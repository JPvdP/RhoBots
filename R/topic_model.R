# =============================================================================
# topic_model.R  --  Topic modeling pipeline on top of `embed_texts()`.
#
# Implements the four-stage BERTopic algorithm (embed -> reduce -> cluster
# -> extract topic terms) with native R packages.  Designed to be extended
# to mirror more of Python BERTopic's capabilities (vectorizers, custom
# representation models, topic reduction, dynamic topic modeling, ...).
# =============================================================================

# -----------------------------------------------------------------------------
# Document-term matrix
# -----------------------------------------------------------------------------

#' Build a sparse document-term matrix from a character vector
#'
#' Tokenizes each document (lowercase, split on non-alphanumeric), removes
#' stopwords, optionally generates n-grams, then applies document-frequency
#' filtering.  Returns a sparse matrix suitable for the c-TF-IDF step.
#'
#' @param docs Character vector of documents.
#' @param min_df Minimum document frequency: terms appearing in fewer than
#'   this many documents are dropped.
#' @param max_df_frac Maximum document-frequency fraction: terms appearing
#'   in more than this fraction of documents are dropped.
#' @param stopwords Character vector of tokens to remove before n-gram
#'   construction.
#' @param ngram_range Integer vector of length 2, e.g. \code{c(1, 2)} to
#'   include both unigrams and bigrams.  Default \code{c(1, 1)} (unigrams
#'   only).
#' @return A sparse \code{dgCMatrix} of dimensions \code{length(docs)} x
#'   vocab size, with \code{colnames} set to the vocabulary.
#' @examples
#' docs <- c("the cat sat on the mat", "the dog chased the cat",
#'           "a cat and a dog", "the mat is on the floor")
#' build_dtm(docs, min_df = 1L)
#' @export
build_dtm <- function(docs, min_df = 2, max_df_frac = 0.95,
                      stopwords = character(), ngram_range = c(1L, 1L)) {
  tokens <- lapply(docs, function(d) {
    t <- tolower(d)
    t <- gsub("[^a-z0-9 ]+", " ", t)
    t <- strsplit(t, "\\s+", perl = TRUE)[[1]]
    t <- t[nchar(t) >= 2]
    t[!t %in% stopwords]
  })

  if (ngram_range[2L] > 1L) {
    tokens <- lapply(tokens, function(toks) {
      result <- if (ngram_range[1L] == 1L) toks else character(0)
      for (n in seq(max(2L, ngram_range[1L]), ngram_range[2L])) {
        len <- length(toks)
        if (len >= n) {
          ng <- vapply(seq_len(len - n + 1L), function(j)
            paste(toks[j:(j + n - 1L)], collapse = " "), character(1))
          result <- c(result, ng)
        }
      }
      result
    })
  }

  vocab <- sort(unique(unlist(tokens)))
  df_count <- tabulate(match(unlist(lapply(tokens, unique)), vocab),
                       nbins = length(vocab))
  N <- length(docs)
  keep <- df_count >= min_df & df_count <= floor(max_df_frac * N)
  vocab <- vocab[keep]
  vocab_idx <- stats::setNames(seq_along(vocab), vocab)

  rows <- integer(0); cols <- integer(0); vals <- integer(0)
  for (i in seq_along(tokens)) {
    toks <- tokens[[i]]
    toks <- toks[toks %in% names(vocab_idx)]
    if (!length(toks)) next
    tab <- table(toks)
    rows <- c(rows, rep(i, length(tab)))
    cols <- c(cols, unname(vocab_idx[names(tab)]))
    vals <- c(vals, as.integer(tab))
  }

  Matrix::sparseMatrix(i = rows, j = cols, x = vals,
                       dims = c(length(docs), length(vocab)),
                       dimnames = list(NULL, vocab))
}

# -----------------------------------------------------------------------------
# Class-based TF-IDF
# -----------------------------------------------------------------------------

#' Class-based TF-IDF (c-TF-IDF) for cluster-level topic terms
#'
#' Treats each cluster as one big document and ranks terms by class TF
#' multiplied by a class-based inverse document frequency.  This is the
#' BERTopic-flavoured TF-IDF that produces interpretable topic descriptors.
#'
#' @param dtm A sparse document-term matrix from [build_dtm()].
#' @param cluster_ids Integer vector of cluster assignments, one per row of
#'   `dtm`.  Use `-1` for noise/unassigned documents.
#' @param top_n Number of top terms to return per cluster.
#' @param reduce_frequent_words If \code{TRUE}, apply a square-root to the
#'   class TF before multiplying by IDF.  This down-weights terms that are
#'   very frequent within a class and often improves topic interpretability.
#'   Mirrors the \code{reduce_frequent_words} option in Python BERTopic's
#'   \code{ClassTfidfTransformer}.  Default \code{FALSE}.
#' @return A data frame with columns `topic`, `rank`, `term`, `score`.
#' @examples
#' docs <- c("the cat sat on mat", "a dog chased cat",
#'           "machine learning models", "deep learning neural nets")
#' dtm  <- build_dtm(docs, min_df = 1L)
#' c_tf_idf(dtm, cluster_ids = c(0L, 0L, 1L, 1L), top_n = 3L)
#' @export
c_tf_idf <- function(dtm, cluster_ids, top_n = 10,
                     reduce_frequent_words = FALSE) {
  classes <- sort(unique(cluster_ids))
  vocab   <- colnames(dtm)

  agg <- Matrix::Matrix(0, nrow = length(classes), ncol = ncol(dtm),
                        dimnames = list(as.character(classes), vocab))
  for (k in seq_along(classes)) {
    rows <- which(cluster_ids == classes[k])
    if (length(rows) == 0) next
    agg[k, ] <- Matrix::colSums(dtm[rows, , drop = FALSE])
  }

  class_totals <- pmax(Matrix::rowSums(agg), 1)
  tf <- agg / class_totals
  if (reduce_frequent_words) tf <- sqrt(tf)

  A    <- mean(class_totals)
  docf <- Matrix::colSums(agg > 0)
  idf  <- log(1 + A / pmax(docf, 1))
  ctfidf <- as.matrix(tf) * matrix(idf, nrow = nrow(tf), ncol = length(idf),
                                   byrow = TRUE)

  n_terms <- ncol(ctfidf)
  topics <- lapply(seq_along(classes), function(k) {
    n <- min(top_n, n_terms)
    o <- order(ctfidf[k, ], decreasing = TRUE)[seq_len(n)]
    data.frame(
      topic = classes[k],
      rank  = seq_len(n),
      term  = vocab[o],
      score = ctfidf[k, o],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, topics)
}

# -----------------------------------------------------------------------------
# HDBSCAN cluster-selection helpers
# -----------------------------------------------------------------------------

# EOM (excess-of-mass) is the native output of dbscan::hdbscan().
# Leaf selection is not directly exposed in the R package, so we derive it
# by traversing the full SNN dendrogram (gen_hdbscan_tree = TRUE):
#   - Recurse into a split when BOTH children have >= min_pts points.
#   - When only one or neither child is large enough to split further,
#     the whole subtree becomes one leaf cluster.
#   - Subtrees with fewer than min_pts points in total become noise (-1).
.hdbscan_leaf <- function(reduced, min_pts) {
  h    <- dbscan::hdbscan(reduced, minPts = min_pts, gen_hdbscan_tree = TRUE)
  dend <- h$hdbscan_tree
  n    <- nrow(reduced)

  assignments <- integer(n)   # 0 = noise initially
  next_id     <- 1L

  process <- function(d) {
    size <- attr(d, "members")
    if (is.null(size) || size < min_pts) return()

    if (is.leaf(d)) return()

    left       <- d[[1L]]
    right      <- d[[2L]]
    left_size  <- attr(left,  "members") %||% 1L
    right_size <- attr(right, "members") %||% 1L

    if (left_size >= min_pts && right_size >= min_pts) {
      process(left)
      process(right)
    } else {
      pts <- as.integer(labels(d))
      assignments[pts] <<- next_id
      next_id <<- next_id + 1L
    }
  }

  if (attr(dend, "members") >= min_pts) process(dend)

  h$cluster <- assignments
  h
}

# -----------------------------------------------------------------------------
# Full BERTopic-style pipeline
# -----------------------------------------------------------------------------

#' Fit a BERTopic-style topic model
#'
#' Runs the four-stage BERTopic pipeline:
#' 1. Embed documents with the supplied encoder.
#' 2. Reduce embedding dimensionality with UMAP.
#' 3. Cluster the reduced space with HDBSCAN.
#' 4. Extract per-topic terms with c-TF-IDF.
#'
#' Also computes a separate 2-D UMAP projection of the same embeddings, for
#' visualization.
#'
#' This implementation will be extended over time to mirror more of the
#' Python BERTopic package's capabilities (custom vectorizers, topic
#' reduction, representation models, dynamic topic modeling).
#'
#' @param encoder A loaded encoder, as returned by [load_hf_bert()].
#' @param docs Character vector of documents.
#' @param embeddings Optional pre-computed embedding matrix (rows = documents,
#'   columns = dimensions).  When supplied, `encoder` is ignored for embedding
#'   and its value may be \code{NULL}.
#' @param dim_reduction_model A dimensionality-reduction model object from
#'   [umap_reduction()], [pca_reduction()], or [no_reduction()].  When
#'   \code{NULL} (default), a UMAP model is built from the legacy
#'   \code{umap_*} parameters.
#' @param cluster_model A clustering model from [hdbscan_clustering()],
#'   [kmeans_clustering()], or [agglomerative_clustering()].  When \code{NULL}
#'   (default), HDBSCAN is used with the legacy \code{hdbscan_*} parameters.
#' @param representation_model Optional representation model from
#'   [cvalue_representation()], [pos_representation()], or similar.  When
#'   \code{NULL} (default) standard c-TF-IDF is used.
#' @param umap_n_neighbors UMAP `n_neighbors` parameter (default 15).
#' @param umap_n_components Reduced dimensionality for clustering (default 5).
#' @param umap_min_dist UMAP `min_dist` (default 0).
#' @param umap_metric UMAP distance metric (default `"cosine"`).
#' @param hdbscan_min_pts HDBSCAN `minPts` (default 10).
#' @param hdbscan_method HDBSCAN cluster-extraction method: \code{"eom"}
#'   (excess of mass, default) or \code{"leaf"}.
#' @param top_n_terms Number of terms per topic in the output (default 10).
#' @param language Language for built-in stopword list (default
#'   \code{"english"}).  Passed to [get_stopwords()].
#' @param stopwords Character vector of words to drop before c-TF-IDF.
#'   Replaces the built-in list when supplied.
#' @param extra_stopwords Additional stopwords appended to the built-in
#'   (or user-supplied) list.
#' @param ngram_range Integer vector of length 2 specifying the minimum and
#'   maximum n-gram sizes for c-TF-IDF (default \code{c(1L, 1L)} = unigrams
#'   only).
#' @param reduce_frequent_words If \code{TRUE}, apply a square-root IDF
#'   dampening to very frequent words (default \code{FALSE}).
#' @param seed Random seed for reproducibility (default 42).
#' @param verbose Whether to print progress messages (default TRUE).
#' @return A list (class `bertopic_fit`) with elements:
#'   `embeddings`, `reduced`, `layout2d`, `clusters`, `topic_terms`,
#'   `hdbscan`, `docs`.
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("pritamdeka/S-Scibert-snli-multinli-stsb")
#'   fit <- fit_bertopic(enc, docs = my_abstracts)
#'   print_topics(fit)
#' }
fit_bertopic <- function(encoder               = NULL,
                         docs,
                         embeddings            = NULL,
                         # Pluggable models (take precedence when supplied)
                         dim_reduction_model   = NULL,
                         cluster_model         = NULL,
                         representation_model  = NULL,
                         # Legacy UMAP params  --  used to build default umap_reduction()
                         umap_n_neighbors      = 15,
                         umap_n_components     = 5,
                         umap_min_dist         = 0.0,
                         umap_metric           = "cosine",
                         # Legacy HDBSCAN params  --  used to build default hdbscan_clustering()
                         hdbscan_min_pts       = 10,
                         hdbscan_method        = c("eom", "leaf"),
                         top_n_terms           = 10,
                         language              = "english",
                         stopwords             = NULL,
                         extra_stopwords       = NULL,
                         ngram_range           = c(1L, 1L),
                         reduce_frequent_words = FALSE,
                         seed                  = 42,
                         verbose               = TRUE) {
  hdbscan_method <- match.arg(hdbscan_method)

  # Build default models from legacy params when not explicitly supplied
  if (is.null(dim_reduction_model))
    dim_reduction_model <- umap_reduction(n_neighbors  = umap_n_neighbors,
                                           n_components = umap_n_components,
                                           min_dist     = umap_min_dist,
                                           metric       = umap_metric)
  if (is.null(cluster_model))
    cluster_model <- hdbscan_clustering(min_pts = hdbscan_min_pts,
                                         method  = hdbscan_method)

  # --- [1] Embeddings -------------------------------------------------------
  if (!is.null(embeddings)) {
    if (nrow(embeddings) != length(docs))
      stop("'embeddings' must have the same number of rows as 'docs'.")
    if (verbose) message("[1/4] Using ", length(docs), " pre-computed embeddings.")
    emb <- embeddings
  } else {
    if (is.null(encoder))
      stop("Provide either 'encoder' or pre-computed 'embeddings'.")
    if (verbose) message("[1/4] Embedding ", length(docs), " documents...")
    emb <- embed_texts(encoder, docs, verbose = verbose)
  }

  # --- [2] Dimensionality reduction -----------------------------------------
  if (verbose) message("[2/4] Reducing dimensions (",
                        class(dim_reduction_model)[1L], ")...")
  dr_result           <- dim_reduce(dim_reduction_model, emb,
                                     seed = seed, verbose = verbose)
  reduced             <- dr_result$embedding
  dim_reduction_model <- dr_result$model   # store fitted model

  # --- [3] Clustering -------------------------------------------------------
  if (verbose) message("[3/4] Clustering (", class(cluster_model)[1L], ")...")
  cl_result     <- cluster_docs(cluster_model, reduced, seed = seed)
  cluster_ids   <- cl_result$labels
  cluster_model <- cl_result$model         # store fitted model

  # --- [4] c-TF-IDF ---------------------------------------------------------
  if (verbose) message("[4/4] Extracting topic terms (c-TF-IDF)...")
  # Build final stopword list:
  #   stopwords = NULL  -> use language defaults
  #   stopwords = c(...)  -> replaces defaults entirely
  #   extra_stopwords   -> always appended on top (accepts vector, file, or df)
  base_sw  <- if (!is.null(stopwords)) stopwords else
    if (nchar(language) > 0L) .english_stopwords else character(0)
  extra_sw <- if (!is.null(extra_stopwords))
    load_stopwords(extra_stopwords) else character(0)
  final_sw <- unique(c(base_sw, extra_sw))

  if (is.null(representation_model)) {
    dtm <- build_dtm(docs, stopwords = final_sw, ngram_range = ngram_range)
  } else if (inherits(representation_model, "pos_representation")) {
    dtm <- pos_dtm(
      docs,
      pos             = representation_model$pos,
      patterns        = representation_model$patterns,
      lemmatize       = representation_model$lemmatize,
      language        = representation_model$language,
      model_dir       = representation_model$model_dir,
      min_df          = representation_model$min_df,
      max_df_frac     = representation_model$max_df_frac,
      extra_stopwords = extra_sw,
      verbose         = verbose
    )
  } else if (inherits(representation_model, "cvalue_representation")) {
    dtm <- .cvalue_dtm(docs, representation_model, final_sw, verbose)
  } else {
    stop("Unsupported representation_model class: ",
         paste(class(representation_model), collapse = "/"),
         "\nUse pos_representation() or cvalue_representation().")
  }

  topic_terms <- c_tf_idf(dtm, cluster_ids, top_n = top_n_terms,
                           reduce_frequent_words = reduce_frequent_words)

  # 2-D layout for visualization (always UMAP for quality)
  if (verbose) message("Computing 2-D layout for visualization...")
  viz_metric     <- if (inherits(dim_reduction_model, "umap_reduction"))
    dim_reduction_model$metric else "cosine"
  viz_neighbors  <- if (inherits(dim_reduction_model, "umap_reduction"))
    dim_reduction_model$n_neighbors else 15L
  set.seed(seed)
  layout2d <- uwot::umap(emb, n_neighbors = viz_neighbors,
                          n_components = 2L, min_dist = 0.1,
                          metric = viz_metric, verbose = FALSE)

  # --- Derived state -------------------------------------------------------
  topics_nonnoise <- sort(setdiff(unique(cluster_ids), -1L))

  topic_sizes <- setNames(
    vapply(topics_nonnoise, function(t) sum(cluster_ids == t), integer(1)),
    as.character(topics_nonnoise)
  )

  topic_labels <- .generate_topic_labels(topic_terms, topics_nonnoise,
                                         nr_words = 4L)
  if (any(cluster_ids == -1L))
    topic_labels[["-1"]] <- "-1_outliers"

  # L2-normalise embeddings for similarity arithmetic
  enorms <- sqrt(rowSums(emb^2))
  enorms[enorms == 0] <- 1
  emb_n <- emb / enorms

  if (length(topics_nonnoise) == 0L) {
    topic_centroids <- matrix(numeric(0), nrow = 0L, ncol = ncol(emb))
  } else {
    topic_centroids <- do.call(rbind, lapply(topics_nonnoise, function(t) {
      colMeans(emb_n[cluster_ids == t, , drop = FALSE])
    }))
    rownames(topic_centroids) <- as.character(topics_nonnoise)
    cent_norms <- sqrt(rowSums(topic_centroids^2))
    cent_norms[cent_norms == 0] <- 1
    topic_centroids <- sweep(topic_centroids, 1L, cent_norms, "/")
  }

  representative_docs <- setNames(lapply(topics_nonnoise, function(t) {
    idx <- which(cluster_ids == t)
    sims <- as.vector(emb_n[idx, , drop = FALSE] %*%
                        topic_centroids[as.character(t), ])
    top_k <- min(3L, length(idx))
    docs[idx[order(sims, decreasing = TRUE)[seq_len(top_k)]]]
  }), as.character(topics_nonnoise))
  # -------------------------------------------------------------------------

  structure(
    list(
      embeddings            = emb,
      reduced               = reduced,
      dim_reduction_model   = dim_reduction_model,
      cluster_model         = cluster_model,
      representation_model  = representation_model,
      layout2d              = layout2d,
      clusters              = cluster_ids,
      topic_terms           = topic_terms,
      topic_sizes           = topic_sizes,
      topic_labels          = topic_labels,
      topic_centroids       = topic_centroids,
      representative_docs   = representative_docs,
      docs                  = docs,
      dtm                   = dtm,
      top_n_terms           = top_n_terms,
      reduce_frequent_words = reduce_frequent_words
    ),
    class = c("bertopic_fit", "list")
  )
}

#' Pretty-print discovered topics
#'
#' @param fit A fit object from [fit_bertopic()].
#' @param max_topics Maximum number of topics to display.
#' @return Called for its side effect (printing); returns \code{NULL} invisibly.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   print_topics(fit)
#' }
#' @export
print_topics <- function(fit, max_topics = 20) {
  tt <- fit$topic_terms
  topics <- sort(unique(tt$topic))
  topics <- topics[topics != -1]
  if (length(topics) > max_topics) topics <- topics[seq_len(max_topics)]
  for (t in topics) {
    rows <- tt[tt$topic == t, ]
    terms <- rows$term[order(rows$rank)]
    cat(sprintf("Topic %2d (%d docs): %s\n",
                t, sum(fit$clusters == t),
                paste(terms, collapse = ", ")))
  }
  noise <- sum(fit$clusters == -1)
  if (noise > 0)
    cat(sprintf("  (%d documents unassigned / noise)\n", noise))
  invisible(fit)
}

#' Print method for bertopic_fit objects
#' @param x A bertopic_fit object.
#' @param ... Unused.
#' @return Invisibly returns \code{x}.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   print(fit)
#' }
#' @export
print.bertopic_fit <- function(x, ...) {
  n_topics <- length(setdiff(unique(x$clusters), -1L))
  n_noise  <- sum(x$clusters == -1L)
  cat("<bertopic_fit>\n")
  cat("  documents:      ", length(x$docs), "\n")
  cat("  topics found:   ", n_topics, "\n")
  cat("  noise points:   ", n_noise, "\n")
  cat("  embedding dim:  ", ncol(x$embeddings), "\n")
  invisible(x)
}
