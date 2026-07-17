# =============================================================================
# zero_shot.R  --  Confirmatory topic modeling without unsupervised clustering.
#
# The user supplies topic labels + descriptions; each document is assigned to
# the most similar label by cosine similarity in embedding space.
# =============================================================================

#' Zero-shot topic modeling with user-defined topic labels
#'
#' Instead of discovering topics through unsupervised clustering, the user
#' supplies a named character vector of topic descriptions.  Each description
#' is embedded and every document is assigned to the nearest topic by cosine
#' similarity.  Documents whose best similarity falls below \code{threshold}
#' are labelled as noise (\code{-1}).
#'
#' The returned object is a full \code{bertopic_fit} compatible with all
#' Rhobots accessor and visualisation functions (\code{get_topic_info()},
#' \code{visualize_barchart()}, \code{topic_quality()}, etc.).
#'
#' @param docs Character vector of documents.
#' @param labels Named character vector of topic descriptions.  Names become
#'   topic labels (e.g. \code{"Climate policy"}); values are the text used to
#'   compute the embedding anchor (e.g. \code{"carbon emissions renewable energy
#'   net-zero"}).
#' @param embeddings Optional pre-computed numeric matrix for \code{docs}
#'   (\code{nrow = length(docs)}).
#' @param encoder Optional encoder from \code{\link{load_hf_bert}}.  Required
#'   when \code{embeddings} is \code{NULL}, or to embed the label descriptions.
#' @param label_embeddings Optional pre-computed numeric matrix for the label
#'   descriptions (\code{nrow = length(labels)}).  When both \code{embeddings}
#'   and \code{label_embeddings} are supplied, no encoder is needed.
#' @param threshold Minimum cosine similarity for assignment.  Documents below
#'   this threshold are assigned to noise (\code{-1}).  Default \code{0.0}
#'   forces every document into some topic.
#' @param ngram_range Integer vector \code{c(min, max)} for n-gram extraction
#'   in the c-TF-IDF step (default \code{c(1L, 2L)}).
#' @param top_n_terms Number of c-TF-IDF terms stored per topic (default 10).
#' @param extra_stopwords Additional stopwords  --  character vector, file path,
#'   or data frame (same formats accepted by \code{\link{fit_bertopic}}).
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return A \code{bertopic_fit} object.  The extra field
#'   \code{$label_names} records the original user-supplied label names.
#' @seealso \code{\link{fit_bertopic}}, \code{\link{guided_fit_bertopic}}
#' @examples
#' \dontrun{
#'   enc    <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   labels <- c(
#'     "Climate change" = "carbon emissions global warming climate",
#'     "Machine learning" = "neural network deep learning model training"
#'   )
#'   fit <- zero_shot_topics(abstracts, labels = labels, encoder = enc)
#'   get_topic_info(fit)
#' }
#' @export
zero_shot_topics <- function(docs,
                              labels,
                              embeddings       = NULL,
                              encoder          = NULL,
                              label_embeddings = NULL,
                              threshold        = 0.0,
                              ngram_range      = c(1L, 2L),
                              top_n_terms      = 10L,
                              extra_stopwords  = character(0L),
                              verbose          = TRUE) {
  stopifnot(is.character(docs), length(docs) > 0L)
  if (!is.character(labels) || is.null(names(labels)) || length(labels) < 2L)
    stop("'labels' must be a named character vector with at least two entries.")

  .l2 <- function(m) {
    n <- sqrt(rowSums(m^2)); n[n == 0] <- 1; m / n
  }

  # --- [1] Embed documents ---------------------------------------------------
  if (!is.null(embeddings)) {
    if (nrow(embeddings) != length(docs))
      stop("'embeddings' must have one row per document.")
    if (verbose) message("[1/3] Using ", length(docs), " pre-computed document embeddings.")
    emb <- embeddings
  } else {
    if (is.null(encoder))
      stop("Provide 'encoder' or pre-computed 'embeddings'.")
    if (verbose) message("[1/3] Embedding ", length(docs), " documents...")
    emb <- embed_texts(encoder, docs, normalize = TRUE)
  }

  # --- [2] Embed labels ------------------------------------------------------
  if (!is.null(label_embeddings)) {
    if (nrow(label_embeddings) != length(labels))
      stop("'label_embeddings' must have one row per label.")
    if (verbose) message("[2/3] Using pre-computed label embeddings.")
    lbl_emb <- label_embeddings
  } else {
    if (is.null(encoder))
      stop("Provide 'encoder' or 'label_embeddings' to embed topic descriptions.")
    if (verbose) message("[2/3] Embedding ", length(labels), " topic descriptions...")
    lbl_emb <- embed_texts(encoder, as.character(labels), normalize = TRUE)
  }

  emb_n <- .l2(emb)
  lbl_n <- .l2(lbl_emb)

  # --- [3] Assign documents to nearest label ---------------------------------
  if (verbose) message("[3/3] Assigning documents to topics...")
  sim_mat  <- emb_n %*% t(lbl_n)
  best_idx <- max.col(sim_mat, ties.method = "first")
  best_sim <- sim_mat[cbind(seq_len(nrow(sim_mat)), best_idx)]
  cluster_ids <- ifelse(best_sim >= threshold, best_idx - 1L, -1L)

  # --- c-TF-IDF representation -----------------------------------------------
  extra_sw <- if (length(extra_stopwords) > 0L)
    load_stopwords(extra_stopwords) else character(0L)
  final_sw <- unique(c(.english_stopwords, extra_sw))

  dtm         <- build_dtm(docs, stopwords = final_sw, ngram_range = ngram_range)
  topic_terms <- c_tf_idf(dtm, cluster_ids, top_n = top_n_terms)

  topics_nn <- sort(setdiff(unique(cluster_ids), -1L))

  topic_sizes <- setNames(
    vapply(topics_nn, function(t) sum(cluster_ids == t), integer(1L)),
    as.character(topics_nn)
  )

  topic_labels <- setNames(
    lapply(topics_nn, function(t) {
      nm <- names(labels)[t + 1L]
      paste0(t, "_", gsub("[^a-zA-Z0-9]+", "_", tolower(nm)))
    }),
    as.character(topics_nn)
  )
  if (any(cluster_ids == -1L))
    topic_labels[["-1"]] <- "-1_outliers"

  # Centroids
  topic_centroids <- if (length(topics_nn) > 0L) {
    cents <- do.call(rbind, lapply(topics_nn, function(t)
      colMeans(emb_n[cluster_ids == t, , drop = FALSE])
    ))
    rownames(cents) <- as.character(topics_nn)
    cents <- .l2(cents)
    cents
  } else {
    matrix(numeric(0), 0L, ncol(emb))
  }

  representative_docs <- setNames(lapply(topics_nn, function(t) {
    idx  <- which(cluster_ids == t)
    sims <- as.vector(emb_n[idx, , drop = FALSE] %*%
                        topic_centroids[as.character(t), ])
    k    <- min(3L, length(idx))
    docs[idx[order(sims, decreasing = TRUE)[seq_len(k)]]]
  }), as.character(topics_nn))

  set.seed(42L)
  layout2d <- uwot::umap(emb, n_components = 2L, min_dist = 0.1,
                          n_neighbors = min(15L, nrow(emb) - 1L),
                          metric = "cosine", verbose = FALSE)

  structure(
    list(
      embeddings            = emb,
      reduced               = emb_n,
      dim_reduction_model   = NULL,
      cluster_model         = NULL,
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
      reduce_frequent_words = FALSE,
      label_names           = names(labels)
    ),
    class = c("bertopic_fit", "list")
  )
}
