# =============================================================================
# guided.R — Guided / seeded topic modeling.
#
# The user supplies seed words for each expected topic; they are turned into
# virtual "anchor" documents that are embedded and added to the corpus before
# UMAP + HDBSCAN.  The anchors pull nearby real documents toward semantically
# appropriate clusters.  Seed documents are removed from the output.
# =============================================================================

#' Guided topic modeling with user-supplied seed words
#'
#' Extends \code{\link{fit_bertopic}} with a \code{seed_topic_list} argument.
#' For each seed topic the user supplies a character vector of representative
#' words; these are concatenated into virtual "anchor" documents, embedded, and
#' added to the corpus before dimensionality reduction and clustering.  The
#' anchors bias the latent space so that documents semantically close to the
#' seed words cluster together.  Seed documents are removed from the fit
#' before the object is returned.
#'
#' @param docs Character vector of documents.
#' @param seed_topic_list Named list of character vectors — one entry per
#'   expected topic.  Example:
#'   \code{list("Energy" = c("solar", "wind", "renewable", "battery"),
#'              "Economics" = c("labour", "wages", "capital", "trade"))}.
#' @param embeddings Optional pre-computed embedding matrix for \code{docs}.
#'   The encoder is still required to embed the seed words.
#' @param encoder An encoder from \code{\link{load_hf_bert}}, used to embed
#'   both documents (when \code{embeddings} is \code{NULL}) and seed words.
#' @param n_anchor_weight Number of times each anchor document is replicated
#'   before adding to the corpus.  Higher values give seeds more influence on
#'   the UMAP layout and HDBSCAN clustering (default 3).
#' @param ... Additional arguments forwarded to \code{\link{fit_bertopic}}
#'   (e.g. \code{umap_n_neighbors}, \code{hdbscan_min_pts}).
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return A \code{bertopic_fit} object.  The attribute \code{seed_map}
#'   records which cluster each seed topic was assigned to.
#' @seealso \code{\link{fit_bertopic}}, \code{\link{zero_shot_topics}}
#' @export
guided_fit_bertopic <- function(docs,
                                 seed_topic_list,
                                 embeddings      = NULL,
                                 encoder         = NULL,
                                 n_anchor_weight = 3L,
                                 ...,
                                 verbose         = TRUE) {
  if (!is.list(seed_topic_list) || is.null(names(seed_topic_list)))
    stop("'seed_topic_list' must be a named list of character vectors.")
  if (is.null(encoder))
    stop("'encoder' is required to embed seed words.")

  n_real   <- length(docs)
  n_labels <- length(seed_topic_list)

  # Build seed documents: concatenate words, repeat n_anchor_weight times
  seed_texts <- rep(
    vapply(seed_topic_list, paste, character(1L), collapse = " "),
    each = n_anchor_weight
  )
  n_seeds <- length(seed_texts)
  all_docs <- c(docs, seed_texts)

  if (verbose)
    message("Guided fit: ", n_labels, " seed topics × ", n_anchor_weight,
            " anchors each (", n_seeds, " virtual documents added).")

  # Embed: real docs from pre-computed matrix if available, seeds always fresh
  if (!is.null(embeddings)) {
    if (nrow(embeddings) != n_real)
      stop("'embeddings' must have one row per document.")
    if (verbose) message("  Embedding seed words...")
    seed_emb <- embed_texts(encoder, seed_texts, normalize = TRUE)
    all_emb  <- rbind(embeddings, seed_emb)
  } else {
    all_emb <- NULL   # fit_bertopic will embed all_docs
  }

  # Fit on real + seed documents
  fit_all <- fit_bertopic(
    docs       = all_docs,
    embeddings = all_emb,
    encoder    = encoder,
    verbose    = verbose,
    ...
  )

  # Identify which cluster each seed topic anchored in
  seed_clusters    <- fit_all$clusters[(n_real + 1L):(n_real + n_seeds)]
  seed_topic_index <- rep(seq_len(n_labels), each = n_anchor_weight)
  seed_map <- vapply(seq_len(n_labels), function(i) {
    cl <- seed_clusters[seed_topic_index == i]
    cl <- cl[cl != -1L]
    if (length(cl) == 0L) return(NA_integer_)
    as.integer(names(sort(table(cl), decreasing = TRUE))[1L])
  }, integer(1L))
  names(seed_map) <- names(seed_topic_list)

  if (verbose) {
    for (nm in names(seed_map)) {
      t <- seed_map[[nm]]
      if (is.na(t)) message("  '", nm, "': seed was assigned to noise")
      else          message("  '", nm, "': anchored to Topic ", t)
    }
  }

  # Strip seed documents from the fit
  real_idx    <- seq_len(n_real)
  fit_out     <- fit_all
  fit_out$docs       <- docs
  fit_out$clusters   <- fit_all$clusters[real_idx]
  fit_out$embeddings <- fit_all$embeddings[real_idx, , drop = FALSE]
  fit_out$reduced    <- fit_all$reduced[real_idx, , drop = FALSE]
  fit_out$layout2d   <- fit_all$layout2d[real_idx, , drop = FALSE]

  # Rebuild DTM and c-TF-IDF on real docs only
  # (fit_all$dtm was built on all_docs; slice it back to real rows)
  fit_out$dtm <- fit_all$dtm[real_idx, , drop = FALSE]

  topics_nn <- sort(setdiff(unique(fit_out$clusters), -1L))
  fit_out$topic_terms <- c_tf_idf(
    fit_out$dtm, fit_out$clusters,
    top_n = fit_all$top_n_terms,
    reduce_frequent_words = fit_all$reduce_frequent_words %||% FALSE
  )
  fit_out$topic_sizes <- setNames(
    vapply(topics_nn, function(t) sum(fit_out$clusters == t), integer(1L)),
    as.character(topics_nn)
  )

  # Recompute centroids from real docs
  .l2 <- function(m) { n <- sqrt(rowSums(m^2)); n[n == 0] <- 1; m / n }
  emb_n <- .l2(fit_out$embeddings)
  fit_out$topic_centroids <- if (length(topics_nn) > 0L) {
    cents <- do.call(rbind, lapply(topics_nn, function(t)
      colMeans(emb_n[fit_out$clusters == t, , drop = FALSE])
    ))
    rownames(cents) <- as.character(topics_nn)
    .l2(cents)
  } else {
    matrix(numeric(0), 0L, ncol(emb_n))
  }

  fit_out$representative_docs <- setNames(lapply(topics_nn, function(t) {
    idx  <- which(fit_out$clusters == t)
    sims <- as.vector(emb_n[idx, , drop = FALSE] %*%
                        fit_out$topic_centroids[as.character(t), ])
    k <- min(3L, length(idx))
    docs[idx[order(sims, decreasing = TRUE)[seq_len(k)]]]
  }), as.character(topics_nn))

  # Apply seed-based labels where a mapping was found
  fit_out$topic_labels <- fit_all$topic_labels[
    intersect(names(fit_all$topic_labels), c("-1", as.character(topics_nn)))
  ]
  for (i in seq_along(seed_map)) {
    t <- seed_map[i]
    if (!is.na(t) && as.character(t) %in% as.character(topics_nn)) {
      nm <- names(seed_map)[i]
      fit_out$topic_labels[[as.character(t)]] <-
        paste0(t, "_", gsub("[^a-zA-Z0-9]+", "_", tolower(nm)))
    }
  }
  if (any(fit_out$clusters == -1L))
    fit_out$topic_labels[["-1"]] <- "-1_outliers"

  attr(fit_out, "seed_map") <- seed_map
  fit_out
}
