# =============================================================================
# topic_manipulation.R  --  Post-fit operations on a bertopic_fit object.
#
# reduce_topics()    --  iteratively merge the most similar topics
# reduce_outliers()  --  reassign noise (-1) documents to the nearest real topic
# merge_topics()     --  manually collapse a specific set of topics into one
# =============================================================================

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

# Recompute all derived state (topic_terms, sizes, labels, centroids,
# representative_docs) from the current fit$clusters.  Called after any
# operation that changes cluster assignments.
.recompute_topics <- function(fit) {
  cluster_ids     <- fit$clusters
  topics_nonnoise <- sort(setdiff(unique(cluster_ids), -1L))

  topic_terms <- c_tf_idf(fit$dtm, cluster_ids, top_n = fit$top_n_terms,
                           reduce_frequent_words = fit$reduce_frequent_words %||% FALSE)

  topic_sizes <- setNames(
    vapply(topics_nonnoise, function(t) sum(cluster_ids == t), integer(1)),
    as.character(topics_nonnoise)
  )

  topic_labels <- .generate_topic_labels(topic_terms, topics_nonnoise,
                                          nr_words = 4L)
  if (any(cluster_ids == -1L))
    topic_labels[["-1"]] <- "-1_outliers"

  emb    <- fit$embeddings
  enorms <- sqrt(rowSums(emb^2))
  enorms[enorms == 0] <- 1
  emb_n  <- emb / enorms

  if (length(topics_nonnoise) == 0L) {
    topic_centroids <- matrix(numeric(0), nrow = 0L, ncol = ncol(emb))
  } else {
    topic_centroids <- do.call(rbind, lapply(topics_nonnoise, function(t) {
      colMeans(emb_n[cluster_ids == t, , drop = FALSE])
    }))
    rownames(topic_centroids) <- as.character(topics_nonnoise)
    cn <- sqrt(rowSums(topic_centroids^2))
    cn[cn == 0] <- 1
    topic_centroids <- sweep(topic_centroids, 1L, cn, "/")
  }

  representative_docs <- setNames(lapply(topics_nonnoise, function(t) {
    idx  <- which(cluster_ids == t)
    sims <- as.vector(emb_n[idx, , drop = FALSE] %*%
                        topic_centroids[as.character(t), ])
    top_k <- min(3L, length(idx))
    fit$docs[idx[order(sims, decreasing = TRUE)[seq_len(top_k)]]]
  }), as.character(topics_nonnoise))

  fit$topic_terms         <- topic_terms
  fit$topic_sizes         <- topic_sizes
  fit$topic_labels        <- topic_labels
  fit$topic_centroids     <- topic_centroids
  fit$representative_docs <- representative_docs
  fit
}

# Reassign topic IDs to a compact 0-based sequence after merges.
# Topics are renumbered in ascending order of their current IDs;
# noise (-1) is preserved.
.renumber_topics <- function(fit) {
  topics_nonnoise <- sort(setdiff(unique(fit$clusters), -1L))
  if (length(topics_nonnoise) == 0L) return(fit)
  new_clusters <- fit$clusters
  for (i in seq_along(topics_nonnoise)) {
    new_clusters[fit$clusters == topics_nonnoise[i]] <- i - 1L
  }
  fit$clusters <- new_clusters
  fit
}

# -----------------------------------------------------------------------------
# reduce_topics
# -----------------------------------------------------------------------------

#' Reduce the number of topics by iteratively merging the most similar pair
#'
#' At each step the two non-noise topics whose centroids have the highest
#' cosine similarity are merged.  The loop continues until the desired number
#' of topics remains.  After the loop, topics are renumbered 0, 1, 2, ... in
#' order of their (post-merge) IDs and all derived state is recomputed.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param nr_topics Target number of non-noise topics.
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return An updated \code{bertopic_fit} with renumbered topics.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   fit <- reduce_topics(fit, nr_topics = 5L)
#' }
#' @export
reduce_topics <- function(fit, nr_topics, verbose = TRUE) {
  if (nr_topics < 1L) stop("nr_topics must be at least 1.")

  n_current <- length(setdiff(unique(fit$clusters), -1L))

  if (nr_topics >= n_current) {
    message("Already at or below ", nr_topics, " topics (currently ",
            n_current, "). No changes made.")
    return(fit)
  }

  if (verbose)
    message("Reducing from ", n_current, " to ", nr_topics, " topics...")

  while (TRUE) {
    n_now <- length(setdiff(unique(fit$clusters), -1L))
    if (n_now <= nr_topics) break

    centroids <- fit$topic_centroids
    if (nrow(centroids) <= 1L) break

    sim_mat       <- centroids %*% t(centroids)
    diag(sim_mat) <- -Inf
    best          <- arrayInd(which.max(sim_mat), dim(sim_mat))

    t1   <- as.integer(rownames(centroids)[best[1, 1]])
    t2   <- as.integer(rownames(centroids)[best[1, 2]])
    keep <- min(t1, t2)
    drop <- max(t1, t2)

    fit$clusters[fit$clusters == drop] <- keep
    fit <- .recompute_topics(fit)
  }

  fit <- .renumber_topics(fit)
  fit <- .recompute_topics(fit)

  if (verbose)
    message("Done. Topics remaining: ",
            length(setdiff(unique(fit$clusters), -1L)))
  fit
}

# -----------------------------------------------------------------------------
# reduce_outliers
# -----------------------------------------------------------------------------

#' Reassign noise documents to the nearest real topic
#'
#' Documents assigned to the noise cluster (\code{-1}) by HDBSCAN are
#' reassigned to the most similar non-noise topic.  Only documents whose
#' best similarity exceeds \code{threshold} are reassigned; the rest remain
#' as noise.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param strategy How to measure similarity:
#'   \describe{
#'     \item{\code{"embeddings"} (default)}{Cosine similarity between the
#'       document embedding and each topic centroid.}
#'     \item{\code{"c-tf-idf"}}{Cosine similarity between the document's
#'       term-frequency vector and each topic's c-TF-IDF vector.  Useful
#'       when embedding quality is low or unavailable.}
#'   }
#' @param threshold Minimum similarity required to reassign a document.
#'   Documents below the threshold remain as noise (\code{-1}).
#'   Default \code{0.0} reassigns all noise documents.
#' @param verbose Print a reassignment summary (default \code{TRUE}).
#' @return An updated \code{bertopic_fit}.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   fit <- reduce_outliers(fit, threshold = 0.1)
#' }
#' @export
reduce_outliers <- function(fit, strategy = "embeddings", threshold = 0.0,
                             verbose = TRUE) {
  noise_idx <- which(fit$clusters == -1L)

  if (length(noise_idx) == 0L) {
    if (verbose) message("No noise documents to reassign.")
    return(fit)
  }
  if (nrow(fit$topic_centroids) == 0L)
    stop("No non-noise topics in this fit  --  nothing to assign to.")

  if (strategy == "embeddings") {
    emb   <- fit$embeddings[noise_idx, , drop = FALSE]
    norms <- sqrt(rowSums(emb^2))
    norms[norms == 0] <- 1
    emb_n <- emb / norms

    sim_mat   <- emb_n %*% t(fit$topic_centroids)
    best_idx  <- max.col(sim_mat, ties.method = "first")
    best_sim  <- sim_mat[cbind(seq_len(nrow(sim_mat)), best_idx)]
    topic_ids <- as.integer(rownames(fit$topic_centroids))

    fit$clusters[noise_idx] <- ifelse(best_sim >= threshold,
                                       topic_ids[best_idx], -1L)

  } else if (strategy == "c-tf-idf") {
    # Build a full topic c-TF-IDF matrix (topics x vocab) from non-noise docs
    non_noise  <- fit$clusters != -1L
    class_ids  <- fit$clusters[non_noise]
    dtm_nn     <- fit$dtm[non_noise, , drop = FALSE]
    classes    <- sort(unique(class_ids))
    n_terms    <- ncol(fit$dtm)

    class_mat <- Matrix::Matrix(0, nrow = length(classes), ncol = n_terms,
                                dimnames = list(as.character(classes),
                                                colnames(fit$dtm)))
    for (k in seq_along(classes)) {
      idx <- which(class_ids == classes[k])
      class_mat[k, ] <- Matrix::colSums(dtm_nn[idx, , drop = FALSE])
    }

    class_totals <- pmax(Matrix::rowSums(class_mat), 1)
    tf           <- class_mat / class_totals
    A            <- mean(class_totals)
    docf         <- Matrix::colSums(class_mat > 0)
    idf          <- log(1 + A / pmax(docf, 1))
    ctf          <- tf * Matrix::Matrix(rep(idf, each = nrow(tf)),
                                        nrow = nrow(tf))

    rn <- sqrt(Matrix::rowSums(ctf^2))
    rn[rn == 0] <- 1
    ctf_norm <- ctf / rn

    noise_dtm  <- fit$dtm[noise_idx, , drop = FALSE]
    dn         <- sqrt(Matrix::rowSums(noise_dtm^2))
    dn[dn == 0] <- 1
    noise_norm <- noise_dtm / dn

    sim_mat  <- as.matrix(noise_norm %*% Matrix::t(ctf_norm))
    best_idx <- max.col(sim_mat, ties.method = "first")
    best_sim <- sim_mat[cbind(seq_len(nrow(sim_mat)), best_idx)]

    fit$clusters[noise_idx] <- ifelse(best_sim >= threshold,
                                       classes[best_idx], -1L)

  } else {
    stop("Unknown strategy '", strategy, "'. Use \"embeddings\" or \"c-tf-idf\".")
  }

  n_reassigned <- sum(fit$clusters[noise_idx] != -1L)
  if (verbose)
    message("Reassigned ", n_reassigned, " / ", length(noise_idx),
            " noise documents.")

  .recompute_topics(fit)
}

# -----------------------------------------------------------------------------
# merge_topics
# -----------------------------------------------------------------------------

#' Manually merge a set of topics into one
#'
#' All topics in \code{topics_to_merge} are combined into the one with the
#' smallest ID.  All derived state (c-TF-IDF terms, labels, centroids,
#' representative docs) is recomputed from the merged assignments.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param topics_to_merge Integer vector of at least two topic IDs to merge.
#' @return An updated \code{bertopic_fit}.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   fit <- merge_topics(fit, topics_to_merge = c(0L, 1L))
#' }
#' @export
merge_topics <- function(fit, topics_to_merge) {
  topics_to_merge <- sort(unique(as.integer(topics_to_merge)))

  existing <- unique(fit$clusters[fit$clusters != -1L])
  missing  <- topics_to_merge[!topics_to_merge %in% existing]
  if (length(missing) > 0L)
    stop("Topic IDs not found in this fit: ", paste(missing, collapse = ", "))
  if (length(topics_to_merge) < 2L)
    stop("Provide at least 2 topic IDs to merge.")

  keep <- topics_to_merge[1L]
  for (t in topics_to_merge[-1L]) {
    fit$clusters[fit$clusters == t] <- keep
  }

  .recompute_topics(fit)
}
