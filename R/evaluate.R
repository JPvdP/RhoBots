# =============================================================================
# evaluate.R  --  Topic quality metrics for bertopic_fit objects
# =============================================================================

#' Evaluate topic quality for a fitted BERTopic model
#'
#' Computes four families of metrics that characterise different aspects of
#' topic quality:
#' \describe{
#'   \item{**Cohesion**}{How tightly documents cluster around their topic
#'     centroid (mean cosine similarity of each document to its centroid;
#'     higher is better).}
#'   \item{**Separation**}{How distinct topics are from each other (pairwise
#'     cosine similarity between L2-normalised topic centroids; lower mean is
#'     better, i.e. topics point in different directions in embedding space).}
#'   \item{**Overlap**}{How much vocabulary topics share (pairwise Jaccard
#'     similarity of the top-\code{top_n} c-TF-IDF terms; lower mean is
#'     better).}
#'   \item{**Distribution**}{How balanced and noise-free the topic assignments
#'     are (normalised size entropy, coefficient of variation, noise ratio).}
#' }
#' A **silhouette score** in the full embedding space is also computed.
#' It is the standard cluster-quality measure: values near 1 mean documents
#' sit close to their own centroid and far from the nearest other cluster;
#' values near 0 or below indicate overlapping or mis-assigned topics.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param top_n Number of top c-TF-IDF terms per topic used for the Jaccard
#'   overlap computation. Default 10.
#' @param sample_size Maximum number of non-noise documents used when computing
#'   the silhouette score (which requires an \eqn{n \times n} distance matrix).
#'   Set to \code{NULL} for exact computation. Default 2000.
#' @return A list of class \code{topic_quality} with elements:
#' \describe{
#'   \item{\code{cohesion}}{List with \code{global} (scalar) and
#'     \code{per_topic} (named vector) mean doc-to-centroid cosine similarity.}
#'   \item{\code{separation}}{List with \code{mean_inter_topic_similarity}
#'     (scalar) and \code{centroid_similarity} (symmetric matrix).}
#'   \item{\code{overlap}}{List with \code{mean_jaccard} (scalar) and
#'     \code{jaccard_matrix} (symmetric matrix of pairwise Jaccard scores).}
#'   \item{\code{distribution}}{List with \code{counts} (named integer vector),
#'     \code{noise_ratio}, \code{entropy} (normalised, in \eqn{[0,1]}), and
#'     \code{cv} (coefficient of variation of topic sizes).}
#'   \item{\code{silhouette}}{List with \code{global}, \code{per_topic},
#'     \code{sampled} (logical), and \code{sample_n}.}
#' }
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   q   <- topic_quality(fit)
#'   print(q)
#' }
#' @export
topic_quality <- function(fit, top_n = 10L, sample_size = 2000L) {
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object from fit_bertopic().")

  topics_nn <- sort(unique(fit$clusters[fit$clusters != -1L]))
  n_topics  <- length(topics_nn)
  n_total   <- length(fit$clusters)
  n_noise   <- sum(fit$clusters == -1L)

  nonnoise  <- fit$clusters != -1L
  emb_nn    <- fit$embeddings[nonnoise, , drop = FALSE]
  cl_nn     <- fit$clusters[nonnoise]

  # --- cohesion ---------------------------------------------------------------
  # Embeddings and centroids are already L2-normalised -> cosine_sim = dot.
  doc_cent_sim   <- rowSums(
    emb_nn * fit$topic_centroids[as.character(cl_nn), , drop = FALSE]
  )
  cohesion_topic <- tapply(doc_cent_sim, cl_nn, mean)
  cohesion_glob  <- mean(doc_cent_sim)

  # --- separation -------------------------------------------------------------
  if (n_topics >= 2L) {
    cent_sim <- fit$topic_centroids %*% t(fit$topic_centroids)
    dimnames(cent_sim) <- list(rownames(fit$topic_centroids),
                               rownames(fit$topic_centroids))
    mean_inter_sim <- mean(cent_sim[upper.tri(cent_sim)])
  } else {
    cent_sim       <- matrix(1, 1L, 1L,
                             dimnames = list(as.character(topics_nn),
                                            as.character(topics_nn)))
    mean_inter_sim <- NA_real_
  }

  # --- overlap ----------------------------------------------------------------
  term_sets <- lapply(as.character(topics_nn), function(t) {
    tt <- fit$topic_terms[fit$topic_terms$topic == as.integer(t), ]
    head(tt$term[order(tt$rank)], top_n)
  })
  names(term_sets) <- as.character(topics_nn)

  jac_mat <- matrix(NA_real_, n_topics, n_topics,
                    dimnames = list(as.character(topics_nn),
                                   as.character(topics_nn)))
  diag(jac_mat) <- 1
  if (n_topics >= 2L) {
    for (i in seq_len(n_topics - 1L)) {
      for (j in (i + 1L):n_topics) {
        a <- term_sets[[i]]; b <- term_sets[[j]]
        s <- length(intersect(a, b)) / length(union(a, b))
        jac_mat[i, j] <- jac_mat[j, i] <- s
      }
    }
  }
  mean_jac <- if (n_topics >= 2L) mean(jac_mat[upper.tri(jac_mat)]) else NA_real_

  # --- distribution -----------------------------------------------------------
  topic_counts <- setNames(
    vapply(topics_nn, function(t) sum(cl_nn == t), integer(1L)),
    as.character(topics_nn)
  )
  probs       <- topic_counts / sum(topic_counts)
  entropy_raw <- -sum(probs * log(probs + .Machine$double.eps))
  norm_ent    <- if (n_topics > 1L) entropy_raw / log(n_topics) else 1
  cv          <- if (mean(topic_counts) > 0)
    sd(topic_counts) / mean(topic_counts) else NA_real_

  # --- silhouette -------------------------------------------------------------
  nonnoise_idx <- which(nonnoise)
  if (!is.null(sample_size) && length(nonnoise_idx) > sample_size) {
    samp    <- sample(nonnoise_idx, sample_size)
    sampled <- TRUE
  } else {
    samp    <- nonnoise_idx
    sampled <- FALSE
  }
  emb_s <- fit$embeddings[samp, , drop = FALSE]
  cl_s  <- fit$clusters[samp]

  if (n_topics >= 2L) {
    # Cosine distance = 1 - cosine_similarity; diagonal = 0 (self).
    cos_dist <- 1 - emb_s %*% t(emb_s)

    unique_cls <- unique(cl_s)
    a_vals <- numeric(nrow(emb_s))
    for (t in unique_cls) {
      idx_t <- which(cl_s == t)
      if (length(idx_t) > 1L)
        a_vals[idx_t] <- rowSums(cos_dist[idx_t, idx_t, drop = FALSE]) /
                         (length(idx_t) - 1L)
    }

    b_vals <- vapply(seq_len(nrow(emb_s)), function(i) {
      other_cls <- unique_cls[unique_cls != cl_s[i]]
      if (length(other_cls) == 0L) return(0)
      min(vapply(other_cls, function(t) {
        mean(cos_dist[i, cl_s == t])
      }, numeric(1L)))
    }, numeric(1L))

    denom    <- pmax(a_vals, b_vals)
    sil_vals <- ifelse(denom == 0, 0, (b_vals - a_vals) / denom)
    sil_topic <- tapply(sil_vals, cl_s, mean)
    sil_glob  <- mean(sil_vals)
  } else {
    sil_vals  <- rep(NA_real_, length(samp))
    sil_topic <- setNames(NA_real_, as.character(topics_nn))
    sil_glob  <- NA_real_
  }

  # --- assemble ---------------------------------------------------------------
  structure(
    list(
      cohesion = list(
        global    = cohesion_glob,
        per_topic = cohesion_topic
      ),
      separation = list(
        mean_inter_topic_similarity = mean_inter_sim,
        centroid_similarity         = cent_sim
      ),
      overlap = list(
        mean_jaccard   = mean_jac,
        jaccard_matrix = jac_mat
      ),
      distribution = list(
        counts      = topic_counts,
        noise_ratio = n_noise / n_total,
        entropy     = norm_ent,
        cv          = cv
      ),
      silhouette = list(
        global    = sil_glob,
        per_topic = sil_topic,
        sampled   = sampled,
        sample_n  = length(samp)
      ),
      n_topics = n_topics,
      n_docs   = n_total,
      n_noise  = n_noise
    ),
    class = c("topic_quality", "list")
  )
}

#' @export
print.topic_quality <- function(x, ...) {
  cat("<topic_quality>\n")
  cat(sprintf("  Topics: %d   Docs: %d   Noise: %d (%.1f%%)\n\n",
              x$n_topics, x$n_docs, x$n_noise,
              100 * x$distribution$noise_ratio))
  cat("  Embedding space:\n")
  cat(sprintf("    Cohesion   (doc->centroid cos sim):     %.3f  [higher is better]\n",
              x$cohesion$global))
  cat(sprintf("    Separation (mean inter-centroid sim):   %.3f  [lower is better]\n",
              x$separation$mean_inter_topic_similarity))
  sil_note <- if (isTRUE(x$silhouette$sampled))
    sprintf(" (n=%d)", x$silhouette$sample_n) else ""
  cat(sprintf("    Silhouette score%s:                    %.3f  [higher is better]\n",
              sil_note, x$silhouette$global))
  cat("\n  Vocabulary:\n")
  cat(sprintf("    Mean pairwise Jaccard overlap:         %.3f  [lower is better]\n",
              x$overlap$mean_jaccard))
  cat("\n  Topic size distribution:\n")
  cat(sprintf("    Normalised entropy:                    %.3f  [higher is more balanced]\n",
              x$distribution$entropy))
  cat(sprintf("    Coefficient of variation:              %.3f  [v more balanced]\n",
              x$distribution$cv))
  invisible(x)
}
