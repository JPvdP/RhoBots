# =============================================================================
# coherence.R — Lexical coherence metrics (NPMI, CV) for bertopic_fit objects.
#
# Complements the embedding-space metrics in evaluate.R with word-level
# measures that are standard in the LDA / topic-modeling literature.
# =============================================================================

#' Compute lexical coherence for discovered topics
#'
#' Measures how often the top terms of each topic co-occur in the same
#' documents.  High coherence means the words form a semantically tight
#' cluster — they appear together rather than in isolation.
#'
#' Two measures are supported:
#' \describe{
#'   \item{\code{"npmi"}}{Normalised Pointwise Mutual Information, the standard
#'     measure in recent topic-model benchmarks.  Ranges from \code{-1}
#'     (terms never co-occur) to \code{1} (terms always appear together).
#'     Values above \code{0.1} are generally considered good; above \code{0.3}
#'     is excellent.}
#'   \item{\code{"cv"}}{The C\eqn{_V} coherence measure, a log-conditional
#'     variant that is slightly more discriminative on small corpora.  Higher
#'     is better; typical range \code{-4} to \code{0}.}
#' }
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param top_n Number of top c-TF-IDF terms per topic to include in pairwise
#'   co-occurrence calculations (default 10).
#' @param measure Coherence measure: \code{"npmi"} (default) or \code{"cv"}.
#' @return A list of class \code{topic_coherence} with elements:
#' \describe{
#'   \item{\code{per_topic}}{Named numeric vector of per-topic scores.}
#'   \item{\code{mean}}{Mean coherence across all non-noise topics.}
#'   \item{\code{measure}}{Which measure was used.}
#'   \item{\code{top_n}}{Number of terms used.}
#' }
#' @seealso \code{\link{topic_quality}}
#' @export
topic_coherence <- function(fit, top_n = 10L, measure = c("npmi", "cv")) {
  measure <- match.arg(measure)
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object.")

  topics_nn <- sort(setdiff(unique(fit$clusters), -1L))
  if (length(topics_nn) == 0L)
    stop("No non-noise topics found.")

  n_docs  <- nrow(fit$dtm)

  # fit$dtm is a sparse document-term matrix (DTM): rows = documents,
  # columns = vocabulary terms, values = term counts.
  # bin_dtm converts it to a logical indicator: TRUE where a term appears
  # at least once, FALSE otherwise.  We care only about PRESENCE, not frequency,
  # because coherence measures how reliably two terms co-appear in the same
  # document, regardless of how many times each appears.
  bin_dtm <- fit$dtm > 0

  # P(w) = fraction of documents in which term w appears.
  # colSums on the logical matrix gives the number of documents per term.
  # Dividing by n_docs converts to a probability in [0, 1].
  doc_freq <- Matrix::colSums(bin_dtm) / n_docs

  per_topic <- vapply(topics_nn, function(t) {
    tt    <- fit$topic_terms[fit$topic_terms$topic == t, ]
    terms <- head(tt$term[order(tt$rank)], top_n)
    # Only keep terms that actually appear in the DTM vocabulary.
    # If the corpus was filtered after fitting, some terms may be absent.
    terms <- terms[terms %in% colnames(fit$dtm)]
    if (length(terms) < 2L) return(NA_real_)

    # Extract the indicator columns for just the top terms of this topic.
    # cols is a sparse logical sub-matrix: n_docs × length(terms).
    cols  <- bin_dtm[, terms, drop = FALSE]

    # combn() generates all unordered pairs of column indices.
    # For top_n = 10 terms this gives C(10,2) = 45 pairs.
    # Each pair (w1, w2) is one observation in the coherence average.
    # WHY pairs and not individual terms?  Coherence is a RELATIONAL measure:
    # a term that appears in every document is trivially frequent but tells us
    # nothing about topic cohesion.  What matters is how reliably word pairs
    # co-occur — that is the signal that distinguishes tight topics from loose ones.
    pairs <- utils::combn(seq_len(ncol(cols)), 2L)

    vals <- apply(pairs, 2L, function(idx) {
      w1 <- idx[1L]; w2 <- idx[2L]
      p1  <- doc_freq[terms[w1]]       # P(w1): marginal probability of word 1
      p2  <- doc_freq[terms[w2]]       # P(w2): marginal probability of word 2
      # P(w1, w2): fraction of documents where BOTH words appear.
      # The logical AND (&) finds rows where both indicator columns are TRUE.
      p12 <- sum(cols[, w1] & cols[, w2]) / n_docs

      if (measure == "npmi") {
        # NPMI — Normalised Pointwise Mutual Information
        #
        # Raw PMI = log( P(w1,w2) / (P(w1) × P(w2)) )
        #   Positive when words co-occur more often than chance.
        #   Negative when they avoid each other.
        #
        # NPMI divides by −log P(w1,w2), which maps the range to [−1, +1]:
        #   −1  words never co-occur (p12 → 0)
        #    0  words are statistically independent
        #   +1  words always appear together (p12 = p1 = p2)
        #
        # Normalisation removes the trivial dependence on term frequency:
        # two rare co-occurring words get a high NPMI even if their absolute
        # count is small, which makes the measure comparable across topics
        # of different sizes.
        if (p12 == 0) return(-1)       # perfectly non-co-occurring pair
        log_norm <- -log(p12)
        if (log_norm == 0) return(1)   # p12 = 1: always appear together
        log(p12 / (p1 * p2)) / log_norm
      } else {
        # CV — log-conditional probability with Laplace smoothing
        #
        # CV(w1, w2) = log( (count(w1 ∧ w2) + 1) / (count(w1) + 1) )
        #
        # This is a smoothed estimate of log P(w2 | w1): how often does w2
        # appear given that w1 already appears?  Higher values (closer to 0)
        # mean w2 reliably follows w1.
        #
        # The +1 smoothing (Laplace / add-one) prevents log(0) when the pair
        # never co-occurs, while shrinking the score only slightly for
        # well-attested pairs.
        d12 <- sum(cols[, w1] & cols[, w2])  # raw co-occurrence count
        d1  <- sum(cols[, w1])                # raw count of w1
        if (d1 == 0) return(NA_real_)
        log((d12 + 1) / (d1 + 1))
      }
    })
    mean(vals, na.rm = TRUE)
  }, numeric(1L))
  names(per_topic) <- as.character(topics_nn)

  structure(
    list(
      per_topic = per_topic,
      mean      = mean(per_topic, na.rm = TRUE),
      measure   = measure,
      top_n     = top_n,
      n_topics  = length(topics_nn)
    ),
    class = c("topic_coherence", "list")
  )
}

#' @export
print.topic_coherence <- function(x, ...) {
  cat(sprintf("<topic_coherence>  measure = %s   top_n = %d\n",
              toupper(x$measure), x$top_n))
  hint <- if (x$measure == "npmi")
    "[>0.1 good, >0.3 excellent; ↑ better]" else "[higher is better]"
  cat(sprintf("  Mean %-5s: %6.3f  %s\n", toupper(x$measure), x$mean, hint))
  cat("\n  Per topic:\n")
  for (nm in names(x$per_topic)) {
    v <- x$per_topic[[nm]]
    bar <- if (!is.na(v)) strrep("|", max(0L, round((v + 1) * 10)))
           else "  (NA — fewer than 2 terms in vocab)"
    cat(sprintf("    Topic %-4s %6.3f  %s\n", paste0(nm, ":"), v, bar))
  }
  invisible(x)
}
