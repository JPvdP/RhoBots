# =============================================================================
# representation.R — Post-fit topic representation models.
#
# apply_mmr()  Maximal Marginal Relevance: re-ranks topic terms to balance
#              relevance to the topic with diversity across selected terms.
# =============================================================================

#' Refine topic representations with Maximal Marginal Relevance (MMR)
#'
#' After fitting a topic model, the top terms for each topic are selected
#' purely by c-TF-IDF score, which can produce redundant terms (e.g.
#' \emph{model}, \emph{models}, \emph{modeling}).  MMR re-ranks the candidate
#' terms by jointly maximising their relevance to the topic and their
#' diversity from already-selected terms.
#'
#' The algorithm mirrors Python BERTopic's
#' \code{MaximalMarginalRelevance} representation model:
#' \enumerate{
#'   \item For each topic, take the top \code{top_n_candidates} terms from
#'     c-TF-IDF as the candidate pool.
#'   \item Embed every unique candidate term and a \emph{topic reference
#'     string} (the candidates joined into one string) using \code{encoder}.
#'   \item Greedily select \code{top_n} terms by
#'     \deqn{MMR_i = (1 - \lambda)\,\text{sim}(w_i, \text{topic}) -
#'           \lambda\,\max_{s \in S}\text{sim}(w_i, s)}
#'     where \eqn{S} is the set of already-selected terms and
#'     \eqn{\lambda} = \code{diversity}.
#'   \item Original c-TF-IDF scores are preserved for the selected terms;
#'     only the selection and ranking change.
#' }
#'
#' \strong{Note:} MMR operates on the existing \code{topic_terms} in the fit
#' object.  If you subsequently call \code{\link{reduce_topics}},
#' \code{\link{merge_topics}}, or \code{\link{reduce_outliers}}, topic terms
#' are recomputed from c-TF-IDF and MMR needs to be re-applied.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param encoder An encoder from \code{\link{load_hf_bert}}, used to embed
#'   candidate terms.
#' @param diversity Controls the relevance-diversity trade-off.  \code{0.0}
#'   gives pure relevance (close to the original c-TF-IDF ranking);
#'   \code{1.0} gives maximum diversity.  Default \code{0.1}.
#' @param top_n Number of terms to keep per topic after MMR.  Defaults to
#'   \code{fit$top_n_terms}.
#' @param top_n_candidates Size of the candidate pool drawn from c-TF-IDF
#'   before MMR selection.  Must be \eqn{\geq} \code{top_n}.  Increasing
#'   this gives MMR a broader pool to diversify from.  Defaults to
#'   \code{top_n} (matching Python BERTopic's default behaviour).
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return An updated \code{bertopic_fit} with \code{topic_terms} and
#'   \code{topic_labels} replaced by the MMR-selected representation.
#' @export
apply_mmr <- function(fit,
                       encoder,
                       diversity        = 0.1,
                       top_n            = NULL,
                       top_n_candidates = NULL,
                       verbose          = TRUE) {
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object.")
  if (diversity < 0 || diversity > 1)
    stop("'diversity' must be between 0 and 1.")

  topics_nonnoise <- sort(setdiff(unique(fit$clusters), -1L))
  if (length(topics_nonnoise) == 0L) {
    message("No non-noise topics to update.")
    return(fit)
  }

  if (is.null(top_n))            top_n            <- fit$top_n_terms
  if (is.null(top_n_candidates)) top_n_candidates <- top_n
  top_n_candidates <- max(as.integer(top_n_candidates), as.integer(top_n))

  # --- 1. Collect candidate terms and topic reference strings --------------
  # For each topic, pull the top top_n_candidates terms ranked by c-TF-IDF.
  # These are the terms MMR will choose from (the "candidate pool").
  # A larger pool gives MMR more vocabulary to diversify into; a smaller pool
  # keeps it close to the original c-TF-IDF selection.
  candidates_by_topic <- stats::setNames(
    lapply(topics_nonnoise, function(t) {
      tt <- fit$topic_terms[fit$topic_terms$topic == t, ]
      tt$term[order(tt$rank)][seq_len(min(top_n_candidates, nrow(tt)))]
    }),
    as.character(topics_nonnoise)
  )

  # all_unique_terms: deduplicated pool across ALL topics (many terms appear
  # in several topics).  Embedding once and indexing is far cheaper than
  # embedding separately per topic.
  all_unique_terms <- unique(unlist(candidates_by_topic, use.names = FALSE))

  # topic_ref_strings: for each topic we create a "topic reference" by
  # concatenating all candidate terms into one string, then embed it.
  # WHY? The encoder turns this bag-of-words string into a vector that sits
  # near the centroid of those terms in embedding space — a cheap proxy for
  # "what this topic is about".  MMR uses this vector as the relevance target.
  topic_ref_strings <- vapply(
    as.character(topics_nonnoise),
    function(t) paste(candidates_by_topic[[t]], collapse = " "),
    character(1)
  )

  # --- 2. Embed terms and topic references, then L2-normalise --------------
  # We embed all candidate terms AND all topic reference strings in a single
  # encoder call (one batch pass is cheaper than many small calls).
  # The result is a single matrix; we split it by position afterwards.
  n_terms  <- length(all_unique_terms)
  n_topics <- length(topics_nonnoise)

  if (verbose)
    message("Embedding ", n_terms, " candidate terms and ",
            n_topics, " topic references...")

  all_texts <- c(all_unique_terms, topic_ref_strings)
  all_emb   <- embed_texts(encoder, all_texts, verbose = FALSE)

  # L2-normalise every row so that the dot product between two vectors equals
  # their cosine similarity.  Cosine similarity is invariant to scale, which
  # matters here because words and topic references have very different raw
  # embedding magnitudes.
  norms <- sqrt(rowSums(all_emb^2))
  norms[norms == 0] <- 1          # guard against zero vectors
  all_emb_n <- all_emb / norms

  # Split the normalised matrix into term embeddings and topic embeddings.
  # term_emb:  n_unique_terms × hidden_size  (one row per unique candidate term)
  # topic_emb: n_topics       × hidden_size  (one row per topic reference)
  term_emb  <- all_emb_n[seq_len(n_terms), , drop = FALSE]
  topic_emb <- all_emb_n[n_terms + seq_len(n_topics), , drop = FALSE]

  # term_idx maps each term string → its row position in term_emb.
  # We use this to look up embeddings by name when building cand_emb below.
  term_idx  <- stats::setNames(seq_len(n_terms), all_unique_terms)

  # --- 3. Greedy MMR selection, one topic at a time -----------------------
  # For each topic we greedily build a set S of top_n terms that balances:
  #   relevance  — how similar a candidate is to the topic reference
  #   redundancy — how similar it is to terms already in S
  #
  # At each step k we pick the remaining candidate that maximises:
  #   MMR(w) = (1 − λ) × sim(w, topic) − λ × max_{s∈S} sim(w, s)
  # where λ = diversity (default 0.1 → strongly favour relevance).
  #
  # A diversity of 0 reduces to pure relevance ordering (same as c-TF-IDF rank).
  # A diversity of 1 would greedily maximise dissimilarity at the expense of
  # any relationship to the topic.  Values of 0.1–0.3 work well in practice.
  new_rows <- lapply(seq_along(topics_nonnoise), function(i) {
    t          <- topics_nonnoise[i]
    candidates <- candidates_by_topic[[as.character(t)]]
    n_cand     <- length(candidates)
    if (n_cand == 0L) return(NULL)

    # Rows of term_emb corresponding to this topic's candidates.
    cand_emb  <- term_emb[term_idx[candidates], , drop = FALSE]
    topic_vec <- topic_emb[i, ]    # the topic reference vector

    # word_doc_sim: cosine similarity of each candidate to the topic reference.
    # Shape: n_cand.  This is the "relevance" score.
    # Matrix multiplication (cand_emb × topic_vec) works because both are L2-
    # normalised, so dot product = cosine similarity.
    word_doc_sim  <- as.vector(cand_emb %*% topic_vec)

    # word_word_sim: pairwise cosine similarities between all candidates.
    # Shape: n_cand × n_cand.  Entry [i, j] = sim(candidate_i, candidate_j).
    # We use this to penalise candidates that are similar to already-chosen terms.
    word_word_sim <- cand_emb %*% t(cand_emb)

    n_select  <- min(as.integer(top_n), n_cand)
    selected  <- integer(n_select)    # indices (into candidates[]) of chosen terms
    remaining <- seq_len(n_cand)      # indices of candidates not yet chosen

    # Step k=1: always pick the most relevant candidate (no redundancy yet).
    selected[1L] <- which.max(word_doc_sim)
    remaining    <- remaining[remaining != selected[1L]]

    # Steps k=2…n_select: pick greedily by MMR score.
    for (k in seq(2L, n_select)) {
      if (length(remaining) == 0L) break

      # Relevance of each remaining candidate to the topic reference.
      relevance  <- word_doc_sim[remaining]

      # Redundancy of each remaining candidate relative to ALREADY SELECTED terms.
      # For each remaining candidate, take its maximum similarity to any selected
      # term.  The max (not mean) ensures we strongly penalise any candidate that
      # closely mirrors even ONE already-chosen term — preventing near-duplicates.
      redundancy <- apply(
        word_word_sim[remaining, selected[seq_len(k - 1L)], drop = FALSE],
        1L, max
      )

      # MMR score: trade off relevance against redundancy, controlled by diversity.
      best        <- remaining[which.max((1 - diversity) * relevance -
                                           diversity * redundancy)]
      selected[k] <- best
      remaining   <- remaining[remaining != best]
    }

    selected_terms <- candidates[selected[selected != 0L]]

    # Preserve original c-TF-IDF scores for the selected terms.
    # MMR changes WHICH terms are shown and their ORDER, but not the scores —
    # the score column still reflects how characteristic each term is of the topic.
    orig <- fit$topic_terms[fit$topic_terms$topic == t, ]
    score_lookup <- stats::setNames(orig$score, orig$term)

    data.frame(
      topic = t,
      rank  = seq_along(selected_terms),
      term  = selected_terms,
      score = unname(score_lookup[selected_terms]),
      stringsAsFactors = FALSE
    )
  })

  # --- 4. Rebuild topic_terms (noise topic unchanged) ----------------------
  noise_rows    <- fit$topic_terms[fit$topic_terms$topic == -1L, ]
  updated_terms <- do.call(rbind, c(list(noise_rows), new_rows))
  rownames(updated_terms) <- NULL
  fit$topic_terms <- updated_terms

  # --- 5. Refresh topic labels ---------------------------------------------
  new_labels <- .generate_topic_labels(updated_terms, topics_nonnoise,
                                        nr_words = 4L)
  for (nm in names(new_labels)) fit$topic_labels[[nm]] <- new_labels[[nm]]

  if (verbose)
    message("Done. MMR applied to ", length(topics_nonnoise), " topics.")
  fit
}
