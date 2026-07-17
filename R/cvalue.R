# =============================================================================
# cvalue.R  --  C-value multi-word term recognition (Frantzi et al. 2000).
#
# C-value scores candidate multi-word terms by combining raw corpus frequency
# with a penalty for appearing as a constituent of longer terms, weighted by
# log2(term_length).  This surfaces domain-specific compound terms that simple
# frequency counts inflate or that n-gram document-frequency filters miss.
#
# Reference:
#   Frantzi, K., Ananiadou, S., & Mima, H. (2000). Automatic recognition of
#   multi-word terms: the C-value/NC-value method. International Journal on
#   Digital Libraries, 3(2), 115 - 130.
#   https://personalpages.manchester.ac.uk/staff/sophia.ananiadou/ijodl2000.pdf
# =============================================================================

#' Construct a C-value representation model
#'
#' Use as the \code{representation_model} argument in \code{\link{fit_bertopic}}
#' to select the n-gram vocabulary using the C-value algorithm (Frantzi et al.
#' 2000) rather than plain document-frequency filtering.  Unigrams are always
#' kept alongside the C-value-selected multi-word terms.
#'
#' @param max_n Maximum n-gram length to evaluate (default 4).  Longer n-grams
#'   are more informative but also rarer; values of 3 - 5 cover most terminology.
#' @param threshold Minimum C-value score for a multi-word term to be retained
#'   (default 0).  Increase to surface only well-established compound terms.
#' @param min_freq Minimum corpus frequency for a candidate n-gram to be
#'   considered (default 2).
#' @return An object of class \code{cvalue_representation}.
#' @seealso \code{\link{cvalue_terms}}, \code{\link{fit_bertopic}},
#'   \code{\link{pos_representation}}
#' @examples
#' m <- cvalue_representation(max_n = 3L, threshold = 0.5)
#' @export
cvalue_representation <- function(max_n     = 4L,
                                   threshold = 0.0,
                                   min_freq  = 2L) {
  structure(
    list(max_n     = as.integer(max_n),
         threshold = as.numeric(threshold),
         min_freq  = as.integer(min_freq)),
    class = "cvalue_representation"
  )
}

#' Compute C-value scores for candidate multi-word terms
#'
#' Implements the C-value method of Frantzi, Ananiadou & Mima (2000).  For
#' each candidate n-gram \eqn{t} (n >= 2):
#'
#' \deqn{
#'   C\text{-value}(t) = \log_2|t| \times
#'   \begin{cases}
#'     f(t) & \text{if } P(t) = \emptyset \\
#'     f(t) - \dfrac{1}{|P(t)|} \displaystyle\sum_{a \in P(t)} f(a)
#'       & \text{otherwise}
#'   \end{cases}
#' }
#'
#' where \eqn{|t|} is word count, \eqn{f(t)} is corpus frequency, and
#' \eqn{P(t)} is the set of longer candidate terms that contain \eqn{t} as a
#' contiguous sub-sequence.
#'
#' @param docs Character vector of documents.
#' @param max_n Maximum n-gram length to consider (default 4).
#' @param min_freq Minimum corpus frequency for a candidate term (default 2).
#' @param threshold Minimum C-value to include in the returned table (default 0).
#' @param stopwords Character vector of tokens to remove before n-gram
#'   extraction (default none).
#' @return A data frame with columns \code{term} (underscore-separated tokens),
#'   \code{freq}, \code{n_words}, and \code{cvalue}, sorted by descending
#'   C-value.  Terms with \code{cvalue < threshold} are excluded.
#' @references Frantzi, K., Ananiadou, S., & Mima, H. (2000). Automatic
#'   recognition of multi-word terms: the C-value/NC-value method.
#'   \emph{International Journal on Digital Libraries}, 3(2), 115 - 130.
#' @seealso \code{\link{cvalue_representation}}, \code{\link{fit_bertopic}}
#' @examples
#' docs <- c("sea level rise and climate change", "sea level is rising",
#'           "climate change impacts sea level", "Arctic ice melt sea level")
#' cvalue_terms(docs, max_n = 3L, min_freq = 2L)
#' @export
cvalue_terms <- function(docs,
                          max_n     = 4L,
                          min_freq  = 2L,
                          threshold = 0.0,
                          stopwords = character(0L)) {
  max_n <- as.integer(max_n)
  if (max_n < 2L) stop("'max_n' must be at least 2.")

  empty <- data.frame(term = character(), freq = integer(),
                      n_words = integer(), cvalue = numeric(),
                      stringsAsFactors = FALSE)

  # --- Step 1: tokenise -------------------------------------------------------
  # Lowercase, strip punctuation and special characters (keeping letters,
  # digits, and spaces), split on whitespace, and drop tokens shorter than
  # 2 characters or in the stopword list.
  # WHY lowercase + strip?  We want "Sea Level Rise" and "sea level rise" to
  # count as the same term.  Punctuation would fragment compound words.
  tokens <- lapply(docs, function(d) {
    t <- tolower(gsub("[^a-z0-9 ]+", " ", d))
    t <- strsplit(trimws(t), "\\s+", perl = TRUE)[[1L]]
    t[nchar(t) >= 2L & !t %in% stopwords]
  })

  # --- Step 2: count n-grams for n in 2:max_n ---------------------------------
  # For each n (bigrams, trigrams, ..., max_n-grams), slide a window of width n
  # across each document's token sequence and collect the resulting strings.
  # Tokens within an n-gram are joined with "_" as a separator (e.g. "sea_level").
  # After collecting all occurrences across the corpus, table() counts them and
  # we discard n-grams that appear fewer than min_freq times  --  they are too
  # rare to be reliable terminology.
  counts_by_n <- vector("list", max_n - 1L)
  for (n in 2L:max_n) {
    ng_raw <- unlist(lapply(tokens, function(toks) {
      len <- length(toks)
      if (len < n) return(character(0L))
      # Sliding window: for position j, grab tokens[j .. j+n-1].
      vapply(seq_len(len - n + 1L),
             function(j) paste(toks[j:(j + n - 1L)], collapse = "_"),
             character(1L))
    }), use.names = FALSE)

    if (!length(ng_raw)) next
    tab <- table(ng_raw)
    tab <- tab[tab >= min_freq]
    if (!length(tab)) next

    counts_by_n[[n - 1L]] <- data.frame(
      term    = names(tab),
      freq    = as.integer(tab),
      n_words = n,
      stringsAsFactors = FALSE
    )
  }

  present <- !vapply(counts_by_n, is.null, logical(1L))
  if (!any(present)) return(empty)

  ngrams_df <- do.call(rbind, counts_by_n[present])
  rownames(ngrams_df) <- NULL

  # --- Step 3: build containment index ----------------------------------------
  # This is the key data structure for C-value.
  #
  # The C-value formula penalises a candidate term t if it frequently appears
  # as a PART OF a longer term.  Example:
  #   "sea level"       appears 80 times in the corpus
  #   "sea level rise"  appears 60 of those times
  # Without the penalty, "sea level" scores as very frequent (80).  But most
  # of those occurrences are really about "sea level rise"  --  the bigram is not
  # independently informative.  The penalty reduces the effective frequency of
  # "sea level" by the mean frequency of the longer terms that contain it:
  #   C-value("sea level") = log2(2) x (80 - 60) = 1 x 20 = 20.
  #
  # The containment index is a list:
  #   key   = a candidate term t (string, "_"-joined)
  #   value = numeric vector of corpus frequencies of every longer term
  #           that contains t as a contiguous sub-sequence.
  #
  # We only process n-grams with n >= 3, because bigrams (n=2) have no
  # shorter sub-sequences of length >= 2 to penalise.
  containment <- list()  # term -> frequencies of longer containing terms

  for (i in seq_len(nrow(ngrams_df))) {
    n <- ngrams_df$n_words[i]
    if (n < 3L) next   # bigrams cannot contain shorter tracked sub-sequences

    words  <- strsplit(ngrams_df$term[i], "_", fixed = TRUE)[[1L]]
    f_long <- ngrams_df$freq[i]

    # Enumerate all contiguous sub-sequences of length 2 to (n-1).
    # For each, append f_long to that sub-sequence's entry in containment[].
    for (sub_len in 2L:(n - 1L)) {
      for (start in seq_len(n - sub_len + 1L)) {
        sub <- paste(words[start:(start + sub_len - 1L)], collapse = "_")
        containment[[sub]] <- c(containment[[sub]], f_long)
      }
    }
  }

  # --- Step 4: compute C-value ------------------------------------------------
  # C-value(t) = log2(|t|) x adjusted_frequency(t)
  #
  #   adjusted_frequency(t) =
  #     f(t)                      if t is NOT nested inside any longer term
  #     f(t) - mean(f(b) for b in P(t))   otherwise
  #
  # where |t| = number of words, f(t) = corpus frequency, P(t) = the set of
  # longer candidate terms that contain t as a sub-sequence.
  #
  # The log2(|t|) factor rewards longer terms: a 4-word term that survives
  # the penalty scores twice as high as an equally frequent 2-word term.
  # This reflects the intuition that longer specific compound terms carry more
  # information than shorter generic ones.
  #
  # In code: containment[[t]] is NULL when t has no longer parent terms (the
  # "not nested" case), so is.null(cf) -> use raw frequency.
  cvalue_vec <- vapply(seq_len(nrow(ngrams_df)), function(i) {
    t  <- ngrams_df$term[i]
    n  <- ngrams_df$n_words[i]
    f  <- ngrams_df$freq[i]
    cf <- containment[[t]]    # frequencies of all longer terms that contain t
    if (is.null(cf)) log2(n) * f else log2(n) * (f - mean(cf))
  }, numeric(1L))

  ngrams_df$cvalue <- cvalue_vec

  # --- Step 5: filter and return ----------------------------------------------
  ngrams_df <- ngrams_df[ngrams_df$cvalue >= threshold, , drop = FALSE]
  ngrams_df[order(ngrams_df$cvalue, decreasing = TRUE), ]
}

# ---------- internal: build DTM from C-value vocabulary ----------------------

.cvalue_dtm <- function(docs, repr_model, stopwords, verbose) {
  if (verbose) message("    Computing C-value multi-word terms...")
  cv <- cvalue_terms(docs,
                      max_n     = repr_model$max_n,
                      min_freq  = repr_model$min_freq,
                      threshold = repr_model$threshold,
                      stopwords = stopwords)

  if (verbose) {
    n_cv <- nrow(cv)
    message("    Found ", n_cv, " multi-word term(s) with C-value >= ",
            repr_model$threshold, ".")
  }

  # Build full DTM with all requested n-gram lengths
  full_dtm <- build_dtm(docs,
                         stopwords   = stopwords,
                         ngram_range = c(1L, repr_model$max_n))

  if (nrow(cv) == 0L) return(full_dtm)   # no C-value terms; keep unigrams only

  # build_dtm uses spaces between tokens; cvalue_terms uses underscores
  cv_terms_space <- gsub("_", " ", cv$term, fixed = TRUE)

  # Determine word count for each DTM column (unigrams have no spaces)
  col_has_space <- grepl(" ", colnames(full_dtm), fixed = TRUE)

  # Keep: all unigrams + multi-word terms approved by C-value
  keep <- !col_has_space | colnames(full_dtm) %in% cv_terms_space
  full_dtm[, keep, drop = FALSE]
}
