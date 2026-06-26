# =============================================================================
# pos_representation.R — POS-filtered document-term matrices via udpipe.
#
# Allows the representation model to focus on specific grammatical categories
# (NOUN, VERB, ADJ …) or multi-word POS patterns (ADJ+NOUN, NOUN+NOUN …)
# instead of the default bag-of-words vocabulary.
# =============================================================================

#' Construct a POS-based representation model
#'
#' Use as the \code{representation_model} argument in \code{\link{fit_bertopic}}
#' to restrict the c-TF-IDF vocabulary to tokens whose Universal POS (UPOS) tag
#' is in \code{pos}, or to phrases that match specific POS-tag sequences.
#'
#' Requires the \pkg{udpipe} package
#' (\code{install.packages("udpipe")}).  A language model (~15 MB) is
#' downloaded from the Universal Dependencies collection on first use and
#' cached in \code{model_dir}.
#'
#' Common UPOS tags:
#' \describe{
#'   \item{\code{NOUN}}{Common nouns (default, with \code{PROPN})}
#'   \item{\code{PROPN}}{Proper nouns}
#'   \item{\code{VERB}}{Main verbs — use for action-focused topics}
#'   \item{\code{ADJ}}{Adjectives}
#'   \item{\code{ADV}}{Adverbs}
#' }
#'
#' @param pos Character vector of UPOS tags to retain as single tokens.
#'   Default \code{c("NOUN","PROPN")}.  Set to \code{NULL} to skip single-token
#'   filtering and rely on \code{patterns} alone.
#' @param patterns Optional list of UPOS sequences to extract as multi-word
#'   phrases, e.g. \code{list(c("ADJ","NOUN"), c("NOUN","NOUN"))}.  Matched
#'   consecutive tokens are joined with \code{"_"} and added to the vocabulary.
#' @param lemmatize Use lemmatised forms rather than surface tokens (default
#'   \code{TRUE}).
#' @param language Language name understood by udpipe, e.g. \code{"english"},
#'   \code{"dutch"}, \code{"french"}.  Default \code{"english"}.
#' @param model_dir Directory to cache the udpipe language model.  Defaults to
#'   a session-scoped temporary directory.
#' @param min_df Minimum document frequency for a term to survive (default 2).
#' @param max_df_frac Maximum document-frequency fraction (default 0.95).
#' @return An object of class \code{pos_representation}.
#' @seealso \code{\link{pos_dtm}}, \code{\link{fit_bertopic}},
#'   \code{\link{cvalue_representation}}
#' @export
pos_representation <- function(pos         = c("NOUN", "PROPN"),
                                patterns    = NULL,
                                lemmatize   = TRUE,
                                language    = "english",
                                model_dir   = NULL,
                                min_df      = 2L,
                                max_df_frac = 0.95) {
  if (is.null(pos) && is.null(patterns))
    stop("At least one of 'pos' or 'patterns' must be specified.")
  structure(
    list(pos = pos, patterns = patterns, lemmatize = lemmatize,
         language = language, model_dir = model_dir,
         min_df = as.integer(min_df), max_df_frac = max_df_frac),
    class = "pos_representation"
  )
}

#' Build a POS-filtered document-term matrix
#'
#' Annotates each document in \code{docs} with a udpipe language model, keeps
#' only tokens whose UPOS tag is in \code{pos} (and/or phrases matching
#' \code{patterns}), then builds a sparse document-term matrix from the
#' surviving tokens.  The result is compatible with \code{\link{c_tf_idf}}.
#'
#' @param docs Character vector of documents.
#' @param pos UPOS tags to retain.  Default \code{c("NOUN","PROPN")}.
#' @param patterns Optional list of UPOS sequences for multi-word phrases.
#' @param lemmatize Use lemmatised forms (default \code{TRUE}).
#' @param language udpipe language name (default \code{"english"}).
#' @param model_dir Directory for the cached language model.
#' @param min_df Minimum document frequency (default 2).
#' @param max_df_frac Maximum document-frequency fraction (default 0.95).
#' @param extra_stopwords Character vector of additional words to exclude.
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return A sparse \code{dgCMatrix}: documents × POS-filtered vocabulary.
#' @seealso \code{\link{pos_representation}}, \code{\link{build_dtm}}
#' @export
pos_dtm <- function(docs,
                    pos             = c("NOUN", "PROPN"),
                    patterns        = NULL,
                    lemmatize       = TRUE,
                    language        = "english",
                    model_dir       = NULL,
                    min_df          = 2L,
                    max_df_frac     = 0.95,
                    extra_stopwords = character(0L),
                    verbose         = TRUE) {
  .check_udpipe()

  model_path <- .udpipe_model(language, model_dir, verbose)
  model      <- udpipe::udpipe_load_model(model_path)

  if (verbose) message("  Annotating ", length(docs), " documents with udpipe...")
  annot_raw <- udpipe::udpipe_annotate(model, x = docs,
                                        doc_id = seq_along(docs))
  annot <- as.data.frame(annot_raw, stringsAsFactors = FALSE)

  # Drop multi-word token spans (token_id like "3-4") and unparsed tokens
  annot <- annot[!grepl("-", annot$token_id, fixed = TRUE), ]
  annot <- annot[!is.na(annot$upos) & nzchar(trimws(annot$upos)), ]

  word_col <- if (lemmatize) "lemma" else "token"
  annot[[word_col]] <- tolower(trimws(annot[[word_col]]))
  annot <- annot[nchar(annot[[word_col]]) >= 2L, ]

  # Collect per-document token lists
  all_tokens <- vector("list", length(docs))
  for (i in seq_along(docs)) all_tokens[[i]] <- character(0L)

  if (!is.null(pos)) {
    pos_rows <- annot[annot$upos %in% pos, ]
    by_doc   <- split(pos_rows[[word_col]], as.integer(pos_rows$doc_id))
    for (nm in names(by_doc)) {
      idx <- as.integer(nm)
      if (idx >= 1L && idx <= length(docs))
        all_tokens[[idx]] <- c(all_tokens[[idx]], by_doc[[nm]])
    }
  }

  if (!is.null(patterns)) {
    phrases <- .extract_pos_patterns(annot, patterns, word_col)
    if (nrow(phrases) > 0L) {
      by_doc <- split(phrases$phrase, phrases$doc_id)
      for (nm in names(by_doc)) {
        idx <- as.integer(nm)
        if (idx >= 1L && idx <= length(docs))
          all_tokens[[idx]] <- c(all_tokens[[idx]], by_doc[[nm]])
      }
    }
  }

  if (length(extra_stopwords) > 0L) {
    sw_lower <- tolower(extra_stopwords)
    all_tokens <- lapply(all_tokens, function(t) t[!t %in% sw_lower])
  }

  .tokens_to_dtm(all_tokens, length(docs), as.integer(min_df), max_df_frac)
}

# ---------- internal helpers --------------------------------------------------

.check_udpipe <- function() {
  if (!requireNamespace("udpipe", quietly = TRUE))
    stop(
      "Package 'udpipe' is required for POS-based representation.\n",
      "Install it with:  install.packages(\"udpipe\")"
    )
}

.udpipe_model <- function(language, model_dir, verbose) {
  if (is.null(model_dir))
    model_dir <- file.path(tempdir(), "rhobots_udpipe")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  cached <- list.files(model_dir,
                        pattern = paste0("^", language, ".*\\.udpipe$"),
                        full.names = TRUE)
  if (length(cached) > 0L) return(cached[1L])

  if (verbose)
    message("  Downloading udpipe model for '", language, "' to ", model_dir,
            "\n  (one-time download, ~15 MB)...")
  info <- tryCatch(
    udpipe::udpipe_download_model(language = language, model_dir = model_dir),
    error = function(e) stop(
      "Failed to download udpipe model for '", language, "': ",
      conditionMessage(e),
      "\nCheck your internet connection or set 'model_dir' to a directory ",
      "containing an existing <language>*.udpipe file."
    )
  )
  info$file_model
}

.extract_pos_patterns <- function(annot, patterns, word_col) {
  result_list <- list()
  idx <- 0L

  doc_ids <- unique(annot$doc_id)
  for (did in doc_ids) {
    doc_rows <- annot[annot$doc_id == did, ]
    sent_ids <- unique(doc_rows$sentence_id)

    for (sid in sent_ids) {
      sent <- doc_rows[doc_rows$sentence_id == sid, ]
      # Sort by numeric token position (guard against character ordering)
      sent <- sent[order(as.integer(sent$token_id)), ]
      upos  <- sent$upos
      words <- sent[[word_col]]
      n_tok <- nrow(sent)

      for (patt in patterns) {
        plen <- length(patt)
        if (n_tok < plen) next
        for (start in seq_len(n_tok - plen + 1L)) {
          if (identical(upos[start:(start + plen - 1L)], patt)) {
            phrase <- paste(words[start:(start + plen - 1L)], collapse = "_")
            idx <- idx + 1L
            result_list[[idx]] <- list(doc_id = as.integer(did), phrase = phrase)
          }
        }
      }
    }
  }

  if (idx == 0L)
    return(data.frame(doc_id = integer(0L), phrase = character(0L),
                      stringsAsFactors = FALSE))

  data.frame(
    doc_id = vapply(result_list, `[[`, integer(1L),   "doc_id"),
    phrase  = vapply(result_list, `[[`, character(1L), "phrase"),
    stringsAsFactors = FALSE
  )
}

.tokens_to_dtm <- function(token_lists, n_docs, min_df, max_df_frac) {
  vocab <- sort(unique(unlist(token_lists, use.names = FALSE)))
  if (length(vocab) == 0L)
    stop("POS filtering left no tokens. Try broadening the 'pos' argument.")

  df_count <- tabulate(
    match(unlist(lapply(token_lists, unique), use.names = FALSE), vocab),
    nbins = length(vocab)
  )
  keep  <- df_count >= min_df & df_count <= floor(max_df_frac * n_docs)
  vocab <- vocab[keep]
  if (length(vocab) == 0L)
    stop("No POS-filtered terms survived document-frequency filtering. ",
         "Lower 'min_df' or broaden 'pos'.")
  vocab_idx <- stats::setNames(seq_along(vocab), vocab)

  rows <- integer(0L); cols <- integer(0L); vals <- integer(0L)
  for (i in seq_len(n_docs)) {
    toks <- token_lists[[i]]
    toks <- toks[toks %in% names(vocab_idx)]
    if (!length(toks)) next
    tab  <- table(toks)
    rows <- c(rows, rep(i, length(tab)))
    cols <- c(cols, unname(vocab_idx[names(tab)]))
    vals <- c(vals, as.integer(tab))
  }

  Matrix::sparseMatrix(i = rows, j = cols, x = vals,
                       dims = c(n_docs, length(vocab)),
                       dimnames = list(NULL, vocab))
}
