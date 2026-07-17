# =============================================================================
# tokenizers.R  --  Fallback tokenizer for older BERT models that ship vocab.txt
# but not tokenizer.json.  Wraps the CRAN `wordpiece` package to mimic the
# interface of `tok::tokenizer` so the rest of the pipeline doesn't have to
# know which kind of tokenizer is in use.
# =============================================================================

#' Create a WordPiece tokenizer for BERT models that lack `tokenizer.json`
#'
#' Older BERT-family models (SciBERT, BioBERT, the original BERT-base, etc.)
#' ship a `vocab.txt` file but not the HuggingFace fast-tokenizer format.
#' This function builds a tokenizer object exposing the same methods as
#' `tok::tokenizer$from_pretrained()` (`encode_batch`, `enable_padding`,
#' `enable_truncation`) so it can be used interchangeably with
#' [embed_texts()].
#'
#' Uses the CRAN package `wordpiece` for the WordPiece algorithm itself.
#'
#' @param vocab_path Path to the model's `vocab.txt` file.
#' @param do_lower_case Logical, whether to lowercase input before
#'   tokenization.  If `NULL` (the default), inferred from the vocabulary:
#'   if no tokens start with an uppercase letter, the vocab is treated as
#'   uncased.  Pass an explicit value if `tokenizer_config.json` specifies it.
#' @param max_length Default maximum sequence length, including the two
#'   special tokens `[CLS]` and `[SEP]`.  Can be overridden later via
#'   `tokenizer$enable_truncation()`.
#' @return A list with methods `encode_batch(texts)`, `enable_padding()`,
#'   and `enable_truncation(max_length)`.
#' @export
#' @examples
#' \dontrun{
#'   vocab <- hfhub::hub_download("allenai/scibert_scivocab_cased", "vocab.txt")
#'   tk <- make_wordpiece_tokenizer(vocab)
#'   tk$encode_batch(c("This is a sentence.", "Another one."))
#' }
make_wordpiece_tokenizer <- function(vocab_path, do_lower_case = NULL,
                                     max_length = 512L) {
  if (!requireNamespace("wordpiece", quietly = TRUE)) {
    stop("Please install.packages(\"wordpiece\") for WordPiece tokenization.")
  }
  vocab <- wordpiece::load_vocab(vocab_path)

  # Casedness: prefer the explicit do_lower_case (from tokenizer_config.json
  # if available), otherwise infer from the vocab content.
  is_cased <- attr(vocab, "is_cased")
  if (is.null(is_cased)) is_cased <- TRUE
  if (is.null(do_lower_case)) do_lower_case <- !is_cased

  vocab_chars <- as.character(vocab)
  vocab_lookup <- stats::setNames(seq_along(vocab_chars) - 1L, vocab_chars)
  for (tok_name in c("[CLS]", "[SEP]", "[PAD]", "[UNK]")) {
    if (!tok_name %in% names(vocab_lookup))
      stop("Vocab is missing required special token: ", tok_name)
  }
  cls_id <- unname(vocab_lookup[["[CLS]"]])
  sep_id <- unname(vocab_lookup[["[SEP]"]])

  state <- new.env(parent = emptyenv())
  state$max_length <- max_length

  encode_one <- function(txt) {
    if (do_lower_case) txt <- tolower(txt)
    res <- wordpiece::wordpiece_tokenize(text = txt, vocab = vocab)
    if (is.list(res)) res <- res[[1]]
    ids <- as.integer(unname(res))
    if (length(ids) > state$max_length - 2L)
      ids <- ids[seq_len(state$max_length - 2L)]
    ids <- c(cls_id, ids, sep_id)
    list(ids = ids, attention_mask = rep(1L, length(ids)))
  }

  list(
    encode_batch      = function(texts) lapply(texts, encode_one),
    enable_padding    = function() invisible(NULL),  # embed_texts pads
    enable_truncation = function(max_length) {
      state$max_length <- as.integer(max_length); invisible(NULL)
    }
  )
}
