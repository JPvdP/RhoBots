# =============================================================================
# embed.R — Produce sentence embeddings from a loaded encoder.
# =============================================================================

#' Mean-pool token-level hidden states into a sentence vector
#'
#' Takes the encoder's final hidden states `(B, L, H)` and the attention mask
#' `(B, L)`, masks out padding positions, and averages along the sequence
#' dimension to produce `(B, H)` sentence vectors.
#'
#' @param hidden A 3-D `torch_tensor` of shape `(batch, seq_len, hidden)`.
#' @param attention_mask An integer `torch_tensor` of shape `(batch, seq_len)`
#'   with 1 for real tokens and 0 for padding.
#' @return A 2-D `torch_tensor` of shape `(batch, hidden)`.
#' @export
mean_pool <- function(hidden, attention_mask) {
  m <- attention_mask$to(dtype = torch::torch_float())$unsqueeze(-1)
  s <- (hidden * m)$sum(dim = 2)
  d <- m$sum(dim = 2)$clamp(min = 1e-9)
  s / d
}

#' CLS-token pooling: extract the [CLS] hidden state as the sentence vector
#'
#' Returns the first token's hidden state `(B, H)` from a `(B, L, H)` tensor.
#' Used for models whose `1_Pooling/config.json` sets
#' `pooling_mode_cls_token = true` (e.g. some BGE and GTE variants).
#'
#' @param hidden A 3-D `torch_tensor` of shape `(batch, seq_len, hidden)`.
#' @return A 2-D `torch_tensor` of shape `(batch, hidden)`.
#' @export
cls_pool <- function(hidden) {
  hidden$select(2L, 1L)   # position 1 along the sequence dim (1-based)
}

#' Embed a vector of texts to a numeric matrix
#'
#' Tokenizes, runs the encoder forward pass, pools over tokens (mean or CLS,
#' auto-detected from the encoder's pooling configuration), and optionally
#' L2-normalizes the result.  Batches inputs for memory efficiency.
#'
#' An instruction **prefix** can be prepended to every text before
#' tokenization.  This is required for best performance with BGE and E5
#' models:
#' \itemize{
#'   \item BGE (`BAAI/bge-*`): `prefix = "Represent this sentence: "`
#'   \item E5 (`intfloat/e5-*`): `prefix = "passage: "`
#' }
#' If the encoder was loaded with a prefix via [load_hf_bert()]'s `prefix`
#' argument, that value is used automatically and need not be repeated here.
#' Passing `prefix` explicitly always takes precedence.
#'
#' @param encoder A loaded encoder, as returned by [load_hf_bert()] or
#'   [load_specter2()].
#' @param texts A character vector of strings to embed.
#' @param batch_size Number of texts to tokenize and forward together.
#' @param max_length Truncate token sequences to this length (including
#'   special tokens).  Capped at the model's `max_position_embeddings`.
#' @param normalize If `TRUE` (default), L2-normalize each row so cosine
#'   similarity equals dot product.
#' @param device Either `"cpu"` (default) or `"cuda"`.
#' @param prefix String prepended to every text before tokenization.
#'   `NULL` (default) falls back to `encoder$prefix`; `""` disables prefixing
#'   even when the encoder has a stored prefix.
#' @param verbose If `TRUE`, prints batch progress.
#' @return A numeric matrix with `length(texts)` rows and `hidden_size` cols.
#' @export
#' @examples
#' \dontrun{
#'   # General model — no prefix needed
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("First sentence.", "Second sentence."))
#'
#'   # BGE model — set prefix at load time (applied automatically)
#'   enc <- load_hf_bert("BAAI/bge-base-en-v1.5",
#'                        prefix = "Represent this sentence: ")
#'   emb <- embed_texts(enc, docs)
#'
#'   # E5 model — passage prefix for documents, query prefix for queries
#'   enc <- load_hf_bert("intfloat/e5-base-v2", prefix = "passage: ")
#'   doc_emb   <- embed_texts(enc, docs)
#'   query_emb <- embed_texts(enc, queries, prefix = "query: ")
#' }
embed_texts <- function(encoder, texts,
                        batch_size = 32L, max_length = 256L,
                        normalize  = TRUE, device = "cpu",
                        prefix     = NULL,
                        verbose    = interactive()) {
  model     <- encoder$model
  tokenizer <- encoder$tokenizer

  # Resolve prefix: explicit arg > encoder-stored default > none
  if (is.null(prefix)) prefix <- encoder$prefix %||% ""
  # Resolve pooling: encoder-stored setting > mean
  pooling <- encoder$pooling %||% "mean"

  model$eval()
  model$to(device = device)
  tokenizer$enable_padding()
  tokenizer$enable_truncation(max_length)

  if (nzchar(prefix)) texts <- paste0(prefix, texts)

  n   <- length(texts)
  out <- vector("list", ceiling(n / batch_size))
  idx <- 0L

  for (start in seq(1L, n, by = batch_size)) {
    end   <- min(start + batch_size - 1L, n)
    batch <- texts[start:end]
    enc   <- tokenizer$encode_batch(batch)

    ids   <- lapply(enc, function(e) e$ids)
    masks <- lapply(enc, function(e) e$attention_mask)
    Lmax  <- max(vapply(ids, length, integer(1L)))
    pad   <- function(v) c(v, rep(0L, Lmax - length(v)))
    ids_m <- do.call(rbind, lapply(ids,   pad))
    msk_m <- do.call(rbind, lapply(masks, pad))

    input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())$
      to(device = device)
    attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())$
      to(device = device)

    torch::with_no_grad({
      hidden <- model(input_ids, attn_mask)
      pooled <- if (pooling == "cls")
        cls_pool(hidden)
      else
        mean_pool(hidden, attn_mask)
      if (normalize) pooled <- torch::nnf_normalize(pooled, p = 2, dim = 2)
    })

    idx        <- idx + 1L
    out[[idx]] <- as.array(pooled$cpu())
    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }
  do.call(rbind, out)
}
