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

#' Embed a vector of texts to a numeric matrix
#'
#' Tokenizes, runs the encoder forward pass, mean-pools over tokens (masked
#' by the attention mask), and optionally L2-normalizes the result.  Batches
#' inputs for memory efficiency.
#'
#' @param encoder A loaded encoder, as returned by [load_hf_bert()].
#' @param texts A character vector of strings to embed.
#' @param batch_size Number of texts to tokenize and forward together.
#'   Larger uses more memory but runs faster.
#' @param max_length Truncate token sequences to this length (including
#'   special tokens like `[CLS]` and `[SEP]`).  Capped at the model's
#'   `max_position_embeddings`.
#' @param normalize If `TRUE` (default), L2-normalize each row of the output
#'   so cosine similarity equals dot product.  Leave on for retrieval,
#'   clustering, or topic modeling.
#' @param device Either `"cpu"` (default) or `"cuda"` if a GPU is available
#'   and libtorch was installed with CUDA support.
#' @param verbose If `TRUE`, prints batch progress.  Defaults to interactive
#'   sessions only.
#' @return A numeric matrix with `length(texts)` rows and `hidden_size`
#'   columns.
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("First sentence.", "Second sentence."))
#'   # Cosine similarity = dot product (because rows are L2-normalized)
#'   emb %*% t(emb)
#' }
embed_texts <- function(encoder, texts,
                        batch_size = 32, max_length = 256,
                        normalize = TRUE, device = "cpu",
                        verbose = interactive()) {
  model     <- encoder$model
  tokenizer <- encoder$tokenizer

  model$eval()
  model$to(device = device)
  tokenizer$enable_padding()
  tokenizer$enable_truncation(max_length)

  n <- length(texts)
  out <- vector("list", ceiling(n / batch_size))
  idx <- 0

  for (start in seq(1, n, by = batch_size)) {
    end <- min(start + batch_size - 1, n)
    batch <- texts[start:end]
    enc <- tokenizer$encode_batch(batch)

    ids   <- lapply(enc, function(e) e$ids)
    masks <- lapply(enc, function(e) e$attention_mask)
    Lmax  <- max(vapply(ids, length, integer(1)))
    pad   <- function(v) c(v, rep(0L, Lmax - length(v)))
    ids_m <- do.call(rbind, lapply(ids,   pad))
    msk_m <- do.call(rbind, lapply(masks, pad))

    input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())$
      to(device = device)
    attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())$
      to(device = device)

    torch::with_no_grad({
      hidden <- model(input_ids, attn_mask)
      pooled <- mean_pool(hidden, attn_mask)
      if (normalize) pooled <- torch::nnf_normalize(pooled, p = 2, dim = 2)
    })

    idx <- idx + 1
    out[[idx]] <- as.array(pooled$cpu())
    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }
  do.call(rbind, out)
}
