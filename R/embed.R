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
#' S3 generic dispatching on the encoder class.  For local BERT-family models
#' (class `bert_encoder`) see the Details section.  For API-backed models
#' (class `api_embedder`) the function calls the provider's REST endpoint and
#' parameters `max_length`, `device`, `chunk_strategy`, and `chunk_overlap`
#' are ignored.
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
#' **Long-document chunking** (`chunk_strategy`): BERT-family models have a
#' fixed maximum sequence length (usually 512 tokens) and silently truncate
#' longer inputs.  Setting `chunk_strategy = "mean"` (or `"first"`) instead
#' splits each over-length text into overlapping windows of `max_length`
#' tokens, embeds each window, and aggregates:
#' \itemize{
#'   \item `"truncate"` (default): truncate at `max_length`, fast, no overhead.
#'   \item `"mean"`: embed all windows, average their vectors (then normalize
#'     if `normalize = TRUE`). Best quality for long documents.
#'   \item `"first"`: embed only the first window. Good when the lead of a
#'     document (abstract, summary) is the most informative part.
#' }
#' Use `chunk_overlap` to set the number of overlapping tokens between
#' consecutive windows (default 0).
#'
#' @param encoder A loaded encoder: a `bert_encoder` from [load_hf_bert()] /
#'   [load_specter2()], or an `api_embedder` from [load_openai_embedder()] /
#'   [load_cohere_embedder()].
#' @param texts A character vector of strings to embed.
#' @param batch_size Number of texts (or chunks) per forward pass.
#' @param max_length Truncate/chunk token sequences to this length (including
#'   special tokens).  Capped at the model's `max_position_embeddings`.
#' @param normalize If `TRUE` (default), L2-normalize each row so cosine
#'   similarity equals dot product.
#' @param device Either `"cpu"` (default) or `"cuda"`.
#' @param prefix String prepended to every text before tokenization.
#'   `NULL` (default) falls back to `encoder$prefix`; `""` disables prefixing
#'   even when the encoder has a stored prefix.
#' @param chunk_strategy One of `"truncate"` (default), `"mean"`, or
#'   `"first"`.  See Details. Ignored for `api_embedder`.
#' @param chunk_overlap Number of token overlap between consecutive windows
#'   when `chunk_strategy != "truncate"`. Default 0.
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
#'
#'   # Long documents: chunk and mean-pool windows
#'   enc <- load_hf_bert("sentence-transformers/all-mpnet-base-v2")
#'   emb <- embed_texts(enc, long_papers, chunk_strategy = "mean",
#'                       chunk_overlap = 32L)
#'
#'   # API-based (no torch required)
#'   enc <- load_openai_embedder()
#'   emb <- embed_texts(enc, docs)
#' }
embed_texts <- function(encoder, texts, ...) UseMethod("embed_texts")

#' @rdname embed_texts
#' @export
embed_texts.bert_encoder <- function(encoder, texts,
                                      batch_size     = 32L,
                                      max_length     = 256L,
                                      normalize      = TRUE,
                                      device         = "cpu",
                                      prefix         = NULL,
                                      chunk_strategy = c("truncate", "mean", "first"),
                                      chunk_overlap  = 0L,
                                      verbose        = interactive()) {
  chunk_strategy <- match.arg(chunk_strategy)

  if (is.null(prefix)) prefix <- encoder$prefix %||% ""
  if (nzchar(prefix)) texts <- paste0(prefix, texts)

  if (chunk_strategy != "truncate") {
    return(.embed_chunked(encoder, texts, batch_size, max_length, normalize,
                          device, chunk_strategy, as.integer(chunk_overlap),
                          verbose))
  }

  model     <- encoder$model
  tokenizer <- encoder$tokenizer
  pooling   <- encoder$pooling %||% "mean"

  model$eval()
  model$to(device = device)
  tokenizer$enable_padding()
  tokenizer$enable_truncation(max_length)

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
      pooled <- if (pooling == "cls") cls_pool(hidden) else mean_pool(hidden, attn_mask)
      if (normalize) pooled <- torch::nnf_normalize(pooled, p = 2, dim = 2)
    })

    idx        <- idx + 1L
    out[[idx]] <- as.array(pooled$cpu())
    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }
  do.call(rbind, out)
}

#' @export
embed_texts.default <- function(encoder, texts, ...) {
  stop("No embed_texts method for class '",
       paste(class(encoder), collapse = "/"), "'.\n",
       "Use load_hf_bert(), load_specter2(), load_openai_embedder(), ",
       "or load_cohere_embedder() to create a compatible encoder.")
}

# -----------------------------------------------------------------------------
# Long-document chunking
# -----------------------------------------------------------------------------

.embed_chunked <- function(encoder, texts, batch_size, max_length, normalize,
                            device, strategy, overlap, verbose) {
  model     <- encoder$model
  tokenizer <- encoder$tokenizer
  pooling   <- encoder$pooling %||% "mean"

  model$eval()
  model$to(device = device)

  # Tokenize all texts with the model's hard maximum (effectively no truncation
  # for any text that fits within positional embeddings)
  model_max <- encoder$config$max_position_embeddings %||% 512L
  tokenizer$enable_padding()
  tokenizer$enable_truncation(model_max)
  all_enc <- tokenizer$encode_batch(texts)

  # Build a flat list of all chunks with their origin index
  chunk_ids    <- list()
  chunk_masks  <- list()
  chunk_origin <- integer(0)

  for (i in seq_along(texts)) {
    ids    <- all_enc[[i]]$ids
    n_toks <- length(ids)

    if (n_toks <= max_length) {
      chunk_ids    <- c(chunk_ids,    list(ids))
      chunk_masks  <- c(chunk_masks,  list(all_enc[[i]]$attention_mask))
      chunk_origin <- c(chunk_origin, i)
    } else {
      # Strip special tokens from the body, slide a window, re-add specials
      cls_id <- ids[1L]
      sep_id <- ids[n_toks]
      body   <- ids[seq(2L, n_toks - 1L)]

      body_size <- max_length - 2L        # slots for content tokens
      stride    <- max(1L, body_size - overlap)
      starts    <- seq(1L, length(body), by = stride)
      if (strategy == "first") starts <- starts[1L]

      for (s in starts) {
        e     <- min(s + body_size - 1L, length(body))
        chunk <- c(cls_id, body[s:e], sep_id)
        chunk_ids    <- c(chunk_ids,    list(chunk))
        chunk_masks  <- c(chunk_masks,  list(rep(1L, length(chunk))))
        chunk_origin <- c(chunk_origin, i)
      }
    }
  }

  # Batch-embed all chunks
  n_chunks <- length(chunk_ids)
  emb_list <- vector("list", ceiling(n_chunks / batch_size))
  cidx <- 0L

  for (start in seq(1L, n_chunks, by = batch_size)) {
    end   <- min(start + batch_size - 1L, n_chunks)
    b_ids <- chunk_ids[start:end]
    b_msk <- chunk_masks[start:end]

    Lmax  <- max(vapply(b_ids, length, integer(1L)))
    pad   <- function(v) c(v, rep(0L, Lmax - length(v)))
    ids_m <- do.call(rbind, lapply(b_ids, pad))
    msk_m <- do.call(rbind, lapply(b_msk, pad))

    input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())$to(device = device)
    attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())$to(device = device)

    torch::with_no_grad({
      hidden <- model(input_ids, attn_mask)
      pooled <- if (pooling == "cls") cls_pool(hidden) else mean_pool(hidden, attn_mask)
    })

    cidx <- cidx + 1L
    emb_list[[cidx]] <- as.array(pooled$cpu())
    if (verbose) message(sprintf("  embedded %d / %d chunks", end, n_chunks))
  }

  all_chunk_emb <- do.call(rbind, emb_list)

  # Aggregate chunks back to one vector per original text
  n      <- length(texts)
  hidden_size <- ncol(all_chunk_emb)
  result <- matrix(0, nrow = n, ncol = hidden_size)

  for (i in seq_len(n)) {
    cidx_i <- which(chunk_origin == i)
    if (length(cidx_i) == 1L) {
      result[i, ] <- all_chunk_emb[cidx_i, ]
    } else {
      result[i, ] <- colMeans(all_chunk_emb[cidx_i, , drop = FALSE])
    }
  }

  if (normalize) {
    norms <- sqrt(rowSums(result^2))
    norms[norms == 0] <- 1
    result <- result / norms
  }
  result
}
