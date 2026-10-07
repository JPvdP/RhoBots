# =============================================================================
# embed.R  --  Convert a loaded encoder + a character vector into a numeric matrix
#
# WHAT THIS FILE DOES
# -------------------
# After load_hf_bert() has set up the model and tokenizer, the functions in
# this file handle the actual inference pipeline:
#
#   texts (character vector)
#     |
#     v
#   tokenizer  -> integer token IDs + attention mask
#     |
#     v
#   model forward pass  -> hidden states (B, L, hidden_size)
#     |
#     v
#   pooling  -> one vector per sentence  (B, hidden_size)
#     |
#     v
#   L2 normalisation  -> unit-norm rows  (B, hidden_size)
#     |
#     v
#   numeric matrix  (n_texts x hidden_size)
#
# The embed_texts() function is an S3 generic: calling it on a bert_encoder
# dispatches to embed_texts.bert_encoder(), and on an api_embedder to
# embed_texts.api_embedder() (defined in api_embedders.R).
# =============================================================================


# -----------------------------------------------------------------------------
# mean_pool
#
# BERT processes every token individually.  After the final transformer layer,
# each token has its own hidden-state vector of shape (hidden_size,).  To get
# ONE vector representing the whole sentence we average ("pool") the token
# vectors  --  but we must exclude padding tokens, which are artificial fillers
# added to make all sequences in a batch the same length.
#
# HOW IT WORKS:
#
#   Step 1  --  convert the attention mask to float and add a trailing dimension:
#             shape (B, L) -> (B, L, 1), so it broadcasts against (B, L, H).
#
#   Step 2  --  mask hidden states: multiply hidden * mask.
#             Padding positions (mask=0) become zero vectors.
#             Real token positions (mask=1) are unchanged.
#
#   Step 3  --  sum along the sequence dimension (dim 2) -> (B, H).
#
#   Step 4  --  divide by the number of real tokens per row.
#             clamp(min=1e-9) avoids division by zero if a row is all padding
#             (extremely rare but possible with empty strings).
#
# RESULT: a (B, H) tensor where each row is the average of the real token
# vectors for that sentence.
# -----------------------------------------------------------------------------

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
#' @examples
#' \dontrun{
#'   enc    <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   tokens <- enc$tokenizer$encode_batch(c("hello world"))
#'   ids    <- torch::torch_tensor(matrix(tokens[[1L]]$ids, nrow = 1L),
#'                                  dtype = torch::torch_long())
#'   mask   <- torch::torch_tensor(matrix(tokens[[1L]]$attention_mask, nrow = 1L),
#'                                  dtype = torch::torch_long())
#'   hidden <- enc$model(ids, mask)
#'   mean_pool(hidden, mask)
#' }
#' @export
mean_pool <- function(hidden, attention_mask) {
  # Expand mask to (B, L, 1) for broadcasting against (B, L, H)
  m <- attention_mask$to(dtype = torch::torch_float())$unsqueeze(-1)

  # Sum of real-token hidden states per sentence: shape (B, H)
  s <- (hidden * m)$sum(dim = 2)

  # Count of real tokens per sentence (avoid /0 with clamp)
  d <- m$sum(dim = 2)$clamp(min = 1e-9)

  s / d   # element-wise division: (B, H) / (B, 1) -> (B, H)
}


# -----------------------------------------------------------------------------
# cls_pool
#
# Some models (e.g. certain BGE variants) are trained to put the sentence
# representation in the very first token's hidden state rather than averaging
# all tokens.  This first token is the [CLS] (classification) token inserted
# by the tokenizer at position 0 (0-based) / 1 (1-based in R).
#
# select(dim, index) extracts a slice along the given dimension.
# Here dim=2 is the sequence dimension, index=1 selects the first position
# (using R's 1-based indexing), returning shape (B, H).
# -----------------------------------------------------------------------------

#' CLS-token pooling: extract the CLS hidden state as the sentence vector
#'
#' Returns the first token's hidden state `(B, H)` from a `(B, L, H)` tensor.
#' Used for models whose `1_Pooling/config.json` sets
#' `pooling_mode_cls_token = true` (e.g. some BGE and GTE variants).
#'
#' @param hidden A 3-D `torch_tensor` of shape `(batch, seq_len, hidden)`.
#' @return A 2-D `torch_tensor` of shape `(batch, hidden)`.
#' @examples
#' \dontrun{
#'   enc    <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   tokens <- enc$tokenizer$encode_batch(c("hello world"))
#'   ids    <- torch::torch_tensor(matrix(tokens[[1L]]$ids, nrow = 1L),
#'                                  dtype = torch::torch_long())
#'   mask   <- torch::torch_tensor(matrix(tokens[[1L]]$attention_mask, nrow = 1L),
#'                                  dtype = torch::torch_long())
#'   hidden <- enc$model(ids, mask)
#'   cls_pool(hidden)
#' }
#' @export
cls_pool <- function(hidden) {
  hidden$select(2L, 1L)   # select position 1 (= [CLS]) along the sequence dim
}


# -----------------------------------------------------------------------------
# embed_texts  --  S3 generic
#
# Using R's S3 dispatch system, calling embed_texts(encoder, texts) routes to:
#   embed_texts.bert_encoder   for models loaded with load_hf_bert() / load_specter2()
#   embed_texts.api_embedder   for API-backed models (OpenAI, Cohere)
#   embed_texts.default        for anything else  --  produces a helpful error
#
# This means users always call the same function regardless of encoder type,
# which is the core idea of object-oriented dispatch.
# -----------------------------------------------------------------------------

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
#' @param gc_every Release memory held by finished batches every `gc_every`
#'   batches (default 1).  Torch tensors live outside R's heap, so R does not
#'   see how much memory they use and may not free them in time on large
#'   corpora.  Higher values trade a little memory for slightly less overhead.
#'   Unless `gc_every = 0`, memory is also released after every encoder layer,
#'   so only one layer's attention tensors are alive at a time.  `0` disables
#'   all cleanup.  Ignored for `api_embedder`.
#' @param ... Not used; retained for S3 method compatibility.
#' @return A numeric matrix with `length(texts)` rows and `hidden_size` cols.
#' @export
#' @examples
#' \dontrun{
#'   # General model  --  no prefix needed
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("First sentence.", "Second sentence."))
#'
#'   # BGE model  --  set prefix at load time (applied automatically)
#'   enc <- load_hf_bert("BAAI/bge-base-en-v1.5",
#'                        prefix = "Represent this sentence: ")
#'   emb <- embed_texts(enc, docs)
#'
#'   # E5 model  --  passage prefix for documents, query prefix for queries
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


# -----------------------------------------------------------------------------
# embed_texts.bert_encoder  --  Main inference loop for local models
#
# OVERVIEW OF STEPS (for the default "truncate" strategy):
#
#  1. Prepend prefix   --  instruction-tuned models like BGE or E5 require a
#     short phrase before each text to signal the embedding task.
#
#  2. Set up tokenizer   --  enable padding (so all texts in a batch get the
#     same length) and truncation (so no text exceeds max_length tokens).
#
#  3. Batch loop   --  process texts in chunks of batch_size to control memory.
#     Within each batch:
#
#     a. Tokenize   --  the tok package returns a list of Encoding objects, each
#        carrying:
#          $ids           : integer vector of token IDs
#          $attention_mask: 1 for real tokens, 0 for padding
#
#     b. Build matrices   --  pad every sequence to Lmax (the longest in this
#        batch) by appending 0s.  Stack rows into an integer matrix.
#
#     c. Convert to tensors   --  wrap the R matrices as torch_long() tensors
#        and move them to the chosen device (CPU or GPU).
#
#     d. Forward pass   --  run through the model architecture defined in
#        architecture.R.  Output shape: (batch_size, Lmax, hidden_size).
#
#     e. Pool   --  collapse the sequence dimension:
#          mean pool : weighted average over non-padding tokens
#          CLS pool  : extract the first token's hidden state
#
#     f. L2 normalise   --  scale each row vector to unit length.  After
#        normalisation, cosine_similarity(a, b) = dot(a, b), making
#        similarity comparisons very cheap (just matrix multiplication).
#
#     g. Back to R   --  convert the GPU tensor to a standard R numeric array.
#
#  4. Free memory   --  torch tensors live outside R's heap, so R's garbage
#     collector cannot see how large they are and rarely runs on its own.
#     Every `gc_every` batches we force a collection (and empty the CUDA
#     cache) so finished batches are actually released.
#
#  5. Restore order   --  rows are written into a pre-allocated matrix in
#     sorted order and permuted back to the input order at the end.
# -----------------------------------------------------------------------------

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
                                      verbose        = interactive(),
                                      gc_every       = 1L,
                                      ...) {
  chunk_strategy <- match.arg(chunk_strategy)
  gc_every       <- as.integer(gc_every)

  # Resolve the prefix: explicit argument beats encoder-stored value.
  if (is.null(prefix)) prefix <- encoder$prefix %||% ""
  if (nzchar(prefix)) texts <- paste0(prefix, texts)

  # Delegate to the chunking path for long-document strategies.
  if (chunk_strategy != "truncate") {
    return(.embed_chunked(encoder, texts, batch_size, max_length, normalize,
                          device, chunk_strategy, as.integer(chunk_overlap),
                          verbose, gc_every))
  }

  model     <- encoder$model
  tokenizer <- encoder$tokenizer
  pooling   <- encoder$pooling %||% "mean"

  model$eval()
  model$to(device = device)

  tokenizer$enable_padding()
  tokenizer$enable_truncation(max_length)

  n <- length(texts)

  # --- Smart batching: sort by approximate length to minimise padding waste ---
  # Texts within a batch are padded to the longest sequence in that batch.
  # Sorting by character length (a cheap proxy for token length) groups similar-
  # length texts together, dramatically reducing wasted padding computation.
  # We record the original order so results are returned in the input order.
  order_idx    <- order(nchar(texts, type = "bytes"))
  texts_sorted <- texts[order_idx]

  # Pre-allocate the result matrix  --  avoids repeated rbind across batches.
  # hidden_size is not known until after the first forward pass, so we fill it in then.
  result  <- NULL
  n_batch <- 0L

  for (start in seq(1L, n, by = batch_size)) {
    end   <- min(start + batch_size - 1L, n)

    enc   <- tokenizer$encode_batch(texts_sorted[start:end])
    batch_arr <- .forward_batch(model,
                                lapply(enc, function(e) e$ids),
                                lapply(enc, function(e) e$attention_mask),
                                pooling, normalize, device,
                                layer_gc = gc_every > 0L)
    rm(enc)

    # Allocate result matrix on first batch now that hidden_size is known.
    # Write rows straight to their original positions, so no reordering copy
    # of the full result is needed at the end.
    if (is.null(result)) result <- matrix(0, nrow = n, ncol = ncol(batch_arr))
    result[order_idx[start:end], ] <- batch_arr

    n_batch <- n_batch + 1L
    if (gc_every > 0L && n_batch %% gc_every == 0L) .release_memory(device)

    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }
  .release_memory(device)

  result
}


# -----------------------------------------------------------------------------
# .forward_batch  --  one forward pass, returned as a plain R matrix
#
# Keeping the tensors local to this function means they become unreachable
# as soon as it returns, so the next .release_memory() call can free them.
# -----------------------------------------------------------------------------

.forward_batch <- function(model, ids, masks, pooling, normalize, device,
                           layer_gc = TRUE) {
  # Free each encoder layer's intermediates before the next one runs.
  old <- options(rhobots.layer_gc = layer_gc)
  on.exit(options(old), add = TRUE)

  Lmax  <- max(vapply(ids, length, integer(1L)))
  pad   <- function(v) c(v, rep(0L, Lmax - length(v)))
  ids_m <- do.call(rbind, lapply(ids,   pad))
  msk_m <- do.call(rbind, lapply(masks, pad))

  input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())$to(device = device)
  attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())$to(device = device)

  torch::with_no_grad({
    hidden <- model(input_ids, attn_mask)
    pooled <- if (pooling == "cls") cls_pool(hidden) else mean_pool(hidden, attn_mask)
    if (normalize) pooled <- torch::nnf_normalize(pooled, p = 2, dim = 2)
  })

  as.matrix(pooled$cpu())
}


# -----------------------------------------------------------------------------
# .release_memory  --  force R to finalise unreachable torch tensors
#
# A torch tensor looks like a tiny external pointer to R, even when it holds
# hundreds of MB.  R therefore sees no memory pressure and postpones garbage
# collection, letting dead batch tensors pile up.  gc() runs their finalisers;
# on CUDA we additionally hand cached blocks back to the driver.
# -----------------------------------------------------------------------------

.release_memory <- function(device) {
  gc(verbose = FALSE)
  if (startsWith(as.character(device), "cuda") && torch::cuda_is_available()) {
    torch::cuda_empty_cache()
  }
  invisible(NULL)
}

#' @rdname embed_texts
#' @export
embed_texts.default <- function(encoder, texts, ...) {
  stop("No embed_texts method for class '",
       paste(class(encoder), collapse = "/"), "'.\n",
       "Use load_hf_bert(), load_specter2(), load_openai_embedder(), ",
       "or load_cohere_embedder() to create a compatible encoder.")
}


# =============================================================================
# Long-document chunking helper
#
# BERT-family models have a hard maximum sequence length (max_position_
# embeddings, typically 512 tokens).  Documents longer than this are
# silently truncated in the default path, losing potentially important content.
#
# STRATEGY
# --------
# Rather than truncating, we work through the corpus in blocks of documents
# (so memory does not scale with corpus size) and for each block:
#   1. Tokenize without truncation (up to the model's hard maximum).
#   2. For texts that fit within max_length: use as-is (one chunk).
#   3. For texts that exceed max_length: slide a window of max_length tokens
#      over the body (stripping and re-adding [CLS] / [SEP] each time).
#   4. Embed the block's chunks in length-sorted batches.
#   5. Aggregate per-document: average all chunk vectors (strategy="mean"),
#      or use only the first chunk's vector (strategy="first").
# Finally the per-document vectors are L2-normalised.
#
# WHY PRESERVE [CLS] AND [SEP]?
# The model was pre-trained to always see these special tokens at the sentence
# boundaries.  Stripping them and re-adding one pair per chunk keeps each
# chunk well-formed from the model's perspective.
#
# WHY ALLOW OVERLAP?
# When chunk_overlap > 0, consecutive windows share some tokens.  This helps
# avoid boundary artifacts where a key sentence is split between two chunks
# and neither chunk captures its full context.
# =============================================================================

.embed_chunked <- function(encoder, texts, batch_size, max_length, normalize,
                            device, strategy, overlap, verbose, gc_every = 1L) {
  model     <- encoder$model
  tokenizer <- encoder$tokenizer
  pooling   <- encoder$pooling %||% "mean"

  model$eval()
  model$to(device = device)

  # Tokenize with the model's absolute maximum  --  essentially "no truncation"
  # for any document that fits within the positional embedding range.
  model_max <- encoder$config$max_position_embeddings %||% 512L
  tokenizer$enable_padding()
  tokenizer$enable_truncation(model_max)

  # Documents are processed in blocks so that only one block's token IDs and
  # chunk lists are held in memory at a time, instead of the whole corpus.
  n          <- length(texts)
  block_size <- max(2048L, batch_size * 64L)
  result     <- NULL
  n_batch    <- 0L

  for (b_start in seq(1L, n, by = block_size)) {
    b_end <- min(b_start + block_size - 1L, n)
    if (verbose && n > block_size) {
      message(sprintf("  documents %d - %d / %d", b_start, b_end, n))
    }

    all_enc <- tokenizer$encode_batch(texts[b_start:b_end])

    # Split every document into chunks.  Each element of `per_doc` holds the
    # chunk token-ID vectors for one document; chunk_origin[i] records which
    # (block-local) document chunk i came from so we can reassemble later.
    per_doc <- lapply(all_enc, function(e) {
      # encode_batch() pads every document to the longest one in the block;
      # keep only the real tokens so [SEP] and the token count are correct.
      ids    <- e$ids[e$attention_mask == 1L]
      n_toks <- length(ids)

      # Document fits in one chunk  --  use as-is.
      if (n_toks <= max_length) return(list(ids))

      # Extract special tokens from the first and last positions.
      # Standard tokenisers always place [CLS] first and [SEP] last.
      cls_id <- ids[1L]
      sep_id <- ids[n_toks]

      # The "body" is everything between [CLS] and [SEP].
      body <- ids[seq(2L, n_toks - 1L)]

      # Each chunk window holds (max_length - 2) body tokens, leaving 2 slots
      # for re-attaching [CLS] and [SEP].
      body_size <- max_length - 2L

      # stride controls how many tokens the window advances each step.
      # stride = body_size - overlap means consecutive chunks share `overlap`
      # tokens at their boundary.  stride must be at least 1.
      stride <- max(1L, body_size - overlap)

      # Starting positions of each window within the body.
      starts <- seq(1L, length(body), by = stride)
      if (strategy == "first") starts <- starts[1L]   # only use the first window

      lapply(starts, function(s) {
        e <- min(s + body_size - 1L, length(body))   # end of this window
        c(cls_id, body[s:e], sep_id)                 # re-add special tokens
      })
    })
    rm(all_enc)

    chunk_ids    <- unlist(per_doc, recursive = FALSE)
    chunk_origin <- rep(seq_along(per_doc), lengths(per_doc))
    rm(per_doc)

    # Every chunk consists of real tokens only, so its mask is all ones.
    # Sort chunks by length for the same padding-efficiency reason as the main path.
    n_chunks     <- length(chunk_ids)
    chunk_order  <- order(lengths(chunk_ids))
    chunk_ids    <- chunk_ids[chunk_order]
    chunk_origin <- chunk_origin[chunk_order]

    chunk_emb <- NULL

    for (start in seq(1L, n_chunks, by = batch_size)) {
      end   <- min(start + batch_size - 1L, n_chunks)
      b_ids <- chunk_ids[start:end]

      # Normalisation is applied to the aggregated document vector, not
      # individual chunks.
      batch_arr <- .forward_batch(model, b_ids,
                                  lapply(b_ids, function(v) rep(1L, length(v))),
                                  pooling, normalize = FALSE, device,
                                  layer_gc = gc_every > 0L)

      if (is.null(chunk_emb)) {
        chunk_emb <- matrix(0, nrow = n_chunks, ncol = ncol(batch_arr))
      }
      chunk_emb[start:end, ] <- batch_arr

      n_batch <- n_batch + 1L
      if (gc_every > 0L && n_batch %% gc_every == 0L) .release_memory(device)

      if (verbose) message(sprintf("  embedded %d / %d chunks", end, n_chunks))
    }

    # Aggregate chunk embeddings back into one vector per document: sum the
    # chunks of each document with rowsum() and divide by the chunk count.
    # This gives equal weight to each window, which is a reasonable default.
    block_res <- rowsum(chunk_emb, chunk_origin, reorder = TRUE) /
                 tabulate(chunk_origin, nbins = b_end - b_start + 1L)

    # L2 normalise the aggregated document vectors.  Done per block (rows are
    # independent) so the full result matrix is never copied.
    if (normalize) {
      norms             <- sqrt(rowSums(block_res^2))
      norms[norms == 0] <- 1   # avoid dividing a zero vector by zero
      block_res         <- block_res / norms
    }

    if (is.null(result)) result <- matrix(0, nrow = n, ncol = ncol(block_res))
    result[b_start:b_end, ] <- block_res
    rm(chunk_ids, chunk_emb, block_res)
  }
  .release_memory(device)

  result   # (n_texts x hidden_size) numeric matrix
}
