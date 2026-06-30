# =============================================================================
# architecture.R — BERT-family transformer encoder as torch nn_modules.
#
# Mirrors HuggingFace's BertModel class hierarchy exactly so the parameter
# names in a safetensors checkpoint map one-to-one onto the R submodule
# paths.  Config-driven, so the same code works for any BERT-architecture
# model (BERT-base, MiniLM, SciBERT, BioBERT, ...) — only the dimensions in
# `config.json` change.
#
# These modules are package-internal.  End users go through `load_hf_bert()`.
# =============================================================================

#' BERT input embedding layer (word + position + token-type, then LayerNorm)
#' @keywords internal
#' @noRd
bert_embeddings <- torch::nn_module(
  "BertEmbeddings",
  initialize = function(config) {
    self$word_embeddings <- torch::nn_embedding(config$vocab_size,
                                                config$hidden_size)
    self$position_embeddings <- torch::nn_embedding(config$max_position_embeddings,
                                                    config$hidden_size)
    self$token_type_embeddings <- torch::nn_embedding(config$type_vocab_size,
                                                      config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(input_ids) {
    L <- input_ids$size(2)
    pos_ids <- torch::torch_arange(start = 0, end = L - 1,
                                   dtype = torch::torch_long(),
                                   device = input_ids$device)$
      unsqueeze(1)$expand_as(input_ids)
    tt_ids <- torch::torch_zeros_like(input_ids)
    # nn_embedding in R-torch uses 1-based indices (R convention) but the
    # tokenizer and the PyTorch-trained weights are 0-based.  The weight
    # matrix rows themselves line up either way — row 0 in PyTorch is row 1
    # in R-torch — so we just add 1 at the call site to keep the lookup
    # accessing the same underlying row.
    x <- self$word_embeddings(input_ids + 1L) +
         self$position_embeddings(pos_ids + 1L) +
         self$token_type_embeddings(tt_ids + 1L)
    self$LayerNorm(x)
  }
)

#' BERT multi-head self-attention
#' @keywords internal
#' @noRd
bert_self_attention <- torch::nn_module(
  "BertSelfAttention",
  initialize = function(config) {
    self$num_heads <- config$num_attention_heads
    self$head_dim  <- config$hidden_size %/% config$num_attention_heads
    self$query <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$key   <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$value <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },
  forward = function(x, mask) {
    B <- x$size(1); L <- x$size(2); H <- x$size(3)
    reshape_heads <- function(t) {
      t$view(c(B, L, self$num_heads, self$head_dim))$transpose(2, 3)
    }
    q <- reshape_heads(self$query(x))
    k <- reshape_heads(self$key(x))
    v <- reshape_heads(self$value(x))
    scores <- torch::torch_matmul(q, k$transpose(3, 4)) / sqrt(self$head_dim)
    scores <- scores + mask
    attn   <- torch::nnf_softmax(scores, dim = -1)
    out    <- torch::torch_matmul(attn, v)
    out$transpose(2, 3)$contiguous()$view(c(B, L, H))
  }
)

#' BERT attention output projection + residual + layer norm
#' @keywords internal
#' @noRd
bert_self_output <- torch::nn_module(
  "BertSelfOutput",
  initialize = function(config) {
    self$dense     <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(hidden, input_tensor) {
    self$LayerNorm(self$dense(hidden) + input_tensor)
  }
)

#' BERT attention block (self-attention + output projection)
#' @keywords internal
#' @noRd
bert_attention <- torch::nn_module(
  "BertAttention",
  initialize = function(config) {
    # `self` is reserved in R6 / nn_module so the submodule is `self_`.
    # The weight loader renames "attention.self.X" -> "attention.self_.X".
    self$self_  <- bert_self_attention(config)
    self$output <- bert_self_output(config)
  },
  forward = function(x, mask) {
    a <- self$self_(x, mask)
    self$output(a, x)
  }
)

#' BERT intermediate feed-forward projection (hidden -> 4*hidden, then GeLU)
#' @keywords internal
#' @noRd
bert_intermediate <- torch::nn_module(
  "BertIntermediate",
  initialize = function(config) {
    self$dense <- torch::nn_linear(config$hidden_size, config$intermediate_size)
  },
  forward = function(x) torch::nnf_gelu(self$dense(x))
)

#' BERT output feed-forward projection + residual + layer norm
#' @keywords internal
#' @noRd
bert_output <- torch::nn_module(
  "BertOutput",
  initialize = function(config) {
    self$dense     <- torch::nn_linear(config$intermediate_size,
                                       config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(hidden, input_tensor) {
    self$LayerNorm(self$dense(hidden) + input_tensor)
  }
)

#' Pfeiffer bottleneck adapter module (used by SPECTER2 and compatible models)
#'
#' Implements the adapter described in Pfeiffer et al. (2020):
#'   down-project (H → d), ReLU, up-project (d → H), add skip connection.
#' Inserted after the FFN sublayer of each BertLayer by [load_specter2()].
#'
#' The weight layout matches the HuggingFace `adapters` library convention:
#' `adapter_down.0.{weight,bias}` (Sequential wrapper) and
#' `adapter_up.{weight,bias}`.
#'
#' @keywords internal
#' @noRd
bert_adapter <- torch::nn_module(
  "BertAdapter",
  initialize = function(hidden_size, reduction_factor = 16L) {
    d <- as.integer(hidden_size / reduction_factor)
    # nn_sequential so state_dict keys are "adapter_down.0.weight/bias",
    # matching the HuggingFace adapters library naming convention.
    self$adapter_down <- torch::nn_sequential(torch::nn_linear(hidden_size, d))
    self$adapter_up   <- torch::nn_linear(d, hidden_size)
  },
  forward = function(x) {
    self$adapter_up(torch::nnf_relu(self$adapter_down(x))) + x
  }
)

#' BERT layer (one transformer block: attention + feed-forward)
#'
#' An optional Pfeiffer adapter can be attached after construction by
#' assigning a [bert_adapter] instance to `layer$adapter` (done automatically
#' by [load_specter2()]).
#'
#' @keywords internal
#' @noRd
bert_layer <- torch::nn_module(
  "BertLayer",
  initialize = function(config) {
    self$attention    <- bert_attention(config)
    self$intermediate <- bert_intermediate(config)
    self$output       <- bert_output(config)
    self$adapter      <- NULL   # set by load_specter2() after construction
  },
  forward = function(x, mask) {
    a <- self$attention(x, mask)
    i <- self$intermediate(a)
    h <- self$output(i, a)
    if (!is.null(self$adapter)) h <- self$adapter(h)
    h
  }
)

#' BERT encoder (stack of N transformer layers)
#' @keywords internal
#' @noRd
bert_encoder <- torch::nn_module(
  "BertEncoder",
  initialize = function(config) {
    self$layer <- torch::nn_module_list(
      lapply(seq_len(config$num_hidden_layers),
             function(.) bert_layer(config))
    )
  },
  forward = function(x, mask) {
    for (i in seq_along(self$layer)) {
      x <- self$layer[[i]](x, mask)
    }
    x
  }
)

#' Complete BERT model (embeddings + encoder).
#'
#' Not normally called directly — use [load_hf_bert()] instead.  This is the
#' module that gets instantiated and into which weights are loaded.
#'
#' @param config A list with fields `vocab_size`, `hidden_size`,
#'   `num_hidden_layers`, `num_attention_heads`, `intermediate_size`,
#'   `max_position_embeddings`, `type_vocab_size`, `layer_norm_eps`.
#' @return A torch `nn_module` representing the full BERT encoder.
#' @keywords internal
#' @noRd
bert_model <- torch::nn_module(
  "BertModel",
  initialize = function(config) {
    self$config     <- config
    self$embeddings <- bert_embeddings(config)
    self$encoder    <- bert_encoder(config)
  },
  forward = function(input_ids, attention_mask) {
    # Additive attention mask: 0 where attending, -1e4 where padded.
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2)$unsqueeze(2) * -1e4
    x <- self$embeddings(input_ids)
    self$encoder(x, ext)
  }
)
