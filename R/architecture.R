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
    # RoBERTa (and XLM-RoBERTa, CamemBERT) reserve position 0 as padding
    # and start real positions at 2 (padding_idx = 1).  Standard BERT starts
    # at 0.  We store the offset at construction time so forward() is clean.
    self$pos_offset <- if (config$model_type %in%
                           c("roberta", "xlm-roberta", "camembert")) 2L else 0L
  },
  forward = function(input_ids) {
    L <- input_ids$size(2)
    pos_ids <- torch::torch_arange(start = self$pos_offset,
                                   end   = L - 1 + self$pos_offset,
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

# =============================================================================
# MPNet architecture modules (Song et al., 2020 — arXiv:2004.09297)
#
# MPNet differs from BERT in three ways:
#   1. No token_type_embeddings; position IDs start at 2 (padding_idx = 1).
#   2. A shared relative position bias (32 learned buckets × num_heads) is
#      added to attention scores in every layer.
#   3. Attention weight names: q/k/v/o instead of self.query/key/value;
#      the o projection lives in MPNetAttention (not MPNetSelfAttention).
#
# The R module hierarchy mirrors the HuggingFace Python class hierarchy exactly
# so that checkpoint keys map one-to-one without any extra remapping logic.
# =============================================================================

#' MPNet input embedding layer (word + position, then LayerNorm; no token type)
#' @keywords internal
#' @noRd
mpnet_embeddings <- torch::nn_module(
  "MPNetEmbeddings",
  initialize = function(config) {
    self$word_embeddings     <- torch::nn_embedding(config$vocab_size,
                                                    config$hidden_size)
    self$position_embeddings <- torch::nn_embedding(config$max_position_embeddings,
                                                    config$hidden_size)
    self$LayerNorm           <- torch::nn_layer_norm(config$hidden_size,
                                                     eps = config$layer_norm_eps)
  },
  forward = function(input_ids) {
    L <- input_ids$size(2)
    # MPNet position IDs start at 2 (padding_idx = 1, same as RoBERTa).
    # R torch_arange is inclusive, so (start=2, end=L+1) → [2, 3, ..., L+1].
    pos_ids <- torch::torch_arange(
      start  = 2L,
      end    = L + 1L,
      dtype  = torch::torch_long(),
      device = input_ids$device
    )$unsqueeze(1L)$expand_as(input_ids)
    # nn_embedding in R-torch is 1-based: add 1 to convert 0-based IDs.
    x <- self$word_embeddings(input_ids + 1L) +
         self$position_embeddings(pos_ids + 1L)
    self$LayerNorm(x)
  }
)

#' Bucket relative positions for MPNet's learned position bias
#'
#' Translates relative token distances into 32 log-scale buckets.  Negative
#' (leftward) distances occupy buckets 0..15 and positive (rightward) distances
#' occupy 16..31.  The first 8 buckets in each half are exact; the remaining 8
#' cover logarithmically wider ranges up to max_distance = 128.
#'
#' @param relative_position Integer tensor of shape (L, L) with values j - i.
#' @return Long tensor of shape (L, L) with bucket indices in [0, 31].
#' @keywords internal
#' @noRd
.mpnet_position_bucket <- function(relative_position,
                                    num_buckets  = 32L,
                                    max_distance = 128L) {
  nb        <- as.integer(num_buckets) %/% 2L   # 16
  n         <- -relative_position
  # Negative n (i.e. positive relative_position) → high half of buckets.
  ret       <- (n < 0L)$to(dtype = torch::torch_long()) * nb
  n         <- n$abs()
  max_exact <- nb %/% 2L                         # 8 — exact-count region
  is_small  <- n < max_exact
  # Log-scale mapping for |n| >= max_exact.  clamp(min=1) avoids log(0);
  # those positions are always "small" and selected away by torch_where anyway.
  log_ratio  <- log(as.numeric(max_distance) / max_exact)
  val_large  <- max_exact + (
    torch::torch_log(
      n$to(dtype = torch::torch_float())$clamp(min = 1L) / max_exact
    ) / log_ratio * (nb - max_exact)
  )$to(dtype = torch::torch_long())
  val_large  <- val_large$clamp(max = nb - 1L)
  ret + torch::torch_where(is_small, n, val_large)
}

#' MPNet self-attention (Q/K/V projections + relative position bias)
#'
#' Differs from BERT: weight names are q/k/v (not query/key/value), and the
#' output projection o lives in the parent mpnet_attention module.  The r
#' weight (position key projection) is retained for weight loading but unused
#' during inference.
#' @keywords internal
#' @noRd
mpnet_self_attention <- torch::nn_module(
  "MPNetSelfAttention",
  initialize = function(config) {
    self$num_heads <- config$num_attention_heads
    self$head_dim  <- as.integer(config$hidden_size / config$num_attention_heads)
    self$q <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$k <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$v <- torch::nn_linear(config$hidden_size, config$hidden_size)
    # o and r weights live here for checkpoint key compatibility;
    # o is called by the parent mpnet_attention, r is unused in inference.
    self$o <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$r <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },
  forward = function(x, mask, position_bias) {
    B <- x$size(1); L <- x$size(2)
    reshape <- function(t)
      t$view(c(B, L, self$num_heads, self$head_dim))$transpose(2L, 3L)
    q      <- reshape(self$q(x))
    k      <- reshape(self$k(x))
    v      <- reshape(self$v(x))
    scores <- torch::torch_matmul(q, k$transpose(3L, 4L)) / sqrt(self$head_dim)
    scores <- scores + position_bias + mask
    attn   <- torch::nnf_softmax(scores, dim = -1)
    out    <- torch::torch_matmul(attn, v)
    # Return concatenated heads; o projection applied by parent module.
    out$transpose(2L, 3L)$contiguous()$view(c(B, L, self$num_heads * self$head_dim))
  }
)

#' MPNet attention block (self-attention + o projection + residual + LayerNorm)
#'
#' The output projection (self$attn$o) is applied here, not inside
#' mpnet_self_attention, matching HuggingFace's MPNetAttention structure.
#' @keywords internal
#' @noRd
mpnet_attention <- torch::nn_module(
  "MPNetAttention",
  initialize = function(config) {
    self$attn      <- mpnet_self_attention(config)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(x, mask, position_bias) {
    a <- self$attn(x, mask, position_bias)
    self$LayerNorm(self$attn$o(a) + x)
  }
)

#' MPNet transformer layer (attention + FFN, reusing BERT's FFN modules)
#' @keywords internal
#' @noRd
mpnet_layer <- torch::nn_module(
  "MPNetLayer",
  initialize = function(config) {
    self$attention    <- mpnet_attention(config)
    self$intermediate <- bert_intermediate(config)
    self$output       <- bert_output(config)
  },
  forward = function(x, mask, position_bias) {
    a <- self$attention(x, mask, position_bias)
    i <- self$intermediate(a)
    self$output(i, a)
  }
)

#' MPNet encoder (layer stack + shared relative position bias embedding)
#' @keywords internal
#' @noRd
mpnet_encoder <- torch::nn_module(
  "MPNetEncoder",
  initialize = function(config) {
    self$layer <- torch::nn_module_list(
      lapply(seq_len(config$num_hidden_layers), function(.) mpnet_layer(config))
    )
    self$relative_attention_bias <- torch::nn_embedding(
      config$relative_attention_num_buckets,
      config$num_attention_heads
    )
  },
  forward = function(x, mask) {
    bias <- self$compute_position_bias(x)
    for (i in seq_along(self$layer)) {
      x <- self$layer[[i]](x, mask, bias)
    }
    x
  },
  compute_position_bias = function(x) {
    L      <- x$size(2)
    device <- x$device
    # Sequence positions 0..L-1 for relative distance computation.
    idx <- torch::torch_arange(start = 0L, end = L - 1L,
                               dtype = torch::torch_long(), device = device)
    rel <- idx$unsqueeze(1L) - idx$unsqueeze(2L)  # (L, L): j - i
    bucket <- .mpnet_position_bucket(rel)           # (L, L) in [0, 31]
    # nn_embedding is 1-based: convert 0-based bucket indices.
    vals <- self$relative_attention_bias(bucket + 1L)  # (L, L, num_heads)
    vals$permute(c(3L, 1L, 2L))$unsqueeze(1L)          # (1, num_heads, L, L)
  }
)

#' Complete MPNet model (embeddings + encoder)
#' @keywords internal
#' @noRd
mpnet_model <- torch::nn_module(
  "MPNetModel",
  initialize = function(config) {
    self$config     <- config
    self$embeddings <- mpnet_embeddings(config)
    self$encoder    <- mpnet_encoder(config)
  },
  forward = function(input_ids, attention_mask) {
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2L)$unsqueeze(2L) * -1e4
    x <- self$embeddings(input_ids)
    self$encoder(x, ext)
  }
)
