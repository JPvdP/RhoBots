# =============================================================================
# architecture.R  --  BERT-family and MPNet transformer encoders as torch
# nn_modules.
#
# WHAT THIS FILE DOES
# -------------------
# This file implements the forward pass (inference) of transformer-based text
# encoders entirely in R, using the `torch` package.  The goal is to produce
# the same numerical output as the Python HuggingFace `transformers` library
# for the same model weights.
#
# WHY WE MIRROR THE HUGGINGFACE CLASS HIERARCHY
# ----------------------------------------------
# When a model is saved with HuggingFace, every learnable weight gets a name
# that reflects the Python class hierarchy:
#
#   "encoder.layer.0.attention.self.query.weight"
#    ^       ^     ^ ^          ^    ^     ^
#    module  list  i submodule  ...  attr  param
#
# Our R nn_module tree uses the same names and nesting, so a single lookup
# table `model$state_dict()` matches the checkpoint keys one-to-one.  No
# manual remapping needed  --  we just strip optional top-level prefixes.
#
# HOW THESE MODULES RELATE
# ------------------------
# BERT:
#   bert_model
#     +-- bert_embeddings       word + position + token-type -> LayerNorm
#     +-- bert_encoder
#           +-- bert_layer  x N
#                 +-- bert_attention
#                 |     +-- bert_self_attention   (Q/K/V -> scaled dot-product)
#                 |     +-- bert_self_output      (dense -> LayerNorm + residual)
#                 +-- bert_intermediate            (dense -> GeLU)
#                 +-- bert_output                 (dense -> LayerNorm + residual)
#
# MPNet (mpnet_model) follows the same skeleton but swaps in:
#   mpnet_embeddings   (no token-type; positions start at 2)
#   mpnet_encoder      (adds shared relative_attention_bias embedding)
#   mpnet_layer / mpnet_attention / mpnet_self_attention
#
# These modules are package-internal.  End users go through `load_hf_bert()`.
# =============================================================================


# =============================================================================
# BERT ARCHITECTURE
# =============================================================================

# -----------------------------------------------------------------------------
# bert_embeddings
#
# A BERT model never sees raw text.  Instead:
#   1. The tokenizer splits text into sub-word tokens and maps each one to an
#      integer ID from a fixed vocabulary.
#   2. bert_embeddings converts those integer IDs into dense numeric vectors
#      (the "embedding lookup").
#
# Three separate embedding tables are summed together:
#
#   word_embeddings:
#     Each token ID -> a 768-dimensional vector.  This is the core vocabulary
#     lookup.  Vocabulary size is typically 30,522 for standard BERT.
#
#   position_embeddings:
#     Each sequence position (0, 1, 2, ...) -> a 768-dimensional vector.
#     Without this, the model would treat "dog bites man" identically to
#     "man bites dog"  --  order would be invisible.
#
#   token_type_embeddings:
#     Each token belongs to "segment A" (0) or "segment B" (1).  Used during
#     pre-training on sentence-pair tasks (e.g. next-sentence prediction).
#     For single-sentence encoding the IDs are all zero, so this adds the
#     same constant offset to every position  --  effectively a no-op but the
#     lookup table must still be present to match the checkpoint.
#
# The three vectors are summed and then normalised with Layer Normalisation
# (LayerNorm) to keep activations in a stable numerical range before the
# first transformer layer.
# -----------------------------------------------------------------------------

#' BERT input embedding layer (word + position + token-type, then LayerNorm)
#' @keywords internal
#' @noRd
bert_embeddings <- torch::nn_module(
  "BertEmbeddings",
  initialize = function(config) {
    # Vocabulary lookup: maps each token ID to a hidden_size-dimensional vector.
    # Shape of the weight matrix: (vocab_size, hidden_size), e.g. (30522, 768).
    self$word_embeddings <- torch::nn_embedding(config$vocab_size,
                                                config$hidden_size)

    # Position lookup: maps each sequence position to a hidden_size vector.
    # max_position_embeddings is the maximum sequence length the model supports
    # (typically 512 for BERT, 514 for RoBERTa because of the offset below).
    self$position_embeddings <- torch::nn_embedding(config$max_position_embeddings,
                                                    config$hidden_size)

    # Segment lookup: distinguishes sentence A (ID=0) from sentence B (ID=1).
    # For single-sentence tasks all IDs are 0  --  this embedding still adds a
    # constant vector but is needed to match the saved checkpoint weights.
    self$token_type_embeddings <- torch::nn_embedding(config$type_vocab_size,
                                                      config$hidden_size)

    # Layer Normalisation: re-centres and re-scales each hidden_size-dimensional
    # vector independently.  After summing three embedding tables the values can
    # have arbitrary scale; LayerNorm brings them back to mean~=0 and std~=1.
    # eps is a small constant added to the variance to avoid dividing by zero.
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)

    # RoBERTa (and XLM-RoBERTa, CamemBERT) reserve position index 0 for a
    # padding sentinel and index 1 as a "start" placeholder.  Real sequence
    # positions therefore begin at 2 instead of 0.  Standard BERT starts at 0.
    # We store the offset once so the forward() method stays readable.
    self$pos_offset <- if (config$model_type %in%
                           c("roberta", "xlm-roberta", "camembert")) 2L else 0L
  },

  forward = function(input_ids) {
    # input_ids is an integer matrix of shape (batch_size, seq_len), where
    # each cell holds the vocabulary index of one sub-word token.

    L <- input_ids$size(2)   # sequence length (number of tokens per row)

    # Build position IDs: a matrix of the same shape as input_ids where
    # row i, column j = j + pos_offset  (same positions broadcast to every row).
    # torch_arange(start, end) in R torch is inclusive on both ends, so
    # (start=offset, end=L-1+offset) gives exactly L values.
    pos_ids <- torch::torch_arange(start  = self$pos_offset,
                                   end    = L - 1 + self$pos_offset,
                                   dtype  = torch::torch_long(),
                                   device = input_ids$device)$
      unsqueeze(1)$        # shape (L,) -> (1, L)
      expand_as(input_ids) # broadcast to (batch_size, L)  --  same positions for every sentence

    # Segment IDs: all zeros for single-sentence encoding.
    tt_ids <- torch::torch_zeros_like(input_ids)

    # IMPORTANT  --  R vs Python indexing:
    # nn_embedding in R-torch uses 1-based indices (like all of R), while the
    # vocabulary IDs from the tokeniser and the checkpoint's weight rows are
    # 0-based (Python convention).  Internally the storage is identical: the
    # weight matrix rows are in the same order.  Adding +1 at the call site
    # maps Python row 0 -> R index 1, row 1 -> index 2, etc., so we access the
    # same underlying vectors.
    x <- self$word_embeddings(input_ids + 1L) +
         self$position_embeddings(pos_ids + 1L) +
         self$token_type_embeddings(tt_ids + 1L)

    self$LayerNorm(x)   # returns shape (batch_size, seq_len, hidden_size)
  }
)


# -----------------------------------------------------------------------------
# bert_self_attention  --  Scaled Dot-Product Multi-Head Attention
#
# Attention is the central operation in a transformer.  It lets every token
# "look at" every other token and decide how much to borrow from it.
#
# STEP-BY-STEP:
#
# 1. Three projections
#    Each token's hidden vector x_i is linearly projected three times:
#      Q_i = x_i . W_Q   (query  --  "what am I looking for?")
#      K_i = x_i . W_K   (key    --  "what do I offer?")
#      V_i = x_i . W_V   (value  --  "what information do I share?")
#
# 2. Attention scores
#    How much should position i attend to position j?
#      score(i,j) = Q_i . K_j  / sqrt(head_dim)
#    Dividing by sqrt(head_dim) prevents scores from growing too large
#    (which would push the softmax into a regime where gradients vanish).
#
# 3. Masking padding tokens
#    Padding tokens are dummy entries added to make all sequences in a batch
#    the same length.  We force their scores to -10,000 so that after softmax
#    they receive ~= 0 attention weight and contribute nothing.
#
# 4. Softmax -> attention weights
#    Converting scores to non-negative weights that sum to 1 across j.
#
# 5. Weighted sum of values
#    out_i = sum_j  attn(i,j) . V_j
#
# MULTI-HEAD TRICK
# ----------------
# Instead of one Q/K/V projection over the full hidden_size = 768, we split
# into num_heads = 12 independent heads, each working in head_dim = 64
# dimensions.  This lets different heads specialise: one might learn syntactic
# relationships, another co-reference, etc.  The heads are processed in
# parallel by reshaping tensors rather than by looping.
# -----------------------------------------------------------------------------

#' BERT multi-head self-attention
#' @keywords internal
#' @noRd
bert_self_attention <- torch::nn_module(
  "BertSelfAttention",
  initialize = function(config) {
    self$num_heads <- config$num_attention_heads          # e.g. 12
    self$head_dim  <- config$hidden_size %/% config$num_attention_heads  # e.g. 64

    # Linear projections for query, key, and value.
    # Each maps hidden_size -> hidden_size (= num_heads x head_dim).
    # These are the W_Q, W_K, W_V matrices described above.
    self$query <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$key   <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$value <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },

  forward = function(x, mask) {
    # x    : (batch B, seq_len L, hidden H)
    # mask : (B, 1, 1, L) additive mask  --  0 for real tokens, -1e4 for padding
    B <- x$size(1); L <- x$size(2); H <- x$size(3)

    # Project and split into heads.
    # After nn_linear: shape (B, L, H).
    # view() reshapes to (B, L, num_heads, head_dim).
    # transpose(2, 3) moves the head dimension before L to get (B, num_heads, L, head_dim).
    # This layout lets us batch-matmul over B and num_heads simultaneously.
    reshape_heads <- function(t) {
      t$view(c(B, L, self$num_heads, self$head_dim))$transpose(2, 3)
    }
    q <- reshape_heads(self$query(x))   # (B, num_heads, L, head_dim)
    k <- reshape_heads(self$key(x))
    v <- reshape_heads(self$value(x))

    # Scaled dot-product attention scores.
    # k$transpose(3, 4) swaps the last two dims: (B, num_heads, head_dim, L)
    # torch_matmul broadcasts over (B, num_heads) and computes (L, head_dim) x (head_dim, L)
    # giving scores of shape (B, num_heads, L, L).
    # Dividing by sqrt(head_dim) stabilises gradients (Vaswani et al., 2017).
    scores <- torch::torch_matmul(q, k$transpose(3, 4)) / sqrt(self$head_dim)

    # Add the additive mask.  Where mask = -1e4 (padding), scores become very
    # negative so softmax assigns those positions ~= 0 weight.
    scores <- scores + mask

    # Softmax over the last dimension (key positions) -> attention weights
    # that sum to 1 across all key positions for each query position.
    attn <- torch::nnf_softmax(scores, dim = -1)   # (B, num_heads, L, L)

    # Weighted sum of value vectors.
    # (B, num_heads, L, L) x (B, num_heads, L, head_dim) -> (B, num_heads, L, head_dim)
    out <- torch::torch_matmul(attn, v)

    # Merge heads: transpose back to (B, L, num_heads, head_dim) then
    # view() to (B, L, H) where H = num_heads x head_dim.
    # contiguous() ensures the tensor memory is contiguous after the transpose.
    out$transpose(2, 3)$contiguous()$view(c(B, L, H))
  }
)


# -----------------------------------------------------------------------------
# bert_self_output  --  Output projection + residual connection + LayerNorm
#
# After the multi-head attention aggregates information from across the
# sequence, two more operations stabilise training and let gradients flow:
#
#   1. Dense projection (hidden -> hidden): mixes the attended representations.
#   2. Residual connection:  output = x + dense(attn_output).
#      Adding the original input x means the layer only needs to learn
#      a "correction" on top of what was already there.  This prevents
#      gradients from vanishing as the network gets deeper.
#   3. LayerNorm: re-normalises the result.
# -----------------------------------------------------------------------------

#' BERT attention output projection + residual + layer norm
#' @keywords internal
#' @noRd
bert_self_output <- torch::nn_module(
  "BertSelfOutput",
  initialize = function(config) {
    # Linear W_O: projects the concatenated head output back to hidden_size.
    self$dense     <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(hidden, input_tensor) {
    # hidden       : attention output from bert_self_attention  (B, L, H)
    # input_tensor : the original x fed into attention          (B, L, H)
    # dense() projects, then we add the residual and normalise.
    self$LayerNorm(self$dense(hidden) + input_tensor)
  }
)


# -----------------------------------------------------------------------------
# bert_attention  --  Combines bert_self_attention and bert_self_output
#
# This is one complete attention sublayer: Q/K/V -> scores -> weighted sum ->
# output projection -> residual -> LayerNorm.
#
# Note on naming: in R6 / nn_module, `self` is already the module object.
# Python's `self.self` (the attribute name in BertAttention) is illegal in R,
# so the submodule is stored as `self$self_` instead.  The weight loader
# renames keys: "attention.self.query.weight" -> "attention.self_.query.weight".
# -----------------------------------------------------------------------------

#' BERT attention block (self-attention + output projection)
#' @keywords internal
#' @noRd
bert_attention <- torch::nn_module(
  "BertAttention",
  initialize = function(config) {
    # Renamed from `.self` to `.self_` to avoid collision with R's `self` keyword.
    # The weight loader in weight_loading.R handles the key rename automatically.
    self$self_  <- bert_self_attention(config)
    self$output <- bert_self_output(config)
  },
  forward = function(x, mask) {
    a <- self$self_(x, mask)   # multi-head attention: (B, L, H)
    self$output(a, x)          # output projection + residual + LayerNorm
  }
)


# -----------------------------------------------------------------------------
# bert_intermediate  --  Feed-Forward Network, first half: expand to 4x width
#
# Each transformer layer contains a two-layer feed-forward network (FFN)
# applied independently to every token position.  The first half expands
# the hidden_size to a larger intermediate_size (typically 4x = 3072 for
# BERT-base) and applies the GeLU activation function.
#
# WHY GeLU?
# The Gaussian Error Linear Unit is a smooth, non-linear activation that
# outperforms ReLU in language models.  Unlike ReLU which hard-gates at 0,
# GeLU weights inputs by their Gaussian CDF, allowing small negative values
# to pass with reduced magnitude rather than being zeroed.
# -----------------------------------------------------------------------------

#' BERT intermediate feed-forward projection (hidden_size -> 4xhidden, then GeLU)
#' @keywords internal
#' @noRd
bert_intermediate <- torch::nn_module(
  "BertIntermediate",
  initialize = function(config) {
    # Expands hidden_size (e.g. 768) to intermediate_size (e.g. 3072).
    self$dense <- torch::nn_linear(config$hidden_size, config$intermediate_size)
  },
  forward = function(x) torch::nnf_gelu(self$dense(x))
)


# -----------------------------------------------------------------------------
# bert_output  --  Feed-Forward Network, second half: project back to hidden_size
#
# Mirrors bert_self_output: projects back down from intermediate_size to
# hidden_size, adds a residual connection to the attention output, and
# normalises.  This second LayerNorm + residual is the "post-FFN sublayer".
# -----------------------------------------------------------------------------

#' BERT output feed-forward projection + residual + layer norm
#' @keywords internal
#' @noRd
bert_output <- torch::nn_module(
  "BertOutput",
  initialize = function(config) {
    # Projects intermediate_size back down to hidden_size.
    self$dense     <- torch::nn_linear(config$intermediate_size,
                                       config$hidden_size)
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(hidden, input_tensor) {
    # hidden       : output of bert_intermediate (the expanded + activated repr.)
    # input_tensor : output of the attention sublayer, used as the residual
    self$LayerNorm(self$dense(hidden) + input_tensor)
  }
)


# -----------------------------------------------------------------------------
# bert_adapter  --  Pfeiffer Bottleneck Adapter (used by SPECTER2)
#
# Adapters are small modules inserted inside frozen pre-trained layers to
# enable parameter-efficient fine-tuning (PEFT).  Instead of updating all
# 110M parameters during fine-tuning, only the adapter weights (typically
# < 1% of the model) are trained.
#
# Architecture (Pfeiffer et al., 2020  --  "AdapterFusion"):
#
#   x --> down-project (H -> d) --> ReLU --> up-project (d -> H) --> + x
#                                                                      ^
#                                                                  skip / residual
#
# The bottleneck dimension d = H / reduction_factor (typically d = 48 for
# reduction_factor=16 and H=768).  This makes the adapter tiny but expressive.
# -----------------------------------------------------------------------------

#' Pfeiffer bottleneck adapter module (used by SPECTER2 and compatible models)
#' @keywords internal
#' @noRd
bert_adapter <- torch::nn_module(
  "BertAdapter",
  initialize = function(hidden_size, reduction_factor = 16L) {
    d <- as.integer(hidden_size / reduction_factor)  # bottleneck dimension

    # nn_sequential wraps the linear layer so that its state_dict key is
    # "adapter_down.0.weight" (the "0" comes from the sequential index).
    # This matches the naming convention used by the HuggingFace `adapters`
    # library so weights load without any remapping.
    self$adapter_down <- torch::nn_sequential(torch::nn_linear(hidden_size, d))
    self$adapter_up   <- torch::nn_linear(d, hidden_size)
  },
  forward = function(x) {
    # Down-project, activate, up-project, add skip connection.
    self$adapter_up(torch::nnf_relu(self$adapter_down(x))) + x
  }
)


# -----------------------------------------------------------------------------
# bert_layer  --  One complete transformer block
#
# A BERT encoder is a stack of N identical layers.  Each layer has two
# sublayers (each with their own LayerNorm + residual):
#
#   Sublayer 1: Multi-head self-attention  (bert_attention)
#   Sublayer 2: Position-wise FFN          (bert_intermediate + bert_output)
#
# Schematically for a single token position:
#
#   x_in --> attention sublayer --> a --> FFN sublayer --> x_out
#
# An optional adapter can be grafted on after the FFN (used by SPECTER2).
# -----------------------------------------------------------------------------

#' BERT layer (one transformer block: attention + feed-forward)
#' @keywords internal
#' @noRd
bert_layer <- torch::nn_module(
  "BertLayer",
  initialize = function(config) {
    self$attention    <- bert_attention(config)     # attention sublayer
    self$intermediate <- bert_intermediate(config)  # FFN first half (expand)
    self$output       <- bert_output(config)        # FFN second half (contract)
    self$adapter      <- NULL   # populated by load_specter2() when an adapter is needed
  },
  forward = function(x, mask) {
    a <- self$attention(x, mask)      # attention: contextualise each token
    i <- self$intermediate(a)         # FFN expand: (B, L, H) -> (B, L, 4H)
    h <- self$output(i, a)            # FFN contract: back to (B, L, H) + residual

    # Apply Pfeiffer adapter if one was injected (SPECTER2 only).
    if (!is.null(self$adapter)) h <- self$adapter(h)
    h
  }
)


# -----------------------------------------------------------------------------
# bert_encoder  --  Stack of N transformer layers
#
# Simply applies each bert_layer in sequence.  The hidden states flow from
# layer 0 (shallowest, closest to raw token embeddings) to layer N-1
# (deepest, most contextualised).  Only the final layer's output is returned
# here; for retrieval / sentence embedding, that output is then pooled
# (mean or CLS) by embed_texts().
# -----------------------------------------------------------------------------

#' BERT encoder (stack of N transformer layers)
#' @keywords internal
#' @noRd
bert_encoder <- torch::nn_module(
  "BertEncoder",
  initialize = function(config) {
    # nn_module_list stores a numbered list of submodules whose state_dict
    # keys are "layer.0.attention...", "layer.1.attention...", etc.
    # This matches HuggingFace's ModuleList naming and allows direct weight loading.
    self$layer <- torch::nn_module_list(
      lapply(seq_len(config$num_hidden_layers),
             function(.) bert_layer(config))
    )
  },
  forward = function(x, mask) {
    # Pass hidden states through each transformer layer in turn.
    for (i in seq_along(self$layer)) {
      x <- self$layer[[i]](x, mask)
    }
    x   # (batch_size, seq_len, hidden_size)
  }
)


# -----------------------------------------------------------------------------
# bert_model  --  The complete BERT encoder
#
# Combines embeddings and the encoder stack.  Also converts the binary
# attention mask (1 = real token, 0 = padding) into an *additive* mask that
# can be directly added to attention scores before softmax.
#
# WHY AN ADDITIVE MASK?
# The attention score for padding positions is set to -10,000 before softmax.
# Because exp(-10000) ~= 0, padding tokens receive essentially zero attention
# weight and their value vectors contribute nothing to the output.  Critically,
# an additive mask is differentiable (unlike a hard Boolean mask), which
# matters during pre-training.
# -----------------------------------------------------------------------------

#' Complete BERT model (embeddings + encoder)
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
    # Convert the 0/1 attention mask to an additive mask for attention scores.
    # (1 - mask) flips: real=0, padding=1.
    # Multiplying by -1e4 gives: real=0, padding=-10000.
    # unsqueeze(2)$unsqueeze(2) inserts two dimensions so the shape becomes
    # (B, 1, 1, L) and broadcasts correctly onto attention scores (B, H, L, L).
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2)$unsqueeze(2) * -1e4

    x <- self$embeddings(input_ids)   # (B, L, hidden_size)
    self$encoder(x, ext)              # (B, L, hidden_size)
  }
)


# =============================================================================
# MPNet ARCHITECTURE  (Song et al., 2020  --  arXiv:2004.09297)
#
# WHAT MAKES MPNET DIFFERENT FROM BERT?
# --------------------------------------
# MPNet was designed to fix a fundamental mismatch in XLNet / permuted LM
# pre-training.  For sentence embedding (the inference task here), three
# practical differences matter:
#
#  1. NO TOKEN-TYPE EMBEDDINGS
#     MPNet only encodes a single segment, so there is no segment-B ID.
#     The embeddings.token_type_embeddings weight does not exist in the
#     checkpoint, and we simply do not create that submodule.
#
#  2. POSITION IDs START AT 2 (same as RoBERTa)
#     Index 0 and 1 are reserved (padding_idx = 1).  Sequence positions are
#     2, 3, 4, ..., L+1.
#
#  3. RELATIVE POSITION BIAS (the key architectural innovation)
#     BERT encodes position absolutely: each position gets its own learned
#     vector.  MPNet instead learns *relative* position biases: a lookup
#     table of shape (num_buckets, num_heads) = (32, 12) where the bucket
#     index encodes how far apart two tokens are (in a log-scale scheme).
#     This bias is added directly to attention scores:
#
#       score(i, j) += relative_bias(bucket(j - i))
#
#     The same bias table is shared across all 12 transformer layers.
#     Benefits: the model can generalise to sequences longer than it saw
#     during training, and nearby vs. distant tokens are treated differently
#     even for the same absolute distance.
#
# WEIGHT KEY NAMING
# -----------------
# HuggingFace MPNet uses different attribute names than BERT:
#   attention.self.query  ->  attention.attn.q
#   attention.self.key    ->  attention.attn.k
#   attention.self.value  ->  attention.attn.v
# Plus a new "o" (output projection) and "r" (position key, pre-training only).
# The LayerNorm lives at attention.LayerNorm (not attention.output.LayerNorm).
# Our R hierarchy mirrors these Python names exactly so weight loading is
# automatic with no extra remapping.
# =============================================================================


# -----------------------------------------------------------------------------
# mpnet_embeddings  --  Word + absolute-position embeddings, no token-type
# -----------------------------------------------------------------------------

#' MPNet input embedding layer (word + position, then LayerNorm; no token type)
#' @keywords internal
#' @noRd
mpnet_embeddings <- torch::nn_module(
  "MPNetEmbeddings",
  initialize = function(config) {
    # Same as BERT: vocabulary lookup from token ID to hidden_size vector.
    self$word_embeddings     <- torch::nn_embedding(config$vocab_size,
                                                    config$hidden_size)

    # Absolute position embeddings.  max_position_embeddings is typically 514
    # for MPNet: positions 0 and 1 are reserved, real positions are 2..513.
    self$position_embeddings <- torch::nn_embedding(config$max_position_embeddings,
                                                    config$hidden_size)

    # Layer Normalisation applied after summing word + position embeddings.
    self$LayerNorm           <- torch::nn_layer_norm(config$hidden_size,
                                                     eps = config$layer_norm_eps)
    # Note: no token_type_embeddings  --  MPNet is always single-segment.
  },

  forward = function(input_ids) {
    L <- input_ids$size(2)   # sequence length

    # MPNet position IDs start at 2 (padding_idx = 1, same as RoBERTa).
    # torch_arange in R-torch is inclusive on both ends, so
    # arange(start=2, end=L+1) produces exactly L values: [2, 3, ..., L+1].
    pos_ids <- torch::torch_arange(
      start  = 2L,
      end    = L + 1L,
      dtype  = torch::torch_long(),
      device = input_ids$device
    )$unsqueeze(1L)$expand_as(input_ids)   # broadcast to (batch_size, L)

    # +1L shifts 0-based IDs to R's 1-based embedding indexing (same as BERT).
    x <- self$word_embeddings(input_ids + 1L) +
         self$position_embeddings(pos_ids + 1L)

    self$LayerNorm(x)   # (batch_size, seq_len, hidden_size)
  }
)


# -----------------------------------------------------------------------------
# .mpnet_position_bucket  --  Convert relative distances to bucket indices
#
# MPNet uses 32 buckets to represent relative token distances.  The first
# 16 buckets cover negative relative positions (token j is to the LEFT of i),
# and the last 16 cover positive (token j is to the RIGHT of i).
#
# Within each half, the first 8 buckets are exact (distance 0, 1, 2, ..., 7),
# and the remaining 8 cover logarithmically wider ranges up to max_distance
# = 128.  Log-scale bucketing means the model distinguishes nearby tokens
# precisely but treats very distant tokens as "just far away", which is
# sufficient in practice.
#
# For a sequence of length L, the input is an (L x L) integer tensor where
# entry [i, j] = j - i  (positive = rightward, negative = leftward).
#
# EXAMPLE for L=4:
#   relative positions:       buckets:
#    0  1  2  3               0  17  18  19
#   -1  0  1  2               1   0  17  18
#   -2 -1  0  1               2   1   0  17
#   -3 -2 -1  0               3   2   1   0
# -----------------------------------------------------------------------------

#' Bucket relative positions for MPNet's learned position bias
#' @keywords internal
#' @noRd
.mpnet_position_bucket <- function(relative_position,
                                    num_buckets  = 32L,
                                    max_distance = 128L) {
  # Split the 32 buckets into two halves of 16.
  nb <- as.integer(num_buckets) %/% 2L   # nb = 16

  # Flip the sign: n > 0 means j is to the LEFT of i (negative relative pos).
  n <- -relative_position

  # Tokens to the RIGHT of i (n < 0, i.e. relative_position > 0) go into the
  # upper half of buckets (16..31).  We set the base offset accordingly.
  # (n < 0)$to(torch_long()) converts the boolean mask to 0/1, then x nb adds
  # 16 for rightward positions and 0 for leftward.
  ret <- (n < 0L)$to(dtype = torch::torch_long()) * nb

  # From here on we work with the absolute distance.
  n <- n$abs()

  # The first max_exact = 8 buckets are exact (one bucket per distance unit).
  max_exact <- nb %/% 2L   # = 8

  # is_small marks positions that fall in the exact-count region.
  is_small <- n < max_exact

  # For larger distances: map [max_exact, max_distance) logarithmically into
  # the remaining (nb - max_exact) = 8 buckets.
  # clamp(min=1) prevents log(0) for distance 0; those positions are "small"
  # and will be selected away by torch_where below, so this doesn't affect output.
  log_ratio <- log(as.numeric(max_distance) / max_exact)   # scalar, computed in R
  val_large <- max_exact + (
    torch::torch_log(
      n$to(dtype = torch::torch_float())$clamp(min = 1L) / max_exact
    ) / log_ratio * (nb - max_exact)
  )$to(dtype = torch::torch_long())

  # Clamp so no bucket index exceeds nb - 1 = 15 (the maximum within each half).
  val_large <- val_large$clamp(max = nb - 1L)

  # Combine: use the exact distance for "small" positions, log bucket for large.
  # torch_where selects element-wise: is_small -> n, else -> val_large.
  # Then add the half-offset stored in ret.
  ret + torch::torch_where(is_small, n, val_large)
}


# -----------------------------------------------------------------------------
# mpnet_self_attention  --  Q/K/V attention with relative position bias
#
# Very similar to bert_self_attention with two differences:
#
#  1. The weight names are q/k/v instead of query/key/value.
#  2. The relative position bias (computed once per forward pass in
#     mpnet_encoder) is added to the attention scores before softmax.
#     This is the key operation that makes MPNet position-aware in a relative
#     rather than absolute sense.
#
#  3. Two extra weights live here for checkpoint compatibility:
#     - o : the output projection (called by the PARENT mpnet_attention module)
#     - r : a position-key projection used only during pre-training;
#           it is present in the weight file but never called during inference.
# -----------------------------------------------------------------------------

#' MPNet self-attention (Q/K/V projections + relative position bias)
#' @keywords internal
#' @noRd
mpnet_self_attention <- torch::nn_module(
  "MPNetSelfAttention",
  initialize = function(config) {
    self$num_heads <- config$num_attention_heads
    self$head_dim  <- as.integer(config$hidden_size / config$num_attention_heads)

    # Query, key, value projection matrices  --  same role as in BERT but with
    # shorter names (q/k/v) to match the HuggingFace checkpoint keys.
    self$q <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$k <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$v <- torch::nn_linear(config$hidden_size, config$hidden_size)

    # Output projection (o): lives here so that the state_dict key is
    # "attention.attn.o.weight", matching the HuggingFace checkpoint.
    # However, it is CALLED from the parent mpnet_attention module.
    self$o <- torch::nn_linear(config$hidden_size, config$hidden_size)

    # Position-key projection (r): used during MPNet pre-training to compute
    # relative position scores.  Not used during inference (sentence embedding).
    # We keep it here purely so the weight key exists for checkpoint loading.
    self$r <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },

  forward = function(x, mask, position_bias) {
    # position_bias : (1, num_heads, L, L)  --  computed once by mpnet_encoder
    B <- x$size(1); L <- x$size(2)

    # Project and reshape exactly as in bert_self_attention.
    reshape <- function(t)
      t$view(c(B, L, self$num_heads, self$head_dim))$transpose(2L, 3L)

    q <- reshape(self$q(x))   # (B, num_heads, L, head_dim)
    k <- reshape(self$k(x))
    v <- reshape(self$v(x))

    # Scaled dot-product scores, same formula as BERT.
    scores <- torch::torch_matmul(q, k$transpose(3L, 4L)) / sqrt(self$head_dim)

    # Add relative position bias BEFORE adding the padding mask.
    # position_bias broadcasts from (1, num_heads, L, L) to (B, num_heads, L, L).
    # This shifts scores based on how far apart each (query, key) pair is.
    scores <- scores + position_bias + mask

    attn <- torch::nnf_softmax(scores, dim = -1)
    out  <- torch::torch_matmul(attn, v)   # (B, num_heads, L, head_dim)

    # Merge heads back to (B, L, hidden_size).
    # Note: self$o is NOT applied here  --  it is called by mpnet_attention below.
    out$transpose(2L, 3L)$contiguous()$view(c(B, L, self$num_heads * self$head_dim))
  }
)


# -----------------------------------------------------------------------------
# mpnet_attention  --  Output projection + residual + LayerNorm
#
# In HuggingFace's MPNet, the output projection (self.attn.o) is applied
# inside MPNetAttention rather than inside MPNetSelfAttention.  This is an
# unusual design choice (BERT's W_O lives in BertSelfOutput), but we mirror
# it exactly so that weight key paths like "attention.attn.o.weight" remain
# correct.
#
# The LayerNorm here sits at "attention.LayerNorm" (not "attention.output.
# LayerNorm" as in BERT)  --  another naming difference handled by our hierarchy.
# -----------------------------------------------------------------------------

#' MPNet attention block (self-attention + o projection + residual + LayerNorm)
#' @keywords internal
#' @noRd
mpnet_attention <- torch::nn_module(
  "MPNetAttention",
  initialize = function(config) {
    self$attn      <- mpnet_self_attention(config)

    # LayerNorm applied after the output projection.
    # Its state_dict key will be "attention.LayerNorm.weight", matching HuggingFace.
    self$LayerNorm <- torch::nn_layer_norm(config$hidden_size,
                                           eps = config$layer_norm_eps)
  },
  forward = function(x, mask, position_bias) {
    # a : raw attention output, shape (B, L, H), heads already merged
    a <- self$attn(x, mask, position_bias)

    # Apply the output projection self$attn$o (note: we reach into the child
    # module here), add the residual x, then normalise.
    self$LayerNorm(self$attn$o(a) + x)
  }
)


# -----------------------------------------------------------------------------
# mpnet_layer  --  One complete MPNet transformer block
#
# Identical structure to bert_layer except:
#  - The attention sublayer is mpnet_attention (which takes position_bias).
#  - The FFN sublayers (bert_intermediate, bert_output) are shared unchanged:
#    their weight keys ("intermediate.dense.*", "output.dense.*",
#    "output.LayerNorm.*") are the same in both BERT and MPNet checkpoints.
# -----------------------------------------------------------------------------

#' MPNet transformer layer (attention + FFN, reusing BERT's FFN modules)
#' @keywords internal
#' @noRd
mpnet_layer <- torch::nn_module(
  "MPNetLayer",
  initialize = function(config) {
    self$attention    <- mpnet_attention(config)

    # The feed-forward network (expand -> GeLU -> contract -> residual -> LayerNorm)
    # has the same structure and weight key names as in BERT, so we reuse these.
    self$intermediate <- bert_intermediate(config)
    self$output       <- bert_output(config)
  },
  forward = function(x, mask, position_bias) {
    a <- self$attention(x, mask, position_bias)   # attention sublayer
    i <- self$intermediate(a)                      # FFN expand to 4x width
    self$output(i, a)                              # FFN contract + residual
  }
)


# -----------------------------------------------------------------------------
# mpnet_encoder  --  Layer stack + shared relative position bias
#
# Two responsibilities beyond bert_encoder:
#
#  1. Owns the relative_attention_bias embedding table (32 x num_heads).
#     A single table is shared by all transformer layers  --  every layer sees
#     the same position signal, which keeps parameter count low.
#
#  2. Calls compute_position_bias() ONCE before the layer loop, then passes
#     the resulting (1, num_heads, L, L) tensor to every layer.  Computing it
#     once avoids redundant table lookups on every layer's forward pass.
# -----------------------------------------------------------------------------

#' MPNet encoder (layer stack + shared relative position bias embedding)
#' @keywords internal
#' @noRd
mpnet_encoder <- torch::nn_module(
  "MPNetEncoder",
  initialize = function(config) {
    # Stack of N transformer layers (same pattern as bert_encoder).
    self$layer <- torch::nn_module_list(
      lapply(seq_len(config$num_hidden_layers), function(.) mpnet_layer(config))
    )

    # Learned relative position bias table.
    # Shape: (num_buckets, num_heads) = (32, 12) for standard MPNet.
    # Each row gives the bias for one "distance bucket" per attention head.
    # Stored as a standard nn_embedding so the weights are automatically
    # tracked and loadable from the checkpoint key "encoder.relative_attention_bias.weight".
    self$relative_attention_bias <- torch::nn_embedding(
      config$relative_attention_num_buckets,
      config$num_attention_heads
    )
  },

  forward = function(x, mask) {
    # Compute position bias once for the current sequence length and reuse
    # across all layers  --  this is an efficiency choice, not a correctness
    # requirement.
    bias <- self$compute_position_bias(x)   # (1, num_heads, L, L)

    for (i in seq_along(self$layer)) {
      x <- self$layer[[i]](x, mask, bias)
    }
    x   # (batch_size, seq_len, hidden_size)
  },

  compute_position_bias = function(x) {
    L      <- x$size(2)    # current sequence length
    device <- x$device

    # Build an (L,) integer tensor of 0-based sequence positions [0, 1, ..., L-1].
    # We want relative distances, not absolute positions, so the actual values
    # do not matter  --  only differences between them do.
    idx <- torch::torch_arange(start = 0L, end = L - 1L,
                               dtype = torch::torch_long(), device = device)

    # Outer difference matrix: rel[i, j] = j - i  (shape L x L).
    # unsqueeze(1L) turns (L,) into (1, L)   --  the "memory" (key) positions.
    # unsqueeze(2L) turns (L,) into (L, 1)   --  the "context" (query) positions.
    # Broadcasting the subtraction gives the full (L, L) relative position matrix.
    rel <- idx$unsqueeze(1L) - idx$unsqueeze(2L)

    # Map each relative distance to one of 32 log-scale buckets.
    bucket <- .mpnet_position_bucket(rel)   # (L, L), values in [0, 31]

    # Look up the learned bias for each bucket.
    # R-torch nn_embedding is 1-based, so we add 1 to the 0-based bucket index.
    # Result shape: (L, L, num_heads).
    vals <- self$relative_attention_bias(bucket + 1L)

    # Rearrange to (1, num_heads, L, L) to broadcast over the batch dimension
    # when added to attention scores of shape (B, num_heads, L, L).
    # permute(c(3, 1, 2)) moves the heads dimension first: (num_heads, L, L).
    # unsqueeze(1L) prepends the batch dimension: (1, num_heads, L, L).
    vals$permute(c(3L, 1L, 2L))$unsqueeze(1L)
  }
)


# -----------------------------------------------------------------------------
# mpnet_model  --  Complete MPNet encoder
#
# Structurally identical to bert_model: convert the binary attention mask
# to an additive mask, run through embeddings, then through the encoder stack.
# The difference is that mpnet_embeddings and mpnet_encoder are used instead
# of their BERT equivalents.
# -----------------------------------------------------------------------------

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
    # Convert binary mask (1 = real, 0 = padding) to additive scores mask.
    # Same logic as bert_model: real tokens -> 0, padding -> -10,000.
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2L)$unsqueeze(2L) * -1e4

    x <- self$embeddings(input_ids)   # (B, L, hidden_size)
    self$encoder(x, ext)              # (B, L, hidden_size)
  }
)


# =============================================================================
# CLASSIFICATION AND TOKEN-LABELING HEADS
#
# Fine-tuned models add a small "head" on top of the BERT/RoBERTa backbone to
# map contextualised hidden states to task outputs (sentiment labels, NER tags,
# VAD regression scores, ...).  The backbone weights are identical to those used
# for embedding; only the head is new.
#
# WHY TWO DIFFERENT HEAD DESIGNS?
# --------------------------------
# BERT fine-tuning papers introduced the "pooler" pattern: the [CLS] token's
# hidden state is passed through a dense layer + tanh (the BertPooler), then a
# linear classifier.  This is BertForSequenceClassification.
#
# RoBERTa (and XLM-RoBERTa, CamemBERT) uses a slightly different head called
# RobertaClassificationHead: it skips the separate pooler and instead applies
# dropout -> dense -> tanh -> dropout -> out_proj directly on the [CLS] vector.
# The weight key names differ too (classifier.dense / classifier.out_proj vs.
# pooler.dense / classifier).
#
# For token classification (NER, POS tagging), ALL architectures use the same
# head: a single linear layer applied to every token position independently.
# The label set (e.g. B-PER, I-PER, O, ...) is read from config.json's id2label.
#
# WEIGHT KEY MAPPING
# ------------------
# These R modules are structured to mirror the Python class hierarchy exactly,
# so that checkpoint keys map one-to-one after the top-level prefix is stripped
# by .normalize_key() in weight_loading.R:
#
#   BertForSequenceClassification:
#     bert.embeddings.*  ->  embeddings.*
#     bert.encoder.*     ->  encoder.*
#     bert.pooler.*      ->  pooler.*       (bert_pooler below)
#     classifier.*       ->  classifier.*
#
#   RobertaForSequenceClassification / XLMRobertaForSequenceClassification:
#     roberta.embeddings.*          ->  embeddings.*
#     roberta.encoder.*             ->  encoder.*
#     classifier.dense.*            ->  classifier.dense.*
#     classifier.out_proj.*         ->  classifier.out_proj.*
#
#   BertForTokenClassification / RobertaForTokenClassification:
#     bert./roberta.embeddings.*    ->  embeddings.*
#     bert./roberta.encoder.*       ->  encoder.*
#     classifier.*                  ->  classifier.*
# =============================================================================


# -----------------------------------------------------------------------------
# bert_pooler  --  CLS-token pooler used by BertForSequenceClassification
#
# During BERT pre-training the [CLS] token is used for next-sentence prediction.
# Fine-tuned classifiers read the same token's hidden state, pass it through a
# dense layer and tanh to produce the "pooled" sentence representation, then
# feed that into the final linear classifier.
#
# Weight keys: pooler.dense.weight / pooler.dense.bias
# -----------------------------------------------------------------------------

#' BERT CLS-token pooler (dense + tanh on the CLS hidden state)
#' @keywords internal
#' @noRd
bert_pooler <- torch::nn_module(
  "BertPooler",
  initialize = function(config) {
    # Projects the CLS hidden state: hidden_size -> hidden_size.
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },
  forward = function(hidden_states) {
    # hidden_states: (B, L, H).  Take position 1 (R 1-based) = [CLS] token.
    cls <- hidden_states[, 1, ]                 # (B, H)
    torch::torch_tanh(self$dense(cls))          # (B, H)
  }
)


# -----------------------------------------------------------------------------
# roberta_classification_head  --  RoBERTa's built-in classification head
#
# Unlike BERT's separate pooler, RoBERTa packs everything into one module:
# CLS extract -> dropout -> dense -> tanh -> dropout -> out_proj.
# In eval() mode dropout is a no-op, so the effective path is:
#   CLS -> dense -> tanh -> out_proj.
#
# Weight keys: classifier.dense.* / classifier.out_proj.*
# -----------------------------------------------------------------------------

#' RoBERTa / XLM-RoBERTa sequence classification head
#' @keywords internal
#' @noRd
roberta_classification_head <- torch::nn_module(
  "RobertaClassificationHead",
  initialize = function(config) {
    dp <- config$hidden_dropout_prob %||% config$classifier_dropout %||% 0.1
    self$dense    <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$dropout  <- torch::nn_dropout(p = dp)
    self$out_proj <- torch::nn_linear(config$hidden_size, config$num_labels)
  },
  forward = function(hidden_states) {
    x <- hidden_states[, 1, ]          # CLS token: (B, H)
    x <- self$dropout(x)               # no-op in eval() mode
    x <- torch::torch_tanh(self$dense(x))
    x <- self$dropout(x)
    self$out_proj(x)                   # (B, num_labels)
  }
)


# -----------------------------------------------------------------------------
# bert_for_classification  --  Complete BERT sequence classifier
#
# Flat structure: embeddings + encoder + pooler + classifier all live at the
# same level so that checkpoint keys (after stripping "bert.") map directly
# to R nn_module attribute paths.  This mirrors BertForSequenceClassification.
# -----------------------------------------------------------------------------

#' BERT-style sequence classification model (backbone + pooler + classifier)
#' @keywords internal
#' @noRd
bert_for_classification <- torch::nn_module(
  "BertForSequenceClassification",
  initialize = function(config) {
    dp <- config$hidden_dropout_prob %||% config$classifier_dropout %||% 0.1
    self$embeddings <- bert_embeddings(config)
    self$encoder    <- bert_encoder(config)
    self$pooler     <- bert_pooler(config)          # CLS -> dense -> tanh
    self$dropout    <- torch::nn_dropout(p = dp)
    self$classifier <- torch::nn_linear(config$hidden_size, config$num_labels)
  },
  forward = function(input_ids, attention_mask) {
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2)$unsqueeze(2) * -1e4      # (B, 1, 1, L): 0 real, -1e4 pad
    x <- self$embeddings(input_ids)          # (B, L, H)
    x <- self$encoder(x, ext)               # (B, L, H)
    x <- self$pooler(x)                     # (B, H)   --  extracts + transforms CLS
    x <- self$dropout(x)                    # no-op at eval time
    self$classifier(x)                      # (B, num_labels) raw logits
  }
)


# -----------------------------------------------------------------------------
# roberta_for_classification  --  Complete RoBERTa sequence classifier
#
# Same flat structure as bert_for_classification but uses roberta_classification_
# head (which has no separate pooler).  Mirrors RobertaForSequenceClassification
# and XLMRobertaForSequenceClassification  --  both use attribute name "roberta"
# for the backbone, so ".normalize_key() strips "roberta." from checkpoint keys.
# -----------------------------------------------------------------------------

#' RoBERTa / XLM-RoBERTa sequence classification model
#' @keywords internal
#' @noRd
roberta_for_classification <- torch::nn_module(
  "RobertaForSequenceClassification",
  initialize = function(config) {
    self$embeddings <- bert_embeddings(config)     # RoBERTa pos offset handled in bert_embeddings
    self$encoder    <- bert_encoder(config)
    self$classifier <- roberta_classification_head(config)
  },
  forward = function(input_ids, attention_mask) {
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2)$unsqueeze(2) * -1e4
    x <- self$embeddings(input_ids)
    x <- self$encoder(x, ext)
    self$classifier(x)     # (B, num_labels)
  }
)


# -----------------------------------------------------------------------------
# bert_for_token_classification  --  Token-level classifier (NER, POS, etc.)
#
# Unlike sequence classifiers that reduce to one vector per sentence, token
# classifiers keep the full (B, L, H) output and apply a linear layer to
# every token position independently.  This gives (B, L, num_labels) logits:
# one label distribution per token.
#
# The same structure works for both BERT and RoBERTa checkpoints because both
# store the classifier weights at the top level (not under "bert." or "roberta.")
# and the backbone keys are already mapped correctly.
# -----------------------------------------------------------------------------

#' BERT / RoBERTa token classification model (NER, POS tagging)
#' @keywords internal
#' @noRd
bert_for_token_classification <- torch::nn_module(
  "BertForTokenClassification",
  initialize = function(config) {
    dp <- config$hidden_dropout_prob %||% config$classifier_dropout %||% 0.1
    self$embeddings <- bert_embeddings(config)
    self$encoder    <- bert_encoder(config)
    self$dropout    <- torch::nn_dropout(p = dp)
    self$classifier <- torch::nn_linear(config$hidden_size, config$num_labels)
  },
  forward = function(input_ids, attention_mask) {
    ext <- (1 - attention_mask$to(dtype = torch::torch_float()))$
      unsqueeze(2)$unsqueeze(2) * -1e4
    x <- self$embeddings(input_ids)    # (B, L, H)
    x <- self$encoder(x, ext)          # (B, L, H)
    x <- self$dropout(x)               # no-op at eval time
    self$classifier(x)                 # (B, L, num_labels)  --  one dist. per token
  }
)
