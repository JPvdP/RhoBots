# =============================================================================
# classify.R  --  Text classification and token labeling (NER) with fine-tuned
# BERT-family models from HuggingFace.
#
# OVERVIEW
# --------
# Fine-tuned classification models share the same BERT/RoBERTa backbone as
# embedding models.  The difference is a small "head" on top that maps the
# final hidden states to task outputs:
#
#   load_hf_classifier()   --   download config + weights, build the right model
#   classify_texts()       --   run inference, return a data.frame of predictions
#
# THREE OUTPUT MODES (determined by problem_type in config.json)
# ---------------------------------------------------------------
#   single_label_classification:
#     softmax over logits -> one label + probability per input text.
#     Example output: data.frame(text, label, score, negative, neutral, positive)
#
#   multi_label_classification:
#     sigmoid per label (each label is independent) -> probability per label.
#     Example output: data.frame(text, toxic, severe_toxic, obscene, ...)
#
#   regression:
#     raw logits without any activation -> continuous scores.
#     Example: VAD model -> data.frame(text, valence, arousal, dominance)
#
# TOKEN LABELING (NER, POS, chunking)
# ------------------------------------
# When config.json's architectures field contains "ForTokenClassification",
# classify_texts() runs the model in NER mode and returns a *list* of
# data.frames (one per input text), each with columns: token, label, score.
# Special tokens ([CLS], [SEP], padding) are automatically excluded.
# =============================================================================


# Special token strings that should be excluded from NER output.
# These are padding or delimiter tokens that do not represent real words.
.SPECIAL_TOKENS <- c("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]",
                     "<cls>", "<sep>")


#' Load a fine-tuned BERT-family classifier from HuggingFace
#'
#' Downloads `config.json`, model weights, and a tokenizer from the HuggingFace
#' Hub and returns an `hf_classifier` object.  Supports sequence classification
#' (sentiment, topic, intent), multi-label classification, regression (e.g. VAD
#' emotion scores), and token classification (NER, POS tagging).
#'
#' The task type and output format are auto-detected from `config.json`:
#' \itemize{
#'   \item `architectures` containing `"ForTokenClassification"` -> NER mode
#'   \item `problem_type = "regression"` -> return raw numeric scores
#'   \item `problem_type = "multi_label_classification"` -> sigmoid per label
#'   \item otherwise -> softmax single-label classification
#' }
#'
#' Supported backbone architectures: the same as [load_hf_bert()]  --  BERT,
#' RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT, MPNet.
#'
#' @param repo_id HuggingFace repo ID, e.g. `"cardiffnlp/twitter-xlm-roberta-base-sentiment"`
#'   or `"RobroKools/vad-bert"`.
#' @param weights_path Optional path to a local weights file (`.safetensors` or
#'   `.bin`).  Overrides Hub weight discovery; config and tokenizer still come
#'   from `repo_id`.
#' @param prefix Optional string prepended to every text before tokenization
#'   (useful for instruction-tuned classifiers).  Default `""`.
#' @return An `hf_classifier` list with elements `model`, `tokenizer`, `config`,
#'   `id2label`, `problem_type`, `task`, `num_labels`, `repo_id`, `prefix`.
#' @export
#' @examples
#' \dontrun{
#'   # Sentiment analysis (single-label, 3 classes)
#'   clf <- load_hf_classifier("cardiffnlp/twitter-xlm-roberta-base-sentiment")
#'   classify_texts(clf, c("I love this!", "Terrible experience."))
#'
#'   # VAD regression (valence, arousal, dominance)
#'   clf <- load_hf_classifier("RobroKools/vad-bert")
#'   classify_texts(clf, c("I am ecstatic!", "Feeling nervous about the exam."))
#'
#'   # Named entity recognition
#'   clf <- load_hf_classifier("dslim/bert-base-NER")
#'   classify_texts(clf, c("Albert Einstein was born in Ulm, Germany."))
#' }
load_hf_classifier <- function(repo_id, weights_path = NULL, prefix = "") {

  if (!requireNamespace("torch", quietly = TRUE))
    stop("The 'torch' package is required. Run install.packages('torch') ",
         "then rhobots_install().")
  if (!torch::torch_is_installed())
    stop("The torch C++ backend is not ready. Run rhobots_install() to set it up.")

  for (pkg in c("hfhub", "jsonlite", "tok")) {
    if (!requireNamespace(pkg, quietly = TRUE))
      stop("Please install.packages(\"", pkg, "\")")
  }

  # -- config ------------------------------------------------------------------
  cfg_path   <- hfhub::hub_download(repo_id, "config.json")
  cfg_raw    <- .distilbert_config(jsonlite::fromJSON(cfg_path))
  model_type <- cfg_raw$model_type %||% "bert"

  .supported <- c("bert", "roberta", "xlm-roberta", "camembert",
                  "distilbert", "mpnet")
  if (!model_type %in% .supported)
    stop("'", repo_id, "' has model_type '", model_type, "', not supported.\n",
         "Supported: ", paste(.supported, collapse = ", "))

  # id2label maps integer index (as character "0", "1", ...) to label name.
  # unlist() flattens the parsed JSON list to a named character vector.
  id2label <- if (!is.null(cfg_raw$id2label)) unlist(cfg_raw$id2label) else NULL

  # num_labels: prefer the length of id2label (most reliable), fall back to
  # the config field, then default to 2.
  num_labels <- if (!is.null(id2label)) length(id2label) else
    as.integer(cfg_raw$num_labels %||% 2L)

  problem_type <- cfg_raw$problem_type %||% "single_label_classification"

  # Detect task from the architectures list:
  # "BertForTokenClassification", "RobertaForTokenClassification", etc. -> NER
  archs <- cfg_raw$architectures %||% character(0)
  task  <- if (any(grepl("TokenClassification", archs))) "ner" else "classification"

  message(sprintf("Task: %s | problem_type: %s | num_labels: %d",
                  task, problem_type, num_labels))

  cfg <- list(
    vocab_size                     = cfg_raw$vocab_size,
    hidden_size                    = cfg_raw$hidden_size,
    num_hidden_layers              = cfg_raw$num_hidden_layers,
    num_attention_heads            = cfg_raw$num_attention_heads,
    intermediate_size              = cfg_raw$intermediate_size,
    max_position_embeddings        = cfg_raw$max_position_embeddings,
    type_vocab_size                = cfg_raw$type_vocab_size      %||% 2L,
    layer_norm_eps                 = cfg_raw$layer_norm_eps        %||% 1e-12,
    hidden_dropout_prob            = cfg_raw$hidden_dropout_prob   %||% 0.1,
    classifier_dropout             = cfg_raw$classifier_dropout,   # may be NULL
    model_type                     = model_type,
    activation                     = cfg_raw$activation,
    num_labels                     = as.integer(num_labels),
    relative_attention_num_buckets = cfg_raw$relative_attention_num_buckets %||% 32L
  )

  # -- weights -----------------------------------------------------------------
  if (!is.null(weights_path)) {
    if (!file.exists(weights_path))
      stop("Local weights_path does not exist: ", weights_path)
    message("Using provided weights file: ", weights_path)
  } else {
    weights_path  <- NULL
    dl_errors     <- list()
    for (fname in c("model.safetensors", "pytorch_model.bin")) {
      weights_path <- tryCatch(
        hfhub::hub_download(repo_id, fname),
        error = function(e) { dl_errors[[fname]] <<- conditionMessage(e); NULL }
      )
      if (!is.null(weights_path)) { message("Using weights file: ", fname); break }
    }
    if (is.null(weights_path))
      stop("Could not download weights from ", repo_id,
           ". Tried model.safetensors and pytorch_model.bin.")
  }

  # -- build model -------------------------------------------------------------
  # Dispatch on task x model_type to pick the right head architecture.
  model <- if (task == "ner" && model_type == "distilbert") {
    distilbert_for_token_classification(cfg)
  } else if (task == "ner") {
    # Token classification head is identical for BERT and RoBERTa  -- 
    # both store "classifier.*" at the top level of the checkpoint.
    bert_for_token_classification(cfg)
  } else if (model_type == "distilbert") {
    # DistilBERT: first token -> pre_classifier -> ReLU -> classifier.
    distilbert_for_classification(cfg)
  } else if (model_type %in% c("roberta", "xlm-roberta", "camembert")) {
    # RoBERTa-family uses its own head (no separate pooler).
    roberta_for_classification(cfg)
  } else {
    # BERT, MPNet-like -> BERT-style head with pooler.
    bert_for_classification(cfg)
  }

  load_bert_weights(model, weights_path, strict = FALSE)
  model$eval()

  # -- tokenizer ---------------------------------------------------------------
  tokenizer <- tryCatch({
    tok::tokenizer$from_pretrained(repo_id)
  }, error = function(e) {
    message("  no tokenizer.json; falling back to WordPiece + vocab.txt")
    vocab_path <- tryCatch(hfhub::hub_download(repo_id, "vocab.txt"),
                           error = function(e2) NULL)
    if (is.null(vocab_path))
      stop("Model ", repo_id, " provides neither tokenizer.json nor vocab.txt.")
    do_lower <- tryCatch({
      tc <- jsonlite::fromJSON(hfhub::hub_download(repo_id, "tokenizer_config.json"))
      if (!is.null(tc$do_lower_case)) isTRUE(tc$do_lower_case) else NULL
    }, error = function(e2) NULL)
    make_wordpiece_tokenizer(vocab_path, do_lower_case = do_lower)
  })

  # NER output needs the token strings, which tok's encodings do not carry.
  id2token <- if (task == "ner") .load_id2token(repo_id) else NULL

  structure(
    list(model        = model,
         tokenizer    = tokenizer,
         config       = cfg,
         repo_id      = repo_id,
         id2label     = id2label,
         id2token     = id2token,
         problem_type = problem_type,
         task         = task,
         num_labels   = as.integer(num_labels),
         prefix       = prefix),
    class = c("hf_classifier", "list")
  )
}


# -----------------------------------------------------------------------------
# .load_id2token  --  vocabulary lookup from token ID to token string
#
# The tok package returns only token IDs and attention masks, so NER output
# looks the token strings up here.  Element i + 1 is the string for ID i, as
# in Python's tokenizer.convert_ids_to_tokens() (e.g. "##ie" for a WordPiece
# continuation, "ĠParis" for a RoBERTa word that follows a space).
# -----------------------------------------------------------------------------

.load_id2token <- function(repo_id) {
  tj <- tryCatch(hfhub::hub_download(repo_id, "tokenizer.json"),
                 error = function(e) NULL)
  if (!is.null(tj)) {
    j     <- jsonlite::fromJSON(tj, simplifyVector = FALSE)
    vocab <- unlist(j$model$vocab)                     # token -> id
    added <- j$added_tokens %||% list()
    ids   <- c(unname(vocab), vapply(added, function(a) as.integer(a$id), 1L))
    toks  <- c(names(vocab),  vapply(added, function(a) a$content, ""))
    out   <- character(max(ids) + 1L)
    out[ids + 1L] <- toks
    return(out)
  }
  # WordPiece models without tokenizer.json: line i of vocab.txt is ID i - 1.
  vp <- tryCatch(hfhub::hub_download(repo_id, "vocab.txt"), error = function(e) NULL)
  if (is.null(vp))
    stop("Model ", repo_id, " provides neither tokenizer.json nor vocab.txt.")
  readLines(vp, encoding = "UTF-8", warn = FALSE)
}


#' Print method for hf_classifier objects
#' @param x An hf_classifier object.
#' @param ... Unused.
#' @return Invisibly returns \code{x}.
#' @examples
#' \dontrun{
#'   cls <- load_hf_classifier("cardiffnlp/twitter-roberta-base-sentiment-latest")
#'   print(cls)
#' }
#' @export
print.hf_classifier <- function(x, ...) {
  cat("<hf_classifier>\n")
  cat("  repo:         ", x$repo_id, "\n")
  cat("  task:         ", x$task, "\n")
  cat("  problem_type: ", x$problem_type, "\n")
  cat("  num_labels:   ", x$num_labels, "\n")
  if (!is.null(x$id2label)) {
    lab_str <- paste(unname(x$id2label), collapse = ", ")
    if (nchar(lab_str) > 60) lab_str <- paste0(substr(lab_str, 1, 57), "...")
    cat("  labels:       ", lab_str, "\n")
  }
  cat("  hidden_size:  ", x$config$hidden_size, "\n")
  cat("  layers:       ", x$config$num_hidden_layers, "\n")
  if (nzchar(x$prefix %||% ""))
    cat("  prefix:       \"", x$prefix, "\"\n", sep = "")
  invisible(x)
}


# =============================================================================
# classify_texts  --  S3 generic + methods
# =============================================================================

#' Classify or label texts with a fine-tuned BERT-family model
#'
#' S3 generic that dispatches on the classifier class returned by
#' [load_hf_classifier()].
#'
#' **Sequence classification** (sentiment, topic, ...): returns a `data.frame`
#' with columns `text`, `label`, `score`, plus one probability column per
#' label.  For `problem_type = "regression"` the label columns contain raw
#' numeric scores.  For `problem_type = "multi_label_classification"` each
#' label column contains a sigmoid probability and there is no single `label`
#' or `score` column.
#'
#' **Token classification / NER**: returns a named `list` of `data.frame`s,
#' one per input text, each with columns `token`, `label`, `score`.  Special
#' tokens (`[CLS]`, `[SEP]`, padding) are automatically excluded.
#'
#' @param classifier An `hf_classifier` from [load_hf_classifier()].
#' @param texts A character vector of strings.
#' @param batch_size Number of texts per forward pass. Default 32.
#' @param max_length Maximum token sequence length (including special tokens).
#'   Sequences are truncated to this value. Default 512.
#' @param verbose Print batch progress. Default FALSE.
#' @param max_tokens Upper limit on padded tokens per forward pass (number of
#'   texts in the batch times the longest one).  Long texts then get smaller
#'   batches, keeping peak memory steady.  `NULL` (default) means no limit
#'   beyond `batch_size`.  See [embed_texts()].
#' @param gc_every Release memory held by finished batches every `gc_every`
#'   batches (default 1); `0` disables cleanup.  See [embed_texts()].
#' @param ... Unused (for future extension).
#' @return For sequence tasks: a `data.frame` with `nrow(texts)` rows.
#'   For NER: a `list` of `data.frame`s, one per input text.
#' @export
#' @examples
#' \dontrun{
#'   clf  <- load_hf_classifier("cardiffnlp/twitter-xlm-roberta-base-sentiment")
#'   res  <- classify_texts(clf, c("I love this!", "Terrible experience."))
#'   res$label   # c("positive", "negative")
#'   res$score   # highest class probability
#'
#'   # VAD regression
#'   clf <- load_hf_classifier("RobroKools/vad-bert")
#'   classify_texts(clf, c("I am ecstatic!"))
#'   # returns: text | valence | arousal | dominance
#'
#'   # NER
#'   clf <- load_hf_classifier("dslim/bert-base-NER")
#'   result <- classify_texts(clf, c("Marie Curie was born in Warsaw."))
#'   result[[1]]   # token | label | score
#' }
classify_texts <- function(classifier, texts, ...) UseMethod("classify_texts")


#' @rdname classify_texts
#' @export
classify_texts.hf_classifier <- function(classifier, texts,
                                          batch_size = 32L,
                                          max_length = 512L,
                                          verbose    = FALSE,
                                          max_tokens = NULL,
                                          gc_every   = 1L,
                                          ...) {
  model      <- classifier$model
  tokenizer  <- classifier$tokenizer
  task       <- classifier$task
  batch_size <- as.integer(batch_size)
  gc_every   <- as.integer(gc_every)
  if (is.null(max_tokens)) max_tokens <- Inf
  if (!is.numeric(max_tokens) || length(max_tokens) != 1L || max_tokens < 1) {
    stop("`max_tokens` must be NULL or a single positive number.")
  }

  model$eval()

  # Apply instruction prefix if the model needs one (output keeps the
  # original texts).
  prefix <- classifier$prefix %||% ""
  inputs <- if (nzchar(prefix)) paste0(prefix, texts) else texts

  tokenizer$enable_padding()
  tokenizer$enable_truncation(as.integer(max_length))

  # Same memory strategy as embed_texts(): tokenize in blocks of documents,
  # batch by token length (longest first, capped by max_tokens), and free
  # torch memory after every batch and encoder layer.
  n          <- length(texts)
  block_size <- max(2048L, batch_size * 64L)
  seq_logits <- NULL                      # (n, num_labels) for sequence tasks
  tok_logits <- vector("list", n)         # per-text (L_i, num_labels) for NER
  ids_all    <- vector("list", n)         # per-text token IDs for NER
  n_batch    <- 0L
  n_done     <- 0L

  for (b_start in seq(1L, n, by = block_size)) {
    b_end <- min(b_start + block_size - 1L, n)

    # encode_batch() pads to the longest text in the block; keep real tokens.
    enc <- tokenizer$encode_batch(inputs[b_start:b_end])
    ids <- lapply(enc, function(e) e$ids[e$attention_mask == 1L])
    rm(enc)

    for (idx in .batch_plan(lengths(ids), batch_size, max_tokens)) {
      logits <- .classify_batch(model, ids[idx], layer_gc = gc_every > 0L)
      rows   <- b_start - 1L + idx              # positions in `texts`

      if (task == "ner") {
        # (B, Lmax, nl): keep each text's real positions only.
        for (j in seq_along(idx)) {
          L_j <- length(ids[[idx[j]]])
          tok_logits[[rows[j]]] <- matrix(logits[j, seq_len(L_j), ], nrow = L_j)
          ids_all[[rows[j]]]    <- ids[[idx[j]]]
        }
      } else {
        if (is.null(seq_logits)) seq_logits <- matrix(0, nrow = n, ncol = ncol(logits))
        seq_logits[rows, ] <- logits
      }

      n_batch <- n_batch + 1L
      if (gc_every > 0L && n_batch %% gc_every == 0L) .release_memory("cpu")

      n_done <- n_done + length(idx)
      if (verbose) message(sprintf("  classified %d / %d", n_done, n))
    }
    rm(ids)
  }
  .release_memory("cpu")

  if (task == "ner") {
    id2token <- classifier$id2token %||% .load_id2token(classifier$repo_id)
    return(.process_ner(tok_logits, ids_all, classifier$id2label, id2token))
  }

  .process_seq_classification(seq_logits, texts, classifier$id2label,
                              classifier$problem_type)
}


# -----------------------------------------------------------------------------
# .classify_batch  --  one forward pass, returned as a plain R array
#
# ids: list of token-ID vectors (real tokens only).  Pads to the batch's
# longest sequence and returns (B, num_labels) or, for token classification,
# (B, Lmax, num_labels).  Tensors stay local so .release_memory() can free
# them once the function returns.
# -----------------------------------------------------------------------------

.classify_batch <- function(model, ids, layer_gc = TRUE) {
  # Free each encoder layer's intermediates before the next one runs.
  old <- options(rhobots.layer_gc = layer_gc)
  on.exit(options(old), add = TRUE)

  lens  <- lengths(ids)
  Lmax  <- max(lens)
  ids_m <- do.call(rbind, lapply(ids,  function(v) c(v, rep(0L, Lmax - length(v)))))
  msk_m <- do.call(rbind, lapply(lens, function(l) rep(c(1L, 0L), c(l, Lmax - l))))

  input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())
  attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())

  torch::with_no_grad({
    logits <- model(input_ids, attn_mask)
  })
  out <- as.array(logits$cpu())
  # A single (B = 1) sequence-classification batch comes back as a vector.
  if (is.null(dim(out)) || length(dim(out)) == 1L) out <- matrix(out, nrow = 1L)
  out
}


#' @rdname classify_texts
#' @export
classify_texts.default <- function(classifier, texts, ...) {
  stop("No classify_texts method for class '",
       paste(class(classifier), collapse = "/"), "'.\n",
       "Use load_hf_classifier() to create a compatible classifier.")
}


# =============================================================================
# Internal post-processing helpers
# =============================================================================

# .softmax  --  numerically stable row-wise softmax
.softmax <- function(mat) {
  # Subtract row max before exp() to prevent overflow (result is identical).
  mat <- mat - apply(mat, 1, max)
  e   <- exp(mat)
  e / rowSums(e)
}

# .sigmoid  --  element-wise sigmoid (for multi-label)
.sigmoid <- function(x) 1 / (1 + exp(-x))


# .process_seq_classification
#
# Converts raw logit matrix (n_texts x num_labels) to a user-friendly
# data.frame.  Output format depends on problem_type.
.process_seq_classification <- function(logits, texts, id2label, problem_type) {

  # Build label name vector from id2label (keys are "0", "1", ... as strings).
  # If id2label is missing, fall back to generic "LABEL_0", "LABEL_1", ... names.
  nl <- ncol(logits)
  label_names <- if (!is.null(id2label)) {
    unname(id2label[as.character(seq(0, nl - 1))])
  } else {
    paste0("LABEL_", seq(0, nl - 1))
  }

  if (problem_type == "regression") {
    # Regression: return raw continuous scores (e.g. VAD dimensions).
    # No activation function  --  just expose the raw linear outputs.
    df           <- as.data.frame(logits)
    colnames(df) <- label_names
    df$text      <- texts
    return(df[, c("text", label_names), drop = FALSE])
  }

  if (problem_type == "multi_label_classification") {
    # Each label is an independent binary decision -> sigmoid per cell.
    probs        <- .sigmoid(logits)
    df           <- as.data.frame(probs)
    colnames(df) <- label_names
    df$text      <- texts
    return(df[, c("text", label_names), drop = FALSE])
  }

  # Default: single_label_classification -> softmax, then argmax.
  probs    <- .softmax(logits)             # (n, num_labels)  --  rows sum to 1
  best_idx <- apply(probs, 1, which.max)  # 1-based R index of winning class

  # Map 1-based R index to 0-based Python id2label key ("0", "1", ...).
  labels <- label_names[best_idx]
  scores <- apply(probs, 1, max)

  # Core columns: text, predicted label, and the winning class probability.
  df <- data.frame(text  = texts,
                   label = labels,
                   score = scores,
                   stringsAsFactors = FALSE)

  # Append one column per label so callers can inspect the full distribution.
  for (i in seq_len(nl)) df[[label_names[i]]] <- probs[, i]

  df
}


# .process_ner
#
# Converts per-text (L_i, num_labels) logit matrices (real tokens only) to a
# list of per-text data.frames with columns token / label / score.
# Special tokens ([CLS], [SEP], etc.) are excluded.
.process_ner <- function(tok_logits, ids_all, id2label, id2token) {
  nl <- ncol(tok_logits[[1]])
  label_names <- if (!is.null(id2label)) {
    unname(id2label[as.character(seq(0, nl - 1))])
  } else {
    paste0("LABEL_", seq(0, nl - 1))
  }

  lapply(seq_along(tok_logits), function(i) {
    probs  <- .softmax(tok_logits[[i]])        # softmax per token: (L, nl)

    # Look up token strings by ID (tok encodings carry IDs only).
    tokens <- id2token[ids_all[[i]] + 1L]

    # Exclude special delimiter tokens  --  these are not real words.
    keep   <- !tokens %in% .SPECIAL_TOKENS
    probs  <- probs[keep, , drop = FALSE]

    data.frame(token = tokens[keep],
               label = label_names[apply(probs, 1, which.max)],
               score = apply(probs, 1, max),
               stringsAsFactors = FALSE)
  })
}
