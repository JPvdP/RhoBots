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
#' RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT (backbone only), MPNet.
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
  cfg_raw    <- jsonlite::fromJSON(cfg_path)
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
  model <- if (task == "ner") {
    # Token classification head is identical for BERT and RoBERTa  -- 
    # both store "classifier.*" at the top level of the checkpoint.
    bert_for_token_classification(cfg)
  } else if (model_type %in% c("roberta", "xlm-roberta", "camembert")) {
    # RoBERTa-family uses its own head (no separate pooler).
    roberta_for_classification(cfg)
  } else {
    # BERT, DistilBERT, MPNet-like -> BERT-style head with pooler.
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

  structure(
    list(model        = model,
         tokenizer    = tokenizer,
         config       = cfg,
         repo_id      = repo_id,
         id2label     = id2label,
         problem_type = problem_type,
         task         = task,
         num_labels   = as.integer(num_labels),
         prefix       = prefix),
    class = c("hf_classifier", "list")
  )
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
                                          ...) {
  model     <- classifier$model
  tokenizer <- classifier$tokenizer
  id2label  <- classifier$id2label
  problem   <- classifier$problem_type
  task      <- classifier$task

  model$eval()

  # Apply instruction prefix if the model needs one.
  prefix <- classifier$prefix %||% ""
  if (nzchar(prefix)) texts <- paste0(prefix, texts)

  tokenizer$enable_padding()
  tokenizer$enable_truncation(as.integer(max_length))

  n         <- length(texts)
  logit_buf <- vector("list", ceiling(n / batch_size))  # stores raw model output
  enc_buf   <- vector("list", ceiling(n / batch_size))  # stores tok Encodings (NER)
  mask_buf  <- vector("list", ceiling(n / batch_size))  # stores attention masks (NER)

  idx <- 0L
  torch::with_no_grad({
    for (start in seq(1L, n, by = as.integer(batch_size))) {
      end   <- min(start + as.integer(batch_size) - 1L, n)
      batch <- texts[start:end]
      idx   <- idx + 1L

      enc   <- tokenizer$encode_batch(batch)
      ids   <- lapply(enc, function(e) e$ids)
      masks <- lapply(enc, function(e) e$attention_mask)
      Lmax  <- max(vapply(ids, length, integer(1L)))

      # Pad every sequence to Lmax by appending zeros (padding token ID).
      pad   <- function(v) c(v, rep(0L, Lmax - length(v)))
      ids_m <- do.call(rbind, lapply(ids,   pad))   # (B, Lmax)
      msk_m <- do.call(rbind, lapply(masks, pad))   # (B, Lmax)

      input_ids <- torch::torch_tensor(ids_m, dtype = torch::torch_long())
      attn_mask <- torch::torch_tensor(msk_m, dtype = torch::torch_long())

      logits <- model(input_ids, attn_mask)   # (B, num_labels) or (B, L, num_labels)
      logit_buf[[idx]] <- as.array(logits$cpu())

      if (task == "ner") {
        enc_buf[[idx]]  <- enc
        mask_buf[[idx]] <- msk_m
      }

      if (verbose) message(sprintf("  classified %d / %d", end, n))
    }
  })

  if (task == "ner") {
    return(.process_ner(logit_buf, enc_buf, mask_buf, id2label))
  }

  # Stack batches into a single (n_texts, num_labels) matrix.
  all_logits <- do.call(rbind, logit_buf)
  .process_seq_classification(all_logits, texts, id2label, problem)
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
# Converts a list of (B, L, num_labels) logit arrays (one element per batch)
# to a list of per-text data.frames with columns token / label / score.
# Padding and special tokens ([CLS], [SEP], etc.) are excluded.
.process_ner <- function(logit_buf, enc_buf, mask_buf, id2label) {
  nl <- dim(logit_buf[[1]])[3]
  label_names <- if (!is.null(id2label)) {
    unname(id2label[as.character(seq(0, nl - 1))])
  } else {
    paste0("LABEL_", seq(0, nl - 1))
  }

  result <- list()

  for (bi in seq_along(logit_buf)) {
    logits_arr <- logit_buf[[bi]]  # (B, L, nl) as R array
    enc        <- enc_buf[[bi]]
    masks      <- mask_buf[[bi]]   # integer matrix (B, L)

    B <- dim(logits_arr)[1]

    for (i in seq_len(B)) {
      # Extract (L, nl) logit slice for this text.
      if (B == 1L) {
        tok_logits <- matrix(logits_arr, nrow = dim(logits_arr)[2])
      } else {
        tok_logits <- matrix(logits_arr[i, , ], nrow = dim(logits_arr)[2])
      }

      # Softmax per token position.
      tok_probs <- .softmax(tok_logits)   # (L, nl)

      # Real (non-padding) positions.
      real_pos <- which(masks[i, ] == 1)

      # Get token strings from the tokenizer Encoding.
      tokens <- enc[[i]]$tokens[real_pos]

      # Exclude special delimiter tokens  --  these are not real words.
      keep   <- !tokens %in% .SPECIAL_TOKENS
      tokens <- tokens[keep]
      probs  <- tok_probs[real_pos[keep], , drop = FALSE]

      pred_idx <- apply(probs, 1, which.max)
      labels   <- label_names[pred_idx]
      scores   <- apply(probs, 1, max)

      result <- c(result, list(
        data.frame(token = tokens, label = labels, score = scores,
                   stringsAsFactors = FALSE)
      ))
    }
  }

  result
}
