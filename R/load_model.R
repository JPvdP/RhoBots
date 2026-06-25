# =============================================================================
# load_model.R — User-facing function to load a HuggingFace BERT-family
# model from R, in one call.  Handles model discovery, format detection,
# tokenizer selection, and weight loading.
# =============================================================================

#' Load a BERT-family model from HuggingFace for use in R
#'
#' Downloads a model's config, weights, and tokenizer from the HuggingFace
#' Hub and returns an "encoder" object that can be passed to [embed_texts()]
#' or [fit_bertopic()].  Works for any model with a standard BERT
#' architecture: vanilla BERT, MiniLM, MPNet, SciBERT, BioBERT, ClinicalBERT,
#' BERT-for-Patents, FinBERT, LegalBERT, the sentence-transformers built on
#' top of these, and so on.
#'
#' The function tries to use modern formats when available and falls back
#' transparently when they aren't:
#'
#' * **Weights**: `model.safetensors` is preferred (device-agnostic, safe,
#'   fast).  If absent, `pytorch_model.bin` is read via R-torch's pickle
#'   loader.  If the model has only `pytorch_model.bin` and that file was
#'   saved on a CUDA device, the load will fail — convert the upstream
#'   model to safetensors first (see HuggingFace's `safetensors/convert`
#'   Space) and pass the result via `weights_path`.
#' * **Tokenizer**: `tokenizer.json` (HuggingFace fast tokenizer via the
#'   `tok` package) is preferred.  If absent, falls back to
#'   [make_wordpiece_tokenizer()] reading `vocab.txt` (requires the
#'   `wordpiece` package).
#'
#' @param repo_id A HuggingFace repo ID, e.g.
#'   `"sentence-transformers/all-MiniLM-L6-v2"` or
#'   `"NetworkIsLife/SciBert_Cased_DAFS"`.
#' @param weights_path Optional path to a local weights file (`.safetensors`
#'   or `.bin`).  Useful when (a) the upstream model lacks safetensors and
#'   you've downloaded a converted version from a PR branch, (b) you've
#'   produced a local conversion, or (c) you want to pin to a specific local
#'   file rather than the latest on the Hub.  When supplied, this overrides
#'   the Hub weight discovery; `config.json` and the tokenizer are still
#'   pulled from `repo_id`.
#' @return A list with three elements:
#'   \describe{
#'     \item{`model`}{the loaded `bert_model` nn_module}
#'     \item{`tokenizer`}{a tokenizer object exposing `encode_batch()`}
#'     \item{`config`}{the architecture config as a list}
#'   }
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("Hello world.", "Another sentence."))
#'
#'   # With a custom weights file (e.g. safetensors from an unmerged PR)
#'   enc <- load_hf_bert("pritamdeka/S-Scibert-snli-multinli-stsb",
#'                       weights_path = "/path/to/local/model.safetensors")
#' }
load_hf_bert <- function(repo_id, weights_path = NULL) {
  required <- c("hfhub", "jsonlite", "tok")
  for (pkg in required) {
    if (!requireNamespace(pkg, quietly = TRUE))
      stop("Please install.packages(\"", pkg, "\")")
  }

  # --- config ---
  cfg_path <- hfhub::hub_download(repo_id, "config.json")
  cfg_raw  <- jsonlite::fromJSON(cfg_path)
  cfg <- list(
    vocab_size              = cfg_raw$vocab_size,
    hidden_size             = cfg_raw$hidden_size,
    num_hidden_layers       = cfg_raw$num_hidden_layers,
    num_attention_heads     = cfg_raw$num_attention_heads,
    intermediate_size       = cfg_raw$intermediate_size,
    max_position_embeddings = cfg_raw$max_position_embeddings,
    type_vocab_size         = cfg_raw$type_vocab_size %||% 2,
    layer_norm_eps          = cfg_raw$layer_norm_eps  %||% 1e-12
  )

  # --- weights ---
  if (!is.null(weights_path)) {
    if (!file.exists(weights_path))
      stop("Local weights_path does not exist: ", weights_path)
    message("Using provided weights file: ", weights_path)
  } else {
    weights_path <- NULL
    download_errors <- list()
    for (filename in c("model.safetensors", "pytorch_model.bin")) {
      weights_path <- tryCatch(
        hfhub::hub_download(repo_id, filename),
        error = function(e) { download_errors[[filename]] <<- conditionMessage(e); NULL }
      )
      if (!is.null(weights_path)) {
        message("Using weights file: ", filename)
        break
      }
    }
    if (is.null(weights_path)) {
      detail <- if (length(download_errors) > 0)
        paste0("\n  ", names(download_errors), ": ", unlist(download_errors), collapse = "")
      else ""
      stop("Could not download model weights from ", repo_id,
           ". Tried model.safetensors and pytorch_model.bin.", detail)
    }
  }

  # --- tokenizer ---
  tokenizer <- tryCatch({
    tok::tokenizer$from_pretrained(repo_id)
  }, error = function(e) {
    message("  no tokenizer.json found; falling back to WordPiece + vocab.txt")
    vocab_path <- tryCatch(hfhub::hub_download(repo_id, "vocab.txt"),
                           error = function(e2) NULL)
    if (is.null(vocab_path))
      stop("Model ", repo_id, " provides neither tokenizer.json nor vocab.txt.")
    do_lower <- tryCatch({
      tc <- jsonlite::fromJSON(hfhub::hub_download(repo_id,
                                                   "tokenizer_config.json"))
      if (!is.null(tc$do_lower_case)) isTRUE(tc$do_lower_case) else NULL
    }, error = function(e2) NULL)
    make_wordpiece_tokenizer(vocab_path, do_lower_case = do_lower)
  })

  # --- assemble ---
  model <- bert_model(cfg)
  load_bert_weights(model, weights_path, strict = FALSE)
  model$eval()
  structure(
    list(model = model, tokenizer = tokenizer, config = cfg, repo_id = repo_id),
    class = c("bert_encoder", "list")
  )
}

#' Print method for bert_encoder objects
#' @param x A bert_encoder object.
#' @param ... Unused.
#' @export
print.bert_encoder <- function(x, ...) {
  cat("<bert_encoder>\n")
  cat("  repo:        ", x$repo_id, "\n")
  cat("  hidden_size: ", x$config$hidden_size, "\n")
  cat("  layers:      ", x$config$num_hidden_layers, "\n")
  cat("  heads:       ", x$config$num_attention_heads, "\n")
  cat("  vocab_size:  ", x$config$vocab_size, "\n")
  cat("  max_len:     ", x$config$max_position_embeddings, "\n")
  invisible(x)
}
