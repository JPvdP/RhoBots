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
  # Fail fast with a clear message before touching the network
  if (!requireNamespace("torch", quietly = TRUE))
    stop("The 'torch' package is required. Run install.packages('torch') ",
         "then rhobots_install().")
  if (!torch::torch_is_installed()) {
    extra <- if (.Platform$OS.type == "windows")
      paste0("\nWindows: first install the Visual C++ Redistributable 2022 —",
             "\n  https://aka.ms/vs/17/release/vc_redist.x64.exe",
             "\nthen restart Windows and run rhobots_install().")
    else
      ""
    stop("The torch C++ backend is not ready. Run rhobots_install() to set it up.", extra)
  }

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

#' Load a SPECTER2 model with a task-specific adapter
#'
#' SPECTER2 \doi{10.48550/arXiv.2211.13308} extends the original SPECTER model
#' with task-specific Pfeiffer adapters trained on millions of citation pairs.
#' This function loads the base encoder from `allenai/specter2_base` and
#' injects the chosen adapter into each transformer layer, matching the
#' behaviour of the Python `adapters` library.
#'
#' Available adapters on the HuggingFace Hub:
#' \describe{
#'   \item{`"allenai/specter2"`}{Proximity / similarity — recommended for
#'     document retrieval and topic modeling (default).}
#'   \item{`"allenai/specter2_adhoc_query"`}{Query-side adapter for asymmetric
#'     retrieval (query vs. document).}
#'   \item{`"allenai/specter2_classification"`}{Trained for paper
#'     classification tasks.}
#' }
#'
#' The returned object is a standard `bert_encoder` compatible with
#' [embed_texts()], [embed_texts_cached()], and [fit_bertopic()].
#'
#' @param adapter HuggingFace repo ID of the adapter checkpoint.  Must contain
#'   a `pytorch_adapter.bin` file and an `adapter_config.json`.
#'   Default: `"allenai/specter2"` (proximity adapter).
#' @param base HuggingFace repo ID of the base model.
#'   Default: `"allenai/specter2_base"`.
#' @param adapter_name Name of the adapter as stored in the checkpoint's weight
#'   keys.  For all official SPECTER2 adapters this is `"[PRX]"`.
#' @return A `bert_encoder` object (same class as [load_hf_bert()]) with
#'   Pfeiffer adapter modules injected into every transformer layer.  The list
#'   also carries `$adapter_repo` recording which adapter was loaded.
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_specter2()   # proximity adapter — best for topic modeling
#'   emb <- embed_texts(enc, c("Graph neural networks for drug discovery.",
#'                              "Climate tipping points and carbon budgets."))
#' }
load_specter2 <- function(adapter      = "allenai/specter2",
                           base         = "allenai/specter2_base",
                           adapter_name = "[PRX]") {
  # ---- [1] Load base model --------------------------------------------------
  message("Loading SPECTER2 base model from: ", base)
  enc <- load_hf_bert(base)

  hidden_size      <- enc$config$hidden_size        # 768
  n_layers         <- enc$config$num_hidden_layers  # 12

  # ---- [2] Read adapter config to get reduction_factor ----------------------
  message("Downloading adapter config from: ", adapter)
  cfg_path <- tryCatch(
    hfhub::hub_download(adapter, "adapter_config.json"),
    error = function(e) stop("Could not download adapter_config.json from '",
                              adapter, "': ", conditionMessage(e))
  )
  adp_cfg          <- jsonlite::fromJSON(cfg_path)
  reduction_factor <- as.integer(adp_cfg$config$reduction_factor %||% 16L)
  bottleneck_dim   <- as.integer(hidden_size / reduction_factor)
  message(sprintf("  Adapter: %s  |  bottleneck: %d -> %d -> %d",
                  adapter_name, hidden_size, bottleneck_dim, hidden_size))

  # ---- [3] Download adapter weights -----------------------------------------
  message("Downloading adapter weights (pytorch_adapter.bin)...")
  adp_path <- tryCatch(
    hfhub::hub_download(adapter, "pytorch_adapter.bin"),
    error = function(e) stop("Could not download pytorch_adapter.bin from '",
                              adapter, "': ", conditionMessage(e))
  )
  adp_weights <- .load_weight_file(adp_path)
  message("  Loaded ", length(adp_weights), " tensors from adapter checkpoint.")

  # Peek at available keys to validate the adapter_name
  all_keys <- names(adp_weights)
  if (length(all_keys) == 0L)
    stop("Adapter weight file appears empty. Check the download at: ", adp_path)

  # ---- [4] Inject adapters into each transformer layer ----------------------
  message("Injecting adapter into ", n_layers, " transformer layers...")
  for (i in seq_len(n_layers)) {
    layer_idx <- i - 1L   # 0-based key index matching Python convention
    prefix    <- sprintf("encoder.layer.%d.output.adapters.%s.",
                         layer_idx, adapter_name)

    down_w_key <- paste0(prefix, "adapter_down.0.weight")
    down_b_key <- paste0(prefix, "adapter_down.0.bias")
    up_w_key   <- paste0(prefix, "adapter_up.weight")
    up_b_key   <- paste0(prefix, "adapter_up.bias")

    missing_keys <- setdiff(c(down_w_key, down_b_key, up_w_key, up_b_key),
                            all_keys)
    if (length(missing_keys) > 0L) {
      # Helpful diagnostic: show a sample of actual keys so the user can
      # spot naming differences (e.g. wrong adapter_name)
      sample_keys <- head(all_keys, 8L)
      stop(
        "Expected adapter keys not found in checkpoint (layer ", i, ").\n",
        "Missing: ", paste(missing_keys, collapse = "\n         "), "\n",
        "Available keys (first 8):\n  ",
        paste(sample_keys, collapse = "\n  "), "\n",
        "Check that 'adapter_name' matches the key pattern above."
      )
    }

    # Build and populate the adapter module
    adp <- bert_adapter(hidden_size, reduction_factor)
    adp$load_state_dict(list(
      "adapter_down.0.weight" = adp_weights[[down_w_key]],
      "adapter_down.0.bias"   = adp_weights[[down_b_key]],
      "adapter_up.weight"     = adp_weights[[up_w_key]],
      "adapter_up.bias"       = adp_weights[[up_b_key]]
    ))

    # Attach to the layer — R torch registers this as a submodule
    enc$model$encoder$layer[[i]]$adapter <- adp
  }

  enc$model$eval()
  message("SPECTER2 ready — base + '", adapter_name, "' adapter loaded.")

  # Annotate the returned object with adapter provenance
  enc$adapter_repo <- adapter
  enc$adapter_name <- adapter_name
  enc
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
  if (!is.null(x$adapter_repo))
    cat("  adapter:     ", x$adapter_name, " (", x$adapter_repo, ")\n", sep = "")
  invisible(x)
}
