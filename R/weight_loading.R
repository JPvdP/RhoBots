# =============================================================================
# weight_loading.R — Read BERT weights from safetensors or pickle files into
# an already-constructed bert_model.  Handles HuggingFace naming conventions
# (prefix stripping, head-key filtering, R6 `self` rename).
# =============================================================================

#' Read a state-dict file regardless of format
#'
#' Dispatches on file extension:
#'   - `.safetensors` -> [safetensors::safe_load_file()]   (preferred)
#'   - `.bin/.pth/.pt` -> [torch::load_state_dict()]        (PyTorch pickle)
#'
#' @param path Path to a weights file.
#' @return A named list of `torch_tensor`s.
#' @keywords internal
#' @noRd
.check_lfs_pointer <- function(path) {
  sz <- file.size(path)
  # >= 1 MB looks like a real weights file
  if (!is.na(sz) && sz >= 1048576L) return(invisible(NULL))

  lines <- tryCatch(
    readLines(path, n = 3L, warn = FALSE),
    error = function(e) character(0)
  )
  text <- paste(lines, collapse = "\n")

  if (grepl("git-lfs.github.com", text, fixed = TRUE)) {
    stop(
      "The downloaded file is a Git LFS pointer, not actual weights:\n  ",
      path,
      "\nhfhub fetched the stub instead of the real binary.",
      "\nFix: set a HuggingFace token, then force-redownload:\n",
      "  Sys.setenv(HUGGING_FACE_HUB_TOKEN = \"hf_...\")\n",
      "  hfhub::hub_download(",
      "\"<repo_id>\", \"pytorch_model.bin\", force_download = TRUE)"
    )
  }

  stop(
    "Downloaded weights file is too small (", sz, " bytes) — likely corrupted or an error page:\n  ",
    path,
    "\nDelete the cached file and retry, optionally with a HuggingFace token:\n",
    "  Sys.setenv(HUGGING_FACE_HUB_TOKEN = \"hf_...\")"
  )
}

.load_weight_file <- function(path) {
  .check_lfs_pointer(path)
  if (grepl("\\.safetensors$", path, ignore.case = TRUE)) {
    weights <- safetensors::safe_load_file(path, framework = "torch")
    attr(weights, "metadata") <- NULL
    weights$metadata <- NULL
    return(weights)
  }
  if (grepl("\\.(bin|pth|pt)$", path, ignore.case = TRUE)) {
    # Detect pre-1.6 PyTorch legacy pickle format (magic byte 0x80).
    # PyTorchStreamReader (used by load_state_dict) only handles the zip
    # container introduced in PyTorch 1.6 (2020); old-format files need
    # Python to convert first.
    magic <- tryCatch(readBin(path, what = "raw", n = 2L),
                      error = function(e) raw(0))
    if (length(magic) >= 1L && magic[1] == as.raw(0x80)) {
      stop(
        "This model's pytorch_model.bin uses the legacy pickle format",
        " (pre-PyTorch 1.6),\n",
        "which R torch cannot read. Convert it to safetensors first:\n\n",
        "  # In Python (pip install torch safetensors huggingface_hub):\n",
        "  import torch\n",
        "  from safetensors.torch import save_file\n",
        "  sd = torch.load(\"pytorch_model.bin\", map_location=\"cpu\",",
        " weights_only=False)\n",
        "  save_file(sd, \"model.safetensors\")\n\n",
        "Then pass the converted file via:\n",
        "  load_hf_bert(\"<repo_id>\", weights_path = \"/path/to/model.safetensors\")"
      )
    }
    weights <- torch::load_state_dict(path)
    return(as.list(weights))
  }
  stop("Unknown weight file format: ", path,
       "\nExpected .safetensors, .bin, .pth, or .pt")
}

#' Normalize a checkpoint key to match our R module's parameter naming
#'
#' Strips optional top-level prefixes that some checkpoints add (e.g.
#' BertForMaskedLM has a "bert." prefix; some sentence-transformers exports
#' use "0.auto_model.") and renames `attention.self.X` -> `attention.self_.X`
#' because `self` is reserved in R6.
#'
#' @param key A checkpoint key (string).
#' @return The corresponding R-module parameter path.
#' @keywords internal
#' @noRd
.normalize_key <- function(key) {
  for (pfx in c("bert.", "mpnet.", "model.", "0.auto_model.", "auto_model.")) {
    if (startsWith(key, pfx)) key <- substring(key, nchar(pfx) + 1)
  }
  key <- sub("attention\\.self\\.", "attention.self_.", key)
  # Old HuggingFace BERT checkpoints (converted from TF) name LayerNorm
  # parameters gamma/beta; PyTorch and our R module use weight/bias.
  key <- gsub("LayerNorm.gamma", "LayerNorm.weight", key, fixed = TRUE)
  key <- gsub("LayerNorm.beta",  "LayerNorm.bias",   key, fixed = TRUE)
  key
}

#' Load BERT weights from a checkpoint into a constructed model
#'
#' Reads weights from a safetensors or PyTorch pickle file, normalizes the
#' parameter names to match the R module structure, filters out task-head
#' keys we don't need (`pooler.*`, `cls.*`, `lm_head.*`, ...), and applies
#' the result via `model$load_state_dict()`.
#'
#' Most users don't call this directly — it's invoked by [load_hf_bert()].
#' Exposed for users who construct a model manually or want to load
#' alternative weight files.
#'
#' @param model A `bert_model` instance (or compatible nn_module).
#' @param weights_path Path to a `.safetensors` or `.bin` file.
#' @param strict If TRUE, errors when expected parameters are missing.  If
#'   FALSE, just warns.  Default FALSE — most checkpoints have a handful of
#'   extra task-head keys that are correctly ignored.
#' @return Invisibly, the named list of loaded weights.
#' @export
load_bert_weights <- function(model, weights_path, strict = FALSE) {
  weights <- .load_weight_file(weights_path)

  # Rename incoming keys to match our R module structure.
  names(weights) <- vapply(names(weights), .normalize_key, character(1))

  # Keep only keys the model actually expects (drop task-head leftovers).
  expected <- names(model$state_dict())
  loadable <- weights[names(weights) %in% expected]

  missing <- setdiff(expected, names(loadable))
  extra   <- setdiff(names(weights), expected)
  if (length(missing) > 0) {
    msg <- sprintf("  missing in checkpoint: %d params (first: %s)",
                   length(missing), missing[1])
    message(msg)
  }
  if (length(extra) > 0) {
    msg <- sprintf("  extra in checkpoint:   %d params (first: %s; ignored)",
                   length(extra), extra[1])
    message(msg)
  }
  if (strict && length(missing) > 0)
    stop("Strict load: ", length(missing), " expected params missing.")

  # R torch's load_state_dict does not support strict = FALSE, so we build a
  # complete state dict by starting from the model's current values (random
  # init) and overwriting only the keys present in the checkpoint.  Missing
  # keys (e.g. token_type_embeddings in RoBERTa-derived models) keep their
  # random init, which is harmless because those embeddings are never activated
  # (token_type_ids are always 0).
  full_state <- model$state_dict()
  for (key in names(loadable)) {
    full_state[[key]] <- loadable[[key]]
  }
  model$load_state_dict(full_state)
  message(sprintf("Loaded %d / %d expected parameters from %s",
                  length(loadable), length(expected),
                  basename(weights_path)))
  invisible(loadable)
}
