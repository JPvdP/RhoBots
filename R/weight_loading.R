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
    # torch::load_state_dict reads PyTorch's modern zipfile-pickle format,
    # which HuggingFace `pytorch_model.bin` files have used since PyTorch
    # 1.6 (2020).  Note: this loader cannot remap CUDA-saved tensors to
    # CPU; for that case, the upstream model needs to be converted to
    # safetensors first.
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
  for (pfx in c("bert.", "model.", "0.auto_model.", "auto_model.")) {
    if (startsWith(key, pfx)) key <- substring(key, nchar(pfx) + 1)
  }
  sub("attention\\.self\\.", "attention.self_.", key)
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

  model$load_state_dict(loadable)
  message(sprintf("Loaded %d / %d expected parameters from %s",
                  length(loadable), length(expected),
                  basename(weights_path)))
  invisible(loadable)
}
