# =============================================================================
# validate_embeddings.R
#
# Validates that Rhobots embeddings match HuggingFace transformers (Python)
# for each supported model.  For every model the script:
#   1. Embeds texts with Rhobots (R)
#   2. Embeds the same texts with transformers.AutoModel (Python)
#   3. Computes per-sentence cosine similarity and element-wise differences
#   4. Reports a pass/fail summary
#
# WHY results should be (nearly) identical
# ----------------------------------------
# Both pipelines use the same weights, the same tokenizer, mean pooling over
# non-padding tokens, and L2 normalisation.  Differences come only from
# float32 rounding across two independent implementations, so cosine
# similarity between corresponding sentence pairs should be > 0.9999 for all
# models tested here (none have an extra Dense projection layer).
#
# NOTE: SPECTER2 is excluded because the Python side requires the `adapters`
# library and separate adapter loading, making it a separate validation task.
#
# Usage:
#   Rscript validate_embeddings.R path/to/texts.csv
#   Rscript validate_embeddings.R path/to/texts.csv --col Abstract --n 200
#   Rscript validate_embeddings.R path/to/texts.csv --python python3
# =============================================================================

suppressPackageStartupMessages(library(Rhobots))

# =============================================================================
# Parse arguments
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) && length(args) >= i + 1) args[i + 1] else default
}

CSV_PATH   <- args[!startsWith(args, "--")][1]
if (is.na(CSV_PATH) || !nzchar(CSV_PATH)) CSV_PATH <- "/Users/janpieter/Desktop/Projects/UU_Sciento/abstracts_by_continent/Middle_East_abstracts.csv"
TEXT_COL   <- get_arg("--col",    "Abstract")
N_TEXTS    <- as.integer(get_arg("--n",     "200"))
MAX_LEN    <- as.integer(get_arg("--maxlen","256"))
BATCH_SIZE <- as.integer(get_arg("--batch", "32"))
PYTHON_BIN <- get_arg("--python", NULL)

# Auto-detect a Python that has pandas + transformers + torch when none given
if (is.null(PYTHON_BIN)) {
  candidates <- c(
    "/Users/janpieter/miniforge3/envs/bertopic_env/bin/python",
    "/opt/anaconda3/envs/bertopic_env/bin/python",
    Sys.which("python3"),
    "/Users/janpieter/miniforge3/bin/python3",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3"
  )
  # Write a temp script — avoids shell-splitting the -c argument
  .check_py <- tempfile(fileext = ".py")
  writeLines("import pandas, transformers, torch", .check_py)
  for (cand in candidates[nzchar(candidates)]) {
    ok <- tryCatch(
      system2(cand, .check_py, stdout = FALSE, stderr = FALSE) == 0L,
      error = function(e) FALSE
    )
    if (ok) { PYTHON_BIN <- cand; break }
  }
  file.remove(.check_py)
  if (is.null(PYTHON_BIN))
    stop("No Python with pandas + transformers + torch found.\n",
         "Pass --python /path/to/python with the right environment.")
  message("Auto-detected Python: ", PYTHON_BIN)
}

THRESHOLD  <- 0.9999   # minimum acceptable cosine similarity

PY_WORKER  <- file.path(dirname(normalizePath(
                tryCatch(sys.frame(1)$ofile, error = function(e) ".")
              )), "validate_embeddings_py.py")
if (!file.exists(PY_WORKER))
  PY_WORKER <- "validate_embeddings_py.py"   # look in working dir

OUT_DIR    <- "validation_results"
dir.create(OUT_DIR, showWarnings = FALSE)

# =============================================================================
# Models to validate
# Notes:
#   SPECTER2  — excluded: Python side requires the `adapters` library
#   SciBERT   — excluded: allenai/scibert_scivocab_uncased uses a legacy pickle
#               pytorch_model.bin that R-torch cannot read; convert to safetensors
#               first (see load_hf_bert() error message for instructions)
# =============================================================================
MODELS <- list(
  "MiniLM-L6"     = list(repo = "sentence-transformers/all-MiniLM-L6-v2",        prefix = ""),
  "MiniLM-L12"    = list(repo = "sentence-transformers/paraphrase-MiniLM-L12-v2", prefix = ""),
  "DistilRoBERTa" = list(repo = "sentence-transformers/all-distilroberta-v1",      prefix = ""),
  "MPNet-base"    = list(repo = "sentence-transformers/all-mpnet-base-v2",         prefix = ""),
  "BGE-base"      = list(repo = "BAAI/bge-base-en-v1.5",
                          prefix = "Represent this sentence: "),
  "E5-base"       = list(repo = "intfloat/e5-base-v2", prefix = "passage: "),
  "Scibert_DAFS"  = list(repo = "NetworkIsLife/SciBert_Cased_DAFS", prefix = "")
)

# =============================================================================
# Helpers
# =============================================================================
log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(..., collapse = ""))
  message(msg)
}

# Row-wise dot product of two L2-normalised matrices = cosine similarity.
cosine_sim <- function(A, B) rowSums(A * B)

# =============================================================================
# Load and sample texts
# =============================================================================
if (!file.exists(CSV_PATH))
  stop("CSV not found: ", CSV_PATH)

df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
if (!TEXT_COL %in% colnames(df))
  stop("Column '", TEXT_COL, "' not found in CSV. Available: ",
       paste(colnames(df), collapse = ", "))

texts <- df[[TEXT_COL]]
texts <- texts[nzchar(trimws(texts))]

if (length(texts) > N_TEXTS) {
  set.seed(42)
  texts <- sample(texts, N_TEXTS)
  log_msg("Sampled ", N_TEXTS, " texts from ", nrow(df), " rows.")
} else {
  log_msg("Using all ", length(texts), " texts.")
}

# Save shared input so Python reads the exact same rows in the same order
input_csv <- file.path(OUT_DIR, "validation_texts.csv")
write.csv(data.frame(text = texts), input_csv, row.names = FALSE)

# =============================================================================
# Check Python is available
# =============================================================================
py_check <- tryCatch(
  system2(PYTHON_BIN, "--version", stdout = TRUE, stderr = TRUE),
  error = function(e) NULL
)
if (is.null(py_check)) {
  stop("Python not found at '", PYTHON_BIN, "'.\n",
       "Install Python + transformers + torch, then pass --python /path/to/python3")
}
log_msg("Python: ", paste(py_check, collapse = " "))

if (!file.exists(PY_WORKER))
  stop("Python worker not found: ", PY_WORKER,
       "\nMake sure validate_embeddings_py.py is in the same directory.")

# =============================================================================
# Per-model validation
# =============================================================================
results <- list()

for (model_name in names(MODELS)) {
  spec   <- MODELS[[model_name]]
  repo   <- spec$repo
  prefix <- spec$prefix

  log_msg("=== ", model_name, " (", repo, ") ===")

  r_csv  <- file.path(OUT_DIR, paste0(model_name, "_r.csv"))
  py_csv <- file.path(OUT_DIR, paste0(model_name, "_py.csv"))

  row <- list(model = model_name, repo = repo, prefix = prefix,
              n_texts = length(texts), status = "FAIL",
              mean_cosine = NA, min_cosine = NA, p1_cosine = NA,
              mean_abs_diff = NA, max_abs_diff = NA, pass = FALSE)

  # ---- [R] Embed with Rhobots -----------------------------------------------
  pooling <- "mean"
  r_ok <- tryCatch({
    log_msg("  [R] Loading encoder...")
    enc     <- load_hf_bert(repo, prefix = prefix)
    pooling <- enc$pooling %||% "mean"
    log_msg("  [R] Pooling: ", pooling)
    log_msg("  [R] Embedding ", length(texts), " texts...")
    emb_r <- embed_texts(enc, texts, batch_size = BATCH_SIZE,
                         max_length = MAX_LEN, normalize = TRUE,
                         verbose = FALSE)
    write.csv(as.data.frame(emb_r), r_csv, row.names = FALSE)
    log_msg("  [R] Matrix: ", nrow(emb_r), " x ", ncol(emb_r))
    TRUE
  }, error = function(e) {
    log_msg("  [R] ERROR: ", conditionMessage(e))
    FALSE
  })

  if (!r_ok) { results[[model_name]] <- row; next }

  # ---- [Python] Embed with transformers -------------------------------------
  log_msg("  [Py] Running transformers.AutoModel (pooling=", pooling, ")...")
  py_cmd <- sprintf(
    '"%s" "%s" --model "%s" --input "%s" --output "%s" --prefix "%s" --pooling "%s" --col text --max_len %d --batch %d',
    PYTHON_BIN, PY_WORKER, repo,
    normalizePath(input_csv), normalizePath(py_csv),
    prefix, pooling, MAX_LEN, BATCH_SIZE
  )
  py_status <- system(py_cmd)

  if (py_status != 0 || !file.exists(py_csv)) {
    log_msg("  [Py] ERROR: Python script failed (exit code ", py_status, ")")
    row$status <- "PY_FAIL"
    results[[model_name]] <- row
    next
  }

  # ---- Compare --------------------------------------------------------------
  emb_r  <- as.matrix(read.csv(r_csv))
  emb_py <- as.matrix(read.csv(py_csv))

  if (!identical(dim(emb_r), dim(emb_py))) {
    log_msg("  MISMATCH: R matrix ", paste(dim(emb_r), collapse="x"),
            " vs Python ", paste(dim(emb_py), collapse="x"))
    row$status <- "DIM_MISMATCH"
    results[[model_name]] <- row
    next
  }

  cos_sim   <- cosine_sim(emb_r, emb_py)
  abs_diff  <- abs(emb_r - emb_py)

  row$mean_cosine  <- round(mean(cos_sim), 7)
  row$min_cosine   <- round(min(cos_sim), 7)
  row$p1_cosine    <- round(quantile(cos_sim, 0.01), 7)
  row$mean_abs_diff <- round(mean(abs_diff), 8)
  row$max_abs_diff  <- round(max(abs_diff), 6)
  row$pass   <- row$min_cosine >= THRESHOLD
  row$status <- if (row$pass) "PASS" else "WARN"

  log_msg(sprintf("  cosine:   mean=%.7f  min=%.7f  p1=%.7f",
                  row$mean_cosine, row$min_cosine, row$p1_cosine))
  log_msg(sprintf("  abs diff: mean=%.2e  max=%.2e",
                  row$mean_abs_diff, row$max_abs_diff))
  log_msg("  Result: ", row$status,
          if (!row$pass) paste0("  (threshold: ", THRESHOLD, ")") else "")

  results[[model_name]] <- row
}

# =============================================================================
# Summary
# =============================================================================
cat("\n")
cat("=================================================================\n")
cat("  VALIDATION SUMMARY\n")
cat("  Threshold: cosine similarity >= ", THRESHOLD, "\n")
cat("=================================================================\n")

summary_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    model        = r$model,
    status       = r$status,
    mean_cosine  = r$mean_cosine,
    min_cosine   = r$min_cosine,
    p1_cosine    = r$p1_cosine,
    mean_abs_diff = r$mean_abs_diff,
    max_abs_diff = r$max_abs_diff,
    stringsAsFactors = FALSE
  )
}))

print(summary_df, row.names = FALSE, digits = 7)

n_pass <- sum(summary_df$status == "PASS", na.rm = TRUE)
n_fail <- sum(summary_df$status %in% c("FAIL", "WARN", "PY_FAIL", "DIM_MISMATCH"),
              na.rm = TRUE)
cat("\n", n_pass, "/ ", length(results), " models passed.\n")

write.csv(summary_df, file.path(OUT_DIR, "validation_summary.csv"), row.names = FALSE)
log_msg("Results saved to: ", normalizePath(OUT_DIR))

# =============================================================================
# Interpretation note
# =============================================================================
cat("
INTERPRETATION
--------------
Both pipelines load the same weights, use the same tokenizer, and apply
mean pooling + L2 normalisation.  The only differences are float32 rounding
across two independent arithmetic paths.

Expected cosine similarity: > 0.9999 per sentence (typically 0.99999+)
Expected max element-wise difference: < 5e-5

If cosine similarity is substantially below 1 for a model, the likely cause
is a prefix mismatch (R and Python used different instruction prefixes) or a
model that includes a Dense projection layer not loaded by Rhobots.  Run
validate_embeddings_py.py manually to check Python output.
")
