# =============================================================================
# demo_full_test.R — Rhobots full-suite benchmark across 5 encoders
#
# Usage:
#   Rscript demo_full_test.R path/to/abstracts.csv
#   Or set CSV_PATH manually below.
#
# Expects a CSV with at least one column called "Abstract".
# Results and cached embeddings are written to ./rhobots_test_results/
# Each model gets its own sub-folder containing:
#   *.rds          — saved R objects (fit, embeddings, metrics, ...)
#   visuals/*.html — self-contained interactive plotly charts
# =============================================================================

suppressPackageStartupMessages(library(Rhobots))

`%||%` <- function(x, y) if (is.null(x)) y else x

# =============================================================================
# Configuration
# =============================================================================

CSV_PATH <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(CSV_PATH) || !nzchar(CSV_PATH)) CSV_PATH <- "abstracts.csv"

OUT_DIR <- "rhobots_test_results"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
LOG_FILE <- file.path(OUT_DIR, "test_log.txt")
if (file.exists(LOG_FILE)) file.remove(LOG_FILE)

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", paste0(..., collapse = ""))
  message(msg)
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

# Encoders to benchmark.  Each entry: repo_id (character) or a list with
# repo_id + optional loader / prefix fields.
# load_hf_bert() is the default; load_specter2() is used for SPECTER2 adapters.
MODELS <- list(
  "MiniLM-L6"          = "sentence-transformers/all-MiniLM-L6-v2",
  "MiniLM-L12"         = "sentence-transformers/paraphrase-MiniLM-L12-v2",
  "DistilRoBERTa"      = "sentence-transformers/all-distilroberta-v1",
  "MPNet-base"         = "sentence-transformers/all-mpnet-base-v2",
  "SciBERT"            = "allenai/scibert_scivocab_uncased",
  # BGE — SOTA general-purpose; requires document prefix for best performance
  "BGE-base"           = list(repo_id = "BAAI/bge-base-en-v1.5",
                               prefix  = "Represent this sentence: "),
  # E5 — strong multilingual-capable model; passage prefix for documents
  "E5-base"            = list(repo_id = "intfloat/e5-base-v2",
                               prefix  = "passage: "),
  # SPECTER2 — best for scientific abstracts
  "SPECTER2-proximity" = list(repo_id = "allenai/specter2",
                               loader  = "specter2"),
  "SPECTER2-query"     = list(repo_id = "allenai/specter2_adhoc_query",
                               loader  = "specter2")
)

# Seed topics for guided fitting (sustainability-research defaults;
# adjust to match your corpus domain)
SEED_TOPICS <- list(
  "Energy"       = c("solar", "wind", "renewable", "battery", "energy"),
  "Climate"      = c("climate", "temperature", "warming", "emissions", "carbon"),
  "Biodiversity" = c("species", "ecosystem", "habitat", "extinction", "wildlife"),
  "Water"        = c("water", "ocean", "river", "hydrological", "precipitation"),
  "Society"      = c("policy", "governance", "community", "social", "urban")
)

# Zero-shot topic labels — embedding anchor text as values
ZERO_SHOT_LABELS <- c(
  "Energy transition"    = "renewable energy solar wind power transition fossil fuels",
  "Climate change"       = "global warming climate change temperature emissions carbon",
  "Biodiversity"         = "species extinction habitat biodiversity ecosystem conservation",
  "Water resources"      = "water scarcity drought hydrological cycle groundwater",
  "Urban sustainability" = "urban planning sustainable cities infrastructure land use"
)

# =============================================================================
# Load corpus
# =============================================================================

log_msg("Loading abstracts from: ", CSV_PATH)
if (!file.exists(CSV_PATH))
  stop("CSV file not found: ", CSV_PATH,
       "\nPass the path as the first command-line argument or set CSV_PATH.")

df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
if (!"Abstract" %in% colnames(df))
  stop("CSV must contain a column named 'Abstract'. Found: ",
       paste(colnames(df), collapse = ", "))

docs <- df$Abstract
docs <- docs[nzchar(trimws(docs))]
n    <- length(docs)
log_msg("Loaded ", n, " non-empty abstracts")

# Synthetic metadata for functions that require it
set.seed(42)
years  <- sample(2015:2024, n, replace = TRUE)   # for topics_over_time
groups <- sample(c("Group_A", "Group_B"), n, replace = TRUE)  # for compare_topics

# =============================================================================
# Helper: run one block with timing + error capture
# =============================================================================

run_block <- function(label, expr, res) {
  log_msg("  [", label, "]")
  t0 <- proc.time()
  out <- tryCatch(expr, error = function(e) {
    log_msg("    ERROR: ", conditionMessage(e))
    res$errors[[label]] <<- conditionMessage(e)
    NULL
  })
  elapsed <- (proc.time() - t0)[["elapsed"]]
  res$timings[[label]] <<- round(elapsed, 2)
  out
}

# Save a plotly widget as a self-contained HTML file.
# Silently skips if htmlwidgets is not installed or the widget is NULL.
save_html <- function(widget, path) {
  if (is.null(widget)) return(invisible(NULL))
  if (!requireNamespace("htmlwidgets", quietly = TRUE)) {
    log_msg("    (htmlwidgets not installed — skipping HTML export)")
    return(invisible(NULL))
  }
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  htmlwidgets::saveWidget(widget,
                          file          = normalizePath(path, mustWork = FALSE),
                          selfcontained = TRUE,
                          libdir        = NULL)
  log_msg("    Saved: ", basename(path))
}

# =============================================================================
# Main loop — one model at a time
# =============================================================================

all_results <- list()

for (model_name in names(MODELS)) {

  spec      <- MODELS[[model_name]]
  repo_id   <- if (is.list(spec)) spec$repo_id else spec
  loader    <- if (is.list(spec)) spec$loader  %||% "hf_bert" else "hf_bert"
  prefix    <- if (is.list(spec)) spec$prefix  %||% ""        else ""
  model_dir <- file.path(OUT_DIR, model_name)
  dir.create(model_dir, showWarnings = FALSE)

  log_msg("=== ", model_name, "  (", repo_id, ") ===")

  vis_dir <- file.path(model_dir, "visuals")
  dir.create(vis_dir, showWarnings = FALSE)

  res <- list(model = model_name, repo_id = repo_id,
              timings = list(), errors = list())

  # ---- [1] Load encoder ------------------------------------------------
  enc <- run_block("load_encoder", {
    e <- if (loader == "specter2")
           load_specter2(adapter = repo_id)
         else
           load_hf_bert(repo_id, prefix = prefix)
    log_msg("    hidden=", e$config$hidden_size,
            "  layers=", e$config$num_hidden_layers,
            if (nzchar(e$prefix %||% "")) paste0("  prefix=\"", e$prefix, "\"") else "",
            "  pooling=", e$pooling %||% "mean")
    e
  }, res)
  if (is.null(enc)) { all_results[[model_name]] <- res; next }

  # ---- [2] Embed (cached) ----------------------------------------------
  cache_file <- file.path(model_dir, "embeddings.rds")
  emb <- run_block("embed_cached", {
    m <- embed_texts_cached(enc, docs,
                             cache_file = cache_file,
                             verbose    = TRUE)
    log_msg("    matrix ", nrow(m), " x ", ncol(m))
    m
  }, res)
  if (is.null(emb)) { all_results[[model_name]] <- res; next }

  # ---- [3] sweep_topics ------------------------------------------------
  sw <- run_block("sweep_topics", {
    sw <- sweep_topics(
      docs        = docs,
      embeddings  = emb,
      n_neighbors  = c(10L, 15L),
      n_components = c(5L),
      min_pts      = c(10L, 20L),
      sample_size  = 3000L,
      verbose      = FALSE
    )
    saveRDS(sw, file.path(model_dir, "sweep.rds"))
    log_msg("    ", nrow(sw$results), " combinations evaluated")
    sw
  }, res)
  # Visual: heatmap / scatter of n_topics across parameter combinations
  run_block("visual_sweep", {
    p <- visualize_sweep(sw)
    save_html(p, file.path(vis_dir, "sweep.html"))
    p
  }, res)

  # ---- [4] fit_bertopic (default) --------------------------------------
  fit <- run_block("fit_default", {
    f <- fit_bertopic(docs = docs, embeddings = emb, verbose = TRUE)
    n_topics <- length(setdiff(unique(f$clusters), -1L))
    saveRDS(f, file.path(model_dir, "fit_default.rds"))
    log_msg("    ", n_topics, " topics  |  noise=", sum(f$clusters == -1L))
    f
  }, res)

  if (!is.null(fit)) {

    # Visual: 2-D UMAP scatter of all documents coloured by topic
    run_block("visual_topics", {
      p <- visualize_topics(fit)
      save_html(p, file.path(vis_dir, "topics_scatter.html"))
      p
    }, res)

    # Visual: horizontal bar charts of top c-TF-IDF terms per topic
    run_block("visual_barchart", {
      p <- visualize_barchart(fit)
      save_html(p, file.path(vis_dir, "topics_barchart.html"))
      p
    }, res)

    # Visual: hierarchical dendrogram of topic relationships
    run_block("visual_hierarchy", {
      h <- hierarchical_topics(fit)
      saveRDS(h, file.path(model_dir, "hierarchy.rds"))
      p <- visualize_hierarchy(h, fit = fit)
      save_html(p, file.path(vis_dir, "topics_hierarchy.html"))
      p
    }, res)

    # ---- [5] topic_quality ---------------------------------------------
    tq <- run_block("topic_quality", {
      tq <- topic_quality(fit)
      saveRDS(tq, file.path(model_dir, "topic_quality.rds"))
      log_msg("    mean silhouette=",
              round(mean(tq$scores$silhouette, na.rm = TRUE), 3))
      tq
    }, res)
    # Visual: silhouette + density scores per topic as a bubble/bar chart
    run_block("visual_quality", {
      p <- visualize_quality(tq)
      save_html(p, file.path(vis_dir, "topic_quality.html"))
      p
    }, res)

    # ---- [6] topic_coherence (NPMI) ------------------------------------
    run_block("topic_coherence_npmi", {
      tc <- topic_coherence(fit, measure = "npmi")
      saveRDS(tc, file.path(model_dir, "coherence_npmi.rds"))
      log_msg("    mean NPMI=",
              round(mean(tc$coherence$score, na.rm = TRUE), 3))
      tc
    }, res)

    # ---- [7] topic_coherence (CV) --------------------------------------
    run_block("topic_coherence_cv", {
      tc <- topic_coherence(fit, measure = "cv")
      saveRDS(tc, file.path(model_dir, "coherence_cv.rds"))
      log_msg("    mean CV=",
              round(mean(tc$coherence$score, na.rm = TRUE), 3))
      tc
    }, res)

    # ---- [8] compare_topics --------------------------------------------
    comp <- run_block("compare_topics", {
      comp <- compare_topics(fit, groups, verbose = FALSE)
      saveRDS(comp, file.path(model_dir, "compare_topics.rds"))
      log_msg("    ", nrow(comp$table), " topic-group rows")
      comp
    }, res)
    # Visual: grouped bar chart showing topic share per group
    run_block("visual_comparison", {
      p <- visualize_comparison(comp)
      save_html(p, file.path(vis_dir, "topic_comparison.html"))
      p
    }, res)

    # ---- [9] reduce_topics (keep ~50 %) --------------------------------
    run_block("reduce_topics", {
      n_before <- length(setdiff(unique(fit$clusters), -1L))
      n_target <- max(2L, floor(n_before * 0.5))
      fr <- reduce_topics(fit, n_target, verbose = FALSE)
      saveRDS(fr, file.path(model_dir, "fit_reduced.rds"))
      log_msg("    ", n_before, " -> ",
              length(setdiff(unique(fr$clusters), -1L)), " topics")
      fr
    }, res)

    # ---- [10] reduce_outliers ------------------------------------------
    run_block("reduce_outliers", {
      n_out_before <- sum(fit$clusters == -1L)
      fo <- reduce_outliers(fit, strategy = "embeddings", verbose = FALSE)
      saveRDS(fo, file.path(model_dir, "fit_no_outliers.rds"))
      log_msg("    outliers ", n_out_before, " -> ", sum(fo$clusters == -1L))
      fo
    }, res)

    # ---- [11] topics_over_time -----------------------------------------
    tot <- run_block("topics_over_time", {
      tot <- topics_over_time(fit, timestamps = years)
      saveRDS(tot, file.path(model_dir, "topics_over_time.rds"))
      log_msg("    ", nrow(tot), " topic-year rows")
      tot
    }, res)
    # Visual: line chart of topic frequency over time
    run_block("visual_topics_over_time", {
      p <- visualize_topics_over_time(tot)
      save_html(p, file.path(vis_dir, "topics_over_time.html"))
      p
    }, res)
    # Visual: alluvial / stream chart of topic flow across time
    run_block("visual_topic_flow", {
      p <- visualize_topic_flow(tot)
      save_html(p, file.path(vis_dir, "topic_flow.html"))
      p
    }, res)

    # ---- [12] save / load bertopic -------------------------------------
    run_block("persistence", {
      save_path <- file.path(model_dir, "fit_saved")
      save_bertopic(fit, save_path)
      fit_re <- load_bertopic(save_path)
      ok <- identical(fit$topic_terms, fit_re$topic_terms)
      log_msg("    round-trip match: ", ok)
      fit_re
    }, res)

  }  # end !is.null(fit)

  # ---- [13] stability_analysis (independent of base fit) ---------------
  stab <- run_block("stability_analysis", {
    stab <- stability_analysis(docs, emb, n_runs = 3L, verbose = FALSE)
    saveRDS(stab, file.path(model_dir, "stability.rds"))
    log_msg("    mean ARI=", round(stab$mean_ari, 3))
    stab
  }, res)
  # Visual: ARI distribution across run pairs
  run_block("visual_stability", {
    p <- visualize_stability(stab)
    save_html(p, file.path(vis_dir, "stability.html"))
    p
  }, res)

  # ---- [14] pos_representation — VERB ----------------------------------
  run_block("fit_pos_verb", {
    fv <- fit_bertopic(
      docs                 = docs,
      embeddings           = emb,
      representation_model = pos_representation(pos = c("VERB"),
                                                lemmatize = TRUE),
      verbose              = FALSE
    )
    saveRDS(fv, file.path(model_dir, "fit_pos_verb.rds"))
    log_msg("    ", length(setdiff(unique(fv$clusters), -1L)), " topics (VERB repr)")
    fv
  }, res)

  # ---- [15] pos_representation — NOUN + PROPN -------------------------
  run_block("fit_pos_noun", {
    fn <- fit_bertopic(
      docs                 = docs,
      embeddings           = emb,
      representation_model = pos_representation(pos = c("NOUN", "PROPN"),
                                                lemmatize = TRUE),
      verbose              = FALSE
    )
    saveRDS(fn, file.path(model_dir, "fit_pos_noun.rds"))
    log_msg("    ", length(setdiff(unique(fn$clusters), -1L)), " topics (NOUN+PROPN repr)")
    fn
  }, res)

  # ---- [16] pos_representation — ADJ+NOUN patterns --------------------
  run_block("fit_pos_pattern", {
    fp <- fit_bertopic(
      docs                 = docs,
      embeddings           = emb,
      representation_model = pos_representation(
        patterns  = list(c("ADJ", "NOUN"), c("NOUN", "NOUN")),
        lemmatize = TRUE
      ),
      verbose              = FALSE
    )
    saveRDS(fp, file.path(model_dir, "fit_pos_pattern.rds"))
    log_msg("    ", length(setdiff(unique(fp$clusters), -1L)), " topics (ADJ+NOUN pattern)")
    fp
  }, res)

  # ---- [17] cvalue_representation -------------------------------------
  run_block("fit_cvalue", {
    fc <- fit_bertopic(
      docs                 = docs,
      embeddings           = emb,
      representation_model = cvalue_representation(max_n = 3L, threshold = 0.5),
      verbose              = FALSE
    )
    saveRDS(fc, file.path(model_dir, "fit_cvalue.rds"))
    log_msg("    ", length(setdiff(unique(fc$clusters), -1L)), " topics (C-value repr)")
    fc
  }, res)

  # ---- [18] zero_shot_topics ------------------------------------------
  run_block("zero_shot_topics", {
    fz <- zero_shot_topics(
      docs       = docs,
      labels     = ZERO_SHOT_LABELS,
      embeddings = emb,
      encoder    = enc,
      verbose    = FALSE
    )
    saveRDS(fz, file.path(model_dir, "fit_zero_shot.rds"))
    counts <- table(fz$clusters)
    log_msg("    assignments: ",
            paste(names(counts), counts, sep = "=", collapse = "  "))
    fz
  }, res)

  # ---- [19] guided_fit_bertopic ----------------------------------------
  run_block("guided_fit", {
    fg <- guided_fit_bertopic(
      docs            = docs,
      seed_topic_list = SEED_TOPICS,
      embeddings      = emb,
      encoder         = enc,
      verbose         = FALSE
    )
    saveRDS(fg, file.path(model_dir, "fit_guided.rds"))
    log_msg("    ", length(setdiff(unique(fg$clusters), -1L)), " topics (guided)")
    fg
  }, res)

  # ---- [20] fit with ngram_range (bigrams) + reduce_outliers ----------
  run_block("fit_bigrams", {
    fb <- fit_bertopic(
      docs        = docs,
      embeddings  = emb,
      ngram_range = c(1L, 2L),
      verbose     = FALSE
    )
    saveRDS(fb, file.path(model_dir, "fit_bigrams.rds"))
    log_msg("    ", length(setdiff(unique(fb$clusters), -1L)), " topics (bigrams)")
    fb
  }, res)

  res$completed_at <- format(Sys.time())
  all_results[[model_name]] <- res
  log_msg("=== Done: ", model_name, " ===\n")

  # Free memory before next model
  rm(enc, emb)
  gc()
}

# =============================================================================
# Summary table
# =============================================================================

all_steps <- c(
  "load_encoder", "embed_cached",
  "sweep_topics",        "visual_sweep",
  "fit_default",
  "visual_topics",       "visual_barchart",     "visual_hierarchy",
  "topic_quality",       "visual_quality",
  "topic_coherence_npmi", "topic_coherence_cv",
  "compare_topics",      "visual_comparison",
  "reduce_topics", "reduce_outliers",
  "topics_over_time",    "visual_topics_over_time", "visual_topic_flow",
  "persistence",
  "stability_analysis",  "visual_stability",
  "fit_pos_verb", "fit_pos_noun", "fit_pos_pattern", "fit_cvalue",
  "zero_shot_topics", "guided_fit", "fit_bigrams"
)

rows <- lapply(names(all_results), function(nm) {
  r <- all_results[[nm]]
  timing_row <- setNames(
    sapply(all_steps, function(s) r$timings[[s]] %||% NA_real_),
    all_steps
  )
  c(model = nm, as.list(timing_row))
})

summary_df <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
rownames(summary_df) <- NULL

log_msg("\n========== TIMING SUMMARY (seconds) ==========")
print(summary_df, digits = 1)

write.csv(summary_df, file.path(OUT_DIR, "timing_summary.csv"), row.names = FALSE)
saveRDS(all_results, file.path(OUT_DIR, "all_results.rds"))

log_msg("\nAll outputs in: ", normalizePath(OUT_DIR))
log_msg("Log: ", normalizePath(LOG_FILE))
