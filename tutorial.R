# =============================================================================
# RHOBOTS TUTORIAL — End-to-End Topic Modeling on Project Gutenberg Texts
# =============================================================================
#
# This script walks through the complete Rhobots workflow on four classic
# texts from Project Gutenberg, each representing a distinct intellectual
# domain:
#
#   Darwin   (1228) — natural science: evolution, species, geology
#   Smith    (3300) — economics: labour, wages, capital, trade
#   Holmes   (1661) — detective fiction: crime, investigation, London
#   Plato    (1497) — philosophy: justice, the ideal state, virtue
#
# Because these books are written in very different registers, the embedding
# model has a strong signal to work with, making it easy to validate that the
# discovered topics correspond to meaningful intellectual themes.
#
# Required packages:
#   install.packages(c("Rhobots", "gutenbergr"))   # gutenbergr for text data
#
# Workflow:
#   1.  Load and pre-process text data
#   2.  Load a sentence embedding model and embed the corpus
#   3.  Explore the hyperparameter space with sweep_topics()
#   4.  Visualise and interpret the sweep results
#   5.  Fit the final model and evaluate
# =============================================================================

library(Rhobots)
library(gutenbergr)    # install.packages("gutenbergr") if needed


# =============================================================================
# STEP 1 — LOAD AND PRE-PROCESS TEXT DATA
# =============================================================================
# gutenberg_download() returns a data frame with one row per line of text.
# We download four books in one call and retain the title metadata.

cat("Downloading texts from Project Gutenberg...\n")
books_raw <- gutenberg_download(
  c(1228L, 3300L, 1661L, 1497L),
  meta_fields = "title"
)

cat(sprintf("Downloaded %d lines from %d books.\n",
            nrow(books_raw), length(unique(books_raw$gutenberg_id))))

# --- Split raw lines into paragraph-level documents -------------------------
# Each book is a sequence of lines. Blank lines mark paragraph boundaries.
# We join runs of non-blank lines, then keep only paragraphs in the sweet
# spot of 40–400 words: long enough to carry semantic content, short enough
# to stay topically focused.

split_paragraphs <- function(df, min_words = 40L, max_words = 400L) {
  do.call(rbind, lapply(split(df, df$gutenberg_id), function(book) {
    lines   <- book$text
    title   <- book$title[1L]
    gid     <- book$gutenberg_id[1L]

    # Identify blank lines and use them as paragraph boundaries
    is_blank <- trimws(lines) == ""
    grp      <- cumsum(is_blank)          # increment group at every blank line
    lines    <- lines[!is_blank]
    grp      <- grp[!is_blank]

    # Collapse each group of lines into one string
    paras   <- tapply(lines, grp, function(x) paste(trimws(x), collapse = " "))
    n_words <- lengths(strsplit(paras, "\\s+"))

    keep    <- n_words >= min_words & n_words <= max_words
    paras   <- as.character(paras[keep])

    data.frame(
      gutenberg_id = gid,
      title        = title,
      text         = paras,
      stringsAsFactors = FALSE
    )
  }))
}

corpus <- split_paragraphs(books_raw, min_words = 40L, max_words = 400L)

cat(sprintf("\nCorpus after splitting: %d paragraphs\n", nrow(corpus)))
print(table(corpus$title))

# The working vector of documents
docs <- corpus$text


# =============================================================================
# STEP 2 — LOAD A SENTENCE EMBEDDING MODEL AND EMBED THE CORPUS
# =============================================================================
# all-MiniLM-L6-v2 is a fast, general-purpose sentence encoder.  It maps
# each paragraph to a 384-dimensional unit vector where cosine similarity
# approximates semantic similarity.  It's a sensible default before investing
# in a larger or domain-specific model.
#
# For scientific text, consider:
#   enc <- load_hf_bert("allenai/scibert_scivocab_uncased")
# For patent text:
#   enc <- load_hf_bert("anferico/bert-for-patents",
#                       weights_path = "/local/model.safetensors")

cat("\nLoading encoder from HuggingFace...\n")
enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
print(enc)

# embed_texts_cached() runs the encoder on the first call and saves the
# result to disk.  Every subsequent call with the same cache_file loads
# in under a second — no GPU or warm encoder needed.
#
# On a modern CPU, ~1 000 paragraphs takes roughly 30–90 seconds the
# first time.  Subsequent runs are instant.
cat("\nEmbedding corpus (or loading from cache)...\n")
emb <- embed_texts_cached(
  enc,
  docs,
  cache_file = "gutenberg_embeddings.rds",
  normalize  = TRUE     # L2-normalised: cosine similarity = dot product
)

cat(sprintf("Embedding matrix: %d paragraphs × %d dimensions\n",
            nrow(emb), ncol(emb)))


# =============================================================================
# STEP 3 — EXPLORE THE HYPERPARAMETER SPACE
# =============================================================================
# BERTopic has three main parameters that interact:
#
#   n_neighbors  (UMAP)
#     Controls the balance between local and global structure in the manifold
#     learning step.  Small values capture fine-grained local clusters; large
#     values reveal the broader global geometry.  Typical range: 5–50.
#
#   n_components (UMAP)
#     Dimensionality of the compressed space that is fed into HDBSCAN.
#     More components = more signal retained, but also a noisier clustering
#     surface.  Typical range: 3–15.
#
#   min_pts  (HDBSCAN)
#     Minimum number of documents required to form a cluster.  Larger values
#     → fewer, larger topics and more noise.  Smaller values → more topics,
#     finer-grained.  Set relative to corpus size (≈ 1–5 % of corpus).
#
# sweep_topics() runs fit_bertopic + topic_quality for every combination.
# Because the embeddings are pre-computed and passed in, the encoder only
# runs once.  The dominant cost is UMAP (a few seconds per run).

cat("\nRunning hyperparameter sweep...\n")
sw <- sweep_topics(
  docs         = docs,
  embeddings   = emb,
  n_neighbors  = c(10L, 20L, 30L),
  n_components = c(5L, 10L),
  min_pts      = c(10L, 20L, 30L),
  ngram_range    = c(1L, 2L),
  quality_top_n  = 10L,
  quality_sample = 1000L,    # docs used for silhouette (O(n²) step)
  seed    = 42L,
  verbose = TRUE
)
# 18 combinations.  Each takes ~5–15 s on CPU → ≈ 2–5 min total.

print(sw)

# Sorted results table: best silhouette at the top
results_sorted <- sw$results[order(-sw$results$silhouette, na.last = TRUE), ]
cat("\nSweep results (sorted by silhouette):\n")
print(
  results_sorted[, c("n_neighbors", "n_components", "min_pts",
                     "n_topics", "noise_pct", "silhouette",
                     "cohesion", "jaccard")],
  row.names = FALSE, digits = 3L
)


# =============================================================================
# STEP 4 — VISUALISE AND INTERPRET THE SWEEP RESULTS
# =============================================================================
# The heatmap produced by visualize_sweep() shows all 18 combinations at
# once.  Each column is one quality metric, min-max normalised within the
# column so that the best value in each metric is always darkest green.
# Metrics where a lower raw value is better (separation, Jaccard overlap,
# noise %) are inverted before normalisation, so green consistently means
# "more desirable".
#
# Reading guide:
#
#   Uniformly green rows
#     Robust parameter choices — good across all metrics.  Prefer these.
#
#   High silhouette + high noise_pct
#     Topics are tight, but too many documents are unassigned.  Consider
#     reducing min_pts to let smaller clusters form.
#
#   Low Jaccard + high silhouette
#     Topics are both well-separated in embedding space AND lexically
#     distinct.  This is the ideal target region.
#
#   "err" cells
#     The run produced an all-noise result (min_pts was too large for
#     the corpus size).  Ignore these rows.

cat("\nOpening interactive sweep visualisation...\n")
visualize_sweep(sw)

# Extract the recommended parameter set (highest silhouette)
best <- sw$best
cat(sprintf(
  "\nBest parameters identified:\n  n_neighbors  = %d\n  n_components = %d\n  min_pts      = %d\n",
  best$n_neighbors, best$n_components, best$min_pts
))
cat(sprintf(
  "  Expected: %d topics   silhouette = %.3f   noise = %.1f%%\n",
  as.integer(best$n_topics), best$silhouette, best$noise_pct
))


# =============================================================================
# STEP 5 — FINAL MODEL WITH OPTIMAL PARAMETERS
# =============================================================================
# Re-fit on the full corpus using the parameters identified in Step 4.
# We add bigrams (ngram_range = c(1L, 2L)) to capture multi-word phrases
# such as "natural selection" or "division of labour".

cat("\nFitting final model...\n")
fit <- fit_bertopic(
  docs              = docs,
  embeddings        = emb,
  umap_n_neighbors  = best$n_neighbors,
  umap_n_components = best$n_components,
  hdbscan_min_pts   = best$min_pts,
  ngram_range       = c(1L, 2L),
  top_n_terms       = 10L,
  seed              = 42L,
  verbose           = TRUE
)

print(fit)
cat("\n")
print_topics(fit)


# --- 5a. Topic quality metrics -----------------------------------------------
# Four complementary views of how well-defined the topics are:
#
#   Silhouette   Combined measure of intra-topic compactness and
#                inter-topic separation.  [-1, 1]; higher is better.
#
#   Cohesion     Mean cosine similarity of each document to its topic
#                centroid.  [0, 1]; higher is better.
#
#   Separation   Mean pairwise cosine similarity between centroids.
#                [0, 1]; lower means more distinct topics.
#
#   Jaccard      Mean pairwise overlap of the top-10 c-TF-IDF term sets.
#                [0, 1]; lower means topics share fewer words.

cat("\nComputing topic quality metrics...\n")
q <- topic_quality(fit, top_n = 10L)
print(q)

# Four-panel interactive dashboard:
#   top-left  — per-topic silhouette (red→green bars)
#   top-right — document counts per topic + noise class
#   bot-left  — pairwise centroid similarity matrix
#   bot-right — pairwise vocabulary overlap (Jaccard) matrix
visualize_quality(q, fit)


# --- 5b. Topic term bar charts -----------------------------------------------
# c-TF-IDF scores: how distinctive each term is to its topic compared
# with the rest of the corpus.  Taller bars = more characteristic.
# This is the standard way to present topic content to a non-specialist
# audience.

visualize_barchart(fit, top_n = 8L)


# --- 5c. Document map --------------------------------------------------------
# Interactive 2-D scatter (UMAP layout) coloured by topic assignment.
# Hover over any point to read the paragraph text and its topic label.
# Well-separated clouds confirm that the chosen parameters produce
# geometrically coherent topics.

visualize_topics(fit, label_topics = TRUE, n_label_words = 3L)


# --- 5d. Representative documents --------------------------------------------
# The three paragraphs closest to each topic centroid in embedding space.
# Inspecting these is the fastest sanity check: do they actually discuss
# what the topic label suggests?

cat("\nRepresentative documents per topic:\n")
rep_docs <- get_representative_docs(fit)

for (tid in sort(setdiff(unique(fit$clusters), -1L))) {
  lbl <- fit$topic_labels[[as.character(tid)]] %||% paste("Topic", tid)
  cat(sprintf("\n══ Topic %d — %s ══\n", tid, lbl))
  for (doc in rep_docs[[as.character(tid)]]) {
    cat(strwrap(doc, width = 72L, exdent = 2L), sep = "\n")
    cat("\n")
  }
}


# --- 5e. Topic-source alignment ---------------------------------------------
# Because we know which book each paragraph came from, we can check whether
# the discovered topics align with the source texts.  A high-quality model
# should assign most Darwin paragraphs to biology topics, most Smith
# paragraphs to economics topics, and so on.

doc_info <- get_document_info(
  fit,
  metadata = list(
    book         = corpus$title,
    gutenberg_id = corpus$gutenberg_id
  )
)

cat("\nTopic × source book cross-tabulation:\n")
cross <- table(Topic = doc_info$Topic, Book = corpus$title)
print(cross)

# Share of each topic's documents that come from each book
cat("\nRow proportions (how 'pure' each topic is by source):\n")
print(round(prop.table(cross, margin = 1L), 2L))


# --- 5f. Export results ------------------------------------------------------
write.csv(doc_info, "gutenberg_topic_assignments.csv", row.names = FALSE)
cat("\nTopic assignments saved to gutenberg_topic_assignments.csv\n")
cat(sprintf("Columns: %s\n", paste(names(doc_info), collapse = ", ")))
