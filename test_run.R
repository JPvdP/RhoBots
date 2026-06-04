devtools::load_all(".")

docs <- df_sample$Abstract

# --- 0. Embedding cache (compute once, reuse across all experiments) ------
enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")

# First run: embeds and writes embeddings_sample.rds
# Every subsequent run: loads from disk, enc not used
emb <- embed_texts_cached(enc, docs, cache_file = "embeddings_sample.rds")

# --- 0b. Custom stopwords -------------------------------------------------
# From a plain character vector:
domain_sw <- c("study", "paper", "result", "results", "method", "methods",
               "approach", "propose", "proposed", "show", "using", "based")

# From a file (one word per line):
# domain_sw <- load_stopwords("domain_stopwords.txt")

# From a CSV with a "word" column:
# domain_sw <- load_stopwords("stopwords.csv", column = "word")

# See what the built-in English list contains:
cat("Built-in stopwords (first 20):", head(get_stopwords("english"), 20), "\n")

# --- 1. Fit model (embeddings pre-computed, extra stopwords applied) ------
fit <- fit_bertopic(docs             = docs,
                    embeddings        = emb,
                    umap_n_neighbors  = 15,
                    umap_n_components = 5,
                    hdbscan_min_pts   = 5,
                    top_n_terms       = 10,
                    language          = "english",
                    extra_stopwords   = domain_sw,
                    ngram_range       = c(1L, 2L),
                    reduce_frequent_words = TRUE,
                    verbose           = TRUE)

print(fit)

# PCA + k-means (fast, deterministic)
cat("\n--- PCA + k-means ---\n")
fit_km <- fit_bertopic(
  enc, docs,
  dim_reduction_model = pca_reduction(n_components = 10L),
  cluster_model       = kmeans_clustering(k = 8L),
  language = "english", ngram_range = c(1L, 2L),
  reduce_frequent_words = TRUE, verbose = TRUE
)
print(fit_km)

# UMAP + agglomerative
cat("\n--- UMAP + agglomerative ---\n")
fit_agg <- fit_bertopic(
  enc, docs,
  cluster_model = agglomerative_clustering(k = 6L, linkage = "ward.D2"),
  language = "english", ngram_range = c(1L, 2L),
  reduce_frequent_words = TRUE, verbose = TRUE
)
print(fit_agg)

# Leaf method — tends to produce more, finer-grained topics
fit_leaf <- fit_bertopic(enc, docs,
                         umap_n_neighbors      = 15,
                         umap_n_components     = 5,
                         hdbscan_min_pts       = 5,
                         hdbscan_method        = "leaf",
                         top_n_terms           = 10,
                         language              = "english",
                         ngram_range           = c(1L, 2L),
                         reduce_frequent_words = TRUE,
                         verbose               = TRUE)
cat("EOM topics:", length(setdiff(unique(fit$clusters),      -1L)), "\n")
cat("Leaf topics:", length(setdiff(unique(fit_leaf$clusters), -1L)), "\n")

# --- 2. Accessor methods --------------------------------------------------
cat("\n--- get_topic_info() ---\n")
info <- get_topic_info(fit)
print(info)

cat("\n--- get_topic(0) ---\n")
print(get_topic(fit, 0))

cat("\n--- get_topics() (first 3 topics, top 5 terms) ---\n")
all_topics <- get_topics(fit, top_n = 5)
print(all_topics[seq_len(min(3, length(all_topics)))])

cat("\n--- get_representative_docs() (topic 0) ---\n")
print(get_representative_docs(fit, topic = 0))

cat("\n--- get_document_info() (first 10 rows) ---\n")
doc_info <- get_document_info(fit)
print(head(doc_info, 10))

cat("\n--- find_topics('climate') ---\n")
print(find_topics(fit, "climate", top_n = 5))

# --- 3. Transform (predict on held-out docs) ------------------------------
cat("\n--- transform_bertopic() on 5 new docs ---\n")
new_docs <- docs[1:5]
result   <- transform_bertopic(fit, new_docs, encoder = enc)

cat("Assigned topics:", result$topics, "\n")
cat("Probabilities:  ", round(result$probabilities, 3), "\n")

cat("\nAll done.\n")

# --- 5. topics_over_time() ------------------------------------------------
# Assumes df_sample has a "Year" column (or replace with any timestamp column)
# --- 6. fit_topics_over_time() / Sankey flow --------------------------------
if ("Year" %in% names(df_sample)) {
  cat("\n--- fit_topics_over_time() ---\n")
  flow <- fit_topics_over_time(
    encoder    = enc,
    docs       = docs,
    timestamps = df_sample$Year,
    min_similarity  = 0.7,
    bertopic_params = list(
      hdbscan_min_pts       = 3L,
      ngram_range           = c(1L, 2L),
      language              = "english",
      reduce_frequent_words = TRUE,
      top_n_terms           = 8L
    )
  )
  print(flow)
  cat("\nTopic info:\n"); print(flow$topic_info)
  cat("\nTransitions:\n"); print(flow$transitions)

  p_flow <- visualize_topic_flow(flow, color_by = "status")
  print(p_flow)

  p_flow_period <- visualize_topic_flow(flow, color_by = "period")
  print(p_flow_period)
}

if ("Year" %in% names(df_sample)) {
  cat("\n--- topics_over_time() ---\n")
  tot <- topics_over_time(fit, timestamps = df_sample$Year,
                           evolution_tuning = TRUE,
                           global_tuning    = TRUE)
  print(head(tot, 20))

  cat("\n--- topics_over_time() with binning ---\n")
  tot_binned <- topics_over_time(fit, timestamps = df_sample$Year,
                                  nr_bins = 5)
  print(tot_binned)

  cat("\n--- visualize_topics_over_time() ---\n")
  p_tot <- visualize_topics_over_time(tot)
  print(p_tot)
}

# --- 4. Topic manipulation ------------------------------------------------
cat("\n--- reduce_outliers() ---\n")
fit2 <- reduce_outliers(fit, strategy = "embeddings")
cat("Noise before:", sum(fit$clusters  == -1), "\n")
cat("Noise after: ", sum(fit2$clusters == -1), "\n")
print(get_topic_info(fit2))

cat("\n--- reduce_outliers() with c-tf-idf strategy ---\n")
fit3 <- reduce_outliers(fit, strategy = "c-tf-idf")
cat("Noise after (c-tf-idf): ", sum(fit3$clusters == -1), "\n")

cat("\n--- merge_topics() (merge topics 0 and 1) ---\n")
fit4 <- merge_topics(fit, topics_to_merge = c(0L, 1L))
print(get_topic_info(fit4))

cat("\n--- apply_mmr() ---\n")
cat("Before MMR:\n")
print(get_topic_info(fit)[, c("Topic", "Representation")])

fit_mmr <- apply_mmr(fit, enc, diversity = 0.3)

cat("\nAfter MMR (diversity = 0.3):\n")
print(get_topic_info(fit_mmr)[, c("Topic", "Representation")])

# Side-by-side barchart comparison
p_before <- visualize_barchart(fit,     top_n = 8L, n_cols = 4L)
p_after  <- visualize_barchart(fit_mmr, top_n = 8L, n_cols = 4L)

cat("\n--- visualize_barchart() ---\n")
# All topics, top 8 terms, 4 columns
p_bar <- visualize_barchart(fit, top_n = 8L, n_cols = 4L)
print(p_bar)

# Specific topics only
p_bar2 <- visualize_barchart(fit, topics = 0:2, top_n = 10L, n_cols = 3L)
print(p_bar2)

cat("\n--- visualize_topics() ---\n")
# Default: pre-computed 2-D UMAP layout
p1 <- visualize_topics(fit)
print(p1)

# Pick specific dimensions from the clustering UMAP space
p2 <- visualize_topics(fit, dims = c(1L, 2L), label_topics = TRUE)
print(p2)

cat("\n--- reduce_topics() (target: 3 topics) ---\n")
n_start <- nrow(get_topic_info(fit)[get_topic_info(fit)$Topic != -1, ])
cat("Topics before reduction:", n_start, "\n")
fit5 <- reduce_topics(fit, nr_topics = max(3L, ceiling(n_start / 2)))
print(get_topic_info(fit5))
