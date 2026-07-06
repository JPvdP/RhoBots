# =============================================================================
# africa_analysis.R
#
# End-to-end Rhobots demonstration using African research publications.
#
# DATA
# ----
#   Africa_abstracts.csv                          — 3,328 abstracts (EID + Abstract)
#   UU_scopus_country_clean_lonlat_continent.rdata — affiliation/location data
#     (one row per affiliation per paper: EID, clean2 org name, main_country,
#      Continent, Year, lon, lat, iso2)
#
# OUTLINE
# -------
#   Part 1 — Descriptive statistics (orgs, countries, temporal trends)
#   Part 2 — Parameter sweep (which UMAP / HDBSCAN settings work best?)
#   Part 3 — Fit the topic model with the chosen parameters
#   Part 4 — Analysis through the lens of the discovered topics
#             (LLM labeling, topic map, countries per topic, orgs per topic)
#
# OUTPUT
# ------
#   output/                 directory created by the script
#     01_country_bar.html           top African countries
#     02_org_bar.html               top African organisations
#     03_publication_trend.html     publications per year by continent
#     04_collab_heatmap.html        African × world-region co-authorship
#     05_sweep.html                 hyperparameter sweep heatmap
#     06_topics_scatter.html        UMAP document map coloured by topic
#     07_topics_barchart.html       c-TF-IDF term scores per topic
#     08_topic_quality.html         silhouette / cohesion / overlap dashboard
#     09_topic_map_country.html     world map coloured by dominant topic
#     10_countries_per_topic.html   stacked bar: country share within each topic
#     11_orgs_per_topic.html        top African organisations per topic
#     12_topic_over_time.html       topic prevalence 2015–2024
#     13_topic_comparison.html      topic × continent diverging heatmap
#     embeddings_cache.rds          cached embeddings (reused on re-runs)
#
# PREREQUISITES
# -------------
#   library(Rhobots)      — install from GitHub: pak::pak("JPvdP/Rhobots")
#   library(plotly)       — CRAN
#   library(htmlwidgets)  — CRAN
#   library(dplyr)        — CRAN
#   library(tidyr)        — CRAN (for pivot_wider in co-authorship heatmap)
#   Ollama running locally with llama3.2 pulled:
#     ollama serve          (in a terminal)
#     ollama pull llama3.2
# =============================================================================

library(Rhobots)
library(plotly)
library(htmlwidgets)
library(dplyr)
library(tidyr)

# ── helpers ──────────────────────────────────────────────────────────────────
save_html <- function(p, path) {
  if (is.null(p)) return(invisible(NULL))
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  htmlwidgets::saveWidget(p,
                          file = normalizePath(path, mustWork = FALSE),
                          selfcontained = TRUE, libdir = NULL)
  message("  saved: ", basename(path))
  invisible(p)
}

OUT <- "output"
dir.create(OUT, showWarnings = FALSE)

# =============================================================================
# LOAD DATA
# =============================================================================
message("\n=== Loading data ===")

abstracts <- read.csv("Africa_abstracts.csv", stringsAsFactors = FALSE)
load("UU_scopus_country_clean_lonlat_continent.rdata")
affil <- UU_scopus_country_clean

# Keep only rows matching our abstract set (should be all of them)
affil <- affil[affil$EID %in% abstracts$EID, ]

# Paper-level metadata: take one row per paper and record year + a flag
# for whether the paper has at least one African co-author.
paper_meta <- affil |>
  group_by(EID) |>
  summarise(
    Year        = first(Year),
    n_affil     = n(),
    has_africa  = any(Continent == "Africa"),
    n_africa    = sum(Continent == "Africa"),
    .groups     = "drop"
  )

message(sprintf("Abstracts: %d | Affiliation rows: %d | Papers with ≥1 African affil: %d",
                nrow(abstracts), nrow(affil), sum(paper_meta$has_africa)))

# =============================================================================
# PART 1 — DESCRIPTIVE STATISTICS
# =============================================================================
message("\n=== Part 1: Descriptive statistics ===")

# ── 1a. Top African countries by number of affiliated papers ─────────────────
af_country <- affil |>
  filter(Continent == "Africa") |>
  distinct(EID, main_country) |>          # one row per (paper, country)
  count(main_country, name = "n_papers") |>
  arrange(desc(n_papers)) |>
  head(20)

p_country <- plot_ly(af_country |> arrange(n_papers),
                     x = ~n_papers, y = ~reorder(main_country, n_papers),
                     type = "bar", orientation = "h",
                     marker = list(color = "#2ca25f")) |>
  layout(title  = "Top 20 African countries by number of affiliated papers",
         xaxis  = list(title = "Number of papers"),
         yaxis  = list(title = ""),
         margin = list(l = 130))
save_html(p_country, file.path(OUT, "01_country_bar.html"))

# ── 1b. Top African organisations ────────────────────────────────────────────
af_org <- affil |>
  filter(Continent == "Africa", !is.na(clean2), nzchar(clean2)) |>
  distinct(EID, clean2) |>
  count(clean2, name = "n_papers") |>
  arrange(desc(n_papers)) |>
  head(20)

p_org <- plot_ly(af_org |> arrange(n_papers),
                 x = ~n_papers, y = ~reorder(clean2, n_papers),
                 type = "bar", orientation = "h",
                 marker = list(color = "#2171b5")) |>
  layout(title  = "Top 20 African organisations by number of affiliated papers",
         xaxis  = list(title = "Number of papers"),
         yaxis  = list(title = ""),
         margin = list(l = 200))
save_html(p_org, file.path(OUT, "02_org_bar.html"))

# ── 1c. Publication trend by continent ───────────────────────────────────────
trend <- affil |>
  distinct(EID, Continent, Year) |>
  count(Year, Continent, name = "n") |>
  filter(!is.na(Year))

continents <- sort(unique(trend$Continent))
cont_cols  <- setNames(
  c("#e41a1c","#377eb8","#4daf4a","#ff7f00","#984ea3","#a65628"),
  c("Africa","Asia","Europe","North America","Oceania","South America")
)

p_trend <- plot_ly()
for (cont in continents) {
  d <- trend[trend$Continent == cont, ]
  p_trend <- add_trace(p_trend, data = d, x = ~Year, y = ~n,
                       type = "scatter", mode = "lines+markers",
                       name = cont,
                       line   = list(color = cont_cols[cont], width = 2),
                       marker = list(color = cont_cols[cont], size = 5))
}
p_trend <- layout(p_trend,
                  title  = "Affiliated organisations per continent, 2015–2024",
                  xaxis  = list(title = "Year"),
                  yaxis  = list(title = "Number of affiliated organisation rows"),
                  legend = list(title = list(text = "Continent")))
save_html(p_trend, file.path(OUT, "03_publication_trend.html"))

# ── 1d. African × world-region co-authorship heatmap ─────────────────────────
# For each paper, count how many African countries co-occur with each
# world region.
af_eids <- unique(affil$EID[affil$Continent == "Africa"])

colab <- affil |>
  filter(EID %in% af_eids) |>
  distinct(EID, main_country, Continent) |>
  inner_join(
    affil |>
      filter(Continent == "Africa") |>
      distinct(EID, africa_country = main_country),
    by = "EID"
  ) |>
  filter(Continent != "Africa") |>          # cross-continental links only
  count(africa_country, Continent, name = "n") |>
  filter(n >= 5)                            # filter noise

# Keep top 15 African countries for readability
top_af <- af_country$main_country[1:15]
colab_mat <- colab |>
  filter(africa_country %in% top_af) |>
  tidyr::pivot_wider(names_from = Continent, values_from = n, values_fill = 0)

colab_m <- as.matrix(colab_mat[, -1])
rownames(colab_m) <- colab_mat$africa_country

p_colab <- plot_ly(
  x = colnames(colab_m),
  y = rownames(colab_m),
  z = colab_m,
  type = "heatmap",
  colorscale = "Blues",
  hovertemplate = "%{y} × %{x}: %{z} co-authorships<extra></extra>"
) |>
  layout(title  = "African country × world region co-authorship frequency",
         xaxis  = list(title = "World region"),
         yaxis  = list(title = "African country"),
         margin = list(l = 150, b = 60))
save_html(p_colab, file.path(OUT, "04_collab_heatmap.html"))

message("Part 1 done — 4 figures saved.")

# =============================================================================
# PART 2 — EMBED AND PARAMETER SWEEP
# =============================================================================
message("\n=== Part 2: Embedding + parameter sweep ===")

# Load encoder — all-MiniLM-L6-v2 is a fast 384-D general-purpose model.
# For African research corpora a scientific model (SPECTER2, SciBERT) would
# give higher topic coherence; MiniLM is used here for speed.
enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")

docs <- abstracts$Abstract

# embed_texts_cached() runs the encoder on the first call and saves to disk.
# Every subsequent call loads the cached .rds — no GPU or long wait needed.
message("Embedding ", length(docs), " abstracts (cached after first run)...")
emb <- embed_texts_cached(enc, docs,
                           cache_file = file.path(OUT, "embeddings_cache.rds"),
                           normalize  = TRUE,
                           verbose    = TRUE)
message(sprintf("Embedding matrix: %d × %d", nrow(emb), ncol(emb)))

# ── Parameter sweep ──────────────────────────────────────────────────────────
# sweep_topics() evaluates every combination of UMAP and HDBSCAN parameters.
# We test:
#   n_neighbors  — controls how local vs. global the UMAP manifold is.
#                  Small values → fine-grained local clusters.
#                  Large values → smoother, more global structure.
#   n_components — dimensionality of the UMAP space fed to HDBSCAN.
#                  Higher → richer but noisier representation.
#   min_pts      — minimum cluster size in HDBSCAN.
#                  Small → many small topics + low noise.
#                  Large → few big topics + high noise.
#
# For each combination the sweep reports:
#   silhouette   — how well-separated documents are from other topics (higher better)
#   cohesion     — mean cosine similarity within topics (higher better)
#   separation   — mean cosine between topic centroids (lower better = more distinct)
#   jaccard      — vocabulary overlap between topics (lower better)
#   noise %      — share of documents left unclustered (lower usually better)
message("Running parameter sweep (this takes a few minutes)...")

sw <- sweep_topics(
  docs         = docs,
  embeddings   = emb,
  n_neighbors  = c(10L, 20L, 30L),
  n_components = c(5L, 10L),
  min_pts      = c(10L, 20L, 30L),
  ngram_range    = c(1L, 2L),
  quality_top_n  = 10L,
  quality_sample = 1000L,
  seed    = 42L,
  verbose = TRUE
)

# Print the sweep summary table
print(sw)

# visualize_sweep() produces an interactive heatmap:
# - rows  = parameter combinations
# - cols  = quality metrics (all normalised 0–1 within column, green = best)
# - star  = combination with the highest silhouette score (sw$best)
p_sweep <- visualize_sweep(sw)
save_html(p_sweep, file.path(OUT, "05_sweep.html"))

# Inspect the best combination
best <- sw$best
message(sprintf(
  "Best parameters: n_neighbors=%d  n_components=%d  min_pts=%d  silhouette=%.3f",
  best$n_neighbors, best$n_components, best$min_pts,
  sw$results$silhouette[sw$results$n_neighbors == best$n_neighbors &
                        sw$results$n_components == best$n_components &
                        sw$results$min_pts      == best$min_pts]
))
message("Part 2 done — parameter sweep complete.")

# =============================================================================
# PART 3 — FIT THE TOPIC MODEL
# =============================================================================
message("\n=== Part 3: Fitting the topic model ===")

# Use the best parameters identified by the sweep.
# We also add bigrams (ngram_range = c(1,2)) to capture compound terms such as
# "south africa", "climate change", "public health" etc., and apply
# reduce_frequent_words to dampen terms that dominate a single topic.
fit <- fit_bertopic(
  docs                  = docs,
  embeddings            = emb,
  umap_n_neighbors      = best$n_neighbors,
  umap_n_components     = best$n_components,
  hdbscan_min_pts       = best$min_pts,
  ngram_range           = c(1L, 2L),
  top_n_terms           = 10L,
  reduce_frequent_words = TRUE,
  extra_stopwords       = c("study", "paper", "result", "results",
                            "analysis", "data", "method", "methods",
                            "approach", "model", "models", "show",
                            "showed", "significant", "significantly",
                            "using", "used", "associated", "among",
                            "based", "found", "findings"),
  seed = 42L
)

print(fit)
print_topics(fit)

# ── Topic quality ─────────────────────────────────────────────────────────────
q <- topic_quality(fit, top_n = 10L)
print(q)

p_quality <- visualize_quality(q, fit)
save_html(p_quality, file.path(OUT, "08_topic_quality.html"))

# ── UMAP document scatter (coloured by topic) ─────────────────────────────────
p_scatter <- visualize_topics(fit)
save_html(p_scatter, file.path(OUT, "06_topics_scatter.html"))

# ── c-TF-IDF bar charts ───────────────────────────────────────────────────────
p_bars <- visualize_barchart(fit, top_n = 8L)
save_html(p_bars, file.path(OUT, "07_topics_barchart.html"))

message("Part 3 done — topic model fitted.")

# =============================================================================
# PART 4 — ANALYSIS THROUGH TOPIC LENS
# =============================================================================
message("\n=== Part 4: Analysis through topic lens ===")

# ── 4a. Label topics with Ollama ─────────────────────────────────────────────
# label_topics_llm() sends each topic's top terms + 3 representative abstracts
# to a local Ollama model and replaces the auto-generated slug labels with
# a concise 3–5-word human description.
#
# Prerequisites:
#   1. Run `ollama serve` in a terminal (or start the Ollama desktop app).
#   2. Run `ollama pull llama3.2` once to download the model (~2 GB).
#
# If Ollama is not available, comment out the next two lines — the rest of
# the analysis uses the automatic c-TF-IDF slugs.
message("Labeling topics with Ollama llama3.2 (ensure `ollama serve` is running)...")
fit <- label_topics_llm(fit, provider = "ollama", model = "llama3.2",
                        top_n_terms = 10L, n_representative_docs = 3L)

# Show the labels
topic_info <- get_topic_info(fit)
print(topic_info[, c("Topic", "Count", "Name")])

# ── 4b. Attach topic assignments to the paper-level affiliation data ──────────
# fit_bertopic() assigns each document in docs[] to a topic.
# We join those assignments back to the affiliation table using the original
# document order (EIDs are in the same order as docs[]).
paper_topics <- data.frame(
  EID   = abstracts$EID,
  topic = fit$clusters,           # -1 = noise; 0, 1, 2, … = topic IDs
  stringsAsFactors = FALSE
)

# Human-readable topic label, formatted (no underscores, truncated to 30 chars)
.fmt <- function(raw) {
  text <- sub("^-?[0-9]+_", "", raw)   # strip numeric id prefix
  text <- gsub("_", " ", text)          # underscores -> spaces (auto labels)
  text <- trimws(text)
  if (nchar(text) > 30L) paste0(substr(text, 1L, 29L), "…") else text
}

paper_topics$topic_label <- vapply(paper_topics$topic, function(t) {
  if (t == -1L) return("Noise")
  lbl <- fit$topic_labels[[as.character(t)]]
  if (is.null(lbl) || !nzchar(lbl)) paste0("Topic ", t) else .fmt(lbl)
}, character(1))

# Merge: affiliation table gets a topic column (all rows for a paper share one topic)
affil_topics <- affil |>
  left_join(paper_topics, by = "EID")

# ── 4b2. 2-D topic map ────────────────────────────────────────────────────────
# One bubble per topic, positioned at its centroid in the UMAP document space.
# Bubble size = number of documents in the topic (log-scaled for readability).
# Top terms are shown in the hover tooltip.
topics_nonnoise <- sort(setdiff(unique(fit$clusters), -1L))
coords2d        <- fit$layout2d

topic_map_df <- data.frame(
  topic = topics_nonnoise,
  x     = vapply(topics_nonnoise,
                 function(t) mean(coords2d[fit$clusters == t, 1L]), numeric(1L)),
  y     = vapply(topics_nonnoise,
                 function(t) mean(coords2d[fit$clusters == t, 2L]), numeric(1L)),
  n     = vapply(topics_nonnoise,
                 function(t) sum(fit$clusters == t), integer(1L)),
  label = vapply(topics_nonnoise, function(t) {
    lbl <- fit$topic_labels[[as.character(t)]]
    if (is.null(lbl) || !nzchar(lbl)) paste0("Topic ", t) else .fmt(lbl)
  }, character(1L)),
  stringsAsFactors = FALSE
)

# Top terms per topic for hover text
topic_map_df$top_terms <- vapply(topics_nonnoise, function(t) {
  tt <- fit$topic_terms[fit$topic_terms$topic == t, ]
  tt <- tt[order(tt$rank), ]
  paste(head(tt$term, 8L), collapse = ", ")
}, character(1L))

topic_map_df$hover <- paste0(
  "<b>", topic_map_df$label, "</b><br>",
  "Documents: ", topic_map_df$n, "<br>",
  "Terms: ", topic_map_df$top_terms
)

pal_tm <- grDevices::hcl.colors(nrow(topic_map_df), "Dynamic")

p_topic_map <- plot_ly(
  topic_map_df,
  x    = ~x, y = ~y,
  type = "scatter", mode = "markers+text",
  marker = list(
    size    = ~sqrt(n) * 3,
    color   = pal_tm,
    opacity = 0.85,
    line    = list(color = "white", width = 1.5)
  ),
  text         = ~label,
  textposition = "top center",
  textfont     = list(size = 11L),
  hovertext    = ~hover,
  hoverinfo    = "text"
) |>
  layout(
    title  = "2-D topic map (bubble size = document count)",
    xaxis  = list(title = "UMAP 1", zeroline = FALSE, showgrid = FALSE),
    yaxis  = list(title = "UMAP 2", zeroline = FALSE, showgrid = FALSE),
    plot_bgcolor  = "#f7f7f7",
    paper_bgcolor = "white",
    showlegend    = FALSE
  )
save_html(p_topic_map, file.path(OUT, "06b_topic_map_2d.html"))

# ── 4c. World map coloured by dominant topic per country ─────────────────────
# For each country, find its modal (most common) topic across all its papers.
country_topic <- affil_topics |>
  filter(topic != -1L, !is.na(main_country)) |>
  count(main_country, topic, topic_label, name = "n") |>
  group_by(main_country) |>
  slice_max(order_by = n, n = 1L, with_ties = FALSE) |>
  ungroup() |>
  left_join(
    affil |>
      distinct(main_country, lon, lat, iso2) |>
      group_by(main_country) |>
      slice(1L) |>
      ungroup(),
    by = "main_country"
  )

# Unique topics and a colour palette
all_labels <- sort(unique(country_topic$topic_label))
pal <- setNames(
  colorRampPalette(c("#e41a1c","#377eb8","#4daf4a","#ff7f00",
                     "#984ea3","#a65628","#f781bf","#999999",
                     "#66c2a5","#fc8d62","#8da0cb","#e78ac3"))(length(all_labels)),
  all_labels
)

p_map <- plot_ly()
for (lbl in all_labels) {
  d <- country_topic[country_topic$topic_label == lbl, ]
  p_map <- add_trace(p_map, data = d,
                     type = "scattergeo",
                     lat  = ~lat, lon = ~lon,
                     text = ~paste0(main_country, "<br>Dominant topic: ", topic_label,
                                    "<br>Papers in topic: ", n),
                     hoverinfo = "text",
                     mode   = "markers",
                     name   = lbl,
                     marker = list(color = pal[lbl], size = 8,
                                   line  = list(color = "white", width = 0.5)))
}
p_map <- layout(p_map,
                title = "Dominant research topic per country (modal topic across all papers)",
                geo   = list(showland = TRUE, landcolor = "#f5f5f5",
                             showcoastlines = TRUE, coastlinecolor = "#cccccc",
                             showframe = FALSE,
                             projection = list(type = "natural earth")),
                legend = list(title = list(text = "Topic")))
save_html(p_map, file.path(OUT, "09_topic_map_country.html"))

# ── 4d. Countries per topic (stacked bar — share of each continent) ──────────
cont_topic <- affil_topics |>
  filter(topic != -1L) |>
  distinct(EID, Continent, topic, topic_label) |>
  count(topic_label, Continent, name = "n") |>
  group_by(topic_label) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

cont_order <- topic_info$Name[topic_info$Topic != -1L]
cont_cols_named <- c(Africa = "#e41a1c", Asia = "#377eb8", Europe = "#4daf4a",
                     `North America` = "#ff7f00", Oceania = "#984ea3",
                     `South America` = "#a65628")

p_cont <- plot_ly()
for (cont in names(cont_cols_named)) {
  d <- cont_topic[cont_topic$Continent == cont, ]
  p_cont <- add_trace(p_cont, data = d,
                      x = ~topic_label, y = ~pct,
                      type = "bar", name = cont,
                      marker = list(color = cont_cols_named[cont]),
                      hovertemplate = paste0(cont, ": %{y:.1f}%<extra></extra>"))
}
p_cont <- layout(p_cont,
                 title   = "Continent share per topic (% of affiliated organisations)",
                 barmode = "stack",
                 xaxis   = list(title = "", tickangle = -35),
                 yaxis   = list(title = "% of affiliations"),
                 legend  = list(title = list(text = "Continent")))
save_html(p_cont, file.path(OUT, "10_countries_per_topic.html"))

# ── 4e. Top African organisations per topic ───────────────────────────────────
# For each topic, rank African organisations by number of papers in that topic.
af_org_topic <- affil_topics |>
  filter(topic != -1L, Continent == "Africa",
         !is.na(clean2), nzchar(clean2)) |>
  distinct(EID, clean2, topic_label) |>
  count(topic_label, clean2, name = "n") |>
  group_by(topic_label) |>
  slice_max(order_by = n, n = 5L, with_ties = FALSE) |>   # top 5 per topic
  ungroup()

p_orgs <- plot_ly(af_org_topic,
                  x = ~n, y = ~reorder(clean2, n),
                  color = ~topic_label, colors = unname(pal),
                  type  = "bar", orientation = "h",
                  hovertemplate = "%{y}: %{x} papers<extra></extra>") |>
  layout(title   = "Top 5 African organisations per topic",
         barmode = "stack",
         xaxis   = list(title = "Number of papers"),
         yaxis   = list(title = ""),
         legend  = list(title = list(text = "Topic")),
         margin  = list(l = 200))
save_html(p_orgs, file.path(OUT, "11_orgs_per_topic.html"))

# ── 4f. Topic prevalence over time ────────────────────────────────────────────
# topics_over_time() uses the fitted topic assignments to compute how each
# topic's share of publications changes across the 2015–2024 window.
timestamps <- paper_topics |>
  left_join(paper_meta[, c("EID", "Year")], by = "EID") |>
  pull(Year)

tot <- topics_over_time(fit, timestamps = timestamps, nr_bins = 10L,
                        evolution_tuning = TRUE, global_tuning = TRUE)
p_tot <- visualize_topics_over_time(tot, normalize = TRUE)
save_html(p_tot, file.path(OUT, "12_topic_over_time.html"))

# ── 4g. Topic × continent diverging heatmap (compare_topics) ─────────────────
# compare_topics() tests whether each topic is over- or under-represented in
# each group (here: continent of affiliation).  The chi-squared contribution
# captures both direction (positive = over-represented) and magnitude.
#
# compare_topics() requires a full-length groups vector (one entry per document).
# For each paper we take the modal continent across its affiliated organisations.
# Papers with no affiliation data are labelled "Unknown".
paper_continent <- affil |>
  count(EID, Continent) |>
  group_by(EID) |>
  slice_max(n, n = 1L, with_ties = FALSE) |>
  ungroup() |>
  select(EID, Continent)

# Full-length vector aligned with docs[] / fit$clusters
paper_groups <- paper_topics |>
  left_join(paper_continent, by = "EID") |>
  mutate(Continent = ifelse(is.na(Continent), "Unknown", Continent)) |>
  pull(Continent)

comp <- compare_topics(
  fit    = fit,
  groups = paper_groups,
  method = "chi2",
  min_count = 5L
)
p_comp <- visualize_comparison(comp)
save_html(p_comp, file.path(OUT, "13_topic_comparison.html"))

# =============================================================================
# SUMMARY
# =============================================================================
message("\n=== Analysis complete ===")
message("All outputs written to: ", normalizePath(OUT))
message("\nFiles saved:")
for (f in list.files(OUT, full.names = FALSE))
  message("  ", f)

message("\nTopic summary:")
ti <- get_topic_info(fit)
print(ti[ti$Topic != -1L, c("Topic", "Count", "Name")], row.names = FALSE)
