# =============================================================================
# africa_analysis.R
#
# End-to-end Rhobots demonstration using African research publications.
#
# PURPOSE
# -------
# This script walks through a complete topic-modeling workflow on a real
# corpus of 3,328 research abstracts that include at least one African
# co-author.  It is intended as a teaching example: every major step
# is annotated with a WHY, not just a WHAT.
#
# LEARNING GOALS
# --------------
#   1. Understand what it means to "embed" text into a vector space.
#   2. See how a parameter sweep helps you choose model settings.
#   3. Fit a topic model and interpret the results.
#   4. Use topic assignments to answer research questions about geography,
#      institutional collaboration, and temporal trends.
#
# DATA
# ----
#   Africa_abstracts.csv                          — 3,328 abstracts (EID + Abstract)
#   UU_scopus_country_clean_lonlat_continent.rdata — affiliation/location data
#     (one row per affiliation per paper: EID, clean2 org name, main_country,
#      Continent, Year, lon, lat, iso2)
#
# NOTE ON DATA STRUCTURE
# ----------------------
# A single paper can have multiple affiliations (e.g., co-authors from three
# different universities in three countries).  This means the affiliation table
# has MANY ROWS per paper.  When counting papers, we always use:
#
#   distinct(EID, <grouping variable>)  → then count()
#
# rather than just count(), to avoid double-counting a paper once for each
# of its affiliations.
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

# ── Helper: save a plotly figure as a self-contained HTML file ────────────────
# We use saveWidget() from htmlwidgets.  selfcontained = TRUE embeds all
# JavaScript inside the .html file so it opens without an internet connection.
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

# The abstracts table is flat: one row per paper.
abstracts <- read.csv("Africa_abstracts.csv", stringsAsFactors = FALSE)

# The affiliation table has ONE ROW PER AFFILIATION PER PAPER.
# A paper with 5 co-authors from 5 institutions produces 5 rows, all sharing
# the same EID (Elsevier document identifier).
load("UU_scopus_country_clean_lonlat_continent.rdata")
affil <- UU_scopus_country_clean

# Restrict affiliations to the papers in our abstract set.
# (The affiliation file may contain papers outside the Africa subset.)
affil <- affil[affil$EID %in% abstracts$EID, ]

# Compute one summary row per paper:
#   Year       — publication year
#   has_africa — TRUE if ≥1 affiliation is in Africa (should always be TRUE here)
#   n_africa   — how many African affiliations the paper has
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
# Before modeling, always explore the data.  This section asks:
#   - Which countries produce the most Africa-affiliated research?
#   - Which institutions are most prolific?
#   - Has African publication output grown over time?
#   - Which world regions do African researchers collaborate with most?
message("\n=== Part 1: Descriptive statistics ===")

# ── 1a. Top African countries by number of affiliated papers ─────────────────
# WHY distinct(EID, main_country)?
#   A paper with two South African universities would appear TWICE in the
#   affiliation table for "South Africa".  distinct() keeps it once per
#   (paper, country) pair so we count papers, not affiliation rows.
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
# clean2 = a standardised organisation name (raw Scopus names are inconsistent;
# "Univ Cape Town", "University of Cape Town", "UCT" all map to the same clean2).
af_org <- affil |>
  filter(Continent == "Africa", !is.na(clean2), nzchar(clean2)) |>
  distinct(EID, clean2) |>     # again: one (paper, org) pair, not one row per affil
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
# For each (year, continent) pair we count how many affiliation rows there are.
# Note: this counts affiliated ORGANISATIONS per year, not papers, because a
# paper with five European affiliations contributes 5 to Europe's count.
# That is intentional — it captures the volume of international engagement.
trend <- affil |>
  distinct(EID, Continent, Year) |>    # (paper, continent, year) triples
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
# QUESTION: which world regions collaborate most with each African country?
# METHOD:
#   For each paper that has ≥1 African affiliation, count how many times each
#   African country co-appears with each non-African continent.
#   This is a cross-tabulation of (african_country, partner_continent) pairs.
#
# We use an inner_join to link:
#   - Left side:  ALL affiliations (any continent) for papers with an African co-author
#   - Right side: the AFRICAN affiliations for those same papers
# After joining, each row represents one (African country, partner continent) link
# for a single paper.  We then filter to partner_continent != "Africa" to get
# only cross-continental links.
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
  filter(n >= 5)                            # drop very rare pairs (noise)

# Keep only the top 15 African countries (by total paper count) for readability.
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
# WHAT IS EMBEDDING?
# ------------------
# A transformer encoder reads each abstract and produces a dense numeric
# vector — typically 384 to 768 numbers — that captures the semantic meaning
# of the text.  Abstracts about similar topics land close together in this
# high-dimensional space; abstracts about different topics land far apart.
# This is what makes topic modeling with BERTopic semantically coherent:
# it clusters documents by meaning, not by shared vocabulary.
#
# WHY all-MiniLM-L6-v2?
#   It is fast (6 transformer layers, 384 dimensions) and produces competitive
#   embeddings for general English text.  For a corpus of scientific abstracts
#   a domain-specific model (SPECTER2, SciBERT) would give higher topic
#   coherence at the cost of slower inference.  MiniLM-L6 is a good starting
#   point before investing in a heavier model.
message("\n=== Part 2: Embedding + parameter sweep ===")

enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")

docs <- abstracts$Abstract

# embed_texts_cached() runs the encoder on the first call and saves the matrix
# to disk as an .rds file.  Every subsequent call loads the cached version in
# a fraction of a second without re-running the encoder.
# WHY cache?  Embedding 3,328 abstracts takes ~2–5 minutes on a CPU.  If you
# want to try different clustering settings, you do not want to re-embed every
# time.  Caching decouples the slow encoding step from the fast modeling step.
message("Embedding ", length(docs), " abstracts (cached after first run)...")
emb <- embed_texts_cached(enc, docs,
                           cache_file = file.path(OUT, "embeddings_cache.rds"),
                           normalize  = TRUE,   # L2-normalise so cosine = dot product
                           verbose    = TRUE)
message(sprintf("Embedding matrix: %d × %d", nrow(emb), ncol(emb)))

# ── Parameter sweep ──────────────────────────────────────────────────────────
# Before fitting a single model it is good practice to evaluate multiple
# combinations of hyperparameters.  sweep_topics() does this efficiently:
# the encoder runs ONCE (embeddings are already computed), and only the cheap
# UMAP + HDBSCAN steps are repeated for each combination.
#
# The three parameters:
#
#   n_neighbors  (UMAP)
#     Controls the local vs. global balance of the UMAP manifold.
#     SMALL n_neighbors → UMAP attends to very local structure (tight, fine-grained
#       clusters, but may miss larger-scale organisation).
#     LARGE n_neighbors → UMAP attends to more global structure (broader, smoother
#       manifold, topics may merge).
#     Typical range: 10–50.
#
#   n_components (UMAP)
#     The number of dimensions in the reduced space fed into HDBSCAN.
#     More dimensions → richer representation but harder for HDBSCAN to find
#     density peaks (curse of dimensionality).
#     Less dimensions → faster, but may lose fine-grained topic separation.
#     Typical range: 5–15.
#
#   min_pts (HDBSCAN)
#     Minimum number of documents for a group to be considered a "cluster"
#     rather than noise.  This is the most important tuning parameter.
#     SMALL min_pts → many small topics; documents near the edge of clusters
#       are still included; less noise.
#     LARGE min_pts → fewer, larger topics; documents that don't fit
#       a big group are labelled noise (-1).
#     Typical range: 5–30 depending on corpus size.
#
# For each combination the sweep reports five quality metrics:
#   silhouette — how well-separated topics are (higher = better)
#   cohesion   — mean cosine similarity within topics (higher = better)
#   separation — mean cosine distance between topic centroids (lower = more distinct)
#   jaccard    — vocabulary overlap between topics (lower = more distinct)
#   noise %    — share of documents left unassigned (lower is usually better)
message("Running parameter sweep (this takes a few minutes)...")

sw <- sweep_topics(
  docs         = docs,
  embeddings   = emb,
  n_neighbors  = c(10L, 15L, 20L),
  n_components = c(3L, 5L),      # keep low — HDBSCAN struggles above 5 dims
  min_pts      = c(5L, 10L, 15L),
  min_topics   = 8L,             # must find at least 8 topics; optimise silhouette under this constraint
  ngram_range    = c(1L, 2L),    # use both unigrams and bigrams in topic terms
  quality_top_n  = 10L,          # evaluate using the top 10 terms per topic
  quality_sample = 1000L,        # silhouette computed on a sample for speed
  seed    = 42L,
  verbose = TRUE
)

# Print the sweep summary table: one row per parameter combination.
print(sw)

# visualize_sweep() produces an interactive heatmap where:
#   rows  = parameter combinations (18 in total)
#   cols  = quality metrics (all normalised 0–1 within column, green = best)
#   star  = the combination with the highest silhouette score (sw$best)
# WHAT TO LOOK FOR: rows that are consistently green across all columns.
# A combination that scores well on silhouette but poorly on noise% may be
# producing many small tight clusters at the expense of leaving half the corpus
# unassigned — that may or may not be what you want.
p_sweep <- visualize_sweep(sw)
save_html(p_sweep, file.path(OUT, "05_sweep.html"))

# sw$best = the parameter combination with the highest silhouette score.
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
# Now that we have good hyperparameters, we fit the full model.
#
# fit_bertopic() runs the complete four-stage pipeline:
#   1. (Skip — embeddings already computed and passed in via `embeddings`)
#   2. UMAP: reduce the 384-D embeddings to n_components dimensions.
#      A separate 2-D projection is always computed for visualisation.
#   3. HDBSCAN: find clusters in the reduced space.
#   4. c-TF-IDF: for each cluster, compute which terms are most characteristic
#      relative to the rest of the corpus.
#
# EXTRA STOPWORDS
# ---------------
# Generic academic phrases like "study", "results", "analysis" are so common
# across all topics that they score highly on TF (term frequency) but carry no
# discriminative information.  Adding them to extra_stopwords removes them from
# the c-TF-IDF vocabulary so the topic terms focus on content words.
#
# REDUCE_FREQUENT_WORDS = TRUE
# ----------------------------
# Even after stopword removal, some terms may completely dominate a single topic.
# This option applies a square-root dampening to the class TF before multiplying
# by IDF, which compresses very high frequencies and gives more balanced term lists.
message("\n=== Part 3: Fitting the topic model ===")

fit <- fit_bertopic(
  docs                  = docs,
  embeddings            = emb,
  umap_n_neighbors      = best$n_neighbors,
  umap_n_components     = best$n_components,
  hdbscan_min_pts       = best$min_pts,
  ngram_range           = c(1L, 2L),   # include bigrams like "south africa", "climate change"
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

# print() shows the number of topics found, the noise count, and the top terms.
print(fit)
# print_topics() shows a compact table: topic ID, size, top terms.
print_topics(fit)

# ── Label topics with Ollama BEFORE saving any visualisations ────────────────
# All visualisations (scatter, barchart, quality, map, …) pull labels from
# fit$topic_labels.  We run the LLM labeling here — before saving anything —
# so every output file shows human-readable names instead of c-TF-IDF slugs.
#
# label_topics_llm() sends each topic's top terms + 3 representative abstracts
# to a local LLM and asks for a concise 3–5-word description.  The LLM label
# replaces the auto-generated slug inside fit$topic_labels.
#
# Requirements (one-time setup):
#   1. Install Ollama from https://ollama.com
#   2. In a terminal: ollama serve
#   3. In a terminal: ollama pull llama3.2  (downloads ~2 GB)
#
# If Ollama is unavailable, comment out the two lines below — the rest of the
# script falls back to the automatic c-TF-IDF slugs.
message("Labeling topics with Ollama llama3.2 (ensure `ollama serve` is running)...")
fit <- label_topics_llm(fit, provider = "ollama", model = "llama3.2",
                        top_n_terms = 10L, n_representative_docs = 3L)

# Inspect the human-readable labels before producing any output.
topic_info <- get_topic_info(fit)
print(topic_info[, c("Topic", "Count", "Name")])

# ── Topic quality assessment ──────────────────────────────────────────────────
# topic_quality() computes four metrics for the fitted model:
#   cohesion    — how tightly each topic's documents cluster around their centroid
#   separation  — how distinct topics are from one another
#   overlap     — vocabulary overlap between topics (Jaccard similarity of term sets)
#   distribution — balance of topic sizes (entropy, coefficient of variation)
# A silhouette score in the full embedding space is also reported.
q <- topic_quality(fit, top_n = 10L)
print(q)

p_quality <- visualize_quality(q, fit)
save_html(p_quality, file.path(OUT, "08_topic_quality.html"))

# ── UMAP document scatter (coloured by topic) ─────────────────────────────────
# visualize_topics() uses the 2-D UMAP layout (always computed by fit_bertopic,
# regardless of the clustering dimensionality).  Each point is one abstract.
# Colour = topic assignment.  Grey points = noise (topic -1).
# WHAT TO LOOK FOR:
#   - Tight, well-separated clouds of the same colour → coherent topics.
#   - Overlapping clouds → topics that may need to be merged.
#   - Many grey points → either min_pts is too large, or those abstracts
#     are genuinely too diverse to cluster.
p_scatter <- visualize_topics(fit)
save_html(p_scatter, file.path(OUT, "06_topics_scatter.html"))

# ── c-TF-IDF bar charts ───────────────────────────────────────────────────────
# visualize_barchart() shows the top 8 terms per topic and their c-TF-IDF score.
# A high c-TF-IDF means the term appears frequently in this topic but rarely
# across the rest of the corpus — it is characteristic, not just common.
p_bars <- visualize_barchart(fit, top_n = 8L)
save_html(p_bars, file.path(OUT, "07_topics_barchart.html"))

message("Part 3 done — topic model fitted and labeled.")

# =============================================================================
# PART 4 — ANALYSIS THROUGH TOPIC LENS
# =============================================================================
# The topic assignments let us ask research questions that go beyond "what is
# this corpus about?" to "who researches what, where, and how is that changing?"
message("\n=== Part 4: Analysis through topic lens ===")

# topic_info was already printed above; re-fetch for use in the rest of Part 4.

# ── 4b. Attach topic assignments to the affiliation table ─────────────────────
# fit$clusters is a vector of length = number of documents (nrow(abstracts)).
# Position i corresponds to abstracts$Abstract[i] (the same document order we
# passed to fit_bertopic).  Topic -1 means HDBSCAN labelled the document as noise.
paper_topics <- data.frame(
  EID   = abstracts$EID,
  topic = fit$clusters,           # -1 = noise; 0, 1, 2, … = topic IDs
  stringsAsFactors = FALSE
)

# Helper: format a raw topic label for display.
# LLM labels look like "5_Sustainable Agriculture and Food Security".
# Auto-generated labels look like "5_agriculture_soil_crop".
# In both cases we:
#   1. Strip the numeric ID prefix (e.g. "5_")
#   2. Replace underscores with spaces (for auto-generated labels)
#   3. Truncate to 30 characters to keep axis labels readable
.fmt <- function(raw) {
  text <- sub("^-?[0-9]+_", "", raw)   # strip "5_" or "-1_"
  text <- gsub("_", " ", text)          # "agriculture_soil" → "agriculture soil"
  text <- trimws(text)
  if (nchar(text) > 30L) paste0(substr(text, 1L, 29L), "…") else text
}

paper_topics$topic_label <- vapply(paper_topics$topic, function(t) {
  if (t == -1L) return("Noise")
  lbl <- fit$topic_labels[[as.character(t)]]
  if (is.null(lbl) || !nzchar(lbl)) paste0("Topic ", t) else .fmt(lbl)
}, character(1))

# Join topic assignments back to the affiliation table.
# After this join, every affiliation row has a "topic" column indicating
# which topic the paper it belongs to was assigned to.
affil_topics <- affil |>
  left_join(paper_topics, by = "EID")

# ── 4b2. 2-D topic bubble map ────────────────────────────────────────────────
# Instead of showing individual documents, this chart shows ONE BUBBLE PER TOPIC.
# Position = centroid of the topic's documents in the 2-D UMAP layout.
# Size     = number of documents in the topic (log-scaled for readability).
# Hover    = top c-TF-IDF terms.
# This gives a compact overview of where topics sit relative to each other
# in the semantic space, and which topics are large vs. niche.
topics_nonnoise <- sort(setdiff(unique(fit$clusters), -1L))
coords2d        <- fit$layout2d   # n × 2 matrix of 2-D UMAP coordinates

topic_map_df <- data.frame(
  topic = topics_nonnoise,
  # Centroid = average of the 2-D coordinates of all documents in the topic.
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

# Add the top 8 terms per topic for hover text.
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
    size    = ~sqrt(n) * 3,   # sqrt-scaling prevents very large topics from dominating
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
# QUESTION: Is research about certain topics concentrated in certain countries?
# METHOD: For each country, find the modal topic (the topic that appears most
# often across its affiliated papers).  Colour the country on the world map by
# that dominant topic.
#
# slice_max() keeps the single row with the largest n per country (the most
# common topic).  with_ties = FALSE breaks ties by taking the first row.
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

# ── 4d. Continent share per topic (stacked bar) ───────────────────────────────
# QUESTION: Are some topics more internationally collaborative than others?
# Are some topics primarily driven by African institutions or by a specific
# world region?
#
# For each (topic, continent) pair we count how many DISTINCT (paper, continent)
# combinations exist, then express it as a percentage within the topic.
# This normalises for topic size so we can compare topic "flavours" fairly.
cont_topic <- affil_topics |>
  filter(topic != -1L) |>
  distinct(EID, Continent, topic, topic_label) |>
  count(topic_label, Continent, name = "n") |>
  group_by(topic_label) |>
  mutate(pct = n / sum(n) * 100) |>    # convert to percentage within topic
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
# QUESTION: Which African universities and research institutes lead each topic?
# This helps identify institutional specialisation and potential collaboration
# partners for specific research areas.
af_org_topic <- affil_topics |>
  filter(topic != -1L, Continent == "Africa",
         !is.na(clean2), nzchar(clean2)) |>
  distinct(EID, clean2, topic_label) |>    # one (paper, org, topic) triple
  count(topic_label, clean2, name = "n") |>
  group_by(topic_label) |>
  slice_max(order_by = n, n = 5L, with_ties = FALSE) |>   # top 5 orgs per topic
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
# QUESTION: How has the share of each topic changed from 2015 to 2024?
# Is any topic growing rapidly? Has any topic declined?
#
# topics_over_time() takes the fitted topic assignments and a timestamp per
# document.  It bins the timestamps (here: by year, 10 bins for 10 years) and
# for each (topic, year) combination computes the share of that year's documents
# assigned to that topic.
#
# evolution_tuning = TRUE smooths the term representation by averaging with the
# previous time period's representation (stabilises sparse time bins).
# global_tuning    = TRUE blends with the global topic representation.
#
# NOTE: timestamps must be aligned with docs[].  We left_join paper_meta
# (which has one row per EID) to get the Year for each document.
timestamps <- paper_topics |>
  left_join(paper_meta[, c("EID", "Year")], by = "EID") |>
  pull(Year)

tot <- topics_over_time(fit, timestamps = timestamps, nr_bins = 10L,
                        evolution_tuning = TRUE, global_tuning = TRUE)
p_tot <- visualize_topics_over_time(tot, normalize = TRUE)
save_html(p_tot, file.path(OUT, "12_topic_over_time.html"))

# ── 4g. Topic × continent diverging heatmap ───────────────────────────────────
# QUESTION: Which continents are over- or under-represented in each topic,
# relative to what we would expect if topics were distributed uniformly?
#
# compare_topics() tests for statistical over/under-representation using a
# signed chi-squared contribution:
#
#   chi2_contribution(topic, group) = sign(O - E) × sqrt((O - E)² / E)
#
# where O = observed count and E = expected count under independence.
# Positive values = over-represented; negative values = under-represented.
# visualize_comparison() renders this as a blue/white/red diverging heatmap.
#
# IMPORTANT: compare_topics() needs a full-length groups vector — one entry
# per document in the same order as docs[].  We use the modal continent per
# paper (the continent that appears most often among the paper's affiliations).
# Papers with no affiliation data get the label "Unknown".
paper_continent <- affil |>
  count(EID, Continent) |>
  group_by(EID) |>
  slice_max(n, n = 1L, with_ties = FALSE) |>
  ungroup() |>
  select(EID, Continent)

# Build the full-length groups vector.  left_join ensures we get one row per
# document (in the same order as abstracts$EID = docs[]).
paper_groups <- paper_topics |>
  left_join(paper_continent, by = "EID") |>
  mutate(Continent = ifelse(is.na(Continent), "Unknown", Continent)) |>
  pull(Continent)

comp <- compare_topics(
  fit    = fit,
  groups = paper_groups,
  method = "chi2",    # signed chi-squared contribution
  min_count = 5L      # ignore (topic, continent) cells with fewer than 5 papers
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
