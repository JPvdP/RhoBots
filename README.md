# Rhobots

**BERTopic-style topic modeling in pure R — no Python, no conda, no reticulate.**

Rhobots brings the full BERTopic pipeline to R using only native R packages.
Every step — transformer inference, dimensionality reduction, density-based
clustering, c-TF-IDF representation, and interactive visualization — runs
entirely within your R session.

This package was developed for the **Data Analytics for Sustainability** course
at [Utrecht University](https://www.uu.nl).

---

## Why Rhobots?

Most R interfaces to large language models require a working Python environment,
a conda install, or `reticulate` to bridge between the two languages.  That
dependency chain is fragile, version-sensitive, and a real barrier in teaching
settings.

Rhobots removes it entirely.  Transformer weights are loaded via
[`torch`](https://torch.mlverse.org/) and [`safetensors`](https://cran.r-project.org/package=safetensors),
tokenization runs through [`tok`](https://cran.r-project.org/package=tok),
dimensionality reduction uses [`uwot`](https://cran.r-project.org/package=uwot),
and clustering uses [`dbscan`](https://cran.r-project.org/package=dbscan).
The result is a self-contained R package that "just works" in any standard R
installation.

---

## Standing on the Shoulders of Giants: BERTopic

Rhobots is a faithful R port of the BERTopic algorithm, which was designed and
built by [Maarten Grootendorst](https://github.com/MaartenGr).

> **[BERTopic](https://maartengr.github.io/BERTopic/index.html)** is one of
> the most impressive contributions to the NLP and topic modeling community in
> recent years.  It reimagined topic modeling from the ground up: rather than
> relying on bag-of-words assumptions like LDA, it leverages state-of-the-art
> sentence embeddings to capture the true semantic structure of a corpus, then
> combines UMAP, HDBSCAN, and class-based TF-IDF into an elegant four-stage
> pipeline that is simultaneously principled and practical.
>
> The Python package is extraordinarily feature-rich — supporting dozens of
> representation models (KeyBERT, GPT, Zephyr, …), guided topic modeling,
> multimodal inputs, online learning, and much more.  The documentation,
> tutorials, and the research papers behind the method are all exemplary.  If
> you do any serious NLP work, the Python package is worth your time.
>
> Rhobots would not exist without that foundation.  We owe a great debt to
> Maarten and every contributor to the BERTopic project.

---

## Installation

```r
# install.packages("pak")
pak::pak("your-github-username/Rhobots")
```

Or clone this repository and install locally:

```r
devtools::install("path/to/Rhobots")
```

---

## Ready-to-Use Models for Scientometrics and Innometrics

Rhobots ships with `load_hf_bert()`, which can load any BERT-family model
directly from the [HuggingFace Hub](https://huggingface.co/models).  The
following models have been tested and work out of the box for scientometric
and innometric analyses (patent texts, scientific abstracts, policy documents):

| Model | Best for |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Fast general-purpose baseline |
| `allenai/scibert_scivocab_uncased` | Scientific publications |
| `pritamdeka/S-Scibert-snli-multinli-stsb` | Scientific sentence similarity |
| `NetworkIsLife/SciBert_Cased_DAFS` | Domain-adapted scientometrics |
| `anferico/bert-for-patents` | Patent texts |
| `ProsusAI/finbert` | Financial / economic texts |

```r
library(Rhobots)

# Load a domain-specific encoder
enc <- load_hf_bert("pritamdeka/S-Scibert-snli-multinli-stsb")

# Embed and model
fit <- fit_bertopic(enc, docs = my_abstracts)
print_topics(fit)
```

---

## Working with Other Models

Not every model on HuggingFace uses the standard BERT weight layout.  Rhobots
provides lower-level helpers so you can adapt them:

| Function | Purpose |
|---|---|
| `load_hf_bert(repo_id, weights_path = ...)` | Load any BERT-family model; supply a local `weights_path` when the Hub file needs conversion |
| `load_bert_weights(model, weights_path)` | Load weights manually into a constructed model |
| `make_wordpiece_tokenizer(vocab_file)` | Build a WordPiece tokenizer from a raw `vocab.txt` when the repo lacks `tokenizer.json` |
| `mean_pool(hidden, attention_mask)` | Mean-pool token hidden states into sentence vectors (the building block for custom pooling strategies) |

A typical adaptation workflow for a model with only `pytorch_model.bin`:

```r
# Convert .bin to .safetensors via the HuggingFace web Space, then:
enc <- load_hf_bert(
  "some-org/some-model",
  weights_path = "/local/path/to/model.safetensors"
)
```

---

## Core Workflow

```r
library(Rhobots)

# 1. Load an encoder
enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")

# 2. Compute embeddings (cached to disk so you only pay once)
emb <- embed_texts_cached(enc, docs, cache_file = "embeddings.rds")

# 3. Fit a topic model
fit <- fit_bertopic(
  docs                  = docs,
  embeddings            = emb,
  hdbscan_min_pts       = 5,
  ngram_range           = c(1L, 2L),
  extra_stopwords       = c("study", "paper", "result"),
  reduce_frequent_words = TRUE
)

# 4. Inspect results
print(fit)
print_topics(fit)
get_topic_info(fit)
get_representative_docs(fit, topic = 0)
find_topics(fit, "climate")
```

---

## Full Feature Overview

### Embedding

| Function | Description |
|---|---|
| `load_hf_bert()` | Load any BERT-family model from the HuggingFace Hub |
| `embed_texts()` | Embed a character vector using a loaded encoder |
| `embed_texts_cached()` | Embed with on-disk caching; subsequent calls load instantly |
| `save_embeddings()` / `load_embeddings()` | Persist and reload embedding matrices |

### Topic Modeling

| Function | Description |
|---|---|
| `fit_bertopic()` | Four-stage pipeline: embed → reduce → cluster → c-TF-IDF |
| `print_topics()` | Console summary of discovered topics |

### Pluggable Models

Swap in alternative dimensionality-reduction or clustering algorithms by
passing a model object to `fit_bertopic()`:

```r
# PCA + k-means (fast, deterministic)
fit <- fit_bertopic(
  docs = docs, embeddings = emb,
  dim_reduction_model = pca_reduction(n_components = 10L),
  cluster_model       = kmeans_clustering(k = 8L)
)

# UMAP + agglomerative clustering
fit <- fit_bertopic(
  docs = docs, embeddings = emb,
  cluster_model = agglomerative_clustering(k = 6L, linkage = "ward.D2")
)
```

**Dimensionality reduction:** `umap_reduction()`, `pca_reduction()`, `no_reduction()`

**Clustering:** `hdbscan_clustering()`, `kmeans_clustering()`, `agglomerative_clustering()`

### Accessors (mirroring the Python BERTopic API)

| Function | Description |
|---|---|
| `get_topics()` | All topic-term representations as a named list |
| `get_topic()` | Term-score table for a single topic |
| `get_topic_info()` | Topic-level metadata (count, label, top terms) |
| `get_document_info()` | Document-level topic assignments |
| `get_representative_docs()` | Three documents closest to each topic centroid |
| `find_topics()` | Rank topics by similarity to a search term |

### Out-of-Sample Prediction

```r
result <- transform_bertopic(fit, new_docs, encoder = enc)
# result$topics         — assigned topic IDs
# result$probabilities  — cosine similarity to assigned centroid
```

### Post-Fit Topic Manipulation

| Function | Description |
|---|---|
| `reduce_topics(fit, nr_topics)` | Iteratively merge the most similar topics |
| `reduce_outliers()` | Reassign noise documents to the nearest real topic |
| `merge_topics()` | Manually collapse a set of topics into one |
| `apply_mmr()` | Refine topic terms with Maximal Marginal Relevance |

### Dynamic Topic Modeling

```r
# Single model, topics tracked across timestamps
tot <- topics_over_time(fit, timestamps = df$Year)
visualize_topics_over_time(tot)

# Independent model per period, transitions aligned by centroid similarity
flow <- fit_topics_over_time(enc, docs, timestamps = df$Year)
visualize_topic_flow(flow, color_by = "status")
```

### Visualization (requires `plotly`)

| Function | Description |
|---|---|
| `visualize_topics()` | 2-D UMAP scatter plot coloured by topic |
| `visualize_barchart()` | Grid of c-TF-IDF bar charts, one panel per topic |
| `visualize_topics_over_time()` | Line chart of topic frequency over time |
| `visualize_topic_flow()` | Sankey diagram of topic transitions across periods |

---

## Stopwords

```r
# Built-in English list
get_stopwords("english")

# Load domain-specific stopwords from a file, CSV, or character vector
sw <- load_stopwords("domain_stopwords.txt")

fit <- fit_bertopic(docs = docs, embeddings = emb,
                    extra_stopwords = sw)
```

---

## Dependencies

All dependencies are CRAN or R-universe packages — no Python required.

| Package | Role |
|---|---|
| `torch` | Transformer forward passes |
| `safetensors` | Safe, fast weight loading |
| `hfhub` | HuggingFace Hub downloads |
| `tok` | Fast tokenizer (HuggingFace format) |
| `wordpiece` | Fallback WordPiece tokenizer |
| `uwot` | UMAP dimensionality reduction |
| `dbscan` | HDBSCAN clustering |
| `Matrix` | Sparse document-term matrices |
| `plotly` *(suggested)* | Interactive visualizations |

---

## Acknowledgements

This package was built for the **Data Analytics for Sustainability** course at
[Utrecht University](https://www.uu.nl), Faculty of Geosciences.

The algorithm and API design are based on, and inspired by,
[BERTopic](https://maartengr.github.io/BERTopic/index.html) by
[Maarten Grootendorst](https://github.com/MaartenGr) and its many contributors.
Please cite the original work if you publish results obtained with this package:

> Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
> TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
> <https://arxiv.org/abs/2203.05794>

---

## License

MIT © J.P.G. van der Pol, Utrecht University
