# Rhobots (development version)

## Memory

* `embed_texts()` no longer lets finished batches pile up in memory on large
  corpora.  Torch tensors are invisible to R's garbage collector, so dead
  batch tensors were only freed sporadically; they are now released after
  every batch (new argument `gc_every`, default 1; also empties the CUDA
  cache).  On a 3,000-document test this cut peak memory roughly in half
  with no slowdown.
* `chunk_strategy = "mean"` / `"first"` now tokenizes and chunks the corpus in
  blocks of documents instead of all at once, so memory no longer grows with
  corpus size.  Output is unchanged.

# Rhobots 0.1.10

## CRAN policy fixes

* DESCRIPTION rewritten to spell out all acronyms in full (Uniform Manifold
  Approximation and Projection, Hierarchical Density-Based Spatial Clustering
  of Applications with Noise, term frequency-inverse document frequency,
  part-of-speech) and remove author-name citations, eliminating all
  spell-checker NOTEs in the automated CRAN incoming feasibility check.
* nanoflann.hpp moved from inst/include/ to src/ (internal use only);
  removes the 'cc not on path' NOTE on Windows by taking it out of the
  public header path that R's compiled-code checker inspects.

# Rhobots 0.1.9

## CRAN policy fixes

* All `\dontrun{}` example blocks replaced with `\donttest{}` (examples require
  a model download and cannot run in < 5 s; `\dontrun{}` retained only for
  `rhobots_install()` which installs the torch backend).
* Software and package names in DESCRIPTION now consistently wrapped in single
  quotes per CRAN policy.

# Rhobots 0.1.8

## Bug fixes

* `load_specter2()` now correctly injects adapter weights into all transformer
  layers.  The previous code used a nested replacement chain
  (`enc$model$encoder$layer[[i]]$adapter <- adp`) which R desugars into a
  `[[<-` write-back that can reinitialise the submodule; weights from earlier
  layers were silently overwritten by each successive iteration.  Fixed by
  pre-fetching direct layer references (`lapply(seq_len(n_layers), \(i) ...)`).
  SPECTER2 embeddings now match the Python reference (cosine similarity > 0.9999).

* `load_hf_bert()` / `load_specter2()` now correctly read safetensors files
  named `pytorch_model.safetensors` in addition to `model.safetensors`.

* `weight_loading.R` now detects ZIP-archive `.bin` files (PyTorch 1.6+ format,
  magic bytes `PK`) and emits a clear error message with conversion instructions,
  in addition to the pre-existing guard for legacy pickle format (`0x80`).

* `torch::torch_load()` call updated from deprecated `map_location` argument to
  `device` (renamed in torch 0.16.3).

* Default repositories for `load_specter2()` updated to
  `NetworkIsLife/specter2_base` and `NetworkIsLife/specter2`, which host
  safetensors versions of the original `allenai/specter2_base` and
  `allenai/specter2` (the AllenAI repos only provide `.bin` files that R torch
  cannot read directly).

# Rhobots 0.1.7

## New features

* `hdbscan_clustering()` gains `allow_single_cluster` (default `FALSE`,
  matching Python `hdbscan`'s default).  When `FALSE`, EOM never returns a
  single root cluster when sub-structure is present in the condensed tree.
  This fixes cases where diffuse embedding spaces (continuous sentiment data,
  overlapping topics) collapsed to 1 cluster.  On the validation suite,
  datasets that previously returned 1 cluster (Emotion, Yelp Reviews, IMDB,
  Rotten Tomatoes) now match Python's cluster counts within ±1.

* `topic_quality()` gains a `space` argument (`"original"` / `"reduced"`).
  When `"reduced"`, cohesion, separation, and silhouette are measured in the
  UMAP-reduced space rather than the full encoder embedding.  This makes
  quality metrics much more sensitive to different UMAP/HDBSCAN parameter
  choices and is the recommended setting when using `sweep_topics()`.

* `sweep_topics()` gains a corresponding `quality_space` argument that is
  forwarded to `topic_quality()` (default `"reduced"`).

# Rhobots 0.1.6

## New features

* `hdbscan_clustering()` gains a `knn` argument that selects the tree data
  structure used during Borůvka MST construction:
  - `"balltree"` (default) — dual-tree Borůvka with a Ball-tree index
    (bounding hyperspheres; effective pruning in ≥3-D).
  - `"kdtree"` — dual-tree Borůvka with a KD-tree index (axis-aligned
    bounding boxes; faster to build, pruning degrades above three dimensions).
  - `"adaptive"` — pre-computed kNN graph via `dbscan::kNN()`, growing `k`
    until the MST is fully connected (guaranteed connectivity, no cap).
  - `"fixed"` — same as `"adaptive"` but caps `k` at 200 (faster).

  The `"balltree"` default replaces the previous fixed-kNN approach and
  matches the algorithm used by the Python `hdbscan` package's
  `boruvka_balltree` mode.  Validation on two scientific-abstract corpora
  (n = 3,328 and n = 9,035) confirms Adjusted Rand Index > 0.97 and
  Normalised Mutual Information > 0.98 against the Python reference.

# Rhobots 0.1.5

* Initial CRAN submission release.
* Auto-install of `gutenbergr` and `plotly` in `rhobots_demo()`.
* Fixed `useDynLib` being overwritten by roxygen2 regeneration.
* Fixed CUDA error handling and demo topic quality.
* Added `rhobots_demo()` — Gutenberg pipeline walkthrough.
