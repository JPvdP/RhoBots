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
