# Small inline corpus: 6 animal sentences + 6 machine-learning sentences.
# No internet access, no model downloads — only pure-R package functions tested.
small_corpus <- c(
  "The cat sat on the mat and purred loudly",
  "Dogs love to fetch balls and run in the park",
  "A kitten chased the mouse across the floor",
  "Puppies bark and wag their tails with great excitement",
  "Feline predators groom themselves carefully after hunting prey",
  "Canines are loyal companions known for their obedience",
  "Machine learning models optimise parameters on large datasets",
  "Neural networks learn representations using gradient descent",
  "Deep learning has transformed computer vision and language tasks",
  "Attention mechanisms allow transformers to process long sequences",
  "Clustering algorithms group similar data points into clusters",
  "Dimensionality reduction maps high-dimensional data to lower spaces"
)
small_clusters <- c(rep(0L, 6L), rep(1L, 6L))

# Synthetic 12x10 embedding matrix: two well-separated Gaussian clusters
set.seed(42)
small_emb <- rbind(
  matrix(rnorm(6L * 10L, mean =  2, sd = 0.3), nrow = 6L),
  matrix(rnorm(6L * 10L, mean = -2, sd = 0.3), nrow = 6L)
)

# ---------------------------------------------------------------------------
# build_dtm
# ---------------------------------------------------------------------------

test_that("build_dtm returns a sparse dgCMatrix with correct dimensions", {
  dtm <- build_dtm(small_corpus, min_df = 1L)
  expect_s4_class(dtm, "dgCMatrix")
  expect_equal(nrow(dtm), length(small_corpus))
  expect_gt(ncol(dtm), 0L)
})

test_that("build_dtm min_df removes rare terms", {
  dtm_all  <- build_dtm(small_corpus, min_df = 1L)
  dtm_filt <- build_dtm(small_corpus, min_df = 3L)
  expect_lte(ncol(dtm_filt), ncol(dtm_all))
})

test_that("build_dtm ngram_range adds bigram columns", {
  dtm_uni <- build_dtm(small_corpus, min_df = 1L, ngram_range = c(1L, 1L))
  dtm_bi  <- build_dtm(small_corpus, min_df = 1L, ngram_range = c(1L, 2L))
  expect_gt(ncol(dtm_bi), ncol(dtm_uni))
})

# ---------------------------------------------------------------------------
# c_tf_idf
# ---------------------------------------------------------------------------

test_that("c_tf_idf returns a data frame with the expected columns", {
  dtm <- build_dtm(small_corpus, min_df = 1L)
  tt  <- c_tf_idf(dtm, cluster_ids = small_clusters, top_n = 5L)
  expect_s3_class(tt, "data.frame")
  expect_named(tt, c("topic", "rank", "term", "score"))
})

test_that("c_tf_idf includes all unique cluster ids including noise (-1)", {
  clusters_with_noise <- c(-1L, small_clusters[-1L])
  dtm <- build_dtm(small_corpus, min_df = 1L)
  tt  <- c_tf_idf(dtm, cluster_ids = clusters_with_noise, top_n = 5L)
  expect_true(-1L %in% tt$topic)
  expect_equal(sort(unique(tt$topic)), sort(unique(clusters_with_noise)))
})

test_that("c_tf_idf respects top_n per topic", {
  dtm <- build_dtm(small_corpus, min_df = 1L)
  tt  <- c_tf_idf(dtm, cluster_ids = small_clusters, top_n = 3L)
  expect_true(all(table(tt$topic) <= 3L))
})

# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

test_that("umap_reduction stores parameters and class", {
  m <- umap_reduction(n_neighbors = 10L, n_components = 3L, metric = "euclidean")
  expect_s3_class(m, "umap_reduction")
  expect_equal(m$n_neighbors, 10L)
  expect_equal(m$n_components, 3L)
  expect_equal(m$metric, "euclidean")
})

test_that("pca_reduction stores parameters and class", {
  m <- pca_reduction(n_components = 4L)
  expect_s3_class(m, "pca_reduction")
  expect_equal(m$n_components, 4L)
})

test_that("no_reduction returns the correct class", {
  m <- no_reduction()
  expect_s3_class(m, "no_reduction")
})

test_that("hdbscan_clustering stores parameters and class", {
  m <- hdbscan_clustering(min_pts = 5L, knn = "kdtree")
  expect_s3_class(m, "hdbscan_clustering")
  expect_equal(m$min_pts, 5L)
  expect_equal(m$knn, "kdtree")
})

test_that("kmeans_clustering stores k", {
  m <- kmeans_clustering(k = 3L)
  expect_s3_class(m, "kmeans_clustering")
  expect_equal(m$k, 3L)
})

test_that("agglomerative_clustering stores k and linkage", {
  m <- agglomerative_clustering(k = 4L, linkage = "complete")
  expect_s3_class(m, "agglomerative_clustering")
  expect_equal(m$k, 4L)
  expect_equal(m$linkage, "complete")
})

# ---------------------------------------------------------------------------
# dim_reduce / dim_project
# ---------------------------------------------------------------------------

test_that("dim_reduce.no_reduction passes the input matrix through unchanged", {
  m   <- no_reduction()
  out <- dim_reduce(m, small_emb)
  expect_equal(out$embedding, small_emb)
})

test_that("dim_reduce.pca_reduction outputs the correct shape", {
  m   <- pca_reduction(n_components = 3L)
  out <- dim_reduce(m, small_emb)
  expect_equal(nrow(out$embedding), nrow(small_emb))
  expect_equal(ncol(out$embedding), 3L)
})

test_that("dim_project.pca_reduction projects new data to the fitted shape", {
  m    <- pca_reduction(n_components = 3L)
  out  <- dim_reduce(m, small_emb)
  newx <- matrix(rnorm(5L * 10L), nrow = 5L)
  proj <- dim_project(out$model, newx)
  expect_equal(nrow(proj), 5L)
  expect_equal(ncol(proj), 3L)
})

# ---------------------------------------------------------------------------
# cluster_docs
# ---------------------------------------------------------------------------

test_that("cluster_docs.hdbscan_clustering returns labels and model", {
  m   <- hdbscan_clustering(min_pts = 3L)
  out <- cluster_docs(m, small_emb)
  expect_true(all(c("labels", "model") %in% names(out)))
  expect_equal(length(out$labels), nrow(small_emb))
  expect_true(all(out$labels >= -1L))
})

test_that("cluster_docs.kmeans_clustering returns exactly k distinct labels", {
  m   <- kmeans_clustering(k = 2L, nstart = 3L)
  out <- cluster_docs(m, small_emb)
  expect_equal(length(out$labels), nrow(small_emb))
  expect_equal(length(unique(out$labels)), 2L)
})

test_that("cluster_docs.agglomerative_clustering returns k distinct labels", {
  m   <- agglomerative_clustering(k = 2L)
  out <- cluster_docs(m, small_emb)
  expect_equal(length(out$labels), nrow(small_emb))
  expect_equal(length(unique(out$labels)), 2L)
})

# ---------------------------------------------------------------------------
# topic_quality
# ---------------------------------------------------------------------------

test_that("topic_quality returns a topic_quality object with all expected fields", {
  dtm <- build_dtm(small_corpus, min_df = 1L)
  tt  <- c_tf_idf(dtm, cluster_ids = small_clusters, top_n = 5L)

  mock_fit <- structure(
    list(
      embeddings  = small_emb,
      reduced     = small_emb[, 1:3],
      clusters    = small_clusters,
      docs        = small_corpus,
      topic_terms = tt
    ),
    class = "bertopic_fit"
  )

  q <- topic_quality(mock_fit, space = "original")
  expect_s3_class(q, "topic_quality")
  expect_true(all(c("cohesion", "separation", "overlap",
                    "distribution", "silhouette") %in% names(q)))
  expect_true(is.finite(q$silhouette$global))
  expect_true(is.finite(q$cohesion$global))
})

test_that("topic_quality works in reduced space", {
  dtm <- build_dtm(small_corpus, min_df = 1L)
  tt  <- c_tf_idf(dtm, cluster_ids = small_clusters, top_n = 5L)

  mock_fit <- structure(
    list(
      embeddings  = small_emb,
      reduced     = small_emb[, 1:3],
      clusters    = small_clusters,
      docs        = small_corpus,
      topic_terms = tt
    ),
    class = "bertopic_fit"
  )

  q <- topic_quality(mock_fit, space = "reduced")
  expect_s3_class(q, "topic_quality")
  expect_true(is.finite(q$silhouette$global))
})
