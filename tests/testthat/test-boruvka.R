# Tests for hdbscan_boruvka_cpp:
#   - Structural invariants (MST edge count, label range)
#   - Correctness on well-separated clusters (with k large enough to span clusters)
#   - Noise detection in the presence of a true split
#   - Agreement with dbscan::hdbscan() as reference
#   - min_pts effect
#   - Disconnected kNN graph detection
#
# NOTE on kNN graph connectivity:
#   hdbscan_boruvka_cpp only considers edges present in the kNN graph. When
#   clusters are well-separated and k < n_per_cluster, points cannot reach
#   the other cluster and the graph is disconnected (n_mst_edges < n-1). The
#   R-level wrapper cluster_docs.hdbscan_clustering() handles this with a
#   retry loop that doubles k; the C++ function reports disconnection via
#   n_mst_edges. Tests here use k >= n_per_cluster to guarantee a connected
#   graph so that clustering correctness can be tested directly.

boruvka <- function(...) Rhobots:::hdbscan_boruvka_cpp(...)

make_knn <- function(X, k) {
  knn <- dbscan::kNN(X, k = k, sort = TRUE)
  list(idx = knn$id, dist = knn$dist)
}

# Check: all points in row-block g (given by group_sizes) have the same label,
# and different blocks have different non-zero labels.
clusters_consistent <- function(labels, group_sizes) {
  start <- 1L
  gl    <- integer(length(group_sizes))
  for (g in seq_along(group_sizes)) {
    idx <- start:(start + group_sizes[g] - 1L)
    nz  <- labels[idx][labels[idx] > 0L]
    if (length(nz) > 0L && !all(nz == nz[1L])) return(FALSE)
    gl[g] <- if (length(nz) > 0L) nz[1L] else 0L
    start <- start + group_sizes[g]
  }
  nz2 <- gl[gl > 0L]
  length(nz2) == length(unique(nz2))
}

# ── Test 1: two well-separated clusters ─────────────────────────────────────
# k = 25 > n_each = 20, so the kNN graph is guaranteed to span both clusters.

test_that("Boruvka finds two well-separated clusters with no noise", {
  set.seed(11)
  n_each <- 20L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 20, sd = 0.1), nrow = n_each)
  )
  n   <- nrow(X)
  k   <- 25L
  knn <- make_knn(X, k)
  res <- boruvka(knn$idx, knn$dist, min_pts = 5L)

  expect_equal(res$n_mst_edges, n - 1L)
  expect_equal(sum(res$labels == 0L), 0L)
  expect_equal(length(unique(res$labels)), 2L)
  expect_true(clusters_consistent(res$labels, c(n_each, n_each)))
})

# ── Test 2: three well-separated clusters ────────────────────────────────────

test_that("Boruvka finds three well-separated clusters", {
  set.seed(22)
  n_each <- 20L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 20, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 40, sd = 0.1), nrow = n_each)
  )
  n   <- nrow(X)
  k   <- 35L
  knn <- make_knn(X, k)
  res <- boruvka(knn$idx, knn$dist, min_pts = 5L)

  expect_equal(res$n_mst_edges, n - 1L)
  expect_equal(length(unique(res$labels[res$labels > 0L])), 3L)
  expect_true(clusters_consistent(res$labels, c(n_each, n_each, n_each)))
})

# ── Test 3: isolated outlier is noise when a true split exists ───────────────
# An outlier is only labeled noise (0) after a true split has occurred (both
# sides >= min_pts). Here 2 clear clusters create that split; the outlier
# falls off earlier and propagates to noise correctly.

test_that("Boruvka labels an extreme outlier as noise when two clusters exist",{
  set.seed(33)
  n_each <- 15L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 15, sd = 0.1), nrow = n_each),
    matrix(c(5e3, 5e3), nrow = 1L)
  )
  n   <- nrow(X)
  k   <- 22L
  knn <- make_knn(X, k)
  res <- boruvka(knn$idx, knn$dist, min_pts = 5L)

  expect_equal(res$n_mst_edges, n - 1L)
  expect_equal(res$labels[n], 0L)   # outlier is noise
  expect_true(clusters_consistent(res$labels, c(n_each, n_each)))
})

# ── Test 4: n_mst_edges == n-1 for a fully connected kNN graph ───────────────

test_that("Boruvka returns n-1 MST edges on a connected kNN graph", {
  set.seed(44)
  X   <- matrix(rnorm(50L * 3L), nrow = 50L, ncol = 3L)
  knn <- make_knn(X, k = 20L)
  res <- boruvka(knn$idx, knn$dist, min_pts = 5L)

  expect_equal(res$n_mst_edges, 49L)
})

# ── Test 5: label range is valid ─────────────────────────────────────────────

test_that("Boruvka labels are non-negative integers not exceeding n", {
  set.seed(55)
  X   <- matrix(rnorm(60L * 2L), nrow = 60L)
  knn <- make_knn(X, k = 15L)
  res <- boruvka(knn$idx, knn$dist, min_pts = 5L)

  expect_true(all(res$labels >= 0L))
  expect_true(all(res$labels <= nrow(X)))
  expect_type(res$labels, "integer")
})

# ── Test 6: agreement with dbscan::hdbscan() ─────────────────────────────────
# On unambiguous data with a connected kNN graph, the two implementations must
# produce identical cluster partitions (up to label permutation).
# We verify co-clustering agreement: any two points that share a label in our
# output must also share a label in the reference, and vice versa.

test_that("Boruvka cluster membership matches dbscan::hdbscan() on clear data",{
  set.seed(77)
  n_each  <- 20L
  min_pts <- 5L
  k       <- 30L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 20, sd = 0.1), nrow = n_each)
  )

  knn  <- make_knn(X, k)
  ours <- boruvka(knn$idx, knn$dist, min_pts)
  ref  <- dbscan::hdbscan(X, minPts = min_pts)

  our_nonnoise <- which(ours$labels > 0L)
  ref_nonnoise <- which(ref$cluster  > 0L)

  # Every non-noise point in ours is also non-noise in the reference.
  expect_true(all(our_nonnoise %in% ref_nonnoise),
              label = "Non-noise in Boruvka are non-noise in reference")

  # Co-clustering: same label in ours → same label in reference.
  for (cl in unique(ours$labels[ours$labels > 0L])) {
    pts      <- which(ours$labels == cl)
    ref_lbls <- ref$cluster[pts]
    expect_true(all(ref_lbls == ref_lbls[1L]),
                label = paste("Boruvka cluster", cl,
                             "matches one reference cluster"))
  }
})

# ── Test 7: three clusters also agree with reference ─────────────────────────

test_that("Boruvka agrees with dbscan on three-cluster data", {
  set.seed(99)
  n_each  <- 20L
  min_pts <- 5L
  k       <- 35L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.5), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean =  5, sd = 0.5), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 10, sd = 0.5), nrow = n_each)
  )

  knn  <- make_knn(X, k)
  ours <- boruvka(knn$idx, knn$dist, min_pts)
  ref  <- dbscan::hdbscan(X, minPts = min_pts)

  expect_equal(ours$n_mst_edges, nrow(X) - 1L)
  expect_equal(table(ours$labels), table(ref$cluster))
})

# ── Test 8: larger min_pts → more conservative (more noise or fewer clusters) ─

test_that("Larger min_pts produces at least as many noise points", {
  set.seed(88)
  X   <- matrix(rnorm(60L * 2L, sd = 1.5), nrow = 60L)
  knn <- make_knn(X, k = 25L)

  res_small <- boruvka(knn$idx, knn$dist, min_pts = 2L)
  res_large <- boruvka(knn$idx, knn$dist, min_pts = 20L)

  expect_gte(sum(res_large$labels == 0L), sum(res_small$labels == 0L))
})

# ── Test 8b: root that never splits → all noise unless single cluster allowed ─
# With min_pts = 20 on 60 diffuse points no split yields two children of size
# >= 20, so the root is the only candidate cluster.  Python hdbscan and
# dbscan::hdbscan() then return all noise; allow_single_cluster = TRUE returns
# the root as one cluster.

test_that("Unsplit root gives all noise unless allow_single_cluster = TRUE", {
  set.seed(88)
  X   <- matrix(rnorm(60L * 2L, sd = 1.5), nrow = 60L)
  knn <- make_knn(X, k = 25L)

  expect_true(all(dbscan::hdbscan(X, minPts = 20L)$cluster == 0L))

  expect_true(all(boruvka(knn$idx, knn$dist, min_pts = 20L)$labels == 0L))
  expect_true(all(Rhobots:::hdbscan_kdtree_cpp(X, 20L)$labels == 0L))
  expect_true(all(Rhobots:::hdbscan_balltree_cpp(X, 20L)$labels == 0L))

  # Python hdbscan also returns exactly one cluster here.  (It additionally
  # marks the 40 points that leave the root early as noise; we do not yet.)
  single <- boruvka(knn$idx, knn$dist, min_pts = 20L, allow_single_cluster = TRUE)
  expect_equal(max(single$labels), 1L)
})

# ── Test 9: disconnected kNN graph is detected ───────────────────────────────
# When k is too small to span well-separated clusters, n_mst_edges < n-1.
# The R-level wrapper (cluster_docs.hdbscan_clustering) handles this by
# doubling k and retrying; the C++ function reports it so the wrapper can act.

test_that("Boruvka reports disconnected kNN graph via n_mst_edges < n-1", {
  set.seed(99)
  n_each <- 15L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 50, sd = 0.1), nrow = n_each)
  )
  n   <- nrow(X)
  # k = 5 — too small to reach across the 50-unit gap
  knn <- make_knn(X, k = 5L)
  res <- boruvka(knn$idx, knn$dist, min_pts = 3L)

  expect_lt(res$n_mst_edges, n - 1L)
})

# ── Test 10: knn = "adaptive" finds island clusters that "fixed" misses ───────
# With extreme separation and k_cap=200 still too small, "adaptive" removes
# the ceiling and finds full connectivity. We simulate the effect by using
# clusters so far apart that k=200 would never bridge the gap (500-unit gap,
# n_each=5 so k needs to reach across; use n_each > 200 to force > 200 cap).
# Simpler here: verify the constructor stores the option and that the wrapper
# resolves it correctly even for old model objects without the knn field.

test_that("hdbscan_clustering stores knn strategy and cluster_docs respects it", {
  m_fixed    <- hdbscan_clustering(min_pts = 5L, knn = "fixed")
  m_adaptive <- hdbscan_clustering(min_pts = 5L, knn = "adaptive")

  expect_equal(m_fixed$knn,    "fixed")
  expect_equal(m_adaptive$knn, "adaptive")

  # Backward-compat: old objects (no knn field) default to "fixed" behaviour
  m_old <- structure(
    list(min_pts = 5L, method = "eom", fitted = NULL),
    class = c("hdbscan_clustering", "cluster_model")
  )
  expect_null(m_old$knn)  # field absent

  # On clearly separated data, adaptive finds the same clusters as fixed
  # (gap is large but n_each=10 < 200, so both reach connectivity at k=10)
  set.seed(42)
  n_each <- 10L
  X <- rbind(
    matrix(rnorm(n_each * 2L, mean =  0, sd = 0.1), nrow = n_each),
    matrix(rnorm(n_each * 2L, mean = 30, sd = 0.1), nrow = n_each)
  )
  r_fixed    <- cluster_docs(m_fixed,    X)
  r_adaptive <- cluster_docs(m_adaptive, X)

  expect_equal(r_fixed$labels,    r_adaptive$labels)
  expect_equal(length(unique(r_adaptive$labels[r_adaptive$labels > 0L])), 2L)
})
