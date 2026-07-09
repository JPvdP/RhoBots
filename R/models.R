# =============================================================================
# models.R — Pluggable dimensionality-reduction and clustering models.
#
# S3 protocol
# -----------
# Dimensionality reduction
#   dim_reduce(model, X, seed, verbose) → list(embedding, model)
#   dim_project(model, X)              → matrix   (transform new data)
#
# Clustering
#   cluster_docs(model, X, seed)       → list(labels, model)
#   labels: integer vector, -1 = noise, 0, 1, 2, … = cluster IDs
#
# Built-in models
# ---------------
#   umap_reduction()           default dim reduction (uwot)
#   pca_reduction()            stats::prcomp
#   no_reduction()             identity pass-through
#
#   hdbscan_clustering()       default clustering (dbscan)
#   kmeans_clustering()        stats::kmeans  (no noise)
#   agglomerative_clustering() stats::hclust + cutree  (no noise)
# =============================================================================

# -----------------------------------------------------------------------------
# S3 generics
# -----------------------------------------------------------------------------

#' Fit a dimensionality-reduction model and return the reduced matrix
#'
#' @param model A dimensionality-reduction model object.
#' @param X Numeric matrix to reduce.
#' @param seed Random seed (default 42).
#' @param verbose Print progress (default \code{FALSE}).
#' @return A list with \code{$embedding} (reduced matrix) and \code{$model}
#'   (the fitted model, for use with \code{\link{dim_project}}).
#' @export
dim_reduce <- function(model, X, seed = 42L, verbose = FALSE)
  UseMethod("dim_reduce")

#' Project new data using a fitted dimensionality-reduction model
#'
#' @param model A fitted dimensionality-reduction model (returned inside the
#'   list from \code{\link{dim_reduce}}).
#' @param X New numeric matrix to project.
#' @return A numeric matrix with the same number of columns as the training
#'   embedding.
#' @export
dim_project <- function(model, X) UseMethod("dim_project")

#' Fit a clustering model and return cluster labels
#'
#' @param model A clustering model object.
#' @param X Numeric matrix to cluster.
#' @param seed Random seed (default 42).
#' @return A list with \code{$labels} (integer vector; \code{-1} = noise,
#'   \code{0, 1, 2, ...} = cluster IDs) and \code{$model} (the fitted
#'   model).
#' @export
cluster_docs <- function(model, X, seed = 42L) UseMethod("cluster_docs")

# -----------------------------------------------------------------------------
# UMAP reduction  (default)
# -----------------------------------------------------------------------------

#' UMAP dimensionality reduction
#'
#' Wraps \code{uwot::umap}.  This is the default dimensionality-reduction
#' model used by \code{\link{fit_bertopic}}.
#'
#' @param n_neighbors,n_components,min_dist,metric Passed directly to
#'   \code{uwot::umap}.
#' @return A \code{umap_reduction} model object.
#' @export
umap_reduction <- function(n_neighbors  = 15L,
                            n_components = 5L,
                            min_dist     = 0.0,
                            metric       = "cosine") {
  structure(
    list(n_neighbors  = n_neighbors,
         n_components = n_components,
         min_dist     = min_dist,
         metric       = metric,
         fitted       = NULL),
    class = c("umap_reduction", "dim_reduction_model")
  )
}

#' @export
dim_reduce.umap_reduction <- function(model, X, seed = 42L, verbose = FALSE) {
  set.seed(seed)
  res         <- uwot::umap(X,
                             n_neighbors  = model$n_neighbors,
                             n_components = model$n_components,
                             min_dist     = model$min_dist,
                             metric       = model$metric,
                             ret_model    = TRUE,
                             verbose      = verbose)
  model$fitted <- res
  list(embedding = res$embedding, model = model)
}

#' @export
dim_project.umap_reduction <- function(model, X) {
  if (is.null(model$fitted))
    stop("UMAP model not fitted. Run dim_reduce() first.")
  uwot::umap_transform(X, model$fitted)
}

# -----------------------------------------------------------------------------
# PCA reduction
# -----------------------------------------------------------------------------

#' PCA dimensionality reduction
#'
#' Wraps \code{stats::prcomp}.  Unlike UMAP, PCA is deterministic and fast,
#' but often produces lower-quality cluster separation for text embeddings.
#'
#' @param n_components Number of principal components to retain.
#' @param scale. Whether to scale variables before computing PCA
#'   (default \code{FALSE}; BERT embeddings are already normalised).
#' @return A \code{pca_reduction} model object.
#' @export
pca_reduction <- function(n_components = 5L, scale. = FALSE) {
  structure(
    list(n_components = as.integer(n_components),
         scale.       = scale.,
         fitted       = NULL),
    class = c("pca_reduction", "dim_reduction_model")
  )
}

#' @export
dim_reduce.pca_reduction <- function(model, X, seed = 42L, verbose = FALSE) {
  pca          <- stats::prcomp(X, rank. = model$n_components,
                                 center = TRUE, scale. = model$scale.)
  model$fitted <- pca
  list(embedding = pca$x[, seq_len(model$n_components), drop = FALSE],
       model = model)
}

#' @export
dim_project.pca_reduction <- function(model, X) {
  if (is.null(model$fitted))
    stop("PCA model not fitted. Run dim_reduce() first.")
  stats::predict(model$fitted, X)[, seq_len(model$n_components), drop = FALSE]
}

# -----------------------------------------------------------------------------
# No reduction  (identity pass-through)
# -----------------------------------------------------------------------------

#' Skip dimensionality reduction (identity pass-through)
#'
#' Passes the raw embeddings directly to the clustering step.  Useful when
#' the embeddings are already low-dimensional or when you want to cluster in
#' the original space.
#'
#' @return A \code{no_reduction} model object.
#' @export
no_reduction <- function() {
  structure(list(), class = c("no_reduction", "dim_reduction_model"))
}

#' @export
dim_reduce.no_reduction  <- function(model, X, seed = 42L, verbose = FALSE)
  list(embedding = X, model = model)

#' @export
dim_project.no_reduction <- function(model, X) X

# -----------------------------------------------------------------------------
# HDBSCAN clustering  (default)
# -----------------------------------------------------------------------------

#' HDBSCAN clustering
#'
#' Wraps \code{dbscan::hdbscan} (EOM method) or the leaf-selection variant
#' from \code{.hdbscan_leaf()}.  This is the default clustering model used by
#' \code{\link{fit_bertopic}}.
#'
#' For large corpora (\code{n > sample_size}) the function automatically uses
#' an approximate strategy: HDBSCAN is run on a random sample of
#' \code{sample_size} documents, and every remaining document is assigned to
#' the nearest sample cluster centroid.  Noise points from the sample (\code{-1})
#' are kept as a centroid class only if no better option exists.  Set
#' \code{sample_size = Inf} to always run exact HDBSCAN (may OOM for n > ~30K).
#'
#' @param min_pts Minimum cluster size / neighbourhood size (default 10).
#' @param method \code{"eom"} (default) or \code{"leaf"}.
#' @param sample_size Maximum number of documents passed to exact HDBSCAN.
#'   When \code{nrow(X) > sample_size}, approximate mode is used (default 25000).
#' @return An \code{hdbscan_clustering} model object.
#' @export
hdbscan_clustering <- function(min_pts   = 10L,
                                method    = c("eom", "leaf"),
                                sample_size = 25000L) {
  method <- match.arg(method)
  structure(
    list(min_pts = as.integer(min_pts), method = method,
         sample_size = sample_size, fitted = NULL),
    class = c("hdbscan_clustering", "cluster_model")
  )
}

#' @export
cluster_docs.hdbscan_clustering <- function(model, X, seed = 42L) {
  n <- nrow(X)

  if (n <= model$sample_size) {
    # ── Exact HDBSCAN ──────────────────────────────────────────────────────────
    clust <- if (model$method == "leaf") {
      .hdbscan_leaf(X, min_pts = model$min_pts)
    } else {
      dbscan::hdbscan(X, minPts = model$min_pts)
    }
    labels            <- clust$cluster
    labels[labels == 0L] <- -1L
    model$fitted      <- clust
    return(list(labels = labels, model = model))
  }

  # ── Approximate HDBSCAN for large n ─────────────────────────────────────────
  # 1. Draw a stratified random sample.
  # 2. Run exact HDBSCAN on the sample.
  # 3. Compute the centroid of each discovered cluster.
  # 4. Assign every out-of-sample document to its nearest centroid.
  message(sprintf(
    "  n = %d > sample_size = %d: using approximate HDBSCAN (sample + assign).",
    n, model$sample_size
  ))

  set.seed(seed)
  samp_idx  <- sort(sample.int(n, model$sample_size))
  X_samp    <- X[samp_idx, , drop = FALSE]

  clust <- if (model$method == "leaf") {
    .hdbscan_leaf(X_samp, min_pts = model$min_pts)
  } else {
    dbscan::hdbscan(X_samp, minPts = model$min_pts)
  }
  samp_labels           <- clust$cluster
  samp_labels[samp_labels == 0L] <- -1L
  model$fitted          <- clust

  # Compute cluster centroids (including -1 as fallback).
  cluster_ids <- sort(unique(samp_labels))
  centroids   <- do.call(rbind, lapply(cluster_ids, function(cl) {
    colMeans(X_samp[samp_labels == cl, , drop = FALSE])
  }))
  rownames(centroids) <- as.character(cluster_ids)

  # Only use non-noise centroids for assignment of out-of-sample points;
  # fall back to noise (-1) if a document is truly isolated.
  non_noise_ids <- cluster_ids[cluster_ids != -1L]

  labels    <- integer(n)
  labels[samp_idx] <- samp_labels

  rest_idx <- setdiff(seq_len(n), samp_idx)
  if (length(rest_idx) > 0L) {
    X_rest <- X[rest_idx, , drop = FALSE]

    if (length(non_noise_ids) == 0L) {
      labels[rest_idx] <- -1L
    } else {
      cent_nn <- centroids[as.character(non_noise_ids), , drop = FALSE]
      # Nearest centroid via squared Euclidean distance: ||x - c||² = ||x||² - 2xᵀc + ||c||²
      # outer() broadcasts the two norms correctly (avoids R's column-wise recycling).
      dists   <- -2 * tcrossprod(X_rest, cent_nn) +
                 outer(rowSums(X_rest^2), rowSums(cent_nn^2), "+")
      nearest <- non_noise_ids[max.col(-dists)]
      labels[rest_idx] <- nearest
    }
  }

  list(labels = labels, model = model)
}

# -----------------------------------------------------------------------------
# K-means clustering
# -----------------------------------------------------------------------------

#' K-means clustering
#'
#' Wraps \code{stats::kmeans}.  Unlike HDBSCAN, k-means assigns every
#' document to a cluster (no noise label \code{-1}) and requires specifying
#' the number of clusters \code{k} in advance.
#'
#' @param k Number of clusters.
#' @param nstart Number of random restarts (default 10).
#' @param iter.max Maximum iterations (default 300).
#' @return A \code{kmeans_clustering} model object.
#' @export
kmeans_clustering <- function(k, nstart = 10L, iter.max = 300L) {
  if (missing(k)) stop("'k' (number of clusters) is required.")
  structure(
    list(k        = as.integer(k),
         nstart   = as.integer(nstart),
         iter.max = as.integer(iter.max),
         fitted   = NULL),
    class = c("kmeans_clustering", "cluster_model")
  )
}

#' @export
cluster_docs.kmeans_clustering <- function(model, X, seed = 42L) {
  set.seed(seed)
  km           <- stats::kmeans(X, centers = model$k,
                                 nstart   = model$nstart,
                                 iter.max = model$iter.max)
  model$fitted <- km
  list(labels = km$cluster - 1L, model = model)   # 0-based, no noise
}

# -----------------------------------------------------------------------------
# Agglomerative clustering
# -----------------------------------------------------------------------------

#' Agglomerative (hierarchical) clustering
#'
#' Wraps \code{stats::hclust} + \code{stats::cutree}.  Like k-means, every
#' document is assigned to a cluster (no noise label).
#'
#' @param k Number of clusters to cut the dendrogram into.
#' @param linkage Linkage method passed to \code{stats::hclust}
#'   (default \code{"ward.D2"}).
#' @return An \code{agglomerative_clustering} model object.
#' @export
agglomerative_clustering <- function(k, linkage = "ward.D2") {
  if (missing(k)) stop("'k' (number of clusters) is required.")
  structure(
    list(k = as.integer(k), linkage = linkage, fitted = NULL),
    class = c("agglomerative_clustering", "cluster_model")
  )
}

#' @export
cluster_docs.agglomerative_clustering <- function(model, X, seed = 42L) {
  hc           <- stats::hclust(stats::dist(X), method = model$linkage)
  labels       <- stats::cutree(hc, k = model$k) - 1L   # 0-based, no noise
  model$fitted <- hc
  list(labels = labels, model = model)
}
