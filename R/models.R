# =============================================================================
# models.R  --  Pluggable dimensionality-reduction and clustering models.
#
# S3 protocol
# -----------
# Dimensionality reduction
#   dim_reduce(model, X, seed, verbose) -> list(embedding, model)
#   dim_project(model, X)              -> matrix   (transform new data)
#
# Clustering
#   cluster_docs(model, X, seed)       -> list(labels, model)
#   labels: integer vector, -1 = noise, 0, 1, 2, ... = cluster IDs
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
#' @examples
#' \dontrun{
#'   m   <- umap_reduction(n_neighbors = 5L, n_components = 2L)
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("first doc", "second doc", "third doc"))
#'   res <- dim_reduce(m, emb)
#' }
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
#' @examples
#' \dontrun{
#'   m    <- umap_reduction(n_neighbors = 5L, n_components = 2L)
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb  <- embed_texts(enc, c("first doc", "second doc", "third doc"))
#'   res  <- dim_reduce(m, emb)
#'   new_emb <- embed_texts(enc, c("new document"))
#'   dim_project(res$model, new_emb)
#' }
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
#' @examples
#' \dontrun{
#'   m   <- hdbscan_clustering(min_pts = 3L)
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, c("cats and dogs", "machine learning",
#'                              "pets and animals", "neural networks"))
#'   res <- cluster_docs(m, emb)
#'   res$labels
#' }
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
#' @examples
#' m <- umap_reduction(n_neighbors = 10L, n_components = 3L)
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
#' @examples
#' m <- pca_reduction(n_components = 3L)
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
#' @examples
#' m <- no_reduction()
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
#' HDBSCAN density clustering
#'
#' Default clustering model for \code{\link{fit_bertopic}}.  Uses a Boruvka
#' minimum spanning tree on the mutual-reachability kNN graph (Rcpp) instead
#' of the naive Prim's algorithm in \code{dbscan::hdbscan()}, which avoids the
#' O(n^2) memory cost that causes OOM at ~30K+ documents.  Memory is O(n x k)
#' throughout, making it practical at 100K+ documents.
#'
#' @param min_pts Minimum cluster size and core-distance order (default 10).
#' @param method \code{"eom"} (excess of mass, default) or \code{"leaf"}.
#'   The leaf method uses \code{dbscan::hdbscan()} on small corpora only.
#' @return An \code{hdbscan_clustering} model object.
#' @examples
#' m <- hdbscan_clustering(min_pts = 5L)
#' @export
hdbscan_clustering <- function(min_pts = 10L, method = c("eom", "leaf")) {
  method <- match.arg(method)
  structure(
    list(min_pts = as.integer(min_pts), method = method, fitted = NULL),
    class = c("hdbscan_clustering", "cluster_model")
  )
}

#' @export
cluster_docs.hdbscan_clustering <- function(model, X, seed = 42L) {
  min_pts <- model$min_pts

  if (model$method == "leaf") {
    # Leaf method uses the existing dbscan-based helper (small corpora only)
    clust             <- .hdbscan_leaf(X, min_pts = min_pts)
    labels            <- clust$cluster
    labels[labels == 0L] <- -1L
    model$fitted      <- clust
    return(list(labels = labels, model = model))
  }

  # EOM method: Boruvka MST via Rcpp  --  O(n x k) memory, scales to 100K+
  # Pre-compute kNN with dbscan::kNN() (uses kd-trees; fast and low-memory).
  # If the kNN graph is disconnected (nm < n-1), double k and retry until the
  # MST is complete or k exceeds the cap.
  n     <- nrow(X)
  k     <- max(min_pts, 15L)
  k_cap <- min(n - 1L, 200L)

  repeat {
    knn    <- dbscan::kNN(X, k = k, sort = TRUE)
    result <- hdbscan_boruvka_cpp(knn$id, knn$dist, min_pts)
    if (result$n_mst_edges >= n - 1L || k >= k_cap) break
    k <- min(k * 2L, k_cap)
    message(sprintf(
      "  HDBSCAN: kNN graph disconnected (MST edges %d < %d), retrying with k=%d",
      result$n_mst_edges, n - 1L, k))
  }

  labels            <- result$labels
  labels[labels == 0L] <- -1L    # match dbscan convention: 0 -> -1 for noise

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
#' @examples
#' m <- kmeans_clustering(k = 5L)
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
#' @examples
#' m <- agglomerative_clustering(k = 5L)
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
