# =============================================================================
# cache.R — Embedding persistence utilities.
#
# The intended workflow is:
#
#   # Session 1 — compute once and save
#   emb <- embed_texts_cached(enc, docs, cache_file = "emb.rds")
#
#   # Session 2 — load instantly, skip the encoder entirely
#   emb <- embed_texts_cached(cache_file = "emb.rds", texts = docs)
#
#   # Experiment freely with different models
#   fit1 <- fit_bertopic(docs = docs, embeddings = emb,
#                        cluster_model = kmeans_clustering(k = 8))
#   fit2 <- fit_bertopic(docs = docs, embeddings = emb,
#                        cluster_model = hdbscan_clustering(min_pts = 5))
# =============================================================================

#' Compute or load document embeddings from a cache file
#'
#' On the first call (or when \code{overwrite = TRUE}) the encoder is used to
#' compute embeddings and the result is written to \code{cache_file}.  On
#' subsequent calls the cached matrix is read directly, making it free to
#' experiment with different dimensionality-reduction or clustering settings
#' without re-running the expensive forward passes.
#'
#' @param encoder An encoder from \code{\link{load_hf_bert}}.  Required when
#'   computing embeddings; may be \code{NULL} when loading from cache.
#' @param texts Character vector of documents to embed.  Still required even
#'   when loading from cache so the row count can be validated.
#' @param cache_file Path to a \code{.rds} file.  If the file exists and
#'   \code{overwrite = FALSE}, embeddings are loaded from it.  If the file
#'   does not exist (or \code{overwrite = TRUE}), embeddings are computed and
#'   written to this path.  Pass \code{NULL} to skip caching entirely.
#' @param overwrite If \code{TRUE}, always recompute even if the cache file
#'   exists (default \code{FALSE}).
#' @param batch_size,max_length,normalize,device,prefix,chunk_strategy,chunk_overlap
#'   Forwarded to \code{\link{embed_texts}}.  \code{prefix} defaults to
#'   \code{NULL}, inheriting the encoder's stored prefix.
#' @param verbose Print progress messages.
#' @return A numeric matrix with \code{length(texts)} rows.
#' @export
embed_texts_cached <- function(encoder    = NULL,
                                texts,
                                cache_file     = NULL,
                                overwrite      = FALSE,
                                batch_size     = 32L,
                                max_length     = 256L,
                                normalize      = TRUE,
                                device         = "cpu",
                                prefix         = NULL,
                                chunk_strategy = c("truncate", "mean", "first"),
                                chunk_overlap  = 0L,
                                verbose        = interactive()) {
  chunk_strategy <- match.arg(chunk_strategy)
  # --- Try to load from cache ----------------------------------------------
  if (!is.null(cache_file) && file.exists(cache_file) && !overwrite) {
    if (verbose) message("Loading embeddings from cache: ", cache_file)
    emb <- load_embeddings(cache_file)
    if (nrow(emb) != length(texts))
      stop("Cached embeddings have ", nrow(emb), " rows but 'texts' has ",
           length(texts), " elements.\n",
           "Re-run with overwrite = TRUE to refresh the cache.")
    if (verbose)
      message("Loaded ", nrow(emb), " × ", ncol(emb),
              " embedding matrix from cache.")
    return(emb)
  }

  # --- Compute embeddings --------------------------------------------------
  if (is.null(encoder))
    stop("'encoder' is required to compute embeddings ",
         "(no cache found at '", cache_file %||% "<no path>", "').")

  emb <- embed_texts(encoder, texts,
                      batch_size     = batch_size,
                      max_length     = max_length,
                      normalize      = normalize,
                      device         = device,
                      prefix         = prefix,
                      chunk_strategy = chunk_strategy,
                      chunk_overlap  = chunk_overlap,
                      verbose        = verbose)

  # --- Save to cache --------------------------------------------------------
  if (!is.null(cache_file)) {
    save_embeddings(emb, cache_file)
    if (verbose) message("Embeddings cached to: ", cache_file)
  }

  emb
}

#' Save an embedding matrix to disk
#'
#' @param embeddings A numeric matrix (rows = documents, columns = dimensions).
#' @param path File path.  Use a \code{.rds} extension (recommended — lossless,
#'   fast) or \code{.csv} (portable but larger and slower).  If no extension is
#'   given, \code{.rds} is appended automatically.
#' @return The resolved file path, invisibly.
#' @export
save_embeddings <- function(embeddings, path) {
  if (!is.matrix(embeddings) || !is.numeric(embeddings))
    stop("'embeddings' must be a numeric matrix.")

  ext <- tolower(tools::file_ext(path))
  if (ext == "") { path <- paste0(path, ".rds"); ext <- "rds" }

  if (ext == "rds") {
    saveRDS(embeddings, path)
  } else if (ext == "csv") {
    utils::write.csv(embeddings, path, row.names = FALSE)
  } else {
    stop("Unsupported format '.", ext, "'. Use .rds (recommended) or .csv.")
  }

  invisible(path)
}

#' Load an embedding matrix from disk
#'
#' @param path Path to a \code{.rds} or \code{.csv} file previously written by
#'   \code{\link{save_embeddings}}.
#' @return A numeric matrix.
#' @export
load_embeddings <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)

  ext <- tolower(tools::file_ext(path))
  emb <- if (ext == "rds") {
    readRDS(path)
  } else if (ext == "csv") {
    as.matrix(utils::read.csv(path, check.names = FALSE))
  } else {
    stop("Unsupported format '.", ext, "'. Use .rds or .csv.")
  }

  if (!is.matrix(emb) || !is.numeric(emb))
    stop("Loaded file does not contain a numeric matrix.")
  emb
}
