# =============================================================================
# persistence.R  --  Structured save / load for bertopic_fit objects.
#
# Saves the fit to a directory with separate files for the heavy matrices so
# that the model can be inspected or partially loaded without reading everything
# into memory.
# =============================================================================

#' Save a fitted BERTopic model to disk
#'
#' Writes a \code{bertopic_fit} object to a directory.  The fit is split into:
#' \itemize{
#'   \item \code{metadata.json}  --  human-readable summary (topic labels, counts,
#'     parameters).
#'   \item \code{fit.rds}  --  the full fit object minus the large matrices.
#'   \item \code{embeddings.rds}  --  the document embedding matrix (optional).
#'   \item \code{dtm.rds}  --  the sparse document-term matrix (optional).
#' }
#' Splitting the heavy matrices means the directory can be browsed and the
#' metadata inspected without loading gigabytes into R.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param path Directory path.  Created if it does not exist.
#' @param include_embeddings Save the embedding matrix (default \code{TRUE}).
#'   Set to \code{FALSE} to save space; \code{topic_quality()} and
#'   \code{reduce_outliers()} will not work after reloading.
#' @param include_dtm Save the document-term matrix (default \code{TRUE}).
#'   Set to \code{FALSE} to save space; c-TF-IDF recomputation will not work.
#' @param compress Compress \code{.rds} files (default \code{TRUE}).
#' @return Invisibly, the normalised \code{path}.
#' @seealso \code{\link{load_bertopic}}
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   save_bertopic(fit, tempdir())
#' }
#' @export
save_bertopic <- function(fit, path,
                           include_embeddings = TRUE,
                           include_dtm        = TRUE,
                           compress           = TRUE) {
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object.")

  dir.create(path, recursive = TRUE, showWarnings = FALSE)

  # Extract heavy matrices before saving the light object
  emb <- fit$embeddings
  dtm <- fit$dtm
  fit_light             <- fit
  fit_light$embeddings  <- NULL
  fit_light$dtm         <- NULL

  # Human-readable metadata
  topics_nn <- sort(setdiff(unique(fit$clusters), -1L))
  meta <- list(
    rhobots_version = as.character(utils::packageVersion("Rhobots")),
    saved_at        = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    n_docs          = length(fit$docs),
    n_topics        = length(topics_nn),
    n_noise         = sum(fit$clusters == -1L),
    top_n_terms     = fit$top_n_terms %||% NA,
    topic_labels    = fit$topic_labels,
    has_embeddings  = include_embeddings,
    has_dtm         = include_dtm
  )
  jsonlite::write_json(meta,
                       file.path(path, "metadata.json"),
                       auto_unbox = TRUE, pretty = TRUE)

  saveRDS(fit_light, file.path(path, "fit.rds"), compress = compress)

  if (include_embeddings && !is.null(emb))
    saveRDS(emb, file.path(path, "embeddings.rds"), compress = compress)

  if (include_dtm && !is.null(dtm))
    saveRDS(dtm, file.path(path, "dtm.rds"), compress = compress)

  message("Model saved to: ", normalizePath(path))
  invisible(normalizePath(path))
}

#' Load a previously saved BERTopic model
#'
#' Reads a model directory written by \code{\link{save_bertopic}} and
#' reconstructs the \code{bertopic_fit} object.
#'
#' @param path Path to the directory created by \code{\link{save_bertopic}}.
#' @return A \code{bertopic_fit} object.
#' @seealso \code{\link{save_bertopic}}
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit  <- fit_bertopic(docs = abstracts, encoder = enc)
#'   path <- tempdir()
#'   save_bertopic(fit, path)
#'   fit2 <- load_bertopic(path)
#' }
#' @export
load_bertopic <- function(path) {
  if (!dir.exists(path))
    stop("Directory not found: ", path)

  fit_path <- file.path(path, "fit.rds")
  if (!file.exists(fit_path))
    stop("No fit.rds found in '", path, "'. ",
         "Is this a valid Rhobots save directory?")

  fit <- readRDS(fit_path)

  emb_path <- file.path(path, "embeddings.rds")
  if (file.exists(emb_path))
    fit$embeddings <- readRDS(emb_path)

  dtm_path <- file.path(path, "dtm.rds")
  if (file.exists(dtm_path))
    fit$dtm <- readRDS(dtm_path)

  fit
}
