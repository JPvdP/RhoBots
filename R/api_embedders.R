# =============================================================================
# api_embedders.R  --  API-based embedding backends (OpenAI, Cohere)
#
# Returns `api_embedder` objects compatible with embed_texts(),
# embed_texts_cached(), and fit_bertopic()  --  no torch required.
# =============================================================================

#' Load an OpenAI embedding model
#'
#' Returns an `api_embedder` that calls the OpenAI Embeddings API.
#' Requires the \pkg{httr2} package and a valid OpenAI API key.
#'
#' The returned object is compatible with [embed_texts()],
#' [embed_texts_cached()], and [fit_bertopic()].
#'
#' @param model OpenAI embedding model name.
#'   \itemize{
#'     \item `"text-embedding-3-small"` (default)  --  1536-d, fast and cheap
#'     \item `"text-embedding-3-large"`  --  3072-d, highest quality
#'     \item `"text-embedding-ada-002"`  --  legacy 1536-d model
#'   }
#' @param api_key API key.  Defaults to `Sys.getenv("OPENAI_API_KEY")`.
#' @param dimensions Optional integer to request a reduced output dimension
#'   (only supported by `text-embedding-3-*` models).
#' @param base_url API base URL.  Override for Azure OpenAI or API proxies.
#' @return An `api_embedder` object.
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_openai_embedder()   # uses OPENAI_API_KEY env var
#'   emb <- embed_texts(enc, c("Transformers in R.", "Topic modelling."))
#'   fit <- fit_bertopic(enc, docs = abstracts)
#' }
load_openai_embedder <- function(model      = "text-embedding-3-small",
                                  api_key    = Sys.getenv("OPENAI_API_KEY"),
                                  dimensions = NULL,
                                  base_url   = "https://api.openai.com/v1") {
  if (!requireNamespace("httr2", quietly = TRUE))
    stop("Please install.packages('httr2') to use load_openai_embedder().")
  if (!nzchar(api_key))
    stop("OpenAI API key not found. ",
         "Set the OPENAI_API_KEY environment variable or pass api_key= directly.")
  structure(
    list(provider   = "openai",
         model      = model,
         api_key    = api_key,
         dimensions = dimensions,
         base_url   = base_url,
         prefix     = ""),
    class = c("api_embedder", "list")
  )
}

#' Load a Cohere embedding model
#'
#' Returns an `api_embedder` that calls the Cohere Embed API.
#' Requires the \pkg{httr2} package and a valid Cohere API key.
#'
#' The returned object is compatible with [embed_texts()],
#' [embed_texts_cached()], and [fit_bertopic()].
#'
#' @param model Cohere embedding model name.
#'   \itemize{
#'     \item `"embed-english-v3.0"` (default)  --  1024-d English
#'     \item `"embed-multilingual-v3.0"`  --  1024-d, 100+ languages
#'     \item `"embed-english-light-v3.0"`  --  384-d, faster/cheaper
#'   }
#' @param api_key API key.  Defaults to `Sys.getenv("COHERE_API_KEY")`.
#' @param input_type Cohere input type controlling the embedding space used.
#'   \itemize{
#'     \item `"search_document"` (default)  --  for indexing documents
#'     \item `"search_query"`  --  for embedding queries
#'     \item `"clustering"`  --  optimized for clustering
#'     \item `"classification"`  --  optimized for classification
#'   }
#' @return An `api_embedder` object.
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_cohere_embedder()   # uses COHERE_API_KEY env var
#'   emb <- embed_texts(enc, docs)
#' }
load_cohere_embedder <- function(model      = "embed-english-v3.0",
                                  api_key    = Sys.getenv("COHERE_API_KEY"),
                                  input_type = "search_document") {
  if (!requireNamespace("httr2", quietly = TRUE))
    stop("Please install.packages('httr2') to use load_cohere_embedder().")
  if (!nzchar(api_key))
    stop("Cohere API key not found. ",
         "Set the COHERE_API_KEY environment variable or pass api_key= directly.")
  structure(
    list(provider   = "cohere",
         model      = model,
         api_key    = api_key,
         input_type = input_type,
         prefix     = ""),
    class = c("api_embedder", "list")
  )
}

#' @rdname embed_texts
#' @export
embed_texts.api_embedder <- function(encoder,
                                      texts,
                                      batch_size = NULL,
                                      normalize  = TRUE,
                                      prefix     = NULL,
                                      verbose    = interactive(),
                                      ...) {
  if (!requireNamespace("httr2", quietly = TRUE))
    stop("Please install.packages('httr2') to use API embedders.")

  if (is.null(prefix)) prefix <- encoder$prefix %||% ""
  if (nzchar(prefix))  texts  <- paste0(prefix, texts)

  if (is.null(batch_size))
    batch_size <- if (encoder$provider == "openai") 512L else 96L

  if (encoder$provider == "openai") {
    .embed_openai(encoder, texts, batch_size, normalize, verbose)
  } else if (encoder$provider == "cohere") {
    .embed_cohere(encoder, texts, batch_size, normalize, verbose)
  } else {
    stop("Unknown provider: ", encoder$provider)
  }
}

# --- OpenAI -------------------------------------------------------------------
.embed_openai <- function(enc, texts, batch_size, normalize, verbose) {
  n   <- length(texts)
  out <- vector("list", ceiling(n / batch_size))
  idx <- 0L

  for (start in seq(1L, n, by = batch_size)) {
    end   <- min(start + batch_size - 1L, n)
    batch <- texts[start:end]

    body <- list(model = enc$model, input = batch)
    if (!is.null(enc$dimensions)) body$dimensions <- as.integer(enc$dimensions)

    resp <- httr2::request(paste0(enc$base_url, "/embeddings")) |>
      httr2::req_headers(
        Authorization  = paste("Bearer", enc$api_key),
        `Content-Type` = "application/json"
      ) |>
      httr2::req_body_json(body) |>
      httr2::req_error(is_error = \(r) FALSE) |>
      httr2::req_perform()

    if (httr2::resp_status(resp) != 200L) {
      err <- tryCatch(httr2::resp_body_json(resp)$error$message,
                      error = \(e) httr2::resp_body_string(resp))
      stop("OpenAI API error: ", err)
    }

    data  <- httr2::resp_body_json(resp)$data
    data  <- data[order(vapply(data, `[[`, 0L, "index"))]
    mat   <- do.call(rbind, lapply(data, \(x) as.numeric(unlist(x$embedding))))

    idx        <- idx + 1L
    out[[idx]] <- mat
    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }

  result <- do.call(rbind, out)
  if (normalize) {
    norms <- sqrt(rowSums(result^2))
    norms[norms == 0] <- 1
    result <- result / norms
  }
  result
}

# --- Cohere -------------------------------------------------------------------
.embed_cohere <- function(enc, texts, batch_size, normalize, verbose) {
  n   <- length(texts)
  out <- vector("list", ceiling(n / batch_size))
  idx <- 0L

  for (start in seq(1L, n, by = batch_size)) {
    end   <- min(start + batch_size - 1L, n)
    batch <- texts[start:end]

    resp <- httr2::request("https://api.cohere.ai/v2/embed") |>
      httr2::req_headers(
        Authorization  = paste("Bearer", enc$api_key),
        `Content-Type` = "application/json"
      ) |>
      httr2::req_body_json(list(
        model           = enc$model,
        texts           = batch,
        input_type      = enc$input_type,
        embedding_types = list("float")
      )) |>
      httr2::req_error(is_error = \(r) FALSE) |>
      httr2::req_perform()

    if (httr2::resp_status(resp) != 200L) {
      err <- tryCatch(httr2::resp_body_json(resp)$message,
                      error = \(e) httr2::resp_body_string(resp))
      stop("Cohere API error: ", err)
    }

    embeddings <- httr2::resp_body_json(resp)$embeddings$float
    mat        <- do.call(rbind, lapply(embeddings, as.numeric))

    idx        <- idx + 1L
    out[[idx]] <- mat
    if (verbose) message(sprintf("  embedded %d / %d", end, n))
  }

  result <- do.call(rbind, out)
  if (normalize) {
    norms <- sqrt(rowSums(result^2))
    norms[norms == 0] <- 1
    result <- result / norms
  }
  result
}

#' @export
print.api_embedder <- function(x, ...) {
  cat("<api_embedder>\n")
  cat("  provider: ", x$provider, "\n")
  cat("  model:    ", x$model, "\n")
  if (x$provider == "cohere")
    cat("  input_type:", x$input_type, "\n")
  if (!is.null(x$dimensions))
    cat("  dimensions:", x$dimensions, "\n")
  if (nzchar(x$prefix %||% ""))
    cat("  prefix:   \"", x$prefix, "\"\n", sep = "")
  invisible(x)
}
