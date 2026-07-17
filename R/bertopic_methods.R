# =============================================================================
# bertopic_methods.R  --  Accessor methods and out-of-sample prediction for
# bertopic_fit objects.
#
# Mirrors the Python BERTopic API:
#   get_topics(), get_topic(), get_topic_info(), get_document_info(),
#   get_representative_docs(), find_topics(),
#   predict.bertopic_fit() / transform_bertopic().
# =============================================================================

#' Get all topic-term representations
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param top_n Maximum terms per topic. \code{NULL} (default) returns all
#'   stored terms.
#' @return A named list; each element is a data frame with columns
#'   \code{term} and \code{score} sorted by descending score.  Names are
#'   topic IDs (character strings, including \code{"-1"} for noise).
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   get_topics(fit, top_n = 5L)
#' }
#' @export
get_topics <- function(fit, top_n = NULL) {
  tt <- fit$topic_terms
  topics <- sort(unique(tt$topic))
  setNames(lapply(topics, function(t) {
    rows <- tt[tt$topic == t, ]
    rows <- rows[order(rows$rank), , drop = FALSE]
    if (!is.null(top_n))
      rows <- rows[seq_len(min(top_n, nrow(rows))), , drop = FALSE]
    data.frame(term = rows$term, score = rows$score,
               stringsAsFactors = FALSE, row.names = NULL)
  }), as.character(topics))
}

#' Get term-score representation for a single topic
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param topic Integer topic ID.
#' @return A data frame with columns \code{term} and \code{score}, or
#'   \code{NULL} if the topic ID is not found.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   get_topic(fit, topic = 0L)
#' }
#' @export
get_topic <- function(fit, topic) {
  rows <- fit$topic_terms[fit$topic_terms$topic == as.integer(topic), ]
  if (nrow(rows) == 0L) return(NULL)
  rows <- rows[order(rows$rank), c("term", "score"), drop = FALSE]
  rownames(rows) <- NULL
  rows
}

#' Get topic-level metadata as a data frame
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param topic Optional integer. If provided, only the row for that topic
#'   is returned.
#' @return A data frame with columns:
#'   \describe{
#'     \item{\code{Topic}}{Integer topic ID (\code{-1} = noise/unassigned).}
#'     \item{\code{Count}}{Number of documents assigned to this topic.}
#'     \item{\code{Name}}{Auto-generated label (e.g. \code{"0_model_data_..."}).}
#'     \item{\code{Representation}}{Top-5 terms, comma-separated.}
#'   }
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   get_topic_info(fit)
#' }
#' @export
get_topic_info <- function(fit, topic = NULL) {
  all_topics <- sort(unique(fit$clusters))
  rows <- lapply(all_topics, function(t) {
    count   <- sum(fit$clusters == t)
    name    <- fit$topic_labels[[as.character(t)]] %||% as.character(t)
    rep_str <- if (t == -1L) {
      ""
    } else {
      tt <- fit$topic_terms[fit$topic_terms$topic == t, ]
      tt <- tt[order(tt$rank), ]
      paste(tt$term[seq_len(min(5L, nrow(tt)))], collapse = ", ")
    }
    data.frame(Topic = t, Count = count, Name = name,
               Representation = rep_str, stringsAsFactors = FALSE)
  })
  info <- do.call(rbind, rows)
  rownames(info) <- NULL
  if (!is.null(topic))
    info <- info[info$Topic == as.integer(topic), , drop = FALSE]
  info
}

#' Get document-level topic assignments as a data frame
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param metadata An optional named list of additional columns to append,
#'   each with \code{length(fit$docs)} elements.
#' @return A data frame with one row per document and columns
#'   \code{Document}, \code{Topic}, \code{Name}, \code{Top_words}.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   get_document_info(fit)
#' }
#' @export
get_document_info <- function(fit, metadata = NULL) {
  topics    <- fit$clusters
  names_vec <- vapply(topics, function(t) {
    fit$topic_labels[[as.character(t)]] %||% as.character(t)
  }, character(1))
  top_words <- vapply(topics, function(t) {
    if (t == -1L) return("")
    tt <- fit$topic_terms[fit$topic_terms$topic == t, ]
    tt <- tt[order(tt$rank), ]
    paste(tt$term[seq_len(min(5L, nrow(tt)))], collapse = ", ")
  }, character(1))
  info <- data.frame(
    Document  = fit$docs,
    Topic     = topics,
    Name      = names_vec,
    Top_words = top_words,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
  if (!is.null(metadata)) {
    for (nm in names(metadata)) info[[nm]] <- metadata[[nm]]
  }
  info
}

#' Get representative documents for one or all topics
#'
#' Representative documents are the three training documents whose embeddings
#' lie closest (cosine similarity) to the topic centroid.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param topic Optional integer topic ID. When \code{NULL} (default), returns
#'   a named list for all non-noise topics.
#' @return A character vector (single topic) or a named list of character
#'   vectors (all topics).
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   get_representative_docs(fit, topic = 0L)
#' }
#' @export
get_representative_docs <- function(fit, topic = NULL) {
  if (!is.null(topic))
    return(fit$representative_docs[[as.character(as.integer(topic))]])
  fit$representative_docs
}

#' Find topics most similar to a search term
#'
#' Ranks non-noise topics by the c-TF-IDF score of \code{search_term}.  If
#' the term is absent from the vocabulary entirely, falls back to substring
#' matching across each topic's top terms.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param search_term A single character string.
#' @param top_n Number of topics to return (default 5).
#' @return A data frame with columns \code{Topic}, \code{Name}, \code{Score},
#'   sorted by descending score.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(docs = abstracts, encoder = enc)
#'   find_topics(fit, "neural network")
#' }
#' @export
find_topics <- function(fit, search_term, top_n = 5L) {
  tt              <- fit$topic_terms
  topics_nonnoise <- sort(unique(tt$topic[tt$topic != -1L]))

  scores <- vapply(topics_nonnoise, function(t) {
    rows <- tt[tt$topic == t & tt$term == search_term, ]
    if (nrow(rows) > 0L) rows$score[[1L]] else 0
  }, numeric(1))

  if (all(scores == 0)) {
    scores <- vapply(topics_nonnoise, function(t) {
      rows <- tt[tt$topic == t, ]
      sum(grepl(search_term, rows$term, fixed = TRUE))
    }, numeric(1))
  }

  k   <- min(top_n, length(topics_nonnoise))
  ord <- order(scores, decreasing = TRUE)[seq_len(k)]
  data.frame(
    Topic = topics_nonnoise[ord],
    Name  = vapply(topics_nonnoise[ord], function(t) {
      fit$topic_labels[[as.character(t)]] %||% as.character(t)
    }, character(1)),
    Score = scores[ord],
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

#' Predict topics for new documents using a fitted BERTopic model
#'
#' Embeds new documents (or accepts pre-computed embeddings) and assigns each
#' to the nearest topic centroid by cosine similarity.  Noise (\code{-1}) is
#' never assigned as a target  --  all new documents receive a real topic.
#'
#' @param object A \code{bertopic_fit} from \code{\link{fit_bertopic}}.
#' @param new_docs Character vector of documents to predict.
#' @param encoder Optional encoder from \code{\link{load_hf_bert}}.  Required
#'   when \code{embeddings} is \code{NULL}.
#' @param embeddings Optional numeric matrix of pre-computed embeddings
#'   (\code{nrow = length(new_docs)}).  Overrides \code{encoder}.
#' @param ... Unused (for S3 generic compatibility).
#' @return A list with:
#'   \describe{
#'     \item{\code{topics}}{Integer vector of assigned topic IDs.}
#'     \item{\code{probabilities}}{Cosine similarity to the assigned centroid
#'       (proxy for confidence, in \code{[0, 1]} for normalised embeddings).}
#'     \item{\code{all_similarities}}{Numeric matrix
#'       (\code{nrow = length(new_docs)}, \code{ncol = n_topics}) of cosine
#'       similarities to every topic centroid.}
#'   }
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit  <- fit_bertopic(docs = abstracts, encoder = enc)
#'   pred <- predict(fit, new_docs = c("new paper about deep learning"),
#'                   encoder = enc)
#'   pred$topics
#' }
#' @export
predict.bertopic_fit <- function(object, new_docs, encoder = NULL,
                                  embeddings = NULL, ...) {
  if (is.null(embeddings)) {
    if (is.null(encoder))
      stop("Provide either 'encoder' or pre-computed 'embeddings'.")
    embeddings <- embed_texts(encoder, new_docs)
  }

  norms <- sqrt(rowSums(embeddings^2))
  norms[norms == 0] <- 1
  emb_n <- embeddings / norms

  centroids <- object$topic_centroids
  if (is.null(centroids) || nrow(centroids) == 0L)
    stop("No topic centroids in this fit. ",
         "All documents were likely assigned to noise (-1)  --  try lowering ",
         "hdbscan_min_pts or increasing the corpus size, then re-run fit_bertopic().")

  sim_mat <- emb_n %*% t(centroids)
  colnames(sim_mat) <- rownames(centroids)

  topic_ids <- as.integer(rownames(centroids))
  best_idx  <- max.col(sim_mat, ties.method = "first")
  topics    <- topic_ids[best_idx]
  probs     <- sim_mat[cbind(seq_len(nrow(sim_mat)), best_idx)]

  list(
    topics           = topics,
    probabilities    = probs,
    all_similarities = sim_mat
  )
}

#' Predict topics for new documents (standalone alias)
#'
#' Wraps \code{\link{predict.bertopic_fit}}; see that function for full
#' parameter documentation.
#'
#' @param fit A \code{bertopic_fit} from \code{\link{fit_bertopic}}.
#' @param new_docs Character vector of new documents.
#' @param encoder Optional encoder from \code{\link{load_hf_bert}}.
#' @param embeddings Optional pre-computed embedding matrix.
#' @return Same as \code{\link{predict.bertopic_fit}}.
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit  <- fit_bertopic(docs = abstracts, encoder = enc)
#'   pred <- transform_bertopic(fit, new_docs = c("new document"), encoder = enc)
#'   pred$topics
#' }
#' @export
transform_bertopic <- function(fit, new_docs, encoder = NULL,
                                embeddings = NULL) {
  predict.bertopic_fit(fit, new_docs, encoder = encoder,
                        embeddings = embeddings)
}
