# =============================================================================
# topics_over_time.R  --  Compute and visualise how topic representations
# evolve across timestamps.
# =============================================================================

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

# Full vocabulary c-TF-IDF matrix for a DTM subset.
# Returns list(matrix = KxV numeric, frequency = named integer).
.full_ctfidf <- function(dtm, cluster_ids, topics) {
  K    <- length(topics)
  vocab <- colnames(dtm)

  agg <- Matrix::Matrix(0.0, nrow = K, ncol = ncol(dtm),
                         dimnames = list(as.character(topics), vocab))
  for (k in seq_len(K)) {
    idx <- which(cluster_ids == topics[k])
    if (length(idx) > 0L)
      agg[k, ] <- Matrix::colSums(dtm[idx, , drop = FALSE])
  }

  class_totals <- pmax(Matrix::rowSums(agg), 1.0)
  tf   <- agg / class_totals
  A    <- mean(class_totals)
  docf <- Matrix::colSums(agg > 0)
  idf  <- log(1.0 + A / pmax(docf, 1.0))
  mat  <- as.matrix(tf) * matrix(idf, nrow = K, ncol = ncol(dtm), byrow = TRUE)

  list(
    matrix    = mat,
    frequency = stats::setNames(
      vapply(topics, function(t) sum(cluster_ids == t), integer(1L)),
      as.character(topics)
    )
  )
}

# L1-normalise rows of a matrix.
.l1_normalize <- function(mat) {
  rs <- rowSums(abs(mat))
  rs[rs == 0] <- 1
  mat / rs
}

# Bin a vector of timestamps into nr_bins groups.
# Returns a vector of the same type as the input whose values are the
# per-bin median, so the output stays in Date / numeric space.
.bin_timestamps <- function(timestamps, nr_bins) {
  nr_bins  <- as.integer(nr_bins)
  is_date  <- inherits(timestamps, "Date")
  is_posix <- inherits(timestamps, c("POSIXct", "POSIXlt"))
  ts_num   <- as.numeric(timestamps)

  breaks  <- seq(min(ts_num), max(ts_num), length.out = nr_bins + 1L)
  bin_idx <- findInterval(ts_num, breaks, rightmost.closed = TRUE)

  reps <- vapply(seq_len(nr_bins), function(b) {
    vals <- ts_num[bin_idx == b]
    if (length(vals) == 0L) NA_real_ else stats::median(vals)
  }, numeric(1L))

  result <- reps[bin_idx]
  if (is_date)  return(as.Date(result,    origin = "1970-01-01"))
  if (is_posix) return(as.POSIXct(result, origin = "1970-01-01"))
  result
}

# -----------------------------------------------------------------------------
# topics_over_time
# -----------------------------------------------------------------------------

#' Compute how topic representations change over time
#'
#' For each unique timestamp (or time bin), the documents assigned to each
#' topic are collected and a local c-TF-IDF representation is computed.
#' Two optional smoothing passes mirror Python BERTopic's behaviour:
#'
#' \describe{
#'   \item{\code{evolution_tuning}}{Averages each timestamp's representation
#'     with the previous timestamp (L1-normalised) to smooth abrupt changes.}
#'   \item{\code{global_tuning}}{Averages each local representation with the
#'     global c-TF-IDF (computed over all documents) so words that are absent
#'     from a narrow time window are not completely lost.}
#' }
#'
#' @param fit A \code{bertopic_fit} from \code{\link{fit_bertopic}}.
#' @param timestamps A vector of the same length as \code{fit$docs} giving
#'   each document a time label.  Can be numeric, \code{Date}, \code{POSIXct},
#'   or character.
#' @param nr_bins Optional integer.  Bin the timestamps into this many equally
#'   spaced intervals (using the per-bin median as the representative label).
#' @param evolution_tuning Smooth representations across adjacent timestamps
#'   (default \code{TRUE}).
#' @param global_tuning Blend each local representation with the global
#'   c-TF-IDF (default \code{TRUE}).
#' @param top_n Number of top terms to include per (topic, timestamp) row.
#'   Defaults to \code{fit$top_n_terms}.
#' @return A data frame (class \code{topics_over_time}) with columns
#'   \code{Topic}, \code{Words}, \code{Frequency}, \code{Timestamp},
#'   sorted by \code{Timestamp} then \code{Topic}.
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit  <- fit_bertopic(docs = abstracts, encoder = enc)
#'   years <- as.Date(paste0(sample(2015:2023, length(abstracts), replace = TRUE), "-01-01"))
#'   tot  <- topics_over_time(fit, timestamps = years)
#'   visualize_topics_over_time(tot, fit = fit)
#' }
#' @export
topics_over_time <- function(fit,
                              timestamps,
                              nr_bins          = NULL,
                              evolution_tuning = TRUE,
                              global_tuning    = TRUE,
                              top_n            = NULL) {
  n_docs <- length(fit$docs)
  if (length(timestamps) != n_docs)
    stop("'timestamps' must have the same length as fit$docs (", n_docs, ").")

  if (is.null(top_n)) top_n <- fit$top_n_terms

  topics_nonnoise <- sort(setdiff(unique(fit$clusters), -1L))
  if (length(topics_nonnoise) == 0L)
    stop("No non-noise topics in this fit.")

  if (!is.null(nr_bins)) timestamps <- .bin_timestamps(timestamps, nr_bins)

  unique_ts <- sort(unique(timestamps))
  vocab     <- colnames(fit$dtm)

  # --- Global c-TF-IDF (all non-noise documents) ---------------------------
  non_noise  <- fit$clusters != -1L
  global_ctf <- .full_ctfidf(
    fit$dtm[non_noise, , drop = FALSE],
    fit$clusters[non_noise],
    topics_nonnoise
  )
  global_mat <- .l1_normalize(global_ctf$matrix)

  # --- Per-timestamp c-TF-IDF ----------------------------------------------
  ts_data <- lapply(unique_ts, function(ts) {
    idx <- which(timestamps == ts & fit$clusters != -1L)
    if (length(idx) == 0L) return(NULL)
    ts_topics <- sort(unique(fit$clusters[idx]))
    ctf <- .full_ctfidf(fit$dtm[idx, , drop = FALSE],
                         fit$clusters[idx], ts_topics)
    ctf$matrix <- .l1_normalize(ctf$matrix)
    list(ts = ts, ctf = ctf, topics = ts_topics)
  })
  ts_data <- Filter(Negate(is.null), ts_data)

  if (length(ts_data) == 0L)
    stop("No non-noise documents found for any timestamp.")

  # --- Evolution tuning: average with previous timestamp -------------------
  if (evolution_tuning && length(ts_data) > 1L) {
    for (i in seq(2L, length(ts_data))) {
      shared <- intersect(ts_data[[i]]$topics, ts_data[[i - 1L]]$topics)
      for (t in shared) {
        tc <- as.character(t)
        if (tc %in% rownames(ts_data[[i - 1L]]$ctf$matrix)) {
          ts_data[[i]]$ctf$matrix[tc, ] <-
            (ts_data[[i]]$ctf$matrix[tc, ] +
               ts_data[[i - 1L]]$ctf$matrix[tc, ]) / 2.0
        }
      }
    }
  }

  # --- Global tuning: average with global representation -------------------
  if (global_tuning) {
    for (i in seq_along(ts_data)) {
      for (t in ts_data[[i]]$topics) {
        tc <- as.character(t)
        if (tc %in% rownames(global_mat)) {
          ts_data[[i]]$ctf$matrix[tc, ] <-
            (ts_data[[i]]$ctf$matrix[tc, ] + global_mat[tc, ]) / 2.0
        }
      }
    }
  }

  # --- Build output --------------------------------------------------------
  rows <- do.call(rbind, lapply(ts_data, function(entry) {
    do.call(rbind, lapply(entry$topics, function(t) {
      tc     <- as.character(t)
      scores <- entry$ctf$matrix[tc, ]
      top_i  <- order(scores, decreasing = TRUE)[seq_len(min(top_n, length(vocab)))]
      data.frame(
        Topic     = t,
        Words     = paste(vocab[top_i], collapse = ", "),
        Frequency = entry$ctf$frequency[[tc]],
        Timestamp = entry$ts,
        stringsAsFactors = FALSE
      )
    }))
  }))

  rownames(rows) <- NULL
  out <- rows[order(rows$Timestamp, rows$Topic), ]
  class(out) <- c("topics_over_time", "data.frame")
  out
}

# -----------------------------------------------------------------------------
# visualize_topics_over_time
# -----------------------------------------------------------------------------

#' Visualise topic frequency over time
#'
#' Produces an interactive Plotly line chart showing how the relative frequency
#' of each topic changes across timestamps.  Hovering over a point shows the
#' topic's top words at that time.
#'
#' @param tot A \code{topics_over_time} data frame from
#'   \code{\link{topics_over_time}}.
#' @param topics Integer vector of topic IDs to include.  \code{NULL} (default)
#'   shows the 10 most frequent topics overall.
#' @param normalize If \code{TRUE} (default), the Y-axis shows each topic's
#'   share of documents in that time period.  If \code{FALSE}, raw document
#'   counts are shown.
#' @param width,height Plot dimensions in pixels.
#' @return A \code{plotly} figure.
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit  <- fit_bertopic(docs = abstracts, encoder = enc)
#'   years <- as.Date(paste0(sample(2015:2023, length(abstracts), replace = TRUE), "-01-01"))
#'   tot  <- topics_over_time(fit, timestamps = years)
#'   visualize_topics_over_time(tot, fit = fit)
#' }
#' @export
visualize_topics_over_time <- function(tot,
                                        topics    = NULL,
                                        normalize = TRUE,
                                        width     = 900L,
                                        height    = 550L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_topics_over_time().")
  if (!is.data.frame(tot) ||
      !all(c("Topic", "Words", "Frequency", "Timestamp") %in% names(tot)))
    stop("'tot' must be the output of topics_over_time().")

  # Total non-noise docs per timestamp (computed before any topic filter)
  ts_totals <- tapply(tot$Frequency, tot$Timestamp, sum)

  # Select topics
  if (!is.null(topics)) {
    tot <- tot[tot$Topic %in% as.integer(topics), ]
  } else {
    freq_by_topic <- sort(tapply(tot$Frequency, tot$Topic, sum), decreasing = TRUE)
    keep <- as.integer(names(freq_by_topic)[seq_len(min(10L, length(freq_by_topic)))])
    tot  <- tot[tot$Topic %in% keep, ]
  }
  if (nrow(tot) == 0L) stop("No data to display after topic filtering.")

  if (normalize) {
    tot$y_val <- tot$Frequency / ts_totals[as.character(tot$Timestamp)]
    y_title   <- "Share of documents"
    tick_fmt  <- ".0%"
  } else {
    tot$y_val <- tot$Frequency
    y_title   <- "Number of documents"
    tick_fmt  <- NULL
  }

  all_topics <- sort(unique(tot$Topic))
  pal        <- grDevices::hcl.colors(length(all_topics), "Dynamic")
  topic_col  <- stats::setNames(pal, as.character(all_topics))

  p <- plotly::plot_ly(width = width, height = height)

  for (t in all_topics) {
    rows  <- tot[tot$Topic == t, ]
    rows  <- rows[order(rows$Timestamp), ]
    label <- paste0("Topic ", t)

    hover <- paste0(
      "<b>", label, "</b>  --  ", rows$Timestamp, "<br>",
      "Docs: ", rows$Frequency, "<br>",
      "<i>", substr(rows$Words, 1L, 100L), "</i>"
    )

    p <- plotly::add_trace(p,
      x    = rows$Timestamp,
      y    = rows$y_val,
      type = "scatter", mode = "lines+markers",
      name = label,
      line   = list(color = topic_col[as.character(t)], width = 2L),
      marker = list(color = topic_col[as.character(t)], size  = 6L),
      text      = hover,
      hoverinfo = "text"
    )
  }

  y_axis <- list(title = y_title, zeroline = FALSE)
  if (!is.null(tick_fmt)) y_axis$tickformat <- tick_fmt

  plotly::layout(p,
    xaxis         = list(title = "Timestamp"),
    yaxis         = y_axis,
    legend        = list(title = list(text = "<b>Topic</b>")),
    plot_bgcolor  = "#f7f7f7",
    paper_bgcolor = "white"
  )
}
