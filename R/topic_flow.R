# =============================================================================
# topic_flow.R — Longitudinal topic tracking across time periods.
#
# fit_topics_over_time()   Run independent BERTopic models per period and
#                          align topics across adjacent periods by centroid
#                          cosine similarity.
# visualize_topic_flow()   Sankey diagram of topic transitions.
# =============================================================================

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

# Classify each topic's lifecycle status across periods.
.classify_topic_status <- function(topic_info, transitions, periods) {
  topic_info$status <- mapply(function(p, t) {
    p_idx   <- match(p, periods)
    is_first <- p_idx == 1L
    is_last  <- p_idx == length(periods)

    n_in  <- if (!is.null(transitions) && !is_first)
      sum(transitions$period_to == p & transitions$topic_to == t) else 0L
    n_out <- if (!is.null(transitions) && !is_last)
      sum(transitions$period_from == p & transitions$topic_from == t) else 0L

    if (is_first  && n_out == 0L) return("disappears")
    if (is_first  && n_out == 1L) return("continues")
    if (is_first  && n_out  > 1L) return("splits")
    if (is_last   && n_in  == 0L) return("emerges")
    if (is_last   && n_in  == 1L) return("continues")
    if (is_last   && n_in   > 1L) return("merges")
    if (n_in == 0L && n_out == 0L) return("isolated")
    if (n_in == 0L) return("emerges")
    if (n_out == 0L) return("disappears")
    if (n_in  > 1L) return("merges")
    if (n_out > 1L) return("splits")
    "continues"
  }, topic_info$period, topic_info$topic, SIMPLIFY = TRUE)

  topic_info
}

# -----------------------------------------------------------------------------
# fit_topics_over_time
# -----------------------------------------------------------------------------

#' Fit independent BERTopic models per time period and align topics
#'
#' All documents are embedded once using the shared encoder, then one
#' BERTopic model is fitted per time period using the appropriate slice of
#' embeddings.  Because every model lives in the same embedding space,
#' cosine similarity between topic centroids across periods is a meaningful
#' measure of topic continuity.
#'
#' Adjacent periods are compared by computing the full cosine-similarity
#' matrix between their topic centroids.  Any pair whose similarity exceeds
#' \code{min_similarity} becomes a directed link in the transition graph.
#' From that graph each topic is labelled:
#' \describe{
#'   \item{\code{emerges}}{No incoming link from the previous period.}
#'   \item{\code{disappears}}{No outgoing link to the next period.}
#'   \item{\code{continues}}{Exactly one incoming and one outgoing link.}
#'   \item{\code{splits}}{One incoming link, multiple outgoing links.}
#'   \item{\code{merges}}{Multiple incoming links, one outgoing link.}
#'   \item{\code{isolated}}{No links in either direction.}
#' }
#'
#' @param encoder An encoder from \code{\link{load_hf_bert}}.  Used to embed
#'   all documents once before running per-period models.
#' @param docs Character vector of all documents.
#' @param timestamps A vector of the same length as \code{docs} assigning
#'   each document to a time period.  Unique values (sorted) define the
#'   period order.
#' @param min_similarity Minimum cosine similarity between topic centroids in
#'   adjacent periods to count as a link (default \code{0.7}).
#' @param bertopic_params A named list of additional arguments passed to
#'   \code{\link{fit_bertopic}} for every period (e.g.
#'   \code{list(hdbscan_min_pts = 5, ngram_range = c(1L, 2L))}).
#' @param verbose Print progress messages (default \code{TRUE}).
#' @return An object of class \code{bertopic_flow} containing:
#'   \describe{
#'     \item{\code{periods}}{Character vector of period labels in order.}
#'     \item{\code{fits}}{Named list of \code{bertopic_fit} objects.}
#'     \item{\code{transitions}}{Data frame of inter-period topic links:
#'       \code{period_from}, \code{topic_from}, \code{period_to},
#'       \code{topic_to}, \code{similarity}.}
#'     \item{\code{topic_info}}{Data frame with one row per (period, topic):
#'       \code{period}, \code{topic}, \code{label}, \code{count},
#'       \code{status}.}
#'   }
#' @export
fit_topics_over_time <- function(encoder,
                                  docs,
                                  timestamps,
                                  min_similarity  = 0.7,
                                  bertopic_params = list(),
                                  verbose         = TRUE) {
  if (length(timestamps) != length(docs))
    stop("'timestamps' must have the same length as 'docs'.")

  periods   <- as.character(sort(unique(timestamps)))
  n_periods <- length(periods)
  if (n_periods < 2L) stop("Need at least 2 distinct time periods.")

  timestamps <- as.character(timestamps)

  # --- Embed everything once -----------------------------------------------
  if (verbose) message("Embedding all ", length(docs), " documents...")
  all_emb <- embed_texts(encoder, docs, verbose = FALSE)
  if (verbose) message("Done. Running BERTopic on ", n_periods, " periods.\n")

  # --- Fit one model per period --------------------------------------------
  fits <- stats::setNames(
    lapply(periods, function(p) {
      idx  <- which(timestamps == p)
      ndoc <- length(idx)
      if (verbose) message("Period '", p, "': ", ndoc, " documents")

      if (ndoc < 2L) {
        warning("Period '", p, "' has fewer than 2 documents — skipping.")
        return(NULL)
      }

      params <- c(
        list(encoder    = NULL,
             docs       = docs[idx],
             embeddings = all_emb[idx, , drop = FALSE],
             verbose    = FALSE),
        bertopic_params
      )

      tryCatch(
        do.call(fit_bertopic, params),
        error = function(e) {
          warning("Period '", p, "' model failed: ", conditionMessage(e))
          NULL
        }
      )
    }),
    periods
  )

  fits    <- Filter(Negate(is.null), fits)
  periods <- names(fits)
  if (length(periods) < 2L)
    stop("Fewer than 2 periods produced a valid model.")

  # --- Compute inter-period transitions ------------------------------------
  transitions <- do.call(rbind, lapply(seq_len(length(periods) - 1L), function(i) {
    p_from <- periods[i];      fit_from <- fits[[p_from]]
    p_to   <- periods[i + 1L]; fit_to   <- fits[[p_to]]

    t_from <- sort(setdiff(unique(fit_from$clusters), -1L))
    t_to   <- sort(setdiff(unique(fit_to$clusters),   -1L))
    if (length(t_from) == 0L || length(t_to) == 0L) return(NULL)

    cent_f  <- fit_from$topic_centroids[as.character(t_from), , drop = FALSE]
    cent_t  <- fit_to$topic_centroids[as.character(t_to),   , drop = FALSE]
    sim_mat <- cent_f %*% t(cent_t)         # topic_from × topic_to

    links <- which(sim_mat >= min_similarity, arr.ind = TRUE)
    if (nrow(links) == 0L) return(NULL)

    data.frame(
      period_from = p_from,
      topic_from  = t_from[links[, 1L]],
      period_to   = p_to,
      topic_to    = t_to[links[, 2L]],
      similarity  = round(sim_mat[links], 4L),
      stringsAsFactors = FALSE
    )
  }))
  if (!is.null(transitions)) rownames(transitions) <- NULL

  # --- Build topic_info with lifecycle status ------------------------------
  topic_info <- do.call(rbind, lapply(periods, function(p) {
    f      <- fits[[p]]
    topics <- sort(setdiff(unique(f$clusters), -1L))
    if (length(topics) == 0L) return(NULL)
    data.frame(
      period = p,
      topic  = topics,
      label  = vapply(topics, function(t)
        .format_topic_label(f$topic_labels[[as.character(t)]] %||%
                              as.character(t), n_words = 4L), character(1L)),
      count  = vapply(topics, function(t) sum(f$clusters == t), integer(1L)),
      status = NA_character_,
      stringsAsFactors = FALSE
    )
  }))
  if (!is.null(topic_info)) {
    rownames(topic_info) <- NULL
    topic_info <- .classify_topic_status(topic_info, transitions, periods)
  }

  if (verbose) {
    message("\nTransitions found: ",
            if (is.null(transitions)) 0L else nrow(transitions))
    if (!is.null(topic_info)) {
      tbl <- table(topic_info$status)
      message(paste(names(tbl), tbl, sep = ": ", collapse = " | "))
    }
  }

  structure(
    list(periods     = periods,
         fits        = fits,
         transitions = transitions,
         topic_info  = topic_info),
    class = "bertopic_flow"
  )
}

#' Print method for bertopic_flow objects
#' @param x A \code{bertopic_flow} object.
#' @param ... Unused.
#' @export
print.bertopic_flow <- function(x, ...) {
  cat("<bertopic_flow>\n")
  cat("  periods:    ", paste(x$periods, collapse = " → "), "\n")
  cat("  models:     ", length(x$fits), "\n")
  total_topics <- if (!is.null(x$topic_info)) nrow(x$topic_info) else 0L
  cat("  topics:     ", total_topics, "across all periods\n")
  total_links  <- if (!is.null(x$transitions)) nrow(x$transitions) else 0L
  cat("  links:      ", total_links, "(similarity ≥ threshold)\n")
  if (!is.null(x$topic_info)) {
    tbl <- sort(table(x$topic_info$status), decreasing = TRUE)
    cat("  status:     ",
        paste(names(tbl), tbl, sep = "=", collapse = "  "), "\n")
  }
  invisible(x)
}

# -----------------------------------------------------------------------------
# visualize_topic_flow
# -----------------------------------------------------------------------------

#' Sankey diagram of topic flow across periods
#'
#' Produces an interactive Plotly Sankey diagram where each column represents
#' one time period, each node is a topic (sized by document count), and each
#' link is a cross-period topic similarity above the fitted threshold.
#' Link thickness scales with \code{similarity × min(count_from, count_to)}.
#'
#' @param flow A \code{bertopic_flow} object from
#'   \code{\link{fit_topics_over_time}}.
#' @param periods Character vector of period labels to include.  \code{NULL}
#'   (default) shows all periods.
#' @param color_by One of \code{"period"} (all topics in the same period share
#'   a hue) or \code{"status"} (nodes coloured by lifecycle status).
#' @param width,height Plot dimensions in pixels.
#' @return A \code{plotly} figure.
#' @export
visualize_topic_flow <- function(flow,
                                  periods   = NULL,
                                  color_by  = c("period", "status"),
                                  width     = 1100L,
                                  height    = 650L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_topic_flow().")
  if (!inherits(flow, "bertopic_flow"))
    stop("'flow' must be a bertopic_flow object.")

  color_by <- match.arg(color_by)

  use_periods <- if (is.null(periods)) flow$periods else
    intersect(flow$periods, as.character(periods))
  if (length(use_periods) < 2L)
    stop("Need at least 2 periods to visualise flow.")

  ti   <- flow$topic_info[flow$topic_info$period %in% use_periods, ]
  tr   <- flow$transitions[
    flow$transitions$period_from %in% use_periods &
      flow$transitions$period_to   %in% use_periods, ]

  if (nrow(ti) == 0L) stop("No topics found for the selected periods.")

  # --- Assign a unique node index to each (period, topic) ------------------
  ti$node_id <- seq_len(nrow(ti)) - 1L    # 0-based for plotly

  node_lookup <- stats::setNames(
    ti$node_id,
    paste(ti$period, ti$topic, sep = "_")
  )

  # --- Node colours ---------------------------------------------------------
  n_periods <- length(use_periods)
  period_pal <- stats::setNames(
    grDevices::hcl.colors(n_periods, "Dynamic"),
    use_periods
  )

  status_pal <- c(
    continues  = "#4e79a7",
    emerges    = "#59a14f",
    disappears = "#e15759",
    splits     = "#f28e2b",
    merges     = "#b07aa1",
    isolated   = "#bab0ac"
  )

  node_colors <- if (color_by == "period") {
    period_pal[ti$period]
  } else {
    status_pal[ti$status]
  }

  # --- Node positions (x = period, y = rank within period) ------------------
  period_x <- stats::setNames(
    seq(0.01, 0.99, length.out = n_periods),
    use_periods
  )
  node_x <- period_x[ti$period]

  node_y <- unlist(lapply(use_periods, function(p) {
    rows <- ti[ti$period == p, ]
    rows <- rows[order(-rows$count), ]
    n    <- nrow(rows)
    if (n == 1L) return(stats::setNames(0.5, paste(rows$period, rows$topic, sep = "_")))
    ys   <- seq(0.05, 0.95, length.out = n)
    stats::setNames(ys, paste(rows$period, rows$topic, sep = "_"))
  }))
  node_y <- node_y[paste(ti$period, ti$topic, sep = "_")]

  # --- Node labels ----------------------------------------------------------
  node_labels <- paste0(ti$period, ": ", ti$label,
                         " (n=", ti$count, ")")

  # --- Links ----------------------------------------------------------------
  link_src <- link_tgt <- link_val <- link_col <- NULL

  if (!is.null(tr) && nrow(tr) > 0L) {
    from_key <- paste(tr$period_from, tr$topic_from, sep = "_")
    to_key   <- paste(tr$period_to,   tr$topic_to,   sep = "_")
    valid    <- from_key %in% names(node_lookup) & to_key %in% names(node_lookup)
    tr       <- tr[valid, ]

    if (nrow(tr) > 0L) {
      from_key <- paste(tr$period_from, tr$topic_from, sep = "_")
      to_key   <- paste(tr$period_to,   tr$topic_to,   sep = "_")

      count_from <- ti$count[match(from_key, paste(ti$period, ti$topic, sep = "_"))]
      count_to   <- ti$count[match(to_key,   paste(ti$period, ti$topic, sep = "_"))]

      link_src <- unname(node_lookup[from_key])
      link_tgt <- unname(node_lookup[to_key])
      link_val <- pmax(tr$similarity * pmin(count_from, count_to), 1L)
      src_col  <- node_colors[match(from_key, paste(ti$period, ti$topic, sep = "_"))]
      link_col <- gsub("^#(..)(..)(..)$",
                        "rgba(\\1,\\2,\\3,0.45)",
                        src_col)
      # Convert hex to rgba properly
      link_col <- vapply(src_col, function(hex) {
        rgb  <- grDevices::col2rgb(hex)
        sprintf("rgba(%d,%d,%d,0.45)", rgb[1], rgb[2], rgb[3])
      }, character(1L))
    }
  }

  # --- Build Sankey ---------------------------------------------------------
  plotly::plot_ly(
    type        = "sankey",
    orientation = "h",
    width       = width,
    height      = height,
    node = list(
      pad       = 12L,
      thickness = 18L,
      label     = node_labels,
      color     = node_colors,
      x         = node_x,
      y         = node_y
    ),
    link = list(
      source = link_src %||% integer(0),
      target = link_tgt %||% integer(0),
      value  = link_val %||% numeric(0),
      color  = link_col %||% character(0)
    )
  ) |>
    plotly::layout(
      title = list(text = "<b>Topic flow across periods</b>",
                   font = list(size = 14L)),
      font  = list(size = 11L),
      paper_bgcolor = "white"
    )
}
