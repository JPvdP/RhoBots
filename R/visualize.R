# =============================================================================
# visualize.R — Interactive visualizations for bertopic_fit objects.
# =============================================================================

# Short display label for any topic label format.
# Strips the leading "id_" prefix, replaces remaining underscores with spaces,
# and truncates to max_chars with an ellipsis if needed.
.format_topic_label <- function(label, max_chars = 30L, n_words = NULL) {
  text <- sub("^-?[0-9]+_", "", label)   # strip numeric id prefix
  text <- gsub("_", " ", text)           # underscores -> spaces
  text <- trimws(text)
  if (!is.null(n_words)) {
    words <- strsplit(text, "\\s+")[[1L]]
    text  <- paste(words[seq_len(min(n_words, length(words)))], collapse = " ")
  }
  if (nchar(text) > max_chars)
    text <- paste0(substr(text, 1L, max_chars - 1L), "…")
  text
}

#' Visualise documents in topic space
#'
#' Produces an interactive 2-D scatter plot (via \pkg{plotly}) with one point
#' per document, coloured by topic.  Topic centroids are annotated with short
#' labels.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param dims Either \code{NULL} (default) to use the pre-computed 2-D UMAP
#'   layout stored in \code{fit$layout2d}, or an integer vector of length 2
#'   selecting columns from \code{fit$reduced}, e.g. \code{c(1, 3)} to plot
#'   dimension 1 against dimension 3 of the clustering UMAP space.
#' @param label_topics Whether to annotate topic centroids with short labels
#'   (default \code{TRUE}).
#' @param n_label_words Number of topic words to show in centroid annotations
#'   (default 3).
#' @param point_size Marker size for document points (default 5).
#' @param noise_color Hex colour for unassigned (\code{-1}) documents
#'   (default \code{"#cccccc"}).
#' @param width,height Plot dimensions in pixels (defaults: 900 x 700).
#' @return A \code{plotly} figure object.
#' @export
#' Bar charts of top terms per topic
#'
#' Produces a grid of horizontal bar charts (via \pkg{plotly}), one panel per
#' topic, showing c-TF-IDF scores for the top \code{top_n} terms.  The best
#' term is always at the top of each panel.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param topics Integer vector of topic IDs to include.  \code{NULL} (default)
#'   shows all non-noise topics.
#' @param top_n Number of terms to show per topic (default 8).
#' @param n_cols Number of panel columns in the grid (default 4).
#' @param width,height Plot dimensions in pixels.  Auto-scaled to the grid
#'   size when \code{NULL}.
#' @return A \code{plotly} figure object.
#' @export
visualize_barchart <- function(fit,
                                topics = NULL,
                                top_n  = 8L,
                                n_cols = 4L,
                                width  = NULL,
                                height = NULL) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_barchart().")

  all_topics <- sort(setdiff(unique(fit$clusters), -1L))
  if (is.null(topics)) topics <- all_topics
  topics <- as.integer(topics)
  bad <- setdiff(topics, all_topics)
  if (length(bad))
    stop("Topics not found in this fit: ", paste(bad, collapse = ", "))
  if (!length(topics))
    stop("No non-noise topics to display.")

  n_t    <- length(topics)
  n_cols <- min(as.integer(n_cols), n_t)
  n_rows <- ceiling(n_t / n_cols)
  if (is.null(height)) height <- max(350L, 250L * n_rows)
  if (is.null(width))  width  <- max(400L, 260L * n_cols)

  pal  <- grDevices::hcl.colors(n_t, "Dynamic")
  cols <- stats::setNames(pal, as.character(topics))

  # One subplot per topic
  plots <- lapply(seq_along(topics), function(i) {
    t  <- topics[i]
    tt <- fit$topic_terms[fit$topic_terms$topic == t, ]
    tt <- tt[order(tt$rank), ][seq_len(min(top_n, nrow(tt))), ]

    plotly::plot_ly(
      x = tt$score, y = tt$term,
      type = "bar", orientation = "h",
      marker = list(color = cols[as.character(t)],
                    line  = list(color = "white", width = 0.4)),
      hovertemplate = "<b>%{y}</b>: %{x:.4f}<extra></extra>",
      showlegend = FALSE
    ) |>
      plotly::layout(
        xaxis = list(title = "c-TF-IDF", zeroline = FALSE,
                     showgrid = TRUE, gridcolor = "#e0e0e0",
                     tickfont = list(size = 9L)),
        yaxis = list(title = "",
                     categoryorder = "array",
                     categoryarray = rev(tt$term),
                     tickfont = list(size = 10L)),
        plot_bgcolor = "#f7f7f7"
      )
  })

  # Panel title annotations in paper coordinates
  annots <- lapply(seq_along(topics), function(i) {
    t     <- topics[i]
    row_i <- (i - 1L) %/% n_cols
    col_i <- (i - 1L) %% n_cols
    list(
      x         = (col_i + 0.5) / n_cols,
      y         = 1 - row_i / n_rows,
      xref      = "paper", yref = "paper",
      text      = paste0("<b>", .format_topic_label(
        fit$topic_labels[[as.character(t)]] %||% as.character(t)), "</b>"),
      showarrow = FALSE,
      xanchor   = "center", yanchor   = "bottom",
      font      = list(size = 11L)
    )
  })

  plotly::subplot(plots,
                  nrows  = n_rows,
                  shareX = FALSE, shareY = FALSE,
                  margin = 0.08) |>
    plotly::layout(
      width         = width,
      height        = height,
      showlegend    = FALSE,
      annotations   = annots,
      paper_bgcolor = "white"
    )
}

#' Visualise documents in topic space
#'
#' @export
visualize_topics <- function(fit,
                              dims            = NULL,
                              label_topics    = TRUE,
                              max_label_chars = 30L,
                              point_size      = 5L,
                              noise_color     = "#cccccc",
                              width           = 900L,
                              height          = 700L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_topics().")

  # --- Resolve coordinates -------------------------------------------------
  if (is.null(dims)) {
    if (is.null(fit$layout2d))
      stop("No 2-D layout found in fit. Re-run fit_bertopic().")
    coords     <- fit$layout2d
    dim_labels <- c("UMAP 1", "UMAP 2")
  } else {
    dims <- as.integer(dims)
    if (length(dims) != 2L)
      stop("'dims' must be a length-2 integer vector, e.g. c(1, 2).")
    if (any(dims < 1L) || any(dims > ncol(fit$reduced)))
      stop("'dims' values must be between 1 and ", ncol(fit$reduced),
           " (the number of columns in fit$reduced).")
    coords     <- fit$reduced[, dims, drop = FALSE]
    dim_labels <- paste0("UMAP dim ", dims)
  }

  # --- Document data -------------------------------------------------------
  topic_ids <- fit$clusters
  top_label <- vapply(topic_ids, function(t)
    .format_topic_label(fit$topic_labels[[as.character(t)]] %||%
                          as.character(t), max_chars = max_label_chars),
    character(1))

  hover <- paste0(
    "<b>", top_label, "</b><br>",
    "<span style='font-size:11px'>",
    gsub("(.{1,80})(\\s|$)", "\\1<br>", substr(fit$docs, 1L, 160L)),
    "</span>"
  )

  df <- data.frame(
    x = coords[, 1L], y = coords[, 2L],
    topic = topic_ids, label = top_label, hover = hover,
    stringsAsFactors = FALSE
  )

  # --- Colour palette ------------------------------------------------------
  topics_nonnoise <- sort(setdiff(unique(topic_ids), -1L))
  n_t <- length(topics_nonnoise)

  pal       <- if (n_t > 0L) grDevices::hcl.colors(n_t, "Dynamic") else character(0)
  topic_col <- stats::setNames(pal, as.character(topics_nonnoise))

  # --- Build figure --------------------------------------------------------
  p <- plotly::plot_ly(width = width, height = height)

  # Noise documents (grey, behind topic points)
  noise_rows <- df[df$topic == -1L, ]
  if (nrow(noise_rows) > 0L) {
    p <- plotly::add_trace(p,
      x = noise_rows$x, y = noise_rows$y,
      type = "scatter", mode = "markers",
      marker = list(color = noise_color, size = point_size - 1L,
                    opacity = 0.35),
      text = noise_rows$hover, hoverinfo = "text",
      name = "Noise (-1)"
    )
  }

  # One trace per topic (gives per-topic colour + legend entry)
  for (t in topics_nonnoise) {
    rows <- df[df$topic == t, ]
    col  <- topic_col[as.character(t)]
    lbl  <- .format_topic_label(
      fit$topic_labels[[as.character(t)]] %||% as.character(t),
      max_chars = max_label_chars
    )
    p <- plotly::add_trace(p,
      x = rows$x, y = rows$y,
      type = "scatter", mode = "markers",
      marker = list(color = col, size = point_size, opacity = 0.75,
                    line = list(color = "white", width = 0.4)),
      text = rows$hover, hoverinfo = "text",
      name = lbl
    )
  }

  # Centroid annotations
  if (label_topics && n_t > 0L) {
    cx <- vapply(topics_nonnoise, function(t) mean(df$x[df$topic == t]),
                 numeric(1))
    cy <- vapply(topics_nonnoise, function(t) mean(df$y[df$topic == t]),
                 numeric(1))
    cl <- vapply(topics_nonnoise, function(t)
      .format_topic_label(
        fit$topic_labels[[as.character(t)]] %||% as.character(t),
        max_chars = max_label_chars
      ), character(1))

    p <- plotly::add_annotations(p,
      x = cx, y = cy, text = cl,
      showarrow = FALSE,
      font     = list(size = 11L, color = "#1a1a1a"),
      bgcolor  = "rgba(255,255,255,0.78)",
      bordercolor = "rgba(80,80,80,0.25)",
      borderwidth = 1L, borderpad = 3L
    )
  }

  plotly::layout(p,
    xaxis  = list(title = dim_labels[1L], zeroline = FALSE, showgrid = FALSE),
    yaxis  = list(title = dim_labels[2L], zeroline = FALSE, showgrid = FALSE),
    legend = list(title = list(text = "<b>Topic</b>"),
                  itemsizing = "constant"),
    plot_bgcolor  = "#f7f7f7",
    paper_bgcolor = "white",
    margin = list(t = 30L, r = 20L)
  )
}

#' Visualise topic quality metrics
#'
#' Produces a four-panel interactive dashboard (via \pkg{plotly}) from a
#' \code{topic_quality} object returned by \code{\link{topic_quality}}.
#' The panels are:
#' \enumerate{
#'   \item \strong{Silhouette per topic} — horizontal bars coloured from red
#'     (negative) through yellow (zero) to green (positive).
#'   \item \strong{Topic size distribution} — document counts per topic, with
#'     the noise class shown separately.
#'   \item \strong{Centroid similarity} — pairwise cosine similarity matrix
#'     between topic centroids (lower = more distinct).
#'   \item \strong{Vocabulary overlap} — pairwise Jaccard similarity of the
#'     top-\eqn{N} c-TF-IDF term sets (lower = less overlap).
#' }
#'
#' @param q A \code{topic_quality} object from \code{\link{topic_quality}}.
#' @param fit Optional \code{bertopic_fit} used to resolve short topic labels.
#'   When \code{NULL} topics are labelled \code{T0}, \code{T1}, etc.
#' @param width,height Plot dimensions in pixels (defaults: 900 x 750).
#' @return A \code{plotly} figure object.
#' @export
visualize_quality <- function(q, fit = NULL, width = 900L, height = 750L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_quality().")
  if (!inherits(q, "topic_quality"))
    stop("'q' must be a topic_quality object from topic_quality().")

  # --- Topic labels -----------------------------------------------------------
  topics <- sort(as.integer(names(q$distribution$counts)))
  lbls <- if (!is.null(fit)) {
    vapply(topics, function(t)
      .format_topic_label(
        fit$topic_labels[[as.character(t)]] %||% as.character(t)
      ), character(1L))
  } else {
    paste0("T", topics)
  }
  t_chr <- as.character(topics)

  # --- Panel 1: Silhouette per topic ------------------------------------------
  sil_raw  <- as.numeric(q$silhouette$per_topic[t_chr])
  sil_cl   <- pmax(-1, pmin(1, sil_raw))
  col_fn   <- grDevices::colorRamp(c("#d73027", "#fee08b", "#1a9850"))
  sil_hex  <- apply(col_fn((sil_cl + 1) / 2), 1L, function(rgb)
    grDevices::rgb(rgb[1L], rgb[2L], rgb[3L], maxColorValue = 255))

  p_sil <- plotly::plot_ly(
    x = sil_raw, y = lbls, type = "bar", orientation = "h",
    marker = list(color = sil_hex),
    hovertemplate = "<b>%{y}</b><br>Silhouette: %{x:.3f}<extra></extra>",
    showlegend = FALSE
  ) |> plotly::layout(
    xaxis = list(title = "Silhouette score", range = c(-1, 1),
                 zeroline = TRUE, zerolinecolor = "#888888"),
    yaxis = list(title = "", categoryorder = "array",
                 categoryarray = rev(lbls), tickfont = list(size = 10L)),
    title       = list(text = "<b>Silhouette</b>", x = 0,
                       font = list(size = 12L)),
    plot_bgcolor = "#f7f7f7"
  )

  # --- Panel 2: Topic size distribution ---------------------------------------
  cnt_vals  <- as.integer(q$distribution$counts[t_chr])
  all_lbls  <- c(lbls, "Noise (-1)")
  all_cnts  <- c(cnt_vals, q$n_noise)
  all_cols  <- c(rep("#5b9bd5", length(lbls)), "#cccccc")

  p_dist <- plotly::plot_ly(
    x = all_cnts, y = all_lbls, type = "bar", orientation = "h",
    marker = list(color = all_cols),
    hovertemplate = "<b>%{y}</b><br>Documents: %{x}<extra></extra>",
    showlegend = FALSE
  ) |> plotly::layout(
    xaxis = list(title = "Documents"),
    yaxis = list(title = "", categoryorder = "total ascending",
                 tickfont = list(size = 10L)),
    title = list(text = "<b>Topic distribution</b>", x = 0,
                 font = list(size = 12L)),
    plot_bgcolor = "#f7f7f7"
  )

  # --- Shared heatmap helpers -------------------------------------------------
  rwr_scale <- list(
    list(c = 0,   color = "#053061"),
    list(c = 0.5, color = "#f7f7f7"),
    list(c = 1,   color = "#67001f")
  )
  # Convert to the list-of-lists format plotly expects
  cs <- lapply(rwr_scale, function(x) list(x$c, x$color))

  .heatmap <- function(mat, title_text, bar_title) {
    m <- mat
    diag(m) <- NA_real_          # suppress diagonal so scale focuses on off-diag
    plotly::plot_ly(
      z = m, x = lbls, y = lbls, type = "heatmap",
      colorscale  = cs, zmin = 0, zmax = 1,
      hovertemplate = "<b>%{x}</b> vs <b>%{y}</b><br>%{z:.3f}<extra></extra>",
      colorbar = list(title = bar_title, len = 0.45, thickness = 12L)
    ) |> plotly::layout(
      xaxis = list(title = "", tickangle = -40L, tickfont = list(size = 9L)),
      yaxis = list(title = "", tickfont = list(size = 9L),
                   autorange = "reversed"),
      title = list(text = title_text, x = 0, font = list(size = 12L))
    )
  }

  # --- Panel 3: Centroid similarity -------------------------------------------
  sim_mat <- q$separation$centroid_similarity[t_chr, t_chr, drop = FALSE]
  p_sim   <- .heatmap(sim_mat,
                      "<b>Centroid similarity</b>  (↓ better)",
                      "Cosine sim")

  # --- Panel 4: Jaccard overlap -----------------------------------------------
  jac_mat <- q$overlap$jaccard_matrix[t_chr, t_chr, drop = FALSE]
  p_jac   <- .heatmap(jac_mat,
                      "<b>Vocabulary overlap</b>  (↓ better)",
                      "Jaccard")

  # --- Combine ----------------------------------------------------------------
  plotly::subplot(p_sil, p_dist, p_sim, p_jac,
                  nrows = 2L, margin = 0.1,
                  shareX = FALSE, shareY = FALSE) |>
    plotly::layout(
      width         = width,
      height        = height,
      paper_bgcolor = "white",
      showlegend    = FALSE
    )
}
