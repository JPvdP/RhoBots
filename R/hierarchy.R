# =============================================================================
# hierarchy.R — Hierarchical topic modelling via agglomerative merging.
#
# Builds a bottom-up merge tree from a fitted bertopic_fit using the
# cosine distance between topic centroids.  At each internal node the
# function computes merged c-TF-IDF terms so every level of the tree is
# interpretable.
# =============================================================================

#' Build a hierarchical topic tree from a fitted topic model
#'
#' Performs agglomerative clustering on the L2-normalised topic centroids
#' stored in `fit$topic_centroids` using cosine distance.  At each merge
#' node the function pools the c-TF-IDF scores of all constituent topics and
#' records the top terms, so every level of the hierarchy is labelled.
#'
#' The result can be visualised with [visualize_hierarchy()].
#'
#' @param fit A `bertopic_fit` object from [fit_bertopic()].
#' @param method Linkage method passed to [stats::hclust()].  `"ward.D2"`
#'   (default) tends to produce compact, balanced clusters.  Other good
#'   choices: `"average"` (UPGMA), `"complete"`.
#' @param top_n_terms Number of top terms to record at each internal node.
#' @return A list of class `hierarchical_topics` with elements:
#'   \describe{
#'     \item{`hclust`}{the `hclust` object (for use with base R dendrogram
#'       functions if desired)}
#'     \item{`merge_df`}{a data frame with one row per merge, recording
#'       `parent_id`, `child_left`, `child_right`, `distance` (cosine),
#'       `topics` (comma-separated topic IDs), and `terms`}
#'     \item{`topic_ids`}{integer vector of leaf topic IDs in original order}
#'     \item{`method`}{the linkage method used}
#'   }
#' @export
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   fit <- fit_bertopic(enc, docs = abstracts)
#'   h   <- hierarchical_topics(fit)
#'   print(h)
#'   visualize_hierarchy(h, fit = fit)
#' }
hierarchical_topics <- function(fit, method = "ward.D2", top_n_terms = 10L) {
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object from fit_bertopic().")

  centroids <- fit$topic_centroids
  if (is.null(centroids) || nrow(centroids) < 2L)
    stop("Need at least 2 non-noise topics to build a hierarchy. ",
         "The fit has ", nrow(centroids %||% matrix()), " topics.")

  topics    <- as.integer(rownames(centroids))
  n_topics  <- length(topics)

  # Cosine distance: centroids are already L2-normalised in fit_bertopic
  sim_mat  <- centroids %*% t(centroids)
  sim_mat  <- pmin(pmax(sim_mat, -1), 1)   # numerical safety
  dist_mat <- stats::as.dist(1 - sim_mat)

  hc         <- stats::hclust(dist_mat, method = method)
  hc$labels  <- as.character(topics)

  n_merges  <- n_topics - 1L
  n_nodes   <- n_topics + n_merges  # leaves + internal nodes

  # node_topics[[i]] = character vector of original topic IDs under that node
  node_topics <- vector("list", n_nodes)
  for (i in seq_along(topics))
    node_topics[[i]] <- as.character(topics[i])

  merge_df <- data.frame(
    parent_id   = integer(n_merges),
    child_left  = integer(n_merges),
    child_right = integer(n_merges),
    distance    = numeric(n_merges),
    topics      = character(n_merges),
    terms       = character(n_merges),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(n_merges)) {
    left  <- hc$merge[i, 1L]
    right <- hc$merge[i, 2L]

    # hclust merge convention: negative = leaf index, positive = prior merge
    lnode <- if (left  < 0L) -left          else n_topics + left
    rnode <- if (right < 0L) -right         else n_topics + right
    pnode <- n_topics + i

    merged <- c(node_topics[[lnode]], node_topics[[rnode]])
    node_topics[[pnode]] <- merged

    # Pool c-TF-IDF scores across the merged topics: sum per term
    merged_ids <- as.integer(merged)
    tt  <- fit$topic_terms[fit$topic_terms$topic %in% merged_ids, ]
    if (nrow(tt) > 0L) {
      scores  <- tapply(tt$score, tt$term, sum)
      top_trms <- names(sort(scores, decreasing = TRUE))[
        seq_len(min(as.integer(top_n_terms), length(scores)))]
    } else {
      top_trms <- character(0L)
    }

    merge_df[i, ] <- list(
      parent_id   = pnode,
      child_left  = lnode,
      child_right = rnode,
      distance    = hc$height[i],
      topics      = paste(merged, collapse = ", "),
      terms       = paste(top_trms, collapse = ", ")
    )
  }

  structure(
    list(hclust      = hc,
         merge_df    = merge_df,
         node_topics = node_topics,
         topic_ids   = topics,
         method      = method,
         n_topics    = n_topics),
    class = c("hierarchical_topics", "list")
  )
}

#' @export
print.hierarchical_topics <- function(x, ...) {
  cat("<hierarchical_topics>\n")
  cat("  topics:  ", x$n_topics, "\n")
  cat("  merges:  ", x$n_topics - 1L, "\n")
  cat("  method:  ", x$method, "\n")
  cat("\n")
  cat("  Top-level merge (root):\n")
  last <- x$merge_df[nrow(x$merge_df), ]
  cat("    topics: ", last$topics, "\n")
  cat("    terms:  ", last$terms, "\n")
  invisible(x)
}

#' Visualise the hierarchical topic tree as a dendrogram
#'
#' Produces an interactive plotly dendrogram with:
#' \itemize{
#'   \item Leaf labels: topic ID + top c-TF-IDF words (from `fit`)
#'   \item Internal node hover: merged topics and their pooled terms
#'   \item Height axis: cosine distance at which topics were merged
#' }
#'
#' @param h A `hierarchical_topics` object from [hierarchical_topics()].
#' @param fit Optional `bertopic_fit` — when supplied, leaf labels show topic
#'   words; when `NULL`, leaves are labelled by topic ID only.
#' @param n_label_words Number of topic words to show per leaf label.
#' @param width,height Plot dimensions in pixels.
#' @return A `plotly` figure object.
#' @export
visualize_hierarchy <- function(h, fit = NULL,
                                 n_label_words = 3L,
                                 width = 900L, height = 600L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_hierarchy().")
  if (!inherits(h, "hierarchical_topics"))
    stop("'h' must be a hierarchical_topics object from hierarchical_topics().")

  hc       <- h$hclust
  topics   <- h$topic_ids
  n_topics <- h$n_topics
  n_merges <- n_topics - 1L
  n_nodes  <- n_topics + n_merges

  # --- Leaf x-positions (based on hclust leaf ordering) ----------------------
  node_x <- numeric(n_nodes)
  node_y <- numeric(n_nodes)

  # hc$order: indices of leaves in left-to-right dendrogram order
  for (rank in seq_len(n_topics))
    node_x[hc$order[rank]] <- rank

  # Internal nodes: y = merge height
  for (i in seq_len(n_merges))
    node_y[n_topics + i] <- hc$height[i]

  # Compute internal node x-positions (midpoint of children)
  for (i in seq_len(n_merges)) {
    left  <- hc$merge[i, 1L]
    right <- hc$merge[i, 2L]
    lnode <- if (left  < 0L) -left        else n_topics + left
    rnode <- if (right < 0L) -right       else n_topics + right
    pnode <- n_topics + i
    node_x[pnode] <- (node_x[lnode] + node_x[rnode]) / 2
  }

  # --- Build dendrogram segments --------------------------------------------
  x0 <- numeric(0); y0 <- numeric(0)
  x1 <- numeric(0); y1 <- numeric(0)
  hover_txt <- character(0)

  for (i in seq_len(n_merges)) {
    left  <- hc$merge[i, 1L]
    right <- hc$merge[i, 2L]
    lnode <- if (left  < 0L) -left        else n_topics + left
    rnode <- if (right < 0L) -right       else n_topics + right
    pnode <- n_topics + i

    px <- node_x[pnode]
    py <- node_y[pnode]
    lx <- node_x[lnode]; ly <- node_y[lnode]
    rx <- node_x[rnode]; ry <- node_y[rnode]

    row <- h$merge_df[i, ]
    htxt <- paste0(
      "<b>Merged topics:</b> ", row$topics, "<br>",
      "<b>Distance:</b> ", round(row$distance, 4), "<br>",
      "<b>Terms:</b> ", row$terms
    )

    # Left vertical segment
    x0 <- c(x0, lx); y0 <- c(y0, ly); x1 <- c(x1, lx); y1 <- c(y1, py)
    hover_txt <- c(hover_txt, htxt)
    # Right vertical segment
    x0 <- c(x0, rx); y0 <- c(y0, ry); x1 <- c(x1, rx); y1 <- c(y1, py)
    hover_txt <- c(hover_txt, htxt)
    # Horizontal connector
    x0 <- c(x0, lx); y0 <- c(y0, py); x1 <- c(x1, rx); y1 <- c(y1, py)
    hover_txt <- c(hover_txt, htxt)
  }

  # --- Leaf labels -----------------------------------------------------------
  leaf_order <- hc$order  # indices into topics[], 1-based
  leaf_labels <- vapply(seq_len(n_topics), function(rank) {
    leaf_idx  <- leaf_order[rank]
    topic_id  <- topics[leaf_idx]
    if (!is.null(fit)) {
      lbl <- fit$topic_labels[[as.character(topic_id)]] %||% as.character(topic_id)
      parts <- strsplit(lbl, "_")[[1L]]
      words <- parts[seq(2L, min(n_label_words + 1L, length(parts)))]
      paste0(parts[1L], ": ", paste(words, collapse = ", "))
    } else {
      paste0("T", topic_id)
    }
  }, character(1L))

  leaf_x <- node_x[leaf_order]

  # --- Build plotly figure --------------------------------------------------
  # Segments as a scatter trace (mode = "lines")
  seg_x <- as.vector(rbind(x0, x1, NA))
  seg_y <- as.vector(rbind(y0, y1, NA))

  p <- plotly::plot_ly(width = width, height = height) |>
    plotly::add_trace(
      x = seg_x, y = seg_y,
      type = "scatter", mode = "lines",
      line = list(color = "#4472c4", width = 1.5),
      hoverinfo = "none",
      showlegend = FALSE
    ) |>
    plotly::add_trace(
      x = node_x[n_topics + seq_len(n_merges)],
      y = node_y[n_topics + seq_len(n_merges)],
      type = "scatter", mode = "markers",
      marker = list(color = "#4472c4", size = 6L,
                    line = list(color = "white", width = 1L)),
      text = h$merge_df$terms,
      hovertemplate = paste0(
        "<b>Topics:</b> %{customdata}<br>",
        "<b>Distance:</b> %{y:.4f}<br>",
        "<b>Terms:</b> %{text}<extra></extra>"
      ),
      customdata = h$merge_df$topics,
      showlegend = FALSE
    ) |>
    plotly::layout(
      xaxis = list(
        tickvals  = seq_len(n_topics),
        ticktext  = leaf_labels,
        tickangle = -45L,
        tickfont  = list(size = 10L),
        title     = "",
        zeroline  = FALSE,
        showgrid  = FALSE
      ),
      yaxis = list(
        title    = "Cosine distance",
        zeroline = FALSE,
        showgrid = TRUE,
        gridcolor = "#e0e0e0"
      ),
      plot_bgcolor  = "#f7f7f7",
      paper_bgcolor = "white",
      margin = list(b = 120L, t = 30L, l = 60L, r = 20L),
      showlegend = FALSE
    )
  p
}
