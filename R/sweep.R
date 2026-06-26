# =============================================================================
# sweep.R — Parameter sweep over BERTopic hyperparameters
# =============================================================================

# Internal: resolve encoders + embeddings into a named list of embedding matrices.
.resolve_emb_list <- function(docs, encoders, embeddings, verbose) {
  # Normalise singles to named lists
  if (!is.null(encoders) && !is.list(encoders))
    encoders  <- list(encoder = encoders)
  if (!is.null(embeddings) && is.matrix(embeddings))
    embeddings <- list(encoder = embeddings)

  all_names <- union(names(encoders), names(embeddings))
  if (length(all_names) == 0L)
    stop(
      "Provide at least one encoder or pre-computed embedding matrix.\n",
      "Use named lists when supplying multiple: list(minilm = enc1, scibert = enc2)"
    )

  result <- vector("list", length(all_names))
  names(result) <- all_names
  for (nm in all_names) {
    if (!is.null(embeddings) && nm %in% names(embeddings)) {
      result[[nm]] <- embeddings[[nm]]
    } else {
      if (verbose) message("Embedding with '", nm, "'...")
      result[[nm]] <- embed_texts(encoders[[nm]], docs, normalize = TRUE)
    }
  }
  result
}

#' Sweep BERTopic hyperparameters and compare topic quality
#'
#' Runs \code{\link{fit_bertopic}} and \code{\link{topic_quality}} over a full
#' factorial grid of UMAP and HDBSCAN parameters (and optionally multiple
#' embedding models), then returns a tidy data frame of quality metrics for
#' every combination.  Use \code{\link{visualize_sweep}} to compare runs
#' visually.
#'
#' Embeddings are the expensive step.  When sweeping only UMAP/HDBSCAN
#' parameters with a single encoder, pass pre-computed \code{embeddings} so
#' the encoder runs only once:
#' \preformatted{
#'   emb  <- embed_texts(enc, docs, normalize = TRUE)
#'   sw   <- sweep_topics(docs, embeddings = emb, min_pts = c(5, 10, 20))
#' }
#'
#' To compare multiple encoders, pass a named list of either encoders or
#' pre-computed matrices:
#' \preformatted{
#'   sw <- sweep_topics(docs,
#'     embeddings = list(minilm = emb1, scibert = emb2),
#'     n_neighbors = c(5, 15), min_pts = c(5, 10))
#' }
#'
#' @param docs Character vector of documents.
#' @param encoders A single encoder (from \code{\link{load_hf_bert}}) or a
#'   named list of encoders.  Ignored for any model whose embeddings are
#'   supplied via \code{embeddings}.
#' @param embeddings A pre-computed numeric matrix (rows = documents, columns =
#'   embedding dimensions), or a named list of such matrices when comparing
#'   multiple embedding models.  Names must match those in \code{encoders} when
#'   both are supplied.
#' @param n_neighbors Integer vector of UMAP \code{n_neighbors} values to try.
#' @param n_components Integer vector of UMAP \code{n_components} values to try.
#' @param min_pts Integer vector of HDBSCAN \code{min_pts} values to try.
#' @param ngram_range,top_n_terms,extra_stopwords Fixed model parameters passed
#'   to \code{\link{fit_bertopic}} for every run.
#' @param sample_size If not \code{NULL}, draw a random sample of this many
#'   documents before sweeping.  Useful for fast exploration on large corpora.
#' @param quality_top_n Passed to \code{\link{topic_quality}} as \code{top_n}.
#' @param quality_sample Passed to \code{\link{topic_quality}} as
#'   \code{sample_size} for the silhouette computation. Default 500.
#' @param seed Random seed for reproducibility.
#' @param verbose Print one-line progress per combination.
#' @return A list of class \code{topic_sweep} with elements:
#' \describe{
#'   \item{\code{results}}{Data frame with one row per parameter combination
#'     and columns for all swept parameters plus quality metrics.}
#'   \item{\code{best}}{The row of \code{results} with the highest silhouette
#'     score (ignoring failed runs).}
#'   \item{\code{n_docs}}{Number of documents used (after optional sampling).}
#'   \item{\code{sampled}}{Logical: whether a random sample was drawn.}
#'   \item{\code{param_names}}{Character vector of swept parameter names.}
#' }
#' @export
sweep_topics <- function(docs,
                          encoders        = NULL,
                          embeddings      = NULL,
                          n_neighbors     = c(5L, 15L, 30L),
                          n_components    = c(5L, 10L),
                          min_pts         = c(5L, 10L, 20L),
                          ngram_range     = c(1L, 1L),
                          top_n_terms     = 10L,
                          extra_stopwords = NULL,
                          sample_size     = NULL,
                          quality_top_n   = 10L,
                          quality_sample  = 500L,
                          seed            = 42L,
                          verbose         = TRUE) {

  if (is.null(encoders) && is.null(embeddings))
    stop("Provide 'encoders' or pre-computed 'embeddings'.")

  # --- Optional sampling ------------------------------------------------------
  n_total <- length(docs)
  sampled <- !is.null(sample_size) && n_total > sample_size
  if (sampled) {
    set.seed(seed)
    idx  <- sample(n_total, sample_size)
    docs <- docs[idx]
    if (is.matrix(embeddings))
      embeddings <- embeddings[idx, , drop = FALSE]
    if (is.list(embeddings))
      embeddings <- lapply(embeddings, function(m) m[idx, , drop = FALSE])
  }
  n_docs <- length(docs)
  if (verbose) message("Sweeping on ", n_docs, " documents",
                        if (sampled) paste0(" (sampled from ", n_total, ")") else "")

  # --- Embed once per encoder -------------------------------------------------
  emb_list <- .resolve_emb_list(docs, encoders, embeddings, verbose)

  # --- Build parameter grid ---------------------------------------------------
  n_neighbors  <- as.integer(n_neighbors)
  n_components <- as.integer(n_components)
  min_pts      <- as.integer(min_pts)

  grid <- expand.grid(
    model        = names(emb_list),
    n_neighbors  = n_neighbors,
    n_components = n_components,
    min_pts      = min_pts,
    stringsAsFactors = FALSE
  )
  n_runs <- nrow(grid)
  if (verbose) message("Running ", n_runs, " combinations...")

  # Determine which params are actually being swept (> 1 unique value)
  param_swept <- c(
    if (length(names(emb_list)) > 1L)  "model"        else character(0),
    if (length(n_neighbors)     > 1L)  "n_neighbors"  else character(0),
    if (length(n_components)    > 1L)  "n_components" else character(0),
    if (length(min_pts)         > 1L)  "min_pts"      else character(0)
  )

  # --- Run each combination ---------------------------------------------------
  run_results <- vector("list", n_runs)

  for (i in seq_len(n_runs)) {
    row <- grid[i, ]
    if (verbose) {
      message(sprintf("  [%d/%d] %-12s n_neighbors=%-3d n_components=%-3d min_pts=%d",
                      i, n_runs, row$model,
                      row$n_neighbors, row$n_components, row$min_pts))
    }

    run_results[[i]] <- tryCatch({
      fit_i <- fit_bertopic(
        docs              = docs,
        embeddings        = emb_list[[row$model]],
        umap_n_neighbors  = row$n_neighbors,
        umap_n_components = row$n_components,
        hdbscan_min_pts   = row$min_pts,
        ngram_range       = ngram_range,
        top_n_terms       = top_n_terms,
        extra_stopwords   = extra_stopwords,
        seed              = seed,
        verbose           = FALSE
      )
      q_i <- topic_quality(fit_i,
                            top_n       = quality_top_n,
                            sample_size = quality_sample)
      list(q = q_i, error = NA_character_)
    }, error = function(e) {
      list(q = NULL, error = conditionMessage(e))
    })
  }

  # --- Extract metrics into a tidy data frame ---------------------------------
  .get <- function(r, expr, default) {
    if (is.null(r$q)) return(default)
    tryCatch(eval(expr, envir = list(q = r$q)), error = function(e) default)
  }

  results <- grid
  results$n_topics   <- vapply(run_results, .get,
                                numeric(1L), quote(as.numeric(q$n_topics)),  NA_real_)
  results$noise_pct  <- vapply(run_results, .get,
                                numeric(1L), quote(100 * q$distribution$noise_ratio), NA_real_)
  results$silhouette <- vapply(run_results, .get,
                                numeric(1L), quote(q$silhouette$global),     NA_real_)
  results$cohesion   <- vapply(run_results, .get,
                                numeric(1L), quote(q$cohesion$global),       NA_real_)
  results$separation <- vapply(run_results, .get,
                                numeric(1L),
                                quote(q$separation$mean_inter_topic_similarity), NA_real_)
  results$jaccard    <- vapply(run_results, .get,
                                numeric(1L), quote(q$overlap$mean_jaccard),  NA_real_)
  results$entropy    <- vapply(run_results, .get,
                                numeric(1L), quote(q$distribution$entropy),  NA_real_)
  results$error      <- vapply(run_results,
                                function(r) r$error %||% NA_character_, character(1L))

  # --- Best run (highest silhouette) ------------------------------------------
  valid <- which(!is.na(results$silhouette))
  best  <- if (length(valid) > 0L)
    results[valid[which.max(results$silhouette[valid])], , drop = FALSE]
  else
    NULL

  structure(
    list(
      results     = results,
      best        = best,
      n_docs      = n_docs,
      sampled     = sampled,
      param_names = param_swept
    ),
    class = c("topic_sweep", "list")
  )
}

#' @export
print.topic_sweep <- function(x, ...) {
  cat("<topic_sweep>\n")
  cat(sprintf("  Runs:     %d\n", nrow(x$results)))
  cat(sprintf("  Docs:     %d%s\n", x$n_docs,
              if (x$sampled) " (sampled)" else ""))
  failed <- sum(!is.na(x$results$error))
  if (failed > 0L) cat(sprintf("  Failed:   %d run(s)\n", failed))
  cat("\n  Metrics (across all valid runs):\n")
  sil <- x$results$silhouette[!is.na(x$results$silhouette)]
  if (length(sil)) {
    cat(sprintf("    Silhouette:  min=%.3f  median=%.3f  max=%.3f\n",
                min(sil), stats::median(sil), max(sil)))
  }
  if (!is.null(x$best)) {
    b <- x$best
    cat("\n  Best run (highest silhouette):\n")
    if ("model" %in% x$param_names)
      cat(sprintf("    model:       %s\n", b$model))
    cat(sprintf("    n_neighbors: %d\n", b$n_neighbors))
    cat(sprintf("    n_components:%d\n", b$n_components))
    cat(sprintf("    min_pts:     %d\n", b$min_pts))
    cat(sprintf("    n_topics:    %d   silhouette: %.3f\n",
                as.integer(b$n_topics), b$silhouette))
  }
  invisible(x)
}

#' Visualise the results of a parameter sweep
#'
#' Produces an interactive heatmap (via \pkg{plotly}) where each row is one
#' parameter combination and each column is a quality metric.  Within every
#' column the values are min-max normalised to \eqn{[0, 1]} so that colours
#' are comparable across metrics (green = best in column, white = worst).
#' Metrics where a lower raw value is better (separation, Jaccard overlap,
#' noise percentage) are inverted before normalisation.  Hover text shows the
#' actual raw values.
#'
#' @param sweep A \code{topic_sweep} object from \code{\link{sweep_topics}}.
#' @param metrics Character vector selecting which columns of
#'   \code{sweep$results} to display.  Default: all six quality metrics.
#' @param width,height Plot dimensions in pixels.  Height auto-scales to the
#'   number of runs when \code{NULL}.
#' @return A \code{plotly} figure object.
#' @export
visualize_sweep <- function(sweep,
                             metrics = c("silhouette", "cohesion", "separation",
                                         "jaccard", "entropy", "noise_pct"),
                             width   = 900L,
                             height  = NULL) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Please install.packages('plotly') to use visualize_sweep().")
  if (!inherits(sweep, "topic_sweep"))
    stop("'sweep' must be a topic_sweep object from sweep_topics().")

  df      <- sweep$results
  n_runs  <- nrow(df)
  metrics <- intersect(metrics, names(df))
  if (length(metrics) == 0L)
    stop("None of the requested metrics found in sweep$results.")

  # --- Row labels -------------------------------------------------------------
  multi_model <- length(unique(df$model)) > 1L
  row_lbls <- paste0(
    if (multi_model) paste0(df$model, " | ") else "",
    "nbr=",  df$n_neighbors,
    " cmp=", df$n_components,
    " min=", df$min_pts
  )

  # --- Normalise metrics (direction-aware) ------------------------------------
  # Metrics where LOWER raw value = BETTER → invert before normalising
  invert_set <- c("separation", "jaccard", "noise_pct")

  # Human-readable column names and hover labels
  col_names <- c(
    silhouette = "Silhouette",
    cohesion   = "Cohesion",
    separation = "Separation\n(↓ raw)",
    jaccard    = "Vocab\noverlap\n(↓ raw)",
    entropy    = "Size\nentropy",
    noise_pct  = "Noise %\n(↓ raw)"
  )

  .minmax <- function(v) {
    rng <- range(v, na.rm = TRUE)
    if (diff(rng) == 0) return(rep(0.5, length(v)))
    (v - rng[1L]) / diff(rng)
  }

  norm_mat <- matrix(NA_real_, nrow = n_runs, ncol = length(metrics),
                     dimnames = list(row_lbls, metrics))
  raw_mat  <- matrix(NA_real_, nrow = n_runs, ncol = length(metrics),
                     dimnames = list(row_lbls, metrics))

  for (m in metrics) {
    raw  <- as.numeric(df[[m]])
    norm <- .minmax(raw)
    if (m %in% invert_set) norm <- 1 - norm
    raw_mat[, m]  <- raw
    norm_mat[, m] <- norm
  }

  # --- Hover text matrix ------------------------------------------------------
  hover_mat <- matrix("", nrow = n_runs, ncol = length(metrics))
  for (j in seq_along(metrics)) {
    m <- metrics[j]
    hover_mat[, j] <- ifelse(
      is.na(raw_mat[, m]),
      paste0(col_names[m], ": failed"),
      paste0("<b>", row_lbls, "</b><br>",
             col_names[m] %||% m, ": ",
             ifelse(m == "n_topics",
                    formatC(raw_mat[, m], format = "d"),
                    formatC(raw_mat[, m], digits = 3L, format = "f")))
    )
  }

  # --- Annotate best row ------------------------------------------------------
  best_row <- if (!is.null(sweep$best) && "silhouette" %in% metrics) {
    b <- sweep$best
    which(df$model       == b$model        &
          df$n_neighbors == b$n_neighbors  &
          df$n_components == b$n_components &
          df$min_pts     == b$min_pts)[1L]
  } else NULL

  annots <- if (!is.null(best_row)) {
    list(list(
      x = length(metrics) - 0.5,
      y = best_row - 1L,    # 0-indexed
      xref = "x", yref = "y",
      text = "★ best",
      showarrow = FALSE,
      font = list(size = 10L, color = "#2c7bb6"),
      xanchor = "left"
    ))
  } else list()

  # --- Build heatmap ----------------------------------------------------------
  if (is.null(height)) height <- max(400L, 30L * n_runs + 120L)

  pretty_cols <- vapply(metrics, function(m) col_names[m] %||% m, character(1L))

  p <- plotly::plot_ly(
    z           = norm_mat,
    x           = pretty_cols,
    y           = row_lbls,
    type        = "heatmap",
    colorscale  = list(list(0, "#f7f7f7"), list(1, "#1a7a3f")),
    zmin        = 0, zmax = 1,
    text        = hover_mat,
    hovertemplate = "%{text}<extra></extra>",
    showscale   = FALSE
  )

  # Overlay raw value text
  for (j in seq_along(metrics)) {
    m    <- metrics[j]
    vals <- raw_mat[, m]
    txt  <- ifelse(is.na(vals), "err",
             ifelse(m %in% c("n_topics", "noise_pct"),
                    formatC(vals, digits = 1L, format = "f"),
                    formatC(vals, digits = 3L, format = "f")))
    p <- plotly::add_annotations(p,
      x         = rep(j - 1L, n_runs),
      y         = seq_len(n_runs) - 1L,
      text      = txt,
      xref      = "x", yref = "y",
      showarrow = FALSE,
      font      = list(size = 9L, color = "#1a1a1a")
    )
  }

  plotly::layout(p,
    width  = width,
    height = height,
    xaxis  = list(title = "", side = "top",
                  tickfont = list(size = 10L)),
    yaxis  = list(title = "", autorange = "reversed",
                  tickfont = list(size = 10L)),
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    margin = list(l = 160L, t = 80L, r = 20L, b = 20L),
    annotations = annots
  )
}
