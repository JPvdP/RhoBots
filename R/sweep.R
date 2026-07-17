# =============================================================================
# sweep.R  --  Parameter sweep over BERTopic hyperparameters
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
#' @param min_topics Minimum number of topics required in the \code{best}
#'   selection.  When set, \code{best} is the highest-silhouette run among
#'   those that produced at least \code{min_topics} topics.  If no combination
#'   meets the constraint, a warning is issued and \code{best} falls back to
#'   the run with the most topics.  \code{NULL} (default) selects purely by
#'   silhouette.
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
#'     and columns for all swept parameters plus quality metrics (including
#'     \code{n_topics}).}
#'   \item{\code{best}}{The selected row of \code{results}: highest silhouette
#'     among runs satisfying \code{min_topics}, or the run with the most topics
#'     if no run satisfies the constraint.}
#'   \item{\code{min_topics}}{The \code{min_topics} argument (or \code{NULL}).}
#'   \item{\code{best_met_constraint}}{Logical: \code{TRUE} when \code{best}
#'     satisfies the \code{min_topics} constraint (always \code{TRUE} when
#'     \code{min_topics = NULL}).}
#'   \item{\code{n_docs}}{Number of documents used (after optional sampling).}
#'   \item{\code{sampled}}{Logical: whether a random sample was drawn.}
#'   \item{\code{param_names}}{Character vector of swept parameter names.}
#' }
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, abstracts, normalize = TRUE)
#'   sw  <- sweep_topics(abstracts, embeddings = emb,
#'                        n_neighbors = c(5L, 15L), min_pts = c(5L, 10L))
#'   visualize_sweep(sw)
#' }
#' @export
sweep_topics <- function(docs,
                          encoders        = NULL,
                          embeddings      = NULL,
                          n_neighbors     = c(5L, 15L, 30L),
                          n_components    = c(5L, 10L),
                          min_pts         = c(5L, 10L, 20L),
                          min_topics      = NULL,
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
      if (verbose)
        message(sprintf("    -> %d topics  sil=%.3f  noise=%.0f%%",
                        as.integer(q_i$n_topics),
                        q_i$silhouette$global %||% NA_real_,
                        100 * q_i$distribution$noise_ratio))
      list(q = q_i, error = NA_character_)
    }, error = function(e) {
      if (verbose) message("    -> ERROR: ", conditionMessage(e))
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

  # --- Best run: optimise silhouette subject to min_topics constraint ---------
  valid <- which(!is.na(results$silhouette))
  best  <- NULL
  best_met_constraint <- TRUE

  if (length(valid) > 0L) {
    if (!is.null(min_topics)) {
      # Constrained selection: among valid runs with n_topics >= min_topics,
      # pick the one with the highest silhouette.
      ok <- valid[!is.na(results$n_topics[valid]) &
                    results$n_topics[valid] >= min_topics]
      if (length(ok) > 0L) {
        best <- results[ok[which.max(results$silhouette[ok])], , drop = FALSE]
      } else {
        # No run met the constraint  --  fall back to the run with the most topics.
        best_met_constraint <- FALSE
        nt_valid <- valid[!is.na(results$n_topics[valid])]
        best <- results[nt_valid[which.max(results$n_topics[nt_valid])], ,
                        drop = FALSE]
        warning(
          "No sweep combination produced at least ", min_topics, " topics. ",
          "The run with the most topics (", as.integer(best$n_topics), ") ",
          "was selected instead.\n",
          "Consider lowering min_pts or n_components in your sweep grid.",
          call. = FALSE
        )
      }
    } else {
      # Unconstrained: highest silhouette wins.
      best <- results[valid[which.max(results$silhouette[valid])], , drop = FALSE]
    }
  }

  structure(
    list(
      results              = results,
      best                 = best,
      min_topics           = min_topics,
      best_met_constraint  = best_met_constraint,
      n_docs               = n_docs,
      sampled              = sampled,
      param_names          = param_swept
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
  if (!is.null(x$min_topics))
    cat(sprintf("  Constraint: n_topics >= %d\n", x$min_topics))
  failed <- sum(!is.na(x$results$error))
  if (failed > 0L) cat(sprintf("  Failed:   %d run(s)\n", failed))

  # Summary of topic counts and silhouette across all valid runs
  r   <- x$results
  ok  <- !is.na(r$silhouette)
  sil <- r$silhouette[ok]
  nt  <- r$n_topics[ok]
  if (length(sil)) {
    cat("\n  n_topics across runs: min=", min(nt, na.rm = TRUE),
        " median=", stats::median(nt, na.rm = TRUE),
        " max=", max(nt, na.rm = TRUE), "\n", sep = "")
    cat(sprintf("  Silhouette:           min=%.3f  median=%.3f  max=%.3f\n",
                min(sil), stats::median(sil), max(sil)))
  }

  if (!is.null(x$best)) {
    b        <- x$best
    met      <- isTRUE(x$best_met_constraint)
    sel_note <- if (!is.null(x$min_topics))
      if (met) "  (constraint met)"
      else     "  *** constraint NOT met  --  no run reached min_topics ***"
    else ""
    cat("\n  Best run", sel_note, ":\n", sep = "")
    if ("model" %in% x$param_names)
      cat(sprintf("    model:        %s\n", b$model))
    cat(sprintf("    n_neighbors:  %d\n", b$n_neighbors))
    cat(sprintf("    n_components: %d\n", b$n_components))
    cat(sprintf("    min_pts:      %d\n", b$min_pts))
    cat(sprintf("    n_topics:     %d   silhouette: %.3f\n",
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
#' \code{n_topics} is always shown as the first column.  When a
#' \code{min_topics} constraint was passed to \code{\link{sweep_topics}}, rows
#' that did not meet it are prefixed with \code{"[x] "} in the row labels.
#'
#' @param sweep A \code{topic_sweep} object from \code{\link{sweep_topics}}.
#' @param metrics Character vector selecting which columns of
#'   \code{sweep$results} to display.  \code{"n_topics"} is always prepended
#'   regardless of this argument.
#' @param width,height Plot dimensions in pixels.  Height auto-scales to the
#'   number of runs when \code{NULL}.
#' @return A \code{plotly} figure object.
#' @examples
#' \dontrun{
#'   enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb <- embed_texts(enc, abstracts, normalize = TRUE)
#'   sw  <- sweep_topics(abstracts, embeddings = emb,
#'                        n_neighbors = c(5L, 15L), min_pts = c(5L, 10L))
#'   visualize_sweep(sw)
#' }
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

  # If every run failed, return an informative placeholder instead of crashing.
  if (all(is.na(df$silhouette))) {
    n_failed <- sum(!is.na(df$error))
    err_msg  <- df$error[!is.na(df$error)][1L] %||% "unknown error"
    msg <- paste0("All ", n_failed, " sweep run(s) failed.\n",
                  "First error: ", err_msg, "\n\n",
                  "Check sw$results$error for details.")
    message("visualize_sweep: all runs failed  --  ", err_msg)
    return(
      plotly::plot_ly(width = width %||% 900L, height = 300L) |>
      plotly::layout(
        annotations = list(list(
          text      = gsub("\n", "<br>", msg),
          x = 0.5, y = 0.5, xref = "paper", yref = "paper",
          showarrow = FALSE, font = list(size = 13L, color = "#c0392b")
        )),
        xaxis = list(visible = FALSE),
        yaxis = list(visible = FALSE),
        paper_bgcolor = "white", plot_bgcolor = "white"
      )
    )
  }

  # n_topics is always the first column shown
  metrics <- unique(c("n_topics", intersect(metrics, names(df))))
  if (length(metrics) == 0L)
    stop("None of the requested metrics found in sweep$results.")

  # --- Row labels -------------------------------------------------------------
  # When a min_topics constraint was set, prefix rows that didn't meet it
  # with "[x] " so users can immediately spot which combinations are unusable.
  multi_model  <- length(unique(df$model)) > 1L
  min_topics   <- sweep$min_topics
  meets_constraint <- if (!is.null(min_topics) && "n_topics" %in% names(df))
    !is.na(df$n_topics) & df$n_topics >= min_topics
  else
    rep(TRUE, n_runs)

  row_lbls <- paste0(
    ifelse(meets_constraint, "  ", "[x] "),
    if (multi_model) paste0(df$model, " | ") else "",
    "nbr=",  df$n_neighbors,
    " cmp=", df$n_components,
    " min=", df$min_pts
  )

  # --- Normalise metrics (direction-aware) ------------------------------------
  # Metrics where LOWER raw value = BETTER -> invert before normalising
  invert_set <- c("separation", "jaccard", "noise_pct")

  # Human-readable column names and hover labels
  col_names <- c(
    n_topics   = "Topics\nfound",
    silhouette = "Silhouette",
    cohesion   = "Cohesion",
    separation = "Separation\n(v raw)",
    jaccard    = "Vocab\noverlap\n(v raw)",
    entropy    = "Size\nentropy",
    noise_pct  = "Noise %\n(v raw)"
  )

  .minmax <- function(v) {
    rng <- range(v, na.rm = TRUE)
    if (!all(is.finite(rng)) || diff(rng) == 0) return(rep(0.5, length(v)))
    (v - rng[1L]) / diff(rng)
  }

  norm_mat <- matrix(NA_real_, nrow = n_runs, ncol = length(metrics),
                     dimnames = list(row_lbls, metrics))
  raw_mat  <- matrix(NA_real_, nrow = n_runs, ncol = length(metrics),
                     dimnames = list(row_lbls, metrics))

  flat_cols <- character(0)
  for (m in metrics) {
    raw  <- as.numeric(df[[m]])
    norm <- .minmax(raw)
    if (m %in% invert_set) norm <- 1 - norm
    raw_mat[, m]  <- raw
    norm_mat[, m] <- norm
    if (all(norm == 0.5, na.rm = TRUE) && !all(is.na(raw)))
      flat_cols <- c(flat_cols, m)
  }
  if (length(flat_cols) > 0L)
    message("Note: all runs produced identical values for: ",
            paste(flat_cols, collapse = ", "),
            "  --  heatmap colours are uniform for those columns ",
            "(raw values still shown in cell text).")

  # --- Cell text (raw values displayed inside each cell) ----------------------
  fmt_val <- function(vals, m)
    ifelse(is.na(vals), "err",
           ifelse(m == "n_topics",
                  as.character(as.integer(vals)),
                  sprintf("%.3f", vals)))

  cell_mat <- matrix("", nrow = n_runs, ncol = length(metrics))
  for (j in seq_along(metrics))
    cell_mat[, j] <- fmt_val(raw_mat[, metrics[j]], metrics[j])

  # --- Hover text (shown on mouse-over) ---------------------------------------
  pretty_cols <- vapply(metrics,
                        function(m) col_names[m] %||% m, character(1L))
  hover_mat <- matrix("", nrow = n_runs, ncol = length(metrics))
  for (j in seq_along(metrics)) {
    nm <- gsub("\n", " ", pretty_cols[j])
    hover_mat[, j] <- paste0(
      "<b>", row_lbls, "</b><br>",
      nm, ": ", cell_mat[, j]
    )
  }

  # --- Best-row annotation ----------------------------------------------------
  best_row <- if (!is.null(sweep$best)) {
    b <- sweep$best
    which(df$model        == b$model        &
          df$n_neighbors  == b$n_neighbors  &
          df$n_components == b$n_components &
          df$min_pts      == b$min_pts)[1L]
  } else NULL

  best_label <- if (!is.null(best_row)) {
    if (isTRUE(sweep$best_met_constraint)) "  * best" else "  * best (fallback)"
  } else NULL

  # --- Build heatmap ----------------------------------------------------------
  if (is.null(height)) height <- max(400L, 30L * n_runs + 120L)

  # plotly:::map_color() calls scales::col_numeric(domain = range(z, na.rm=TRUE)).
  # If every value in z is NA, range() returns c(Inf, -Inf) and scales crashes.
  # Replace any remaining NAs with the neutral midpoint before passing to plotly.
  norm_mat[is.na(norm_mat)] <- 0.5

  p <- plotly::plot_ly(
    width         = width,
    height        = height,
    z             = norm_mat,
    x             = pretty_cols,
    y             = row_lbls,
    type          = "heatmap",
    colorscale    = list(list(0, "#f7f7f7"), list(1, "#1a7a3f")),
    zmin          = 0,
    zmax          = 1,
    zauto         = FALSE,
    text          = cell_mat,
    hovertext     = hover_mat,
    hovertemplate = "%{hovertext}<extra></extra>",
    texttemplate  = "%{text}",
    textfont      = list(size = 9L, color = "#333333"),
    showscale     = FALSE
  )

  if (!is.null(best_row)) {
    p <- plotly::add_annotations(p,
      x         = length(metrics) - 0.5,
      y         = best_row - 1L,
      xref      = "x",
      yref      = "y",
      text      = best_label,
      showarrow = FALSE,
      xanchor   = "left",
      font      = list(size = 10L, color = "#2c7bb6")
    )
  }

  plotly::layout(p,
    xaxis         = list(title = "", side = "top",
                         tickfont = list(size = 10L)),
    yaxis         = list(title = "", autorange = "reversed",
                         tickfont = list(size = 10L)),
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    margin        = list(l = 160L, t = 80L, r = 20L, b = 20L)
  )
}
