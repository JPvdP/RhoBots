# =============================================================================
# stability.R  --  Topic stability analysis across multiple random seeds.
#
# Answers "how reproducible are these topics?" by running fit_bertopic()
# multiple times and measuring agreement via the Adjusted Rand Index (ARI).
# =============================================================================

# Adjusted Rand Index (pure R, no external dep)
.adj_rand_index <- function(x, y) {
  tab  <- table(x, y)
  n    <- sum(tab)
  c2   <- function(k) k * (k - 1L) / 2L
  sum_c  <- sum(c2(tab))
  sum_a  <- sum(c2(rowSums(tab)))
  sum_b  <- sum(c2(colSums(tab)))
  expect <- sum_a * sum_b / c2(n)
  denom  <- 0.5 * (sum_a + sum_b) - expect
  if (denom == 0) return(1.0)
  (sum_c - expect) / denom
}

#' Measure topic stability across multiple random seeds
#'
#' Runs \code{\link{fit_bertopic}} \code{n_runs} times with different random
#' seeds on the same corpus, then measures how consistent the document
#' assignments are across runs using the Adjusted Rand Index (ARI).
#'
#' ARI = 1 means two clusterings are identical; ARI \eqn{\approx} 0 means
#' agreement no better than chance; negative values indicate systematic
#' disagreement.  A mean ARI above 0.8 across all run pairs indicates a
#' stable, reproducible topic structure.
#'
#' @param docs Character vector of documents.
#' @param embeddings Pre-computed embedding matrix (strongly recommended to
#'   avoid re-embedding for every run).
#' @param n_runs Number of independent fits (default 5).
#' @param seeds Integer vector of random seeds, length \code{n_runs}.
#'   Defaults to \code{42 * 1:n_runs}.
#' @param ... Additional arguments forwarded to \code{\link{fit_bertopic}}.
#' @param verbose Print per-run progress (default \code{TRUE}).
#' @return A list of class \code{stability_result} with elements:
#' \describe{
#'   \item{\code{ari_matrix}}{Symmetric ARI matrix (\code{n_runs x n_runs}).}
#'   \item{\code{mean_ari}}{Mean ARI across all off-diagonal pairs.}
#'   \item{\code{per_doc_stability}}{For each document, the fraction of runs
#'     that agreed on the modal topic assignment (1.0 = always the same).}
#'   \item{\code{n_topics_per_run}}{Integer vector of non-noise topic counts.}
#'   \item{\code{fits}}{List of \code{bertopic_fit} objects (one per run).}
#' }
#' @seealso \code{\link{visualize_stability}}, \code{\link{sweep_topics}}
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb  <- embed_texts(enc, abstracts)
#'   stab <- stability_analysis(abstracts, emb, n_runs = 3L)
#'   stab$mean_ari
#' }
#' @export
stability_analysis <- function(docs, embeddings, n_runs = 5L, seeds = NULL,
                                ..., verbose = TRUE) {
  if (is.null(seeds)) seeds <- 42L * seq_len(n_runs)
  if (length(seeds) != n_runs)
    stop("'seeds' must have length equal to 'n_runs'.")

  if (verbose) message("Running ", n_runs, " fits for stability analysis...")

  fits <- vector("list", n_runs)
  for (i in seq_len(n_runs)) {
    if (verbose) message("  Run [", i, "/", n_runs, "]  seed = ", seeds[i])
    fits[[i]] <- tryCatch(
      fit_bertopic(docs = docs, embeddings = embeddings,
                   seed = seeds[i], verbose = FALSE, ...),
      error = function(e) {
        warning("Run ", i, " (seed=", seeds[i], ") failed: ", conditionMessage(e))
        NULL
      }
    )
  }

  valid   <- !vapply(fits, is.null, logical(1L))
  fits_v  <- fits[valid]
  seeds_v <- seeds[valid]
  k       <- length(fits_v)
  if (k < 2L)
    stop("Fewer than 2 runs succeeded  --  cannot compute stability.")

  # Pairwise ARI
  ari_mat <- matrix(1.0, k, k,
                    dimnames = list(paste0("run_", seeds_v),
                                    paste0("run_", seeds_v)))
  for (i in seq_len(k - 1L)) {
    for (j in (i + 1L):k) {
      v <- .adj_rand_index(fits_v[[i]]$clusters, fits_v[[j]]$clusters)
      ari_mat[i, j] <- ari_mat[j, i] <- v
    }
  }

  # Per-document: fraction of runs that agree on the modal topic
  all_cl  <- do.call(cbind, lapply(fits_v, `[[`, "clusters"))
  per_doc <- apply(all_cl, 1L, function(row) {
    max(table(row)) / length(row)
  })

  n_topics_run <- vapply(fits_v, function(f)
    length(setdiff(unique(f$clusters), -1L)), integer(1L))

  structure(
    list(
      ari_matrix        = ari_mat,
      mean_ari          = mean(ari_mat[upper.tri(ari_mat)]),
      per_doc_stability = per_doc,
      n_topics_per_run  = setNames(n_topics_run, paste0("run_", seeds_v)),
      n_runs            = k,
      seeds             = seeds_v,
      fits              = fits_v
    ),
    class = c("stability_result", "list")
  )
}

#' @export
print.stability_result <- function(x, ...) {
  cat("<stability_result>\n")
  cat(sprintf("  Runs:           %d  (seeds: %s)\n",
              x$n_runs, paste(x$seeds, collapse = ", ")))
  cat(sprintf("  Mean ARI:       %.3f  [1.0 = identical across all runs]\n",
              x$mean_ari))
  cat(sprintf("  Topics/run:     %s\n",
              paste(x$n_topics_per_run, collapse = ", ")))
  cat(sprintf("  Doc stability:  median = %.3f   min = %.3f\n",
              median(x$per_doc_stability), min(x$per_doc_stability)))
  invisible(x)
}

#' Interactive heatmap of pairwise ARI scores
#'
#' @param stab A \code{stability_result} from \code{\link{stability_analysis}}.
#' @param width,height Plot dimensions in pixels (default 550 x 500).
#' @return A \code{plotly} figure.
#' @examples
#' \dontrun{
#'   enc  <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")
#'   emb  <- embed_texts(enc, abstracts)
#'   stab <- stability_analysis(abstracts, emb, n_runs = 3L)
#'   visualize_stability(stab)
#' }
#' @export
visualize_stability <- function(stab, width = 550L, height = 500L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Package 'plotly' is required. Install with install.packages(\"plotly\").")

  ari <- stab$ari_matrix
  rn  <- rownames(ari)
  z   <- ari; diag(z) <- NA_real_

  plotly::plot_ly(width = width, height = height) |>
    plotly::add_heatmap(
      x = rn, y = rn,
      z = z,
      zmin = -1, zmid = 0, zmax = 1,
      colorscale = list(
        c(0,   "#d73027"),
        c(0.5, "#ffffbf"),
        c(1,   "#1a9850")
      ),
      text         = matrix(sprintf("%.3f", ari), nrow = nrow(ari),
                             dimnames = dimnames(ari)),
      texttemplate = "%{text}",
      textfont     = list(size = 13L),
      showscale    = TRUE,
      colorbar     = list(title = "ARI")
    ) |>
    plotly::layout(
      title = list(text = sprintf(
        "Topic stability  --  mean ARI = %.3f", stab$mean_ari)),
      xaxis = list(title = ""),
      yaxis = list(title = "", autorange = "reversed")
    )
}
