# =============================================================================
# compare_topics.R — Class-conditional topic analysis.
#
# Measures which topics are over- or under-represented in user-defined groups
# (e.g. time periods, countries, treatment arms) using signed chi-square
# contributions or log2-ratios.
# =============================================================================

#' Compare topic prevalence across groups
#'
#' Given a grouping variable (e.g. publication year, country, experimental
#' condition) of the same length as the corpus, computes which topics are
#' over- or under-represented in each group relative to a null model of
#' independence.
#'
#' Two statistics are available:
#' \describe{
#'   \item{\code{"chi2"}}{Signed chi-square contribution:
#'     \eqn{sign(O-E) \cdot \sqrt{(O-E)^2/E}}.  Positive = over-represented,
#'     negative = under-represented.  A global chi-square test is also reported.}
#'   \item{\code{"log_ratio"}}{Laplace-smoothed log\eqn{_2} ratio of observed
#'     to expected proportion.  Values above 1 mean the topic is twice as
#'     prevalent in that group as expected; below -1 means half as prevalent.}
#' }
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param groups Character or factor vector, one entry per document, defining
#'   the group membership.  Noise documents (\code{-1}) are excluded from the
#'   analysis but the vector still needs one entry per document.
#' @param method Test statistic: \code{"chi2"} (default) or \code{"log_ratio"}.
#' @param min_count Minimum expected count for a (topic, group) cell to be
#'   included.  Cells below this threshold are silently dropped (default 5).
#' @param verbose Print the contingency table (default \code{TRUE}).
#' @return A list of class \code{compare_topics_result} with elements:
#' \describe{
#'   \item{\code{result}}{Tidy data frame with columns \code{Topic},
#'     \code{Name}, \code{Group}, \code{Observed}, \code{Expected},
#'     \code{Stat}.}
#'   \item{\code{table}}{Raw contingency table (topics × groups).}
#'   \item{\code{global_statistic}, \code{global_pvalue}}{Global chi-square
#'     statistic and p-value (\code{NA} when \code{method = "log_ratio"}).}
#' }
#' @seealso \code{\link{visualize_comparison}}
#' @export
compare_topics <- function(fit, groups,
                            method    = c("chi2", "log_ratio"),
                            min_count = 5L,
                            verbose   = TRUE) {
  method <- match.arg(method)
  if (!inherits(fit, "bertopic_fit"))
    stop("'fit' must be a bertopic_fit object.")

  groups <- as.character(groups)
  if (length(groups) != length(fit$docs))
    stop("'groups' must have the same length as the corpus (", length(fit$docs),
         " documents).")

  topics_nn <- sort(setdiff(unique(fit$clusters), -1L))

  nonnoise <- fit$clusters != -1L
  cl_nn    <- fit$clusters[nonnoise]
  grp_nn   <- groups[nonnoise]

  tab      <- table(Topic = cl_nn, Group = grp_nn)
  expected <- outer(rowSums(tab), colSums(tab)) / sum(tab)

  # Derive group_levels from the contingency table, not from the full groups
  # vector.  Groups that appear only in noise documents are absent from tab;
  # indexing tab with those group names causes "subscript out of bounds".
  group_levels <- colnames(tab)
  topics_nn    <- as.integer(rownames(tab))   # restrict to topics in the table

  if (verbose) {
    cat("Contingency table (topics × groups, non-noise docs only):\n")
    print(tab)
    cat("\n")
  }

  rows <- vector("list", length(topics_nn) * length(group_levels))
  k    <- 0L
  for (t in topics_nn) {
    for (g in group_levels) {
      obs <- tab[as.character(t), g]
      exp <- expected[as.character(t), g]
      if (exp < min_count) next

      if (method == "chi2") {
        contrib <- (obs - exp)^2 / exp
        stat    <- sign(obs - exp) * sqrt(contrib)
      } else {
        row_tot <- sum(tab[as.character(t), ])
        col_tot <- sum(tab[, g])
        n       <- sum(tab)
        p_obs   <- (obs + 0.5) / (row_tot + 1)
        p_exp   <- col_tot / n
        stat    <- log2(p_obs / p_exp)
      }

      k <- k + 1L
      rows[[k]] <- data.frame(
        Topic    = t,
        Name     = fit$topic_labels[[as.character(t)]] %||% as.character(t),
        Group    = g,
        Observed = as.integer(obs),
        Expected = round(exp, 1L),
        Stat     = round(stat, 4L),
        stringsAsFactors = FALSE
      )
    }
  }
  if (k == 0L) {
    warning("No (topic, group) cells met the min_count threshold. ",
            "Try lowering min_count.")
    result_df <- data.frame(
      Topic = integer(0), Name = character(0), Group = character(0),
      Observed = integer(0), Expected = numeric(0), Stat = numeric(0),
      stringsAsFactors = FALSE
    )
  } else {
    result_df <- do.call(rbind, rows[seq_len(k)])
    rownames(result_df) <- NULL
  }

  # Global test
  if (method == "chi2") {
    ct          <- suppressWarnings(chisq.test(tab))
    global_stat <- unname(ct$statistic)
    global_p    <- ct$p.value
  } else {
    global_stat <- NA_real_
    global_p    <- NA_real_
  }

  structure(
    list(
      result           = result_df,
      table            = tab,
      method           = method,
      global_statistic = global_stat,
      global_pvalue    = global_p,
      group_levels     = group_levels,
      topic_ids        = topics_nn
    ),
    class = c("compare_topics_result", "list")
  )
}

#' @export
print.compare_topics_result <- function(x, top_n = 10L, ...) {
  stat_lbl <- if (x$method == "chi2") "Signed chi2" else "log2 ratio"
  cat("<compare_topics_result>\n")
  cat(sprintf("  Method: %-12s  Groups: %s\n",
              x$method, paste(x$group_levels, collapse = ", ")))
  if (!is.na(x$global_pvalue))
    cat(sprintf("  Global chi2 = %.2f   p = %.4f\n",
                x$global_statistic, x$global_pvalue))
  if (nrow(x$result) == 0L) {
    cat("  (no cells met the min_count threshold)\n")
    return(invisible(x))
  }
  cat(sprintf("\n  Top %d associations by |%s|:\n", top_n, stat_lbl))
  ord <- order(abs(x$result$Stat), decreasing = TRUE)
  print(head(x$result[ord, ], top_n), row.names = FALSE)
  invisible(x)
}

#' Interactive heatmap of topic–group associations
#'
#' Displays the signed chi-square contributions or log\eqn{_2} ratios from
#' \code{\link{compare_topics}} as a diverging colour heatmap: blue = over-
#' represented in that group, red = under-represented.
#'
#' Topics are placed on the x-axis (with angled labels) and groups on the
#' y-axis, which keeps the chart readable even when topic labels are long.
#'
#' @param comp A \code{compare_topics_result} from \code{\link{compare_topics}}.
#' @param top_n_topics Restrict to the \code{top_n_topics} topics with the
#'   largest mean absolute statistic.  \code{NULL} (default) shows all topics.
#' @param max_label_chars Maximum characters for topic labels before truncation
#'   with an ellipsis (default 25).
#' @param width,height Plot dimensions in pixels (default 1000 × 420).
#' @return A \code{plotly} figure.
#' @export
visualize_comparison <- function(comp,
                                  top_n_topics   = NULL,
                                  max_label_chars = 25L,
                                  width  = 1000L,
                                  height = 420L) {
  if (!requireNamespace("plotly", quietly = TRUE))
    stop("Package 'plotly' is required. Install with install.packages(\"plotly\").")

  df <- comp$result

  if (!is.null(top_n_topics)) {
    topic_score <- tapply(abs(df$Stat), df$Topic, mean)
    keep <- as.integer(names(sort(topic_score, decreasing = TRUE)[
      seq_len(min(top_n_topics, length(topic_score)))]))
    df <- df[df$Topic %in% keep, ]
  }

  topics_ord <- sort(unique(df$Topic))
  groups_ord <- sort(unique(df$Group))

  # Short display labels: strip numeric prefix, replace underscores, truncate.
  topic_labels <- vapply(as.character(topics_ord), function(t) {
    raw <- df$Name[match(as.integer(t), df$Topic)][1L] %||% t
    txt <- sub("^-?[0-9]+_", "", raw)
    txt <- gsub("_", " ", txt)
    txt <- trimws(txt)
    if (nchar(txt) > max_label_chars)
      paste0(substr(txt, 1L, max_label_chars - 1L), "…")
    else
      txt
  }, character(1L))

  # Build a topics × groups matrix of statistics (NA = cell below min_count).
  z_mat   <- matrix(NA_real_, length(topics_ord), length(groups_ord),
                    dimnames = list(as.character(topics_ord), groups_ord))
  txt_mat <- z_mat
  for (i in seq_len(nrow(df))) {
    r <- as.character(df$Topic[i]); g <- df$Group[i]
    z_mat[r, g]   <- df$Stat[i]
    txt_mat[r, g] <- sprintf("%.2f\n(n=%d)", df$Stat[i], df$Observed[i])
  }

  # Topics on x-axis, groups on y-axis.
  # Transpose z so rows = groups, columns = topics.
  z_t   <- t(z_mat)
  txt_t <- t(txt_mat)

  abs_max <- max(abs(z_mat), na.rm = TRUE)

  stat_label <- if (comp$method == "chi2") "Signed χ²" else "log₂ ratio"
  title_str  <- sprintf("%s by group × topic", stat_label)

  plotly::plot_ly(width = width, height = height) |>
    plotly::add_heatmap(
      x = topic_labels,        # topics on x-axis
      y = groups_ord,           # groups on y-axis (typically short: continent names)
      z = z_t,
      zmin = -abs_max, zmid = 0, zmax = abs_max,
      colorscale = list(
        c(0,   "#d73027"),
        c(0.5, "#f7f7f7"),
        c(1,   "#2166ac")
      ),
      text         = txt_t,
      texttemplate = "%{text}",
      textfont     = list(size = 10L),
      showscale    = TRUE,
      colorbar     = list(title = stat_label, len = 0.8)
    ) |>
    plotly::layout(
      title  = list(text = title_str, x = 0),
      xaxis  = list(title = "", tickangle = -40L,
                    tickfont = list(size = 11L)),
      yaxis  = list(title = "", autorange = "reversed",
                    tickfont = list(size = 12L)),
      margin = list(b = 130L, l = 110L, r = 80L, t = 50L)
    )
}
