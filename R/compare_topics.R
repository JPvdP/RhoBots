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

  topics_nn    <- sort(setdiff(unique(fit$clusters), -1L))
  group_levels <- sort(unique(groups))

  nonnoise <- fit$clusters != -1L
  cl_nn    <- fit$clusters[nonnoise]
  grp_nn   <- groups[nonnoise]

  tab      <- table(Topic = cl_nn, Group = grp_nn)
  expected <- outer(rowSums(tab), colSums(tab)) / sum(tab)

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
#' @param comp A \code{compare_topics_result} from \code{\link{compare_topics}}.
#' @param top_n_topics Restrict to the \code{top_n_topics} topics with the
#'   largest mean absolute statistic.  \code{NULL} (default) shows all topics.
#' @param width,height Plot dimensions in pixels (default 800 × 550).
#' @return A \code{plotly} figure.
#' @export
visualize_comparison <- function(comp,
                                  top_n_topics = NULL,
                                  width  = 800L,
                                  height = 550L) {
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

  row_labels <- vapply(as.character(topics_ord), function(t)
    df$Name[match(as.integer(t), df$Topic)][1L] %||% t,
    character(1L))

  z_mat   <- matrix(NA_real_, length(topics_ord), length(groups_ord),
                    dimnames = list(as.character(topics_ord), groups_ord))
  txt_mat <- z_mat
  for (i in seq_len(nrow(df))) {
    r <- as.character(df$Topic[i]); g <- df$Group[i]
    z_mat[r, g]   <- df$Stat[i]
    txt_mat[r, g] <- sprintf("%.2f\n(n=%d)", df$Stat[i], df$Observed[i])
  }

  abs_max <- max(abs(z_mat), na.rm = TRUE)

  stat_label <- if (comp$method == "chi2")
    "Signed χ²" else "log₂ ratio"
  title_str  <- sprintf("%s by topic × group", stat_label)

  plotly::plot_ly(width = width, height = height) |>
    plotly::add_heatmap(
      x = groups_ord,
      y = row_labels,
      z = z_mat,
      zmin = -abs_max, zmid = 0, zmax = abs_max,
      colorscale = list(
        c(0,   "#d73027"),
        c(0.5, "#f7f7f7"),
        c(1,   "#2166ac")
      ),
      text         = txt_mat,
      texttemplate = "%{text}",
      textfont     = list(size = 11L),
      showscale    = TRUE,
      colorbar     = list(title = stat_label)
    ) |>
    plotly::layout(
      title = list(text = title_str),
      xaxis = list(title = "Group"),
      yaxis = list(title = "", autorange = "reversed")
    )
}
