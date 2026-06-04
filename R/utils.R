#' Default-value operator
#'
#' Returns `y` when `x` is `NULL`, otherwise returns `x`.  Useful for reading
#' config fields that may or may not be present in `config.json`.
#'
#' @param x A value to test for NULL.
#' @param y A default value to use when `x` is NULL.
#' @return `x` if not NULL, otherwise `y`.
#' @keywords internal
#' @noRd
`%||%` <- function(x, y) if (is.null(x)) y else x

# Standard English stopwords (NLTK list).  Used as the default when
# language = "english" in fit_bertopic().
.english_stopwords <- c(
  "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
  "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
  "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
  "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
  "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "having", "do", "does", "did", "doing",
  "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
  "while", "of", "at", "by", "for", "with", "about", "against", "between",
  "into", "through", "during", "before", "after", "above", "below", "to",
  "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
  "further", "then", "once", "here", "there", "when", "where", "why", "how",
  "all", "both", "each", "few", "more", "most", "other", "some", "such",
  "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
  "can", "will", "just", "should", "now", "also", "however", "therefore",
  "thus", "hence", "whereas", "although", "though", "yet", "still",
  "already", "since", "either", "neither", "whether", "via", "per",
  "et", "al", "eg", "ie", "vs", "s", "t", "d", "ll", "re", "ve", "m"
)

# =============================================================================
# Stopword utilities
# =============================================================================

#' Return the built-in stopword list for a language
#'
#' Currently only \code{"english"} is supported; returns an empty character
#' vector for any other value.
#'
#' @param language Language name (default \code{"english"}).
#' @return A character vector of stopwords.
#' @export
get_stopwords <- function(language = "english") {
  if (identical(tolower(language), "english")) return(.english_stopwords)
  warning("No built-in stopwords for '", language, "'. Returning empty vector.")
  character(0)
}

#' Load a stopword list from a character vector, data frame, or file
#'
#' Accepts three input types, making it easy to manage domain-specific
#' stopwords from any source:
#' \describe{
#'   \item{Character vector}{Returned as-is (after deduplication and trimming).}
#'   \item{Data frame}{Words are taken from \code{column} (or the first column
#'     when \code{column} is \code{NULL}).}
#'   \item{File path}{Supports two file types:
#'     \itemize{
#'       \item \strong{Plain text} (\code{.txt} or no extension): one word per
#'         line, blank lines are ignored.
#'       \item \strong{Tabular} (\code{.csv} or \code{.tsv}): read as a
#'         data frame and words extracted from \code{column} (or the first
#'         column).
#'     }
#'   }
#' }
#'
#' The result is typically combined with the built-in list before passing to
#' \code{\link{fit_bertopic}}:
#' \preformatted{
#'   domain_words <- load_stopwords("domain_stop.txt")
#'   fit <- fit_bertopic(enc, docs,
#'                       extra_stopwords = domain_words)
#' }
#'
#' @param source A character vector of words, a file path, or a data frame.
#' @param column Name of the column to use when \code{source} is a data frame
#'   or tabular file.  Defaults to the first column.
#' @return A deduplicated character vector of stopwords (lowercased and
#'   whitespace-trimmed).
#' @export
load_stopwords <- function(source, column = NULL) {
  words <- if (is.data.frame(source)) {
    col <- column %||% names(source)[1L]
    if (!col %in% names(source))
      stop("Column '", col, "' not found in data frame.")
    as.character(source[[col]])

  } else if (is.character(source) && length(source) == 1L &&
             !is.na(source) && file.exists(source)) {
    ext <- tolower(tools::file_ext(source))
    if (ext %in% c("csv", "tsv")) {
      sep <- if (ext == "tsv") "\t" else ","
      df  <- utils::read.csv(source, sep = sep, stringsAsFactors = FALSE,
                              header = TRUE)
      col <- column %||% names(df)[1L]
      if (!col %in% names(df))
        stop("Column '", col, "' not found in '", source, "'.")
      as.character(df[[col]])
    } else {
      lines <- readLines(source, warn = FALSE)
      lines[nchar(trimws(lines)) > 0L]
    }

  } else if (is.character(source)) {
    source

  } else {
    stop("'source' must be a character vector, a file path, or a data frame.")
  }

  unique(tolower(trimws(words[!is.na(words) & nchar(trimws(words)) > 0L])))
}

# =============================================================================
# Build "id_word1_word2_word3_word4" labels for a vector of topic IDs.
# @keywords internal
.generate_topic_labels <- function(topic_terms, topics, nr_words = 4L) {
  setNames(lapply(topics, function(t) {
    tt <- topic_terms[topic_terms$topic == t, ]
    tt <- tt[order(tt$rank), ]
    words <- tt$term[seq_len(min(nr_words, nrow(tt)))]
    paste0(t, "_", paste(words, collapse = "_"))
  }), as.character(topics))
}
