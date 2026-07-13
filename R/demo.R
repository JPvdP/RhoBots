# =============================================================================
# demo.R — Self-contained demonstration of the Rhobots pipeline.
# =============================================================================

# Common words that dominate 19th-century narrative prose but carry no topic
# signal — they swamp c-TF-IDF when only unigrams are used.
.literary_stopwords <- c(
  "said", "one", "upon", "great", "little", "came", "went", "man", "men",
  "mr", "mrs", "miss", "sir", "now", "still", "yet", "also", "well",
  "just", "back", "away", "long", "even", "much", "every", "never",
  "thee", "thou", "thy", "hath", "doth", "ere", "tis", "twas",
  "chapter", "gutenberg", "project", "ebook", "license", "pg"
)

#' Run a quick Rhobots demo using classic novels from Project Gutenberg
#'
#' Downloads a selection of classic novels, splits them into paragraphs, and
#' runs the full BERTopic pipeline: embed → UMAP → HDBSCAN → c-TF-IDF.
#' Prints a topic summary and shows two interactive plots: a 2-D topic map
#' and a bar chart of top terms per topic.
#'
#' The encoder used is `sentence-transformers/all-MiniLM-L6-v2` (~22 MB),
#' which is downloaded once and cached by \pkg{hfhub}.  Subsequent runs skip
#' the download.
#'
#' @param n_per_book Maximum paragraphs to sample per book (default 150).
#'   Lower values are faster; higher values give richer, more stable topics.
#' @param device Device for the encoder: `"cpu"` (default), `"mps"` (Apple
#'   Silicon), or `"cuda"` (NVIDIA GPU).
#' @param seed Integer random seed for UMAP and sampling (default 42).
#' @param verbose Print progress messages (default `TRUE`).
#' @return Invisibly, a list with elements `fit`, `embeddings`, `encoder`,
#'   and `texts` (a data frame with columns `text` and `title`), so you can
#'   continue exploring after the demo.
#' @export
rhobots_demo <- function(n_per_book = 150L,
                          device     = "cpu",
                          seed       = 42L,
                          verbose    = TRUE) {

  if (!requireNamespace("gutenbergr", quietly = TRUE))
    stop("Package 'gutenbergr' is required for the demo.\n",
         "Install it with: install.packages('gutenbergr')")

  # ── [1] Download books ────────────────────────────────────────────────────
  # Six books chosen for maximal thematic contrast across genres and settings.
  book_ids <- c(
    1342L,   # Pride and Prejudice      — Jane Austen       (romance/society)
    84L,     # Frankenstein             — Mary Shelley      (gothic/science)
    2701L,   # Moby-Dick                — Herman Melville   (adventure/sea)
    1661L,   # The Adventures of Sherlock Holmes — Conan Doyle (mystery)
    36L,     # The War of the Worlds    — H.G. Wells        (science fiction)
    74L      # The Adventures of Tom Sawyer — Mark Twain    (childhood/frontier)
  )
  book_labels <- c(
    "Pride and Prejudice",
    "Frankenstein",
    "Moby-Dick",
    "Sherlock Holmes",
    "The War of the Worlds",
    "Tom Sawyer"
  )

  if (verbose)
    message("Downloading ", length(book_ids),
            " classic novels from Project Gutenberg...")

  raw <- tryCatch(
    gutenbergr::gutenberg_download(book_ids,
                                   meta_fields  = "title",
                                   verbose      = FALSE),
    error = function(e)
      stop("Could not reach Project Gutenberg:\n  ", conditionMessage(e),
           "\nCheck your internet connection and try again.")
  )

  # ── [2] Extract paragraphs ────────────────────────────────────────────────
  # Use 250-char minimum: short fragments are noisy and drag down topic quality.
  .to_paragraphs <- function(lines, min_chars = 250L) {
    paras <- character(0)
    buf   <- character(0)
    for (ln in lines) {
      if (nchar(trimws(ln)) == 0L) {
        if (length(buf)) {
          paras <- c(paras, paste(trimws(buf), collapse = " "))
          buf   <- character(0)
        }
      } else {
        buf <- c(buf, ln)
      }
    }
    if (length(buf))
      paras <- c(paras, paste(trimws(buf), collapse = " "))
    paras[nchar(paras) >= min_chars]
  }

  set.seed(seed)
  texts_df <- do.call(rbind, lapply(seq_along(book_ids), function(i) {
    rows  <- raw[raw$gutenberg_id == book_ids[i], ]
    if (nrow(rows) == 0L) {
      warning("No text downloaded for '", book_labels[i], "' (id ", book_ids[i], ")")
      return(NULL)
    }
    paras <- .to_paragraphs(rows$text)
    if (length(paras) == 0L) {
      warning("No paragraphs extracted for '", book_labels[i], "'")
      return(NULL)
    }
    if (length(paras) > n_per_book)
      paras <- sample(paras, n_per_book)
    data.frame(text = paras, title = book_labels[i], stringsAsFactors = FALSE)
  }))

  if (is.null(texts_df) || nrow(texts_df) == 0L)
    stop("No paragraphs could be extracted. Check your internet connection.")

  if (verbose)
    message("  Using ", nrow(texts_df), " paragraphs from ",
            length(unique(texts_df$title)), " books.")

  # ── [3] Embed ─────────────────────────────────────────────────────────────
  if (verbose) message("Loading encoder (all-MiniLM-L6-v2)...")
  enc <- load_hf_bert("sentence-transformers/all-MiniLM-L6-v2")

  if (verbose) message("Embedding paragraphs on '", device, "'...")
  emb <- embed_texts(enc, texts_df$text, device = device, verbose = verbose)

  # ── [4] Fit topic model ───────────────────────────────────────────────────
  if (verbose) message("Fitting topic model...")
  fit <- fit_bertopic(
    docs              = texts_df$text,
    embeddings        = emb,
    # min_pts=5: narrative prose clusters more tightly than academic text;
    # a lower threshold lets HDBSCAN find granular book-specific topics.
    hdbscan_min_pts   = 5L,
    # n_neighbors=30: larger neighbourhood → more global UMAP structure →
    # better separation of books that share the same prose style.
    umap_n_neighbors  = 5L,
    # bigrams capture phrases ("white whale", "dear watson", "mr darcy") that
    # distinguish books far better than individual words alone.
    ngram_range       = c(2L, 3L),
    extra_stopwords   = .literary_stopwords,
    seed              = seed,
    verbose           = verbose
  )

  # ── [5] Results ───────────────────────────────────────────────────────────
  if (verbose) message("\n── Demo results ──────────────────────────────────────")
  print(fit)

  p_topics <- visualize_topics(fit)
  if (!is.null(p_topics)) print(p_topics)

  p_bar <- visualize_barchart(fit)
  if (!is.null(p_bar)) print(p_bar)

  if (verbose)
    message("\nThe returned list contains $fit, $embeddings, $encoder, and ",
            "$texts\nso you can keep exploring — e.g. print_topics(result$fit).")

  invisible(list(fit        = fit,
                 embeddings = emb,
                 encoder    = enc,
                 texts      = texts_df))
}
