# =============================================================================
# emotions.R  --  Simple front end for emotion detection
#
# detect_emotions() wraps load_hf_classifier() + classify_texts() for the
# three kinds of emotion model taught most often:
#
#   "ekman"       one discrete emotion per text (Ekman's six + neutral)
#   "goemotions"  28 fine-grained emotions, several can apply to one text
#   "vad"         continuous valence, arousal and dominance scores
#
# Presets point at checked models whose weights load in R (safetensors) and
# whose output matches Python `transformers`.  Any other HuggingFace repo ID,
# or an already loaded hf_classifier, works too.  Loaded models are cached
# for the R session so repeated calls do not reload them.
# =============================================================================


# Preset models.  `columns` renames the model's output columns where its
# config has no label names (vad-bert outputs LABEL_0..2, documented on its
# model card as valence, arousal, dominance).
.EMOTION_PRESETS <- list(
  ekman = list(
    repo_id     = "NetworkIsLife/emotion-english-distilroberta-base",
    description = "Ekman's six basic emotions + neutral, one per text",
    source      = "j-hartmann/emotion-english-distilroberta-base (Hartmann, 2022)",
    columns     = NULL
  ),
  goemotions = list(
    repo_id     = "SamLowe/roberta-base-go_emotions",
    description = "28 emotions (GoEmotions); several can apply to one text",
    source      = "Demszky et al. (2020), GoEmotions",
    columns     = NULL
  ),
  vad = list(
    repo_id     = "NetworkIsLife/vad-bert",
    description = "valence, arousal and dominance on the EmoBank 1-5 scale",
    source      = "RobroKools/vad-bert, trained on EmoBank (Buechel & Hahn, 2017)",
    columns     = c("valence", "arousal", "dominance")
  )
)

# Session cache of loaded classifiers, keyed by repo ID.
.emotion_cache <- new.env(parent = emptyenv())


#' List the emotion models available as presets
#'
#' Shows the presets that [detect_emotions()] accepts as `model`, with the
#' HuggingFace model behind each one and what it outputs.
#'
#' @return A `data.frame` with columns `preset`, `repo_id`, `output` and
#'   `source`.
#' @export
#' @examples
#' emotion_models()
emotion_models <- function() {
  data.frame(
    preset  = names(.EMOTION_PRESETS),
    repo_id = vapply(.EMOTION_PRESETS, `[[`, "", "repo_id"),
    output  = vapply(.EMOTION_PRESETS, `[[`, "", "description"),
    source  = vapply(.EMOTION_PRESETS, `[[`, "", "source"),
    row.names = NULL, stringsAsFactors = FALSE
  )
}


#' Detect emotions in texts
#'
#' A simple front end to [load_hf_classifier()] and [classify_texts()] for
#' emotion analysis.  Choose a preset (see [emotion_models()]) or pass any
#' HuggingFace text-classification model.  The model is downloaded once and
#' kept in memory for the rest of the R session.
#'
#' The output depends on the kind of model:
#' \itemize{
#'   \item **One emotion per text** (e.g. `"ekman"`): columns `text`,
#'     `emotion` (the most likely emotion), `score` (its probability), and one
#'     probability column per emotion.  Probabilities sum to 1 per text.
#'   \item **Several emotions per text** (e.g. `"goemotions"`): columns
#'     `text`, `emotions` (all emotions with probability >= `threshold`,
#'     most likely first, comma-separated; `""` if none), and one probability
#'     column per emotion.  Each probability is independent, so they do not
#'     sum to 1.
#'   \item **Continuous scores** (e.g. `"vad"`): columns `text` plus one
#'     numeric column per dimension (`valence`, `arousal`, `dominance`).
#'     For `"vad"` the scale is EmoBank's 1 (low) to 5 (high), with 3 as
#'     neutral.
#' }
#' The model used is printed and stored in `attr(result, "model")`, so
#' results can be traced back to it.
#'
#' @param texts A character vector of texts.
#' @param model A preset name (`"ekman"`, `"goemotions"`, `"vad"`; see
#'   [emotion_models()]), a HuggingFace repo ID, or a classifier already
#'   loaded with [load_hf_classifier()].
#' @param threshold For models that allow several emotions per text: minimum
#'   probability for an emotion to be listed in the `emotions` column.
#'   Default 0.5.
#' @param batch_size Number of texts per forward pass.  Default 32.
#' @param max_length Texts longer than this many tokens are truncated.
#'   Default 512.
#' @param max_tokens Upper limit on padded tokens per forward pass, which
#'   keeps memory use moderate on long texts.  Default 4096; `NULL` for no
#'   limit.  See [classify_texts()].
#' @param verbose Print batch progress.  Default `FALSE`.
#' @return A `data.frame` with one row per text (see Details).
#' @export
#' @examples
#' \dontrun{
#'   texts <- c("I just got the job offer!", "My exam was cancelled again.")
#'
#'   detect_emotions(texts)                       # Ekman's six + neutral
#'   detect_emotions(texts, model = "goemotions") # 28 emotions
#'   detect_emotions(texts, model = "vad")        # valence/arousal/dominance
#'
#'   # Any HuggingFace emotion model
#'   detect_emotions(texts, model = "bhadresh-savani/distilbert-base-uncased-emotion")
#' }
detect_emotions <- function(texts,
                            model      = "ekman",
                            threshold  = 0.5,
                            batch_size = 32L,
                            max_length = 512L,
                            max_tokens = 4096,
                            verbose    = FALSE) {
  if (!is.character(texts))
    stop("`texts` must be a character vector.")
  if (anyNA(texts))
    stop("`texts` contains NA values; remove or replace them first.")

  preset <- if (is.character(model) && length(model) == 1L)
    .EMOTION_PRESETS[[model]] else NULL
  clf    <- .get_emotion_classifier(model, preset)

  if (clf$task == "ner")
    stop("'", clf$repo_id, "' is a token-classification (NER) model, not ",
         "an emotion classifier.  Use classify_texts() for NER.")

  label <- if (!is.null(preset)) {
    sprintf("%s -- %s", clf$repo_id, preset$description)
  } else {
    sprintf("%s (%s)", clf$repo_id, clf$problem_type)
  }
  message("Model: ", label)

  raw <- classify_texts(clf, texts,
                        batch_size = batch_size,
                        max_length = max_length,
                        max_tokens = max_tokens,
                        verbose    = verbose)

  out <- .format_emotions(raw, clf$problem_type, threshold,
                          columns = preset$columns)
  attr(out, "model") <- clf$repo_id
  out
}


# -----------------------------------------------------------------------------
# .get_emotion_classifier  --  resolve `model` to a loaded hf_classifier
# -----------------------------------------------------------------------------

.get_emotion_classifier <- function(model, preset) {
  if (inherits(model, "hf_classifier")) return(model)

  if (!is.character(model) || length(model) != 1L)
    stop("`model` must be a preset name, a HuggingFace repo ID, or a ",
         "classifier from load_hf_classifier().")

  if (is.null(preset) && !grepl("/", model, fixed = TRUE))
    stop("Unknown emotion preset '", model, "'.  Available presets: ",
         paste(names(.EMOTION_PRESETS), collapse = ", "),
         ".  Or pass a HuggingFace repo ID such as \"owner/model-name\".")

  repo_id <- if (!is.null(preset)) preset$repo_id else model
  if (is.null(.emotion_cache[[repo_id]])) {
    .emotion_cache[[repo_id]] <- load_hf_classifier(repo_id)
  }
  .emotion_cache[[repo_id]]
}


# -----------------------------------------------------------------------------
# .format_emotions  --  turn classify_texts() output into the tidy result
#
# raw: data.frame from classify_texts().  Single-label output has columns
# text, label, score, <one per label>; multi-label and regression output has
# text, <one per label>.
# -----------------------------------------------------------------------------

.format_emotions <- function(raw, problem_type, threshold = 0.5, columns = NULL) {
  if (problem_type == "regression") {
    if (!is.null(columns)) names(raw)[-1] <- columns
    return(raw)
  }

  if (problem_type == "multi_label_classification") {
    probs <- as.matrix(raw[, -1, drop = FALSE])
    labs  <- colnames(probs)
    raw$emotions <- vapply(seq_len(nrow(probs)), function(i) {
      p    <- probs[i, ]
      keep <- order(p, decreasing = TRUE)
      keep <- keep[p[keep] >= threshold]
      paste(labs[keep], collapse = ", ")
    }, character(1))
    return(raw[, c("text", "emotions", labs), drop = FALSE])
  }

  # Single label: rename label -> emotion for readability.
  names(raw)[names(raw) == "label"] <- "emotion"
  raw
}
