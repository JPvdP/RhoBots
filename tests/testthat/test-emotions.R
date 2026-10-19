# Tests for the detect_emotions() front end.
# No internet access, no model downloads: only the preset table, argument
# checks and output formatting (on fake classify_texts() output) are tested.

texts3 <- c("I am thrilled", "This is awful", "The bus leaves at 9")

test_that("emotion_models lists the three presets with repo IDs", {
  m <- emotion_models()
  expect_s3_class(m, "data.frame")
  expect_equal(m$preset, c("ekman", "goemotions", "vad"))
  expect_true(all(grepl("/", m$repo_id)))
})

test_that("single-label output renames label to emotion", {
  raw <- data.frame(text = texts3, label = c("joy", "anger", "neutral"),
                    score = c(0.9, 0.8, 0.7), anger = c(0.05, 0.8, 0.1),
                    joy = c(0.9, 0.1, 0.2), neutral = c(0.05, 0.1, 0.7))
  out <- Rhobots:::.format_emotions(raw, "single_label_classification")
  expect_equal(names(out), c("text", "emotion", "score", "anger", "joy", "neutral"))
  expect_equal(out$emotion, c("joy", "anger", "neutral"))
})

test_that("multi-label output lists emotions above threshold, most likely first", {
  raw <- data.frame(text = texts3,
                    anger = c(0.10, 0.70, 0.01),
                    joy   = c(0.95, 0.02, 0.03),
                    fear  = c(0.60, 0.80, 0.02))
  out <- Rhobots:::.format_emotions(raw, "multi_label_classification", threshold = 0.5)
  expect_equal(names(out), c("text", "emotions", "anger", "joy", "fear"))
  expect_equal(out$emotions, c("joy, fear", "fear, anger", ""))

  out2 <- Rhobots:::.format_emotions(raw, "multi_label_classification", threshold = 0.9)
  expect_equal(out2$emotions, c("joy", "", ""))
})

test_that("regression output gets preset column names", {
  raw <- data.frame(text = texts3, LABEL_0 = c(4.1, 2.0, 3.0),
                    LABEL_1 = c(4.0, 3.5, 3.0), LABEL_2 = c(3.3, 2.5, 3.2))
  out <- Rhobots:::.format_emotions(raw, "regression",
                                    columns = c("valence", "arousal", "dominance"))
  expect_equal(names(out), c("text", "valence", "arousal", "dominance"))
  expect_equal(out$valence, c(4.1, 2.0, 3.0))

  # Without preset column names the model's own labels are kept.
  expect_equal(names(Rhobots:::.format_emotions(raw, "regression")),
               c("text", "LABEL_0", "LABEL_1", "LABEL_2"))
})

test_that("detect_emotions rejects bad input before loading a model", {
  expect_error(detect_emotions(1:3), "character vector")
  expect_error(detect_emotions(c("ok", NA)), "NA values")
  expect_error(detect_emotions("hi", model = "plutchik"), "Unknown emotion preset")
  expect_error(detect_emotions("hi", model = c("a/b", "c/d")), "preset name")
})
