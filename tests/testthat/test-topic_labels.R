# Tests for custom topic names: set_topic_labels() and their use in
# visualize_topics_over_time().  No model downloads: a mock fit is used.

mock_label_fit <- function() {
  structure(
    list(
      clusters     = c(0L, 0L, 1L, 1L, 2L, -1L),
      topic_labels = list("0"  = "0_crop_soil_farm_food",
                          "1"  = "1_hiv_malaria_child_health",
                          "2"  = "2_school_teacher_pupil_exam",
                          "-1" = "-1_outliers")
    ),
    class = "bertopic_fit"
  )
}

test_that("set_topic_labels accepts a named vector and keeps other labels", {
  fit <- set_topic_labels(mock_label_fit(),
                          c("0" = "Smallholder agriculture", "-1" = "Unassigned"))
  expect_equal(fit$topic_labels[["0"]],  "0_Smallholder agriculture")
  expect_equal(fit$topic_labels[["-1"]], "-1_Unassigned")
  expect_equal(fit$topic_labels[["1"]],  "1_hiv_malaria_child_health")
})

test_that("set_topic_labels accepts a named list", {
  fit <- set_topic_labels(mock_label_fit(), list("2" = "Education"))
  expect_equal(fit$topic_labels[["2"]], "2_Education")
})

test_that("set_topic_labels reads a filled-in spreadsheet, skipping blanks", {
  path <- tempfile(fileext = ".csv")
  write.csv(data.frame(Topic = c(-1L, 0L, 1L, 2L),
                       Name  = c("-1_outliers", "0_crop", "1_hiv", "2_school"),
                       Label = c("", "Agriculture", NA, " Education ")),
            path, row.names = FALSE)
  fit <- set_topic_labels(mock_label_fit(), read.csv(path))
  expect_equal(fit$topic_labels[["0"]], "0_Agriculture")
  expect_equal(fit$topic_labels[["2"]], "2_Education")          # trimmed
  expect_equal(fit$topic_labels[["1"]], "1_hiv_malaria_child_health")
  expect_equal(fit$topic_labels[["-1"]], "-1_outliers")
})

test_that("set_topic_labels gives clear errors", {
  fit <- mock_label_fit()
  expect_error(set_topic_labels(fit, c("7" = "Nope")), "Unknown topic ID")
  expect_error(set_topic_labels(fit, c("Agriculture", "Health")), "named character vector")
  expect_error(set_topic_labels(fit, c("0" = "")), "empty or NA")
  expect_error(set_topic_labels(fit, c("0" = "A", "0" = "B")), "more than once")
  expect_error(set_topic_labels(fit, data.frame(Topic = 0L, Name = "x")), "`Topic` and `Label`")
  expect_error(set_topic_labels(list(), c("0" = "A")), "bertopic_fit")
})

test_that("set_topic_labels suggests read.csv2 for a semicolon CSV read with read.csv", {
  path <- tempfile(fileext = ".csv")
  write.csv2(data.frame(Topic = 0L, Label = "Agriculture"), path, row.names = FALSE)
  expect_error(set_topic_labels(mock_label_fit(), read.csv(path)), "read.csv2")
  fit <- set_topic_labels(mock_label_fit(), read.csv2(path))
  expect_equal(fit$topic_labels[["0"]], "0_Agriculture")
})

test_that("visualize_topics_over_time shows custom names when fit is given", {
  skip_if_not_installed("plotly")
  tot <- data.frame(Topic     = rep(0:1, each = 2L),
                    Words     = "w",
                    Frequency = c(3L, 4L, 2L, 5L),
                    Timestamp = rep(c(2020L, 2021L), 2L))
  fit <- set_topic_labels(mock_label_fit(), c("0" = "Agriculture"))

  names_of <- function(p) vapply(plotly::plotly_build(p)$x$data, `[[`, "", "name")
  expect_equal(names_of(visualize_topics_over_time(tot, fit = fit)),
               c("Agriculture", "hiv malaria child health"))
  expect_equal(names_of(visualize_topics_over_time(tot)),
               c("Topic 0", "Topic 1"))
})
