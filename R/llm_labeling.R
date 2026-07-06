# =============================================================================
# llm_labeling.R — Generate human-readable topic labels via an LLM API.
#
# Supports Anthropic (Claude), OpenAI (GPT), and Ollama (local) via httr2.
# Ollama uses the OpenAI-compatible endpoint at http://localhost:11434/v1 so
# any model pulled with `ollama pull <model>` works without an API key.
# =============================================================================

.check_httr2 <- function() {
  if (!requireNamespace("httr2", quietly = TRUE))
    stop(
      "Package 'httr2' is required for LLM labeling.\n",
      "Install it with:  install.packages(\"httr2\")"
    )
}

.build_label_prompt <- function(fit, topic_id, top_n_terms,
                                 n_docs, custom_prompt) {
  tt    <- fit$topic_terms[fit$topic_terms$topic == topic_id, ]
  tt    <- tt[order(tt$rank), ]
  terms <- paste(head(tt$term, top_n_terms), collapse = ", ")

  reps     <- fit$representative_docs[[as.character(topic_id)]]
  reps     <- head(reps, n_docs)
  docs_str <- paste(sprintf('  %d. "%s"', seq_along(reps), reps), collapse = "\n")

  if (!is.null(custom_prompt)) {
    p <- custom_prompt
    p <- gsub("{terms}", terms,    p, fixed = TRUE)
    p <- gsub("{docs}",  docs_str, p, fixed = TRUE)
    return(p)
  }

  paste0(
    "You are labeling topics discovered by a text mining algorithm.\n",
    "Based on the top terms and representative documents below, provide a ",
    "concise topic label of 3 to 5 words that captures the core theme.\n\n",
    "Top terms: ", terms, "\n\n",
    "Representative documents:\n", docs_str, "\n\n",
    "Respond with ONLY the topic label — no punctuation at the end, ",
    "no explanation, no quotes."
  )
}

.call_anthropic <- function(prompt, api_key, model) {
  .check_httr2()
  model <- model %||% "claude-haiku-4-5-20251001"
  req <- httr2::request("https://api.anthropic.com/v1/messages") |>
    httr2::req_headers(
      `x-api-key`         = api_key,
      `anthropic-version` = "2023-06-01",
      `content-type`      = "application/json"
    ) |>
    httr2::req_body_json(list(
      model      = model,
      max_tokens = 32L,
      messages   = list(list(role = "user", content = prompt))
    )) |>
    httr2::req_error(is_error = function(r) FALSE)

  resp <- httr2::req_perform(req)
  if (httr2::resp_status(resp) != 200L)
    stop("Anthropic API returned status ", httr2::resp_status(resp), ": ",
         httr2::resp_body_string(resp))
  trimws(httr2::resp_body_json(resp)$content[[1L]]$text)
}

.call_openai <- function(prompt, api_key, model) {
  .check_httr2()
  model <- model %||% "gpt-4o-mini"
  req <- httr2::request("https://api.openai.com/v1/chat/completions") |>
    httr2::req_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ) |>
    httr2::req_body_json(list(
      model      = model,
      max_tokens = 32L,
      messages   = list(list(role = "user", content = prompt))
    )) |>
    httr2::req_error(is_error = function(r) FALSE)

  resp <- httr2::req_perform(req)
  if (httr2::resp_status(resp) != 200L)
    stop("OpenAI API returned status ", httr2::resp_status(resp), ": ",
         httr2::resp_body_string(resp))
  trimws(httr2::resp_body_json(resp)$choices[[1L]]$message$content)
}

.call_ollama <- function(prompt, api_key, model) {
  .check_httr2()
  model <- model %||% "llama3.2"
  req <- httr2::request("http://localhost:11434/v1/chat/completions") |>
    httr2::req_headers(`Content-Type` = "application/json") |>
    httr2::req_body_json(list(
      model      = model,
      max_tokens = 32L,
      messages   = list(list(role = "user", content = prompt))
    )) |>
    httr2::req_error(is_error = function(r) FALSE) |>
    httr2::req_timeout(120)

  resp <- httr2::req_perform(req)
  if (httr2::resp_status(resp) != 200L)
    stop("Ollama returned status ", httr2::resp_status(resp),
         ". Is Ollama running? Start it with: ollama serve\n",
         httr2::resp_body_string(resp))
  trimws(httr2::resp_body_json(resp)$choices[[1L]]$message$content)
}

#' Label topics using a large language model
#'
#' For each non-noise topic, builds a prompt containing the top c-TF-IDF terms
#' and up to \code{n_representative_docs} representative documents, sends it to
#' an LLM API, and stores the returned label in \code{fit$topic_labels}.
#'
#' Three providers are supported:
#' \describe{
#'   \item{\code{"anthropic"}}{Anthropic Claude via the Messages API. Requires
#'     \code{api_key} or the \code{ANTHROPIC_API_KEY} environment variable.}
#'   \item{\code{"openai"}}{OpenAI GPT via the Chat Completions API. Requires
#'     \code{api_key} or \code{OPENAI_API_KEY}.}
#'   \item{\code{"ollama"}}{Local Ollama server (OpenAI-compatible endpoint at
#'     \code{http://localhost:11434/v1}). No API key required. Start the server
#'     with \code{ollama serve} and pull a model with
#'     \code{ollama pull llama3.2} before use.}
#' }
#'
#' Requires the \pkg{httr2} package.
#'
#' @param fit A \code{bertopic_fit} object from \code{\link{fit_bertopic}}.
#' @param provider LLM provider: \code{"anthropic"} (default), \code{"openai"},
#'   or \code{"ollama"} (local, no API key needed).
#' @param api_key API key string.  Ignored for \code{"ollama"}.  Defaults to
#'   the relevant environment variable for cloud providers.
#' @param model Model identifier.  Defaults to \code{"claude-haiku-4-5-20251001"}
#'   (Anthropic), \code{"gpt-4o-mini"} (OpenAI), or \code{"llama3.2"} (Ollama).
#' @param top_n_terms Number of top terms included in the prompt (default 10).
#' @param n_representative_docs Number of representative documents included
#'   in the prompt (default 3).
#' @param custom_prompt Optional character string to override the default
#'   prompt template.  Use \code{\{terms\}} and \code{\{docs\}} as placeholders.
#' @param verbose Print each returned label as it arrives (default \code{TRUE}).
#' @return The input \code{fit} with updated \code{$topic_labels}.
#' @export
label_topics_llm <- function(fit,
                              provider              = c("anthropic", "openai", "ollama"),
                              api_key               = NULL,
                              model                 = NULL,
                              top_n_terms           = 10L,
                              n_representative_docs = 3L,
                              custom_prompt         = NULL,
                              verbose               = TRUE) {
  provider <- match.arg(provider)

  if (provider != "ollama" && is.null(api_key)) {
    env_var <- if (provider == "anthropic") "ANTHROPIC_API_KEY" else "OPENAI_API_KEY"
    api_key <- Sys.getenv(env_var)
    if (!nzchar(api_key))
      stop("Provide 'api_key' or set the ", env_var, " environment variable.")
  }

  call_fn <- switch(provider,
    anthropic = .call_anthropic,
    openai    = .call_openai,
    ollama    = .call_ollama
  )

  topics_nn <- sort(setdiff(unique(fit$clusters), -1L))
  if (length(topics_nn) == 0L)
    stop("No non-noise topics to label.")

  if (verbose)
    message("Requesting LLM labels for ", length(topics_nn),
            " topics via ", provider, "...")

  for (t in topics_nn) {
    prompt <- .build_label_prompt(fit, t, top_n_terms,
                                   n_representative_docs, custom_prompt)
    label <- tryCatch(
      call_fn(prompt, api_key, model),
      error = function(e) {
        warning("Failed to label Topic ", t, ": ", conditionMessage(e))
        NULL
      }
    )
    if (!is.null(label) && nzchar(trimws(label))) {
      fit$topic_labels[[as.character(t)]] <- paste0(t, "_", trimws(label))
      if (verbose) message("  Topic ", t, ": ", label)
    }
  }

  fit
}
