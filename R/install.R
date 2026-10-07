# =============================================================================
# install.R  --  One-time setup helper for Rhobots dependencies.
# =============================================================================

#' Check Rhobots system dependencies and print setup instructions
#'
#' Checks whether the 'torch' C++ backend is installed and prints the
#' appropriate setup instructions.  Rhobots requires the 'torch' backend
#' (libtorch + lantern, ~560 MB) to run transformer models.  Call this
#' function after installing the package to find out what still needs to be
#' done.
#'
#' @return Invisible `NULL`, called for its side-effect of printing instructions.
#' @examples
#' rhobots_install()
#' @export
rhobots_install <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    message(
      "The 'torch' package is not installed.\n",
      "Run the following to install it:\n\n",
      "  install.packages('torch')\n",
      "  torch::install_torch()\n\n",
      "Then restart R and load Rhobots again."
    )
    return(invisible(NULL))
  }

  if (torch::torch_is_installed()) {
    message("torch backend is installed and working. Rhobots is ready to use.")
  } else {
    message(
      "The 'torch' package is installed but the C++ backend is missing.\n",
      "Run the following to install it (~560 MB download):\n\n",
      "  torch::install_torch()\n\n",
      "Then restart R and load Rhobots again."
    )
  }

  if (.Platform$OS.type == "windows") {
    message(
      "\nWindows note: if you see a 'lantern.dll' error after installing the\n",
      "backend, install the Microsoft Visual C++ Redistributable 2022 from:\n",
      "  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n",
      "Then restart Windows and try again."
    )
  }

  invisible(NULL)
}
