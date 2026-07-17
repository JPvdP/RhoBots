# =============================================================================
# install.R  --  One-time setup helper for Rhobots dependencies.
# =============================================================================

#' Set up Rhobots system dependencies
#'
#' Installs the torch C++ backend (libtorch + lantern, ~560 MB) if it is not
#' already available.  On Windows it also reminds the user to install the
#' Microsoft Visual C++ Redistributable 2022, which torch requires to load its
#' native libraries.
#'
#' Call this function once immediately after installing the package:
#'
#' ```r
#' library(Rhobots)
#' rhobots_install()
#' # Restart R when prompted, then load the library again.
#' ```
#'
#' On Windows, install the Visual C++ Redistributable **before** calling this
#' function if you see a `lantern.dll` error.  The installer is available at:
#' \url{https://aka.ms/vs/17/release/vc_redist.x64.exe}
#'
#' @param reinstall If `TRUE`, reinstall the torch backend even if it already
#'   appears to be working (default `FALSE`).  Use this when torch loads but
#'   produces unexpected errors.
#' @return Invisible `NULL`.
#' @examples
#' \dontrun{
#'   rhobots_install()
#' }
#' @export
rhobots_install <- function(reinstall = FALSE) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop(
      "The 'torch' package is not installed.\n",
      "Run install.packages('torch') and then call rhobots_install() again."
    )

  if (.Platform$OS.type == "windows") {
    message(
      "Windows detected.\n",
      "torch requires the Microsoft Visual C++ Redistributable 2022.\n",
      "If torch has not loaded correctly, download and install it from:\n",
      "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n",
      "Restart Windows after installation, then re-run rhobots_install().\n"
    )
  }

  if (torch::torch_is_installed() && !reinstall) {
    message("torch backend is already installed and working. Nothing to do.")
    return(invisible(NULL))
  }

  message("Installing torch C++ backend (~560 MB). This may take several minutes...")
  torch::install_torch(reinstall = reinstall)
  message(
    "\nInstallation complete.\n",
    "IMPORTANT: restart your R session now, then run:\n",
    "  library(Rhobots)\n",
    "to verify that everything loads correctly."
  )
  invisible(NULL)
}
