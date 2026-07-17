# =============================================================================
# zzz.R  --  Package hooks
# =============================================================================

# These roxygen tags make useDynLib/importFrom survive NAMESPACE regeneration.
#' @useDynLib Rhobots, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats chisq.test is.leaf median sd setNames
#' @importFrom utils head
NULL

.onAttach <- function(libname, pkgname) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    packageStartupMessage(
      "Rhobots: 'torch' is not installed.\n",
      "Run install.packages('torch') followed by rhobots_install()."
    )
    return(invisible(NULL))
  }

  if (!torch::torch_is_installed()) {
    msg <- paste0(
      "Rhobots: the torch C++ backend is not ready.\n",
      "Run rhobots_install() to complete setup."
    )
    if (.Platform$OS.type == "windows") {
      msg <- paste0(
        msg, "\n",
        "Windows: first install the Visual C++ Redistributable 2022  -- \n",
        "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
      )
    }
    packageStartupMessage(msg)
  }
}
