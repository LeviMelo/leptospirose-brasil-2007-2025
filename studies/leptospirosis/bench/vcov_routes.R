#!/usr/bin/env Rscript
# Can the reported exposure-lag-response surface be obtained WITHOUT storing a
# full posterior configuration?  BREPI-001 asks this explicitly, because
# `control.compute(config = TRUE)` is what made the stopped runs expensive: it
# retains a full latent GMRF configuration for every hyperparameter integration
# point, and under CCD there are dozens of those.
#
# `reduce_cumulative()` needs only the coefficients and the covariance of the
# cross-basis block -- a 9 x 9 matrix.  Three routes could supply it.  This
# script fits the same cheap structure under each and reports whether the
# matrix is available, what it costs, and whether the routes agree.
#
# Usage: Rscript studies/leptospirosis/bench/vcov_routes.R [scale]

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(INLA) })

args <- commandArgs(trailingOnly = TRUE)
SCALE <- if (length(args) >= 1) args[1] else "health_region"
cache <- readRDS(file.path("data/interim/bench", paste0("model_data_", SCALE, ".rds")))
dt <- cache$data; meta <- cache$meta
.cb <- meta$cb_terms; .x <- meta$x_terms

# M2 keeps the spatial field (so the fixed-effect posterior is genuinely
# correlated) while still fitting in well under a minute.
PC_BYM2 <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                phi  = list(prior = "pc", param = c(0.5, 0.5)))
rhs <- c("1", .x, .cb, sprintf(
  "f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
  meta$graph, paste(deparse(PC_BYM2), collapse = "")), "offset(log_offset)")
form <- as.formula(paste("cases ~", paste(rhs, collapse = " + ")))

base_args <- list(
  formula = form, family = "nbinomial", data = dt,
  control.inla = list(strategy = "gaussian", int.strategy = "eb"),
  control.predictor = list(compute = FALSE, link = 1),
  num.threads = "4:1"
)

timed <- function(label, extra) {
  t0 <- Sys.time()
  fit <- do.call(inla, c(base_args, extra))
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  list(label = label, fit = fit, seconds = round(el, 1),
       mb = round(as.numeric(object.size(fit)) / 1024^2, 1))
}

results <- list()

# ---- Route A: control.fixed(correlation.matrix = TRUE) --------------------
# If INLA returns the fixed-effect posterior correlation directly, this is the
# cheapest possible route: one small dense matrix, no latent configuration.
a <- timed("A: control.fixed(correlation.matrix)", list(
  control.fixed = list(mean = 0, prec = 1, mean.intercept = 0,
                       prec.intercept = 0.001, correlation.matrix = TRUE),
  control.compute = list(config = FALSE, return.marginals.predictor = FALSE)
))
corrA <- a$fit$misc$lincomb.derived.correlation.matrix
if (is.null(corrA)) {
  # Some versions expose it here instead.
  corrA <- tryCatch(a$fit$misc$configs$correlation.matrix, error = function(e) NULL)
}
cat(sprintf("[A] %.1fs, %.1f MB, correlation matrix available: %s\n",
            a$seconds, a$mb, !is.null(corrA)))
if (!is.null(corrA)) cat("    dim:", paste(dim(corrA), collapse = " x "), "\n")

# ---- Route B: config = TRUE, read Qinv for the fixed block ---------------
b <- timed("B: config = TRUE", list(
  control.fixed = list(mean = 0, prec = 1, mean.intercept = 0, prec.intercept = 0.001),
  control.compute = list(config = TRUE, return.marginals.predictor = FALSE)
))
vcovB <- NULL
try({
  cfg <- b$fit$misc$configs
  idx <- cfg$contents
  pos <- match(.cb, idx$tag)
  starts <- idx$start[pos]
  Q <- cfg$config[[1]]$Qinv
  vcovB <- as.matrix(Q[starts, starts])
  # INLA stores Qinv with only the lower triangle populated in some builds.
  vcovB[upper.tri(vcovB)] <- t(vcovB)[upper.tri(vcovB)]
}, silent = TRUE)
cat(sprintf("[B] %.1fs, %.1f MB, cross-basis vcov available: %s\n",
            b$seconds, b$mb, !is.null(vcovB)))

# ---- Route C: explicit lincombs ------------------------------------------
# Ask INLA for the posterior of the exact linear combinations the surface
# needs. Exact marginals, no normal approximation, no stored configuration.
# Here: the cumulative log-RR at the 95th vs the median exposure.
cb <- cache$crossbasis
xg <- c(stats::median(cb$x_model, na.rm = TRUE),
        stats::quantile(cb$x_model, 0.95, na.rm = TRUE))
pred <- dlnm::crosspred(cb$cb, coef = rep(0, ncol(cb$cb)),
                        vcov = diag(ncol(cb$cb)), model.link = "log",
                        at = xg, cen = xg[1])
# crosspred's design row for the cumulative effect at the high value:
wrow <- tryCatch(pred$allRRfit, error = function(e) NULL)
lc_weights <- NULL
try({
  # Rebuild the cumulative design row directly from the basis.
  bmat <- dlnm::crossbasis(x = xg, lag = attr(cb$cb, "lag"),
                           argvar = attr(cb$cb, "argvar"),
                           arglag = attr(cb$cb, "arglag"))
  lc_weights <- colSums(bmat[!is.na(rowSums(bmat)), , drop = FALSE])
}, silent = TRUE)

if (!is.null(lc_weights) && length(lc_weights) == length(.cb)) {
  lcs <- inla.make.lincombs(setNames(as.list(lc_weights), .cb))
  names(lcs) <- "cumulative_p95_vs_median"
  cc <- timed("C: lincombs", list(
    control.fixed = list(mean = 0, prec = 1, mean.intercept = 0, prec.intercept = 0.001),
    control.compute = list(config = FALSE, return.marginals.predictor = FALSE),
    lincomb = lcs
  ))
  lcout <- cc$fit$summary.lincomb.derived
  cat(sprintf("[C] %.1fs, %.1f MB, lincomb available: %s\n",
              cc$seconds, cc$mb, !is.null(lcout)))
  if (!is.null(lcout)) print(round(lcout[, c("mean", "sd", "0.025quant", "0.975quant")], 4))
} else {
  cat("[C] could not rebuild the cumulative design row; skipped\n")
}

# ---- Agreement ------------------------------------------------------------
if (!is.null(corrA) && !is.null(vcovB)) {
  sdA <- a$fit$summary.fixed[.cb, "sd"]
  sub <- corrA[.cb, .cb, drop = FALSE]
  vcovA <- diag(sdA) %*% sub %*% diag(sdA)
  rel <- max(abs(vcovA - vcovB) / pmax(abs(vcovB), 1e-12))
  cat(sprintf("\n[agreement] max relative difference between route A and B vcov: %.3e\n", rel))
}

cat("\n[summary] object sizes tell the story: route B retains a latent\n")
cat("[summary] configuration; A and C do not. Under CCD, B stores one per\n")
cat("[summary] integration point, which is where the stopped runs' memory went.\n")
