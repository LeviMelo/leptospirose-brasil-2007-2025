#!/usr/bin/env Rscript
# RQ1 - health-region spatiotemporal DLNM for precipitation and leptospirosis.
#
# Runs the model under a declared execution profile and writes the estimands the
# manuscript reports: the cumulative exposure-response curve, the
# exposure-lag-response surface, the lag-response at a high exposure percentile,
# and the structural coefficients.
#
# The profile is not cosmetic and is recorded in every output:
#
#   development   gaussian latent strategy, empirical-Bayes hyperparameter
#                 integration, no CPO, no DIC/WAIC. Same structure as the
#                 confirmatory fit and roughly an order of magnitude cheaper.
#                 Point estimates are comparable; interval width is NOT, because
#                 empirical Bayes conditions on the posterior mode of the
#                 hyperparameters instead of integrating over it. Never quote an
#                 interval from this profile.
#   confirmatory  simplified-Laplace latent strategy, CCD integration over the
#                 hyperparameters, DIC/WAIC and CPO. This is the fit a claim may
#                 cite.
#
# Neither profile stores a posterior configuration. The reported surface reduces
# from the fixed-effect posterior correlation matrix, which is a 13 x 13 object;
# retaining a full latent configuration per CCD integration point was a
# principal cost of the runs that had to be abandoned (BREPI-001).
#
# Usage:
#   Rscript studies/leptospirosis/11_fit_rq1.R development [scale]
#   Rscript studies/leptospirosis/11_fit_rq1.R confirmatory [scale]

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm)
})
for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R")) source(file.path("R", f))

args <- commandArgs(trailingOnly = TRUE)
PROFILE <- if (length(args) >= 1) args[1] else "development"
SCALE   <- if (length(args) >= 2) args[2] else "health_region"
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "8:1")

stopifnot(PROFILE %in% c("validation", "development", "confirmatory"))

cachefile <- file.path("data/interim/bench", paste0("model_data_", SCALE, ".rds"))
if (!file.exists(cachefile)) {
  stop("run studies/leptospirosis/bench/00_prepare_model_data.R ", SCALE, " first")
}
cache <- readRDS(cachefile)
dt <- cache$data; meta <- cache$meta; cb <- cache$crossbasis
.cb <- meta$cb_terms; .x <- meta$x_terms

prof <- model_execution_profile(PROFILE, num_threads = THREADS)
cc <- prof$controls

PC_PREC <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
PC_BYM2 <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                phi  = list(prior = "pc", param = c(0.5, 0.5)))
dep <- function(x) paste(deparse(x), collapse = "")

rhs <- c(
  "1", .x, .cb,
  sprintf("f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
          meta$graph, dep(PC_BYM2)),
  sprintf("f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)", dep(PC_PREC)),
  sprintf("f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)",
          dep(PC_PREC)),
  sprintf("f(id_st_year, model='iid', constr=TRUE, hyper=%s)", dep(PC_PREC)),
  "offset(log_offset)"
)
form <- as.formula(paste("cases ~", paste(rhs, collapse = " + ")))

cat("[rq1] profile:", PROFILE, "(", prof$scientific_status, ")\n")
cat("[rq1] scale:", SCALE, "| units:", meta$n_space, "| rows:", nrow(dt), "\n")

t0 <- Sys.time()
fit <- inla(
  form, family = "nbinomial", data = dt,
  control.compute = cc$control.compute,
  control.inla = cc$control.inla,
  control.predictor = cc$control.predictor,
  control.fixed = cc$control.fixed,
  num.threads = THREADS, verbose = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("[rq1] fitted in", round(elapsed, 1), "s\n")

# --- estimands ------------------------------------------------------------
coefs <- fit$summary.fixed[.cb, "mean"]
V <- inla_fixed_vcov(fit, .cb)
cat("[rq1] cross-basis covariance route:", attr(V, "route"), "\n")

cumulative <- reduce_cumulative(cb, coef = coefs, vcov = V)
surface <- exposure_lag_surface(cb, coef = coefs, vcov = V)

structural <- as.data.table(fit$summary.fixed[.x, ], keep.rownames = "term")
setnames(structural, c("0.025quant", "0.975quant"), c("lo", "hi"), skip_absent = TRUE)

hyper <- as.data.table(fit$summary.hyperpar, keep.rownames = "hyperparameter")

outdir <- file.path("data/results/rq1_exposure_response", paste0(SCALE, "_", PROFILE))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
fwrite(cumulative, file.path(outdir, "cumulative_exposure_response.csv"))
fwrite(surface$surface, file.path(outdir, "exposure_lag_surface.csv"))
fwrite(surface$lag_curve, file.path(outdir, "lag_response_high_exposure.csv"))
fwrite(structural, file.path(outdir, "structural_coefficients.csv"))
fwrite(hyper, file.path(outdir, "hyperparameters.csv"))

# --- latent fields --------------------------------------------------------
# The model spends most of its degrees of freedom on the spatial field, the
# temporal trend, the seasonal cycle and the residual interaction, and each is
# a scientific object in its own right: where baseline risk is high once
# rainfall is accounted for, what the national trend did, what the fitted
# seasonal shape is. Writing only the cross-basis and the fixed effects threw
# all of that away, and none of it is recoverable from a coefficient table.
#
# `id_space` codes are the health-region codes in the SAME order the fit
# indexed them, so the output is join-ready against geography rather than
# carrying indices only this script can interpret.
space_codes <- sort(unique(dt$area_code))
random <- tidy_random_effects(
  fit,
  labels = list(
    id_space = space_codes,
    id_time = sort(unique(dt$id_time)),
    id_month = sort(unique(dt$id_month)),
    id_st_year = NULL),
  bym2 = "id_space")
fwrite(random, file.path(outdir, "random_effects.csv"))

# Split out the three fields a reader will ask for by name, so nobody has to
# know that `id_space` means "health region" or that a BYM2 term returns two
# stacked halves.
sp <- random[term == "id_space" & component == "combined"]
if (nrow(sp)) {
  setnames(sp, "code", "health_region_code", skip_absent = TRUE)
  fwrite(sp, file.path(outdir, "spatial_field.csv"))
}
tt <- random[term == "id_time"]
if (nrow(tt)) fwrite(tt, file.path(outdir, "temporal_trend.csv"))
ss <- random[term == "id_month"]
if (nrow(ss)) fwrite(ss, file.path(outdir, "seasonal_cycle.csv"))

cat(sprintf("[rq1] latent fields: %d rows across %d term(s)
",
            nrow(random), uniqueN(random$term)))
if (nrow(sp)) {
  cat(sprintf("[rq1] spatial field: %d health regions, effect %.3f to %.3f (90/10 ratio %.2f)
",
              nrow(sp), min(sp$effect), max(sp$effect),
              stats::quantile(sp$effect, 0.9) / stats::quantile(sp$effect, 0.1)))
}

meta_out <- list(
  profile = PROFILE, scientific_status = prof$scientific_status,
  scale = SCALE, elapsed_seconds = round(elapsed, 1),
  n_rows = nrow(dt), n_space = meta$n_space, n_time = meta$n_time,
  vcov_route = attr(V, "route"),
  n_hyperpar = length(fit$mode$theta),
  mlik = as.numeric(fit$mlik[1]),
  dic = tryCatch(fit$dic$dic, error = function(e) NA),
  waic = tryCatch(fit$waic$waic, error = function(e) NA),
  log_score = tryCatch(mean(-log(pmax(fit$cpo$cpo, .Machine$double.eps)), na.rm = TRUE),
                       error = function(e) NA),
  n_cpo_failure = tryCatch(sum(fit$cpo$failure > 0, na.rm = TRUE),
                           error = function(e) NA),
  fitted_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  quotable = identical(PROFILE, "confirmatory")
)
writeLines(jsonlite::toJSON(meta_out, auto_unbox = TRUE, null = "null", digits = 8),
           file.path(outdir, "fit_metadata.json"))

# --- console summary ------------------------------------------------------
# The exposure axis is NOT millimetres. `per_unit_knots = TRUE` maps each health
# region's precipitation to (x - unit median) / unit IQR before a single common
# basis is evaluated, because national raw-mm knots would sit inside the
# Amazonian interquartile range and outside the semi-arid support entirely.
# Every RR below is therefore "per local IQR above the local median", and the
# national distribution of those IQRs is printed so the contrast can be read in
# millimetres for a region of interest.
ustats <- attr(cb$x_model, "unit_stats")
if (!is.null(ustats)) {
  iqr_q <- stats::quantile(ustats$iqr, c(0.1, 0.5, 0.9), na.rm = TRUE)
  cat(sprintf(
    "\n[rq1] exposure scale: unit-relative. Health-region monthly IQR (mm): p10=%.0f p50=%.0f p90=%.0f\n",
    iqr_q[1], iqr_q[2], iqr_q[3]))
}

cat("\n[rq1] cumulative exposure-response, per local IQR above the local median\n")
qs <- stats::quantile(cb$x_model, c(0.50, 0.75, 0.90, 0.95, 0.99), na.rm = TRUE)
for (i in seq_along(qs)) {
  j <- which.min(abs(cumulative$x - qs[i]))
  mm_median <- if (is.null(ustats)) NA_real_ else cumulative$x[j] * stats::median(ustats$iqr, na.rm = TRUE)
  cat(sprintf("   p%-3s  x=%5.2f IQR (~%5.0f mm at the median region)  RR=%5.3f (%5.3f, %5.3f)\n",
              sub("%", "", names(qs)[i]), cumulative$x[j], mm_median,
              cumulative$rr[j], cumulative$rr_lo[j], cumulative$rr_hi[j]))
}
cat(sprintf("\n[rq1] peak lag: %s months (identified only at 1-month resolution); RR at peak: %.3f\n",
            surface$peak$peak_lag, surface$peak$peak_rr))
cat(sprintf("[rq1] cumulative RR at the p%.0f exposure: %.3f (%.3f, %.3f)\n",
            100 * surface$peak$high_percentile, surface$peak$cumulative_rr_at_high,
            surface$peak$cumulative_lo, surface$peak$cumulative_hi))
cat("[rq1] structural coefficients (log rate ratio)\n")
print(structural[, .(term, mean = round(mean, 4), sd = round(sd, 4))])
cat("\n[rq1] wrote", outdir, "\n")
if (!meta_out$quotable) {
  cat("[rq1] NOT QUOTABLE: profile '", PROFILE,
      "' is a development fit. Intervals understate hyperparameter uncertainty.\n", sep = "")
}
