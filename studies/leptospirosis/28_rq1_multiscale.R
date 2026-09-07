#!/usr/bin/env Rscript
# RQ1 across analytic scales -- the modifiable areal unit problem as a result.
#
# The protocol is explicit: "Movement of estimated cumulative risk, peak lag and
# uncertainty across scales is itself a result. It must not be hidden as a
# technical robustness appendix."
#
# The study's central design choice is that the canonical DATA grain
# (municipality-month) and the primary FITTING grain (health region-month) are
# different, on the argument that a distributed-lag surface is not identified
# against a 97.2%-zero outcome. That argument is only credible if the behaviour
# of the estimate across scales is shown rather than asserted.
#
# Three things to watch, and they can disagree:
#   * the POINT estimate -- aggregation averages over within-unit exposure
#     heterogeneity, which attenuates a convex exposure-response toward the
#     mean and should push the p95 RR DOWN as units get bigger;
#   * the INTERVAL -- coarser units have more cases each, so the interval
#     should NARROW even as the point estimate degrades;
#   * the ZERO SHARE -- the mechanism behind both.
# An estimate that gets tighter and smaller as you aggregate is not converging
# on the truth; it is converging on a different estimand.
#
# Every scale uses the SAME model form and the SAME confirmatory profile. A
# scale comparison whose scales also differ in specification measures nothing.
#
# Usage: Rscript studies/leptospirosis/28_rq1_multiscale.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(INLA); library(dlnm)
})
SRC <- c("R/00_io.R", "R/02_crossbasis.R", "R/03_inla_spacetime.R",
         "R/09_exec.R", "studies/leptospirosis/rq1_scale_spec.R")
for (f in SRC) source(f)

OUT <- "data/results/rq1_exposure_response/multiscale"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "6:1")
PKGS <- c("data.table", "arrow", "INLA", "dlnm")

# Coarse scales are cheap; the endemic-municipality tier is the expensive one
# and is also the one most likely to fail to identify. Both outcomes are
# reportable, so it gets a larger budget rather than an exemption.
BUDGETS <- list(
  uf                   = 900,
  microregion          = 1200,
  immediate_region     = 1200,
  health_region        = 1200,
  municipality_endemic = 2400
)
LEVELS <- c("uf", "microregion", "immediate_region", "health_region",
            "municipality_endemic")

rows <- list()
for (lv in LEVELS) {
  cat("\n", strrep("-", 68), "\n[scale] ", lv, "\n", sep = "")
  budget <- limits(wall_seconds = BUDGETS[[lv]], max_rss_mb = 20000,
                   poll_seconds = 5)
  r <- bounded_call(
    function(level, threads) {
      obj <- scale_fit(list(level = level, threads = threads))
      scale_estimand(obj, list(level = level))
    },
    args = list(level = lv, threads = THREADS),
    budget = budget, source_files = SRC, packages = PKGS, label = lv)
  print(r)
  if (identical(r$status, "ok")) {
    rows[[lv]] <- cbind(as.data.table(r$value),
                        seconds = r$wall_seconds, peak_rss_mb = r$peak_rss_mb,
                        ok = TRUE, error = NA_character_)
  } else {
    # A scale that cannot be fitted is a finding about that scale, not a gap in
    # the table. It is recorded with the same columns so the comparison still
    # reads as a comparison.
    rows[[lv]] <- data.table(
      level = lv, n_units = NA_integer_, n_cells = NA_integer_,
      nonzero_share = NA_real_, rr_p95 = NA_real_, rr_lo = NA_real_,
      rr_hi = NA_real_, ci_width = NA_real_, peak_lag = NA_real_,
      dic = NA_real_, waic = NA_real_, cpo_failures = NA_integer_,
      vcov_route = NA_character_, seconds = r$wall_seconds,
      peak_rss_mb = r$peak_rss_mb, ok = FALSE,
      error = sprintf("[%s] %s", r$status, r$error %||% ""))
  }
}

res <- rbindlist(rows, fill = TRUE)
ref <- res[level == "health_region"]
if (nrow(ref) && isTRUE(ref$ok)) {
  res[, rel_to_primary := (rr_p95 - ref$rr_p95) / abs(ref$rr_p95)]
  res[, ci_width_ratio := ci_width / ref$ci_width]
}
fwrite(res, file.path(OUT, "rq1_scale_comparison.csv"))

cat("\n", strrep("=", 68), "\n", sep = "")
cat("RQ1 ACROSS ANALYTIC SCALES (primary = health_region)\n")
cat(strrep("=", 68), "\n", sep = "")
print(res[, .(level, units = n_units, cells = n_cells,
              nonzero = round(nonzero_share, 4),
              RR = round(rr_p95, 3), lo = round(rr_lo, 3), hi = round(rr_hi, 3),
              width = round(ci_width, 3),
              rel = round(rel_to_primary, 3),
              lag = peak_lag, ok, secs = round(seconds, 1))])

fitted <- res[ok == TRUE]
if (nrow(fitted) >= 2) {
  cat("\n-- reading --\n")
  cat(sprintf("Point estimate spans %.3f to %.3f across %d fitted scales.\n",
              min(fitted$rr_p95), max(fitted$rr_p95), nrow(fitted)))
  cat(sprintf("Interval width spans %.3f to %.3f.\n",
              min(fitted$ci_width), max(fitted$ci_width)))
  cat(sprintf("Non-zero cell share spans %.1f%% to %.1f%%.\n",
              100 * min(fitted$nonzero_share), 100 * max(fitted$nonzero_share)))
  co <- suppressWarnings(stats::cor(fitted$n_units, fitted$rr_p95,
                                    use = "complete.obs"))
  cat(sprintf("Correlation between number of units and the estimate: %+.3f\n", co))
  cat("A monotone relationship here is the modifiable areal unit problem,\n")
  cat("not noise: aggregation averages over within-unit exposure variation,\n")
  cat("which attenuates a convex exposure-response toward the mean.\n")
}
failed <- res[ok == FALSE]
if (nrow(failed)) {
  cat("\n-- scales that did not fit, reported not dropped --\n")
  for (i in seq_len(nrow(failed))) {
    cat(sprintf("  %-22s %s\n", failed$level[i], substr(failed$error[i], 1, 100)))
  }
}

assert_no_strays(kill = TRUE)
cat("\nwrote", OUT, "\n")
