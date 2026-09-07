#!/usr/bin/env Rscript
# RQ1 climate product-splice sensitivity (ISSUE_LEDGER BREPI-007).
#
# The exposure panel is spliced: BR-DWGD through 2024-03-20, ERA5-Land after,
# with March 2024 mixed. 90.3% of municipality-months come from BR-DWGD, 9.2%
# from ERA5-Land. The ERA5 tail is additionally calendar-prorated because the
# published daily subset omits one day per month.
#
# The protocol requires that this be shown not to drive the estimand. The
# obvious test -- refit on the unspliced window only -- is CONFOUNDED, because
# the ERA5 period contains the April-May 2024 Rio Grande do Sul floods, the
# largest single leptospirosis event in the series. Restricting to <= 2023
# therefore removes both the product change AND that outbreak, and a movement
# could be either.
#
# So two fits, not one:
#   unspliced_2023   drop 2024-2025  -> product change + RS 2024 both removed
#   drop_2024        drop 2024 only  -> RS 2024 removed, ERA5 2025 retained
# Comparing them against each other separates the two explanations. If
# `drop_2024` moves the estimate and `unspliced_2023` moves it by about the
# same amount, the mover is the outbreak, not the product.
#
# Usage: Rscript studies/leptospirosis/29_rq1_climate_splice.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm)
})
SRC <- c("R/00_io.R", "R/02_crossbasis.R", "R/03_inla_spacetime.R",
         "R/08_sensitivity.R", "R/09_exec.R",
         "studies/leptospirosis/rq1_spec.R")
for (f in SRC) source(f)

OUT <- "data/results/rq1_exposure_response/climate_splice"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
PKGS <- c("data.table", "INLA", "dlnm")

BASE <- list(
  scale = "health_region",
  threads = Sys.getenv("BREPI_BENCH_THREADS", "6:1"),
  interaction = "type1_year", spatial_prec_u = 1,
  drop_covid = FALSE, adjusted = TRUE, seasonal = TRUE, temporal = TRUE
)

PERTURBATIONS <- list(
  perturbation(
    "unspliced_2023", "Restrict to 2007-2023: BR-DWGD only, no ERA5 tail.",
    "exposure", list(year_max = 2023L),
    rationale = paste("Removes the product splice entirely. Confounded with",
                      "the removal of the 2024 Rio Grande do Sul event, which",
                      "is why drop_2024 is fitted alongside.")),
  perturbation(
    "drop_2024", "Full window minus calendar 2024.", "population",
    list(drop_year = 2024L),
    rationale = paste("Removes the Rio Grande do Sul catastrophe while",
                      "retaining the 2025 ERA5-Land months. Isolates the",
                      "outbreak from the product change."))
)

grid <- sensitivity_grid(BASE, PERTURBATIONS)
print(grid)
BUDGET <- limits(wall_seconds = 900, max_rss_mb = 16000, poll_seconds = 5)

res <- run_sensitivity(grid, "fit_fn", "estimand_fn", on_error = "record",
                       budget = BUDGET, source_files = SRC, packages = PKGS)
report <- sensitivity_report(res, estimand = "rr_p95", threshold = 0.20)
fwrite(report, file.path(OUT, "climate_splice_sensitivity.csv"))

cat("\n============ RQ1 CLIMATE PRODUCT-SPLICE SENSITIVITY ============\n")
print(report[, .(run_id, ok, RR = round(rr_p95, 3), lo = round(rr_lo, 3),
                 hi = round(rr_hi, 3), rel = round(relative_change, 3),
                 note, secs = seconds)])
v <- sensitivity_verdict(report)
cat("\n---- verdict ----\n"); print(v)
writeLines(as.character(v), file.path(OUT, "verdict.txt"))

f <- report[ok == TRUE]
if (nrow(f) == 3L) {
  u <- f[run_id == "unspliced_2023", relative_change]
  d <- f[run_id == "drop_2024", relative_change]
  cat("\n---- attribution ----\n")
  cat(sprintf("unspliced_2023 moves the estimate %+.1f%%\n", 100 * u))
  cat(sprintf("drop_2024      moves the estimate %+.1f%%\n", 100 * d))
  if (abs(u) < 1e-9) {
    cat("The splice does not move the estimate at all.\n")
  } else {
    cat(sprintf("The 2024 outbreak accounts for %.0f%% of the movement seen\n",
                100 * min(1, abs(d) / abs(u))))
    cat("when the ERA5 period is removed; the residual is attributable to\n")
    cat("the product change and to the 2025 months.\n")
  }
}
assert_no_strays(kill = TRUE)
cat("\nwrote", OUT, "\n")
