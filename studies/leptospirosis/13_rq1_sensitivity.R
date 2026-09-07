#!/usr/bin/env Rscript
# RQ1 robustness battery.
#
# The primary estimate is the cumulative rate ratio at the p95 local rainfall
# anomaly relative to the local median. This script re-estimates it under every
# prespecified perturbation and reports how far it moves. Materiality is
# declared before the run -- 20% relative change, or any sign change -- so
# "the estimate was robust" is a checkable claim rather than an impression.
#
# Runs use the `development` profile: the battery compares *point estimates*
# across specifications, and benchmarking showed those are stable between
# profiles (p95 RR 3.039 under both) at roughly half the cost. Interval width
# is not comparable across profiles and is not compared here; the confirmatory
# fit supplies the reported interval.
#
# Each fit executes in its own bounded child process. `fit_fn` and
# `estimand_fn` are therefore written to be **self-contained**: they reconstruct
# every object they need from `spec` and from files on disk, and close over
# nothing. A closure over the parent's globals would serialise silently and
# fail in the child, and the resulting error would look like a model problem.
#
# Usage: Rscript studies/leptospirosis/13_rq1_sensitivity.R [scale]

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm)
})
for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R", "09_exec.R",
            "08_sensitivity.R")) {
  source(file.path("R", f))
}
SPEC_FILE <- "studies/leptospirosis/rq1_spec.R"
source(SPEC_FILE)

args <- commandArgs(trailingOnly = TRUE)
SCALE <- if (length(args) >= 1) args[1] else "health_region"
THRESHOLD <- 0.20

# --------------------------------------------------------------------------
# Everything below the line runs in a child process. It takes `spec` and
# nothing else from the parent.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Specification and prespecified perturbations
# --------------------------------------------------------------------------
BASE <- list(
  scale = SCALE,
  threads = Sys.getenv("BREPI_BENCH_THREADS", "6:1"),
  interaction = "type1_year",
  spatial_prec_u = 1,
  drop_covid = FALSE,
  adjusted = TRUE,
  seasonal = TRUE,
  temporal = TRUE
)

PERTURBATIONS <- list(
  perturbation(
    "no_interaction", "No residual space-time interaction.", "specification",
    list(interaction = "none"),
    rationale = paste("The interaction is a nuisance term; removing it bounds",
                      "how much of the exposure signal it absorbs.")),
  perturbation(
    "interaction_monthly", "Type I interaction at monthly resolution.", "specification",
    list(interaction = "type1_month"),
    rationale = paste("Tests whether the annual interaction resolution, chosen",
                      "for cost, discards information the monthly one retains.")),
  perturbation(
    "interaction_kh2", "Knorr-Held type II interaction (annual).", "specification",
    list(interaction = "type2_year"),
    rationale = paste("The structured alternative and the conventional choice,",
                      "so it must be shown. Benchmarking found it does not",
                      "converge at this scale; the budget makes that a reported",
                      "outcome rather than a stalled battery.")),
  perturbation(
    "no_seasonality", "Drop the cyclic seasonal term.", "specification",
    list(seasonal = FALSE),
    rationale = paste("Rainfall is strongly seasonal; this bounds how much of",
                      "the exposure-response is carried by season rather than",
                      "by anomalies within season.")),
  perturbation(
    "no_temporal_trend", "Drop the national monthly random walk.", "specification",
    list(temporal = FALSE),
    rationale = "Tests dependence on the shared national temporal field."),
  perturbation(
    "unadjusted", "No structural covariates.", "specification",
    list(adjusted = FALSE),
    rationale = paste("The structural block is weakly identified against the",
                      "spatial field; this shows whether it matters at all for",
                      "the rainfall estimate.")),
  perturbation(
    "exclude_covid", "Exclude 2020-2021.", "population",
    list(drop_covid = TRUE),
    rationale = paste("The COVID period is a documented surveillance trough",
                      "(confirmation ratio and notification volume both fall),",
                      "so it perturbs the outcome process, not transmission.")),
  perturbation(
    "wide_spatial_prior", "Permissive PC prior on the BYM2 precision.", "prior",
    list(spatial_prec_u = 5),
    rationale = paste("PC priors are weakly informative by design but the",
                      "spatial field competes with the covariates; a five-fold",
                      "looser scale tests that competition."))
)

grid <- sensitivity_grid(BASE, PERTURBATIONS)
print(grid)

# Enforced by process isolation (R/09_exec.R). In-process mechanisms were
# measured not to interrupt an INLA fit at all. 900 s is ample against a ~200 s
# reference fit, so anything reaching it is pathological and the failure is the
# finding.
BUDGET <- limits(wall_seconds = as.numeric(Sys.getenv("BREPI_SENS_BUDGET", "900")),
                 max_rss_mb = 12000, poll_seconds = 5)

res <- run_sensitivity(
  grid, "fit_fn", "estimand_fn", on_error = "record", budget = BUDGET,
  source_files = c("R/00_io.R", "R/02_crossbasis.R", "R/03_inla_spacetime.R",
                   SPEC_FILE),
  packages = c("data.table", "INLA", "dlnm")
)
report <- sensitivity_report(res, estimand = "rr_p95", threshold = THRESHOLD)

outdir <- file.path("data/results/rq1_exposure_response", paste0(SCALE, "_sensitivity"))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
fwrite(report, file.path(outdir, "rq1_sensitivity.csv"))

cat("\n================ RQ1 SENSITIVITY ================\n")
print(report[, .(run_id, class, ok, RR = round(rr_p95, 3),
                 lo = round(rr_lo, 3), hi = round(rr_hi, 3),
                 rel = round(relative_change, 3), note,
                 secs = seconds, rss = peak_rss_mb)])
verdict <- sensitivity_verdict(report)
cat("\n---- verdict ----\n"); print(verdict)
writeLines(as.character(verdict), file.path(outdir, "verdict.txt"))
assert_no_strays(kill = TRUE)
cat("\nwrote", outdir, "\n")
