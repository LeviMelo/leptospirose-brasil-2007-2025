#!/usr/bin/env Rscript
# One checkpointed RQ2 Callaway-Sant'Anna fit.
#
# This runner exists because long multi-fit orchestration obscured which model
# was active on Windows. Each invocation owns exactly one fit and writes a
# status record plus a complete, prefixed result bundle.
# Usage: Rscript studies/leptospirosis/15b_rq2_one.R JOB OUTCOME GRAIN BSTRAP

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow); library(did) })
for (f in c("R/00_io.R", "R/05_did_flood.R",
            "studies/leptospirosis/rq2_spec.R")) source(f)

args <- commandArgs(trailingOnly = TRUE)
JOB <- if (length(args) >= 1L) args[1L] else "primary"
OUTCOME <- if (length(args) >= 2L) args[2L] else "rate"
GRAIN <- if (length(args) >= 3L) as.integer(args[3L]) else 2L
BSTRAP <- if (length(args) >= 4L) identical(tolower(args[4L]), "true") else TRUE
OUTDIR <- "data/results/rq2_flood_disasters"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

spec <- list(
  events = "flood_events.parquet",
  cobrade = c("121", "122", "123", "13214"),
  control_group = "notyettreated", est_method = "dr",
  outcome = OUTCOME, anticipation = 0L, grain = GRAIN,
  min_e = -as.integer(ceiling(6 / GRAIN)),
  max_e = as.integer(floor(12 / GRAIN)),
  exclude_recurrent_within = NULL, endemic_min_cases = NULL,
  baseline_cols = c("baseline_sanitation_z", "baseline_urban_z",
                    "baseline_gdp_z"), xformla = NULL,
  bstrap = BSTRAP, cband = BSTRAP,
  seed = 20260730L, n_boot = 1000L)

spec <- switch(JOB,
  primary = spec,
  recurrence_clean = modifyList(spec, list(exclude_recurrent_within = 12L)),
  nevertreated = modifyList(spec, list(control_group = "nevertreated")),
  drought_placebo = modifyList(spec, list(
    events = "drought_events.parquet", cobrade = c("1411", "1412"))),
  strict_cobrade = modifyList(spec, list(cobrade = c("121", "122", "123"))),
  one_period_anticipation = modifyList(spec, list(anticipation = 1L)),
  endemic_only = modifyList(spec, list(endemic_min_cases = 20L)),
  stop("unknown RQ2 job: ", JOB, call. = FALSE))

prefix <- paste0("corrected_", JOB,
                 if (OUTCOME == "rate") "" else paste0("_", OUTCOME))
status_path <- file.path(OUTDIR, paste0(prefix, "_status.csv"))
started <- Sys.time()
cat(sprintf("[%s] build/fit start %s | outcome=%s grain=%d bootstrap=%s\n",
            JOB, format(started), OUTCOME, GRAIN, BSTRAP))
flush.console()

value <- try(rq2_bundle(
  spec, horizons = sort(unique(as.integer(round(c(3, 6, 12, 17, 24, 36) /
                                                      GRAIN))))), silent = TRUE)
elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
if (inherits(value, "try-error")) {
  fwrite(data.table(job = JOB, ok = FALSE, outcome = OUTCOME, grain = GRAIN,
                    bootstrap = BSTRAP, seconds = elapsed,
                    error = trimws(as.character(value))), status_path)
  stop(value)
}

fwrite(value$shape, file.path(OUTDIR, paste0(prefix, "_shape.csv")))
fwrite(value$event_study,
       file.path(OUTDIR, paste0(prefix, "_event_study.csv")))
fwrite(value$overall, file.path(OUTDIR, paste0(prefix, "_overall.csv")))
fwrite(value$group_time,
       file.path(OUTDIR, paste0(prefix, "_group_time.csv")))
if (!is.null(value$horizons))
  fwrite(value$horizons, file.path(OUTDIR, paste0(prefix, "_horizons.csv")))
if (!is.null(value$honest))
  fwrite(value$honest, file.path(OUTDIR, paste0(prefix, "_honest_did.csv")))
if (JOB == "primary" && OUTCOME == "rate") {
  design_panel <- rq2_build(spec)
  diagnostics <- treatment_assignment_diagnostics(
    design_panel, spec$baseline_cols)
  fwrite(diagnostics$balance,
         file.path(OUTDIR, "corrected_primary_assignment_balance.csv"))
  fwrite(diagnostics$overlap,
         file.path(OUTDIR, "corrected_primary_assignment_overlap.csv"))
}
fwrite(data.table(
  job = JOB, ok = TRUE, outcome = OUTCOME, grain = GRAIN,
  bootstrap = BSTRAP, seconds = elapsed,
  n_pre_periods = value$pretrend$n_pre_periods,
  max_abs_pre_att = value$pretrend$max_abs_pre_att,
  any_pre_excludes_zero = value$pretrend$any_pre_excludes_zero,
  error = NA_character_), status_path)
cat(sprintf("[%s] complete in %.1f s\n", JOB, elapsed))
