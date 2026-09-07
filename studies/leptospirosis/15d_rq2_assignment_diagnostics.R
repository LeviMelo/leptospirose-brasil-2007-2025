#!/usr/bin/env Rscript
# Baseline treatment-assignment diagnostics for the corrected primary RQ2.
# This is intentionally separated from the expensive ATT fit: diagnostics can
# be regenerated without holding a fitted group-time influence-function matrix.

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow) })
for (f in c("R/00_io.R", "R/05_did_flood.R",
            "studies/leptospirosis/rq2_spec.R")) source(f)

outdir <- "data/results/rq2_flood_disasters"
baseline <- c("baseline_sanitation_z", "baseline_urban_z", "baseline_gdp_z")
spec <- list(
  events = "flood_events.parquet", cobrade = c("121", "122", "123", "13214"),
  outcome = "rate", grain = 2L, exclude_recurrent_within = NULL,
  endemic_min_cases = NULL, baseline_cols = baseline)

panel <- rq2_build(spec)
diagnostics <- treatment_assignment_diagnostics(panel, baseline)
fwrite(diagnostics$balance,
       file.path(outdir, "corrected_primary_assignment_balance.csv"))
fwrite(diagnostics$overlap,
       file.path(outdir, "corrected_primary_assignment_overlap.csv"))
cat(sprintf("assignment diagnostics: %d covariates, %d units\n",
            nrow(diagnostics$balance), uniqueN(panel$unit_id)))
