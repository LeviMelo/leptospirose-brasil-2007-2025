#!/usr/bin/env Rscript
# RQ2 - do officially recognised flood disasters cause excess leptospirosis
# beyond the smooth rainfall response?
#
# Staggered difference-in-differences on municipal flood declarations, using
# Callaway & Sant'Anna (2021) group-time ATTs with not-yet-treated controls,
# cross-checked against Sun & Abraham via fixest and against an episodic
# (non-absorbing) distributed-lag specification.
#
# This is an extended replication, not a new design. Halmenschlager, Almeida,
# Ribeiro & Trindade (Health Economics 2025;34(5):855-868, doi:10.1002/hec.4939)
# already applied Callaway-Sant'Anna to monthly municipal panels of Brazilian
# hydrological disasters in the Northeast, 2000-2012, with leptospirosis as a
# named outcome. Our deltas: national rather than Northeast, 2007-2025 rather
# than 2000-2012 (so including the 2024 Rio Grande do Sul catastrophe), SINAN
# notifications rather than SIH morbidity, HonestDiD bounds, an explicit
# treatment of treatment recurrence, and a drought placebo.
#
# Three design facts, measured, that travel with every estimate below:
#
#   * 73.3% of treated municipalities are treated more than once (median gap 17
#     months). The absorbing-treatment assumption behind Callaway-Sant'Anna is
#     violated. Three separate specifications address it and are compared.
#   * 252 municipalities are already treated in the first period and have no
#     pre-treatment observation. Any event-study normalisation drops them.
#   * Recognition is an administrative act. Municipal civil-defence capacity is
#     part of treatment assignment and plausibly correlates with health
#     surveillance capacity, which is the outcome-reporting mechanism. The
#     drought placebo exists to test exactly that path.
#
# Usage: Rscript studies/leptospirosis/15_rq2_did.R [outcome] [grain]

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow) })
SRC <- c("R/00_io.R", "R/02_crossbasis.R", "R/05_did_flood.R",
         "R/08_sensitivity.R", "R/09_exec.R",
         "studies/leptospirosis/rq2_spec.R")
for (f in SRC) source(f)

args <- commandArgs(trailingOnly = TRUE)
OUTCOME <- if (length(args) >= 1) args[1] else "rate"
# Time grain in months. `did::att_gt` costs one doubly-robust cell per
# (cohort, period); at 200 monthly cohorts x 228 months that is 45,600 cells and
# was measured infeasible on this machine. See 15a_rq2_probe.R for the
# measurement and data/results/rq2_flood_disasters/cost_probe.csv for the numbers.
GRAIN <- if (length(args) >= 2) as.integer(args[2]) else
  as.integer(Sys.getenv("BREPI_RQ2_GRAIN", "2"))
SEED <- 20260730L
PKGS <- c("data.table", "arrow", "did", "fixest", "HonestDiD", "dlnm")
outdir <- "data/results/rq2_flood_disasters"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Sibling orchestration scripts run on this machine concurrently. Snapshot the
# engines that already exist so the closing sweep kills only what this run left.
PRE_PIDS <- stray_pids()

BUDGET <- limits(wall_seconds = as.numeric(Sys.getenv("BREPI_RQ2_BUDGET", "900")),
                 max_rss_mb = 20000, poll_seconds = 5)
# callr process isolation remains available, but on this Windows workstation
# its launch/poll path has itself stalled for hours before dispatching a child.
# Direct mode reuses the frozen panel cache and is the operational default for
# this study script; each major table is still status-labelled and the outer
# process is supervised by the invoking workflow.
USE_BOUNDED <- identical(tolower(Sys.getenv("BREPI_RQ2_USE_BOUNDED", "false")),
                         "true")
run_job <- function(fun, args, label) {
  if (USE_BOUNDED) {
    return(bounded_call(fun, args = args, budget = BUDGET,
                        source_files = SRC, packages = PKGS, label = label))
  }
  started <- Sys.time()
  value <- try(do.call(fun, args), silent = TRUE)
  if (inherits(value, "try-error")) {
    return(.run("error", NULL, started, NA_real_, label,
                trimws(as.character(value))))
  }
  .run("ok", value, started, NA_real_, label, NA_character_)
}

# Event window. -6..+12 months, expressed in the analysis grain. The upper end
# sits below the 17-month median inter-episode gap on purpose; the horizon sweep
# inside each bundle reports what happens past it.
MIN_E <- -as.integer(ceiling(6 / GRAIN))
MAX_E <- as.integer(floor(12 / GRAIN))
HORIZONS <- sort(unique(as.integer(round(c(3, 6, 12, 17, 24, 36) / GRAIN))))

BASE <- list(
  events = "flood_events.parquet",
  cobrade = c("121", "122", "123", "13214"),
  control_group = "notyettreated",
  est_method = "dr",
  outcome = OUTCOME,
  # At a coarsened grain, one did period is GRAIN months. Treating it as one
  # month of anticipation is wrong; the primary therefore uses none. The
  # one-period alternative is reported explicitly below.
  anticipation = 0L,
  grain = GRAIN,
  min_e = MIN_E, max_e = MAX_E,
  exclude_recurrent_within = NULL,
  endemic_min_cases = NULL,
  # Unconditional parallel trends is the primary identifying assumption.
  # Repeating a propensity/outcome model for every cohort-period cell made the
  # national estimator >5x slower and is not required for a valid unconditional
  # DiD. Baseline imbalance is diagnosed separately; estimator labels are not
  # varied when xformla is NULL because DR/IPW/reg would then be identical.
  baseline_cols = c("baseline_sanitation_z", "baseline_urban_z",
                    "baseline_gdp_z"),
  xformla = NULL,
  bstrap = TRUE, cband = TRUE,
  seed = SEED, n_boot = 1000L
)

say <- function(...) cat(sprintf(...), "\n", sep = "")
say("[rq2] outcome=%s grain=%d month(s); event window [%d, %d] in grain units",
    OUTCOME, GRAIN, MIN_E, MAX_E)

# --------------------------------------------------------------------------
# 0. Treatment description -- cheap, in process
# --------------------------------------------------------------------------
flood <- as.data.table(arrow::read_parquet("data/panel/flood_events.parquet"))
drought <- as.data.table(arrow::read_parquet("data/panel/drought_events.parquet"))
say("[rq2] flood declarations: %s | drought declarations: %s",
    format(nrow(flood), big.mark = ","), format(nrow(drought), big.mark = ","))

rec <- rbindlist(list(
  cbind(hazard = "flood", declaration_recurrence(flood)),
  cbind(hazard = "drought", declaration_recurrence(drought))), fill = TRUE)
print(rec[, .(hazard, municipalities_with_declaration, share_with_more_than_one,
              mean_declarations, median_gap_months)])
fwrite(rec, file.path(outdir, "recurrence.csv"))

# --------------------------------------------------------------------------
# 1. Bounded fits
# --------------------------------------------------------------------------
spec_of <- function(...) modifyList(BASE, list(...))

JOBS <- list(
  primary = spec_of(),
  recurrence_clean = spec_of(exclude_recurrent_within = 12L),
  nevertreated = spec_of(control_group = "nevertreated"),
  drought_placebo = spec_of(events = "drought_events.parquet",
                            cobrade = c("1411", "1412"))
)

runs <- list()
for (nm in names(JOBS)) {
  say("\n[rq2] ---- bounded fit: %s ----", nm)
  r <- run_job(rq2_bundle,
               args = list(spec = JOBS[[nm]], horizons = HORIZONS),
               label = nm)
  print(r)
  runs[[nm]] <- r
}

emit <- function(nm) {
  r <- runs[[nm]]
  if (!identical(r$status, "ok")) {
    say("[rq2] %s did not produce a result (%s): %s", nm, r$status, r$error)
    return(invisible(NULL))
  }
  v <- r$value
  fwrite(v$shape, file.path(outdir, sprintf("%s_shape.csv", nm)))
  fwrite(v$event_study, file.path(outdir, sprintf("%s_event_study.csv", nm)))
  fwrite(v$overall, file.path(outdir, sprintf("%s_overall.csv", nm)))
  fwrite(v$group_time, file.path(outdir, sprintf("%s_group_time.csv", nm)))
  if (!is.null(v$horizons))
    fwrite(v$horizons, file.path(outdir, sprintf("%s_horizons.csv", nm)))
  if (!is.null(v$honest))
    fwrite(v$honest, file.path(outdir, sprintf("%s_honest_did.csv", nm)))
  say("\n== %s ==", nm)
  print(v$shape)
  print(v$event_study)
  print(v$overall)
  if (!is.null(v$horizons)) { say("-- horizon sweep --"); print(v$horizons) }
  if (!is.null(v$honest)) { say("-- HonestDiD --"); print(v$honest) }
  say("-- pre-trend --"); print(unlist(v$pretrend[c(
    "n_pre_periods", "max_abs_pre_att", "any_pre_excludes_zero")]))
  invisible(v)
}
for (nm in names(runs)) emit(nm)

# --------------------------------------------------------------------------
# 2. Sun-Abraham cross-check and the episodic (non-absorbing) alternative,
#    both at MONTHLY resolution -- neither carries att_gt's cell-count cost.
# --------------------------------------------------------------------------
with_rainfall_basis <- function(panel, lag_max = 3L) {
  d <- data.table::copy(data.table::as.data.table(panel))
  data.table::setorderv(d, c("munic_code", "time_index"))
  cbo <- build_crossbasis(
    d, exposure_col = "precip_mm", group_col = "munic_code",
    time_col = "time_index", lag_max = lag_max, per_unit_knots = TRUE)
  suppressWarnings(bind_crossbasis(d, cbo, prefix = "rain_cb",
                                   drop_incomplete = FALSE))
}

sunab_job <- function(spec) {
  suppressPackageStartupMessages({ library(data.table); library(fixest); library(dlnm) })
  st <- .rq2_state(spec)
  dp <- build_did_panel(st$panel, st$dec, cobrade_include = spec$cobrade,
                        outcome = spec$outcome)
  rain <- with_rainfall_basis(dp)
  designs <- list(
    list(label = "unadjusted", data = dp, terms = character()),
    list(label = "rainfall_dlnm_lag0_3", data = rain$data, terms = rain$terms))
  out <- list()
  for (design in designs) {
    # Poisson is the population-offset count model. OLS is retained only for
    # the unadjusted transformed-outcome comparison, avoiding an expensive
    # fourth fit whose coefficient has no stable percent interpretation.
    families <- if (design$label == "unadjusted") c("poisson", "ols") else "poisson"
    for (fam in families) {
      key <- paste(design$label, fam, sep = "_")
      m <- try(run_sunab(design$data, family = fam,
                         event_window = c(-6L, 12L),
                         adjust_terms = design$terms), silent = TRUE)
      out[[key]] <- if (inherits(m, "try-error"))
        data.table::data.table(family = fam, rainfall_adjustment = design$label,
                               note = trimws(as.character(m)))
      else cbind(family = fam, rainfall_adjustment = design$label,
                 data.table::as.data.table(m$tidy))
    }
  }
  data.table::rbindlist(out, fill = TRUE)
}

dlag_job <- function(spec) {
  suppressPackageStartupMessages({ library(data.table); library(fixest); library(dlnm) })
  st <- .rq2_state(spec)
  rain <- with_rainfall_basis(st$panel)
  designs <- list(
    list(label = "unadjusted", data = st$panel, terms = character()),
    list(label = "rainfall_dlnm_lag0_3", data = rain$data, terms = rain$terms))
  out <- list()
  for (design in designs) {
    families <- if (design$label == "unadjusted") c("poisson", "ols") else "poisson"
    for (fam in families) {
      key <- paste(design$label, fam, sep = "_")
      m <- try(run_episodic_dlag(
        design$data, st$dec, cobrade_include = spec$cobrade,
        leads = 6L, lags = 12L, family = fam,
        adjust_terms = design$terms), silent = TRUE)
      out[[key]] <- if (inherits(m, "try-error"))
        data.table::data.table(family = fam, rainfall_adjustment = design$label,
                               note = trimws(as.character(m)))
      else cbind(family = fam, rainfall_adjustment = design$label,
                 n_episodes = m$n_episodes, n_units = m$n_units_with_episode,
                 data.table::as.data.table(m$tidy))
    }
  }
  data.table::rbindlist(out, fill = TRUE)
}

for (cfg in list(
  list(nm = "sunab", fn = sunab_job, spec = BASE),
  list(nm = "sunab_drought", fn = sunab_job, spec = JOBS$drought_placebo),
  list(nm = "episodic_dlag", fn = dlag_job, spec = BASE),
  list(nm = "episodic_dlag_drought", fn = dlag_job, spec = JOBS$drought_placebo)
)) {
  say("\n[rq2] ---- bounded fit: %s ----", cfg$nm)
  r <- run_job(cfg$fn, args = list(spec = cfg$spec), label = cfg$nm)
  print(r)
  runs[[cfg$nm]] <- r
  if (identical(r$status, "ok")) {
    fwrite(r$value, file.path(outdir, sprintf("%s.csv", cfg$nm)))
    print(r$value)
  }
}

# --------------------------------------------------------------------------
# 3. Prespecified sensitivity grid
# --------------------------------------------------------------------------
PERTURBATIONS <- list(
  perturbation(
    "strict_cobrade", "Only COBRADE 1.2.x (inundacao/enxurrada/alagamento).",
    "exposure", list(cobrade = c("121", "122", "123")),
    rationale = paste("The textbook hydrological definition. Known to lose most",
                      "of the RS 2024 catastrophe, so a large change here is a",
                      "statement about coding practice, not about flooding.")),
  perturbation(
    "nevertreated_controls", "Never-treated municipalities as controls.",
    "specification", list(control_group = "nevertreated"),
    rationale = paste("Never-treated municipalities are drier, smaller and",
                      "administratively weaker, so this is the less defensible",
                      "control group; it is reported to show what it does.")),
  perturbation(
    "asinh_outcome", "asinh of period counts rather than incidence rate.",
    "outcome", list(outcome = "asinh"),
    rationale = paste("asinh(count) and the population-normalised rate weight",
                      "municipalities differently. Both are constructed only",
                      "after raw counts and person-time are aggregated."),
    # A rate and asinh(count) are not the same quantity; the coefficient sits on
    # a different scale entirely, so a relative-change threshold against the
    # reference is meaningless. Read the SIGN and the interval, not the ratio.
    comparable = FALSE),
  perturbation(
    "any_outcome", "Binary any-case-in-period rather than incidence rate.",
    "outcome", list(outcome = "any"),
    rationale = paste("At 97% zeros most of the information is in whether a",
                      "municipality reports at all; this is where a detection",
                      "effect would show up most clearly."),
    comparable = FALSE),
  perturbation(
    "one_period_anticipation", "One coarsened-period anticipation window.",
    "specification", list(anticipation = 1L),
    rationale = paste("Declarations are often filed after the event. At this",
                      "grain the alternative excludes", GRAIN,
                      "months, not one month; that distinction is explicit.")),
  perturbation(
    "endemic_only", "Municipalities with at least 20 confirmed cases.",
    "population", list(endemic_min_cases = 20L),
    rationale = paste("Outcome-conditioned, so the estimand changes to the",
                      "effect where leptospirosis is reported at all. Reported",
                      "because a national zero can hide a real effect in the",
                      "1% of municipalities carrying the case mass."),
    # Outcome-conditioned: the analysed population changes, so this estimates a
    # different quantity on a different outcome distribution.
    comparable = FALSE),
  perturbation(
    "recurrence_clean", "Drop units with a second declaration within 12 months.",
    "population", list(exclude_recurrent_within = 12L),
    rationale = paste("The absorbing-treatment assumption holds over the event",
                      "window for the surviving units. It selects on a",
                      "post-treatment event and loses most treated units, so it",
                      "is a bound, not a preferred specification.")),
  perturbation(
    "drought_placebo", "Drought declarations instead of floods.", "exposure",
    list(events = "drought_events.parquet", cobrade = c("1411", "1412")),
    rationale = paste("Identical bureaucratic pathway, opposite hydrology, no",
                      "plausible leptospirosis mechanism. A non-null effect",
                      "means the design reads administrative capacity rather",
                      "than water, which would be decisive against it."),
    # A placebo is SUPPOSED to differ from the reference. Judging it on the
    # materiality threshold reports the design working as a design failing.
    comparable = FALSE)
)

grid <- sensitivity_grid(BASE, PERTURBATIONS)
print(grid)
res <- if (USE_BOUNDED) {
  run_sensitivity(grid, "rq2_fit", "rq2_estimand", on_error = "record",
                  budget = BUDGET, source_files = SRC, packages = PKGS)
} else {
  run_sensitivity(grid, rq2_fit, rq2_estimand, on_error = "record")
}
report <- sensitivity_report(res, estimand = "att_0_3", threshold = 0.30)
fwrite(report, file.path(outdir, "rq2_sensitivity.csv"))

say("\n================ RQ2 SENSITIVITY ================")
print(report[, .(run_id, class, ok, att_0_3 = round(att_0_3, 4),
                 att_pre = round(att_pre_mean, 4),
                 rel = round(relative_change, 3), note, secs = seconds)])
verdict <- sensitivity_verdict(report)
say("\n---- verdict ----"); print(verdict)
writeLines(as.character(verdict), file.path(outdir, "verdict.txt"))

saveRDS(lapply(runs, function(r) r[c("status", "wall_seconds", "peak_rss_mb",
                                     "error", "value")]),
        file.path(outdir, "rq2_runs.rds"))
say("\n[rq2] run status:")
print(data.table(job = names(runs),
                 status = vapply(runs, `[[`, character(1), "status"),
                 secs = vapply(runs, `[[`, numeric(1), "wall_seconds"),
                 rss_mb = vapply(runs, `[[`, numeric(1), "peak_rss_mb")))

assert_no_strays(kill = TRUE, exclude_pids = PRE_PIDS)
say("\nwrote %s", outdir)
