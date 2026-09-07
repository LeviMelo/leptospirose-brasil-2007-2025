#!/usr/bin/env Rscript
# RQ2 specification -- DEFINITIONS ONLY, no side effects.
#
# Sourced by the orchestration script and, independently, by every bounded child
# process that fits one specification. It must never run an analysis, print, or
# read command-line arguments: sourcing happens once per child.
#
# The estimand is deliberately narrow. `att_0_3` is the mean group-time ATT over
# event months 0 to 3, because that is where a flood-attributable case can be:
# leptospirosis incubates in 2-30 days, and SINAN adds a median 7 days to
# notification plus about 10 more to digitisation. An effect first appearing at
# +8 months is not leptospirosis from that flood.

suppressPackageStartupMessages({ library(data.table) })

RQ2_PANEL <- "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"

#' Load and cache the panel and a declaration table once per process.
.rq2_state <- function(spec) {
  key <- paste0(".brepi_rq2_", spec$events)
  if (!exists(key, envir = globalenv())) {
    panel <- read_panel(RQ2_PANEL, validate = FALSE, prepare = TRUE)
    dec <- data.table::as.data.table(
      arrow::read_parquet(file.path("data/panel", spec$events)))
    assign(key, list(panel = panel, dec = dec), envir = globalenv())
  }
  get(key, envir = globalenv())
}

#' Coarsen a monthly DiD panel to a courser time grain.
#'
#' `did::att_gt` estimates one doubly-robust cell per (cohort, period). At
#' 200 monthly cohorts x 228 months that is 45,600 cells and an influence
#' function of n x 45,600 -- measured to be beyond this machine. Coarsening the
#' time index divides the cell count by the square of the grain, at the cost of
#' event-time resolution. The grain is a declared part of the specification, not
#' a silent implementation detail, and the monthly evidence is retained through
#' the Sun-Abraham and episodic distributed-lag fits, which do not have this
#' cost structure.
rq2_coarsen <- function(dp, grain = 1L, outcome = "rate",
                        baseline_cols = character()) {
  coarsen_did_panel(dp, grain = grain, outcome = outcome,
                    baseline_cols = baseline_cols)
}

# Freeze structural characteristics at the first panel period. did::att_gt()
# requires time-invariant covariates in a panel design; passing contemporaneous
# sanitation/GDP would condition on variables that can change after treatment.
rq2_freeze_baseline <- function(dp) {
  source_cols <- c("sanitation_sewer_share", "urban_share",
                   "gdp_per_capita_asinh")
  miss <- setdiff(source_cols, names(dp))
  if (length(miss)) stop("RQ2 baseline covariate(s) absent: ",
                         paste(miss, collapse = ", "), call. = FALSE)
  out_cols <- c("baseline_sanitation_z", "baseline_urban_z", "baseline_gdp_z")
  firsts <- dp[order(period), lapply(.SD, data.table::first),
               by = unit_id, .SDcols = source_cols]
  for (j in seq_along(source_cols)) {
    x <- as.numeric(firsts[[source_cols[j]]])
    if (any(!is.finite(x))) {
      stop("RQ2 baseline covariate has missing/non-finite values: ",
           source_cols[j], call. = FALSE)
    }
    s <- stats::sd(x)
    if (!is.finite(s) || s <= 0) stop("RQ2 baseline covariate has no variance: ",
                                      source_cols[j], call. = FALSE)
    firsts[, (out_cols[j]) := (x - mean(x)) / s]
  }
  dp[firsts[, c("unit_id", out_cols), with = FALSE], on = "unit_id"]
}

rq2_build <- function(spec) {
  st <- .rq2_state(spec)
  dp <- build_did_panel(
    st$panel, st$dec,
    cobrade_include = spec$cobrade,
    outcome = spec$outcome,
    exclude_recurrent_within = spec$exclude_recurrent_within)
  baseline_cols <- spec$baseline_cols %||% character()
  if (length(baseline_cols)) dp <- rq2_freeze_baseline(dp)
  if (!is.null(spec$endemic_min_cases)) {
    # Outcome-conditioned subset. The estimand becomes "the effect where
    # leptospirosis is reported at all", which must be said, not laundered.
    tot <- dp[, .(total = sum(cases)), by = munic_code]
    dp <- dp[munic_code %in% tot[total >= spec$endemic_min_cases, munic_code]]
  }
  rq2_coarsen(dp, spec$grain %||% 1L, outcome = spec$outcome,
              baseline_cols = baseline_cols)
}

rq2_fit <- function(spec, ...) {
  suppressPackageStartupMessages({ library(data.table); library(did) })
  dp <- rq2_build(spec)
  run_callaway_santanna(
    dp,
    xformla = spec$xformla %||% NULL,
    control_group = spec$control_group,
    est_method = spec$est_method,
    anticipation = spec$anticipation %||% 1L,
    event_window = c(spec$min_e, spec$max_e),
    bstrap = spec$bstrap %||% TRUE,
    cband = spec$cband %||% TRUE,
    seed = spec$seed %||% 20260730L,
    n_boot = spec$n_boot %||% 1000L)
}

rq2_estimand <- function(fit, spec, ...) {
  es <- data.table::as.data.table(fit$tidy)
  # Event months 0-3 at monthly grain; the same calendar span at a coarser one.
  hi <- max(0L, as.integer(round(3 / (spec$grain %||% 1L))))
  win <- es[event_time >= 0 & event_time <= hi]
  pre <- es[event_time < 0 & se > 0]
  data.frame(
    att_0_3 = mean(win$att, na.rm = TRUE),
    att_0 = es[event_time == 0]$att[1],
    att_pre_mean = mean(pre$att, na.rm = TRUE),
    att_pre_max_abs = if (nrow(pre)) max(abs(pre$att), na.rm = TRUE) else NA_real_,
    pre_excludes_zero = if (nrow(pre)) any(pre$lo > 0 | pre$hi < 0) else NA,
    overall_att = as.numeric(fit$overall$overall.att),
    overall_se = as.numeric(fit$overall$overall.se),
    dynamic_att = as.numeric(fit$event_study$overall.att),
    dynamic_se = as.numeric(fit$event_study$overall.se)
  )
}

#' One bounded child, one fitted `att_gt`, every derived table.
#'
#' The group-time object carries an n x (groups x periods) influence function
#' and cannot be returned across the process boundary at this size. Everything
#' that needs it -- the event study, the two aggregations, the horizon sweep
#' that measures recurrence contamination, and the HonestDiD bounds -- is
#' therefore computed in the child, and only compact tables come back.
rq2_bundle <- function(spec, horizons = c(3L, 6L, 12L, 17L, 24L, 36L),
                       honest_e = c(0L, 1L), honest_M = c(0, 0.5, 1, 1.5, 2)) {
  suppressPackageStartupMessages({ library(data.table); library(did) })
  dp <- rq2_build(spec)
  shape <- data.table::data.table(
    n_units = data.table::uniqueN(dp$unit_id),
    n_periods = data.table::uniqueN(dp$period),
    n_treated = data.table::uniqueN(dp[first_treat > 0]$unit_id),
    n_never = data.table::uniqueN(dp[first_treat == 0]$unit_id),
    n_cohorts = data.table::uniqueN(dp[first_treat > 0]$first_treat))
  cs <- run_callaway_santanna(
    dp, xformla = spec$xformla %||% NULL,
    control_group = spec$control_group, est_method = spec$est_method,
    anticipation = spec$anticipation %||% 1L,
    event_window = c(spec$min_e, spec$max_e),
    bstrap = spec$bstrap %||% TRUE,
    cband = spec$cband %||% TRUE,
    seed = spec$seed %||% 20260730L, n_boot = spec$n_boot %||% 1000L)

  hz <- try(aggregate_horizons(
    cs, horizons = horizons, bstrap = spec$bstrap %||% TRUE,
    cband = spec$cband %||% TRUE), silent = TRUE)
  hd <- lapply(honest_e, function(e) {
    h <- try(honest_did(cs, e = e, Mvec = honest_M), silent = TRUE)
    if (inherits(h, "try-error")) {
      return(data.table::data.table(event_time = e, Mbar = NA_real_,
                                    lb = NA_real_, ub = NA_real_,
                                    method = "FAILED",
                                    note = trimws(as.character(h))))
    }
    rb <- data.table::as.data.table(h$robust)
    orig <- data.table::as.data.table(h$original)
    rb <- data.table::rbindlist(list(rb, orig), fill = TRUE)
    rb[, event_time := e]
    rb[, breakdown_M := h$breakdown_M]
    rb[]
  })

  gt <- data.table::data.table(
    group = cs$att_gt$group, time = cs$att_gt$t, att = cs$att_gt$att,
    se = cs$att_gt$se)
  gt[, event_time := time - group]

  list(shape = shape,
       event_study = cs$tidy,
       overall = cs$overall_tidy,
       group_time = gt[event_time >= -12L & event_time <= 24L],
       horizons = if (inherits(hz, "try-error")) NULL else hz,
       honest = data.table::rbindlist(hd, fill = TRUE),
       pretrend = cs$pretrend_test,
       spec = spec)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
