#!/usr/bin/env Rscript
# Checkpointed monthly RQ2 count-model cross-checks, with and without rainfall.
# Usage: Rscript studies/leptospirosis/15c_rq2_crosschecks.R flood|drought

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(dlnm); library(fixest)
})
for (f in c("R/00_io.R", "R/02_crossbasis.R", "R/05_did_flood.R")) source(f)

args <- commandArgs(trailingOnly = TRUE)
HAZARD <- if (length(args)) args[1L] else "flood"
if (!HAZARD %in% c("flood", "drought")) stop("hazard must be flood or drought")
OUTDIR <- "data/results/rq2_flood_disasters"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

panel <- read_panel(
  "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet",
  validate = FALSE, prepare = TRUE)
events_file <- if (HAZARD == "flood") "flood_events.parquet" else "drought_events.parquet"
cobrade <- if (HAZARD == "flood") c("121", "122", "123", "13214") else c("1411", "1412")
events <- as.data.table(arrow::read_parquet(file.path("data/panel", events_file)))

rain_basis <- function(data, lag_max = 3L) {
  d <- copy(as.data.table(data))
  setorderv(d, c("munic_code", "time_index"))
  cb <- build_crossbasis(d, exposure_col = "precip_mm",
                         group_col = "munic_code", time_col = "time_index",
                         lag_max = lag_max, per_unit_knots = TRUE)
  suppressWarnings(bind_crossbasis(d, cb, prefix = "rain_cb",
                                   drop_incomplete = FALSE))
}

started <- Sys.time()
cat(sprintf("[%s crosschecks] start %s\n", HAZARD, format(started)))
flush.console()
rows <- list()
record <- function(name, adjustment, expr) {
  t0 <- Sys.time()
  value <- try(force(expr), silent = TRUE)
  seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(value, "try-error")) {
    rows[[length(rows) + 1L]] <<- data.table(
      model = name, rainfall_adjustment = adjustment, status = "error", ok = FALSE,
      seconds = seconds, error = trimws(as.character(value)))
    return(invisible(NULL))
  }
  tidy <- as.data.table(value$tidy)
  tidy[, `:=`(model = name, rainfall_adjustment = adjustment, ok = TRUE,
              status = "ok", seconds = seconds, error = NA_character_)]
  if (!is.null(value$n_episodes)) tidy[, n_episodes := value$n_episodes]
  rows[[length(rows) + 1L]] <<- tidy
  invisible(NULL)
}

RUN_SUNAB <- identical(tolower(Sys.getenv("BREPI_RQ2_RUN_SUNAB", "false")),
                       "true")
if (RUN_SUNAB) {
  dp <- build_did_panel(panel, events, cobrade_include = cobrade, outcome = "rate")
  dp_rain <- rain_basis(dp)
  record("sunab_poisson", "unadjusted",
         run_sunab(dp, family = "poisson", event_window = c(-6L, 12L)))
  record("sunab_poisson", "rainfall_dlnm_lag0_3",
         run_sunab(dp_rain$data, family = "poisson", event_window = c(-6L, 12L),
                   adjust_terms = dp_rain$terms))
  rm(dp, dp_rain); gc()
} else {
  rows[[length(rows) + 1L]] <- data.table(
    model = "sunab_poisson", rainfall_adjustment = "not_run",
    status = "infeasible_memory", ok = FALSE, seconds = NA_real_,
    error = paste("Repaired monthly fit exceeded 16.5 GB while still rising;",
                  "terminated before the declared 20 GB ceiling."))
}

panel_rain <- rain_basis(panel)
record("episodic_poisson", "unadjusted",
       run_episodic_dlag(panel, events, cobrade_include = cobrade,
                         leads = 6L, lags = 12L, family = "poisson"))
record("episodic_poisson", "rainfall_dlnm_lag0_3",
       run_episodic_dlag(panel_rain$data, events, cobrade_include = cobrade,
                         leads = 6L, lags = 12L, family = "poisson",
                         adjust_terms = panel_rain$terms))

result <- rbindlist(rows, fill = TRUE)
fwrite(result, file.path(OUTDIR, paste0("corrected_", HAZARD,
                                       "_monthly_crosschecks.csv")))
elapsed_by_model <- unique(result[, .(model, rainfall_adjustment, seconds)])
fwrite(result[, .(
  ok = all(ok[model == "episodic_poisson"]),
  models = uniqueN(paste(model, rainfall_adjustment)),
  failed = uniqueN(paste(model[!ok], rainfall_adjustment[!ok])),
  seconds = sum(elapsed_by_model$seconds, na.rm = TRUE),
  error = paste(unique(na.omit(error)), collapse = " | "))],
  file.path(OUTDIR, paste0("corrected_", HAZARD,
                          "_monthly_crosschecks_status.csv")))
cat(sprintf("[%s crosschecks] complete in %.1f s; %d failed\n", HAZARD,
            as.numeric(difftime(Sys.time(), started, units = "secs")),
            sum(!result$ok)))
