#!/usr/bin/env Rscript
# RQ2 cost probe -- is did::att_gt feasible at 5,318 units x 228 months x 200
# cohorts before we spend a budget on it?
#
# att_gt loops over (group, period): 200 x 228 = 45,600 doubly-robust cells,
# each a logit plus a weighted regression, and then a multiplier bootstrap over
# an n x 45,600 influence-function matrix. Cost is roughly linear in the number
# of cells and in n, so a unit subsample times the cell loop honestly and the
# extrapolation in n is the cheap direction.
#
# Usage: Rscript studies/leptospirosis/15a_rq2_probe.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow) })
for (f in c("00_io.R", "05_did_flood.R", "09_exec.R")) source(file.path("R", f))

dir.create("data/results/rq2_flood_disasters", recursive = TRUE, showWarnings = FALSE)

probe <- function(frac, time_grain, seed) {
  suppressPackageStartupMessages({ library(data.table); library(arrow); library(did) })
  for (f in c("00_io.R", "05_did_flood.R")) source(file.path("R", f))
  panel <- read_panel(
    "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet",
    validate = FALSE, prepare = TRUE)
  dec <- as.data.table(arrow::read_parquet("data/panel/flood_events.parquet"))
  dp <- build_did_panel(panel, dec, outcome = "asinh", first_only = TRUE)
  if (time_grain == "quarter") {
    dp[, period_q := ((period - 1L) %/% 3L) + 1L]
    dp <- dp[, .(Y = mean(Y), first_treat = first_treat[1]),
             by = .(unit_id, period = period_q)]
    dp[, first_treat := fifelse(first_treat == 0L, 0L,
                                ((first_treat - 1L) %/% 3L) + 1L)]
  }
  set.seed(seed)
  ids <- sort(sample(unique(dp$unit_id), round(frac * uniqueN(dp$unit_id))))
  d <- dp[unit_id %in% ids]
  t0 <- Sys.time()
  a <- did::att_gt(yname = "Y", tname = "period", idname = "unit_id",
                   gname = "first_treat", data = as.data.frame(d), panel = TRUE,
                   control_group = "notyettreated", anticipation = 1L,
                   est_method = "dr", bstrap = FALSE, cband = FALSE,
                   base_period = "universal")
  list(secs = as.numeric(difftime(Sys.time(), t0, units = "secs")),
       n_units = uniqueN(d$unit_id), n_periods = uniqueN(d$period),
       n_groups = uniqueN(d$first_treat[d$first_treat > 0]),
       n_cells = length(a$att), n_na = sum(is.na(a$att)),
       inffunc_mb = as.numeric(object.size(a$inffunc)) / 1024^2)
}

for (cfg in list(list(0.10, "month"), list(0.25, "month"),
                 list(0.25, "quarter"))) {
  r <- bounded_call(probe, args = list(frac = cfg[[1]], time_grain = cfg[[2]],
                                       seed = 20260730L),
                    budget = limits(wall_seconds = 900, max_rss_mb = 24000,
                                    poll_seconds = 5),
                    label = sprintf("probe_%s_%s", cfg[[1]], cfg[[2]]))
  print(r)
  if (identical(r$status, "ok")) print(unlist(r$value))
}
assert_no_strays(kill = TRUE)
