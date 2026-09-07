#!/usr/bin/env Rscript
# Contract tests for count outcomes and temporal aggregation in R/05_did_flood.R.
suppressPackageStartupMessages(library(data.table))
source("R/00_io.R"); source("R/05_did_flood.R")

.pass <- 0L; .fail <- 0L
ok <- function(label, cond) {
  if (isTRUE(cond)) { .pass <<- .pass + 1L; cat("  ok   ", label, "\n") }
  else { .fail <<- .fail + 1L; cat("  FAIL ", label, "\n") }
}
throws <- function(expr, pattern) {
  x <- try(force(expr), silent = TRUE)
  inherits(x, "try-error") && grepl(pattern, as.character(x), fixed = TRUE)
}

monthly <- data.table(
  unit_id = rep(1:2, each = 4),
  munic_code = rep(c("001", "002"), each = 4),
  period = rep(1:4, 2),
  first_treat = c(rep(3L, 4), rep(0L, 4)),
  cases = c(1, 3, 0, 2, 0, 1, 0, 0),
  person_months = c(100, 200, 100, 100, 50, 50, 50, 50),
  baseline_urban = rep(c(0.6, 0.3), each = 4)
)

cat("\nprimitive quantities are aggregated before transformation\n")
q_asinh <- coarsen_did_panel(monthly, grain = 2, outcome = "asinh",
                             baseline_cols = "baseline_urban")
u1p1 <- q_asinh[unit_id == 1 & period == 1]
ok("counts are additive", u1p1$cases == 4)
ok("person-time is additive", u1p1$person_months == 300)
ok("asinh is applied to the aggregated count",
   isTRUE(all.equal(u1p1$Y, asinh(4))))
ok("monthly transformed outcomes were not summed",
   !isTRUE(all.equal(u1p1$Y, asinh(1) + asinh(3))))
ok("offset is the log of aggregated person-time",
   isTRUE(all.equal(u1p1$log_offset, log(300))))

q_rate <- coarsen_did_panel(monthly, grain = 2, outcome = "rate")
ok("rate uses aggregate numerator and denominator",
   isTRUE(all.equal(q_rate[unit_id == 1 & period == 1]$Y,
                    4 / 300 * 1e5)))
q_any <- coarsen_did_panel(monthly, grain = 2, outcome = "any")
ok("any-event remains binary after coarsening",
   identical(sort(unique(q_any$Y)), c(0, 1)))

cat("\ntreatment and baseline contracts survive coarsening\n")
ok("treatment cohort is mapped to the coarser period",
   all(q_asinh[unit_id == 1]$first_treat == 2L))
ok("never-treated remains zero", all(q_asinh[unit_id == 2]$first_treat == 0L))
ok("frozen baseline is carried", all(q_asinh[unit_id == 1]$baseline_urban == 0.6))
early <- rbind(monthly, data.table(
  unit_id = 3L, munic_code = "003", period = 1:4,
  first_treat = rep(2L, 4), cases = 0, person_months = 100,
  baseline_urban = 0.4))
q_early <- suppressMessages(coarsen_did_panel(
  early, grain = 2, outcome = "rate", baseline_cols = "baseline_urban"))
ok("a cohort entering the first coarse period is excluded",
   !3L %in% q_early$unit_id)
bad <- copy(monthly); bad[unit_id == 1 & period == 4, baseline_urban := 0.7]
ok("time-varying baseline is rejected",
   throws(coarsen_did_panel(bad, 2, "rate", "baseline_urban"),
          "varies within unit"))
bad_pt <- copy(monthly); bad_pt[1, person_months := 0]
ok("invalid rate denominator is rejected",
   throws(coarsen_did_panel(bad_pt, 1, "rate"), "positive person-time"))

cat("\nassignment diagnostics weight baseline once per unit\n")
design <- data.table(
  unit_id = rep(1:6, each = 2), period = rep(1:2, 6),
  first_treat = rep(c(2L, 2L, 2L, 0L, 0L, 0L), each = 2),
  baseline_x = rep(c(1, 2, 3, 2, 3, 4), each = 2))
dg <- suppressWarnings(treatment_assignment_diagnostics(design, "baseline_x"))
ok("each unit enters the propensity diagnostic once", nrow(dg$propensity) == 6)
ok("balance exposes a standardised mean difference",
   nrow(dg$balance) == 1 && is.finite(dg$balance$standardized_mean_difference))
ok("overlap is reported rather than assumed",
   dg$overlap$n_treated == 3 && dg$overlap$n_control == 3 &&
     is.finite(dg$overlap$share_outside_common_support))

cat(sprintf("\n%d passed, %d failed\n", .pass, .fail))
if (.fail) quit(status = 1L)
