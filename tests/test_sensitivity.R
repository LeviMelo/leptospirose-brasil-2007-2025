#!/usr/bin/env Rscript
# Tests for R/08_sensitivity.R, focused on the comparability rule.
#
# A robustness battery routinely contains two kinds of perturbation: ones that
# re-measure the SAME quantity a different way, and ones that deliberately
# measure a DIFFERENT quantity -- a recoded outcome, a restricted population, a
# placebo. Only the first kind can be judged against a relative-change
# threshold.
#
# Conflating them is not a cosmetic problem. The RQ2 grid recoded its outcome
# from asinh(count) to a rate and the report called it a 3,708% movement; the
# drought placebo, whose entire purpose is to be null, was flagged as a SIGN
# CHANGE. Both are the design working exactly as intended, and both appeared in
# the verdict as robustness failures.
#
# Usage: Rscript tests/test_sensitivity.R
suppressPackageStartupMessages(library(data.table))
source("R/00_io.R"); source("R/08_sensitivity.R")

.pass <- 0L; .fail <- 0L
ok <- function(label, cond) {
  if (isTRUE(cond)) { .pass <<- .pass + 1L; cat("  ok   ", label, "\n") }
  else { .fail <<- .fail + 1L; cat("  FAIL ", label, "\n") }
}

BASE <- list(k = 1)
P <- list(
  perturbation("same_scale", "Same quantity, measured differently.",
               "specification", list(k = 1.1),
               rationale = "A defensible alternative specification."),
  perturbation("rescaled", "A different outcome transformation.", "outcome",
               list(k = 40), rationale = "Recodes the outcome entirely.",
               comparable = FALSE),
  perturbation("placebo", "An exposure with no plausible mechanism.",
               "exposure", list(k = -1),
               rationale = "Placebo; a null result is the expected one.",
               comparable = FALSE)
)
grid <- sensitivity_grid(BASE, P)

fit_fn <- function(spec, ...) spec$k
estimand_fn <- function(fit, spec, ...) data.frame(theta = fit)

res <- run_sensitivity(grid, fit_fn, estimand_fn, verbose = FALSE)
rep <- sensitivity_report(res, estimand = "theta", threshold = 0.20)

cat("\ncomparability is carried through the engine\n")
ok("every result row carries `comparable`", "comparable" %in% names(rep))
ok("the reference run is comparable with itself",
   isTRUE(rep[run_id == "reference", comparable]))
ok("an unmarked perturbation defaults to comparable",
   isTRUE(rep[run_id == "same_scale", comparable]))

cat("\nmateriality applies only to same-scale runs\n")
ok("a same-scale run inside the threshold is not material",
   !rep[run_id == "same_scale", material])
ok("a 40-fold rescaled run is NOT flagged material",
   !rep[run_id == "rescaled", material])
ok("a sign-flipping placebo is NOT flagged as a sign change",
   !rep[run_id == "placebo", sign_flip])

cat("\nbut nothing is hidden\n")
ok("the rescaled run keeps its estimate",
   is.finite(rep[run_id == "rescaled", theta]))
ok("and its relative change is still computed and shown",
   is.finite(rep[run_id == "rescaled", relative_change]))
ok("incomparable runs are labelled, not silently normal",
   rep[run_id == "rescaled", note] == "different estimand - not comparable" &&
   rep[run_id == "placebo", note] == "different estimand - not comparable")

cat("\nthe verdict counts what it actually evaluated\n")
v <- sensitivity_verdict(rep)
ok("only comparable runs are counted as evaluated",
   attr(v, "n_evaluated") == 1L)
ok("no material movement is reported", attr(v, "n_material") == 0L)
ok("the excluded runs are counted", attr(v, "n_incomparable") == 2L)
ok("and the verdict text says the estimand changed",
   grepl("change the estimand", as.character(v), fixed = TRUE))
ok("and names them", grepl("rescaled", as.character(v), fixed = TRUE))

cat("\nthe flag is what does the work, not the values\n")
# Marked comparable, the identical grid must behave as it did before, so the
# test cannot pass merely because these numbers are unusual.
P2 <- lapply(P, function(x) { x$comparable <- TRUE; x })
rep2 <- sensitivity_report(
  run_sensitivity(sensitivity_grid(BASE, P2), fit_fn, estimand_fn,
                  verbose = FALSE),
  estimand = "theta", threshold = 0.20)
ok("marked comparable, the rescaled run IS material",
   isTRUE(rep2[run_id == "rescaled", material]))
ok("marked comparable, the placebo IS a sign change",
   isTRUE(rep2[run_id == "placebo", sign_flip]))
ok("and the verdict then evaluates all three",
   attr(sensitivity_verdict(rep2), "n_evaluated") == 3L)

cat("\nfailures still dominate every other state\n")
fit_bad <- function(spec, ...) if (spec$k == 40) stop("boom") else spec$k
rep3 <- sensitivity_report(
  run_sensitivity(grid, fit_bad, estimand_fn, verbose = FALSE),
  estimand = "theta", threshold = 0.20)
ok("a failed run is reported as failed, not as incomparable",
   rep3[run_id == "rescaled", note] == "FAILED - must be reported, not dropped")

cat(sprintf("\n%d passed, %d failed\n", .pass, .fail))
if (.fail > 0L) quit(status = 1L)
