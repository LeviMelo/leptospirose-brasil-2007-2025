#!/usr/bin/env Rscript
# Tests for R/11_tables.R.
#
# These are mostly INVARIANT tests rather than value tests: a standardised rate
# must reproduce the common rate when every stratum shares it, a Gini must be
# zero under equality, a trend fitted to exact exponential growth must recover
# its rate. Invariants catch the errors that matter here -- a transposed
# weight, a flipped sign, a wrapped month -- which agreeing-to-4-decimals
# against a hand-computed number does not.
#
# Usage: Rscript tests/test_tables.R
suppressPackageStartupMessages(library(data.table))
source("R/00_io.R"); source("R/11_tables.R")

.pass <- 0L; .fail <- 0L
ok <- function(label, cond) {
  if (isTRUE(cond)) { .pass <<- .pass + 1L; cat("  ok   ", label, "\n") }
  else { .fail <<- .fail + 1L; cat("  FAIL ", label, "\n") }
}
near <- function(a, b, tol = 1e-6) isTRUE(all(abs(a - b) < tol))

cat("\npoisson_ci\n")
p <- poisson_ci(10, 1, multiplier = 1)
ok("point estimate is x/pt", near(p$rate, 10))
ok("lower limit matches qgamma(0.025, x)", near(p$lo, qgamma(0.025, 10)))
ok("upper limit matches qgamma(0.975, x + 1)", near(p$hi, qgamma(0.975, 11)))
p0 <- poisson_ci(0, 100, multiplier = 1)
ok("zero counts give a lower limit of exactly 0", near(p0$lo, 0))
ok("zero counts still give a positive upper limit", p0$hi > 0)
ok("interval brackets the estimate", p$lo < p$rate && p$rate < p$hi)

cat("\nbinom_ci\n")
b <- binom_ci(5, 20)
ok("proportion is x/n", near(b$p, 0.25))
ok("Clopper-Pearson lower ~ 0.0866", near(b$lo, 0.08657, tol = 1e-4))
ok("Clopper-Pearson upper ~ 0.4910", near(b$hi, 0.49104, tol = 1e-4))
ok("x = 0 pins the lower limit at 0", near(binom_ci(0, 10)$lo, 0))
ok("x = n pins the upper limit at 1", near(binom_ci(10, 10)$hi, 1))
# A composition -- k categories over one shared total -- is the common case.
bc <- binom_ci(c(10, 20, 70), 100)
ok("a length-1 denominator recycles across categories",
   nrow(bc) == 3L && near(bc$p, c(0.1, 0.2, 0.7)))
ok("a genuine length mismatch is still refused",
   inherits(try(binom_ci(c(1, 2, 3), c(10, 20)), silent = TRUE), "try-error"))
ok("poisson_ci recycles a length-1 person-time",
   nrow(poisson_ci(c(5, 10), 1000, multiplier = 1)) == 2L)
ok("numerator > denominator is refused",
   inherits(try(binom_ci(11, 10), silent = TRUE), "try-error"))

cat("\nrate_table\n")
d <- data.table(g = rep(c("a", "b"), each = 4),
                cases = c(10, 10, 10, 10, 20, 20, 20, 20),
                person_months = rep(12000, 8))
rt <- rate_table(d, by = "g", reference = "a")
ok("one row per stratum", nrow(rt) == 2L)
ok("rate is events / person-years * 1e5",
   near(rt[g == "a", rate], 40 / (48000 / 12) * 1e5))
ok("reference stratum has RR exactly 1", near(rt[g == "a", rr], 1))
ok("RR recovers the ratio of rates", near(rt[g == "b", rr], 2))
ok("RR interval brackets the RR",
   rt[g == "b", rr_lo] < 2 && rt[g == "b", rr_hi] > 2)
ok("an ambiguous reference is refused",
   inherits(try(rate_table(d, by = "g", reference = "zzz"), silent = TRUE),
            "try-error"))

cat("\nproportion_table\n")
d2 <- data.table(g = c("a", "a", "b", "b"), deaths = c(1, 1, 5, 5),
                 outcome_known = c(50, 50, 50, 50))
pt <- proportion_table(d2, by = "g", numerator = "deaths",
                       denominator = "outcome_known")
ok("CFR computed on the named denominator", near(pt[g == "a", p], 2 / 100))
ok("a missing denominator is refused",
   inherits(try(proportion_table(d2, by = "g", numerator = "deaths"),
                silent = TRUE), "try-error"))

cat("\nage_standardise\n")
ages <- names(WHO_WORLD_STANDARD)
# Every age group shares one rate; the age distributions of the two regions are
# wildly different. A correct direct standardisation must return that common
# rate for both, exactly.
set.seed(3)
mk <- function(region, pop) {
  data.table(region = region, age = ages, person_months = pop * 12,
             cases = pop * 12 / 12 * 0.001)   # 0.001 per person-year everywhere
}
d3 <- rbind(mk("young", c(rep(50000, 6), rep(1000, 12))),
            mk("old",   c(rep(1000, 12), rep(50000, 6))))
as1 <- age_standardise(d3, age_col = "age", by = "region", multiplier = 1e5)
ok("ASR equals the common rate regardless of age structure",
   near(as1$asr[1], 0.001 * 1e5, tol = 1e-6) &&
   near(as1$asr[2], 0.001 * 1e5, tol = 1e-6))
ok("crude rates genuinely differ from each other is NOT required here, but ASR agrees",
   near(as1$asr[1], as1$asr[2]))
ok("Fay-Feuer interval brackets the ASR",
   all(as1$lo < as1$asr) && all(as1$hi > as1$asr))
ok("the standard is returned with the result",
   identical(attr(as1, "standard"), WHO_WORLD_STANDARD))
d4 <- copy(d3); d4[1, age := "not-an-age-group"]
ok("an age group absent from the standard is refused, not dropped",
   inherits(try(age_standardise(d4, age_col = "age", by = "region"),
                silent = TRUE), "try-error"))
ok("a NULL standard is refused",
   inherits(try(age_standardise(d3, age_col = "age", standard = NULL),
                silent = TRUE), "try-error"))

cat("\nrebase_standard\n")
tensor_groups <- c("00-04", "05-09", "10-14", "15-19", "20-24", "25-29",
                   "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
                   "60-64", "65-69", "70-74", "75-79", "80+")
rb <- rebase_standard(WHO_WORLD_STANDARD, tensor_groups)
ok("total weight is conserved", near(sum(rb), sum(WHO_WORLD_STANDARD)))
ok("the open-ended group absorbs the collapsed tail",
   near(rb[["80+"]], WHO_WORLD_STANDARD[["80-84"]] + WHO_WORLD_STANDARD[["85+"]]))
ok("uncollapsed groups keep their weight, zero padding notwithstanding",
   near(rb[["00-04"]], WHO_WORLD_STANDARD[["0-4"]]) &&
   near(rb[["45-49"]], WHO_WORLD_STANDARD[["45-49"]]))
ok("output is named and ordered as requested",
   identical(names(rb), tensor_groups))
ok("rebasing onto the same grouping is the identity",
   near(sum(abs(rebase_standard(WHO_WORLD_STANDARD, names(WHO_WORLD_STANDARD)) -
                WHO_WORLD_STANDARD)), 0))
coarse <- rebase_standard(WHO_WORLD_STANDARD, c("0-19", "20-59", "60+"))
ok("a much coarser grouping still conserves the total",
   near(sum(coarse), sum(WHO_WORLD_STANDARD)))
ok("unparseable labels are refused",
   inherits(try(rebase_standard(WHO_WORLD_STANDARD, c("young", "old")),
                silent = TRUE), "try-error"))
# The invariant that matters: standardising with a rebased standard on data
# where every age shares one rate must still return that rate.
d5 <- rbind(
  data.table(region = "a", age = tensor_groups,
             person_months = c(rep(60000, 8), rep(6000, 9)) * 12,
             cases = c(rep(60000, 8), rep(6000, 9)) * 0.001),
  data.table(region = "b", age = tensor_groups,
             person_months = c(rep(6000, 9), rep(60000, 8)) * 12,
             cases = c(rep(6000, 9), rep(60000, 8)) * 0.001))
as2 <- age_standardise(d5, age_col = "age", by = "region", standard = rb,
                       multiplier = 1e5)
ok("a rebased standard still recovers a common rate",
   near(as2$asr[1], 100, tol = 1e-6) && near(as2$asr[2], 100, tol = 1e-6))

cat("\nconcentration\n")
eq <- data.table(u = 1:100, cases = rep(5, 100), person_months = rep(1200, 100))
ce <- concentration(eq, unit = "u")
ok("Gini is ~0 when every unit is identical",
   abs(ce$summary$gini_cases[1]) < 1e-9)
ok("no unit reports zero under equality", ce$summary$units_with_zero[1] == 0L)
sk <- data.table(u = 1:100, cases = c(500, rep(0, 99)), person_months = rep(1200, 100))
cs <- concentration(sk, unit = "u")
ok("Gini approaches (n-1)/n when one unit holds everything",
   near(cs$summary$gini_cases[1], 99 / 100, tol = 1e-9))
ok("the zero share is reported", near(cs$summary$zero_share[1], 0.99))
ok("Lorenz cumulative shares end at 1",
   near(max(cs$lorenz$cum_cases), 1) && near(max(cs$lorenz$cum_pop), 1))
ok("Lorenz cumulative case share is non-decreasing",
   all(diff(cs$lorenz$cum_cases) >= -1e-12))

cat("\ntrend_apc\n")
yrs <- 2007:2025
growth <- data.table(year = yrs, person_months = 1.2e6,
                     cases = round(100 * 1.10^(seq_along(yrs) - 1)))
ta <- trend_apc(growth)
ok("AAPC recovers exact 10%/yr exponential growth", near(ta$aapc, 10, tol = 0.15))
ok("AAPC interval brackets the truth", ta$aapc_lo < 10 && ta$aapc_hi > 10)
ok("Mann-Kendall tau is +1 on a strictly increasing series",
   near(ta$mk_tau, 1, tol = 1e-9))
decline <- copy(growth)[, cases := rev(cases)]
td <- trend_apc(decline)
ok("a falling series gives a negative AAPC", td$aapc < 0)
ok("Mann-Kendall tau is -1 on a strictly decreasing series",
   near(td$mk_tau, -1, tol = 1e-9))
ok("short series return NA rather than a fitted slope",
   is.na(trend_apc(growth[1:3])$aapc))

cat("\ncircular_season\n")
one <- data.table(month = 1:12, cases = c(120, rep(0, 11)))
c1 <- circular_season(one)
ok("all cases in one month give resultant length 1", near(c1$resultant_r, 1))
ok("and that month is the peak", c1$peak_month == 1L)
flat <- data.table(month = 1:12, cases = rep(50, 12))
c2 <- circular_season(flat)
ok("a flat year gives resultant length ~0", c2$resultant_r < 1e-9)
# The wrap-around case: December and January are adjacent, so their midpoint
# must be the year boundary, not June.
dj <- data.table(month = 1:12, cases = c(100, rep(0, 10), 100))
c3 <- circular_season(dj)
ok("a Dec-Jan peak does not average to mid-year",
   c3$peak_month %in% c(12L, 1L))
jun <- data.table(month = 1:12, cases = c(rep(0, 5), 200, rep(0, 6)))
ok("a June peak is reported as June", circular_season(jun)$peak_month == 6L)
# Line-level input: one row per case, no count column.
lvl <- data.table(month = c(rep(6L, 200), rep(7L, 50)))
cl <- circular_season(lvl, numerator = NULL)
ok("line-level input is counted when numerator = NULL",
   cl$cases == 250L && cl$peak_month == 6L)
ok("an absent numerator column is refused with a usable hint",
   inherits(try(circular_season(lvl), silent = TRUE), "try-error"))
ok("an absent month column is refused",
   inherits(try(circular_season(lvl, month_col = "nope", numerator = NULL),
                silent = TRUE), "try-error"))
ok("months outside 1-12 are refused",
   inherits(try(circular_season(data.table(month = 0:11, cases = 1)),
                silent = TRUE), "try-error"))

cat("\ncompleteness_table\n")
line <- data.table(
  year = rep(c(2010L, 2011L), each = 5),
  evolucao = c("1", "2", "9", NA, "", "1", "1", "1", "9", "7")
)
ct <- completeness_table(line, fields = "evolucao", by = "year",
                         unknown_values = "9",
                         valid_values = list(evolucao = c("1", "2", "3")))
ok("the four states partition every record",
   all(ct$n_missing + ct$n_unknown + ct$n_invalid + ct$n_valid == ct$n))
ok("an explicit ignorado code counts as unknown, not missing",
   ct[year == 2010, n_unknown] == 1L)
ok("NA and empty string both count as missing",
   ct[year == 2010, n_missing] == 2L)
ok("a value outside the dictionary counts as invalid",
   ct[year == 2011, n_invalid] == 1L)
ok("shares sum to one", all(near(ct$share_missing + ct$share_unknown +
                                 ct$share_invalid + ct$share_valid, 1)))
ok("an absent column is refused",
   inherits(try(completeness_table(line, fields = "nope"), silent = TRUE),
            "try-error"))

cat(sprintf("\n%d passed, %d failed\n", .pass, .fail))
if (.fail > 0L) quit(status = 1L)
