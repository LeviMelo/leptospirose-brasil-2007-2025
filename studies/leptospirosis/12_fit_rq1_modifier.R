#!/usr/bin/env Rscript
# H2 - is the rainfall-leptospirosis slope steeper where sanitation is worse?
#
# This is the study's policy quantity, and it is the *identified* one. The
# structural main effects are not: `sanitation_sewer_share` moves from -1.64
# without a spatial field to -0.24 with the full structure, because a
# near-time-invariant municipal characteristic competes directly with a BYM2
# field for the same between-area variation (BREPI-016).
#
# The cross-basis x sanitation interaction does not have that problem. It is
# identified from *within-area* variation: whether a given health region's
# response to its own rainfall anomalies is larger when its sewer coverage is
# lower. The spatial field absorbs the level, not the slope.
#
# Reported as the cumulative exposure-response evaluated at contrasting
# sanitation values, with the full joint covariance of base and interaction
# blocks (never marginal SEs - see crossbasis_at_modifier()).
#
# Usage:
#   Rscript studies/leptospirosis/12_fit_rq1_modifier.R [profile] [scale]

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm)
})
for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R")) source(file.path("R", f))

args <- commandArgs(trailingOnly = TRUE)
PROFILE <- if (length(args) >= 1) args[1] else "confirmatory"
SCALE   <- if (length(args) >= 2) args[2] else "health_region"
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "8:1")
MODIFIER <- "sanitation_sewer_share"

cache <- readRDS(file.path("data/interim/bench", paste0("model_data_", SCALE, ".rds")))
dt <- cache$data; meta <- cache$meta; cb <- cache$crossbasis
.cb <- meta$cb_terms; .x <- meta$x_terms

# Rebuild the `bound` structure the modifier binder expects.
bound <- list(data = dt, terms = .cb, basis = cb)
bound <- bind_crossbasis_modifier(bound, MODIFIER)
dt <- data.table::as.data.table(bound$data)
mod <- bound$modifier
cat("[h2] modifier:", MODIFIER, "| centre", round(mod$center, 4),
    "| sd", round(mod$scale, 4), "\n")

prof <- model_execution_profile(PROFILE, num_threads = THREADS)
cc <- prof$controls

PC_PREC <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
PC_BYM2 <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                phi  = list(prior = "pc", param = c(0.5, 0.5)))
dep <- function(x) paste(deparse(x), collapse = "")

# The modifier's own main effect replaces the raw covariate: keeping both would
# be collinear by construction.
x_other <- setdiff(.x, MODIFIER)
rhs <- c(
  "1", x_other, mod$main_term, .cb, mod$interaction_terms,
  sprintf("f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
          meta$graph, dep(PC_BYM2)),
  sprintf("f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)", dep(PC_PREC)),
  sprintf("f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)",
          dep(PC_PREC)),
  sprintf("f(id_st_year, model='iid', constr=TRUE, hyper=%s)", dep(PC_PREC)),
  "offset(log_offset)"
)
form <- as.formula(paste("cases ~", paste(rhs, collapse = " + ")))

cat("[h2] profile:", PROFILE, "(", prof$scientific_status, ")\n")
t0 <- Sys.time()
fit <- inla(
  form, family = "nbinomial", data = dt,
  control.compute = cc$control.compute, control.inla = cc$control.inla,
  control.predictor = cc$control.predictor, control.fixed = cc$control.fixed,
  num.threads = THREADS, verbose = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("[h2] fitted in", round(elapsed, 1), "s\n")

all_terms <- c(.cb, mod$interaction_terms)
coef_all <- setNames(fit$summary.fixed[all_terms, "mean"], all_terms)
V_all <- inla_fixed_vcov(fit, all_terms)
cat("[h2] covariance route:", attr(V_all, "route"), "\n")

# Evaluate at contrasting sanitation coverage. Percentiles of the observed
# distribution, so both ends are populated by real health regions.
qs <- stats::quantile(dt[[MODIFIER]], c(0.10, 0.50, 0.90), na.rm = TRUE)
outdir <- file.path("data/results/rq1_exposure_response", paste0(SCALE, "_", PROFILE, "_h2"))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

x_hi <- as.numeric(stats::quantile(cb$x_model, 0.95, na.rm = TRUE))
rows <- list()
for (i in seq_along(qs)) {
  at_mod <- crossbasis_at_modifier(coef_all, V_all, mod, value = as.numeric(qs[i]))
  cum <- reduce_cumulative(cb, coef = at_mod$coef, vcov = at_mod$vcov)
  j <- which.min(abs(cum$x - x_hi))
  rows[[i]] <- data.table(
    sanitation_percentile = names(qs)[i],
    sanitation_value = as.numeric(qs[i]),
    modifier_z = at_mod$z,
    rr_p95 = cum$rr[j], rr_lo = cum$rr_lo[j], rr_hi = cum$rr_hi[j]
  )
  fwrite(cum, file.path(outdir, sprintf("cumulative_at_sanitation_%s.csv",
                                        gsub("%", "", names(qs)[i]))))
}
tab <- rbindlist(rows)
fwrite(tab, file.path(outdir, "h2_effect_modification.csv"))

interaction_summary <- as.data.table(
  fit$summary.fixed[mod$interaction_terms, ], keep.rownames = "term")
fwrite(interaction_summary, file.path(outdir, "h2_interaction_coefficients.csv"))

cat("\n[h2] cumulative RR at the p95 rainfall anomaly, by sanitation coverage\n")
print(tab[, .(sanitation_percentile,
              sewer_share = round(sanitation_value, 3),
              RR = round(rr_p95, 3),
              lo = round(rr_lo, 3), hi = round(rr_hi, 3))])

ratio <- tab$rr_p95[1] / tab$rr_p95[3]
cat(sprintf("\n[h2] ratio of rainfall RR, worst-sanitation p10 vs best p90: %.3f\n", ratio))
cat("[h2] H2 predicts this ratio > 1 (steeper rainfall slope where sanitation is worse).\n")
cat("[h2] wrote", outdir, "\n")
if (!identical(PROFILE, "confirmatory")) {
  cat("[h2] NOT QUOTABLE: development profile.\n")
}
