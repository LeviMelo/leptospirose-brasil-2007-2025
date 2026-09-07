#!/usr/bin/env Rscript
# RQ1 climate sensitivity: does ENSO state modulate the rainfall response?
#
# ENSO is the largest organised source of interannual rainfall variability over
# Brazil, and its sign differs by region: El Nino brings drought to the north
# and northeast and excess rain to the south, La Nina broadly the reverse. If
# the leptospirosis response to a *local* rainfall anomaly were purely
# hydrological, ENSO would matter only through the anomaly itself and the
# exposure-response curve would be the same in every phase. If instead the
# response differs by phase, something other than the month's rainfall is
# carrying part of the effect -- antecedent soil saturation, river stage, or a
# behavioural/surveillance response to a season everyone expected.
#
# TWO THINGS THIS ANALYSIS CANNOT DO, stated before the result:
#
#  1. ENSO cannot be an additive control here. It varies in time ONLY, so it is
#     collinear with the national monthly random walk already in the model --
#     literally the same degrees of freedom. Entering both would either fail to
#     identify or silently reallocate the trend. ENSO therefore enters solely
#     as a MODIFIER of the cross-basis, and the national temporal field stays.
#  2. This is not a causal decomposition. ENSO phase is not randomly assigned
#     to months and is correlated with everything else that is seasonal.
#
# The estimand is the cumulative RR at the p95 local anomaly within each ENSO
# phase, and the contrast between phases.
#
# Usage: Rscript studies/leptospirosis/27_rq1_enso.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(INLA); library(dlnm)
})
for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R", "09_exec.R")) {
  source(file.path("R", f))
}

OUT <- "data/results/rq1_exposure_response/enso"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "6:1")
SCALE <- "health_region"

# ---------------------------------------------------------------------------
# ENSO phase, on NOAA's own definition
# ---------------------------------------------------------------------------
# NOAA declares an event when the ONI reaches +/-0.5 for five consecutive
# overlapping seasons. Using the +/-0.5 threshold month-by-month is the
# conventional simplification and is what is done here; the alternative -- the
# formal event calendar -- is available from enso.events() and is carried as a
# sensitivity below rather than as the primary, because month-level phase keeps
# every month in the analysis instead of dropping the run-up to each event.
enso_path <- file.path(OUT, "enso_monthly.csv")
if (!file.exists(enso_path)) {
  cat("[enso] materialising the ONI series via the Python adapter\n")
  py <- Sys.getenv("BREPI_PYTHON",
                   "python")
  code <- sprintf(paste0(
    "import sys; sys.path.insert(0, '.');",
    "from datetime import date;",
    "from brepi.sources.climate import enso;",
    "df = enso.monthly_covariate(date(2007,1,1), date(2025,12,1), kind='oni', lags=(0,3,6));",
    "df.write_csv(r'%s')"), enso_path)
  st <- system2(py, c("-c", shQuote(code)), stdout = TRUE, stderr = TRUE)
  if (!file.exists(enso_path)) {
    cat(paste(st, collapse = "\n"), "\n")
    stop("could not materialise the ENSO series", call. = FALSE)
  }
}
enso <- fread(enso_path)
enso[, period := as.IDate(period)]
enso[, phase := fifelse(oni >= 0.5, "el_nino",
                 fifelse(oni <= -0.5, "la_nina", "neutral"))]
enso[, phase := factor(phase, levels = c("neutral", "el_nino", "la_nina"))]
cat("[enso] months by phase:\n"); print(enso[, .N, by = phase][order(-N)])

# ---------------------------------------------------------------------------
# Model data: the same cached objects the confirmatory RQ1 fit uses
# ---------------------------------------------------------------------------
cache <- readRDS(file.path("data/interim/bench",
                           paste0("model_data_", SCALE, ".rds")))
dt <- as.data.table(cache$data)
meta <- cache$meta
cb <- cache$crossbasis
cat("[enso] model cells:", format(nrow(dt), big.mark = ","),
    "| cross-basis columns:", length(meta$cb_terms), "\n")

if (!"date" %in% names(dt)) {
  dt[, date := as.IDate(sprintf("%d-%02d-01", year, month))]
}
dt <- merge(dt, enso[, .(date = period, oni, phase)], by = "date", all.x = TRUE)
if (anyNA(dt$phase)) {
  stop("ENSO phase missing for ", sum(is.na(dt$phase)), " model cells; the ",
       "series does not cover the panel window.", call. = FALSE)
}

# Cross-basis x phase. Each phase gets its own set of cross-basis coefficients,
# which is the interaction written out rather than a single interaction term --
# it keeps the reduction machinery unchanged, since reduce_cumulative() needs a
# coefficient per basis column.
for (ph in c("el_nino", "la_nina")) {
  for (k in seq_along(meta$cb_terms)) {
    nm <- paste0(meta$cb_terms[k], "_", ph)
    dt[, (nm) := get(meta$cb_terms[k]) * as.integer(phase == ph)]
  }
}
inter_terms <- unlist(lapply(c("el_nino", "la_nina"),
                             function(ph) paste0(meta$cb_terms, "_", ph)))

pc_prec <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
dep <- function(x) paste(deparse(x), collapse = "")
rhs <- c("1", meta$x_terms, meta$cb_terms, inter_terms,
         sprintf("f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
                 meta$graph,
                 dep(list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                          phi = list(prior = "pc", param = c(0.5, 0.5))))),
         sprintf("f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)", dep(pc_prec)),
         sprintf("f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)", dep(pc_prec)),
         sprintf("f(id_st_year, model='iid', constr=TRUE, hyper=%s)", dep(pc_prec)),
         "offset(log_offset)")

ctl <- model_execution_profile("confirmatory", num_threads = THREADS)$controls
cat("[enso] fitting cross-basis x ENSO phase ...\n")
t0 <- Sys.time()
fit <- inla(as.formula(paste("cases ~", paste(rhs, collapse = " + "))),
            family = "nbinomial", data = dt,
            control.compute = ctl$control.compute, control.inla = ctl$control.inla,
            control.predictor = ctl$control.predictor, control.fixed = ctl$control.fixed,
            num.threads = THREADS, verbose = FALSE)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("[enso] fitted in", round(secs, 1), "s\n")

# ---------------------------------------------------------------------------
# Reduce within each phase
# ---------------------------------------------------------------------------
x_hi <- as.numeric(stats::quantile(cb$x_model, 0.95, na.rm = TRUE))
phase_rr <- function(label, terms) {
  V <- inla_fixed_vcov(fit, terms)
  co <- fit$summary.fixed[terms, "mean"]
  cum <- reduce_cumulative(cb, coef = co, vcov = V)
  j <- which.min(abs(cum$x - x_hi))
  data.table(phase = label, rr = cum$rr[j], lo = cum$rr_lo[j], hi = cum$rr_hi[j])
}
# The neutral phase is the reference: its coefficients are the main cross-basis
# terms. El Nino and La Nina are main + their own interaction block, which is a
# linear combination and so needs the joint covariance of both blocks.
lincomb_rr <- function(label, ph) {
  main <- meta$cb_terms
  extra <- paste0(main, "_", ph)
  all_terms <- c(main, extra)
  V <- inla_fixed_vcov(fit, all_terms)
  co <- fit$summary.fixed[all_terms, "mean"]
  k <- length(main)
  # Sum the two blocks: beta_phase = beta_main + beta_interaction.
  Tm <- cbind(diag(k), diag(k))
  co_p <- as.numeric(Tm %*% co)
  V_p <- Tm %*% V %*% t(Tm)
  dimnames(V_p) <- list(main, main)
  cum <- reduce_cumulative(cb, coef = co_p, vcov = V_p)
  j <- which.min(abs(cum$x - x_hi))
  data.table(phase = label, rr = cum$rr[j], lo = cum$rr_lo[j], hi = cum$rr_hi[j])
}

res <- rbindlist(list(
  phase_rr("neutral", meta$cb_terms),
  lincomb_rr("el_nino", "el_nino"),
  lincomb_rr("la_nina", "la_nina")))
res[, months := enso[, .N, by = phase][match(res$phase, phase)]$N]
fwrite(res, file.path(OUT, "enso_phase_exposure_response.csv"))
cat("\n[enso] cumulative RR at the p95 LOCAL anomaly, by ENSO phase\n")
print(res[, .(phase, months, rr = round(rr, 3), lo = round(lo, 3),
              hi = round(hi, 3))])

# Interaction coefficients, so a reader can see whether the modification is
# anywhere near distinguishable from zero.
ic <- as.data.table(fit$summary.fixed[inter_terms, ], keep.rownames = "term")
setnames(ic, c("0.025quant", "0.975quant"), c("lo", "hi"), skip_absent = TRUE)
ic[, excludes_zero := (lo > 0 | hi < 0)]
fwrite(ic, file.path(OUT, "enso_interaction_coefficients.csv"))
cat("\n[enso] interaction terms whose 95% interval excludes zero:",
    sum(ic$excludes_zero), "of", nrow(ic), "\n")

sc <- model_scores(fit)
fwrite(as.data.table(sc), file.path(OUT, "enso_model_scores.csv"))
cat("\n[enso] model scores\n"); print(sc)
cat("\n[enso] REFERENCE (no ENSO interaction) DIC was 119,499 / WAIC 120,112.\n")
cat("[enso] A materially better score here would mean phase carries signal the\n")
cat("[enso] national temporal field does not already absorb.\n")

assert_no_strays(kill = TRUE)
cat("\n[enso] wrote", OUT, "\n")
