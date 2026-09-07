#!/usr/bin/env Rscript
# RQ1 model specification -- DEFINITIONS ONLY, no side effects.
#
# This file is sourced by the orchestration script AND, independently, by every
# bounded child process that fits a specification. It must therefore never run
# an analysis, print, or read command-line arguments: sourcing it may happen
# dozens of times per battery.
#
# Splitting specification from orchestration is not merely a requirement of
# process isolation. It means the model can be loaded and inspected -- what is
# adjusted for, what the estimand is, how the exposure enters -- without
# executing anything, which is what a reviewer or a co-author actually wants.

suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm)
})

.state <- function(scale) {
  # Memoised per process: the battery runs one fit per child, but a future
  # multi-fit child must not re-read a 100k-row RDS per call.
  if (!exists(".brepi_state", envir = globalenv())) {
    for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R")) {
      source(file.path("R", f))
    }
    cache <- readRDS(file.path("data/interim/bench",
                               paste0("model_data_", scale, ".rds")))
    dt <- data.table::as.data.table(cache$data)
    # INLA needs a distinct index copy per f(); the Knorr-Held type II term
    # cannot reuse id_space / id_time_year.
    dt[, id_space_g2 := id_space]
    dt[, id_time_g_year := id_time_year]
    assign(".brepi_state",
           list(dt = dt, meta = cache$meta, cb = cache$crossbasis),
           envir = globalenv())
  }
  get(".brepi_state", envir = globalenv())
}

fit_fn <- function(spec, ...) {
  suppressPackageStartupMessages({ library(data.table); library(INLA); library(dlnm) })
  st <- .state(spec$scale)
  dep <- function(x) paste(deparse(x), collapse = "")
  pc_prec <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
  spatial_prior <- list(
    prec = list(prior = "pc.prec", param = c(spec$spatial_prec_u, 0.01)),
    phi  = list(prior = "pc", param = c(0.5, 0.5))
  )

  int_term <- switch(
    spec$interaction,
    none        = NULL,
    type1_year  = sprintf("f(id_st_year, model='iid', constr=TRUE, hyper=%s)", dep(pc_prec)),
    type1_month = sprintf("f(id_st_month, model='iid', constr=TRUE, hyper=%s)", dep(pc_prec)),
    type2_year  = sprintf(paste0("f(id_space_g2, model='iid', group=id_time_g_year, ",
                                 "control.group=list(model='rw1', scale.model=TRUE), hyper=%s)"),
                          dep(pc_prec)),
    stop("unknown interaction: ", spec$interaction)
  )
  covars <- if (isTRUE(spec$adjusted)) st$meta$x_terms else character()
  rhs <- c(
    "1", covars, st$meta$cb_terms,
    sprintf("f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
            st$meta$graph, dep(spatial_prior)),
    if (isTRUE(spec$temporal))
      sprintf("f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)", dep(pc_prec)),
    if (isTRUE(spec$seasonal))
      sprintf("f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)",
              dep(pc_prec)),
    int_term, "offset(log_offset)"
  )

  d <- st$dt
  if (!is.null(spec$year_max)) {
    # Climate product-splice sensitivity. BR-DWGD covers the panel through
    # 2024-03-20 and ERA5-Land thereafter; restricting to <= 2023 gives an
    # UNSPLICED exposure series. Note that this also removes the 2024 Rio
    # Grande do Sul catastrophe, so on its own it confounds "the splice" with
    # "the largest single outbreak in the series" -- which is why `drop_year`
    # exists below to separate them.
    d <- d[year <= spec$year_max]
  }
  if (!is.null(spec$drop_year)) {
    d <- d[!(year %in% spec$drop_year)]
  }
  if (isTRUE(spec$drop_covid)) {
    # 2020-2021 is a surveillance artefact, not an epidemiological trough: the
    # confirmation ratio and notification volume both collapse. Excluding it
    # tests whether the exposure-response is carried by that period.
    d <- d[!(year %in% c(2020L, 2021L))]
  }

  ctl <- model_execution_profile("development", num_threads = spec$threads)$controls
  inla(as.formula(paste("cases ~", paste(rhs, collapse = " + "))),
       family = "nbinomial", data = d,
       control.compute = ctl$control.compute, control.inla = ctl$control.inla,
       control.predictor = ctl$control.predictor, control.fixed = ctl$control.fixed,
       num.threads = spec$threads, verbose = FALSE)
}

estimand_fn <- function(fit, spec, ...) {
  st <- .state(spec$scale)
  cbt <- st$meta$cb_terms
  # The battery's comparison is between POINT estimates, and the point estimate
  # does not depend on the covariance at all. So a fit that returns no
  # covariance must still yield a comparable row, with the interval blanked and
  # the reason recorded -- not abort the run. One such abort already cost this
  # battery its reference and therefore its whole report.
  V <- tryCatch(inla_fixed_vcov(fit, cbt), error = function(e) e)
  route <- if (inherits(V, "error")) NA_character_ else attr(V, "route")
  vcov_note <- if (inherits(V, "error")) conditionMessage(V) else NA_character_
  if (inherits(V, "error")) V <- NULL

  cum <- reduce_cumulative(st$cb, coef = fit$summary.fixed[cbt, "mean"], vcov = V)
  x_hi <- as.numeric(stats::quantile(st$cb$x_model, 0.95, na.rm = TRUE))
  j <- which.min(abs(cum$x - x_hi))
  data.frame(
    rr_p95 = cum$rr[j], rr_lo = cum$rr_lo[j], rr_hi = cum$rr_hi[j],
    log_rr_p95 = log(cum$rr[j]),
    mlik = as.numeric(fit$mlik[1]),
    n_hyper = length(fit$mode$theta),
    vcov_route = route,
    vcov_note = vcov_note,
    mode_status = as.integer(fit$misc$mode.status %||% NA_integer_)
  )
}
