#!/usr/bin/env Rscript
# RQ1 multi-scale specification -- DEFINITIONS ONLY, no side effects.
#
# Sourced by the orchestration script and, independently, by every bounded child
# that fits one scale.
#
# The protocol is explicit that this is a result and not an appendix: "Movement
# of estimated cumulative risk, peak lag and uncertainty across scales is itself
# a result. It must not be hidden as a technical robustness appendix." The
# modifiable areal unit problem is not a nuisance here -- the study's central
# design choice is that the canonical data grain (municipality) and the primary
# fitting grain (health region) are different, and the justification for that
# choice is exactly how the estimate behaves across the two.
#
# Every scale is fitted with the SAME model form and the SAME confirmatory
# profile. A scale comparison in which the scales also differ in specification
# measures nothing.

suppressPackageStartupMessages({
  library(data.table); library(INLA); library(dlnm); library(arrow)
})

SCALE_PANEL <- "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"

# Which column defines the area at each scale, and which graph goes with it.
SCALE_DEF <- list(
  municipality_endemic = list(key = "munic_code",
                              graph = "data/panel/graphs/municipality_endemic.adj",
                              endemic_min = 20L),
  health_region        = list(key = "health_region_code",
                              graph = "data/panel/graphs/health_region.adj"),
  immediate_region     = list(key = "immediate_region_code",
                              graph = "data/panel/graphs/immediate_region.adj"),
  microregion          = list(key = "microregion_code",
                              graph = "data/panel/graphs/microregion.adj"),
  uf                   = list(key = "uf_code",
                              graph = "data/panel/graphs/uf.adj")
)

.scale_state <- function() {
  if (!exists(".brepi_scale_panel", envir = globalenv())) {
    for (f in c("00_io.R", "02_crossbasis.R", "03_inla_spacetime.R")) {
      source(file.path("R", f))
    }
    p <- read_panel(SCALE_PANEL, validate = FALSE, prepare = TRUE)
    assign(".brepi_scale_panel", p, envir = globalenv())
  }
  get(".brepi_scale_panel", envir = globalenv())
}

#' Aggregate the municipal panel to one analytic scale.
#'
#' Cases and person-months are EXTENSIVE and are summed. Precipitation is
#' INTENSIVE and is averaged weighted by person-months, which is the dasymetric
#' choice: a health region's exposure is what its population experienced, not
#' what its territory did. Structural shares are population-weighted for the
#' same reason.
scale_panel <- function(level) {
  def <- SCALE_DEF[[level]]
  if (is.null(def)) stop("unknown scale: ", level, call. = FALSE)
  d <- data.table::copy(.scale_state())
  if (!is.null(def$endemic_min)) {
    tot <- d[, .(total = sum(cases, na.rm = TRUE)), by = munic_code]
    keep <- tot[total >= def$endemic_min, munic_code]
    d <- d[munic_code %in% keep]
  }
  d[, area_code := as.character(get(def$key))]
  d <- d[!is.na(area_code) & area_code != ""]

  XT <- c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh")
  agg <- d[, {
    pm <- sum(person_months, na.rm = TRUE)
    out <- list(cases = sum(cases, na.rm = TRUE),
                person_months = pm,
                precip_mm = stats::weighted.mean(precip_mm, person_months,
                                                 na.rm = TRUE))
    for (v in XT) {
      out[[v]] <- stats::weighted.mean(get(v), person_months, na.rm = TRUE)
    }
    out
  }, by = .(area_code, date, year, month, time_index)]
  agg[, log_offset := log(pmax(person_months, 1))]
  data.table::setorderv(agg, c("area_code", "time_index"))
  agg[]
}

#' The adjacency graph for a scale, built from the units actually analysed.
#'
#' A pre-built graph and an independently-recomputed unit set will eventually
#' disagree, and INLA's report of the disagreement is an index dump rather than
#' a diagnosis: the endemic-municipality fit failed with "f(id_space).
#' Covariate does not match 'values' 225 times" because the shipped graph has
#' 496 nodes and the >= 20-case rule selects 497. One municipality.
#'
#' The graph must therefore be derived FROM the analysed set, not matched to
#' it. For the fixed administrative scales the shipped graph is definitional
#' and is used as-is; for the outcome-conditioned endemic subset it is built
#' here and cached, so the two can never drift apart.
scale_graph <- function(level, codes) {
  def <- SCALE_DEF[[level]]
  if (is.null(def$endemic_min)) return(def$graph)
  dir.create("data/panel/graphs", recursive = TRUE, showWarnings = FALSE)
  f <- sprintf("data/panel/graphs/municipality_endemic_min%d_n%d.adj",
               def$endemic_min, length(codes))
  if (!file.exists(f)) {
    .need("sf")
    geo <- sf::st_read("data/panel/geo_municipality.gpkg", quiet = TRUE)
    build_graph(geo, code_col = "munic_code", codes = codes, file = f)
  }
  f
}

#' Fit one scale. Same form, same profile, everywhere.
scale_fit <- function(spec, ...) {
  suppressPackageStartupMessages({
    library(data.table); library(INLA); library(dlnm)
  })
  def <- SCALE_DEF[[spec$level]]
  d <- scale_panel(spec$level)

  cbo <- build_crossbasis(d, lag_max = 3L, group_col = "area_code",
                          time_col = "time_index", per_unit_knots = TRUE)
  bound <- bind_crossbasis(d, cbo)
  dt <- bound$data
  cb_terms <- bound$terms

  area_levels <- sort(unique(dt$area_code))
  dt[, id_space := as.integer(factor(area_code, levels = area_levels))]
  graph_file <- scale_graph(spec$level, area_levels)
  dt[, id_time := as.integer(factor(time_index))]
  dt[, id_month := month]
  dt[, id_st_year := as.integer(factor(paste(id_space, year)))]

  pc_prec <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
  dep <- function(x) paste(deparse(x), collapse = "")
  XT <- c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh")
  rhs <- c("1", XT, cb_terms,
           sprintf("f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
                   graph_file,
                   dep(list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                            phi = list(prior = "pc", param = c(0.5, 0.5))))),
           sprintf("f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)", dep(pc_prec)),
           sprintf("f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)", dep(pc_prec)),
           sprintf("f(id_st_year, model='iid', constr=TRUE, hyper=%s)", dep(pc_prec)),
           "offset(log_offset)")

  ctl <- model_execution_profile("confirmatory", num_threads = spec$threads)$controls
  fit <- inla(as.formula(paste("cases ~", paste(rhs, collapse = " + "))),
              family = "nbinomial", data = dt,
              control.compute = ctl$control.compute, control.inla = ctl$control.inla,
              control.predictor = ctl$control.predictor,
              control.fixed = ctl$control.fixed,
              num.threads = spec$threads, verbose = FALSE)
  list(fit = fit, cb = cbo, cb_terms = cb_terms,
       n_units = data.table::uniqueN(dt$area_code), n_cells = nrow(dt),
       nonzero_share = mean(dt$cases > 0, na.rm = TRUE))
}

#' The estimand, evaluated identically at every scale.
scale_estimand <- function(obj, spec, ...) {
  fit <- obj$fit
  V <- tryCatch(inla_fixed_vcov(fit, obj$cb_terms), error = function(e) NULL)
  cum <- reduce_cumulative(obj$cb, coef = fit$summary.fixed[obj$cb_terms, "mean"],
                           vcov = V)
  x_hi <- as.numeric(stats::quantile(obj$cb$x_model, 0.95, na.rm = TRUE))
  j <- which.min(abs(cum$x - x_hi))

  # Peak lag, on the same 0-3 grid at every scale.
  surf <- tryCatch(exposure_lag_surface(obj$cb, fit, obj$cb_terms,
                                        at_percentile = 0.95),
                   error = function(e) NULL)
  peak <- if (!is.null(surf) && !is.null(surf$lag_curve))
    surf$lag_curve$lag[which.max(surf$lag_curve$rr)] else NA_real_

  sc <- tryCatch(model_scores(fit), error = function(e) NULL)
  data.frame(
    level = spec$level,
    n_units = obj$n_units, n_cells = obj$n_cells,
    nonzero_share = obj$nonzero_share,
    rr_p95 = cum$rr[j], rr_lo = cum$rr_lo[j], rr_hi = cum$rr_hi[j],
    ci_width = cum$rr_hi[j] - cum$rr_lo[j],
    peak_lag = peak,
    dic = if (is.null(sc)) NA_real_ else sc$dic,
    waic = if (is.null(sc)) NA_real_ else sc$waic,
    cpo_failures = if (is.null(sc)) NA_integer_ else sc$n_cpo_failure,
    vcov_route = if (is.null(V)) NA_character_ else attr(V, "route")
  )
}
