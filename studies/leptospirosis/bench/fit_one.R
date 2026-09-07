#!/usr/bin/env Rscript
# Fit ONE benchmark specification and record its cost.  BREPI-002.
#
# Every spec is a (structure, controls) pair.  Structures nest M0 -> M4 so a
# runtime jump is attributable to one added component; the diagnostic ladder
# then adds one expensive INLA option at a time to the same structure.  Nothing
# here is confirmatory: the outputs exist to explain cost, not to be quoted.
#
# Usage:
#   Rscript studies/leptospirosis/bench/fit_one.R <spec> [scale]
#
# Writes: data/results/benchmarks/<spec>.json  (one row of the cost table)

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table)
  library(INLA)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: fit_one.R <spec> [scale]")
SPEC <- args[1]
SCALE <- if (length(args) >= 2) args[2] else "health_region"
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "4:1")

cache <- readRDS(file.path("data/interim/bench", paste0("model_data_", SCALE, ".rds")))
dt <- cache$data
meta <- cache$meta
graph <- meta$graph

PC_PREC <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
PC_BYM2 <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)),
                phi  = list(prior = "pc", param = c(0.5, 0.5)))

# --------------------------------------------------------------------------
# Structures. Each returns the right-hand side as a character vector of terms.
# --------------------------------------------------------------------------
.cb  <- meta$cb_terms
.x   <- meta$x_terms

f_bym2 <- sprintf(
  "f(id_space, model='bym2', graph='%s', scale.model=TRUE, constr=TRUE, hyper=%s)",
  graph, paste(deparse(PC_BYM2), collapse = ""))
f_rw1 <- sprintf(
  "f(id_time, model='rw1', scale.model=TRUE, constr=TRUE, hyper=%s)",
  paste(deparse(PC_PREC), collapse = ""))
f_seas <- sprintf(
  "f(id_month, model='rw1', cyclic=TRUE, scale.model=TRUE, constr=TRUE, hyper=%s)",
  paste(deparse(PC_PREC), collapse = ""))
f_int_year <- sprintf(
  "f(id_st_year, model='iid', constr=TRUE, hyper=%s)",
  paste(deparse(PC_PREC), collapse = ""))
f_int_month <- sprintf(
  "f(id_st_month, model='iid', constr=TRUE, hyper=%s)",
  paste(deparse(PC_PREC), collapse = ""))

# Knorr-Held type II: independent across space, RW1 within each area over time.
# Written as a *grouped* effect, so the precision is a Kronecker product rather
# than a diagonal. This is the interaction the pipeline actually requests, and
# it is structurally far denser than the type I forms above -- which is why it
# is benchmarked separately rather than assumed to cost the same.
dt[, id_space_g2 := id_space]
dt[, id_time_g_year := id_time_year]
f_kh2_year <- sprintf(paste0(
  "f(id_space_g2, model='iid', group=id_time_g_year, ",
  "control.group=list(model='rw1', scale.model=TRUE), hyper=%s)"),
  paste(deparse(PC_PREC), collapse = ""))
dt[, id_space_g3 := id_space]
dt[, id_time_g_month := id_time]
f_kh2_month <- sprintf(paste0(
  "f(id_space_g3, model='iid', group=id_time_g_month, ",
  "control.group=list(model='rw1', scale.model=TRUE), hyper=%s)"),
  paste(deparse(PC_PREC), collapse = ""))

STRUCTURES <- list(
  M0  = c("1"),
  M1  = c("1", .x, .cb),
  M2  = c("1", .x, .cb, f_bym2),
  M3  = c("1", .x, .cb, f_bym2, f_rw1, f_seas),
  M4y = c("1", .x, .cb, f_bym2, f_rw1, f_seas, f_int_year),
  M4m = c("1", .x, .cb, f_bym2, f_rw1, f_seas, f_int_month),
  K2y = c("1", .x, .cb, f_bym2, f_rw1, f_seas, f_kh2_year),
  K2m = c("1", .x, .cb, f_bym2, f_rw1, f_seas, f_kh2_month)
)

# --------------------------------------------------------------------------
# Control ladder. `dev` is the cheapest thing that still fits the same model.
# --------------------------------------------------------------------------
ctl <- function(strategy = "gaussian", int_strategy = "eb",
                dic = FALSE, waic = FALSE, cpo = FALSE, config = FALSE) {
  list(
    control.compute = list(dic = dic, waic = waic, cpo = cpo, config = config,
                           return.marginals.predictor = FALSE),
    control.inla = list(strategy = strategy, int.strategy = int_strategy)
  )
}

CONTROLS <- list(
  dev        = ctl(),
  ccd        = ctl(int_strategy = "ccd"),
  adaptive   = ctl(strategy = "adaptive"),
  slaplace   = ctl(strategy = "simplified.laplace"),
  dicwaic    = ctl(dic = TRUE, waic = TRUE),
  cpo        = ctl(cpo = TRUE),
  config     = ctl(config = TRUE),
  full       = ctl(strategy = "adaptive", int_strategy = "ccd",
                   dic = TRUE, waic = TRUE, cpo = TRUE, config = TRUE)
)

# Spec name is "<structure>__<control>", e.g. "M4y__ccd".
parts <- strsplit(SPEC, "__", fixed = TRUE)[[1]]
if (length(parts) != 2) stop("spec must be <structure>__<control>, got: ", SPEC)
struct_name <- parts[1]; ctl_name <- parts[2]
if (is.null(STRUCTURES[[struct_name]])) stop("unknown structure: ", struct_name)
if (is.null(CONTROLS[[ctl_name]])) stop("unknown control: ", ctl_name)

rhs <- c(STRUCTURES[[struct_name]], "offset(log_offset)")
form <- as.formula(paste("cases ~", paste(rhs, collapse = " + ")))
cc <- CONTROLS[[ctl_name]]

# Latent size: fixed effects + every random-effect index actually in the model.
latent <- 1L
if (any(.x %in% rhs)) latent <- latent + length(.x)
if (any(.cb %in% rhs)) latent <- latent + length(.cb)
if (grepl("id_space", paste(rhs, collapse = ""))) latent <- latent + 2L * meta$n_space
if (grepl("id_time,", paste(rhs, collapse = ""))) latent <- latent + meta$n_time
if (grepl("id_month", paste(rhs, collapse = ""))) latent <- latent + 12L
if (grepl("id_st_year", paste(rhs, collapse = ""))) latent <- latent + max(dt$id_st_year)
if (grepl("id_st_month", paste(rhs, collapse = ""))) latent <- latent + max(dt$id_st_month)
if (grepl("id_space_g2", paste(rhs, collapse = ""))) {
  latent <- latent + meta$n_space * max(dt$id_time_year)
}
if (grepl("id_space_g3", paste(rhs, collapse = ""))) {
  latent <- latent + meta$n_space * meta$n_time
}

cat("[bench]", SPEC, "| latent ~", latent, "| rows", nrow(dt), "\n")
cat("[bench] controls:", paste(names(unlist(cc)), unlist(cc), sep = "=", collapse = " "), "\n")

t0 <- Sys.time()
fit <- try(inla(
  form, family = "nbinomial", data = dt,
  control.compute = cc$control.compute,
  control.inla = cc$control.inla,
  control.predictor = list(compute = FALSE, link = 1),
  control.fixed = list(mean = 0, prec = 1, mean.intercept = 0,
                       prec.intercept = 0.001),
  num.threads = THREADS,
  verbose = FALSE
), silent = TRUE)
wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

dir.create("data/results/benchmarks", recursive = TRUE, showWarnings = FALSE)
outfile <- file.path("data/results/benchmarks", paste0(SPEC, ".json"))

if (inherits(fit, "try-error")) {
  res <- list(spec = SPEC, structure = struct_name, control = ctl_name,
              scale = SCALE, ok = FALSE, wall_seconds = wall,
              error = as.character(fit))
} else {
  cpu <- tryCatch(as.list(fit$cpu.used), error = function(e) list())
  names(cpu) <- gsub("[^A-Za-z0-9]+", "_", tolower(names(cpu)))
  res <- list(
    spec = SPEC, structure = struct_name, control = ctl_name, scale = SCALE,
    ok = TRUE, wall_seconds = round(wall, 2),
    latent_size = latent, n_rows = nrow(dt),
    n_hyperpar = length(fit$mode$theta),
    cpu = lapply(cpu, function(z) round(as.numeric(z), 2)),
    mlik = tryCatch(round(as.numeric(fit$mlik[1]), 2), error = function(e) NA),
    dic = tryCatch(round(fit$dic$dic, 2), error = function(e) NA),
    waic = tryCatch(round(fit$waic$waic, 2), error = function(e) NA),
    log_score = tryCatch(
      round(mean(-log(pmax(fit$cpo$cpo, .Machine$double.eps)), na.rm = TRUE), 4),
      error = function(e) NA),
    # The cross-basis coefficients are what a later reduction consumes; keeping
    # them lets the ladder show whether an added component attenuates the
    # exposure signal, which is the scientific half of BREPI-004.
    cb_mean = tryCatch(round(fit$summary.fixed[.cb, "mean"], 4),
                       error = function(e) NA),
    cb_sd = tryCatch(round(fit$summary.fixed[.cb, "sd"], 4),
                     error = function(e) NA),
    x_mean = tryCatch(round(fit$summary.fixed[.x, "mean"], 4),
                      error = function(e) NA),
    object_mb = round(as.numeric(object.size(fit)) / 1024^2, 1)
  )
}

writeLines(jsonlite::toJSON(res, auto_unbox = TRUE, null = "null", digits = 6),
           outfile)
cat("[bench] wrote", outfile, "in", round(wall, 1), "s\n")
