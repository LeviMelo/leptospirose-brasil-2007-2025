#!/usr/bin/env Rscript
# Prepare the RQ1 model data once, so that fit benchmarks time the *fit*.
#
# BREPI-002 requires runtime to be attributable to a model component or a
# diagnostic.  Aggregation, cross-basis construction and graph contraction are
# data preparation and must not appear inside a fit timing.  This script does
# them once and caches the result; `fit_one.R` loads the cache and does nothing
# but call `inla()`.
#
# Usage:
#   Rscript studies/leptospirosis/bench/00_prepare_model_data.R [scale]
#
# Writes: data/interim/bench/model_data_<scale>.rds
#         data/interim/bench/graph_<scale>.adj (copied/contracted)

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
})

for (f in c("00_io.R", "00_analysis_spec.R", "02_crossbasis.R", "03_inla_spacetime.R")) {
  source(file.path("R", f))
}

args <- commandArgs(trailingOnly = TRUE)
SCALE <- if (length(args) >= 1) args[1] else "health_region"

PANEL <- Sys.getenv(
  "BREPI_PANEL",
  "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"
)
OUTDIR <- "data/interim/bench"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

X_TERMS <- c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh")

cat("[prepare] panel:", PANEL, "\n")
panel <- read_panel(PANEL, validate = TRUE, prepare = TRUE)
cat("[prepare] municipality-month rows:", format(nrow(panel), big.mark = ","), "\n")

cat("[prepare] aggregating to", SCALE, "\n")
agg <- aggregate_panel(panel, SCALE, drop_na_key = TRUE)
setorderv(agg, c("id_space", "id_time"))
cat("[prepare] units:", uniqueN(agg$id_space),
    " periods:", uniqueN(agg$id_time),
    " rows:", format(nrow(agg), big.mark = ","), "\n")

# Cross-basis: exposure is precipitation, lags 0-3 months.  Constructed on the
# aggregated panel, grouped by unit so no lag leaks across space.
cat("[prepare] building cross-basis\n")
cb <- build_crossbasis(agg, exposure_col = "precip_mm",
                       group_col = "area_code", time_col = "time_index",
                       lag_max = 3L)
bound <- bind_crossbasis(agg, cb)
model_data <- data.table::as.data.table(bound$data)
cat("[prepare] cross-basis edge rows dropped:", bound$edge_rows_dropped, "\n")
model_data <- add_inla_indices(model_data, interaction_time_scale = "month")

# The annual interaction index must be present too; add_inla_indices only emits
# the scale it was asked for, so derive the annual one explicitly and keep both
# so a single cached object serves every interaction spec.
n_space <- max(model_data$id_space)
model_data[, id_st_month := (id_time - 1L) * n_space + id_space]
model_data[, id_time_year := (id_time - 1L) %/% 12L + 1L]
model_data[, id_st_year := (id_time_year - 1L) * n_space + id_space]

cb_terms <- bound$terms
stopifnot(all(cb_terms %in% names(model_data)))

missing_x <- setdiff(X_TERMS, names(model_data))
if (length(missing_x)) {
  stop("[prepare] adjustment block missing: ", paste(missing_x, collapse = ", "))
}

# Numerical conditioning of the design block. BREPI-001 asks explicitly whether
# the cross-basis is well conditioned after aggregation; answer it here, once,
# rather than inferring it from a slow fit.
des <- as.matrix(model_data[, c(cb_terms, X_TERMS), with = FALSE])
des <- des[stats::complete.cases(des), , drop = FALSE]
sv <- svd(scale(des))$d
cond <- max(sv) / min(sv)
cat("[prepare] design block: ", ncol(des), " columns, condition number ",
    format(cond, digits = 4), "\n", sep = "")
if (!is.finite(cond) || cond > 1e6) {
  cat("[prepare] WARNING: design block is ill-conditioned; the fit may be slow\n")
  cat("[prepare] for reasons that are numerical rather than structural.\n")
}

graph <- file.path("data/panel/graphs", paste0(SCALE, ".adj"))
if (!file.exists(graph)) stop("[prepare] graph not found: ", graph)

meta <- list(
  scale = SCALE,
  panel = PANEL,
  n_space = n_space,
  n_time = max(model_data$id_time),
  n_rows = nrow(model_data),
  cb_terms = cb_terms,
  x_terms = X_TERMS,
  graph = graph,
  design_condition_number = cond,
  zero_share = mean(model_data$cases == 0),
  total_cases = sum(model_data$cases),
  prepared_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

out <- file.path(OUTDIR, paste0("model_data_", SCALE, ".rds"))
saveRDS(list(data = model_data, crossbasis = cb, meta = meta), out, compress = FALSE)
cat("[prepare] wrote", out, "\n")
cat("[prepare] zero share:", round(meta$zero_share, 4),
    " total cases:", meta$total_cases, "\n")
str(meta[c("scale", "n_space", "n_time", "n_rows", "design_condition_number")])
