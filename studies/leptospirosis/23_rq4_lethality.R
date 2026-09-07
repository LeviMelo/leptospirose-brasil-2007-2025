#!/usr/bin/env Rscript
# RQ4 -- are the determinants of leptospirosis lethality distinct from the
# determinants of its incidence?
#
# The prior evidence is strong and entirely descriptive. Case fatality falls
# monotonically from 14.5% to 1.2% across municipal incidence bands, and the
# RQ5 regimes separate clusters with near-identical incidence (2.1 to 2.4 per
# 100k) whose case fatality runs from 2.0% to 13.1%. The same organism cannot
# be six times more lethal in one cluster than another, so either the case
# definition, the ascertainment, or the care pathway differs -- and all three
# are health-system properties rather than transmission properties.
#
# This fits the lethality model at the SAME geography and with the SAME
# covariates as the RQ1 confirmatory incidence fit (health region), because the
# question is comparative and a difference between two models that also differ
# in structure is uninterpretable.
#
# The threat to validity that dominates here is not confounding, it is the
# denominator. Case fatality can only be computed on cases whose outcome was
# recorded, and that share moves from 45% in 2007 to 71% in 2025 (T7c). The
# model adjusts for it; the sensitivity below shows what omitting it does.
#
# Usage: Rscript studies/leptospirosis/23_rq4_lethality.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(INLA)
})
for (f in c("00_io.R", "01_descriptive.R", "03_inla_spacetime.R",
            "09_exec.R", "11_tables.R", "12_severity.R")) {
  source(file.path("R", f))
}

OUT <- "data/results/rq4_lethality"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
GRAPH <- "data/panel/graphs/health_region.adj"
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "6:1")
w <- function(x, name) { fwrite(x, file.path(OUT, name))
                         cat("  wrote", name, "-", nrow(x), "rows\n") }
rule <- function(s) cat("\n", strrep("-", 70), "\n", s, "\n", sep = "")

# ---------------------------------------------------------------------------
# Numerator: deaths and known outcomes per health region-year
# ---------------------------------------------------------------------------
rule("Assembling the lethality panel")
line <- as.data.table(read_parquet("data/interim/lept_line_level.parquet"))
line[, is_case := !is.na(classi_fin) & classi_fin == "confirmado"]
conf <- line[is_case == TRUE]
conf[, year := as.integer(src_year)]
conf[, died := !is.na(evolucao) & evolucao == "obito_por_leptospirose"]
conf[, known := evolucao_state == "valid"]
conf[, hospit := !is.na(ate_hosp) & ate_hosp == "sim"]
conf[, lab_confirmed := !is.na(criterio) & criterio == "clinico_laboratorial"]
conf[, onset := as.IDate(DT_SIN_PRI)]
conf[, notified := as.IDate(DT_NOTIFIC)]
conf[, delay := as.numeric(notified - onset)]
conf[delay < 0 | delay > 365, delay := NA_real_]

# The line level keys on a six-digit municipality; the panel keys on seven.
# Crosswalk through the panel itself rather than by recomputing a check digit.
panel <- read_panel(
  "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet",
  validate = TRUE, prepare = TRUE)
xwalk <- unique(panel[, .(munic_code, health_region_code, uf_code, region)])
xwalk[, munic6 := substr(munic_code, 1, 6)]
conf[, munic6 := ID_MN_RESI]
conf <- merge(conf, xwalk[, .(munic6, health_region_code, region)],
              by = "munic6", all.x = TRUE)
unmatched <- sum(is.na(conf$health_region_code))
cat("confirmed cases:", format(nrow(conf), big.mark = ","),
    "| without a health region:", unmatched,
    sprintf("(%.2f%%)", 100 * unmatched / nrow(conf)), "\n")
conf <- conf[!is.na(health_region_code)]

sev <- severity_panel(
  conf, by = c("health_region_code", "year"),
  deaths = "died", outcome_known = "known", cases = "is_case",
  extra = list(
    hosp_share = quote(mean(hospit[known], na.rm = TRUE)),
    lab_share  = quote(mean(lab_confirmed, na.rm = TRUE)),
    median_delay = quote(stats::median(delay, na.rm = TRUE))))
cat("cells:", nrow(sev), "| dropped for a zero denominator:",
    attr(sev, "cells_dropped"), "cells holding",
    attr(sev, "cases_dropped"), "cases\n")
cat("crude CFR:", sprintf("%.2f%%", 100 * sum(sev$deaths) / sum(sev$outcome_known)),
    "\n")

# ---------------------------------------------------------------------------
# Covariates: the same structural block as the RQ1 fit, population-weighted to
# health region-year so the two models see the same construct.
# ---------------------------------------------------------------------------
COVARS <- c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh")
cov_hr <- panel[, {
  wgt <- person_months
  as.list(c(vapply(COVARS, function(v)
    stats::weighted.mean(get(v), wgt, na.rm = TRUE), numeric(1)),
    person_months = sum(wgt), cases_panel = sum(cases)))
}, by = .(health_region_code, year)]

sev <- merge(sev, cov_hr, by = c("health_region_code", "year"), all.x = TRUE)
incomplete <- sev[!stats::complete.cases(sev[, ..COVARS])]
if (nrow(incomplete)) {
  cat("dropping", nrow(incomplete), "cell(s) with a missing covariate\n")
  sev <- sev[stats::complete.cases(sev[, ..COVARS])]
}
# The shared covariates are deliberately left on their RAW scale -- shares in
# [0, 1] and asinh GDP -- because the RQ1 confirmatory fit used them raw
# (sanitation mean 0.451 sd 0.308; urban 0.778/0.143; gdp_asinh 1.857/0.538),
# and compare_determinants() differences two coefficients. Standardising here
# and not there would make every difference an artefact of the rescaling.
# Only the auxiliary covariates, which are never compared, are standardised.
sev[, completeness_z := as.numeric(scale(completeness))]
sev[, median_delay := as.numeric(scale(median_delay))]
sev[is.na(median_delay), median_delay := 0]
sev[, hosp_share_z := as.numeric(scale(hosp_share))]

# Index columns keyed to the adjacency graph's own ordering.
hr_codes <- sort(unique(panel$health_region_code))
sev <- sev[health_region_code %in% hr_codes]
sev[, id_space := match(health_region_code, hr_codes)]
sev[, id_time := year - min(year) + 1L]
cat("model cells:", nrow(sev), "| health regions:", uniqueN(sev$id_space),
    "| years:", uniqueN(sev$id_time), "\n")
w(sev, "rq4_lethality_panel.csv")

# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
rule("Fitting")
ctl <- model_execution_profile("confirmatory", num_threads = THREADS)$controls
BUDGET <- limits(wall_seconds = 900, max_rss_mb = 12000, poll_seconds = 5)

sev[, completeness := completeness_z]   # fit_severity() adjusts on this name

primary <- fit_severity(sev, covariates = COVARS, graph = GRAPH,
                        control = ctl, adjust_completeness = TRUE,
                        num_threads = THREADS)
print(primary)
w(primary$odds, "rq4_primary_odds.csv")

# Health-system block. Kept separate from the primary because hospitalisation
# share and notification delay are plausibly on the causal path between the
# structural covariates and death, so adjusting for them in the primary would
# block part of the very effect being estimated.
health_block <- fit_severity(
  sev, covariates = c(COVARS, "hosp_share_z", "median_delay"),
  graph = GRAPH, control = ctl, adjust_completeness = TRUE,
  num_threads = THREADS)
print(health_block)
w(health_block$odds, "rq4_health_system_odds.csv")

# What omitting the completeness adjustment does. This is the specification a
# reader would write by default, so the size of the difference is the argument
# for the adjustment.
unadjusted <- fit_severity(sev, covariates = COVARS, graph = GRAPH,
                           control = ctl, adjust_completeness = FALSE,
                           num_threads = THREADS)
w(unadjusted$odds, "rq4_unadjusted_odds.csv")

cat("\ncompleteness adjustment: effect on the structural coefficients\n")
cmp <- merge(primary$odds[term %in% COVARS, .(term, or_adj = or)],
             unadjusted$odds[term %in% COVARS, .(term, or_unadj = or)],
             by = "term")
cmp[, shift_pct := round(100 * (or_adj - or_unadj) / or_unadj, 1)]
print(cmp)
w(cmp, "rq4_completeness_adjustment_effect.csv")

# ---------------------------------------------------------------------------
# The comparative question
# ---------------------------------------------------------------------------
rule("Are the determinants of lethality distinct from those of incidence?")
inc_path <- "data/results/rq1_exposure_response/health_region_confirmatory/structural_coefficients.csv"
if (!file.exists(inc_path)) inc_path <- NA_character_
if (is.na(inc_path)) {
  cat("SKIPPED: the RQ1 confirmatory fixed effects are not on disk. Run\n",
      "studies/leptospirosis/11_fit_rq1.R first; the comparison is the point\n",
      "of RQ4 and must not be replaced by a side-by-side eyeball.\n")
} else {
  inc <- fread(inc_path)
  cat("incidence coefficients read from", inc_path, "\n")
  nm <- names(inc)
  term_col <- intersect(c("term", "variable", "parameter", "V1"), nm)[1]
  mean_col <- intersect(c("mean", "estimate", "log_rr"), nm)[1]
  sd_col   <- intersect(c("sd", "se", "std_error"), nm)[1]
  if (is.na(term_col) || is.na(mean_col) || is.na(sd_col)) {
    cat("SKIPPED: could not identify term/mean/sd columns in", inc_path, "\n")
    cat("columns present:", paste(nm, collapse = ", "), "\n")
  } else {
    inc_tab <- inc[, .(term = get(term_col), mean = get(mean_col), sd = get(sd_col))]
    cd <- compare_determinants(inc_tab, primary,
                               label_incidence = "incidence",
                               label_severity = "lethality")
    print(cd)
    w(cd, "rq4_determinant_comparison.csv")
    cat("\n", attr(cd, "scale_caveat"), "\n", sep = "")
    cat("\n", attr(cd, "independence_caveat"), "\n", sep = "")
  }
}

# ---------------------------------------------------------------------------
# Where lethality is high after adjustment
# ---------------------------------------------------------------------------
rule("Residual spatial variation in lethality")
sp <- primary$fit$summary.random$id_space
if (!is.null(sp)) {
  n <- uniqueN(sev$id_space)
  # BYM2 returns the combined effect first, then the structured component.
  combined <- as.data.table(sp[seq_len(n), ])
  combined[, health_region_code := hr_codes[ID]]
  combined[, or := exp(mean)]
  setorder(combined, -or)
  top <- rbind(head(combined, 10), tail(combined, 10))
  w(combined[, .(health_region_code, log_or = mean, sd, or)],
    "rq4_spatial_effects.csv")
  cat("\nhighest and lowest health-region lethality effects (odds ratio,\n",
      "relative to the national mean, after adjustment):\n", sep = "")
  print(top[, .(health_region_code, or = round(or, 3),
                lo = round(exp(`0.025quant`), 3),
                hi = round(exp(`0.975quant`), 3))])
  cat("\nspread: the 90th/10th percentile ratio of the health-region effect is",
      sprintf("%.2f", quantile(combined$or, 0.9) / quantile(combined$or, 0.1)),
      "\n")
}

hy <- as.data.table(primary$fit$summary.hyperpar, keep.rownames = "hyper")
w(hy, "rq4_hyperparameters.csv")
print(hy[, .(hyper, mean = round(mean, 3), sd = round(sd, 3))])

if (!is.null(primary$scores)) {
  cat("\nmodel scores:\n"); print(primary$scores)
}
assert_no_strays(kill = TRUE)
rule("done")
cat("wrote", OUT, "\n")
