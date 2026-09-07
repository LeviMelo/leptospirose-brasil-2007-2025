#!/usr/bin/env Rscript
# RQ5 -- do Brazilian municipalities fall into distinct transmission regimes?
#
# The hypothesis, prespecified: leptospirosis in Brazil is not one epidemiology
# with a national average, but at least three -- metropolitan-flood,
# rural-occupational and Amazonian-riverine -- whose seasonality, level and
# lethality differ enough that a pooled effect averages unlike things.
#
# The descriptive stage already found the sharpest evidence for this without
# any clustering: the Northeast peaks in JUNE while every other macroregion
# peaks in JANUARY. That is not a nuance of a shared seasonal curve, it is a
# different curve, and it follows the Northeast's April-July rainy season. The
# clustering asks whether that contrast, plus level and structure, resolves
# into a small number of coherent types.
#
# Method: CLR-transformed monthly composition (seasonality is compositional),
# plus log incidence, plus the structural block; PCA; Gaussian mixture with K
# chosen by BIC; and a bootstrap ARI, because a K that BIC likes but resampling
# does not reproduce is not a typology.
#
# Restricted to municipalities with >= 20 confirmed cases. Below that a monthly
# profile is noise: the median non-zero municipality-month has exactly one case.
# The restriction is reported, not hidden, and it retains ~81% of case mass.
#
# Usage: Rscript studies/leptospirosis/18_rq5_regimes.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow) })
for (f in c("00_io.R", "10_regimes.R")) source(file.path("R", f))

PANEL <- "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"
OUT <- "data/results/rq5_regimes"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
SEED <- 20260730L

panel <- read_panel(PANEL, validate = TRUE, prepare = TRUE)
cat("[rq5] panel:", format(nrow(panel), big.mark = ","), "rows\n")

STRUCTURAL <- intersect(
  c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh"),
  names(panel)
)

feat <- regime_features(panel, by = "munic_code", min_events = 20L,
                        structural = STRUCTURAL)
print(feat)

fit <- fit_regimes(feat, k_range = 2:8, seed = SEED)
print(fit)
fwrite(fit$criterion, file.path(OUT, "regime_model_selection.csv"))

cat("\n[rq5] bootstrap stability at the selected K and its neighbours\n")
stab <- lapply(unique(c(max(2, fit$k - 1), fit$k, fit$k + 1)), function(k) {
  s <- regime_stability(feat, k = k, n_boot = 40L, seed = SEED)
  cat(sprintf("   k=%d  mean ARI %.3f  min %.3f\n", k, s$mean_ari, s$min_ari))
  data.table(k = k, mean_ari = s$mean_ari, min_ari = s$min_ari)
})
stab <- rbindlist(stab)
fwrite(stab, file.path(OUT, "regime_stability.csv"))

sel <- stab[k == fit$k]
if (nrow(sel) && is.finite(sel$mean_ari) && sel$mean_ari < 0.5) {
  cat("\n[rq5] WARNING: mean ARI", round(sel$mean_ari, 3), "< 0.50 at the BIC-selected K.\n")
  cat("[rq5] The partition is not reproducible under resampling and must NOT be\n")
  cat("[rq5] presented as a typology. Report the seasonal contrast directly instead.\n")
}

prof <- regime_profiles(fit, panel, by = "munic_code")
cat("\n[rq5] regime summary\n")
print(prof$summary[, .(regime, units, cases, deaths,
                       incidence = round(incidence_per_100k_yr, 1),
                       cfr_pct = round(cfr_pct, 2))])

cat("\n[rq5] seasonal peak month by regime\n")
print(unique(prof$seasonality[, .(regime, peak_month)])[order(regime)])

cat("\n[rq5] regional composition of each regime (municipality counts)\n")
if (nrow(prof$composition)) {
  print(dcast(prof$composition, regime ~ region, value.var = "units", fill = 0))
}

assign_dt <- data.table(munic_code = names(fit$assignment),
                        regime = as.integer(fit$assignment))
geo <- unique(panel[, .(munic_code, name, uf_abbr, region)])
fwrite(merge(assign_dt, geo, by = "munic_code", all.x = TRUE),
       file.path(OUT, "regime_assignment.csv"))
fwrite(prof$seasonality, file.path(OUT, "regime_seasonality.csv"))
fwrite(prof$summary, file.path(OUT, "regime_summary.csv"))
cat("\n[rq5] wrote", OUT, "\n")
