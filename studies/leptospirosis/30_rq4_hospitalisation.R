#!/usr/bin/env Rscript
# RQ4, second outcome: hospitalisation, and an independent check on lethality.
#
# The protocol asks whether the determinants of HOSPITALISATION and DEATH
# differ from the determinants of reported incidence. 23_rq4_lethality.R
# answered it for death. This answers it for hospitalisation, and then does
# something the death model could not: it re-asks the lethality question using
# a system that does not know what SINAN recorded.
#
# THREE MODELS, and the third is the point:
#
#   (a) Hospitalisation among confirmed SINAN cases.
#       Binomial(sinan_hospitalised / sinan_confirmed). Cleaner than the
#       lethality model in one important respect: ATE_HOSP is 93.4% valid
#       against EVOLUCAO's 64.6%, so there is no large ascertained-denominator
#       problem to adjust away.
#
#   (b) In-hospital fatality among SIH admissions.
#       Binomial(sih_a27_deaths_in_hospital / sih_a27_admissions). This is a
#       lethality estimate built from hospital billing records, generated with
#       no reference to SINAN at all.
#
#   (c) The comparison of (b) against the SINAN case-fatality model.
#       If the spatial pattern of lethality is REAL, two systems that do not
#       observe each other should show the same one. If it is an artefact of
#       who gets notified, they should not. This is the strongest test of the
#       RQ4 result available in these data, and it is only possible now that
#       SIH covers 2008-2024 rather than four early years.
#
# Usage: Rscript studies/leptospirosis/30_rq4_hospitalisation.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(INLA)
})
for (f in c("00_io.R", "01_descriptive.R", "03_inla_spacetime.R", "09_exec.R",
            "11_tables.R", "12_severity.R")) {
  source(file.path("R", f))
}

OUT <- "data/results/rq4_lethality"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
GRAPH <- "data/panel/graphs/health_region.adj"
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "6:1")
COVARS <- c("sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh")
w <- function(x, name) { fwrite(x, file.path(OUT, name))
                         cat("  wrote", name, "-", nrow(x), "rows\n") }
rule <- function(s) cat("\n", strrep("-", 70), "\n", s, "\n", sep = "")

# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
rule("Assembling")
tri <- as.data.table(read_parquet("data/panel/triangulation_municipality_year.parquet"))
panel <- read_panel(
  "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet",
  validate = FALSE, prepare = TRUE)

cov_hr <- panel[, {
  wgt <- person_months
  as.list(c(vapply(COVARS, function(v)
    stats::weighted.mean(get(v), wgt, na.rm = TRUE), numeric(1)),
    person_months = sum(wgt)))
}, by = .(health_region_code, year)]

hr <- tri[!is.na(health_region_code),
          .(sinan_confirmed = sum(sinan_confirmed, na.rm = TRUE),
            sinan_hospitalised = sum(sinan_hospitalised, na.rm = TRUE),
            sinan_deaths = sum(sinan_deaths, na.rm = TRUE),
            sih_adm = sum(sih_a27_admissions, na.rm = TRUE),
            sih_deaths = sum(sih_a27_deaths_in_hospital, na.rm = TRUE),
            sih_covered = any(sih_covered)),
          by = .(health_region_code, year)]
hr <- merge(hr, cov_hr, by = c("health_region_code", "year"), all.x = TRUE)
hr <- hr[stats::complete.cases(hr[, ..COVARS])]

hr_codes <- sort(unique(panel$health_region_code))
hr <- hr[health_region_code %in% hr_codes]
hr[, id_space := match(health_region_code, hr_codes)]
hr[, id_time := year - min(year) + 1L]

cat("health region-years:", nrow(hr), "| regions:", uniqueN(hr$id_space), "\n")
cat("SINAN confirmed:", format(sum(hr$sinan_confirmed), big.mark = ","),
    "| hospitalised:", format(sum(hr$sinan_hospitalised), big.mark = ","),
    sprintf("(%.1f%%)\n", 100 * sum(hr$sinan_hospitalised) / sum(hr$sinan_confirmed)))
cat("SIH admissions:", format(sum(hr$sih_adm), big.mark = ","),
    "| in-hospital deaths:", format(sum(hr$sih_deaths), big.mark = ","),
    sprintf("(%.1f%%)\n", 100 * sum(hr$sih_deaths) / max(sum(hr$sih_adm), 1)), "\n")

ctl <- model_execution_profile("confirmatory", num_threads = THREADS)$controls

# ---------------------------------------------------------------------------
# (a) Hospitalisation among confirmed cases
# ---------------------------------------------------------------------------
rule("(a) Hospitalisation risk among confirmed SINAN cases")
hosp_panel <- severity_panel(
  hr[sinan_confirmed > 0], by = c("health_region_code", "year"),
  deaths = "sinan_hospitalised", outcome_known = "sinan_confirmed",
  cases = "sinan_confirmed",
  extra = list(sanitation_sewer_share = quote(sanitation_sewer_share[1]),
               urban_share = quote(urban_share[1]),
               gdp_per_capita_asinh = quote(gdp_per_capita_asinh[1]),
               id_space = quote(id_space[1]), id_time = quote(id_time[1])))
cat("cells:", nrow(hosp_panel), "| dropped for a zero denominator:",
    attr(hosp_panel, "cells_dropped"), "\n")

# `completeness` here is identically 1 by construction (the denominator IS the
# case count), so there is nothing to adjust for and adjusting would add a
# constant column. Stated rather than silently skipped.
hosp_fit <- fit_severity(hosp_panel, covariates = COVARS, graph = GRAPH,
                         control = ctl, adjust_completeness = FALSE,
                         num_threads = THREADS)
print(hosp_fit)
w(hosp_fit$odds, "rq4b_hospitalisation_odds.csv")

# ---------------------------------------------------------------------------
# (b) In-hospital fatality among SIH admissions
# ---------------------------------------------------------------------------
rule("(b) In-hospital fatality among SIH A27 admissions")
sih <- hr[sih_covered == TRUE & sih_adm > 0]
cat("SIH-covered health region-years:", nrow(sih),
    "| years", min(sih$year), "-", max(sih$year), "\n")
sih_panel <- severity_panel(
  sih, by = c("health_region_code", "year"),
  deaths = "sih_deaths", outcome_known = "sih_adm", cases = "sih_adm",
  extra = list(sanitation_sewer_share = quote(sanitation_sewer_share[1]),
               urban_share = quote(urban_share[1]),
               gdp_per_capita_asinh = quote(gdp_per_capita_asinh[1]),
               id_space = quote(id_space[1]), id_time = quote(id_time[1])))
sih_fit <- fit_severity(sih_panel, covariates = COVARS, graph = GRAPH,
                        control = ctl, adjust_completeness = FALSE,
                        num_threads = THREADS)
print(sih_fit)
w(sih_fit$odds, "rq4c_sih_inhospital_fatality_odds.csv")

# ---------------------------------------------------------------------------
# (c) Do two independent systems see the same lethality geography?
# ---------------------------------------------------------------------------
rule("(c) SINAN case fatality vs SIH in-hospital fatality, by health region")
sinan_sp_path <- file.path(OUT, "rq4_spatial_effects.csv")
if (!file.exists(sinan_sp_path)) {
  cat("SKIPPED: run 23_rq4_lethality.R first; its spatial effects are the\n")
  cat("comparison target and must not be re-derived here.\n")
} else {
  sinan_sp <- fread(sinan_sp_path)
  n <- uniqueN(sih_panel$id_space)
  sp <- sih_fit$fit$summary.random$id_space
  sih_sp <- as.data.table(sp[seq_len(n), ])
  sih_sp[, health_region_code := hr_codes[ID]]
  sih_sp[, or_sih := exp(mean)]

  cmp <- merge(sinan_sp[, .(health_region_code = as.character(health_region_code),
                            or_sinan = or)],
               sih_sp[, .(health_region_code = as.character(health_region_code),
                          or_sih)],
               by = "health_region_code")
  cmp <- cmp[is.finite(or_sinan) & is.finite(or_sih)]
  w(cmp, "rq4d_lethality_system_agreement.csv")

  ps <- stats::cor(log(cmp$or_sinan), log(cmp$or_sih), method = "pearson")
  sp_r <- stats::cor(cmp$or_sinan, cmp$or_sih, method = "spearman")
  ct <- stats::cor.test(log(cmp$or_sinan), log(cmp$or_sih))
  cat(sprintf("\nhealth regions in both models: %d\n", nrow(cmp)))
  cat(sprintf("Pearson correlation of log odds ratios: %+.3f (95%% CI %+.3f to %+.3f, p = %.3g)\n",
              ps, ct$conf.int[1], ct$conf.int[2], ct$p.value))
  cat(sprintf("Spearman rank correlation:              %+.3f\n", sp_r))

  # Agreement on the extremes is what matters for a policy reading.
  cmp[, q_sinan := cut(or_sinan, stats::quantile(or_sinan, 0:4 / 4),
                       include.lowest = TRUE, labels = 1:4)]
  cmp[, q_sih := cut(or_sih, stats::quantile(or_sih, 0:4 / 4),
                     include.lowest = TRUE, labels = 1:4)]
  tab <- table(cmp$q_sinan, cmp$q_sih)
  cat("\nquartile agreement (rows SINAN, columns SIH):\n"); print(tab)
  cat(sprintf("\nexact quartile agreement: %.1f%%; within one quartile: %.1f%%\n",
              100 * sum(diag(tab)) / sum(tab),
              100 * sum(tab[abs(row(tab) - col(tab)) <= 1]) / sum(tab)))
  cat("\nREADING: a positive correlation means two systems that do not observe\n")
  cat("each other place lethality in the same health regions, which is what a\n")
  cat("real severity gradient predicts and a notification artefact does not.\n")
  cat("A null or negative correlation would indict the RQ4 spatial result.\n")
}

assert_no_strays(kill = TRUE)
rule("done"); cat("wrote", OUT, "\n")
