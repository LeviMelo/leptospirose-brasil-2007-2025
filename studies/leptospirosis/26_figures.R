#!/usr/bin/env Rscript
# Manuscript figures.
#
# Every figure is built from a result table already on disk, never from a fit
# held in memory. That is deliberate: a figure regenerated from a stale table is
# a visible inconsistency with the number in the text, whereas a figure
# regenerated from a re-run model is an invisible one.
#
# Three wording obligations are enforced HERE rather than left to the writing,
# because a caption is where a reader meets the number and is exactly where the
# qualification gets lost (ISSUE_LEDGER BREPI-014, -015, -016):
#
#   * the RQ1 exposure is per LOCAL IQR above the LOCAL median, never per mm;
#   * the peak lag of 0 months is a monthly-aliasing artefact, not a finding;
#   * the structural main effects are not separately identified against the
#     spatial field.
#
# Usage: Rscript studies/leptospirosis/26_figures.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(ggplot2)
})
for (f in c("00_io.R", "01_descriptive.R", "07_figures.R", "11_tables.R")) {
  source(file.path("R", f))
}

RES <- "data/results"
FIG <- file.path(RES, "figures")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

# The exposure-scale sentence, written once and reused, so no figure can carry
# a different wording of it than any other.
EXPOSURE_SCALE_RAW <- paste(
  "Exposure is the monthly precipitation anomaly expressed as",
  "(x - unit median) / unit IQR: a rate ratio is per LOCAL IQR above the",
  "LOCAL median, not per millimetre. Health-region monthly IQR spans",
  "81 mm (p10) to 211 mm (p90).")
EXPOSURE_SCALE <- wrap_caption(EXPOSURE_SCALE_RAW)
LAG_CAVEAT_RAW <- paste(
  "Lag is resolved monthly. Leptospirosis incubates in 2-30 days, so the",
  "apparent peak at lag 0 is an aliasing artefact of the monthly panel and",
  "is not evidence of an immediate effect.")
LAG_CAVEAT <- wrap_caption(LAG_CAVEAT_RAW)
STRUCTURAL_CAVEAT_RAW <- paste(
  "Structural main effects are not separately identified against the BYM2",
  "spatial field and are not interpreted; the identified structural quantity",
  "is the precipitation x sanitation interaction.")
STRUCTURAL_CAVEAT <- wrap_caption(STRUCTURAL_CAVEAT_RAW)

made <- character()
skip <- character()
fig <- function(name, expr, columns = 2, height = 110) {
  # ggplot is lazy: an aesthetic error surfaces when the object is RENDERED,
  # not when it is built. Guarding only construction let one broken figure
  # abort the whole run, which is the opposite of what a figure driver should
  # do -- a missing panel is a gap, an aborted run is no figures at all.
  fail <- function(e) {
    msg <- trimws(conditionMessage(e))
    skip <<- c(skip, sprintf("%s -- %s", name, msg))
    cat(sprintf("  SKIP  %-34s %s\n", name, substr(msg, 1, 90)))
    invisible(NULL)
  }
  tryCatch({
    p <- force(expr)
    f <- file.path(FIG, paste0(name, ".png"))
    save_figure(p, f, columns = columns, height_mm = height)
    made <<- c(made, name)
    cat(sprintf("  ok    %-34s %s\n", name, basename(f)))
    invisible(p)
  }, error = fail)
}
have <- function(path) file.exists(path)

cat("\n-- RQ1 -----------------------------------------------------------\n")
surf_path <- file.path(RES, "rq1_exposure_response/health_region_confirmatory",
                       "exposure_lag_surface.csv")
cum_path <- file.path(RES, "rq1_exposure_response/health_region_confirmatory",
                      "cumulative_exposure_response.csv")
lag_path <- file.path(RES, "rq1_exposure_response/health_region_confirmatory",
                      "lag_response_high_exposure.csv")
if (have(surf_path) && have(cum_path) && have(lag_path)) {
  surf <- list(surface = fread(surf_path), cumulative = fread(cum_path),
               lag_curve = fread(lag_path))
  fig("F1_exposure_lag_surface",
      plot_exposure_lag_surface(
        surf, exposure_label = "Precipitation anomaly (local IQR above local median)") +
        patchwork::plot_annotation(
          caption = wrap_caption(EXPOSURE_SCALE_RAW, LAG_CAVEAT_RAW)),
      columns = 2, height = 120)
} else {
  skip <- c(skip, "F1 -- RQ1 surface tables absent")
  cat("  SKIP  F1_exposure_lag_surface            RQ1 tables absent\n")
}

h2_path <- file.path(RES, "rq1_exposure_response/health_region_confirmatory_h2",
                     "h2_effect_modification.csv")
if (have(h2_path)) {
  h2 <- fread(h2_path)
  nm <- names(h2)
  xcol <- intersect(c("sanitation_percentile", "percentile", "label"), nm)[1]
  fig("F2_h2_sanitation_modification", {
    d <- copy(h2)
    d[, lab := factor(get(xcol), levels = get(xcol))]
    ggplot(d, aes(x = lab, y = rr_p95)) +
      geom_hline(yintercept = 1, linetype = 2, linewidth = 0.3, colour = "grey40") +
      geom_pointrange(aes(ymin = rr_lo, ymax = rr_hi), colour = brepi_palette(1),
                      linewidth = 0.6, fatten = 3) +
      labs(x = "Sanitation coverage percentile of the health region",
           y = "Cumulative rate ratio at the p95 anomaly",
           title = "Rainfall response is steeper where sanitation is worse",
           caption = wrap_caption(EXPOSURE_SCALE_RAW, STRUCTURAL_CAVEAT_RAW)) +
      brepi_theme()
  }, columns = 1, height = 85)
}

sens_path <- file.path(RES, "rq1_exposure_response/health_region_sensitivity",
                       "rq1_sensitivity.csv")
if (have(sens_path)) {
  fig("F3_rq1_sensitivity", {
    s <- fread(sens_path)
    s[, lab := factor(run_id, levels = rev(run_id))]
    fitted <- s[ok == TRUE]
    failed <- s[ok == FALSE]
    ref_rr <- fitted[run_id == "reference", rr_p95][1]
    floor_rr <- min(fitted$rr_lo, na.rm = TRUE)
    p <- ggplot(fitted, aes(x = lab, y = rr_p95)) +
      geom_hline(yintercept = ref_rr,
                 linetype = 2, linewidth = 0.3, colour = "grey40") +
      geom_pointrange(aes(ymin = rr_lo, ymax = rr_hi),
                      colour = brepi_palette(1), linewidth = 0.5, fatten = 2.5) +
      coord_flip() +
      labs(x = NULL, y = "Cumulative RR at the p95 anomaly",
           title = "Prespecified robustness battery",
           subtitle = sprintf(
             "Largest movement %.1f%%; %d specification(s) failed to fit and are reported as such",
             100 * max(abs(fitted$relative_change), na.rm = TRUE), nrow(failed)),
           caption = EXPOSURE_SCALE) +
      brepi_theme()
    if (nrow(failed)) {
      p <- p + geom_text(data = failed, aes(x = lab, y = floor_rr),
                         label = "did not fit", size = 2.2, hjust = 0,
                         colour = "grey30")
    }
    p
  }, columns = 1, height = 85)
}

cat("\n-- Descriptive ---------------------------------------------------\n")
lor_path <- file.path(RES, "01_descriptive/T6b_lorenz.csv")
if (have(lor_path)) {
  fig("F4_concentration", plot_concentration(fread(lor_path)),
      columns = 1, height = 85)
}

seas_path <- file.path(RES, "01_descriptive/T5_seasonality.csv")
panel_path <- "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"
if (have(panel_path)) {
  panel <- read_panel(panel_path, validate = FALSE, prepare = TRUE)
  seas <- try(seasonality_decomposition(panel, by = "region"), silent = TRUE)
  if (!inherits(seas, "try-error")) {
    fig("F5_seasonality_ridges",
        plot_seasonality_ridges(seas, by = "region"),
        columns = 1, height = 95)
  }
}

sero_path <- file.path(RES, "01_descriptive/T11c_reservoir_by_region.csv")
if (have(sero_path)) {
  fig("F6_reservoir_composition", plot_reservoir_composition(fread(sero_path)),
      columns = 2, height = 95)
}

cat("\n-- RQ3 -----------------------------------------------------------\n")
sil_path <- file.path(RES, "rq3_ascertainment/16_surveillance_completeness_map.csv")
geo_path <- "data/panel/geo_municipality.gpkg"
if (have(sil_path) && have(geo_path)) {
  fig("F7_surveillance_completeness", {
    geo <- sf::st_read(geo_path, quiet = TRUE)
    tbl <- fread(sil_path)
    lv <- c("corroborated", "SINAN only", "DETECTION-SILENT (proven)",
            "silent in all systems")
    tbl[, cls := factor(surveillance_class, levels = lv)]
    g <- merge(geo, as.data.frame(tbl[, .(munic_code, cls)]),
               by = "munic_code", all.x = TRUE)
    ggplot(g) +
      geom_sf(aes(fill = cls), colour = NA) +
      scale_fill_manual(values = stats::setNames(brepi_palette(4), lv),
                        name = NULL, na.value = "grey90", drop = FALSE) +
      labs(title = "What each municipality's silence means",
           caption = wrap_caption(
             "A municipality with an A27 death certificate or hospital",
             "admission and zero SINAN confirmations is DETECTION-silent: the",
             "disease was recognised by another system and never notified.",
             "Municipalities silent in every system cannot be adjudicated.")) +
      brepi_theme(grid = "none") +
      theme(axis.line = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank())
  }, columns = 1, height = 120)
}

cnt_path <- file.path(RES, "rq3_ascertainment/13_detection_counterfactual.csv")
if (have(cnt_path)) {
  fig("F8_detection_counterfactual", {
    d <- fread(cnt_path)
    ggplot(d, aes(x = factor(reference_quantile), y = ratio)) +
      geom_hline(yintercept = 1, linetype = 2, linewidth = 0.3, colour = "grey40") +
      geom_col(fill = brepi_palette(1), width = 0.6) +
      geom_text(aes(label = sprintf("%.2fx", ratio)), vjust = -0.4, size = 2.6) +
      labs(x = "Reference: every state notifies at least at this quantile of state effort",
           y = "Expected / observed confirmed cases",
           title = "How much disease the country would have measured",
           caption = wrap_caption(
             "A RANGE, not an estimate: the choice of reference is the estimate.",
             "Assumes effort acts on detection rather than on transmission, and",
             "that the reference state is ascertainment-complete, which it is not.")) +
      brepi_theme(grid = "y")
  }, columns = 1, height = 85)
}

cat("\n-- RQ4 -----------------------------------------------------------\n")
cmp_path <- file.path(RES, "rq4_lethality/rq4_determinant_comparison.csv")
if (have(cmp_path)) {
  fig("F9_determinant_contrast",
      plot_determinant_contrast(
        fread(cmp_path),
        labels = c(sanitation_sewer_share = "Sewer coverage",
                   urban_share = "Urban share",
                   gdp_per_capita_asinh = "GDP per capita (asinh)")) +
        labs(caption = wrap_caption(
          "Coefficients are on each model's own link scale (log rate ratio",
          "versus log odds ratio); case fatality is not rare, so read the sign",
          "and the ordering, not the magnitude.")),
      columns = 1, height = 80)
}

sp_path <- file.path(RES, "rq4_lethality/rq4_spatial_effects.csv")
if (have(sp_path)) {
  fig("F10_lethality_spatial", {
    d <- fread(sp_path)
    setorder(d, -or)
    d[, rank := .I]
    ggplot(d, aes(x = rank, y = or)) +
      geom_hline(yintercept = 1, linetype = 2, linewidth = 0.3, colour = "grey40") +
      geom_line(colour = brepi_palette(1), linewidth = 0.6) +
      scale_y_continuous(trans = "log10") +
      labs(x = "Health region, ranked", y = "Lethality odds ratio vs national mean",
           title = "Residual health-region variation in case fatality",
           subtitle = sprintf("90th/10th percentile ratio %.2f-fold after adjustment",
                              quantile(d$or, 0.9) / quantile(d$or, 0.1)),
           caption = wrap_caption("Adjusted for sanitation, urban share, GDP per capita",
                           "and outcome completeness.")) +
      brepi_theme()
  }, columns = 1, height = 80)
}

cat("\n-- RQ5 -----------------------------------------------------------\n")
reg_path <- file.path(RES, "rq5_regimes/regime_summary.csv")
seasr_path <- file.path(RES, "rq5_regimes/regime_seasonality.csv")
if (have(reg_path) && have(seasr_path)) {
  fig("F11_regime_profiles", {
    rs <- fread(reg_path); sr <- fread(seasr_path)
    sr[, regime := factor(regime)]
    a <- ggplot(sr, aes(x = month, y = share, colour = regime)) +
      geom_line(linewidth = 0.6) +
      scale_x_continuous(breaks = c(1, 4, 7, 10, 12)) +
      scale_colour_manual(values = brepi_palette(uniqueN(sr$regime)), name = "Regime") +
      labs(x = "Month", y = "Share of annual cases", title = "Seasonal profile") +
      brepi_theme()
    b <- ggplot(rs, aes(x = incidence_per_100k_yr, y = cfr_pct,
                        size = units, label = regime)) +
      geom_point(colour = brepi_palette(1), alpha = 0.75) +
      geom_text(size = 2.4, vjust = -1.1, show.legend = FALSE) +
      scale_size_continuous(name = "Municipalities") +
      labs(x = "Incidence per 100,000 person-years", y = "Case fatality (%)",
           title = "Burden and lethality",
           caption = wrap_caption("Regimes with near-identical incidence differ",
                           "six-fold in case fatality.")) +
      brepi_theme()
    a | b
  }, columns = 2, height = 90)
}

cat("\n-- RQ2 -----------------------------------------------------------\n")
es_path <- file.path(RES, "rq2_flood_disasters/primary_event_study.csv")
if (have(es_path)) {
  fig("F12_flood_event_study", {
    es <- fread(es_path)
    plot_event_study(list(tidy = es), anticipation = 1L)
  }, columns = 1, height = 85)
} else {
  cat("  SKIP  F12_flood_event_study              RQ2 not yet complete\n")
  skip <- c(skip, "F12 -- RQ2 event study not yet on disk")
}
hd_path <- file.path(RES, "rq2_flood_disasters/primary_honest_did.csv")
if (have(hd_path)) {
  fig("F13_honest_did", plot_honest_did(fread(hd_path)), columns = 1, height = 80)
}

cat("\n==================================================================\n")
cat(sprintf("%d figure(s) written to %s\n", length(made), FIG))
if (length(skip)) {
  cat(sprintf("%d skipped:\n", length(skip)))
  for (s in skip) cat("   ", s, "\n")
}
writeLines(c(sprintf("made: %s", paste(made, collapse = ", ")),
             sprintf("skipped: %s", paste(skip, collapse = " | "))),
           file.path(FIG, "_manifest.txt"))
