# Figure: ascertainment depth explains the case-fatality geography.
#
# Serves Links 3b and 3c of docs/RESEARCH_JOURNAL.md, and nothing else. The
# three panels are not independent and cannot be read in any order:
#
#   (a) the cross-sectional gradient -- case fatality rises and incidence falls
#       across quintiles of hospitalisation share;
#   (b) the answer to "that is just geography" -- within macro-regions the
#       hospitalisation-share association survives and the sanitation
#       association does not;
#   (c) the answer to "those territories just differ" -- the same movement
#       inside ONE state across a surveillance shock and back.
#
# Remove (b) and (a) is a regional confound. Remove (c) and (a) is cross-
# sectional only. Remove (a) and the other two explain nothing.
#
# No title inside the artwork: the journal puts it at the head of the legend.

source("paper/R/theme_ress.R")
suppressPackageStartupMessages(library(arrow))

RES <- "data/results"
OUT <- "paper/figures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- panel a ---
grad <- as.data.table(read_parquet(file.path(RES, "ascertainment_depth/gradient_by_quintile.parquet")))
grad <- grad[banded_by == "hospitalisation share"][order(quintile)]
stopifnot(nrow(grad) == 5L)

# Incidence is plotted on a rescaled secondary axis. The scale factor is derived
# from the data rather than hard-coded so the two series stay comparable if the
# numbers move.
SF <- max(grad$cfr_hi) / max(grad$incidence_per_100k)
grad[, `:=`(x = factor(quintile), hs_lab = pct_br(100 * band_median, 0))]

pa <- ggplot(grad, aes(x)) +
  geom_col(aes(y = cfr_pct), fill = AZUL, width = 0.62) +
  geom_errorbar(aes(ymin = cfr_lo, ymax = cfr_hi), width = 0.15,
                linewidth = 0.35, colour = TINTA) +
  geom_line(aes(y = incidence_per_100k * SF, group = 1), colour = VERMELHO,
            linewidth = 0.55, linetype = "22") +
  geom_point(aes(y = incidence_per_100k * SF), colour = VERMELHO, size = 1.5) +
  geom_text(aes(y = cfr_hi, label = pct_br(cfr_pct, 1)), vjust = -0.75,
            size = 2.1, colour = TINTA) +
  scale_x_discrete(
    "Quintil de internação entre casos confirmados\n(mediana do quintil)",
    labels = grad$hs_lab
  ) +
  scale_y_continuous(
    "Letalidade (%)", limits = c(0, 20), expand = c(0, 0),
    sec.axis = sec_axis(~ . / SF, name = "Incidência (por 100.000 pessoas-ano)")
  ) +
  tema_ress() +
  theme(panel.grid.major.x = element_blank(),
        axis.title.y.right = element_text(colour = VERMELHO),
        axis.text.y.right  = element_text(colour = VERMELHO))

# ---------------------------------------------------------------- panel b ---
rep_json <- jsonlite::fromJSON(file.path(RES, "ascertainment_depth/ascertainment_depth_report.json"))
wr <- rep_json$within_region
regs <- names(wr)
mk <- function(reg, key) {
  v <- wr[[reg]][[key]]
  if (is.null(v)) NA_real_ else as.numeric(v[1])
}
b <- rbindlist(lapply(regs, function(r) data.table(
  region = r,
  n = as.integer(wr[[r]]$n),
  hosp = mk(r, "hosp_share_vs_cfr"),
  sewer = mk(r, "sewer_vs_cfr")
)))
b <- b[!is.na(hosp)]
bl <- melt(b, id.vars = c("region", "n"), variable.name = "indice", value.name = "rho")
bl[, indice := factor(fifelse(indice == "hosp", "Internação", "Esgotamento"),
                      levels = c("Internação", "Esgotamento"))]
bl[, region := factor(region, levels = rev(ORD_REGIAO))]
bl[, rotulo := paste0(region, " (n=", n, ")")]
bl[, rotulo := factor(rotulo, levels = unique(rotulo[order(match(region, rev(ORD_REGIAO)))]))]

# A rank correlation over nine units carries no weight, and drawing it at full
# strength beside n = 75 would invite the reader to average them. It is shown --
# hiding it would be selection -- but dimmed and labelled.
N_MIN <- 15L
bl[, frouxo := n < N_MIN]
nota <- bl[frouxo == TRUE, .(rho = max(rho, na.rm = TRUE)), by = rotulo]

pb <- ggplot(bl, aes(rho, rotulo, fill = indice, alpha = frouxo)) +
  geom_vline(xintercept = 0, colour = TINTA2, linewidth = 0.3) +
  geom_col(position = position_dodge(width = 0.68), width = 0.62) +
  scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.32), guide = "none") +
  scale_fill_manual(NULL, values = c("Internação" = AZUL, "Esgotamento" = "#C8A15A")) +
  scale_x_continuous("Correlação de Spearman com a letalidade",
                     limits = c(-0.85, 0.9), breaks = seq(-0.6, 0.8, 0.2),
                     labels = function(z) num_br(z, 1)) +
  labs(y = NULL) +
  tema_ress() +
  theme(panel.grid.major.y = element_blank(),
        legend.position = "bottom", legend.margin = margin(t = -3))

if (nrow(nota)) {
  pb <- pb + geom_text(data = nota, aes(x = 0.86, y = rotulo, label = "n insuficiente"),
                       inherit.aes = FALSE, hjust = 1, size = 1.9, colour = TINTA2,
                       fontface = "italic")
}

# ---------------------------------------------------------------- panel c ---
rs <- as.data.table(read_parquet(file.path(RES, "rs2024/rs_annual_series.parquet")))
rs <- rs[year >= 2019][order(year)]
cl <- rbind(
  rs[, .(year, valor = hosp_share_pct, lo = hosp_share_lo, hi = hosp_share_hi,
         serie = "Internação (%)")],
  rs[, .(year, valor = lab_share_pct, lo = lab_share_lo, hi = lab_share_hi,
         serie = "Confirmação laboratorial (%)")]
)
cl[, serie := factor(serie, levels = c("Internação (%)", "Confirmação laboratorial (%)"))]

pc <- ggplot(cl, aes(year, valor, colour = serie, fill = serie)) +
  annotate("rect", xmin = 2023.5, xmax = 2024.5, ymin = 0, ymax = 100,
           fill = "#F0D9D5", alpha = 0.55) +
  annotate("text", x = 2024, y = 97, label = "enchente", size = 2.1,
           colour = VERMELHO, fontface = "bold") +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.16, colour = NA) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.3) +
  scale_colour_manual(NULL, values = c(AZUL, VERDE)) +
  scale_fill_manual(NULL, values = c(AZUL, VERDE)) +
  scale_x_continuous(NULL, breaks = 2019:2025) +
  scale_y_continuous("Proporção dos casos confirmados (%)", limits = c(0, 100),
                     expand = c(0, 0)) +
  tema_ress() +
  theme(legend.position = "bottom", legend.margin = margin(t = -3))

fig <- (pa | pb) / pc + plot_layout(heights = c(1, 0.92))
fig <- tags_minusculos(fig)
salvar(fig, file.path(OUT, "fig_ascertainment_depth.png"), columns = 2, height_mm = 150)

cat("\nlegend numbers (verify against the journal before writing the caption):\n")
cat(sprintf("  a: CFR %s -> %s across quintiles; incidence %s -> %s\n",
            pct_br(grad$cfr_pct[1], 2), pct_br(grad$cfr_pct[5], 2),
            num_br(grad$incidence_per_100k[1], 2), num_br(grad$incidence_per_100k[5], 2)))
cat(sprintf("  b: regions where hospitalisation beats sanitation: %d of %d\n",
            sum(b$hosp > b$sewer), nrow(b)))
w <- rs[year %in% c(2023, 2024, 2025)]
cat(sprintf("  c: hosp share %s -> %s -> %s ; lab %s -> %s -> %s\n",
            pct_br(w$hosp_share_pct[1]), pct_br(w$hosp_share_pct[2]), pct_br(w$hosp_share_pct[3]),
            pct_br(w$lab_share_pct[1]), pct_br(w$lab_share_pct[2]), pct_br(w$lab_share_pct[3])))
