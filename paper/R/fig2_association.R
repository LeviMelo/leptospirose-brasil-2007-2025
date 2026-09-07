# Figure 2 -- the primary association, shown continuously, and the two checks
# that discriminate a denominator effect from a real severity difference.
#
# Panel (a) is the figure the first draft never made: a scatter of the
# hospitalisation proportion H against reported case fatality, one point per
# health region, so the "opposite corners" claim in the text is something a
# reader can actually see rather than infer from a categorical map. No
# quintile compression here -- every included region is a point.
#
# Panel (b) is the construct-validity check: if narrow ascertainment loses
# mild cases preferentially, severe-case incidence should fall far less across
# H than non-severe incidence. Two lines, one log axis, no compression.
#
# Panel (c) is the triangulation across three independent systems -- SINAN
# notified incidence, SIH hospital admissions, SIM population mortality -- at
# quintiles of H. The three series live on genuinely different scales (per
# 100,000 vs per 1,000,000 person-years), and review was explicit that a dual
# y-axis is not acceptable. Each series is therefore indexed to its own Q1
# value (Q1 = 100) so the three trajectories share one axis without implying a
# false common unit; the true units and values are given in the notes and in
# `data/results/triangulation/three_systems_by_stratum.csv` so nothing is
# hidden behind the index.

source("paper/R/theme_ress.R")
suppressPackageStartupMessages(library(arrow))

RES <- "data/results"; OUT <- "paper/figures"

reg <- as.data.table(read_parquet(file.path(RES, "analysis_panel/region_totals.parquet")))
reg <- reg[hosp_known >= 10]
reg[, H_pct := 100 * H]; reg[, cfr_pct := 100 * cfr]

# ------------------------------------------------------------------ panel a --
fit <- glm(cbind(deaths, outcome_known - deaths) ~ H, data = reg, family = binomial())
grid <- data.table(H = seq(min(reg$H), max(reg$H), length.out = 200))
pr <- predict(fit, newdata = grid, se.fit = TRUE)
grid[, `:=`(fit = plogis(pr$fit), lo = plogis(pr$fit - 1.96 * pr$se.fit),
            hi = plogis(pr$fit + 1.96 * pr$se.fit), H_pct = 100 * H)]

# Regions with zero deaths sit exactly on y = 0 and are part of the evidence --
# they are the low-H end of the gradient. `expand` therefore lifts the panel
# floor a little below zero so a marker centred on zero is drawn whole rather
# than half-clipped by the axis line, and `coord_cartesian` (not scale limits)
# holds the x window so a region at H = 100% is not dropped by the scale.
# The size legend sits over the upper-left quadrant, which is empty: no region
# combines low H with high letalidade -- that emptiness is the finding, and the
# legend is placed there precisely because there is nothing to obscure.
pa <- ggplot() +
  geom_ribbon(data = grid, aes(H_pct, ymin = 100 * lo, ymax = 100 * hi),
              fill = AZUL, alpha = 0.15) +
  geom_point(data = reg, aes(H_pct, cfr_pct, size = cases), colour = AZUL,
             alpha = 0.55, stroke = 0) +
  geom_line(data = grid, aes(H_pct, 100 * fit), colour = TINTA, linewidth = 0.6) +
  scale_size_area(name = "Casos", max_size = 5.2, breaks = c(50, 500, 2000)) +
  scale_x_continuous("Proporção de casos internados (%)",
                     breaks = seq(0, 100, 25),
                     expand = expansion(mult = 0.03)) +
  scale_y_continuous("Letalidade (%)",
                     expand = expansion(mult = c(0.055, 0.05))) +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, NA)) +
  tema_ress() +
  theme(legend.position = "inside", legend.position.inside = c(0.15, 0.83),
        legend.background = element_blank(),
        legend.key.size = unit(3, "mm"), legend.title = element_text(size = 6.5))

# ------------------------------------------------------------------ panel b --
sev <- fread(file.path(RES, "construct_validity/severity_incidence_by_quintile.csv"))
sev <- sev[subset == "all confirmed cases" &
          severity_definition == "jaundice OR renal OR haemorrhage (primary)"]
sv <- rbind(
  sev[, .(q, tipo = "Grave", inc = severe_per_100k, lo = severe_lo, hi = severe_hi)],
  sev[, .(q, tipo = "Não grave", inc = nonsevere_per_100k, lo = nonsevere_lo, hi = nonsevere_hi)]
)
sv[, tipo := factor(tipo, levels = c("Não grave", "Grave"))]
r_ns <- sev$nonsevere_per_100k[1] / sev$nonsevere_per_100k[5]
r_sv <- sev$severe_per_100k[1] / sev$severe_per_100k[5]

pb <- ggplot(sv, aes(factor(q), inc, colour = tipo, group = tipo)) +
  geom_line(linewidth = 0.55) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.1, linewidth = 0.3) +
  geom_point(size = 1.5) +
  # Each ratio is written at the right-hand *end of its own line*, vertically
  # centred on that line's Q5 value, so the label's position is the label's
  # referent. The previous placement floated both callouts inside the panel:
  # the green one landed near the floor of the axis, well below the green
  # series, close enough to the gold line's continuation to be misread. Room
  # for the callouts comes from expanding the discrete axis to the right, not
  # from moving the labels inward. Two lines rather than one because a 55 mm
  # panel does not have 15 mm of margin to spare.
  annotate("text", x = 5.18, y = sev$nonsevere_per_100k[5],
           label = paste0("queda de\n", num_br(r_ns, 1), "x"),
           size = 2.1, colour = VERDE, fontface = "bold",
           hjust = 0, vjust = 0.5, lineheight = 0.95) +
  annotate("text", x = 5.18, y = sev$severe_per_100k[5],
           label = paste0("queda de\n", num_br(r_sv, 1), "x"),
           size = 2.1, colour = "#B8860B", fontface = "bold",
           hjust = 0, vjust = 0.5, lineheight = 0.95) +
  scale_colour_manual(NULL, values = c("Não grave" = VERDE, "Grave" = "#B8860B"),
                      guide = guide_legend(nrow = 2)) +
  scale_x_discrete("Quintil da proporção de casos internados\n(I = detecção mais ampla, V = mais estreita)",
                   labels = c("I", "II", "III", "IV", "V"),
                   expand = expansion(add = c(0.35, 1.75))) +
  # Extra floor space so the second line of the "Não grave" callout, which is
  # centred on the lowest point in the panel, is not clipped by the panel edge.
  scale_y_continuous("Incidência (/100.000 pessoas-ano)", trans = "log10",
                     labels = function(z) num_br(z, 2),
                     expand = expansion(mult = c(0.10, 0.06))) +
  tema_ress() +
  theme(legend.position = "bottom", legend.margin = margin(t = -4),
        panel.grid.major.x = element_blank())

# ------------------------------------------------------------------ panel c --
tri <- fread(file.path(RES, "triangulation/three_systems_by_stratum.csv"))
setorder(tri, bin)
idx <- function(x) 100 * x / x[1]
td <- rbind(
  tri[, .(bin, serie = "Notificação", valor = idx(sinan_incidence_per_100k))],
  tri[, .(bin, serie = "Internação hospitalar", valor = idx(sih_admission_per_100k))],
  tri[, .(bin, serie = "Mortalidade populacional", valor = idx(sim_mortality_per_1m))]
)
td[, serie := factor(serie, levels = c("Notificação", "Internação hospitalar",
                                       "Mortalidade populacional"))]

pc <- ggplot(td, aes(factor(bin), valor, colour = serie, group = serie)) +
  geom_hline(yintercept = 100, linewidth = 0.3, colour = "grey55", linetype = "22") +
  geom_line(linewidth = 0.55) + geom_point(size = 1.5) +
  scale_colour_manual(NULL, values = c("Notificação" = AZUL,
                                       "Internação hospitalar" = "#7B3F82",
                                       "Mortalidade populacional" = VERMELHO),
                      guide = guide_legend(nrow = 3)) +
  scale_x_discrete("Quintil da proporção de casos internados\n(I = detecção mais ampla, V = mais estreita)",
                   labels = c("I", "II", "III", "IV", "V")) +
  # Axis choice, stated because it is contestable. The panel's claim is that
  # SIM carries no downward gradient while SINAN and SIH fall to 19 and 38.
  # Two constraints bound the axis: it must start at 0, or a flat line drawn
  # high on a truncated axis exaggerates its flatness exactly as a truncated
  # axis exaggerates a trend; and it must reach 147, the SIM value at Q4,
  # because clipping the one excursion in the series the panel calls flat
  # would be arguing by cropping. c(0, 159) is therefore already the tightest
  # honest window -- there is no range left to tighten. The 100 reference line
  # does the work a tighter axis would have done: SIM returns to it at Q5
  # (108), and the reader can see the two falling series leave it and not come
  # back. Adding y = 50 and y = 150 gridlines is the only change here, so the
  # eye can judge vertical distance from the reference without measuring.
  scale_y_continuous("Índice (Quintil I = 100)",
                     limits = c(0, max(td$valor) * 1.08),
                     breaks = seq(0, 150, 50)) +
  tema_ress() +
  theme(legend.position = "bottom", legend.margin = margin(t = -4),
        panel.grid.major.x = element_blank())

fig <- tags_minusculos((pa | pb | pc) + plot_layout(widths = c(1.15, 1, 1)))
salvar(fig, file.path(OUT, "figura2_associacao.png"), columns = 2, height_mm = 82)

# H is on the 0-1 scale here, so ten percentage points is 0.10, not 10.
cat(sprintf("\n(a) OR por 10pp de H (GLM simples nao ajustado): %.3f\n",
            exp(0.10 * coef(fit)["H"])))
cat(sprintf("(b) grave %.2fx ; nao grave %.2fx\n", r_sv, r_ns))
cat(sprintf("(c) SINAN Q1->Q5 %.2f -> %.2f (/100k) | SIH %.2f -> %.2f (/100k) | SIM %.2f -> %.2f (/1M)\n",
            tri$sinan_incidence_per_100k[1], tri$sinan_incidence_per_100k[5],
            tri$sih_admission_per_100k[1], tri$sih_admission_per_100k[5],
            tri$sim_mortality_per_1m[1], tri$sim_mortality_per_1m[5]))
