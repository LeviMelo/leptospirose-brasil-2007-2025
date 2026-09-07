# Figure 3 -- the mechanism inside one state, before and after a surveillance shock.
#
# Every other result in the paper is cross-sectional, and a cross-sectional
# gradient always invites the reply that the territories simply differ. The
# late-April/May 2024 flood in Rio Grande do Sul supplies the perturbation:
# mass case-finding in one state over a few months, holding the population, the
# health system, the pathogen and the notification form fixed.
#
# The first version of this figure was a single annual line chart. Annual
# resolution is exactly the resolution at which this event disappears -- a
# three-month surge averaged over twelve months looks like a mildly unusual
# year. The three panels here are the event, its two surveillance markers, and
# the check that the movement is specific to Rio Grande do Sul.
#
#   (a) monthly confirmed cases, on a LINEAR count axis. A log or square-root
#       axis would make the pre-event seasonality easier to read, but the whole
#       claim of the panel is one of magnitude -- 598 confirmed cases in May
#       2024 against a pre-event monthly maximum of 114 -- and a compressive
#       transform is precisely what would soften it. The pre-event level is
#       instead carried by a labelled reference line at that maximum, so the
#       baseline is legible as a number even where the bars are small.
#
#   (b) H and the laboratory-confirmed proportion, aggregated to QUARTERS, not
#       months. At monthly resolution the denominators outside the surge run
#       4-50 confirmed cases, so a single month's Clopper-Pearson interval on H
#       is around 40 percentage points wide -- wider than the ~35-point signal
#       the panel exists to show. Quarterly totals put the routine denominators
#       at 30-150 and roughly halve the intervals, without smoothing across the
#       event: the flood falls inside one quarter (2024 Q2) and the recovery
#       inside the next. Numerators and denominators are summed before the
#       ratio is formed; the intervals are exact (Clopper-Pearson) on the
#       quarterly totals.
#
#   (c) the same contrast run across states: change in H between 2024 and the
#       2019-2023 baseline, for every UF with at least 150 confirmed cases in
#       2024. If the fall in H were a national artefact of 2024 -- a change to
#       the form, to the case definition, to coding practice -- it would appear
#       in the other states too. It does not.
#
# No dual y-axis anywhere; each panel carries one unit.

source("paper/R/theme_ress.R")

RES <- "data/results"; OUT <- "paper/figures"

# Same output as the house helper; formatC warns that big.mark and decimal.mark
# are both "." for integers, which floods stdout for no reason here.
int_br <- function(x) suppressWarnings(
  formatC(as.integer(round(x)), big.mark = ".", format = "d"))

# Typographic minus, so axis ticks match the "−30,4 pp" labels.
menos <- function(x) gsub("-", "−", format(x, trim = TRUE))

mes <- fread(file.path(RES, "rs2024_event/rs_monthly_series.csv"))
uf  <- fread(file.path(RES, "rs2024_event/state_H_2024_vs_pre.csv"))

mes[, data := as.Date(month_start)]
setorder(mes, data)
stopifnot(nrow(mes) == 84L, !anyNA(mes$data))

# The flood window used throughout the paper: symptom onset May-July 2024.
FL_INI <- as.Date("2024-05-01"); FL_FIM <- as.Date("2024-08-01")
X_LIM  <- c(as.Date("2018-11-15"), as.Date("2026-02-15"))

# ---------------------------------------------------------------- helpers ---
# Clopper-Pearson, estimate first, matching the house convention.
cp <- function(x, n) {
  x <- as.numeric(x); n <- as.numeric(n)
  ci <- mapply(function(a, b) {
    if (is.na(a) || is.na(b) || b == 0) return(c(NA_real_, NA_real_))
    binom.test(a, b)$conf.int[1:2]
  }, x, n)
  data.table(est = x / n, lo = ci[1, ], hi = ci[2, ])
}

# Wilson score limits, used only inside the Newcombe hybrid below.
wilson <- function(x, n, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2); p <- x / n; d <- 1 + z^2 / n
  h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))
  c((p + z^2 / (2 * n) - h) / d, (p + z^2 / (2 * n) + h) / d)
}

# Newcombe's hybrid score interval for a difference of two independent
# proportions. A difference of binomials has no exact interval of the
# Clopper-Pearson kind; the hybrid score method is the standard alternative and
# holds its coverage far better than a Wald difference at these denominators.
newcombe <- function(x1, n1, x2, n2) {
  p1 <- x1 / n1; p2 <- x2 / n2
  w1 <- wilson(x1, n1); w2 <- wilson(x2, n2); d <- p1 - p2
  c(d - sqrt((p1 - w1[1])^2 + (w2[2] - p2)^2),
    d + sqrt((w1[2] - p1)^2 + (p2 - w2[1])^2))
}

# ================================================================= panel a ===
maxpre <- mes[data < FL_INI, max(confirmed)]
pico   <- mes[data == as.Date("2024-05-01")]

mes[, dentro := data >= FL_INI & data < FL_FIM]

pa <- ggplot(mes, aes(data + 14, confirmed)) +
  annotate("rect", xmin = FL_INI, xmax = FL_FIM, ymin = 0, ymax = Inf,
           fill = VERMELHO, alpha = 0.10) +
  geom_hline(yintercept = maxpre, linetype = "22", linewidth = 0.3,
             colour = TINTA2) +
  geom_col(aes(fill = dentro), width = 24) +
  annotate("text", x = as.Date("2019-02-01"), y = maxpre + 26,
           label = paste0("máximo mensal pré-evento: ", int_br(maxpre)),
           size = 2.0, colour = TINTA2, hjust = 0) +
  annotate("text", x = pico$data + 14, y = pico$confirmed + 30,
           label = int_br(pico$confirmed), size = 2.3, colour = VERMELHO,
           fontface = "bold") +
  annotate("text", x = FL_FIM + 40, y = 520, label = "enchente\nmai–jul 2024",
           size = 2.1, colour = VERMELHO, hjust = 0, lineheight = 0.95) +
  scale_fill_manual(values = c(`FALSE` = AZUL, `TRUE` = VERMELHO), guide = "none") +
  scale_x_date(NULL, limits = X_LIM, date_breaks = "1 year", date_labels = "%Y",
               expand = c(0, 0)) +
  scale_y_continuous("Casos confirmados no mês", limits = c(0, 660),
                     breaks = seq(0, 600, 200), expand = c(0, 0)) +
  tema_ress() +
  theme(axis.text.x = element_blank(), panel.grid.major.x = element_blank(),
        plot.margin = margin(2, 3, 0, 2))

# ================================================================= panel b ===
# Quarterly totals; numerators and denominators summed before the ratio.
mes[, trim := as.Date(sprintf("%d-%02d-01", onset_year,
                              3L * ((onset_month - 1L) %/% 3L) + 1L))]
tri <- mes[, .(hospitalised = sum(hospitalised), hosp_known = sum(hosp_known),
               lab_confirmed = sum(lab_confirmed), crit_known = sum(crit_known),
               confirmed = sum(confirmed)), by = trim]
setorder(tri, trim)

tri <- rbind(
  cbind(tri[, .(trim, n = hosp_known)], cp(tri$hospitalised, tri$hosp_known),
        marcador = "Proporção de casos internados"),
  cbind(tri[, .(trim, n = crit_known)], cp(tri$lab_confirmed, tri$crit_known),
        marcador = "Confirmação laboratorial")
)
tri[, marcador := factor(marcador, levels = c("Confirmação laboratorial",
                                              "Proporção de casos internados"))]
tri[, meio := trim + 45]

PAL_MARC <- c("Confirmação laboratorial" = VERDE,
              "Proporção de casos internados" = AZUL)

pb <- ggplot(tri, aes(meio, 100 * est, colour = marcador, fill = marcador)) +
  annotate("rect", xmin = FL_INI, xmax = FL_FIM, ymin = 0, ymax = 100,
           fill = VERMELHO, alpha = 0.10) +
  geom_ribbon(aes(ymin = 100 * lo, ymax = 100 * hi), alpha = 0.16, colour = NA) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 0.9, stroke = 0) +
  # The resolution and the interval method ride in the legend title: as a free
  # annotation they had to sit somewhere, and every free corner of this panel
  # is crossed either by a ribbon or by the flood shading.
  scale_colour_manual("Trimestres, IC95%", values = PAL_MARC,
                      guide = guide_legend(nrow = 2)) +
  scale_fill_manual("Trimestres, IC95%", values = PAL_MARC,
                    guide = guide_legend(nrow = 2)) +
  scale_x_date(NULL, limits = X_LIM, date_breaks = "1 year", date_labels = "%Y",
               expand = c(0, 0)) +
  # Two lines: on one line this title is taller than the panel, and its top
  # ends up level with the "b" tag.
  scale_y_continuous("Proporção dos casos\nconfirmados (%)",
                     limits = c(0, 100), breaks = seq(0, 100, 25),
                     expand = c(0, 0)) +
  tema_ress() +
  theme(panel.grid.major.x = element_blank(),
        legend.position = c(0.015, 0.03), legend.justification = c(0, 0),
        legend.background = element_rect(fill = alpha("white", 0.75), colour = NA),
        legend.margin = margin(1, 2, 1, 1), legend.key.height = unit(2.6, "mm"),
        legend.text = element_text(size = 6.2),
        legend.title = element_text(size = 6.2, face = "plain", colour = TINTA2),
        # Room for the "b" tag above the rotated axis title, which otherwise
        # reaches the top of the plot area and sits shoulder to shoulder with it.
        plot.tag.position = c(0, 1), plot.margin = margin(8, 3, 2, 2))

# ================================================================= panel c ===
uf <- uf[cases_2024 >= 150]
stopifnot(nrow(uf) == 7L)
uf[, c("dif_lo", "dif_hi") := as.data.table(t(mapply(
  newcombe, hosp_2024, hosp_known_2024, hosp_pre, hosp_known_pre)))]
uf[, `:=`(dif = 100 * (H_2024 - H_pre), dif_lo = 100 * dif_lo, dif_hi = 100 * dif_hi)]
setorder(uf, -dif)
uf[, uf_f := factor(uf_abbr, levels = uf_abbr)]   # most negative last => top
uf[, destaque := uf_abbr == "RS"]
uf[, rotulo := paste0(ifelse(dif > 0, "+", "−"), num_br(abs(dif), 1), " pp")]

# The value labels sit in a fixed right-hand column rather than floating at
# each bar's end: at these magnitudes a label tracking the bar would sit almost
# on the zero line for six of the seven states and read as noise.
pc <- ggplot(uf, aes(dif, uf_f, fill = destaque, colour = destaque)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = TINTA2) +
  geom_col(width = 0.62) +
  geom_errorbar(aes(xmin = dif_lo, xmax = dif_hi), orientation = "y",
                width = 0.22, linewidth = 0.32, colour = TINTA) +
  geom_text(aes(x = 19, label = rotulo), hjust = 1, size = 2.05, colour = TINTA) +
  scale_fill_manual(values = c(`FALSE` = "#A8AFB8", `TRUE` = VERMELHO), guide = "none") +
  scale_colour_manual(values = c(`FALSE` = "#A8AFB8", `TRUE` = VERMELHO), guide = "none") +
  scale_x_continuous("Variação da proporção de casos internados em 2024\nfrente a 2019–2023 (pontos percentuais)",
                     limits = c(-35, 19.5), breaks = seq(-30, 10, 10),
                     labels = menos, expand = c(0, 0)) +
  scale_y_discrete(NULL, expand = expansion(add = 0.62)) +
  tema_ress() +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 7, colour = TINTA, face = "bold"),
        plot.margin = margin(4, 2, 2, 4))

# ================================================================== figura ===
fig <- pa + pb + pc + plot_layout(design = "AAAACCC\nBBBBCCC")
salvar(tags_minusculos(fig), file.path(OUT, "figura3_rs2024.png"),
       columns = 2, height_mm = 95)

# =================================================================== dados ===
rot_trim <- function(d) sprintf("%s-T%d", format(d, "%Y"),
                                (as.integer(format(d, "%m")) - 1L) %/% 3L + 1L)
q <- function(qtr, marc) {
  r <- tri[trim == as.Date(qtr) & grepl(marc, marcador)]
  sprintf("%s (%s–%s; n=%d)", pct_br(100 * r$est), num_br(100 * r$lo, 1),
          num_br(100 * r$hi, 1), r$n)
}

cat("\n================ FIGURA 3 — números para a legenda ================\n")
cat(sprintf("Série mensal: %s a %s (mês de início dos sintomas), %d meses.\n",
            format(min(mes$data), "%m/%Y"), format(max(mes$data), "%m/%Y"), nrow(mes)))

cat("\n--- painel (a): casos confirmados por mês, eixo LINEAR ---\n")
cat(sprintf("  mai/2024 = %s casos confirmados (notificados %s; incidência %s/100 mil pessoas-ano)\n",
            int_br(pico$confirmed), int_br(pico$notified),
            num_br(pico$incidence_per_100k_py, 1)))
cat(sprintf("  jun/2024 = %s ; jul/2024 = %s\n",
            int_br(mes[data == as.Date("2024-06-01"), confirmed]),
            int_br(mes[data == as.Date("2024-07-01"), confirmed])))
cat(sprintf("  máximo mensal pré-evento (jan/2019–abr/2024) = %s ; razão mai/2024 = %s×\n",
            int_br(maxpre), num_br(pico$confirmed / maxpre, 1)))
cat(sprintf("  mediana mensal pré-evento = %s casos ; razão mai/2024 = %s×\n",
            int_br(mes[data < FL_INI, median(confirmed)]),
            num_br(pico$confirmed / mes[data < FL_INI, median(confirmed)], 1)))
cat(sprintf("  janela sombreada: mai–jul/2024, %s casos confirmados (%s dos casos de 2024)\n",
            int_br(mes[dentro == TRUE, sum(confirmed)]),
            pct_br(100 * mes[dentro == TRUE, sum(confirmed)] /
                     mes[onset_year == 2024, sum(confirmed)])))

cat("\n--- painel (b): marcadores trimestrais, IC 95% Clopper-Pearson ---\n")
cat("  agregação trimestral (não mensal): ver comentário no topo do script.\n")
for (tq in c("2023-10-01", "2024-01-01", "2024-04-01", "2024-07-01",
             "2024-10-01", "2025-01-01")) {
  cat(sprintf("  %s  H %s | laboratorial %s\n",
              rot_trim(as.Date(tq)), q(tq, "^H"), q(tq, "Confirma")))
}
pre_h <- mes[onset_year %in% 2019:2023, .(x = sum(hospitalised), n = sum(hosp_known))]
pre_l <- mes[onset_year %in% 2019:2023, .(x = sum(lab_confirmed), n = sum(crit_known))]
fl_h  <- mes[dentro == TRUE, .(x = sum(hospitalised), n = sum(hosp_known))]
fl_l  <- mes[dentro == TRUE, .(x = sum(lab_confirmed), n = sum(crit_known))]
po_h  <- mes[onset_year == 2025, .(x = sum(hospitalised), n = sum(hosp_known))]
po_l  <- mes[onset_year == 2025, .(x = sum(lab_confirmed), n = sum(crit_known))]
fmt <- function(d) sprintf("%s (%s–%s; n=%s)", pct_br(100 * d$est),
                           num_br(100 * d$lo, 1), num_br(100 * d$hi, 1), int_br(d$n))
for (r in list(list("2019–2023", pre_h, pre_l), list("mai–jul/2024", fl_h, fl_l),
               list("2025", po_h, po_l))) {
  cat(sprintf("  %-13s H %s | laboratorial %s\n", r[[1]],
              fmt(cbind(cp(r[[2]]$x, r[[2]]$n), n = r[[2]]$n)),
              fmt(cbind(cp(r[[3]]$x, r[[3]]$n), n = r[[3]]$n))))
}
# The reversion is real but incomplete, and the figure shows that; the caption
# should not say "returns to baseline" when the 2025 interval clears it.
for (r in list(list("H", fl_h, pre_h, po_h), list("laboratorial", fl_l, pre_l, po_l))) {
  d1 <- newcombe(r[[2]]$x, r[[2]]$n, r[[3]]$x, r[[3]]$n)
  d2 <- newcombe(r[[4]]$x, r[[4]]$n, r[[3]]$x, r[[3]]$n)
  cat(sprintf("  %-13s mai–jul/2024 − 2019–2023 = %s pp (IC95%% %s a %s) ; 2025 − 2019–2023 = %s pp (IC95%% %s a %s)\n",
              r[[1]], num_br(100 * (r[[2]]$x / r[[2]]$n - r[[3]]$x / r[[3]]$n), 1),
              num_br(100 * d1[1], 1), num_br(100 * d1[2], 1),
              num_br(100 * (r[[4]]$x / r[[4]]$n - r[[3]]$x / r[[3]]$n), 1),
              num_br(100 * d2[1], 1), num_br(100 * d2[2], 1)))
}

cat("\n--- painel (c): variação de H, 2024 frente a 2019–2023 ---\n")
cat(sprintf("  %d UF com ≥150 casos confirmados em 2024\n", nrow(uf)))
for (i in seq_len(nrow(uf))) {
  r <- uf[i]
  cat(sprintf("  %-3s casos 2024 %s | H 2024 %s | H 2019–2023 %s | Δ %s pp (IC95%% %s a %s) | RC %s (%s–%s)\n",
              r$uf_abbr, int_br(r$cases_2024), pct_br(100 * r$H_2024),
              pct_br(100 * r$H_pre), num_br(r$dif, 1), num_br(r$dif_lo, 1),
              num_br(r$dif_hi, 1), num_br(r$or_hosp_2024_vs_pre, 2),
              num_br(r$or_lo, 2), num_br(r$or_hi, 2)))
}
cat(sprintf("  maior queda: %s (%s pp); segunda maior: %s (%s pp); razão %s×\n",
            uf[.N, uf_abbr], num_br(uf[.N, dif], 1), uf[.N - 1, uf_abbr],
            num_br(uf[.N - 1, dif], 1), num_br(uf[.N, dif] / uf[.N - 1, dif], 1)))
cat("==================================================================\n")
