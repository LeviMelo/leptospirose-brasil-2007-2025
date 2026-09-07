# Figure 1 -- the geography that motivates the question, shown quantitatively.
#
# The earlier version was a single 3x3 bivariate choropleth: a reader had to
# decode a categorical colour matrix to recover two continuous quantities, and
# it showed neither incidence nor case fatality in a form that could be read
# off the page, and it omitted the exposure (H) entirely. Review was explicit
# that this figure was "being asked to carry an argument it cannot visually
# support".
#
# Three panels, one geometry, one visual grammar: sequential, colourblind-safe
# palettes, quantitative legends in the quantity's own units, low-count regions
# shaded grey rather than coloured (a grey cell answering a question the map
# is not equipped to answer is honest; forcing it into a tercile is not).
#
# The dense South/Southeast coastal corridor is illegible at national extent in
# any of the three panels, so a single shared inset window -- chosen by case
# mass with brepi's own zoom_windows() rather than by eye -- is cropped from
# all three and stacked below them. One window serves all three panels because
# the point of the inset is comparison across panels in the same place, not
# three unrelated magnifications.

source("paper/R/theme_ress.R")
suppressPackageStartupMessages({library(sf); library(arrow); library(patchwork); library(ggspatial)})
source("R/00_io.R")             # defines .need(), used by 13_figure_fusion.R
source("R/13_figure_fusion.R")

RES <- "data/results"; OUT <- "paper/figures"

reg <- as.data.table(read_parquet(file.path(RES, "analysis_panel/region_totals.parquet")))
geo <- st_read(file.path(OUT, "geo_health_region.gpkg"), quiet = TRUE)
geo$health_region_code <- as.character(geo$health_region_code)

MIN_CASES <- 10L   # a rate on fewer than 10 cases is not worth colouring

# The "no estimate" grey has to be read as *absence of a value*, not as the
# bottom of any of the three ramps. The earlier "#D9D4C8" sat at roughly the
# same lightness as the pale end of the incidence and letalidade ramps, so a
# grey region and a near-zero region looked alike. This grey is several steps
# darker and fully neutral (R=G=B to within two points), which separates it
# from every colour on all three ramps in hue *and* in greyscale.
CINZA_SEM <- "#B3AFA7"
MARCA     <- "#B02418"   # frame colour, matching crop_to_window()'s default

reg[, estimable := cases >= MIN_CASES]
reg[, cfr_pct := 100 * cfr]
reg[, H_pct := 100 * H]

mp <- merge(geo, reg[, .(health_region_code, cases, incidence_per_100k, cfr_pct,
                         H_pct, estimable)], by = "health_region_code", all.x = TRUE)
mp$estimable[is.na(mp$estimable)] <- FALSE

# The shared inset window: centred on the case mass of the South/Southeast
# corridor, chosen by the same algorithm the module ships rather than by eye.
# s2 rejects centroids of the simplified boundary at a few self-touching
# vertices (the same issue prep_geo.R documents for the national outline);
# planar geometry is exact enough for a window-selection centroid.
old_s2 <- sf_use_s2(); sf_use_s2(FALSE)
ctr <- suppressWarnings(st_centroid(st_geometry(geo)))
sf_use_s2(old_s2)
xy <- st_coordinates(ctr)
w <- zoom_windows(xy[, 1], xy[, 2], weight = reg$cases[match(geo$health_region_code,
                  reg$health_region_code)], k = 1L, span = 0.30)

# Legend geometry is set once, here, rather than per panel: at 180 mm across
# three panels each colourbar has ~60 mm of column to live in, and the earlier
# 9 mm key forced four break labels into a strip narrower than the labels
# themselves. A 34 mm bar leaves each break label its own space, and the white
# tick marks put the label under a mark on the bar rather than under empty
# gradient, so "5,8" cannot be read as belonging to the wrong position.
legenda_barra <- guide_colourbar(theme = theme(
  legend.key.width  = unit(34, "mm"),
  legend.key.height = unit(2.8, "mm"),
  legend.frame      = element_rect(colour = "grey45", linewidth = 0.15),
  legend.ticks      = element_line(colour = "white", linewidth = 0.35),
  legend.ticks.length = unit(2.8, "mm"),   # full-height ticks
  legend.text       = element_text(size = 6.5, margin = margin(t = 0.3, unit = "mm"))
))


# RESS requires maps to carry a north indicator and a cartographic scale.
# ggspatial computes the bar length from the panel's own projection, so the bar
# stays correct if the extent or CRS changes -- unlike a hand-derived
# degrees-per-kilometre constant, which silently goes wrong at a different
# latitude. Drawn on the three national panels only; the insets are
# magnifications of a framed window and carry the frame badge instead.
escala_e_norte <- function(p) {
  p +
    ggspatial::annotation_scale(
      location = "bl", width_hint = 0.30, height = unit(1.2, "mm"),
      text_cex = 0.45, pad_x = unit(2, "mm"), pad_y = unit(2, "mm")) +
    ggspatial::annotation_north_arrow(
      location = "bl", which_north = "true",
      height = unit(6, "mm"), width = unit(5, "mm"),
      pad_x = unit(3, "mm"), pad_y = unit(5, "mm"),
      style = ggspatial::north_arrow_orienteering(text_size = 5))
}

one_panel <- function(fillvar, title, legend_title, palette_fn, digits = 1) {
  ggplot(mp) +
    geom_sf(aes(fill = .data[[fillvar]]), colour = "white", linewidth = 0.04) +
    geom_sf(data = mp[!mp$estimable, ], fill = CINZA_SEM, colour = "white",
            linewidth = 0.04) +
    palette_fn +
    coord_sf(expand = FALSE) +
    labs(title = title, fill = legend_title) +
    guides(fill = legenda_barra) +
    tema_mapa_ress() +
    theme(legend.position = "bottom",
          legend.margin = margin(t = 1, b = 0),
          plot.title = element_text(size = 8, hjust = 0.5, face = "plain"))
}

# A handful of small, sparsely-populated regions post incidence and letalidade
# rates several times the interquartile spread (max incidence 35.3/100k against
# a 95th percentile of 6.0; max letalidade 41.2% against 23.8%). A linear scale
# that reaches those maxima paints every other region a uniform pale colour,
# which defeats the point of a quantitative map. The scale is capped at the
# 98th percentile among estimable regions and values above it are squished to
# the top colour, with the top break labelled "+" so the cap is visible rather
# than silently hiding the outliers.
cap_scale <- function(values, colours, top_break, digits, unit = "") {
  cap <- as.numeric(stats::quantile(values, 0.98, na.rm = TRUE))
  scale_fill_gradientn(
    colours = colours, na.value = CINZA_SEM, limits = c(0, cap),
    oob = scales::squish,
    breaks = seq(0, cap, length.out = 4),
    labels = function(x) {
      lab <- num_br(x, digits)
      lab[length(lab)] <- paste0(lab[length(lab)], "+")
      lab
    })
}
pal_inc <- cap_scale(reg[estimable == TRUE]$incidence_per_100k,
                     c("#F4F2ED", "#9EC4D8", "#2A6C8C"), digits = 1)
pal_cfr <- cap_scale(reg[estimable == TRUE]$cfr_pct,
                     c("#F4F2ED", "#E8A0A8", "#A6243A"), digits = 0)
pal_h   <- scale_fill_gradientn(colours = c("#F4F2ED", "#C9AED6", "#5B2C6F"),
                                na.value = CINZA_SEM, labels = function(x) num_br(x, 0))

# Panel letters are plot tags, not text baked into the title. The earlier
# "a  Incidência" put a plain-weight letter inside a centred 8 pt title, which
# is not what tags_minusculos() produces anywhere else in the paper (bold, 9 pt,
# lowercase, flush left, outside the panel). One tag device for the whole
# figure set; the titles now carry only the quantity.
pA0 <- one_panel("incidence_per_100k", "Incidência (/100.000 pessoas-ano)", NULL, pal_inc)
pB0 <- one_panel("cfr_pct", "Letalidade (%)", NULL, pal_cfr)
pC0 <- one_panel("H_pct", "Proporção de casos internados (%)", NULL, pal_h)

# The inset window has to be marked on the national maps or the bottom row is
# three unplaced magnifications. The frame is drawn in the same colour as the
# inset border, and one frame per panel because one window serves all three.
# It is added *after* the insets are cropped from the base panels, so the
# rectangle does not reappear lying exactly on top of each inset's own border.
moldura <- function(p) {
  p + geom_rect(data = w, inherit.aes = FALSE,
                aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                fill = NA, colour = MARCA, linewidth = 0.45)
}

# Shared inset row: the same window cropped from each panel, legends stripped
# (the top row already carries them), border colour matched across the three.
# Badges are dropped in favour of the figure-wide tags, so d/e/f are set in one
# style rather than two.
insA <- crop_to_window(pA0, w) + theme(plot.title = element_blank())
insB <- crop_to_window(pB0, w) + theme(plot.title = element_blank())
insC <- crop_to_window(pC0, w) + theme(plot.title = element_blank())

# What the grey means, said on the map rather than only in the caption. Placed
# in the empty south-west corner of the frame (no health region lies west of
# 58 W below 26 S), so it covers no data at any output size.
pA <- escala_e_norte(moldura(pA0)) +
  annotate("rect", xmin = -73.6, xmax = -70.6, ymin = -20.4, ymax = -18.9,
           fill = CINZA_SEM, colour = "grey45", linewidth = 0.15) +
  annotate("text", x = -69.9, y = -19.65, label = "sem estimativa",
           hjust = 0, vjust = 0.5, size = 2.15, colour = TINTA2)
pB <- moldura(pB0)
pC <- moldura(pC0)

fig <- tags_minusculos((pA | pB | pC) / (insA | insB | insC) +
                       plot_layout(heights = c(1.6, 1)))
salvar(fig, file.path(OUT, "figura1_geografia.png"), columns = 2, height_mm = 130)

cat("\n", describe_windows(w, what = "massa de casos", labels = "D-F"), "\n", sep = "")
cat(sprintf("regioes estimaveis (>= %d casos): %d de %d\n", MIN_CASES,
            sum(mp$estimable), nrow(mp)))
cat(sprintf("incidencia: %.2f a %.2f /100k | letalidade: %.1f%% a %.1f%% | H: %.0f%% a %.0f%%\n",
            min(reg$incidence_per_100k, na.rm=TRUE), max(reg$incidence_per_100k, na.rm=TRUE),
            min(reg$cfr_pct, na.rm=TRUE), max(reg$cfr_pct, na.rm=TRUE),
            min(reg$H_pct, na.rm=TRUE), max(reg$H_pct, na.rm=TRUE)))
