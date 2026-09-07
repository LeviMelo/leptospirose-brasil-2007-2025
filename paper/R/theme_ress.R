# Shared visual grammar for the RESS manuscript.
#
# One file, because the alternative -- each figure choosing its own greys, its
# own region colours, its own legend position -- is what makes a figure set look
# assembled rather than designed.
#
# Journal constraints encoded here rather than remembered:
#   * No title inside the artwork. The title belongs at the head of the legend.
#     None of these helpers sets one, and `salvar()` does not add one.
#   * Composite panels are lettered in LOWERCASE (a, b, c).
#   * 300 dpi minimum; single column 90 mm, double column 180 mm.
#   * Palette readable under deuteranopia and in greyscale (the qualitative
#     ramp varies in lightness as well as hue).

suppressPackageStartupMessages({
  library(ggplot2); library(data.table); library(scales)
  library(patchwork); library(grid); library(ragg)
})

TINTA  <- "#1A1A1A"
TINTA2 <- "#5A5550"
CINZA  <- "#E8E4DC"

# Macro-regions, ordered north to south so a legend reads like the country.
PAL_REGIAO <- c(
  "Norte"        = "#1B7A6E",
  "Nordeste"     = "#C8562B",
  "Centro-Oeste" = "#A38B2E",
  "Sudeste"      = "#3D5A98",
  "Sul"          = "#7B3F82"
)
ORD_REGIAO <- names(PAL_REGIAO)

# Sequential ramp for "more is more" quantities. Stops short of black so the
# darkest bin still admits a boundary line.
RAMP_SEQ <- c("#F7F4EC", "#E8D6A8", "#DCA85C", "#C86E33", "#A63F26", "#6E1F26")

AZUL     <- "#3D5A98"
VERMELHO <- "#B02418"
VERDE    <- "#1B7A6E"

MESES <- c("Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
           "Jul", "Ago", "Set", "Out", "Nov", "Dez")

#' Portuguese decimal marks. A Brazilian journal reads 3,99 and not 3.99.
num_br <- function(x, d = 1) {
  format(round(x, d), decimal.mark = ",", big.mark = ".", nsmall = d, trim = TRUE)
}
int_br <- function(x) formatC(as.integer(round(x)), big.mark = ".", format = "d")
pct_br <- function(x, d = 1) paste0(num_br(x, d), "%")

tema_ress <- function(base = 7.5) {
  theme_minimal(base_size = base) +
    theme(
      text             = element_text(colour = TINTA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = CINZA, linewidth = 0.22),
      axis.title       = element_text(size = base - 0.5),
      axis.text        = element_text(size = base - 1, colour = TINTA2),
      legend.title     = element_text(size = base - 0.5, face = "bold"),
      legend.text      = element_text(size = base - 1),
      legend.key.height = unit(3, "mm"),
      strip.text       = element_text(size = base - 0.5, face = "bold", hjust = 0),
      plot.margin      = margin(2, 3, 2, 2),
      plot.tag         = element_text(size = base + 1, face = "bold", hjust = 0)
    )
}

tema_mapa_ress <- function(base = 7.5) {
  theme_void(base_size = base) +
    theme(text = element_text(colour = TINTA),
          legend.title = element_text(size = base - 0.5, face = "bold"),
          legend.text  = element_text(size = base - 1),
          plot.tag     = element_text(size = base + 1, face = "bold", hjust = 0))
}

#' Uppercase panel tags: RESS asks for (A), (B), (C) rather than lowercase.
tags_paineis <- function(p) {
  p + plot_annotation(tag_levels = "A") &
    theme(plot.tag = element_text(size = 9, face = "bold", hjust = 0))
}
#' Retained name so existing figure scripts keep working.
tags_minusculos <- tags_paineis

#' Save at journal geometry. `columns = 1` is 90 mm, `columns = 2` is 180 mm.
salvar <- function(plot, file, columns = 2, height_mm = 110, dpi = 300) {
  width_mm <- if (columns == 1) 90 else 180
  agg_png(file, width = width_mm, height = height_mm, units = "mm",
          res = dpi, background = "white")
  print(plot)
  dev.off()
  cat(sprintf("  %s (%.0f x %.0f mm, %d dpi)\n", basename(file),
              width_mm, height_mm, dpi))
  invisible(file)
}
