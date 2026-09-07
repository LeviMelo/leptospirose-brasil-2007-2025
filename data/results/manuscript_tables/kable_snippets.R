# Manuscript tables. Generated strings only -- do not reformat in R.
# Written by studies/leptospirosis/71_manuscript_tables.py
#
# colClasses = "character" is not optional. Every cell is already a Portuguese
# formatted string, and R would read "4.941" (four thousand nine hundred and
# forty-one region-years) as the number 4.941 -- which happens to print the
# same today and would silently lose a digit the first time a count ends in a
# zero. Read everything as text and let the CSV be the single source of truth
# for how a number looks.

# --- Tabela 1 -------------------------------------------------------------
t1 <- read.csv("data/results/manuscript_tables/table1_descritiva.csv",
               check.names = FALSE, colClasses = "character",
               encoding = "UTF-8")

knitr::kable(
  t1,
  format    = "latex",          # or "pipe" / "html"
  booktabs  = TRUE,
  align     = c("l", rep("r", ncol(t1) - 1L)),
  escape    = FALSE,
  caption   = paste("Epidemiologia descritiva da leptospirose confirmada por",
                    "macrorregião, Brasil, 2007-2025. Incidência por 100.000",
                    "pessoas-ano; H = proporção de casos internados entre os",
                    "casos com o campo de internação preenchido, indicador",
                    "inverso da amplitude de detecção. Intervalos exatos",
                    "(Poisson gama; Clopper-Pearson).")
) |>
  kableExtra::kable_styling(latex_options = c("scale_down")) |>
  kableExtra::row_spec(nrow(t1) - 1L, hline_after = TRUE)   # regra antes de Brasil

# --- Tabela 2 -------------------------------------------------------------
t2   <- read.csv("data/results/manuscript_tables/table2_modelos.csv",
                 check.names = FALSE, colClasses = "character",
                 encoding = "UTF-8")
nota <- read.csv("data/results/manuscript_tables/table2_nota.csv",
                 check.names = FALSE, colClasses = "character",
                 encoding = "UTF-8")$Nota

knitr::kable(
  t2,
  format   = "latex",
  booktabs = TRUE,
  align    = c("l", "l", "r", "r", "l"),
  escape   = FALSE,
  caption  = paste("Associação entre a proporção de casos internados (H) e a",
                   "letalidade notificada da leptospirose, região de saúde x",
                   "ano, Brasil, 2007-2025. OR por +10 pontos percentuais de H;",
                   "modelos A-F ajustados por INLA (logito binomial, campo",
                   "espacial BYM2 e RW1 sobre o ano).")
) |>
  kableExtra::kable_styling(latex_options = c("scale_down")) |>
  kableExtra::column_spec(2, width = "38mm") |>
  kableExtra::column_spec(5, width = "52mm") |>
  kableExtra::footnote(general = nota, general_title = "", threeparttable = TRUE)
