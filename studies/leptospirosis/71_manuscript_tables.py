"""The two manuscript tables, formatted for print.

This script does no estimation. Every number it prints was computed elsewhere
and is read back from the result files, because a table that recomputes its own
numbers is a second analysis that can silently disagree with the first.

  Table 1  descriptive epidemiology by macro-region
           <- data/results/descriptives/table1_by_region.csv

  Table 2  the model sequence
           <- data/results/primary_model/model_estimates.csv
              data/results/primary_model/primary_model_report.json

Table 1 is the orientation table. Its columns are not the usual demographic
inventory: they are the four quantities the study question turns on -- how many
cases a territory notifies per head, how often those cases die, how narrow the
ascertainment was (H), and how much of the outcome field was actually filled --
plus laboratory confirmation, because a reader is entitled to know whether the
case definition itself moves across the country. Completeness travels beside
the estimate it conditions rather than in a separate data-quality appendix.

Table 2 replaces an earlier "candidate explanations" table that mixed
monotonicity labels, correlation coefficients and prose, and was correctly
described in review as internal exploratory notes pasted into a manuscript. A
model table states a specification, an estimate with an interval, the number of
observations behind it, and nothing else.

One number in Table 2 is not stored in any result file: the number of
region-years behind the fixed-effects estimate. It is reconstructed here by
replaying the exact filter documented in 68_primary_model.R, and the
reconstruction is refused unless it reproduces the region count that script
recorded (306 of 426). If the panel ever changes underneath, this fails loudly
instead of printing a stale count.

Outputs to ``data/results/manuscript_tables/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS

DESC = PATHS.results / "descriptives" / "table1_by_region.csv"
MODEL_CSV = PATHS.results / "primary_model" / "model_estimates.csv"
MODEL_JSON = PATHS.results / "primary_model" / "primary_model_report.json"
PANEL = PATHS.results / "analysis_panel" / "region_year_panel.parquet"
OUT = PATHS.results / "manuscript_tables"

DASH = "–"   # en dash, the range separator
EMDASH = "—"  # em dash, "not estimated"


# ---------------------------------------------------------------------------
# Portuguese number formatting. Mirrors num_br/int_br in paper/R/theme_ress.R
# so a number printed by a figure and the same number printed by a table look
# identical on the page.
# ---------------------------------------------------------------------------
def num_br(x: float, d: int = 1) -> str:
    s = f"{x:,.{d}f}"                      # 1,234.5
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def int_br(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def ci_br(est: float, lo: float, hi: float, d: int = 1, mult: float = 1.0) -> str:
    return (f"{num_br(est * mult, d)} ({num_br(lo * mult, d)}{DASH}"
            f"{num_br(hi * mult, d)})")


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def build_table1() -> tuple[pl.DataFrame, list[dict]]:
    d = pl.read_csv(DESC)
    total = float(d.filter(pl.col("stratum") == "Brasil")["cases"][0])

    regions = d.filter(pl.col("stratum") != "Brasil").sort(
        "incidence_per_100k", descending=True)
    ordered = pl.concat([regions, d.filter(pl.col("stratum") == "Brasil")])

    rows, raw = [], []
    for r in ordered.iter_rows(named=True):
        share = 100.0 * r["cases"] / total
        rows.append({
            "Macrorregião": r["stratum"],
            "Casos n (%)": f"{int_br(r['cases'])} ({num_br(share, 1)})",
            "Incidência (IC95%)": ci_br(
                r["incidence_per_100k"], r["incidence_lo"], r["incidence_hi"], 2),
            "Letalidade % (IC95%)": ci_br(r["cfr"], r["cfr_lo"], r["cfr_hi"], 2, 100),
            "H % (IC95%)": ci_br(r["H"], r["H_lo"], r["H_hi"], 1, 100),
            "Confirmação laboratorial %": num_br(100 * r["share_lab"], 1),
            "Completude do desfecho %": num_br(100 * r["outcome_completeness"], 1),
        })
        raw.append({
            "stratum": r["stratum"],
            "cases": int(r["cases"]),
            "share_of_cases_pct": share,
            "person_years": r["person_years"],
            "incidence_per_100k": r["incidence_per_100k"],
            "incidence_lo": r["incidence_lo"],
            "incidence_hi": r["incidence_hi"],
            "cfr_pct": 100 * r["cfr"],
            "cfr_lo_pct": 100 * r["cfr_lo"],
            "cfr_hi_pct": 100 * r["cfr_hi"],
            "H_pct": 100 * r["H"],
            "H_lo_pct": 100 * r["H_lo"],
            "H_hi_pct": 100 * r["H_hi"],
            "share_lab_pct": 100 * r["share_lab"],
            "outcome_completeness_pct": 100 * r["outcome_completeness"],
            "deaths": int(r["deaths"]),
            "outcome_known": int(r["outcome_known"]),
        })
    return pl.DataFrame(rows), raw


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def fe_cells(report: dict) -> int:
    """Region-years behind the fixed-effects fit, by replaying 68_primary_model.R.

    A region contributes to a within estimate only if it has both a death and a
    survivor somewhere in its series and is observed in more than one year;
    otherwise its own fixed effect predicts it perfectly and beta gets nothing
    from it. That filter is what produced the 306 regions in the report.
    """
    p = pl.read_parquet(PANEL)
    dt = p.filter((pl.col("outcome_known") > 0) & (pl.col("hosp_known") > 0))
    per = dt.group_by("health_region_code").agg(
        d=pl.col("deaths").sum(), n=pl.col("outcome_known").sum(),
        yrs=pl.len())
    usable = per.filter((pl.col("d") > 0) & (pl.col("d") < pl.col("n"))
                        & (pl.col("yrs") > 1))["health_region_code"]
    fedt = dt.filter(pl.col("health_region_code").is_in(usable.implode()))

    want = report["within_region"]["regions_contributing"]
    got = len(usable)
    if got != want:
        raise SystemExit(
            f"refusing to print a reconstructed cell count: the fixed-effects "
            f"filter now yields {got} regions, but primary_model_report.json "
            f"recorded {want}. The panel or the filter changed; re-run "
            f"68_primary_model.R before rebuilding this table.")
    return fedt.height


def build_table2() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    est = pl.read_csv(MODEL_CSV)
    rep = json.loads(MODEL_JSON.read_text(encoding="utf-8"))

    def get(model: str, term: str) -> dict:
        r = est.filter((pl.col("model") == model) & (pl.col("term") == term))
        if r.height != 1:
            raise SystemExit(f"expected exactly one row for {model}/{term}")

        return r.row(0, named=True)

    def orh(model: str) -> str:
        r = get(model, "H10")
        return ci_br(r["estimate"], r["lower"], r["upper"], 3)

    n_fe = fe_cells(rep)
    mods = rep["models"]
    checks = rep["threat_checks"]
    wr = rep["within_region"]

    # Cell counts come from the estimates file, which carries a `cells` row for
    # every INLA fit; FE is the one reconstruction (see fe_cells).
    def cells(model: str) -> int:
        return int(float(get(model, "cells")["estimate"]))

    spec = {
        "A": "Espaço (BYM2) + tempo (RW1), sem exposição",
        "B": "A + H (contínuo)",
        "C": "B + composição etária e sexo",
        "D": "C restrito a células com completude do desfecho ≥ 95%",
        "E": "C, com o desfecho redefinido como letalidade apenas entre internados",
        "F": "C restrito a células com ≥ 10 casos por trás de H",
        "FE": "Efeitos fixos de região e de ano + composição; erros-padrão "
              "robustos por conglomerado (região)",
    }

    obs = {
        "A": ("Referência sem exposição. Variância espacial marginal "
              + ci_br(mods["A_space_time"]["spatial_marginal_variance"]["estimate"],
                      mods["A_space_time"]["spatial_marginal_variance"]["lower"],
                      mods["A_space_time"]["spatial_marginal_variance"]["upper"], 3)),
        "B": ("A variância espacial marginal cai para "
              + ci_br(mods["B_plus_H"]["spatial_marginal_variance"]["estimate"],
                      mods["B_plus_H"]["spatial_marginal_variance"]["lower"],
                      mods["B_plus_H"]["spatial_marginal_variance"]["upper"], 3)
              + " ao entrar H"),
        "C": ("Modelo principal. OR por +10 pp de casos com 60 anos ou mais "
              + ci_br(mods["C_plus_case_mix"]["or_age60_per_10pp"]["estimate"],
                      mods["C_plus_case_mix"]["or_age60_per_10pp"]["lower"],
                      mods["C_plus_case_mix"]["or_age60_per_10pp"]["upper"], 3)),
        "D": (f"Retém {num_br(100 * mods['D_complete_outcomes']['cell_share_retained'], 1)}%"
              " das células; a completude seletiva do desfecho não explica o gradiente"),
        "E": ("Afasta a explicação puramente aritmética: com a composição "
              "internados/não internados fixada, resta gradiente residual "
              f"(letalidade entre internados {num_br(100 * checks['E_within_hospitalised']['within_hospitalised_case_fatality'], 2)}%)"),
        "F": (f"Retém {num_br(100 * checks['F_hosp_known_ge10']['case_share_retained'], 1)}%"
              " dos casos; o gradiente não é obra de células com poucos casos"),
        "FE": (f"{int_br(wr['regions_contributing'])} de {int_br(wr['regions_total'])}"
               " regiões contribuem, "
               f"{num_br(100 * wr['case_share_retained'], 1)}% dos casos; "
               "estimativa identificada apenas pela variação de H dentro da "
               "própria região"),
    }

    spec_table = [
        ("A", "A_space_time", None, cells("A_space_time")),
        ("B", "B_plus_H", orh("B_plus_H"), cells("B_plus_H")),
        ("C", "C_plus_case_mix", orh("C_plus_case_mix"), cells("C_plus_case_mix")),
        ("D", "D_complete_outcomes", orh("D_complete_outcomes"),
         cells("D_complete_outcomes")),
        ("E", "E_within_hospitalised", orh("E_within_hospitalised"),
         cells("E_within_hospitalised")),
        ("F", "F_hosp_known_ge10", orh("F_hosp_known_ge10"),
         cells("F_hosp_known_ge10")),
        ("FE", "FE_region_year_H_case_mix", orh("FE_region_year_H_case_mix"), n_fe),
    ]

    rows, raw = [], []
    for label, model, or_txt, n in spec_table:
        rows.append({
            "Modelo": label,
            "Especificação": spec[label],
            "OR de H por +10 pp (IC95%)": or_txt if or_txt else EMDASH,
            "Células": int_br(n),
            "Observação": obs[label],
        })
        h = (get(model, "H10") if or_txt else None)
        raw.append({
            "label": label,
            "model": model,
            "specification_pt": spec[label],
            "or_H_per_10pp": h["estimate"] if h else None,
            "lower": h["lower"] if h else None,
            "upper": h["upper"] if h else None,
            "cells": n,
        })

    phiC = mods["C_plus_case_mix"]["phi"]
    ab = rep["spatial_variance_absorbed_by_H"]
    foot = (
        "BYM2 no modelo C: φ = "
        + ci_br(phiC["estimate"], phiC["lower"], phiC["upper"], 3)
        + ", a fração da variância marginal do efeito espacial que é "
        "estruturada entre vizinhos (não é a fração da variação da letalidade "
        "que é espacial). Ao entrar H, a variância espacial de A cai "
        f"{num_br(100 * ab['proportion_absorbed_marginal'], 1)}% na escala "
        "marginal e "
        f"{num_br(100 * ab['proportion_absorbed_empirical'], 1)}% na escala "
        "empírica (variância dos efeitos regionais a posteriori); A e B são "
        "ajustes separados sobre as mesmas linhas, de modo que a comparação é "
        "entre resumos a posteriori e não uma posteriori da diferença. "
        "Células = região-ano. A unidade de observação é o território: toda "
        "estimativa é uma associação ecológica entre a amplitude de detecção "
        "de um território e sua letalidade notificada."
    )
    footer = pl.DataFrame([{"Nota": foot}])

    raw_extra = {
        "phi_model_C": {k: phiC[k] for k in ("estimate", "lower", "upper")},
        "spatial_variance_absorbed_marginal": ab["proportion_absorbed_marginal"],
        "spatial_variance_absorbed_empirical": ab["proportion_absorbed_empirical"],
        "marginal_variance_A": ab["marginal_variance_A"],
        "marginal_variance_B": ab["marginal_variance_B"],
        "fe_cells_reconstructed": n_fe,
        "fe_regions_contributing": wr["regions_contributing"],
        "fe_regions_total": wr["regions_total"],
        "footnote_pt": foot,
    }
    return pl.DataFrame(rows), footer, {"models": raw, **raw_extra}


# ---------------------------------------------------------------------------
# The R side. The snippets deliberately contain no formatting logic: the CSVs
# already hold the final strings, so the manuscript only chooses column widths
# and alignment. Anything that has to decide how many decimals a number gets
# belongs here in Python, where it is version-controlled next to the source.
# ---------------------------------------------------------------------------
R_SNIPPETS = r'''# Manuscript tables. Generated strings only -- do not reformat in R.
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
'''


def show(name: str, df: pl.DataFrame) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=200,
                   tbl_width_chars=240, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print(df)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    t1, t1_raw = build_table1()
    t2, t2_foot, t2_raw = build_table2()

    t1.write_csv(OUT / "table1_descritiva.csv")
    t2.write_csv(OUT / "table2_modelos.csv")
    t2_foot.write_csv(OUT / "table2_nota.csv")
    (OUT / "manuscript_tables.json").write_text(
        json.dumps({"table1": t1_raw, "table2": t2_raw}, indent=2,
                   ensure_ascii=False),
        encoding="utf-8")
    (OUT / "kable_snippets.R").write_text(R_SNIPPETS, encoding="utf-8")

    show("Tabela 1 - Epidemiologia descritiva por macrorregião, Brasil, 2007-2025",
         t1)
    show("Tabela 2 - Sequência de modelos", t2)
    print("\nNota da Tabela 2:\n" + t2_foot["Nota"][0])

    print("\nwritten:")
    for f in ("table1_descritiva.csv", "table2_modelos.csv", "table2_nota.csv",
              "manuscript_tables.json", "kable_snippets.R"):
        print("  ", OUT / f)


if __name__ == "__main__":
    main()
