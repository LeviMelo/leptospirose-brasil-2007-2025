"""Descriptive table for the manuscript, rebuilt on corrected denominators.

Two things changed against the first draft and both move published numbers, so
this is not a cosmetic re-run.

1. **The analytic population is 66,358, not 66,516.** The earlier study flow
   excluded 151 records in a single row labelled "onset outside the window".
   There are in fact two exclusions: 155 confirmed records whose symptom-onset
   date is *invalid* and so cannot be placed in any year, and 105 whose onset
   year is valid but outside 2007–2025. Treating the undatable ones as in-window
   inflated the denominator by 109. A further 49 have no residence municipality.

2. **The hospitalisation proportion is computed on its own valid denominator.**
   The earlier figure divided hospitalised cases by *all* confirmed cases,
   silently reading a blank field as "not hospitalised". On the valid
   denominator the national proportion is 72.2%, not 69.7%.

Both corrections were forced by review: a blank is not a negative answer, and a
study flow that does not reconcile to the last record is not auditable.

Outputs to ``data/results/descriptives/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "descriptives"
PANEL = PATHS.results / "analysis_panel"
REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def _block(g: pl.DataFrame) -> dict:
    cases = int(g["cases"].sum())
    py = float(g["person_years"].sum())
    inc, inc_lo, inc_hi = poisson_ci(cases, py, scale=1e5)
    cfr, cfr_lo, cfr_hi = binom_ci(int(g["deaths"].sum()), int(g["outcome_known"].sum()))
    h, h_lo, h_hi = binom_ci(int(g["hospitalised"].sum()), int(g["hosp_known"].sum()))
    lab, lab_lo, lab_hi = binom_ci(int(g["lab_confirmed"].sum()), int(g["crit_known"].sum()))
    sev, sev_lo, sev_hi = binom_ci(int(g["severe"].sum()), int(g["sev_known"].sum()))
    return {
        "cases": cases,
        "person_years": py,
        "incidence_per_100k": float(inc), "incidence_lo": float(inc_lo), "incidence_hi": float(inc_hi),
        "cfr": float(cfr), "cfr_lo": float(cfr_lo), "cfr_hi": float(cfr_hi),
        "H": float(h), "H_lo": float(h_lo), "H_hi": float(h_hi),
        "share_lab": float(lab), "share_lab_lo": float(lab_lo), "share_lab_hi": float(lab_hi),
        "share_severe": float(sev), "share_severe_lo": float(sev_lo), "share_severe_hi": float(sev_hi),
        # Completeness travels with the estimates it conditions, so a reader can
        # see which denominators an estimate rests on without a separate table.
        "outcome_completeness": float(g["outcome_known"].sum() / cases),
        "hosp_completeness": float(g["hosp_known"].sum() / cases),
        "severity_completeness": float(g["sev_known"].sum() / cases),
        "deaths": int(g["deaths"].sum()),
        "outcome_known": int(g["outcome_known"].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(PANEL / "region_year_panel.parquet")

    rows = []
    for reg in REGIONS:
        g = panel.filter(pl.col("region") == reg)
        if g.height == 0:
            continue
        rows.append({"stratum": reg, **_block(g)})
    rows.append({"stratum": "Brasil", **_block(panel)})
    tab = pl.DataFrame(rows)
    tab.write_csv(OUT / "table1_by_region.csv")

    # The national series, for the trend sentence and for a reader who wants to
    # see that completeness is not drifting under the estimates.
    yr = panel.group_by("year").agg(
        pl.col(["cases", "deaths", "outcome_known", "hospitalised", "hosp_known",
                "lab_confirmed", "crit_known", "person_years"]).sum()
    ).sort("year").with_columns(
        (1e5 * pl.col("cases") / pl.col("person_years")).alias("incidence_per_100k"),
        (pl.col("deaths") / pl.col("outcome_known")).alias("cfr"),
        (pl.col("hospitalised") / pl.col("hosp_known")).alias("H"),
        (pl.col("lab_confirmed") / pl.col("crit_known")).alias("share_lab"),
        (pl.col("outcome_known") / pl.col("cases")).alias("outcome_completeness"),
    )
    yr.write_csv(OUT / "national_by_year.csv")

    nat = rows[-1]
    report = {
        "analytic_population": nat["cases"],
        "national": {k: nat[k] for k in
                     ("incidence_per_100k", "cfr", "H", "share_lab", "share_severe",
                      "outcome_completeness", "hosp_completeness", "deaths", "outcome_known")},
        "incidence_range_across_regions": {
            "max_region": tab[:-1].sort("incidence_per_100k", descending=True)[0, "stratum"],
            "max": float(tab[:-1]["incidence_per_100k"].max()),
            "min_region": tab[:-1].sort("incidence_per_100k")[0, "stratum"],
            "min": float(tab[:-1]["incidence_per_100k"].min()),
        },
        "cfr_range_across_regions": {
            "max_region": tab[:-1].sort("cfr", descending=True)[0, "stratum"],
            "max": float(tab[:-1]["cfr"].max()),
            "min_region": tab[:-1].sort("cfr")[0, "stratum"],
            "min": float(tab[:-1]["cfr"].min()),
        },
        "year_range": [int(yr["year"].min()), int(yr["year"].max())],
    }
    (OUT / "descriptives_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(tab.select("stratum", "cases", "incidence_per_100k", "cfr", "H",
                     "share_lab", "share_severe", "outcome_completeness")
             .to_pandas().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
