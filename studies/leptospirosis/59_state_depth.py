"""Detection depth and case fatality by federative unit.

The paper's gradient is measured across health regions, but the literature it
argues with is written at state level: the comparator analyses in RESS and
elsewhere describe one state at a time. To place this study's claim against
those papers, the same index has to exist at their unit of analysis.

The index is defined exactly as everywhere else in the paper -- confirmed cases
recorded as hospitalised, divided by confirmed cases -- and is aggregated from
primitive numerators and denominators, never from means of ratios. Note the
direction: a HIGH hospitalisation share means surveillance reaches only the
severe end, i.e. SHALLOW detection.

Outputs to ``data/results/state_depth/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci
from brepi.config import PATHS

OUT = PATHS.results / "state_depth"

#: A state with fifty confirmed cases over nineteen years carries a case-fatality
#: estimate too imprecise to rank. The correlation is reported with and without
#: such states rather than silently filtering them.
MIN_CASES = 200


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")

    uf = hr.group_by("uf_abbr").agg(
        pl.col("region").first(),
        pl.col("cases").sum(),
        pl.col("deaths").sum(),
        pl.col("hospitalised").sum(),
        pl.col("rq4_outcome_known").sum(),
        pl.col("person_years").sum(),
    )

    rows = []
    for r in uf.iter_rows(named=True):
        dep, dep_lo, dep_hi = binom_ci(r["hospitalised"], r["cases"])
        cfr, cfr_lo, cfr_hi = binom_ci(r["deaths"], r["rq4_outcome_known"])
        rows.append(
            {
                **{k: r[k] for k in ("uf_abbr", "region", "cases", "deaths",
                                     "hospitalised", "rq4_outcome_known")},
                "depth_share": float(dep), "depth_lo": float(dep_lo),
                "depth_hi": float(dep_hi),
                "cfr": float(cfr), "cfr_lo": float(cfr_lo), "cfr_hi": float(cfr_hi),
                "incidence_per_100k": 1e5 * r["cases"] / r["person_years"],
            }
        )

    out = pl.DataFrame(rows).sort("depth_share")

    # The national figure the state values are read against. Computed from the
    # same primitives so the comparison is exact rather than approximately right.
    nat_dep = hr["hospitalised"].sum() / hr["cases"].sum()
    nat_cfr = hr["deaths"].sum() / hr["rq4_outcome_known"].sum()
    out = out.with_columns(
        pl.lit(nat_dep).alias("national_depth_share"),
        pl.lit(nat_cfr).alias("national_cfr"),
    )
    out.write_csv(OUT / "depth_and_cfr_by_state.csv")

    # Does the health-region gradient survive at the unit the comparator
    # literature is written in? Reported on all 27 units and again excluding the
    # two smallest, so the answer does not rest on states with fifty cases.
    big = out.filter(pl.col("cases") >= MIN_CASES)
    rho_all = stats.spearmanr(out["depth_share"], out["cfr"])
    rho_big = stats.spearmanr(big["depth_share"], big["cfr"])
    summary = {
        "n_states": out.height,
        "spearman_depth_vs_cfr": float(rho_all.statistic),
        "spearman_p": float(rho_all.pvalue),
        "n_states_min_cases": big.height,
        "min_cases": MIN_CASES,
        "spearman_depth_vs_cfr_min_cases": float(rho_big.statistic),
        "spearman_p_min_cases": float(rho_big.pvalue),
        "national_depth_share": float(nat_dep),
        "national_cfr": float(nat_cfr),
    }
    (OUT / "state_depth_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"national depth share {100 * nat_dep:.1f}%  cfr {100 * nat_cfr:.1f}%")
    print(
        f"spearman depth vs cfr: {rho_all.statistic:.3f} (n={out.height}) ; "
        f"{rho_big.statistic:.3f} (n={big.height}, >={MIN_CASES} cases)"
    )
    print(
        out.select(
            "uf_abbr", "cases",
            (100 * pl.col("depth_share")).round(1).alias("depth%"),
            (100 * pl.col("cfr")).round(1).alias("cfr%"),
        ).to_pandas().to_string(index=False)
    )


if __name__ == "__main__":
    main()
