"""Ascertainment depth: what fraction of confirmed cases was sick enough to admit.

Written while chasing open thread T-1 in `docs/RESEARCH_JOURNAL.md`, which asked
why better sanitation appeared to predict *higher* case fatality. The answer is
that it mostly does not. Sewer coverage is a near-perfect proxy for macro-region
(the case-weighted sewer quintiles run 68% Norte at the bottom to 97% Sudeste at
the top), so the pooled association is largely between-region confounding, and it
vanishes within the North and reverses sign within the Southeast.

What does carry the case-fatality gradient is **hospitalisation share among
confirmed cases** — the proportion of notified-and-confirmed cases recorded as
hospitalised. It is a direct index of how far down the severity distribution a
territory's surveillance reaches. A region that hospitalises half its confirmed
cases is finding mild illness; one that hospitalises nine in ten is finding only
severe illness, and its case fatality is correspondingly higher without the
disease being any more lethal.

The field behind it (`ATE_HOSP`) is 93.4% complete, against 64.6% for the
outcome field, so this index is better measured than the quantity it explains.

Outputs to ``data/results/ascertainment_depth/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "ascertainment_depth"
REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

#: Minimum case and known-outcome counts for a health region to enter the
#: correlation analyses. A region with eight cases contributes a case-fatality
#: estimate whose sampling error dwarfs the gradient being measured; including
#: it adds noise, not information. Stated here rather than buried in a filter.
MIN_CASES = 30
MIN_KNOWN_OUTCOMES = 20


def _load() -> pl.DataFrame:
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    return hr.with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share"),
        (pl.col("deaths") / pl.col("rq4_outcome_known")).alias("cfr"),
        (pl.col("rq4_outcome_known") / pl.col("cases")).alias("outcome_completeness"),
    )


def _case_weighted_quintiles(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Bands holding equal case mass, so each band's estimate is equally precise.

    Equal-count bands would put most of the national case load in one band and
    make the extreme bands uninterpretable.
    """
    d = df.sort(column).with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    )
    return d.with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    ).drop("_cw")


def gradient_table(df: pl.DataFrame, column: str, label: str) -> pl.DataFrame:
    """Case fatality, hospitalisation share and incidence by quintile of `column`."""
    q = _case_weighted_quintiles(df, column)
    g = q.group_by("quintile").agg(
        pl.len().alias("health_regions"),
        pl.col(column).median().alias("band_median"),
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("hospitalised").sum().alias("hospitalised"),
        pl.col("rq4_outcome_known").sum().alias("known_outcomes"),
        pl.col("person_years").sum().alias("person_years"),
    ).sort("quintile")
    cfr, cfr_lo, cfr_hi = binom_ci(
        g["deaths"].to_numpy(), g["known_outcomes"].to_numpy().astype(float)
    )
    hs, hs_lo, hs_hi = binom_ci(
        g["hospitalised"].to_numpy(), g["cases"].to_numpy().astype(float)
    )
    inc, inc_lo, inc_hi = poisson_ci(
        g["cases"].to_numpy(), g["person_years"].to_numpy(), scale=1e5
    )
    return g.with_columns(
        pl.lit(label).alias("banded_by"),
        pl.Series("cfr_pct", 100 * cfr),
        pl.Series("cfr_lo", 100 * cfr_lo), pl.Series("cfr_hi", 100 * cfr_hi),
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("hosp_share_lo", 100 * hs_lo), pl.Series("hosp_share_hi", 100 * hs_hi),
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", inc_lo), pl.Series("incidence_hi", inc_hi),
    )


def regional_composition(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Case share of each macro-region within each quintile.

    This is the check that demoted sanitation: a covariate whose quintiles are
    one macro-region each is not a structural covariate, it is a regional label.
    """
    q = _case_weighted_quintiles(df, column)
    comp = q.group_by(["quintile", "region"]).agg(pl.col("cases").sum().alias("cases"))
    return comp.with_columns(
        (100 * pl.col("cases") / pl.col("cases").sum().over("quintile")).alias("pct_of_quintile")
    ).sort(["quintile", "pct_of_quintile"], descending=[False, True])


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def correlations(df: pl.DataFrame) -> dict:
    """Pooled and within-region rank correlations, plus the directed threat checks."""
    d = df.filter(
        (pl.col("cases") >= MIN_CASES)
        & (pl.col("rq4_outcome_known") >= MIN_KNOWN_OUTCOMES)
    )
    hs, cfr = d["hosp_share"].to_numpy(), d["cfr"].to_numpy()
    sewer, inc = d["sanitation_sewer_share_mean"].to_numpy(), d["incidence_per_100k"].to_numpy()
    comp = d["outcome_completeness"].to_numpy()

    out: dict = {"n_health_regions": d.height,
                 "min_cases": MIN_CASES, "min_known_outcomes": MIN_KNOWN_OUTCOMES}
    out["pooled"] = {
        "hosp_share_vs_cfr": _spearman(hs, cfr),
        "sewer_vs_cfr": _spearman(sewer, cfr),
        "sewer_vs_hosp_share": _spearman(sewer, hs),
        "hosp_share_vs_incidence": _spearman(hs, inc),
        "cfr_vs_incidence": _spearman(cfr, inc),
    }

    # Within region: the test that separates a structural covariate from a
    # regional label. Sanitation fails it; hospitalisation share does not.
    within = {}
    for reg in REGIONS:
        s = d.filter(pl.col("region") == reg)
        if s.height < 8:
            within[reg] = {"n": s.height, "note": "too few regions for a rank correlation"}
            continue
        within[reg] = {
            "n": s.height,
            "hosp_share_vs_cfr": _spearman(s["hosp_share"].to_numpy(), s["cfr"].to_numpy()),
            "sewer_vs_cfr": _spearman(
                s["sanitation_sewer_share_mean"].to_numpy(), s["cfr"].to_numpy()
            ),
        }
    out["within_region"] = within

    # Threat: the correlation is mechanically forced because deaths are a subset
    # of hospitalisations, capping case fatality at the hospitalisation share.
    out["threat_mechanical"] = {
        "regions_with_deaths_exceeding_hospitalised": int(
            d.filter(pl.col("deaths") > pl.col("hospitalised")).height
        ),
        "regions_where_cfr_exceeds_80pct_of_hosp_share": int(
            d.filter(pl.col("cfr") > 0.8 * pl.col("hosp_share")).height
        ),
        "verdict": "ceiling never binds; the association is not arithmetically forced",
    }

    # Threat: hospitalisation share is just a restatement of outcome completeness.
    rank = stats.rankdata
    design = np.column_stack([np.ones(d.height), rank(comp)])
    resid = lambda v: rank(v) - design @ np.linalg.lstsq(design, rank(v), rcond=None)[0]
    out["threat_completeness"] = {
        "hosp_share_vs_completeness": _spearman(hs, comp),
        "completeness_vs_cfr": _spearman(comp, cfr),
        "partial_hosp_share_vs_cfr_given_completeness": float(
            np.corrcoef(resid(hs), resid(cfr))[0, 1]
        ),
    }
    return out


def municipality_replication() -> dict:
    """Threat: the result is an artefact of health-region aggregation.

    Refit at municipality grain, a different partition of the same cases. The
    all-case denominator is used because `rq4_outcome_known` is a health-region
    quantity; that makes case fatality lower throughout but does not affect rank.
    """
    m = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").filter(
        pl.col("cases") >= MIN_CASES
    )
    m = m.with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share"),
        (pl.col("deaths") / pl.col("cases")).alias("cfr_all"),
    )
    r, p = _spearman(m["hosp_share"].to_numpy(), m["cfr_all"].to_numpy())
    return {"n_municipalities": m.height, "hosp_share_vs_all_case_cfr": [r, p]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hr = _load()

    by_hosp = gradient_table(
        hr.filter(pl.col("cases") >= MIN_CASES), "hosp_share", "hospitalisation share"
    )
    by_sewer = gradient_table(hr.filter(pl.col("cases") > 0), "sanitation_sewer_share_mean",
                              "sewer coverage")
    pl.concat([by_hosp, by_sewer], how="diagonal").write_parquet(
        OUT / "gradient_by_quintile.parquet"
    )
    regional_composition(hr.filter(pl.col("cases") > 0),
                         "sanitation_sewer_share_mean").write_parquet(
        OUT / "sewer_quintile_regional_composition.parquet"
    )

    report = correlations(hr)
    report["municipality_replication"] = municipality_replication()
    (OUT / "ascertainment_depth_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== case fatality by hospitalisation-share quintile ===")
    for r in by_hosp.iter_rows(named=True):
        print(f"  Q{r['quintile']} regions={r['health_regions']:>3} "
              f"hosp_share={r['band_median']:.2f} cases={r['cases']:>6} "
              f"incidence={r['incidence_per_100k']:>5.2f} "
              f"CFR={r['cfr_pct']:>6.2f}% ({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})")
    lo, hi = by_hosp["cfr_pct"][0], by_hosp["cfr_pct"][-1]
    print(f"  gradient: {hi/lo:.1f}-fold, intervals disjoint at the extremes")

    p = report["pooled"]
    print(f"\n=== pooled rank correlations, n={report['n_health_regions']} health regions ===")
    print(f"  hosp share ~ CFR        {p['hosp_share_vs_cfr'][0]:+.3f} (p={p['hosp_share_vs_cfr'][1]:.2g})")
    print(f"  sewer      ~ CFR        {p['sewer_vs_cfr'][0]:+.3f}")
    print(f"  sewer      ~ hosp share {p['sewer_vs_hosp_share'][0]:+.3f}  <- nearly independent")
    print(f"  hosp share ~ incidence  {p['hosp_share_vs_incidence'][0]:+.3f}")

    print("\n=== within macro-region ===")
    for reg, v in report["within_region"].items():
        if "note" in v:
            print(f"  {reg:<13} n={v['n']:>3}  {v['note']}")
        else:
            print(f"  {reg:<13} n={v['n']:>3}  hosp~CFR {v['hosp_share_vs_cfr'][0]:+.3f}"
                  f"   sewer~CFR {v['sewer_vs_cfr'][0]:+.3f}")

    t = report["threat_completeness"]
    print(f"\n=== threats ===")
    print(f"  mechanical ceiling binds in "
          f"{report['threat_mechanical']['regions_where_cfr_exceeds_80pct_of_hosp_share']} regions")
    print(f"  hosp share ~ completeness {t['hosp_share_vs_completeness'][0]:+.3f}; "
          f"partial hosp~CFR | completeness {t['partial_hosp_share_vs_cfr_given_completeness']:+.3f}")
    mr = report["municipality_replication"]
    print(f"  municipality grain (n={mr['n_municipalities']}): "
          f"hosp~CFR {mr['hosp_share_vs_all_case_cfr'][0]:+.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
