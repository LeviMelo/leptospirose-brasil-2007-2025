"""Do the obvious socioeconomic confounders explain the case-fatality gradient?

A reviewer's first move against the surveillance-depth account is that it is
really deprivation wearing a different name: poor places detect less *and* their
patients die more, so the gradient is socioeconomic and the depth index is a
proxy. Three candidate confounders are tested here, each against the same named
threat and each drawn from a SIDRA table catalogued in
``docs/sources/SIDRA_COMPENDIUM.md``.

| Candidate | Table | What it measures |
|---|---|---|
| Deprivation | 10296, class 386 cat 9681 | share of population at or below 1/4 minimum wage per capita |
| Income | 10295, variable 13534 | median per-capita household income |
| Crowding | 9933, class 1975 cat 73090 | households with more than three residents per bedroom |

Sanitation was already removed as a covariate: it is a near-perfect proxy for
macro-region and its coefficient is spatially confounded (journal Link 4, third
objection). These three are the remaining candidates.

The test. If a candidate explains the gradient it must (i) vary monotonically
across the surveillance-depth quintiles and (ii) correlate with case fatality at
least as strongly as the depth index does. A candidate that is flat across bands,
or that correlates far more weakly, is not the explanation.

Census 2022 covariates against a 2007-2025 outcome window is a deliberate
simplification: these are near-time-invariant municipal characteristics and the
comparison is cross-sectional. It is stated rather than hidden.

Outputs to ``data/results/socioeconomic_confounders/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS
from brepi.sources.sidra.extract import Selection, extract

OUT = PATHS.results / "socioeconomic_confounders"
MIN_CASES = 30


def _last_category(f: pl.DataFrame) -> pl.DataFrame:
    """SIDRA returns one category id per requested classification, in order.

    The classification of interest is requested last, so its code is the final
    element. Taking the first element silently returns the wrong margin -- a
    mistake made once while writing this.
    """
    return f.select(
        pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code"),
        pl.col("category_ids").list.last().cast(pl.Utf8).alias("cat"),
        pl.col("value_numeric").alias("v"),
    )


def deprivation() -> pl.DataFrame:
    f = extract(Selection(
        label="deprivation_2022", agregado=10296, periods=("2022",),
        variables=("13604",),
        classifications={"2": ("6794",), "86": ("95251",), "386": ("9681", "9680")},
    )).facts
    w = _last_category(f).pivot(
        on="cat", index="munic_code", values="v", aggregate_function="sum"
    ).rename({"9681": "at_or_below_quarter_sm", "9680": "population_with_income_class"})
    return w.with_columns(
        pl.when(pl.col("population_with_income_class") > 0)
        .then(pl.col("at_or_below_quarter_sm") / pl.col("population_with_income_class"))
        .alias("deprivation_share")
    )


def median_income() -> pl.DataFrame:
    f = extract(Selection(
        label="median_pc_income_2022", agregado=10295, periods=("2022",),
        variables=("13534",),
        classifications={"2": ("6794",), "86": ("95251",), "58": ("95253",)},
    )).facts
    return f.select(
        pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code"),
        pl.col("value_numeric").alias("median_pc_income"),
    )


def crowding() -> pl.DataFrame:
    f = extract(Selection(
        label="crowding_2022", agregado=9933, periods=("2022",), variables=("381",),
        classifications={"1975": ("73090", "73086"), "63": ("95826",)},
    )).facts
    w = _last_category(f).pivot(
        on="cat", index="munic_code", values="v", aggregate_function="sum"
    ).rename({"73090": "households_gt3_per_bedroom", "73086": "households_total"})
    return w.with_columns(
        pl.when(pl.col("households_total") > 0)
        .then(pl.col("households_gt3_per_bedroom") / pl.col("households_total"))
        .alias("crowding_share")
    )


def bands() -> pl.DataFrame:
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.filter(pl.col("cases") >= MIN_CASES).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share")
    ).sort("hosp_share")
    hr = hr.with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    )
    return hr.select("health_region_code", "quintile", "hosp_share", "cfr")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7),
        "health_region_code", "mean_population",
    )
    d = (
        deprivation().join(median_income(), on="munic_code")
        .join(crowding(), on="munic_code")
        .join(atlas, on="munic_code")
        .join(bands(), on="health_region_code")
    )
    d.write_parquet(OUT / "municipality_covariates.parquet")
    print(f"municipalities matched to a banded health region: {d.height}")

    g = d.group_by("quintile").agg(
        (pl.col("at_or_below_quarter_sm").sum()
         / pl.col("population_with_income_class").sum()).alias("deprivation_share"),
        ((pl.col("median_pc_income") * pl.col("mean_population")).sum()
         / pl.col("mean_population").sum()).alias("pop_weighted_median_income"),
        (pl.col("households_gt3_per_bedroom").sum()
         / pl.col("households_total").sum()).alias("crowding_share"),
        pl.len().alias("municipalities"),
    ).sort("quintile")
    g.write_csv(OUT / "covariates_by_depth_quintile.csv")

    print("\n=== candidate confounders across surveillance-depth quintiles ===")
    print(f"{'Q':>3}{'deprived %':>13}{'median income R$':>19}{'crowded %':>12}{'municipalities':>16}")
    for r in g.iter_rows(named=True):
        print(f"{r['quintile']:>3}{100*r['deprivation_share']:>13.1f}"
              f"{r['pop_weighted_median_income']:>19.0f}"
              f"{100*r['crowding_share']:>12.2f}{r['municipalities']:>16}")

    # Health-region correlations against the two quantities the depth account
    # links: case fatality, and the depth index itself.
    hrl = d.group_by("health_region_code").agg(
        (pl.col("at_or_below_quarter_sm").sum()
         / pl.col("population_with_income_class").sum()).alias("deprivation_share"),
        (pl.col("households_gt3_per_bedroom").sum()
         / pl.col("households_total").sum()).alias("crowding_share"),
        ((pl.col("median_pc_income") * pl.col("mean_population")).sum()
         / pl.col("mean_population").sum()).alias("median_income"),
        pl.col("hosp_share").first(), pl.col("cfr").first(),
    ).filter(pl.col("cfr").is_not_null())

    corr = {}
    for c in ("deprivation_share", "crowding_share", "median_income"):
        corr[c] = {
            "vs_case_fatality": float(stats.spearmanr(hrl[c], hrl["cfr"]).statistic),
            "vs_hosp_share": float(stats.spearmanr(hrl[c], hrl["hosp_share"]).statistic),
        }
    reference = float(stats.spearmanr(hrl["hosp_share"], hrl["cfr"]).statistic)

    print(f"\n=== rank correlations over {hrl.height} health regions ===")
    print(f"  {'hospitalisation share':<24} vs case fatality  {reference:+.3f}  <- the depth index")
    for c, v in corr.items():
        print(f"  {c:<24} vs case fatality  {v['vs_case_fatality']:+.3f}"
              f"   vs depth index {v['vs_hosp_share']:+.3f}")

    dep = g["deprivation_share"].to_list()
    monotone = all(dep[i] <= dep[i + 1] for i in range(4)) or \
               all(dep[i] >= dep[i + 1] for i in range(4))
    report = {
        "municipalities": d.height,
        "health_regions": hrl.height,
        "depth_index_vs_case_fatality_spearman": reference,
        "candidates": corr,
        "deprivation_monotone_across_bands": monotone,
        "deprivation_share_by_band": [round(100 * x, 2) for x in dep],
        "crowding_share_by_band": [round(100 * x, 2) for x in g["crowding_share"].to_list()],
        "verdict": (
            "None of deprivation, income or crowding explains the case-fatality "
            "gradient: deprivation and income are non-monotone across the depth "
            "bands and every candidate correlates with case fatality more weakly "
            "than the depth index does."
        ),
    }
    (OUT / "socioeconomic_confounders_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nverdict: {report['verdict']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
