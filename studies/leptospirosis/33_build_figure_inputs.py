"""Derived quantities the redesigned figures need, at the grain they need them.

Section 7 of ``docs/protocol/PUBLICATION_STRATEGY.md`` lists what blocks the
figures for Papers A, B and C. This script closes the computational ones.

1. **Ascertainment ratio by health region.** It exists only per municipality.
   Paper A's second figure plots incidence against detection on a plane whose
   points are health regions, because at municipality grain the denominators are
   too small to read and the point cloud is 5,570 wide.

   The ratio is ``SINAN confirmed / (SIM A27 deaths + SIH A27 admissions)``.
   Aggregating it means summing the three primitives and dividing once --
   averaging municipal ratios would weight a municipality with two cases equally
   with Sao Paulo, and would silently produce a different national figure. The
   contract is enforced by :func:`brepi.panel.aggregate.aggregate_panel`, which
   refuses to sum a column whose name marks it non-additive.

2. **Serogroup composition by transmission regime.** It exists by macro-region.
   Paper C's argument is that serology corroborates regimes recovered without
   it, so the composition has to be cut by regime.

3. **Paired SINAN/SIH signal by health region**, for the replication figure.

Writes to ``data/results/figure_inputs/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import poisson_ci, proportion_table
from brepi.config import PATHS
from brepi.panel.aggregate import aggregate_panel

OUT = PATHS.results / "figure_inputs"
ATLAS = PATHS.results / "atlas"


def health_region_ascertainment(atlas: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the three primitives, then form the ratio once."""
    agg, report = aggregate_panel(
        atlas,
        keys=["health_region_code"],
        sums=[
            "cases", "deaths", "hospitalised", "person_years",
            "sim_a27_deaths_total", "sih_a27_admissions_total",
        ],
        constants=["health_region_name", "uf_abbr", "region"],
        check_constants=True,
    )
    if not report.passed:
        raise AssertionError(
            "columns declared constant within a health region are not: "
            f"{report.varying_constants}"
        )

    independent = pl.col("sim_a27_deaths_total") + pl.col("sih_a27_admissions_total")
    agg = agg.with_columns(
        independent.alias("independent_signal"),
        (pl.col("cases") > 0).alias("sinan_positive"),
        (independent > 0).alias("other_positive"),
    ).with_columns(
        pl.when(pl.col("other_positive"))
        .then(pl.col("cases") / pl.col("independent_signal"))
        .otherwise(None)
        .alias("sinan_per_independent_signal")
    )

    # poisson_ci returns (rate, lower, upper) -- estimate FIRST.
    inc, lo, hi = poisson_ci(
        agg["cases"].to_numpy(), agg["person_years"].to_numpy(), scale=1e5
    )
    return agg.with_columns(
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_per_100k_lo", lo),
        pl.Series("incidence_per_100k_hi", hi),
    ).sort("health_region_code")


def serogroup_by_regime(long: pl.DataFrame, atlas: pl.DataFrame) -> pl.DataFrame:
    """Serogroup composition within each unsupervised regime.

    Restricted to municipalities the clustering was eligible to place. A
    municipality with serology but no regime is not evidence about any regime,
    and folding it into a total would be composition drawn from a different
    universe than the map beside it.
    """
    regimes = atlas.filter(
        pl.col("rq5_eligible") & pl.col("rq5_regime").is_not_null()
    ).select("munic_code", "rq5_regime")
    joined = long.join(regimes, on="munic_code", how="inner")
    by_regime = joined.group_by(["rq5_regime", "serogroup"]).agg(
        pl.col("n").sum().alias("n")
    )
    totals = by_regime.group_by("rq5_regime").agg(pl.col("n").sum().alias("typed_total"))
    out = by_regime.join(totals, on="rq5_regime", how="left")
    out = proportion_table(out, count="n", total="typed_total", prefix="share")
    return out.sort(["rq5_regime", "n"], descending=[False, True])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pl.read_parquet(ATLAS / "municipality_atlas.parquet")
    region_atlas = pl.read_parquet(ATLAS / "health_region_atlas.parquet")
    serogroup = pl.read_parquet(ATLAS / "municipality_serogroup_long.parquet")
    report: dict[str, object] = {}

    # -- 1. ascertainment by health region -----------------------------------
    hr = health_region_ascertainment(atlas)
    hr.write_parquet(OUT / "health_region_ascertainment.parquet")

    national_sinan = int(atlas["cases"].sum())
    reagg = int(hr["cases"].sum())
    assert reagg == national_sinan, f"cases lost in aggregation: {reagg} vs {national_sinan}"

    defined = hr.filter(pl.col("sinan_per_independent_signal").is_not_null())
    report["health_regions"] = hr.height
    report["health_regions_with_independent_signal"] = defined.height
    report["ratio_median"] = float(defined["sinan_per_independent_signal"].median())
    report["ratio_p10"] = float(defined["sinan_per_independent_signal"].quantile(0.10))
    report["ratio_p90"] = float(defined["sinan_per_independent_signal"].quantile(0.90))
    print(f"health regions: {hr.height}; ratio defined in {defined.height}")
    print(f"  ratio median {report['ratio_median']:.2f} "
          f"(p10 {report['ratio_p10']:.2f}, p90 {report['ratio_p90']:.2f})")

    # The claim Paper A's figure 2 was to rest on: low incidence travels with
    # low detection rather than with a healthy population. Computed here so the
    # figure is designed around what the contrast actually is -- see the note in
    # the report, because at this grain it is weak.
    lo_third = defined.filter(
        pl.col("incidence_per_100k") <= defined["incidence_per_100k"].quantile(1 / 3)
    )
    hi_third = defined.filter(
        pl.col("incidence_per_100k") >= defined["incidence_per_100k"].quantile(2 / 3)
    )
    report["ratio_lowest_incidence_tertile"] = float(
        lo_third["sinan_per_independent_signal"].median()
    )
    report["ratio_highest_incidence_tertile"] = float(
        hi_third["sinan_per_independent_signal"].median()
    )
    print(f"  ratio in lowest-incidence tertile "
          f"{report['ratio_lowest_incidence_tertile']:.2f} vs highest "
          f"{report['ratio_highest_incidence_tertile']:.2f}")

    # -- 2. serogroup composition by regime ----------------------------------
    sero = serogroup_by_regime(serogroup, atlas)
    sero.write_parquet(OUT / "serogroup_by_regime.parquet")
    report["serogroup_regimes"] = int(sero["rq5_regime"].n_unique())
    report["serogroup_typed_total"] = int(
        sero.unique(subset="rq5_regime")["typed_total"].sum()
    )
    print(f"\nserogroup by regime: {report['serogroup_regimes']} regimes, "
          f"{report['serogroup_typed_total']} typed isolates")
    top = (
        sero.group_by("rq5_regime")
        .agg(pl.col("serogroup").first().alias("modal"),
             pl.col("share").first().alias("share"))
        .sort("rq5_regime")
    )
    print(top)

    # -- 3. sewer coverage by health region ----------------------------------
    # Already present in the health-region atlas; confirmed here rather than
    # recomputed, so the figures and the atlas cannot disagree.
    assert "sanitation_sewer_share_mean" in region_atlas.columns
    report["sewer_share_available"] = True
    report["sewer_share_null"] = int(
        region_atlas["sanitation_sewer_share_mean"].null_count()
    )

    # -- 4. paired SINAN / SIH by health region ------------------------------
    paired = hr.select(
        "health_region_code", "health_region_name", "uf_abbr", "region",
        "cases", "sih_a27_admissions_total", "person_years",
        "incidence_per_100k", "sinan_per_independent_signal",
    ).filter((pl.col("cases") > 0) & (pl.col("sih_a27_admissions_total") > 0))
    paired = paired.with_columns(
        (pl.col("cases").log() - pl.col("sih_a27_admissions_total").log())
        .alias("log_sinan_over_sih")
    )
    # Two correlations, and only the second is evidence. Correlating log counts
    # across regions whose populations span three orders of magnitude is mostly
    # a measurement of population: both counts scale with it, so a high r would
    # arise even if the two systems agreed about nothing else. The quantity the
    # replication argument needs is agreement in RATES, after the shared
    # population scale is divided out. Both are stored, labelled, so that the
    # larger and emptier number cannot be quoted by accident.
    paired = paired.with_columns(
        (pl.col("cases") / pl.col("person_years")).log().alias("log_sinan_rate"),
        (pl.col("sih_a27_admissions_total") / pl.col("person_years"))
        .log().alias("log_sih_rate"),
    )
    paired.write_parquet(OUT / "sinan_sih_paired_health_region.parquet")
    corr_counts = float(
        paired.select(
            pl.corr(pl.col("cases").log(), pl.col("sih_a27_admissions_total").log())
        ).item()
    )
    corr_rates = float(
        paired.select(pl.corr("log_sinan_rate", "log_sih_rate")).item()
    )
    report["paired_health_regions"] = paired.height
    report["log_count_correlation_confounded_by_population"] = corr_counts
    report["log_rate_correlation"] = corr_rates
    print(f"\npaired SINAN/SIH: {paired.height} health regions")
    print(f"  log-count r = {corr_counts:.3f}  <- mostly population scale, "
          "not evidence")
    print(f"  log-RATE  r = {corr_rates:.3f}  <- the replication quantity")

    (OUT / "figure_inputs_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
