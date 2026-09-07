"""SUPERSEDED Paper A exploratory threads (retained for provenance only).

This module used the pre-audit mixed-geography panel and includes a
mathematically coupled incidence--fatality exploration. It is not a current
result producer. Use ``35_paperA_audit.py`` and
``docs/protocol/PAPER_A_SCIENTIFIC_SPEC.md`` instead.

An argument that states a finding and stops is not an argument. Each block here
follows one thread in ``docs/protocol/ARGUMENTO_A.md`` to the point where it
either closes or is honestly declared open.

1. **Who are the 413?** A count is not a finding. Where are they, how large, in
   which states, and how do they differ from the corroborated municipalities on
   the covariates that would explain them away?
2. **Is the silence shrinking?** If notification has improved over nineteen
   years, the classification is a snapshot of a moving system and must say so.
3. **Does the case-fatality inversion survive?** It is the strongest link, so it
   gets the hardest tests: within macro-region, within year, and restricted to
   laboratory-confirmed cases -- because if the gradient is really clinical
   versus laboratory confirmation, it is a different finding with a different
   name.
4. **Santa Catarina.** It carries the country's highest incidence and, if the
   thesis holds, should carry among its lowest case fatality -- not because the
   organism is milder but because mild cases are found there. A thesis that
   predicts a specific place's profile and gets it right is worth more than one
   that only fits the average.

Writes to ``data/results/paperA/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "paperA"
SILENT = "DETECTION-SILENT (proven)"
CORROB = "corroborated"


def _cfr(counts: np.ndarray, totals: np.ndarray):
    # binom_ci returns (proportion, lower, upper) -- estimate FIRST. Unpacking
    # it as (lo, mid, hi) puts the lower bound in the point-estimate column and
    # renders every interval nonsensical without raising.
    est, lo, hi = binom_ci(counts, totals)
    return 100 * est, 100 * lo, 100 * hi


def profile_of_the_silent(atlas: pl.DataFrame) -> dict:
    """What distinguishes the 413 from the municipalities that corroborate."""
    out: dict = {}
    cls = atlas.group_by("surveillance_class").agg(
        pl.len().alias("municipalities"),
        pl.col("mean_population").sum().alias("population"),
        pl.col("mean_population").median().alias("median_population"),
        pl.col("sanitation_sewer_share_mean").median().alias("median_sewer"),
        pl.col("gdp_per_capita_mean").median().alias("median_gdp"),
        pl.col("urban_share_mean").median().alias("median_urban"),
    ).sort("municipalities", descending=True)
    out["by_class"] = cls.to_dicts()

    silent = atlas.filter(pl.col("surveillance_class") == SILENT)
    corr = atlas.filter(pl.col("surveillance_class") == CORROB)

    # Where they are. Concentration matters: 413 spread evenly is a national
    # property, 413 in four states is a state-capacity story.
    by_uf = (
        silent.group_by("uf_abbr").agg(pl.len().alias("silent"))
        .join(atlas.group_by("uf_abbr").agg(pl.len().alias("total")), on="uf_abbr")
        .with_columns((pl.col("silent") / pl.col("total")).alias("share_of_uf"))
        .sort("silent", descending=True)
    )
    out["by_uf"] = by_uf.to_dicts()
    top5 = by_uf.head(5)
    out["top5_uf_share_of_all_silent"] = float(top5["silent"].sum() / silent.height)

    by_region = (
        silent.group_by("region").agg(pl.len().alias("silent"))
        .join(atlas.group_by("region").agg(pl.len().alias("total")), on="region")
        .with_columns((pl.col("silent") / pl.col("total")).alias("share_of_region"))
        .sort("silent", descending=True)
    )
    out["by_region"] = by_region.to_dicts()

    # The covariate that would explain them away: are they simply poorer and
    # less served? If so, detection silence is a proxy for deprivation and not
    # an independent finding.
    out["median_sewer_silent"] = float(silent["sanitation_sewer_share_mean"].median())
    out["median_sewer_corroborated"] = float(corr["sanitation_sewer_share_mean"].median())
    out["median_gdp_silent"] = float(silent["gdp_per_capita_mean"].median())
    out["median_gdp_corroborated"] = float(corr["gdp_per_capita_mean"].median())
    out["median_pop_silent"] = float(silent["mean_population"].median())
    out["median_pop_corroborated"] = float(corr["mean_population"].median())
    return out


def temporal(panel: pl.DataFrame) -> pl.DataFrame:
    """National incidence and case fatality by year, with exact intervals."""
    annual = panel.group_by("year").agg(
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("person_months").sum().alias("person_months"),
        (pl.col("cases") > 0).sum().alias("municipality_months_with_cases"),
    ).sort("year")
    py = annual["person_months"].to_numpy() / 12
    inc, ilo, ihi = poisson_ci(annual["cases"].to_numpy(), py, scale=1e5)
    cfr, clo, chi = _cfr(annual["deaths"].to_numpy(), annual["cases"].to_numpy())
    return annual.with_columns(
        pl.Series("person_years", py),
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", ilo), pl.Series("incidence_hi", ihi),
        pl.Series("cfr_pct", cfr),
        pl.Series("cfr_lo", clo), pl.Series("cfr_hi", chi),
    )


def notifying_municipalities(panel: pl.DataFrame) -> pl.DataFrame:
    """How many municipalities notify at least one case each year.

    The classification in Paper A is cumulative over nineteen years. If the
    number notifying has risen steeply, the 413 are a residue of a system that
    is improving, and the paper must frame them as such.
    """
    return panel.group_by("year").agg(
        (pl.col("cases").sum().over("munic_code") > 0).sum().alias("_ignore"),
    ).sort("year")


def cfr_by_incidence(atlas: pl.DataFrame, *, within: str | None = None) -> pl.DataFrame:
    """Case fatality by municipal incidence band, optionally within a stratum.

    Bands follow the published cut points. Aggregation sums deaths and cases and
    divides once -- averaging municipal case fatalities would let a
    three-case municipality outvote a capital.
    """
    banded = atlas.filter(pl.col("cases") > 0).with_columns(
        pl.when(pl.col("incidence_per_100k") < 1).then(pl.lit("1 <1"))
        .when(pl.col("incidence_per_100k") < 3).then(pl.lit("2 1-3"))
        .when(pl.col("incidence_per_100k") < 10).then(pl.lit("3 3-10"))
        .when(pl.col("incidence_per_100k") < 30).then(pl.lit("4 10-30"))
        .otherwise(pl.lit("5 30+")).alias("band")
    )
    keys = ["band"] if within is None else [within, "band"]
    agg = banded.group_by(keys).agg(
        pl.len().alias("municipalities"),
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
    ).sort(keys)
    cfr, lo, hi = _cfr(agg["deaths"].to_numpy(), agg["cases"].to_numpy())
    return agg.with_columns(
        pl.Series("cfr_pct", cfr), pl.Series("cfr_lo", lo), pl.Series("cfr_hi", hi)
    )


def cfr_by_incidence_laboratory(atlas: pl.DataFrame) -> pl.DataFrame:
    """The inversion restricted to laboratory-confirmed cases.

    ``criterion_epi`` counts cases confirmed on epidemiological criteria alone.
    If the gradient is really that low-incidence places confirm clinically and
    high-incidence places confirm by laboratory, then removing the clinical
    cases should flatten it. If it survives, that reading is closed.
    """
    lab = atlas.filter(pl.col("cases") > 0).with_columns(
        (pl.col("cases") - pl.col("criterion_epi").fill_null(0)).alias("lab_cases")
    ).filter(pl.col("lab_cases") > 0)
    banded = lab.with_columns(
        pl.when(pl.col("incidence_per_100k") < 1).then(pl.lit("1 <1"))
        .when(pl.col("incidence_per_100k") < 3).then(pl.lit("2 1-3"))
        .when(pl.col("incidence_per_100k") < 10).then(pl.lit("3 3-10"))
        .when(pl.col("incidence_per_100k") < 30).then(pl.lit("4 10-30"))
        .otherwise(pl.lit("5 30+")).alias("band")
    )
    agg = banded.group_by("band").agg(
        pl.len().alias("municipalities"),
        pl.col("lab_cases").sum().alias("lab_cases"),
        pl.col("cases").sum().alias("all_cases"),
        pl.col("deaths").sum().alias("deaths"),
        (pl.col("criterion_epi").sum() / pl.col("cases").sum()).alias("epi_share"),
    ).sort("band")
    # Deaths are not split by criterion in the panel, so this is an upper bound
    # on laboratory case fatality: all deaths over laboratory cases only.
    cfr, lo, hi = _cfr(agg["deaths"].to_numpy(), agg["lab_cases"].to_numpy())
    return agg.with_columns(
        pl.Series("cfr_upper_bound_pct", cfr),
        pl.Series("cfr_ub_lo", lo), pl.Series("cfr_ub_hi", hi),
    )


def state_profiles(atlas: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """State-level incidence, case fatality and detection composition."""
    agg = atlas.group_by("uf_abbr").agg(
        pl.len().alias("municipalities"),
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("person_years").sum().alias("person_years"),
        (pl.col("surveillance_class") == SILENT).sum().alias("detection_silent"),
        (pl.col("surveillance_class") == CORROB).sum().alias("corroborated"),
        pl.col("criterion_epi").sum().alias("criterion_epi"),
        pl.col("sanitation_sewer_share_mean").median().alias("median_sewer"),
        pl.col("region").first().alias("region"),
    )
    inc, ilo, ihi = poisson_ci(
        agg["cases"].to_numpy(), agg["person_years"].to_numpy(), scale=1e5
    )
    cfr, clo, chi = _cfr(agg["deaths"].to_numpy(), agg["cases"].to_numpy())
    return agg.with_columns(
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", ilo), pl.Series("incidence_hi", ihi),
        pl.Series("cfr_pct", cfr), pl.Series("cfr_lo", clo), pl.Series("cfr_hi", chi),
        (pl.col("criterion_epi") / pl.col("cases")).alias("epi_criterion_share"),
        (pl.col("detection_silent") / pl.col("municipalities")).alias("share_silent"),
    ).sort("incidence_per_100k", descending=True)


def outcome_completeness(atlas: pl.DataFrame, region_atlas: pl.DataFrame) -> pl.DataFrame:
    """State case fatality, naive and restricted to cases with known outcome.

    This closes the thread that nearly sank the argument. At state level the
    case-fatality inversion vanishes (Spearman +0.09), and Piaui reports a case
    fatality near zero -- which is not an epidemiological finding, because
    leptospirosis does not have a case fatality near zero anywhere.

    The resolution is that the notification system fails in two ways that push
    case fatality in OPPOSITE directions:

    * mild cases never detected  -> case fatality inflated
    * outcome field never filled -> deaths become 'unknown', case fatality
      deflated toward zero

    Where both operate the second masks the first. So the naive ratio
    deaths/cases is uninterpretable across states, and the quantity that means
    anything is deaths over cases *with a recorded outcome*.
    """
    known = region_atlas.group_by("uf_abbr").agg(
        pl.col("rq4_outcome_known").sum().alias("outcome_known"),
        pl.col("cases").sum().alias("region_cases"),
    )
    st = atlas.group_by("uf_abbr").agg(
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("person_years").sum().alias("person_years"),
        pl.col("region").first().alias("region"),
    ).join(known, on="uf_abbr", how="left")
    inc, ilo, ihi = poisson_ci(
        st["cases"].to_numpy(), st["person_years"].to_numpy(), scale=1e5
    )
    naive, nlo, nhi = _cfr(st["deaths"].to_numpy(), st["cases"].to_numpy())
    adj, alo, ahi = _cfr(
        st["deaths"].to_numpy(), st["outcome_known"].to_numpy().astype(float)
    )
    return st.with_columns(
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", ilo), pl.Series("incidence_hi", ihi),
        pl.Series("cfr_naive_pct", naive),
        pl.Series("cfr_known_pct", adj),
        pl.Series("cfr_known_lo", alo), pl.Series("cfr_known_hi", ahi),
        (pl.col("outcome_known") / pl.col("region_cases")).alias("outcome_share"),
    ).sort("incidence_per_100k")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
    report: dict = {}

    # -- 1. who the 413 are --------------------------------------------------
    prof = profile_of_the_silent(atlas)
    report["silent_profile"] = prof
    print("=== the 413 ===")
    print(f"  top 5 states hold {prof['top5_uf_share_of_all_silent']:.1%} of them")
    for r in prof["by_uf"][:6]:
        print(f"    {r['uf_abbr']}: {r['silent']} of {r['total']} "
              f"({r['share_of_uf']:.1%} of the state)")
    print(f"  median population   silent {prof['median_pop_silent']:,.0f} vs "
          f"corroborated {prof['median_pop_corroborated']:,.0f}")
    print(f"  median sewer share  silent {prof['median_sewer_silent']:.3f} vs "
          f"corroborated {prof['median_sewer_corroborated']:.3f}")
    print(f"  median GDP p.c.     silent {prof['median_gdp_silent']:,.0f} vs "
          f"corroborated {prof['median_gdp_corroborated']:,.0f}")

    # -- 2. is it shrinking? -------------------------------------------------
    annual = temporal(panel)
    annual.write_parquet(OUT / "national_annual.parquet")
    notif = (
        panel.group_by(["year", "munic_code"]).agg(pl.col("cases").sum().alias("c"))
        .filter(pl.col("c") > 0).group_by("year").agg(pl.len().alias("notifying"))
        .sort("year")
    )
    notif.write_parquet(OUT / "notifying_municipalities.parquet")
    print("\n=== trend ===")
    j = annual.join(notif, on="year")
    for r in j.iter_rows(named=True):
        if r["year"] in (2007, 2011, 2015, 2019, 2020, 2023, 2025):
            print(f"  {r['year']}: incidence {r['incidence_per_100k']:.2f}, "
                  f"CFR {r['cfr_pct']:.1f}%, {r['notifying']} municipalities notifying")
    first, last = notif["notifying"][0], notif["notifying"][-1]
    report["notifying_first_year"] = int(first)
    report["notifying_last_year"] = int(last)
    report["notifying_peak"] = int(notif["notifying"].max())

    # -- 3. does the inversion survive? --------------------------------------
    overall = cfr_by_incidence(atlas)
    overall.write_parquet(OUT / "cfr_by_incidence.parquet")
    within_region = cfr_by_incidence(atlas, within="region")
    within_region.write_parquet(OUT / "cfr_by_incidence_region.parquet")
    lab = cfr_by_incidence_laboratory(atlas)
    lab.write_parquet(OUT / "cfr_by_incidence_laboratory.parquet")
    print("\n=== inversion, all cases ===")
    print(overall.select("band", "municipalities", "cases", "deaths",
                         "cfr_pct", "cfr_lo", "cfr_hi"))
    print("\n=== inversion within macro-region (CFR %) ===")
    pivot = within_region.pivot(on="band", index="region", values="cfr_pct")
    print(pivot)
    report["inversion_holds_in_regions"] = int(sum(
        1 for row in pivot.iter_rows(named=True)
        if (row.get("1 <1") or 0) > (row.get("5 30+") or 1e9)
    ))
    print("\n=== inversion, laboratory-confirmed only (upper bound) ===")
    print(lab.select("band", "lab_cases", "epi_share", "cfr_upper_bound_pct"))

    # -- 4. states, and Santa Catarina in particular -------------------------
    states = state_profiles(atlas, panel)
    states.write_parquet(OUT / "state_profiles.parquet")
    print("\n=== states by incidence ===")
    print(states.select("uf_abbr", "region", "incidence_per_100k", "cfr_pct",
                        "epi_criterion_share", "share_silent").head(8))
    print("...")
    print(states.select("uf_abbr", "region", "incidence_per_100k", "cfr_pct",
                        "epi_criterion_share", "share_silent").tail(6))
    sc = states.filter(pl.col("uf_abbr") == "SC").to_dicts()[0]
    report["santa_catarina"] = {
        k: (float(v) if isinstance(v, (int, float)) else v)
        for k, v in sc.items() if k != "region"
    }
    report["national_cfr"] = float(
        100 * atlas["deaths"].sum() / atlas["cases"].sum()
    )
    print(f"\nSanta Catarina: incidence {sc['incidence_per_100k']:.2f}/100k, "
          f"CFR {sc['cfr_pct']:.2f}% (national {report['national_cfr']:.2f}%), "
          f"{sc['detection_silent']} detection-silent of {sc['municipalities']}")

    # Rank correlation between state incidence and state case fatality: the
    # inversion as a single number at the level a health ministry acts on.
    rho = float(
        states.select(pl.corr("incidence_per_100k", "cfr_pct", method="spearman")).item()
    )
    report["state_incidence_cfr_spearman"] = rho
    print(f"state-level Spearman(incidence, CFR) = {rho:.3f}")

    # -- 5. the thread that nearly sank it: outcome completeness -------------
    region_atlas = pl.read_parquet(
        PATHS.results / "atlas" / "health_region_atlas.parquet"
    )
    oc = outcome_completeness(atlas, region_atlas)
    oc.write_parquet(OUT / "state_outcome_completeness.parquet")
    print("\n=== outcome completeness ===")
    print(oc.select("uf_abbr", "incidence_per_100k", "outcome_share",
                    "cfr_naive_pct", "cfr_known_pct").head(6))
    rho_naive = float(
        oc.select(pl.corr("incidence_per_100k", "cfr_naive_pct", method="spearman")).item()
    )
    rho_known = float(
        oc.select(pl.corr("incidence_per_100k", "cfr_known_pct", method="spearman")).item()
    )
    report["state_spearman_naive"] = rho_naive
    report["state_spearman_known_outcome"] = rho_known
    thresholds = {}
    for thr in (0.80, 0.90):
        g = oc.filter(pl.col("outcome_share") >= thr)
        thresholds[f"{thr:.2f}"] = {
            "states": g.height,
            "spearman_naive": float(
                g.select(pl.corr("incidence_per_100k", "cfr_naive_pct",
                                 method="spearman")).item()
            ),
        }
        print(f"  states with outcome recorded >= {thr:.0%}: n={g.height}, "
              f"Spearman(incidence, naive CFR) = "
              f"{thresholds[f'{thr:.2f}']['spearman_naive']:.3f}")
    report["state_spearman_by_outcome_threshold"] = thresholds
    report["outcome_share_vs_incidence_spearman"] = float(
        oc.select(pl.corr("outcome_share", "incidence_per_100k",
                          method="spearman")).item()
    )
    print(f"  all 27 states: naive {rho_naive:.3f}, known-outcome {rho_known:.3f}")
    print(f"  Spearman(outcome completeness, incidence) = "
          f"{report['outcome_share_vs_incidence_spearman']:.3f}")

    (OUT / "paperA_threads.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    raise RuntimeError(
        "34_paperA_threads.py is superseded and deliberately disabled; "
        "run 35_paperA_audit.py for the residence-aligned Paper A outputs."
    )
