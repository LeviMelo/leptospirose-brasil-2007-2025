"""Stage 0 - descriptive epidemiology of leptospirosis in Brazil, 2007-2025.

This is a standalone output in its own right: the first national description of
the series that includes 2024 (the Rio Grande do Sul catastrophe) and 2025, and
the first to quantify the COVID surveillance trough and the flood spike on the
same national footing.

Everything here is generated. No number in the paper should be typed by hand.

Run:
    python studies/leptospirosis/04_descriptive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.denominators.rates import WHO_WORLD_STANDARD, age_standardised_rate
from brepi.sources.datasus import sinan

YEARS = range(2007, 2026)
OUT = PATHS.results / "01_descriptive"

REGION_ORDER = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]


def show(title: str, frame: pl.DataFrame, n: int = 30) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(frame.head(n).to_pandas().to_string(index=False))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
    print(f"panel {panel.height:,} x {panel.width}")

    # ---------------------------------------------------------------- 1 ----
    national = (
        panel.group_by("year")
        .agg(
            pl.col("cases").sum(),
            pl.col("deaths").sum(),
            pl.col("hospitalised").sum(),
            pl.col("criterion_epi").sum(),
            pl.col("flood_contact").sum(),
            pl.col("population").sum().alias("pop_months"),
        )
        .with_columns((pl.col("pop_months") / 12).alias("population"))
        .with_columns(
            (pl.col("cases") / pl.col("population") * 100_000).alias("incidence"),
            (pl.col("deaths") / pl.col("cases") * 100).alias("cfr_pct"),
            (pl.col("hospitalised") / pl.col("cases") * 100).alias("hosp_pct"),
            (pl.col("criterion_epi") / pl.col("cases") * 100).alias("crit_epi_pct"),
            (pl.col("deaths") / pl.col("population") * 1_000_000).alias("mortality_per_1M"),
        )
        .sort("year")
        .drop("pop_months")
    )
    show("1. NATIONAL SERIES, 2007-2025", national)
    national.write_csv(OUT / "01_national_series.csv")

    # ---------------------------------------------------------------- 2 ----
    regional = (
        panel.group_by("region", "year")
        .agg(pl.col("cases").sum(), pl.col("deaths").sum(),
             (pl.col("population").sum() / 12).alias("population"))
        .with_columns((pl.col("cases") / pl.col("population") * 100_000).alias("incidence"))
        .sort("region", "year")
    )
    reg_wide = (
        regional.pivot(on="region", index="year", values="incidence")
        .select(["year"] + [c for c in REGION_ORDER if c in regional["region"].unique()])
        .sort("year")
    )
    show("2. INCIDENCE PER 100,000 BY MACROREGION", reg_wide.with_columns(
        pl.exclude("year").round(2)))
    regional.write_csv(OUT / "02_regional_series.csv")

    # ---------------------------------------------------------------- 3 ----
    # COVID trough and 2024 spike, quantified against a pre-period expectation
    # rather than against a single adjacent year.
    base = national.filter(pl.col("year").is_between(2015, 2019))
    exp_inc = float(base["incidence"].mean())
    breaks = national.filter(pl.col("year") >= 2020).with_columns(
        (pl.col("incidence") / exp_inc - 1).alias("vs_2015_2019")
    ).select("year", "cases", "incidence", "vs_2015_2019", "cfr_pct", "crit_epi_pct")
    show(f"3. DEPARTURE FROM THE 2015-2019 MEAN ({exp_inc:.2f} per 100k)", breaks)
    breaks.write_csv(OUT / "03_breaks.csv")

    # ---------------------------------------------------------------- 4 ----
    seasonality = (
        panel.group_by("region", "month")
        .agg(pl.col("cases").sum())
        .with_columns(
            (pl.col("cases") / pl.col("cases").sum().over("region") * 100).alias("pct_of_annual")
        )
        .sort("region", "month")
    )
    seas_wide = seasonality.pivot(on="region", index="month", values="pct_of_annual").sort("month")
    show("4. SEASONALITY: % OF REGIONAL CASES BY CALENDAR MONTH",
         seas_wide.with_columns(pl.exclude("month").round(1)), n=12)
    seasonality.write_csv(OUT / "04_seasonality.csv")

    # Concentration of the season: how many months hold half the cases.
    conc = []
    for reg, g in seasonality.group_by("region"):
        v = np.sort(g["pct_of_annual"].to_numpy())[::-1]
        conc.append({"region": reg[0], "months_for_50pct": int(np.searchsorted(np.cumsum(v), 50) + 1),
                     "peak_month": int(g.sort("pct_of_annual", descending=True)["month"][0])})
    show("4b. SEASONAL CONCENTRATION", pl.DataFrame(conc).sort("months_for_50pct"))

    # ---------------------------------------------------------------- 5 ----
    mun = (
        panel.group_by("munic_code", "name", "uf_abbr", "region")
        .agg(pl.col("cases").sum(), pl.col("deaths").sum(),
             (pl.col("population").sum() / 12 / 19).alias("mean_population"),
             pl.col("any_case").sum().alias("active_months"))
        .with_columns(
            (pl.col("cases") / (pl.col("mean_population") * 19) * 100_000).alias("annual_incidence")
        )
    )
    show("5. HIGHEST-BURDEN MUNICIPALITIES BY TOTAL CONFIRMED CASES",
         mun.sort("cases", descending=True)
            .select("name", "uf_abbr", "cases", "deaths", "annual_incidence", "active_months")
            .with_columns(pl.col("annual_incidence").round(1)), n=20)
    show("5b. HIGHEST-INCIDENCE MUNICIPALITIES (>=20 cases, per 100k/yr)",
         mun.filter(pl.col("cases") >= 20).sort("annual_incidence", descending=True)
            .select("name", "uf_abbr", "cases", "annual_incidence", "mean_population")
            .with_columns(pl.col("annual_incidence").round(1),
                          pl.col("mean_population").round(0)), n=20)
    mun.write_csv(OUT / "05_municipal_burden.csv")

    # ---------------------------------------------------------------- 6 ----
    # Silent municipalities: the RQ3 population. Characterised, not just counted.
    silent = mun.with_columns((pl.col("cases") == 0).alias("silent"))
    sil = (
        silent.group_by("region", "silent")
        .agg(pl.len().alias("n"), pl.col("mean_population").sum().alias("pop"))
        .pivot(on="silent", index="region", values="n")
        .rename({"false": "reporting", "true": "silent"})
    )
    show("6. SILENT MUNICIPALITIES BY REGION", sil)

    popbands = silent.with_columns(
        pl.when(pl.col("mean_population") < 5_000).then(pl.lit("1 <5k"))
        .when(pl.col("mean_population") < 20_000).then(pl.lit("2 5-20k"))
        .when(pl.col("mean_population") < 100_000).then(pl.lit("3 20-100k"))
        .when(pl.col("mean_population") < 500_000).then(pl.lit("4 100-500k"))
        .otherwise(pl.lit("5 500k+")).alias("size_band")
    ).group_by("size_band").agg(
        pl.len().alias("municipalities"),
        pl.col("silent").sum().alias("silent"),
        pl.col("cases").sum().alias("cases"),
    ).with_columns((pl.col("silent") / pl.col("municipalities") * 100).round(1).alias("pct_silent")
    ).sort("size_band")
    show("6b. SILENCE BY POPULATION SIZE - the detection gradient", popbands)
    popbands.write_csv(OUT / "06_silence_by_size.csv")

    # ---------------------------------------------------------------- 7 ----
    print("\nloading SINAN line list for the demographic profile (cached)")
    ll, _ = sinan.fetch_range("LEPT", YEARS)
    conf = ll.filter(pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1")
    conf = conf.with_columns(
        decode_age(pl.col("NU_IDADE_N")).alias("age_years"),
        # Canonical sex vocabulary is female/male, matching the population
        # tensor. The tensor is the denominator, so it is the table every
        # numerator must be joinable to; emitting the raw SINAN F/M here left
        # this file unable to join to the age-standardised rates at all.
        pl.col("CS_SEXO").cast(pl.Utf8).str.strip_chars()
          .replace_strict({"M": "male", "F": "female"}, default=None)
          .alias("sex"),
        pl.col("CS_RACA").cast(pl.Utf8).str.strip_chars().alias("race"),
        pl.col("EVOLUCAO").cast(pl.Utf8).str.strip_chars().alias("evolucao"),
    ).with_columns(age_band(pl.col("age_years")).alias("age_group"))

    demo = (
        conf.filter(pl.col("age_group").is_not_null() & pl.col("sex").is_in(["male", "female"]))
        .group_by("sex", "age_group")
        .agg(pl.len().alias("cases"),
             (pl.col("evolucao") == "2").sum().alias("deaths"))
        .with_columns((pl.col("deaths") / pl.col("cases") * 100).round(2).alias("cfr_pct"))
        .sort("age_group", "sex")
    )
    show("7. CASES AND CASE FATALITY BY AGE AND SEX", demo, n=40)
    demo.write_csv(OUT / "07_age_sex_profile.csv")

    male = conf.filter(pl.col("sex") == "male").height
    print(f"\n   male share: {100 * male / conf.filter(pl.col('sex').is_in(['male','female'])).height:.1f}%")
    print(f"   median age: {conf['age_years'].median():.0f} years")
    print(f"   working-age (20-59) share: "
          f"{100 * conf.filter(pl.col('age_years').is_between(20, 59)).height / conf.height:.1f}%")

    # ---------------------------------------------------------------- 8 ----
    pop_strat = pl.read_parquet(PATHS.interim / "population_tensor_long.parquet")
    num = (
        conf.with_columns(
            pl.col("_src_year").cast(pl.Int32).alias("year"),
        )
        .filter(pl.col("age_group").is_not_null() & pl.col("sex").is_in(["male", "female"]))
        .join(
            pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
            .select("munic_code", "uf_abbr").unique(),
            left_on=pl.col("SG_UF").cast(pl.Utf8), right_on="uf_abbr", how="left",
        )
    )
    asr_national = age_standardised_rate(
        conf.filter(pl.col("age_group").is_not_null())
            .with_columns(pl.col("_src_year").cast(pl.Int32).alias("year"))
            .group_by("year", "age_group").agg(pl.len().alias("cases")),
        pop_strat.group_by("year", "age_group").agg(pl.col("population").sum()),
        by=["year"],
        standard=WHO_WORLD_STANDARD,
    )
    show("8. AGE-STANDARDISED INCIDENCE (WHO world standard, per 100,000)",
         asr_national.select("year", "cases", "asr", "asr_lo", "asr_hi")
                     .with_columns(pl.col("asr", "asr_lo", "asr_hi").round(2)))
    asr_national.write_csv(OUT / "08_age_standardised.csv")

    print(f"\nwrote tables to {OUT}")
    return 0


def decode_age(col: pl.Expr) -> pl.Expr:
    """Decode SINAN ``NU_IDADE_N`` to completed years.

    The field packs a unit prefix into the leading digit: 1 = hours,
    2 = days, 3 = months, 4 = years. Reading it as an integer turns a
    3-month-old into a 3-year-old and a 4-year-old into 4004. This is the
    single most common silent error in SINAN demographic tabulations.
    """
    s = col.cast(pl.Utf8).str.strip_chars().str.zfill(4)
    unit = s.str.slice(0, 1)
    value = s.str.slice(1, 3).cast(pl.Int32, strict=False)
    return (
        pl.when(unit == "4").then(value)
        .when(unit.is_in(["1", "2", "3"])).then(pl.lit(0))
        .otherwise(None)
    )


def age_band(age: pl.Expr) -> pl.Expr:
    """Five-year bands matching the denominator tensor's scheme."""
    expr = pl.when(age >= 80).then(pl.lit("80+"))
    for lo in range(75, -1, -5):
        expr = expr.when(age >= lo).then(pl.lit(f"{lo:02d}-{lo + 4:02d}"))
    return expr.otherwise(None)


if __name__ == "__main__":
    raise SystemExit(main())
