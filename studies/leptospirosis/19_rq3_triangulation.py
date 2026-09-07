"""RQ3 - the three-system cross-tabulation and the detection-silent municipalities.

This is the part of RQ3 that does not depend on a model.

A municipality that has never notified a confirmed leptospirosis case might
have no leptospirosis (transmission-silent) or no surveillance
(detection-silent). Those two are observationally identical inside SINAN, and
no amount of modelling of SINAN alone can separate them. They are *not*
identical once a second, independently-generated system is brought in: a death
certificate coded A27 and a hospital admission billed as A27 are both produced
without any notification ever being filed. A municipality with A27 deaths or
A27 admissions and zero SINAN confirmations is therefore demonstrably
detection-silent -- the disease was in front of the health system, was
recognised well enough to be written on a death certificate or an invoice, and
still did not become a notification.

Everything here is computed inside the window where the relevant systems
actually exist (see ``sim_covered`` / ``sih_covered`` in the panel). The
three-way table is restricted to the intersection window; the SINAN-vs-SIM
table uses the longer mortality window and says so.

Run:
    python studies/leptospirosis/19_rq3_triangulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS

OUT = PATHS.results / "rq3_ascertainment"
PANEL = PATHS.panel / "triangulation_municipality_year.parquet"


def show(title: str, frame: pl.DataFrame, n: int = 40) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(frame.head(n).to_pandas().to_string(index=False))


def pop_band(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col < 5_000).then(pl.lit("1 <5k"))
        .when(col < 20_000).then(pl.lit("2 5-20k"))
        .when(col < 100_000).then(pl.lit("3 20-100k"))
        .otherwise(pl.lit("4 100k+"))
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PANEL.exists():
        raise SystemExit(f"{PANEL} absent. Run 18_build_triangulation_panel.py first.")
    tri = pl.read_parquet(PANEL)
    results: dict[str, object] = {}

    sim_years = sorted(tri.filter(pl.col("sim_covered"))["year"].unique().to_list())
    sih_years = sorted(tri.filter(pl.col("sih_covered"))["year"].unique().to_list())
    has_sih = bool(sih_years)
    both = sorted(set(sim_years) & set(sih_years)) if has_sih else []
    # SIH absence is a legitimate state -- the RD series starts in 2008 and the
    # extraction is windowed -- so it must print, not raise. Formatting the
    # bound before checking `has_sih` crashed on an empty sequence and made a
    # reportable data condition look like a code failure.
    sih_span = f"{min(sih_years)}-{max(sih_years)}" if has_sih else "(absent)"
    print(f"coverage  SINAN {tri['year'].min()}-{tri['year'].max()} | "
          f"SIM {min(sim_years)}-{max(sim_years)} | "
          f"SIH {sih_span}")
    results["windows"] = {
        "sinan": [int(tri["year"].min()), int(tri["year"].max())],
        "sim": [int(min(sim_years)), int(max(sim_years))],
        "sih": [int(min(sih_years)), int(max(sih_years))] if has_sih else None,
        "three_system_intersection": (
            [int(min(both)), int(max(both))] if both else None
        ),
    }

    # ------------------------------------------------------------------ 1 ---
    # Municipality-level collapse inside each system's own window. Aggregating
    # a system over years it did not publish would understate it; aggregating
    # SINAN over the same restricted window keeps the comparison fair.
    def collapse(years: list[int], cols: dict[str, str]) -> pl.DataFrame:
        sub = tri.filter(pl.col("year").is_in(years))
        return (
            sub.group_by("munic_code", "name", "uf_abbr", "region")
            .agg(
                *[pl.col(src).sum().alias(dst) for dst, src in cols.items()],
                pl.col("population").mean().alias("pop"),
            )
        )

    # --- SINAN vs SIM, on the mortality window ---
    ms = collapse(sim_years, {
        "sinan_confirmed": "sinan_confirmed",
        "sinan_deaths": "sinan_deaths",
        "sim_a27_deaths": "sim_a27_deaths",
    }).with_columns(
        (pl.col("sinan_confirmed") > 0).alias("sinan"),
        (pl.col("sim_a27_deaths") > 0).alias("sim"),
    )
    cross2 = (
        ms.group_by("sinan", "sim")
        .agg(pl.len().alias("municipalities"),
             pl.col("pop").sum().round(0).alias("population"),
             pl.col("sim_a27_deaths").sum().alias("sim_deaths"),
             pl.col("sinan_confirmed").sum().alias("sinan_cases"))
        .sort("sinan", "sim")
    )
    show(f"1. SINAN CONFIRMED vs SIM A27 DEATH, municipality level, "
         f"{min(sim_years)}-{max(sim_years)}", cross2)
    cross2.write_csv(OUT / "01_cross_sinan_sim.csv")

    silent_sim = ms.filter(~pl.col("sinan") & pl.col("sim"))
    results["sinan_vs_sim"] = {
        "window": [int(min(sim_years)), int(max(sim_years))],
        "municipalities_sim_positive_sinan_zero": silent_sim.height,
        "deaths_in_them": int(silent_sim["sim_a27_deaths"].sum()),
        "population_in_them": int(silent_sim["pop"].sum()),
    }
    print(f"\n   SIM-positive but SINAN-silent: {silent_sim.height} municipalities, "
          f"{int(silent_sim['sim_a27_deaths'].sum())} A27 deaths, "
          f"{int(silent_sim['pop'].sum()):,} residents")

    # ------------------------------------------------------------------ 2 ---
    # The full three-system table, on the intersection window.
    if not has_sih:
        print("\n[2] three-system table SKIPPED: no SIH extraction present. "
              "Every SIH-derived number below is unavailable, not zero.")
        results["three_system"] = None
    else:
        m3 = collapse(both, {
            "sinan_confirmed": "sinan_confirmed",
            "sinan_deaths": "sinan_deaths",
            "sim_a27_deaths": "sim_a27_deaths",
            "sih_a27_admissions": "sih_a27_admissions",
            "sih_a27_principal": "sih_a27_principal",
        }).with_columns(
            (pl.col("sinan_confirmed") > 0).alias("sinan"),
            (pl.col("sim_a27_deaths") > 0).alias("sim"),
            (pl.col("sih_a27_admissions") > 0).alias("sih"),
        )
        cross3 = (
            m3.group_by("sinan", "sih", "sim")
            .agg(pl.len().alias("municipalities"),
                 pl.col("pop").sum().round(0).alias("population"),
                 pl.col("sinan_confirmed").sum().alias("sinan_cases"),
                 pl.col("sih_a27_admissions").sum().alias("sih_admissions"),
                 pl.col("sim_a27_deaths").sum().alias("sim_deaths"))
            .sort("sinan", "sih", "sim", descending=[True, True, True])
        )
        show(f"2. THREE-SYSTEM CROSS-TABULATION, municipality level, "
             f"{min(both)}-{max(both)}", cross3)
        cross3.write_csv(OUT / "02_cross_three_system.csv")

        detection_silent = m3.filter(
            ~pl.col("sinan") & (pl.col("sim") | pl.col("sih"))
        )
        never_seen = m3.filter(~pl.col("sinan") & ~pl.col("sim") & ~pl.col("sih"))
        results["three_system"] = {
            "window": [int(min(both)), int(max(both))],
            "municipalities": m3.height,
            "detection_silent": {
                "municipalities": detection_silent.height,
                "population": int(detection_silent["pop"].sum()),
                "sim_deaths": int(detection_silent["sim_a27_deaths"].sum()),
                "sih_admissions": int(detection_silent["sih_a27_admissions"].sum()),
                "sih_principal": int(detection_silent["sih_a27_principal"].sum()),
            },
            "absent_from_all_three": {
                "municipalities": never_seen.height,
                "population": int(never_seen["pop"].sum()),
            },
        }
        print(f"\n   CROSS-SYSTEM DISCORDANT (zero SINAN, but A27 deaths and/or admissions): "
              f"{detection_silent.height} municipalities")
        print(f"      population: {int(detection_silent['pop'].sum()):,}")
        print(f"      A27 deaths there: {int(detection_silent['sim_a27_deaths'].sum())}")
        print(f"      A27 admissions there: "
              f"{int(detection_silent['sih_a27_admissions'].sum())}")
        print(f"   absent from ALL THREE systems: {never_seen.height} municipalities, "
              f"{int(never_seen['pop'].sum()):,} residents")
        detection_silent.sort("sih_a27_admissions", descending=True).write_csv(
            OUT / "03_detection_silent_municipalities.csv"
        )
        show("   the twenty largest detection-silent municipalities",
             detection_silent.sort(
                 ["sih_a27_admissions", "sim_a27_deaths"], descending=True
             ).select("name", "uf_abbr", "pop", "sim_a27_deaths",
                      "sih_a27_admissions").with_columns(pl.col("pop").round(0)),
             n=20)

        # By population band: does the detection deficit follow size?
        band = (
            m3.with_columns(pop_band(pl.col("pop")).alias("pop_band"))
            .group_by("pop_band")
            .agg(
                pl.len().alias("municipalities"),
                (~pl.col("sinan")).sum().alias("sinan_silent"),
                (~pl.col("sinan") & (pl.col("sim") | pl.col("sih"))).sum()
                .alias("detection_silent"),
                (~pl.col("sinan") & ~pl.col("sim") & ~pl.col("sih")).sum()
                .alias("silent_in_all_three"),
            )
            .with_columns(
                (pl.col("sinan_silent") / pl.col("municipalities")).round(3)
                .alias("pct_sinan_silent"),
                (pl.col("detection_silent") / pl.col("sinan_silent")).round(3)
                .alias("share_of_silence_proven_detection"),
            )
            .sort("pop_band")
        )
        show("2b. SILENCE BY POPULATION BAND, and how much of it the other two "
             "systems can already falsify", band)
        band.write_csv(OUT / "04_silence_by_population_band.csv")

    # ------------------------------------------------------------------ 3 ---
    # National annual series of the three systems, so the reader can see that
    # they move independently.
    ann = (
        tri.group_by("year")
        .agg(
            pl.col("sinan_confirmed").sum(),
            pl.col("sinan_deaths").sum(),
            pl.col("sim_a27_deaths").sum(),
            pl.col("sih_a27_admissions").sum(),
            pl.col("sih_a27_principal").sum(),
        )
        .sort("year")
        .with_columns(
            (pl.col("sim_a27_deaths") / pl.col("sinan_deaths")).round(3)
            .alias("sim_over_sinan_deaths"),
            (pl.col("sih_a27_admissions") / pl.col("sinan_confirmed")).round(3)
            .alias("sih_over_sinan_cases"),
        )
    )
    show("3. NATIONAL ANNUAL SERIES OF THE THREE SYSTEMS "
         "(nulls are years the system does not publish, not zeros)", ann)
    ann.write_csv(OUT / "05_national_annual_series.csv")

    # ------------------------------------------------------------------ 4 ---
    # State-level version of the decisive comparison.
    state = (
        tri.filter(pl.col("year").is_in(both if has_sih else sim_years))
        .group_by("munic_code", "uf_abbr")
        .agg(pl.col("sinan_confirmed").sum(), pl.col("sinan_deaths").sum(),
             pl.col("sim_a27_deaths").sum(),
             pl.col("sih_a27_admissions").sum() if has_sih
             else pl.lit(None).alias("sih_a27_admissions"),
             pl.col("population").mean().alias("pop"))
        .group_by("uf_abbr")
        .agg(
            pl.len().alias("municipalities"),
            (pl.col("sinan_confirmed") == 0).sum().alias("sinan_silent"),
            ((pl.col("sinan_confirmed") == 0) &
             ((pl.col("sim_a27_deaths") > 0) |
              (pl.col("sih_a27_admissions").fill_null(0) > 0)))
            .sum().alias("detection_silent"),
            pl.col("sinan_confirmed").sum(),
            pl.col("sinan_deaths").sum(),
            pl.col("sim_a27_deaths").sum(),
            pl.col("sih_a27_admissions").sum().alias("sih_a27_admissions"),
            pl.col("pop").sum(),
        )
        .with_columns(
            (pl.col("sim_a27_deaths") / pl.col("sinan_deaths")).round(3)
            .alias("sim_over_sinan_deaths"),
            (pl.col("detection_silent") / pl.col("sinan_silent")).round(3)
            .alias("share_of_silence_falsified"),
        )
        .sort("detection_silent", descending=True)
    )
    show("4. BY STATE: silent municipalities and how many of them the other "
         "systems falsify", state)
    state.write_csv(OUT / "06_by_state.csv")

    (OUT / "07_triangulation_summary.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote tables to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
