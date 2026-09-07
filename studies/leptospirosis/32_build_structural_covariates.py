"""Structural covariates the four manuscripts need and the panel lacks.

Four blocks, each traceable to a specific claim in
``docs/protocol/PUBLICATION_STRATEGY.md`` section 10.3. Nothing is extracted
because it is available.

1. **Agriculture and livestock** (PAM 1612, PPM 3939). The only municipal
   covariates observed every year rather than at census anchors. Rice area is
   the flooded-field occupational route; cattle and swine density is the rural
   reservoir. Papers B and C.
2. **Favelas** (Censo 2022, tables 9883/9887/10344). The Reis and Hagan urban
   gradient, nationally measurable for the first time. Table 10344 gives sewage
   inside versus outside settlements in the same municipality, which is the
   within-municipality contrast that answers "that is not sanitation, it is
   poverty". Paper B.
3. **PNSB disease occurrence** (354). A municipal statement about leptospirosis
   made to a sanitation survey rather than to the health system, and therefore
   an ascertainment signal that inherits none of the assumptions behind
   SINAN x SIM x SIH. Paper A.
4. **PNSB service existence** (1238, 2000 and 2008). Not a new covariate -- a
   check on an existing one. Sewer coverage is anchored at 2010 and 2022 and
   back-extrapolated to 2007, and that extrapolation carries Paper B's central
   modifier. Where PNSB 2008 says a municipality has no sewer network at all,
   the extrapolated coverage fraction had better be near zero.

Writes to ``data/results/structural/``. Reconciliation figures are printed and
asserted, not merely logged: a silent join failure here would move a published
effect estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS
from brepi.sources.agriculture import (
    build_crop_area,
    build_livestock_density,
    crop_area_selection,
    livestock_selection,
)
from brepi.sources.favelas import (
    build_favela_profile,
    build_inside_outside_sewage,
    favela_count_selection,
    favela_population_selection,
    inside_outside_sewage_selection,
)
from brepi.sources.sanitation import (
    build_disease_flag,
    build_service_flags,
    compare_with_notifications,
    disease_occurrence_selection,
    service_existence_selection,
)
from brepi.sources.sidra.extract import Selection, extract

YEARS = range(2007, 2025)  # PAM/PPM final year is 2024
LIVESTOCK = ["cattle", "swine", "equine", "sheep", "goat"]
CROPS = ["rice", "sugarcane", "maize", "soy", "cassava"]

OUT = PATHS.results / "structural"


def _land_area(refresh: bool) -> pl.DataFrame:
    facts = extract(
        Selection(agregado=4714, periods=("2022",), variables=("6318",),
                  label="land_area_2022"),
        refresh=refresh,
    ).facts
    return facts.select(
        pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code"),
        pl.col("value_numeric").alias("area_km2"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
    spine = panel.select("munic_code").unique()
    report: dict[str, object] = {"spine_municipalities": spine.height}
    print(f"spine: {spine.height} municipalities")

    land = _land_area(args.refresh)
    assert land.height == 5570, f"land area covers {land.height}, expected 5570"

    # -- 1. agriculture ------------------------------------------------------
    herds = build_livestock_density(
        extract(livestock_selection(LIVESTOCK, YEARS), refresh=args.refresh).facts,
        land,
    )
    crops = build_crop_area(
        extract(crop_area_selection(CROPS, YEARS), refresh=args.refresh).facts,
        land,
    )
    agriculture = herds.join(crops.drop("area_ha"), on=["munic_code", "year"], how="full",
                             coalesce=True).sort(["munic_code", "year"])
    agriculture.write_parquet(OUT / "agriculture_municipality_year.parquet")
    report["agriculture_rows"] = agriculture.height
    report["agriculture_years"] = [
        int(agriculture["year"].min()), int(agriculture["year"].max())
    ]
    print(f"agriculture: {agriculture.height} municipality-years")

    # -- 2. favelas ----------------------------------------------------------
    profile = build_favela_profile(
        extract(favela_count_selection(), refresh=args.refresh).facts,
        extract(favela_population_selection(), refresh=args.refresh).facts,
        spine,
    )
    assert profile.height == spine.height, "favela profile lost or gained municipalities"
    inside_outside = build_inside_outside_sewage(
        extract(inside_outside_sewage_selection(), refresh=args.refresh).facts
    )
    profile.write_parquet(OUT / "favela_profile_municipality.parquet")
    inside_outside.write_parquet(OUT / "favela_sewage_inside_outside.parquet")
    n_with = int(profile["has_favela"].sum())
    report["municipalities_with_favela"] = n_with
    report["favela_population_total"] = float(profile["favela_population"].sum())
    report["inside_outside_municipalities"] = inside_outside.height
    print(f"favelas: {n_with} municipalities with settlements; "
          f"{report['favela_population_total']:,.0f} residents; "
          f"inside/outside table covers {inside_outside.height}")

    # -- 3. PNSB disease occurrence -----------------------------------------
    flags = build_disease_flag(
        extract(disease_occurrence_selection("leptospirosis"), refresh=args.refresh).facts
    )
    flags.write_parquet(OUT / "pnsb_leptospirosis_2008.parquet")

    windows = {"2008": (2008, 2008), "2007_2009": (2007, 2009), "2007_2025": (2007, 2025)}
    comparison = {}
    for name, (lo, hi) in windows.items():
        notified = (
            panel.filter(pl.col("year").is_between(lo, hi))
            .group_by("munic_code")
            .agg(pl.col("cases").sum().alias("cases"))
        )
        comparison[name] = compare_with_notifications(flags, notified)
        c = comparison[name]
        print(f"PNSB {name}: {c['declared']} declared, "
              f"{c['declared_with_no_notification']} silent "
              f"({c['declared_silent_pct']:.1f}%); reverse "
              f"{c['denied_with_notification']} ({c['denied_notified_pct']:.1f}%)")
    report["pnsb_comparison"] = comparison

    # -- 4. PNSB service existence, as a check on the extrapolation ----------
    services = build_service_flags(
        extract(service_existence_selection(), refresh=args.refresh).facts
    )
    services.write_parquet(OUT / "pnsb_services.parquet")

    socio = pl.read_parquet(
        PATHS.panel / "lept_panel_rq1_socioeconomic_municipality_month.parquet"
    )
    extrapolated = (
        socio.filter(pl.col("year") == 2008)
        .group_by("munic_code")
        .agg(pl.col("sanitation_sewer_share").mean().alias("sewer_share_2008"))
    )
    check = (
        services.filter(pl.col("year") == 2008)
        .select("munic_code", "pnsb_has_sewer_network")
        .join(extrapolated, on="munic_code", how="inner")
        .filter(pl.col("sewer_share_2008").is_not_null())
    )
    by_flag = check.group_by("pnsb_has_sewer_network").agg(
        pl.len().alias("n"),
        pl.col("sewer_share_2008").median().alias("median_share"),
        pl.col("sewer_share_2008").quantile(0.9).alias("p90_share"),
    ).sort("pnsb_has_sewer_network")
    print("\nback-extrapolation check (census sewer share in 2008 vs PNSB 2008):")
    print(by_flag)
    report["extrapolation_check"] = by_flag.to_dicts()

    (OUT / "structural_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
