"""WP3 - assemble the leptospirosis municipality-month panel.

Builds the complete 5,570 x 228 spine and left-joins the outcome and
denominator blocks onto it. Environmental blocks attach later by the same
mechanism as their source modules land.

The ordering is deliberate and is the reason the panel is trustworthy: the
spine is enumerated first, from the lattice, with no reference to the data.
Every municipality-month therefore exists whether or not anything was ever
reported there, and the 97% of cells that are zero are recorded as zeros
rather than being absent. In a study whose central question is how much of
the observed spatial variation is detection rather than transmission, a panel
that dropped those cells would have destroyed the evidence before analysis
began.

Run:
    python studies/leptospirosis/03_assemble_panel.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice
from brepi.geo.health_regions import load_official_crosswalk
from brepi.panel.spine import PanelBuild, SpineSpec, build_spine, sparsity_report

START, END = date(2007, 1, 1), date(2025, 12, 1)


def main() -> int:
    PATHS.ensure()
    interim = PATHS.interim

    print("[1/5] spine")
    muns = lattice.load_municipalities(year=2022)
    spec = SpineSpec(
        municipalities=tuple(sorted(muns["code7"].to_list())),
        start=START,
        end=END,
        grain="month",
    )
    spine = build_spine(spec)
    print(f"      {spine.height:,} rows = {len(spec.municipalities):,} municipalities "
          f"x {spec.n_periods} months")

    # Geographic keys for the multi-scale analysis. Every aggregation the study
    # reports must come off this table, not from ad hoc code slicing.
    geo_cols = [c for c in ("code7", "uf_abbr", "region", "microregion",
                            "immediate_region", "intermediate_region", "name")
                if c in muns.columns]
    # A Região de Saúde is a SUS administrative geography, not an IBGE one.
    # Do not infer it from a name or a spatial overlay: an externally supplied,
    # dated official crosswalk is required.  The main municipality panel remains
    # valid without it, while the R pipeline will visibly skip that scale.
    health_path = os.environ.get("BREPI_HEALTH_REGIONS")
    if health_path:
        health_vintage = os.environ.get("BREPI_HEALTH_REGIONS_VINTAGE", "")
        mapping, report = load_official_crosswalk(
            health_path, vintage=health_vintage, lattice_year=2022
        )
        print(f"      health regions: {report.n_municipalities:,} municipalities; "
              f"vintage={report.vintage}")
        muns = lattice.attach_health_regions(muns, mapping)
        if muns["health_region"].null_count():
            raise ValueError(
                "BREPI_HEALTH_REGIONS does not cover the full 2022 municipality lattice; "
                "do not fit a partially defined health-region model."
            )
        geo_cols += ["health_region", "health_region_name"]
    else:
        print("      health-region crosswalk not supplied; health-region models are disabled")
    builder = PanelBuild(spine)
    geo = muns.select(geo_cols)
    geo_renames = {
        old: new for old, new in {
            "code7": "munic_code",
            "immediate_region": "immediate_region_code",
            "microregion": "microregion_code",
            "health_region": "health_region_code",
        }.items() if old in geo.columns
    }
    builder.add(
        geo.rename(geo_renames),
        name="geography",
        on=("munic_code",),
        min_coverage=0.999,
    )

    print("[2/5] outcomes (SINAN confirmed leptospirosis)")
    cases = pl.read_parquet(interim / "lept_confirmed_mun_month.parquet")
    cases = cases.with_columns(
        lattice.code6_to_code7_expr("munic_code6").alias("munic_code")
    ).drop("munic_code6")
    builder.add(
        cases,
        name="sinan_lept",
        fill={
            "cases": 0, "deaths": 0, "hospitalised": 0,
            "criterion_epi": 0, "flood_contact": 0, "cases_residence_fallback": 0,
        },
    )

    print("[3/5] denominators")
    pop = pl.read_parquet(interim / "population_municipal_year.parquet")
    builder.add(pop, name="population", on=("munic_code", "year"), min_coverage=0.99)

    panel = builder.spine
    print(builder.audit().to_pandas().to_string(index=False))

    print("[4/5] derived measures")
    panel = panel.with_columns(
        # ``period`` remains the Python join key; ``date`` is the public
        # analysis contract consumed by the R modelling layer.
        pl.col("period").alias("date"),
        # Person-months is the correct exposure for a monthly count model; the
        # annual population is constant within a year by construction, which is
        # a documented approximation, not an oversight.
        (pl.col("population") / 12.0).alias("person_months"),
    ).with_columns(
        (pl.col("cases") / pl.col("person_months") * 100_000).alias("incidence_per_100k_pm"),
        (pl.col("cases") > 0).alias("any_case"),
    )

    print("[5/5] sparsity and write")
    stats = sparsity_report(panel, count_col="cases")
    print(json.dumps(stats, indent=1, default=float))
    (PATHS.reports / "panel_sparsity.json").write_text(
        json.dumps(stats, indent=1, default=float), encoding="utf-8"
    )

    # Gates: the spine must be complete and the outcome mass must be preserved.
    assert panel.height == spec.n_rows, "spine height changed during assembly"
    lost = int(cases["cases"].sum()) - int(panel["cases"].sum())
    print(f"      cases in source {int(cases['cases'].sum()):,}; "
          f"in panel {int(panel['cases'].sum()):,}; lost {lost:,}")
    if lost:
        missing = cases.join(
            panel.select("munic_code").unique(), on="munic_code", how="anti"
        )
        print("      municipalities in the outcome file but absent from the 2022 lattice:")
        print(missing.group_by("munic_code").agg(pl.col("cases").sum())
              .sort("cases", descending=True).head(15))
        raise RuntimeError(
            f"municipality panel lost {lost} geocoded cases; resolve source "
            "municipality keys before continuing"
        )

    out = PATHS.panel / "lept_panel_municipality_month.parquet"
    panel.write_parquet(out)
    builder.audit().write_csv(PATHS.reports / "panel_join_audit.csv")
    print(f"      wrote {out} ({panel.height:,} x {panel.width})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
