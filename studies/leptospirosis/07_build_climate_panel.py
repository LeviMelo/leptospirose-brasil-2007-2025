"""Build the climate-complete municipality-month panel for leptospirosis RQ1.

This is study orchestration, not a data-source adapter.  The reusable work
(product routing, source-side aggregation, coverage accounting and provenance)
lives in :mod:`brepi.sources.climate.brdwgd`; this script declares the study
window and the explicit policy for the incomplete ERA5-Land tail.

The published ERA5-Land municipal artefacts currently contain one fewer day
than the calendar in each 2024/2025 month.  We retain the observed sum, expose
the coverage, and provide a transparent calendar-prorated total for modelling.
The raw and prorated values therefore remain available for sensitivity
analysis.  Nothing is silently imputed.

Run from ``brepi/``:
    python \
        studies/leptospirosis/07_build_climate_panel.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice
from brepi.io import cache
from brepi.panel.spine import attach
from brepi.sources.climate.brdwgd import cache_keys, monthly_municipal

YEARS = range(2007, 2026)
BASE_PANEL = "lept_panel_municipality_month.parquet"
OUTPUT_PANEL = "lept_panel_rq1_municipality_month.parquet"
SNAPSHOT = "lepto_climate_2007_2025_20260730"


def main() -> int:
    PATHS.ensure()
    base_path = PATHS.panel / BASE_PANEL
    if not base_path.exists():
        raise FileNotFoundError(
            f"{base_path} is absent; run 03_assemble_panel.py before this stage"
        )

    panel = pl.read_parquet(base_path)
    codes = panel["munic_code"].unique().sort().to_list()
    expected = len(codes) * len(YEARS) * 12
    print(f"[1/4] monthly precipitation: {len(codes):,} municipalities x "
          f"{len(YEARS) * 12} months")
    climate = monthly_municipal(
        YEARS,
        variables=("pr",),
        munic_codes=codes,
        download=False,
        require_complete_months=False,
    )
    climate = climate.with_columns(
        pl.col("pr_total").cast(pl.Float64).alias("precip_mm_observed"),
        pl.col("coverage_fraction").cast(pl.Float64).alias("climate_coverage"),
        pl.col("product").alias("climate_product"),
        (pl.col("n_days_expected") - pl.col("n_days"))
        .cast(pl.Int16)
        .alias("precip_missing_days"),
    ).with_columns(
        # Calendar prorating is explicit and reversible. It is a small
        # correction for the systematically omitted final day of ERA5 months,
        # not a claim to have observed that day's rainfall.
        (pl.col("precip_mm_observed") / pl.col("climate_coverage"))
        .alias("precip_mm"),
    )
    value_columns = [
        "precip_mm",
        "precip_mm_observed",
        "climate_coverage",
        "precip_missing_days",
        "rx1day",
        "r1",
        "r10",
        "r20",
        "r50",
    ]
    climate = climate.select(
        "munic_code",
        "period",
        *[pl.col(name).cast(pl.Float64) for name in value_columns],
        "climate_product",
    ).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("territorial_imputation"),
        pl.lit(None, dtype=pl.Utf8).alias("territorial_source_codes"),
        pl.lit(None, dtype=pl.Int16).alias("territorial_source_count"),
    )
    proxies = lattice.impute_created_unit_covariates(
        climate,
        codes,
        value_columns,
        source_lattice_year=2010,
    ).with_columns(
        pl.lit("territorial_parent_proxy").alias("climate_product")
    )
    climate = pl.concat(
        [climate, proxies.select(climate.columns)], how="vertical_relaxed"
    ).with_columns(
        (pl.col("precip_missing_days") > 0).alias("precip_prorated"),
        pl.col("territorial_imputation").is_not_null().alias("territorial_imputed"),
    ).sort(["munic_code", "period"])
    if climate.height != expected:
        missing = expected - climate.height
        raise ValueError(
            f"completed climate lattice has {climate.height:,} rows, expected "
            f"{expected:,} ({missing:,} missing)"
        )

    print("[2/4] attach to the frozen outcome/denominator spine")
    enriched, report = attach(
        panel,
        climate,
        name="monthly_precipitation",
    )
    if report.coverage != 1.0 or report.unmatched_source_keys:
        raise ValueError(report.summary())
    if enriched.height != panel.height:
        raise AssertionError("climate join changed the panel spine")
    print("      " + report.summary())

    print("[3/4] quality report and source snapshot")
    quality_by_product = (
        climate.group_by("climate_product")
        .agg(
            pl.len().alias("municipality_months"),
            pl.col("precip_prorated").sum().alias("prorated_cells"),
            pl.col("climate_coverage").min().alias("minimum_coverage"),
        )
        .sort("climate_product")
    )
    quality = {
        "base_panel": str(base_path),
        "output_panel": str(PATHS.panel / OUTPUT_PANEL),
        "rows": enriched.height,
        "municipalities": len(codes),
        "start": "2007-01-01",
        "end": "2025-12-01",
        "precipitation_policy": {
            "observed_column": "precip_mm_observed",
            "model_column": "precip_mm",
            "incomplete_months": "calendar-prorated; flagged by precip_prorated",
            "territorial_vintage": (
                "five municipalities absent from the source lattice use recorded "
                "parent covariates; flagged by territorial_imputed"
            ),
            "product_change": (
                "BR-DWGD through 2024-03-20; ERA5-Land thereafter; "
                "March 2024 is a mixed-product month"
            ),
            "primary_unspliced_sensitivity_end": "2023-12-01",
        },
        "by_product": quality_by_product.to_dicts(),
    }
    report_path = PATHS.reports / "lepto_climate_panel_quality.json"
    report_path.write_text(
        json.dumps(quality, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    keys = (
        cache_keys("brdwgd", ("pr",), date(2007, 1, 1), date(2024, 3, 20))
        + cache_keys(
            "era5land",
            ("pr",),
            date(2024, 3, 21),
            date(2025, 12, 31),
        )
    )
    manifest_path = cache.write_manifest(SNAPSHOT, keys)
    print(quality_by_product)
    print(f"      manifest: {manifest_path}")

    print("[4/4] write derived RQ1 panel")
    out = PATHS.panel / OUTPUT_PANEL
    enriched.write_parquet(out)
    print(f"      wrote {out} ({enriched.height:,} x {enriched.width})")
    print(f"      quality: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
