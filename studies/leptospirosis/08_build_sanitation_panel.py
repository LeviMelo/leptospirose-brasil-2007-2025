"""Build harmonised census sanitation anchors and attach them to RQ1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS
from brepi.geo.lattice import (
    impute_created_unit_covariates,
    load_municipalities,
)
from brepi.io.cache import write_manifest
from brepi.sources.sanitation import (
    build_sewage_anchors,
    expand_census_anchors,
    sewage_selection_2010,
    sewage_selection_2022,
)
from brepi.sources.sidra.extract import extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result_2010 = extract(sewage_selection_2010(), refresh=args.refresh)
    result_2022 = extract(sewage_selection_2022(), refresh=args.refresh)
    anchors, margins = build_sewage_anchors(
        result_2010.facts,
        result_2022.facts,
    )

    target_codes = (
        load_municipalities(2022)["code7"].cast(pl.Utf8).sort().to_list()
    )
    anchor_2010 = anchors.filter(pl.col("year") == 2010)
    proxies = impute_created_unit_covariates(
        anchor_2010,
        target_codes,
        ["sanitation_sewer_share"],
        code_column="munic_code",
        key_columns=("year",),
        weight_column="sanitation_households",
        source_lattice_year=2010,
    ).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("sanitation_households"),
        pl.lit(1394, dtype=pl.Int32).alias("sanitation_source_table"),
    )
    proxies = proxies.select(anchors.columns)
    anchors = pl.concat([anchors, proxies], how="vertical_relaxed").sort(
        ["munic_code", "year"]
    )
    counts = anchors.group_by("munic_code").len()
    bad_counts = counts.filter(pl.col("len") != 2)
    if bad_counts.height:
        raise RuntimeError(
            f"{bad_counts.height} municipalities lack exactly two anchors"
        )

    annual = expand_census_anchors(anchors)
    anchor_path = PATHS.interim / "sanitation" / "census_sewage_anchors.parquet"
    annual_path = PATHS.interim / "sanitation" / "census_sewage_annual.parquet"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.write_parquet(anchor_path)
    annual.write_parquet(annual_path)

    panel_path = PATHS.panel / "lept_panel_rq1_municipality_month.parquet"
    panel = pl.read_parquet(panel_path).with_columns(
        pl.col("date").dt.year().alias("_year")
    )
    joined = panel.join(
        annual,
        left_on=["munic_code", "_year"],
        right_on=["munic_code", "year"],
        how="left",
        validate="m:1",
    ).drop("_year")
    missing = joined.filter(pl.col("sanitation_sewer_share").is_null()).height
    if missing:
        raise RuntimeError(
            f"sanitation join left {missing} municipality-month cells missing"
        )
    output = (
        PATHS.panel / "lept_panel_rq1_structural_municipality_month.parquet"
    )
    joined.write_parquet(output, compression="zstd")

    cache_keys = [*result_2010.cache_keys, *result_2022.cache_keys]
    manifest = write_manifest("sanitation_census_2010_2022", cache_keys)
    report = {
        "source_tables": {
            "2010": 1394,
            "2022": 6805,
            "rejected": {
                "9860": (
                    "all 44,560 requested 2010 municipal cells returned "
                    "NOT_AVAILABLE despite metadata advertising the period"
                ),
                "3154": (
                    "preliminary-universe table; 119 municipal category "
                    "margins failed against its published total"
                ),
            },
        },
        "estimand": (
            "share of permanent private households using public sewer or "
            "stormwater network; 2010/2022 universe wording retained as a "
            "documented comparability sensitivity"
        ),
        "source_rows": {
            "2010": result_2010.facts.height,
            "2022": result_2022.facts.height,
        },
        "anchor_rows": anchors.height,
        "annual_rows": annual.height,
        "panel_rows": joined.height,
        "municipalities": joined["munic_code"].n_unique(),
        "territorial_proxy_anchors": proxies.height,
        "territorial_proxy_codes": proxies["munic_code"].to_list(),
        "missing_panel_cells": missing,
        "margin_checks": margins,
        "interpolation_policy": {
            "2007_2009": "nearest 2010 anchor",
            "2010_2022": "linear between census anchors",
            "2023_2025": "nearest 2022 anchor",
            "sensitivity_required": [
                "anchor-only models",
                "step-function carry-forward",
                "exclude extrapolated 2023-2025",
                "exclude five 2013-successor territorial proxies",
                "stable-area/AMC analysis",
            ],
        },
        "output": str(output),
        "manifest": str(manifest),
    }
    report_path = PATHS.reports / "lepto_sanitation_panel_quality.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
