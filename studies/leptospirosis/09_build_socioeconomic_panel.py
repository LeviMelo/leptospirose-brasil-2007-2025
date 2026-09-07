"""Build census urbanisation anchors and attach them to the study panel."""

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
from brepi.sources.sidra.extract import extract
from brepi.sources.socioeconomic import (
    build_real_gdp_per_capita,
    build_urban_anchors,
    gdp_deflator_selection,
    municipal_gdp_selection,
    urban_selection_2010,
    urban_selection_2022,
)
from brepi.temporal import expand_anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    result_2010 = extract(urban_selection_2010(), refresh=args.refresh)
    result_2022 = extract(urban_selection_2022(), refresh=args.refresh)
    anchors, margins = build_urban_anchors(
        result_2010.facts,
        result_2022.facts,
    )
    target_codes = (
        load_municipalities(2022)["code7"].cast(pl.Utf8).sort().to_list()
    )
    old = anchors.filter(pl.col("year") == 2010)
    proxies = impute_created_unit_covariates(
        old,
        target_codes,
        ["urban_share"],
        code_column="munic_code",
        key_columns=("year",),
        weight_column="urban_population_support",
        source_lattice_year=2010,
    ).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("urban_population_support"),
        pl.lit(202, dtype=pl.Int32).alias("urban_source_table"),
    ).select(anchors.columns)
    anchors = pl.concat([anchors, proxies], how="vertical_relaxed").sort(
        ["munic_code", "year"]
    )
    annual = expand_anchors(
        anchors,
        entities="munic_code",
        time="year",
        years=range(2007, 2026),
        value_columns=["urban_share"],
        prefix="urban",
        provenance_columns=[
            "urban_source_table",
            "territorial_imputation",
        ],
    )

    gdp_result = extract(municipal_gdp_selection(), refresh=args.refresh)
    deflator_result = extract(gdp_deflator_selection(), refresh=args.refresh)
    population = pl.read_parquet(
        PATHS.interim / "population_municipal_year.parquet"
    )
    gdp_observed = build_real_gdp_per_capita(
        gdp_result.facts,
        deflator_result.facts,
        population,
    ).with_columns(
        pl.lit(5938, dtype=pl.Int32).alias("gdp_source_table"),
        pl.lit("direct_observation").alias("territorial_imputation"),
        pl.col("munic_code").alias("territorial_source_codes"),
        pl.lit(1, dtype=pl.Int16).alias("territorial_source_count"),
    )
    gdp_blocks: list[pl.DataFrame] = [gdp_observed]
    gdp_proxy_rows = 0
    for year in range(2007, 2024):
        year_frame = gdp_observed.filter(pl.col("year") == year)
        proxies_year = impute_created_unit_covariates(
            year_frame,
            target_codes,
            ["gdp_per_capita"],
            code_column="munic_code",
            key_columns=("year",),
            weight_column="population",
            source_lattice_year=year,
            include_same_year=True,
        ).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("population"),
            pl.lit(5938, dtype=pl.Int32).alias("gdp_source_table"),
        ).select(gdp_observed.columns)
        gdp_proxy_rows += proxies_year.height
        if proxies_year.height:
            gdp_blocks.append(proxies_year)
    gdp_anchors = pl.concat(gdp_blocks, how="vertical_relaxed").sort(
        ["munic_code", "year"]
    )
    gdp_counts = gdp_anchors.group_by("munic_code").len()
    bad_gdp_counts = gdp_counts.filter(pl.col("len") != 17)
    if bad_gdp_counts.height:
        raise RuntimeError(
            f"{bad_gdp_counts.height} municipalities lack 17 GDP anchors"
        )
    gdp_annual = expand_anchors(
        gdp_anchors,
        entities="munic_code",
        time="year",
        years=range(2007, 2026),
        value_columns=["gdp_per_capita"],
        prefix="gdp",
        provenance_columns=[
            "gdp_source_table",
            "territorial_imputation",
        ],
    ).with_columns(
        (pl.col("gdp_per_capita") / 10_000)
        .arcsinh()
        .alias("gdp_per_capita_asinh")
    )

    interim = PATHS.interim / "socioeconomic"
    interim.mkdir(parents=True, exist_ok=True)
    anchors.write_parquet(interim / "census_urban_anchors.parquet")
    annual.write_parquet(interim / "census_urban_annual.parquet")
    gdp_anchors.write_parquet(interim / "gdp_real_2023_anchors.parquet")
    gdp_annual.write_parquet(interim / "gdp_real_2023_annual.parquet")

    input_path = (
        PATHS.panel / "lept_panel_rq1_structural_municipality_month.parquet"
    )
    panel = pl.read_parquet(input_path).with_columns(
        pl.col("date").dt.year().alias("_year")
    )
    joined = panel.join(
        annual,
        left_on=["munic_code", "_year"],
        right_on=["munic_code", "year"],
        how="left",
        validate="m:1",
    ).join(
        gdp_annual,
        left_on=["munic_code", "_year"],
        right_on=["munic_code", "year"],
        how="left",
        validate="m:1",
        suffix="_gdp",
    ).drop("_year")
    missing = joined.filter(
        pl.col("urban_share").is_null()
        | pl.col("gdp_per_capita").is_null()
        | pl.col("gdp_per_capita_asinh").is_null()
    ).height
    if missing:
        raise RuntimeError(
            f"socioeconomic joins left {missing} panel cells incomplete"
        )
    output = (
        PATHS.panel / "lept_panel_rq1_socioeconomic_municipality_month.parquet"
    )
    joined.write_parquet(output, compression="zstd")

    manifest = write_manifest(
        "socioeconomic_census_gdp",
        [
            *result_2010.cache_keys,
            *result_2022.cache_keys,
            *gdp_result.cache_keys,
            *deflator_result.cache_keys,
        ],
    )
    report = {
        "source_tables": {"2010": 202, "2022": 9923},
        "gdp_sources": {
            "municipal_nominal": 5938,
            "national_deflator": 6784,
            "price_basis": "2023 BRL",
            "model_transform": "asinh(gdp_per_capita / 10000)",
            "observed_years": [2007, 2023],
            "forecast_years": [2024, 2025],
        },
        "estimand": "share of resident population in urban household situation",
        "anchor_rows": anchors.height,
        "annual_rows": annual.height,
        "panel_rows": joined.height,
        "municipalities": joined["munic_code"].n_unique(),
        "territorial_proxy_anchors": proxies.height,
        "territorial_proxy_codes": proxies["munic_code"].to_list(),
        "missing_panel_cells": missing,
        "gdp_territorial_proxy_anchors": gdp_proxy_rows,
        "margin_checks": margins,
        "output": str(output),
        "manifest": str(manifest),
    }
    report_path = PATHS.reports / "lepto_socioeconomic_panel_quality.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
