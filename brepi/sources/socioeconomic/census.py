"""Semantically harmonised Census urbanisation anchors."""

from __future__ import annotations

import polars as pl

from brepi.sources.sidra.extract import Selection, check_margins


class SocioeconomicError(RuntimeError):
    """A socioeconomic source or derived indicator violates its contract."""


def urban_selection_2010() -> Selection:
    """2010 municipal population by urban/rural situation, sex total."""
    return Selection(
        agregado=202,
        periods=("2010",),
        variables=("93",),
        classifications={"2": ("0",), "1": ("0", "1", "2")},
        label="urban_population_census_2010",
    )


def urban_selection_2022() -> Selection:
    """2022 municipal population by urban/rural situation."""
    return Selection(
        agregado=9923,
        periods=("2022",),
        variables=("93",),
        classifications={"1": ("6795", "1", "2")},
        label="urban_population_census_2022",
    )


def _axis(facts: pl.DataFrame) -> pl.DataFrame:
    return (
        facts.with_row_index("_row")
        .select("_row", "classification_ids", "category_ids")
        .explode(["classification_ids", "category_ids"])
        .filter(pl.col("classification_ids") == "1")
        .select("_row", pl.col("category_ids").alias("category"))
    )


def _derive(
    facts: pl.DataFrame,
    *,
    year: int,
    table: int,
    total: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    margin = check_margins(
        facts,
        total,
        ("1", "2"),
        classification="1",
    )
    if not margin["passed"] or margin["n_incomplete"]:
        raise SocioeconomicError(
            f"table {table} urban/rural margins failed: "
            f"{margin['n_mismatched']} mismatched, "
            f"{margin['n_incomplete']} incomplete"
        )
    frame = facts.with_row_index("_row").join(
        _axis(facts), on="_row", how="inner"
    )
    wide = (
        frame.filter(pl.col("category").is_in([total, "1"]))
        .select(
            pl.col("locality_id").cast(pl.Utf8).alias("munic_code"),
            "category",
            "value_numeric",
        )
        .pivot(
            on="category",
            index="munic_code",
            values="value_numeric",
            aggregate_function="first",
        )
    )
    if total not in wide.columns or "1" not in wide.columns:
        raise SocioeconomicError(f"table {table} lacks total or urban category")
    invalid = wide.filter(
        pl.col(total).is_null()
        | (pl.col(total) <= 0)
        | pl.col("1").is_null()
    )
    if invalid.height:
        raise SocioeconomicError(
            f"table {table} has {invalid.height} invalid urban anchors"
        )
    return (
        wide.with_columns(
            pl.lit(year, dtype=pl.Int32).alias("year"),
            (pl.col("1") / pl.col(total)).alias("urban_share"),
            pl.col(total).alias("urban_population_support"),
            pl.lit(table, dtype=pl.Int32).alias("urban_source_table"),
            pl.lit("direct_observation").alias("territorial_imputation"),
            pl.col("munic_code").alias("territorial_source_codes"),
            pl.lit(1, dtype=pl.Int16).alias("territorial_source_count"),
        ).select(
            "munic_code",
            "year",
            "urban_share",
            "urban_population_support",
            "urban_source_table",
            "territorial_imputation",
            "territorial_source_codes",
            "territorial_source_count",
        ),
        margin,
    )


def build_urban_anchors(
    facts_2010: pl.DataFrame,
    facts_2022: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build 2010/2022 urban population shares with exact margin checks."""
    old, margin_old = _derive(facts_2010, year=2010, table=202, total="0")
    new, margin_new = _derive(
        facts_2022, year=2022, table=9923, total="6795"
    )
    anchors = pl.concat([old, new], how="vertical_relaxed").sort(
        ["munic_code", "year"]
    )
    return anchors, {"2010": margin_old, "2022": margin_new}
