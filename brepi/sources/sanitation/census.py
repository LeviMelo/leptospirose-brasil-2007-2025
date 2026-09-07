"""Census sanitation anchors and transparent between-census expansion.

The 2010 and 2022 anchors deliberately come from different SIDRA tables.
Table 9860 advertises both periods in metadata but returned only ``...`` for
every 2010 municipal cell when verified on 2026-07-30. Treating those cells as
data, or silently substituting 2022, would manufacture a longitudinal series.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from brepi.sources.sidra.extract import Selection, check_margins
from brepi.temporal import AnchorExpansionError, expand_anchors

CLASSIFICATION_2010 = "11558"
TOTAL_2010 = "0"
NETWORK_2010 = "92855"
MARGIN_COMPONENTS_2010 = (
    "92855",
    "92856",
    "92857",
    "92858",
    "92859",
    "119522",
    "119523",
)

CLASSIFICATION_2022 = "11558"
TOTAL_2022 = "46292"
NETWORK_2022 = "72110"
MARGIN_COMPONENTS_2022 = (
    "46290",
    "72112",
    "72113",
    "92858",
    "72114",
    "72115",
    "92861",
)


class SanitationError(RuntimeError):
    """A census sanitation cube or interpolation violates its contract."""


def sewage_selection_2010() -> Selection:
    """2010 exact public/stormwater sewer share from final SIDRA table 1394."""
    return Selection(
        agregado=1394,
        periods=("2010",),
        variables=("96",),
        classifications={
            "1": ("0",),  # dwelling situation: total
            "458": ("0",),  # bathroom/toilet count: total
            "125": ("0",),  # dwelling type: total
            "63": ("0",),  # occupancy condition: total
            CLASSIFICATION_2010: (
                TOTAL_2010,
                *MARGIN_COMPONENTS_2010,
            )
        },
        label="sanitation_census_2010_sewage",
    )


def sewage_selection_2022() -> Selection:
    """2022 exact public/stormwater sewer share from SIDRA table 6805."""
    return Selection(
        agregado=6805,
        periods=("2022",),
        variables=("381",),
        classifications={
            CLASSIFICATION_2022: (
                TOTAL_2022,
                NETWORK_2022,
                *MARGIN_COMPONENTS_2022,
            )
        },
        label="sanitation_census_2022_sewage",
    )


def _classification_axis(
    facts: pl.DataFrame, classification: str
) -> pl.DataFrame:
    return (
        facts.with_row_index("_row")
        .select("_row", "classification_ids", "category_ids")
        .explode(["classification_ids", "category_ids"])
        .filter(pl.col("classification_ids") == classification)
        .select("_row", pl.col("category_ids").alias("category"))
    )


def _derive_anchor(
    facts: pl.DataFrame,
    *,
    year: int,
    variable: str,
    classification: str,
    total: str,
    network: str,
    margin_components: Sequence[str],
    source_table: int,
) -> tuple[pl.DataFrame, dict[str, object]]:
    required = {
        "locality_id",
        "period",
        "variable_id",
        "value_numeric",
        "value_status",
        "classification_ids",
        "category_ids",
    }
    missing = required - set(facts.columns)
    if missing:
        raise SanitationError(f"SIDRA facts missing columns: {sorted(missing)}")

    selected = facts.filter(
        (pl.col("variable_id") == variable)
        & (pl.col("period").cast(pl.Int32) == year)
    )
    if selected.height == 0:
        raise SanitationError(f"table {source_table} returned no {year} facts")
    margin = check_margins(
        selected,
        total,
        margin_components,
        classification=classification,
    )
    if not margin["passed"] or margin["n_incomplete"]:
        raise SanitationError(
            f"table {source_table} sewage margins failed: "
            f"{margin['n_mismatched']} mismatched, "
            f"{margin['n_incomplete']} incomplete groups"
        )

    frame = selected.with_row_index("_row").join(
        _classification_axis(selected, classification),
        on="_row",
        how="inner",
    )
    wide = (
        frame.filter(pl.col("category").is_in([total, network]))
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
    if total not in wide.columns or network not in wide.columns:
        raise SanitationError(
            f"table {source_table} lacks total or exact-network category"
        )
    if wide.filter(pl.col(total).is_null() | (pl.col(total) <= 0)).height:
        raise SanitationError(
            f"table {source_table} has null/non-positive household totals"
        )

    anchors = (
        wide.with_columns(
            pl.lit(year, dtype=pl.Int32).alias("year"),
            (pl.col(network) / pl.col(total)).alias("sanitation_sewer_share"),
            pl.col(total).alias("sanitation_households"),
            pl.lit(source_table, dtype=pl.Int32).alias("sanitation_source_table"),
            pl.lit("direct_observation").alias("territorial_imputation"),
            pl.col("munic_code").alias("territorial_source_codes"),
            pl.lit(1, dtype=pl.Int16).alias("territorial_source_count"),
        )
        .select(
            "munic_code",
            "year",
            "sanitation_sewer_share",
            "sanitation_households",
            "sanitation_source_table",
            "territorial_imputation",
            "territorial_source_codes",
            "territorial_source_count",
        )
        .sort("munic_code")
    )
    bad_share = anchors.filter(
        pl.col("sanitation_sewer_share").is_null()
        | ~pl.col("sanitation_sewer_share").is_finite()
        | (pl.col("sanitation_sewer_share") < 0)
        | (pl.col("sanitation_sewer_share") > 1)
    )
    if bad_share.height:
        raise SanitationError(f"{bad_share.height} invalid sewer-share anchors")
    return anchors, margin


def build_sewage_anchors(
    facts_2010: pl.DataFrame,
    facts_2022: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Derive semantically harmonised anchors with separate margin checks."""
    anchor_2010, margin_2010 = _derive_anchor(
        facts_2010,
        year=2010,
        variable="96",
        classification=CLASSIFICATION_2010,
        total=TOTAL_2010,
        network=NETWORK_2010,
        margin_components=MARGIN_COMPONENTS_2010,
        source_table=1394,
    )
    anchor_2022, margin_2022 = _derive_anchor(
        facts_2022,
        year=2022,
        variable="381",
        classification=CLASSIFICATION_2022,
        total=TOTAL_2022,
        network=NETWORK_2022,
        margin_components=MARGIN_COMPONENTS_2022,
        source_table=6805,
    )
    return (
        pl.concat([anchor_2010, anchor_2022], how="vertical_relaxed").sort(
            ["munic_code", "year"]
        ),
        {"2010": margin_2010, "2022": margin_2022},
    )


def expand_census_anchors(
    anchors: pl.DataFrame,
    *,
    years: Sequence[int] = tuple(range(2007, 2026)),
    value_columns: Sequence[str] = ("sanitation_sewer_share",),
) -> pl.DataFrame:
    """Expand two census anchors with explicit interpolation/extrapolation.

    For every municipality/value, years between the two anchors are linearly
    interpolated; earlier/later years carry the nearest anchor. Anchor source
    and territorial-proxy status remain attached to each annual estimate.
    """
    provenance = [
        column
        for column in (
            "sanitation_source_table",
            "territorial_imputation",
        )
        if column in anchors.columns
    ]
    try:
        expanded = expand_anchors(
            anchors,
            entities="munic_code",
            time="year",
            years=years,
            value_columns=value_columns,
            prefix="sanitation",
            provenance_columns=provenance,
        )
    except AnchorExpansionError as exc:
        raise SanitationError(str(exc).replace("temporal", "census")) from exc
    return expanded.with_columns(
        pl.when(pl.col("sanitation_method") == "linear_between_anchors")
        .then(pl.lit("linear_between_censuses"))
        .otherwise(pl.col("sanitation_method"))
        .alias("sanitation_method")
    )
