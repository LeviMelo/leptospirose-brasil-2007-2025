"""Transparent expansion of sparse temporal anchors onto an annual lattice."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


class AnchorExpansionError(RuntimeError):
    """An anchor panel cannot support the requested temporal expansion."""


def expand_anchors(
    anchors: pl.DataFrame,
    *,
    entities: str,
    time: str,
    years: Sequence[int],
    value_columns: Sequence[str],
    prefix: str,
    provenance_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Linearly expand entity-specific anchors with typed edge policies.

    Interior years are linearly interpolated. Years outside the observed anchor
    range carry the nearest anchor. Every output row records the method, lower
    and upper anchor years, interpolation fraction, and requested provenance at
    both endpoints. This function aligns intensive covariates; it does not
    disaggregate counts or manufacture event histories.
    """
    required = {
        entities,
        time,
        *value_columns,
        *provenance_columns,
    }
    missing = required - set(anchors.columns)
    if missing:
        raise AnchorExpansionError(
            f"anchor frame missing columns: {sorted(missing)}"
        )
    output_years = sorted({int(year) for year in years})
    if not output_years:
        raise ValueError("years must not be empty")
    if not value_columns:
        raise ValueError("value_columns must not be empty")

    rows: list[dict[str, object]] = []
    for group in anchors.partition_by(entities, as_dict=False):
        entity = group[entities][0]
        group = group.sort(time)
        anchor_years = [int(value) for value in group[time].to_list()]
        if len(anchor_years) < 2:
            raise AnchorExpansionError(
                f"{entity} has fewer than two temporal anchors"
            )
        if len(set(anchor_years)) != len(anchor_years):
            raise AnchorExpansionError(
                f"{entity} has duplicate temporal anchors"
            )
        for year in output_years:
            if year <= anchor_years[0]:
                lo = hi = 0
                fraction = 0.0
                method = (
                    "nearest_anchor_backcast"
                    if year < anchor_years[0]
                    else "observed_anchor"
                )
            elif year >= anchor_years[-1]:
                lo = hi = len(anchor_years) - 1
                fraction = 0.0
                method = (
                    "nearest_anchor_forecast"
                    if year > anchor_years[-1]
                    else "observed_anchor"
                )
            else:
                hi = next(
                    index
                    for index, anchor_year in enumerate(anchor_years)
                    if anchor_year > year
                )
                lo = hi - 1
                fraction = (year - anchor_years[lo]) / (
                    anchor_years[hi] - anchor_years[lo]
                )
                method = "linear_between_anchors"
            row: dict[str, object] = {
                entities: entity,
                "year": year,
                f"{prefix}_method": method,
                f"{prefix}_anchor_lo": anchor_years[lo],
                f"{prefix}_anchor_hi": anchor_years[hi],
                f"{prefix}_fraction": fraction,
            }
            for column in provenance_columns:
                row[f"{column}_lo"] = group[column][lo]
                row[f"{column}_hi"] = group[column][hi]
            for column in value_columns:
                low = float(group[column][lo])
                high = float(group[column][hi])
                row[column] = low + fraction * (high - low)
            rows.append(row)
    return pl.DataFrame(rows).sort([entities, "year"])
