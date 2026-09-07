"""Conservative transfer between versioned areal geographies.

Administrative codes are not spatial identities. A code belongs to a named
geography *version*, and changing from one version to another is a support
transformation with semantics that depend on the variable:

* extensive quantities (cases, deaths, people) must conserve mass;
* intensive fields (mean rainfall, temperature, proportions with a known
  support) must conserve the target-support weighted mean;
* rates are never transferred directly: numerator and denominator move
  separately and the rate is reconstructed;
* adjacency and identifiers are topology, not measurements, and must be
  recomputed on the target geography.

The correspondence table is an explicit sparse transfer operator. For each
source/target intersection it carries:

``source_fraction``
    Fraction of the source-supported mass assigned to the target. Columns of
    the mathematical operator sum to one. Used for extensive quantities.
``target_fraction``
    Fraction of the target support represented by the source intersection.
    Rows of the operator sum to one. Used for intensive fields.

Area overlap, population-weighted/dasymetric overlap and exact atomic
crosswalks can all populate this contract. The method and exactness remain data
columns so an areal assumption cannot silently become an administrative fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import polars as pl


class TransferError(ValueError):
    """A geography transfer is undefined or violates its invariants."""


class VariableKind(StrEnum):
    EXTENSIVE = "extensive"
    INTENSIVE = "intensive"
    RATE = "rate"


@dataclass(frozen=True)
class TransferDiagnostics:
    source_version: str
    target_version: str
    kind: VariableKind
    source_units: int
    target_units: int
    links: int
    methods: tuple[str, ...]
    exact_share: float
    max_weight_error: float
    max_conservation_error: float | None


REQUIRED_CORRESPONDENCE = {
    "source_code": pl.Utf8,
    "target_code": pl.Utf8,
    "source_fraction": pl.Float64,
    "target_fraction": pl.Float64,
    "method": pl.Utf8,
    "exact": pl.Boolean,
}


def correspondence_from_support(
    support: pl.DataFrame,
    *,
    source_column: str,
    target_column: str,
    mass_column: str,
    source_version: str,
    target_version: str,
    method: str,
    exact: bool,
    expected_source_codes: Sequence[str] | None = None,
    expected_target_codes: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Compile atomic/intersection support into a sparse transfer operator.

    Each row of ``support`` is an atom assigned to one source and one target
    unit. ``mass_column`` declares the support measure: intersection area,
    population, households, facilities, pixel area, or another non-negative
    quantity justified by the estimand. Multiple atoms for the same pair are
    summed.

    Supplying the expected code sets turns spatial coverage into an executable
    contract. This catches a clipped overlay or a missing administrative member
    before its weights are normalised to a deceptively valid-looking matrix.
    """
    needed = {source_column, target_column, mass_column}
    missing = needed - set(support.columns)
    if missing:
        raise TransferError(f"support table lacks {sorted(missing)}")
    if not method.strip():
        raise TransferError("support correspondence requires a named method")
    if not support.schema[mass_column].is_numeric():
        raise TransferError(f"support mass {mass_column!r} is not numeric")
    invalid = support.filter(
        pl.col(source_column).is_null()
        | pl.col(target_column).is_null()
        | pl.col(mass_column).is_null()
        | ~pl.col(mass_column).is_finite()
        | (pl.col(mass_column) <= 0)
    )
    if invalid.height:
        raise TransferError(
            f"support table has {invalid.height} null/non-positive/non-finite atom(s)"
        )

    observed_sources = set(
        support[source_column].cast(pl.Utf8).unique().to_list()
    )
    observed_targets = set(
        support[target_column].cast(pl.Utf8).unique().to_list()
    )
    if expected_source_codes is not None:
        absent = sorted(set(map(str, expected_source_codes)) - observed_sources)
        if absent:
            raise TransferError(
                f"support omits {len(absent)} expected source unit(s), e.g. {absent[:5]}"
            )
    if expected_target_codes is not None:
        absent = sorted(set(map(str, expected_target_codes)) - observed_targets)
        if absent:
            raise TransferError(
                f"support omits {len(absent)} expected target unit(s), e.g. {absent[:5]}"
            )

    links = (
        support.select(
            pl.col(source_column).cast(pl.Utf8).alias("source_code"),
            pl.col(target_column).cast(pl.Utf8).alias("target_code"),
            pl.col(mass_column).cast(pl.Float64).alias("_mass"),
        )
        .group_by(["source_code", "target_code"])
        .agg(pl.col("_mass").sum())
    )
    source_totals = links.group_by("source_code").agg(
        pl.col("_mass").sum().alias("_source_mass")
    )
    target_totals = links.group_by("target_code").agg(
        pl.col("_mass").sum().alias("_target_mass")
    )
    out = (
        links.join(source_totals, on="source_code")
        .join(target_totals, on="target_code")
        .with_columns(
            (pl.col("_mass") / pl.col("_source_mass")).alias("source_fraction"),
            (pl.col("_mass") / pl.col("_target_mass")).alias("target_fraction"),
            pl.lit(method).alias("method"),
            pl.lit(exact).alias("exact"),
            pl.lit(source_version).alias("source_version"),
            pl.lit(target_version).alias("target_version"),
            pl.col("_mass").alias("support_mass"),
        )
        .select(
            "source_code",
            "target_code",
            "source_fraction",
            "target_fraction",
            "method",
            "exact",
            "source_version",
            "target_version",
            "support_mass",
        )
        .sort(["source_code", "target_code"])
    )
    validate_correspondence(out, kind=VariableKind.EXTENSIVE)
    validate_correspondence(out, kind=VariableKind.INTENSIVE)
    return out


def validate_correspondence(
    correspondence: pl.DataFrame,
    *,
    kind: VariableKind | str,
    tolerance: float = 1e-8,
) -> float:
    """Validate the sparse operator and return its maximum weight-sum error."""
    kind = VariableKind(kind)
    missing = set(REQUIRED_CORRESPONDENCE) - set(correspondence.columns)
    if missing:
        raise TransferError(f"correspondence lacks {sorted(missing)}")
    if correspondence.is_empty():
        raise TransferError("correspondence is empty")
    duplicate_links = correspondence.height - correspondence.select(
        "source_code", "target_code"
    ).unique().height
    if duplicate_links:
        raise TransferError(
            f"correspondence has {duplicate_links} duplicate source-target links"
        )
    for column in ("source_fraction", "target_fraction"):
        invalid = correspondence.filter(
            pl.col(column).is_null()
            | ~pl.col(column).is_finite()
            | (pl.col(column) < 0)
            | (pl.col(column) > 1 + tolerance)
        )
        if invalid.height:
            raise TransferError(
                f"{column} has {invalid.height} invalid weight(s)"
            )

    if kind in (VariableKind.EXTENSIVE, VariableKind.RATE):
        sums = correspondence.group_by("source_code").agg(
            pl.col("source_fraction").sum().alias("_sum")
        )
        axis = "source"
    else:
        sums = correspondence.group_by("target_code").agg(
            pl.col("target_fraction").sum().alias("_sum")
        )
        axis = "target"
    error = float(sums.select((pl.col("_sum") - 1).abs().max()).item())
    if error > tolerance:
        examples = sums.filter((pl.col("_sum") - 1).abs() > tolerance).head(5).to_dicts()
        raise TransferError(
            f"{kind.value} operator is not closed over the {axis} geography; "
            f"maximum weight-sum error {error:.3g}, e.g. {examples}"
        )
    return error


def transfer_extensive(
    frame: pl.DataFrame,
    correspondence: pl.DataFrame,
    value_columns: Sequence[str],
    *,
    source_version: str,
    target_version: str,
    source_column: str = "source_code",
    target_column: str = "target_code",
    group_columns: Sequence[str] = (),
    tolerance: float = 1e-8,
) -> tuple[pl.DataFrame, TransferDiagnostics]:
    """Transfer counts/masses and assert conservation within every group."""
    weight_error = validate_correspondence(
        correspondence, kind=VariableKind.EXTENSIVE, tolerance=tolerance
    )
    _validate_frame(frame, source_column, group_columns, value_columns)
    missing_sources = frame.select(pl.col(source_column).cast(pl.Utf8).alias("source_code")).unique().join(
        correspondence.select("source_code").unique(),
        on="source_code",
        how="anti",
    )
    if missing_sources.height:
        raise TransferError(
            f"{missing_sources.height} source unit(s) have no correspondence, "
            f"e.g. {missing_sources.head(5)['source_code'].to_list()}"
        )

    left = frame.rename({source_column: "source_code"})
    joined = left.join(
        correspondence.select(
            "source_code", "target_code", "source_fraction"
        ),
        on="source_code",
        how="inner",
    )
    transferred = (
        joined.with_columns(
            *[
                (pl.col(name).cast(pl.Float64) * pl.col("source_fraction")).alias(name)
                for name in value_columns
            ]
        )
        .group_by([*group_columns, "target_code"])
        .agg(*[pl.col(name).sum().alias(name) for name in value_columns])
        .rename({"target_code": target_column})
        .sort([*group_columns, target_column])
    )

    source_totals = frame.group_by(list(group_columns)).agg(
        *[pl.col(name).cast(pl.Float64).sum().alias(name) for name in value_columns]
    ) if group_columns else frame.select(
        *[pl.col(name).cast(pl.Float64).sum().alias(name) for name in value_columns]
    )
    target_totals = transferred.group_by(list(group_columns)).agg(
        *[pl.col(name).sum().alias(name) for name in value_columns]
    ) if group_columns else transferred.select(
        *[pl.col(name).sum().alias(name) for name in value_columns]
    )
    suffix = "_target"
    comparison = source_totals.join(
        target_totals, on=list(group_columns), how="inner", suffix=suffix
    ) if group_columns else pl.concat(
        [source_totals, target_totals.rename({name: name + suffix for name in value_columns})],
        how="horizontal",
    )
    conservation_error = max(
        float(
            comparison.select(
                (pl.col(name) - pl.col(name + suffix)).abs().max()
            ).item()
        )
        for name in value_columns
    )
    scale = max(
        1.0,
        max(float(source_totals.select(pl.col(name).abs().max()).item()) for name in value_columns),
    )
    if conservation_error > tolerance * scale:
        raise TransferError(
            f"extensive transfer failed conservation: absolute error "
            f"{conservation_error:.6g}"
        )
    return transferred, _diagnostics(
        correspondence,
        source_version,
        target_version,
        VariableKind.EXTENSIVE,
        weight_error,
        conservation_error,
    )


def transfer_intensive(
    frame: pl.DataFrame,
    correspondence: pl.DataFrame,
    value_columns: Sequence[str],
    *,
    source_version: str,
    target_version: str,
    source_column: str = "source_code",
    target_column: str = "target_code",
    group_columns: Sequence[str] = (),
    tolerance: float = 1e-8,
) -> tuple[pl.DataFrame, TransferDiagnostics]:
    """Transfer means/fields using target-support fractions."""
    weight_error = validate_correspondence(
        correspondence, kind=VariableKind.INTENSIVE, tolerance=tolerance
    )
    _validate_frame(frame, source_column, group_columns, value_columns)
    left = frame.rename({source_column: "source_code"})
    joined = left.join(
        correspondence.select(
            "source_code", "target_code", "target_fraction"
        ),
        on="source_code",
        how="inner",
    )
    out = (
        joined.with_columns(
            *[
                (pl.col(name).cast(pl.Float64) * pl.col("target_fraction")).alias(name)
                for name in value_columns
            ]
        )
        .group_by([*group_columns, "target_code"])
        .agg(*[pl.col(name).sum().alias(name) for name in value_columns])
        .rename({"target_code": target_column})
        .sort([*group_columns, target_column])
    )
    expected_targets = correspondence["target_code"].n_unique()
    observed_per_group = out.group_by(list(group_columns)).agg(
        pl.col(target_column).n_unique().alias("_n")
    ) if group_columns else pl.DataFrame({"_n": [out[target_column].n_unique()]})
    if observed_per_group.filter(pl.col("_n") != expected_targets).height:
        raise TransferError(
            "intensive transfer lacks one or more source values needed to cover "
            "every target in at least one group"
        )
    return out, _diagnostics(
        correspondence,
        source_version,
        target_version,
        VariableKind.INTENSIVE,
        weight_error,
        None,
    )


def transfer_rate(
    frame: pl.DataFrame,
    correspondence: pl.DataFrame,
    *,
    numerator: str,
    denominator: str,
    output: str,
    source_version: str,
    target_version: str,
    source_column: str = "source_code",
    target_column: str = "target_code",
    group_columns: Sequence[str] = (),
    tolerance: float = 1e-8,
) -> tuple[pl.DataFrame, TransferDiagnostics]:
    """Transfer a rate by conserving its numerator and denominator separately."""
    out, diagnostics = transfer_extensive(
        frame,
        correspondence,
        [numerator, denominator],
        source_version=source_version,
        target_version=target_version,
        source_column=source_column,
        target_column=target_column,
        group_columns=group_columns,
        tolerance=tolerance,
    )
    if out.filter(pl.col(denominator) <= 0).height:
        raise TransferError("transferred rate denominator is non-positive")
    out = out.with_columns(
        (pl.col(numerator) / pl.col(denominator)).alias(output)
    )
    return out, TransferDiagnostics(
        **{**diagnostics.__dict__, "kind": VariableKind.RATE}
    )


def _validate_frame(
    frame: pl.DataFrame,
    code_column: str,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> None:
    needed = {code_column, *group_columns, *value_columns}
    missing = needed - set(frame.columns)
    if missing:
        raise TransferError(f"source frame lacks {sorted(missing)}")
    keys = [code_column, *group_columns]
    duplicates = frame.height - frame.select(keys).unique().height
    if duplicates:
        raise TransferError(f"source frame has {duplicates} duplicate row(s) on {keys}")
    for name in value_columns:
        if not frame.schema[name].is_numeric():
            raise TransferError(f"{name!r} is not numeric")
        if frame.filter(pl.col(name).is_null() | ~pl.col(name).is_finite()).height:
            raise TransferError(f"{name!r} contains null or non-finite values")


def _diagnostics(
    correspondence: pl.DataFrame,
    source_version: str,
    target_version: str,
    kind: VariableKind,
    weight_error: float,
    conservation_error: float | None,
) -> TransferDiagnostics:
    return TransferDiagnostics(
        source_version=source_version,
        target_version=target_version,
        kind=kind,
        source_units=correspondence["source_code"].n_unique(),
        target_units=correspondence["target_code"].n_unique(),
        links=correspondence.height,
        methods=tuple(sorted(correspondence["method"].unique().to_list())),
        exact_share=float(correspondence["exact"].mean()),
        max_weight_error=weight_error,
        max_conservation_error=conservation_error,
    )


__all__ = [
    "REQUIRED_CORRESPONDENCE",
    "TransferDiagnostics",
    "TransferError",
    "VariableKind",
    "correspondence_from_support",
    "transfer_extensive",
    "transfer_intensive",
    "transfer_rate",
    "validate_correspondence",
]
