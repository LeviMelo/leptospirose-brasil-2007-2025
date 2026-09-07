"""Rate construction on the denominator tensor, with the guardrails that matter.

Standardisation is routine. The part that is not routine, and that this module
enforces rather than documents, is **numerator-denominator measurement
equivalence**: a rate is only interpretable when the numerator and the
denominator classify people the same way.

Two cases arise constantly in Brazilian routine-data epidemiology and both are
usually got wrong silently.

Colour/race
    IBGE's ``cor ou raca`` is *self-declared* by the respondent. SINAN's
    ``CS_RACA`` is *recorded by a health professional* at notification, often
    by observation and often left as ``ignorado``. These are different
    measurements of different constructs. Dividing one by the other produces
    differential misclassification whose direction is known from the
    literature: professional attribution over-assigns ``branca`` relative to
    self-declaration in populations that self-declare ``parda``, so
    colour-specific rates for ``preta``/``parda`` are biased downward and
    ``branca`` upward. Additionally SINAN's ``ignorado`` share is large and
    varies by state and year, so the bias is not even constant.

    This module therefore refuses to compute colour-stratified rates unless
    the caller passes ``acknowledge_race_measurement_gap=True``, and attaches
    the caveat to the returned frame so it survives into the output.

Age
    SINAN codes age in ``NU_IDADE_N`` with a unit prefix (hours/days/months/
    years). Decoding it as an integer without the prefix silently turns a
    3-month-old into a 3-year-old. Use
    :func:`brepi.sources.datasus.sinan.decode_age` before grouping.

Standard populations shipped here are the WHO World Standard (2000-2025) and
Segi's World Standard, both on the 5-year scheme in
:data:`brepi.denominators.tensor.AGE_GROUPS`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from brepi.denominators.tensor import AGE_GROUPS

#: WHO World Standard Population (Ahmad et al. 2001), collapsed to the 17-group
#: scheme with an 80+ terminal group. Weights sum to 1.
WHO_WORLD_STANDARD: dict[str, float] = {
    "0-4": 0.0886, "5-9": 0.0869, "10-14": 0.0860, "15-19": 0.0847,
    "20-24": 0.0822, "25-29": 0.0793, "30-34": 0.0761, "35-39": 0.0715,
    "40-44": 0.0659, "45-49": 0.0604, "50-54": 0.0537, "55-59": 0.0455,
    "60-64": 0.0372, "65-69": 0.0296, "70-74": 0.0221, "75-79": 0.0152,
    "80+": 0.0151,
}

#: Segi world standard, retained because much of the older Brazilian
#: leptospirosis literature standardises to it and comparability matters.
SEGI_WORLD_STANDARD: dict[str, float] = {
    "0-4": 0.12, "5-9": 0.10, "10-14": 0.09, "15-19": 0.09, "20-24": 0.08,
    "25-29": 0.08, "30-34": 0.06, "35-39": 0.06, "40-44": 0.06, "45-49": 0.06,
    "50-54": 0.05, "55-59": 0.04, "60-64": 0.04, "65-69": 0.03, "70-74": 0.02,
    "75-79": 0.01, "80+": 0.01,
}

RACE_MEASUREMENT_CAVEAT = (
    "Colour-stratified rates combine a professional-attributed numerator "
    "(SINAN CS_RACA) with a self-declared denominator (IBGE census). These are "
    "not measurement-equivalent; differential misclassification is expected to "
    "bias rates for preta/parda downward and branca upward, by an amount that "
    "varies with the state- and year-specific share of CS_RACA='ignorado'."
)


@dataclass(frozen=True)
class RateResult:
    frame: pl.DataFrame
    caveats: tuple[str, ...] = ()


def crude_rate(
    numerator: pl.DataFrame,
    denominator: pl.DataFrame,
    *,
    by: list[str],
    per: float = 100_000.0,
    count_col: str = "cases",
    pop_col: str = "population",
) -> pl.DataFrame:
    """Join counts to population and return a crude rate per ``per`` persons.

    Left-joins on the denominator so that strata with population but no cases
    become explicit zeros. That is not a convenience: in this study the zeros
    are the object of inquiry, and a rate table that silently omits them
    misstates both the spatial distribution and the zero-inflation structure.
    """
    joined = (
        denominator.group_by(by)
        .agg(pl.col(pop_col).sum())
        .join(numerator.group_by(by).agg(pl.col(count_col).sum()), on=by, how="left")
        .with_columns(pl.col(count_col).fill_null(0))
    )
    return joined.with_columns(
        (pl.col(count_col) / pl.col(pop_col) * per).alias("rate"),
        pl.lit(per).alias("rate_per"),
    ).sort(by)


def age_standardised_rate(
    numerator: pl.DataFrame,
    denominator: pl.DataFrame,
    *,
    by: list[str],
    standard: dict[str, float] | None = None,
    per: float = 100_000.0,
    count_col: str = "cases",
    pop_col: str = "population",
    age_col: str = "age_group",
) -> pl.DataFrame:
    """Directly age-standardised rate with an analytic variance.

    Variance follows the standard result for a weighted sum of independent
    Poisson counts, Var(ASR) = sum_i (w_i / n_i)^2 * d_i, giving a normal-
    approximation confidence interval. For strata with very few events prefer
    the model-based estimates rather than these intervals.
    """
    std = standard or WHO_WORLD_STANDARD
    missing = set(AGE_GROUPS) - set(std)
    if missing:
        raise ValueError(f"standard population lacks age groups: {sorted(missing)}")

    w = pl.DataFrame(
        {age_col: list(std.keys()), "_w": [v / sum(std.values()) for v in std.values()]}
    )
    keys = by + [age_col]
    strata = (
        denominator.group_by(keys)
        .agg(pl.col(pop_col).sum())
        .join(numerator.group_by(keys).agg(pl.col(count_col).sum()), on=keys, how="left")
        .with_columns(pl.col(count_col).fill_null(0))
        .join(w, on=age_col, how="inner")
    )
    strata = strata.with_columns(
        pl.when(pl.col(pop_col) > 0)
        .then(pl.col(count_col) / pl.col(pop_col))
        .otherwise(0.0)
        .alias("_r"),
    ).with_columns(
        (pl.col("_w") * pl.col("_r")).alias("_wr"),
        pl.when(pl.col(pop_col) > 0)
        .then((pl.col("_w") / pl.col(pop_col)) ** 2 * pl.col(count_col))
        .otherwise(0.0)
        .alias("_v"),
    )
    out = strata.group_by(by).agg(
        (pl.col("_wr").sum() * per).alias("asr"),
        (pl.col("_v").sum() * per**2).alias("_var"),
        pl.col(count_col).sum().alias(count_col),
        pl.col(pop_col).sum().alias(pop_col),
    )
    return out.with_columns(
        pl.col("_var").sqrt().alias("asr_se"),
        (pl.col("asr") - 1.96 * pl.col("_var").sqrt()).clip(lower_bound=0.0).alias("asr_lo"),
        (pl.col("asr") + 1.96 * pl.col("_var").sqrt()).alias("asr_hi"),
    ).drop("_var").sort(by)


def stratified_rate(
    numerator: pl.DataFrame,
    denominator: pl.DataFrame,
    *,
    by: list[str],
    acknowledge_race_measurement_gap: bool = False,
    **kwargs,
) -> RateResult:
    """Crude rate with a hard stop on unacknowledged colour stratification.

    Raises if ``by`` includes ``colour`` and the caller has not explicitly
    acknowledged the measurement gap. The intent is not ceremony: it is that
    the caveat should be impossible to omit from the output, because in
    practice it always is omitted.
    """
    if "colour" in by and not acknowledge_race_measurement_gap:
        raise ValueError(
            "colour-stratified rates require acknowledge_race_measurement_gap=True. "
            + RACE_MEASUREMENT_CAVEAT
        )
    frame = crude_rate(numerator, denominator, by=by, **kwargs)
    caveats: tuple[str, ...] = ()
    if "colour" in by:
        caveats = (RACE_MEASUREMENT_CAVEAT,)
        frame = frame.with_columns(pl.lit(RACE_MEASUREMENT_CAVEAT).alias("measurement_caveat"))
    return RateResult(frame, caveats)


def interpolation_sensitivity_weight(
    frame: pl.DataFrame, *, halflife_years: float = 6.0
) -> pl.DataFrame:
    """Attach a down-weight based on distance from the nearest census.

    ``exp(-d / halflife)`` on ``years_from_census``. Intended for a sensitivity
    analysis that re-fits excluding or down-weighting heavily interpolated
    denominators, not for the primary fit. The default half-life is roughly
    half an intercensal interval.
    """
    if "years_from_census" not in frame.columns:
        raise ValueError("frame lacks years_from_census; build it from the tensor")
    return frame.with_columns(
        (-pl.col("years_from_census") / halflife_years).exp().alias("denominator_weight")
    )


__all__ = [
    "WHO_WORLD_STANDARD",
    "SEGI_WORLD_STANDARD",
    "RACE_MEASUREMENT_CAVEAT",
    "RateResult",
    "crude_rate",
    "age_standardised_rate",
    "stratified_rate",
    "interpolation_sensitivity_weight",
]
