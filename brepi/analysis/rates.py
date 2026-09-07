"""Exact interval estimation for counts and proportions, on polars frames.

The R side of this study already carries these (``R/11_tables.R``); this is the
Python counterpart, so that a table assembled during data preparation does not
have to be handed to R merely to acquire a confidence interval.

Two families, and the distinction matters:

* **Counts in person-time** -- an incidence rate. The exact interval inverts
  the Poisson test and is expressible in closed form through the gamma
  quantile. Normal approximations fail precisely where epidemiology is most
  interesting: a municipality with two cases in nineteen years.
* **Counts out of a fixed denominator** -- a proportion, e.g. case fatality.
  Clopper-Pearson inverts the binomial test through the beta quantile. Wald
  intervals on a CFR of 0/8 give the width-zero interval [0, 0], which is a
  statement no one intends to make.

Zero numerators are handled explicitly rather than falling out of the maths:
the lower bound of an exact Poisson interval at ``x = 0`` is 0, not the gamma
quantile at shape 0 (undefined). Getting this wrong yields ``NaN`` in exactly
the rows a reader inspects first.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

__all__ = [
    "poisson_ci", "binom_ci", "rate_table", "proportion_table",
]


def poisson_ci(
    count: np.ndarray | list[int] | int,
    person_time: np.ndarray | list[float] | float,
    *,
    conf: float = 0.95,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact (Garwood) Poisson rate and interval.

    Returns ``(rate, lower, upper)`` already multiplied by ``scale`` -- pass
    ``scale=1e5`` for a rate per 100,000 person-time units.

    ``person_time`` may be length-1 and is recycled, which is the common case
    of a shared denominator across strata.
    """
    x = np.asarray(count, dtype=float)
    pt = np.asarray(person_time, dtype=float)
    if pt.size == 1 and x.size != 1:
        pt = np.repeat(pt, x.size)
    if x.shape != pt.shape:
        raise ValueError(
            f"count has shape {x.shape} but person_time has shape {pt.shape}; "
            "pass matching lengths or a single shared denominator")

    a = (1.0 - conf) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # x = 0 has an exact lower bound of zero; stats.gamma.ppf(a, 0) is not
        # defined and would propagate NaN into every zero-count row.
        lo = np.where(x > 0, stats.gamma.ppf(a, np.maximum(x, 1e-12)), 0.0)
        hi = stats.gamma.ppf(1.0 - a, x + 1.0)
        rate = np.where(pt > 0, x / pt, np.nan)
        lo = np.where(pt > 0, lo / pt, np.nan)
        hi = np.where(pt > 0, hi / pt, np.nan)
    return rate * scale, lo * scale, hi * scale


def binom_ci(
    count: np.ndarray | list[int] | int,
    total: np.ndarray | list[int] | int,
    *,
    conf: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clopper-Pearson proportion and exact interval.

    Returns ``(proportion, lower, upper)``. A zero denominator yields ``NaN``
    throughout rather than a silent zero: "no cases with known outcome" and "a
    case fatality of zero" are different findings and must not share an
    encoding.
    """
    x = np.asarray(count, dtype=float)
    n = np.asarray(total, dtype=float)
    if n.size == 1 and x.size != 1:
        n = np.repeat(n, x.size)
    if x.shape != n.shape:
        raise ValueError(
            f"count has shape {x.shape} but total has shape {n.shape}")
    if np.any(x[n > 0] > n[n > 0]):
        raise ValueError("count exceeds total in at least one row")

    a = (1.0 - conf) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, x / n, np.nan)
        lo = np.where((n > 0) & (x > 0), stats.beta.ppf(a, x, n - x + 1), 0.0)
        hi = np.where((n > 0) & (x < n), stats.beta.ppf(1 - a, x + 1, n - x), 1.0)
        lo = np.where(n > 0, lo, np.nan)
        hi = np.where(n > 0, hi, np.nan)
    return p, lo, hi


def rate_table(
    df: pl.DataFrame,
    *,
    count: str,
    person_time: str,
    prefix: str,
    conf: float = 0.95,
    scale: float = 1e5,
) -> pl.DataFrame:
    """Attach ``{prefix}``, ``{prefix}_lo``, ``{prefix}_hi`` to ``df``."""
    r, lo, hi = poisson_ci(df[count].to_numpy(), df[person_time].to_numpy(),
                           conf=conf, scale=scale)
    return df.with_columns([
        pl.Series(prefix, r),
        pl.Series(f"{prefix}_lo", lo),
        pl.Series(f"{prefix}_hi", hi),
    ])


def proportion_table(
    df: pl.DataFrame,
    *,
    count: str,
    total: str,
    prefix: str,
    conf: float = 0.95,
) -> pl.DataFrame:
    """Attach ``{prefix}``, ``{prefix}_lo``, ``{prefix}_hi`` to ``df``."""
    p, lo, hi = binom_ci(df[count].to_numpy(), df[total].to_numpy(), conf=conf)
    return df.with_columns([
        pl.Series(prefix, p),
        pl.Series(f"{prefix}_lo", lo),
        pl.Series(f"{prefix}_hi", hi),
    ])
