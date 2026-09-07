"""Derived monthly climate exposure indices, and what cannot be derived.

The distributed-lag models this feeds do not want "rainfall". They want a small
set of physically distinct exposures, because the lag structure differs between
them: leptospirosis transmission responds to *inundation* (a single extreme day,
or a run of wet days saturating the ground) with a lag of one to three weeks,
and to *seasonal accumulation* on a longer and much weaker lag. Collapsing both
into monthly total precipitation is the standard mistake and it attenuates the
extreme-rainfall coefficient toward the seasonal one.

Two design commitments
----------------------

**Pure functions over daily arrays.** Every index here is a function of a
numpy array of daily values and returns a number. No I/O, no frame schema, no
product knowledge. That is what makes them testable against hand-computed
answers, which :mod:`tests.test_climate` does.

**The module refuses to fake daily-resolution indices.** RX1day, RX5day, CWD,
CDD, R1, R10, R20 and R50 are *not* recoverable from a monthly total. There is
no assumption about within-month distribution that makes them recoverable --
the same 200 mm month is one 200 mm day or twenty 10 mm days, and those are
opposite exposures for leptospirosis. :data:`DAILY_ONLY_INDICES` names them and
:func:`monthly_indices_from_monthly` raises rather than estimating them. Several
published Brazilian leptospirosis papers report "extreme rainfall days" derived
from monthly products; that quantity does not exist.

Indices computed
----------------

===============  ========================================================
``pr_total``     monthly precipitation total, mm
``r1``           wet days, >= 1 mm
``r10``          heavy precipitation days, >= 10 mm
``r20``          very heavy precipitation days, >= 20 mm
``r50``          extremely heavy precipitation days, >= 50 mm
``rx1day``       wettest single day, mm
``rx5day``       wettest 5 consecutive days, mm
``cwd``          longest run of consecutive wet days (>= 1 mm)
``cdd``          longest run of consecutive dry days (< 1 mm)
``sdii``         simple daily intensity index, mm per wet day
``tmax_mean``    mean daily maximum temperature, degC
``tmax_max``     highest daily maximum temperature, degC
``tmin_mean``    mean daily minimum temperature, degC
``tmin_min``     lowest daily minimum temperature, degC
``dtr``          mean diurnal temperature range, degC
===============  ========================================================

Thresholds follow the ETCCDI definitions (Zhang et al. 2011), except R50 which
ETCCDI leaves user-defined; 50 mm/day is the Brazilian operational
heavy-rainfall alert threshold and is what the disaster literature uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

#: Wet-day threshold, mm. ETCCDI. Below this a tipping-bucket gauge and an
#: interpolated grid disagree more than they agree, which is why the standard
#: does not use ">0".
WET_DAY_MM = 1.0

PRECIP_DAY_THRESHOLDS: tuple[float, ...] = (1.0, 10.0, 20.0, 50.0)

#: Indices that are functions of the within-month daily sequence and are
#: therefore NOT recoverable from a monthly aggregate. Not "hard to recover" --
#: not recoverable. Any pipeline that reports one of these from a monthly
#: product has invented it.
DAILY_ONLY_INDICES: frozenset[str] = frozenset(
    {"r1", "r10", "r20", "r50", "rx1day", "rx5day", "cwd", "cdd", "sdii", "tmax_max", "tmin_min", "dtr"}
)

#: Indices a monthly product can legitimately supply.
MONTHLY_SAFE_INDICES: frozenset[str] = frozenset({"pr_total", "tmax_mean", "tmin_mean"})

#: The WMO-recommended climatological normal period, and the one the Brazilian
#: national meteorological service publishes against.
CLIMATOLOGY_BASE: tuple[int, int] = (1991, 2020)


class IndexError_(ValueError):
    """An index was requested from data that cannot support it."""


# --------------------------------------------------------------------------
# Pure scalar indices over a daily array
# --------------------------------------------------------------------------


def _clean(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("daily series must be one-dimensional")
    return arr


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def total_precipitation(daily_mm: Sequence[float] | np.ndarray) -> float:
    """Sum of daily precipitation, mm. NaNs are skipped, not zero-filled."""
    arr = _finite(_clean(daily_mm))
    if arr.size == 0:
        return float("nan")
    if (arr < 0).any():
        raise ValueError("negative precipitation in daily series")
    return float(arr.sum())


def wet_days(daily_mm: Sequence[float] | np.ndarray, threshold: float = WET_DAY_MM) -> int:
    """Count of days at or above ``threshold`` mm (ETCCDI R-nn)."""
    arr = _finite(_clean(daily_mm))
    if (arr < 0).any():
        raise ValueError("negative precipitation in daily series")
    return int((arr >= threshold).sum())


def rx1day(daily_mm: Sequence[float] | np.ndarray) -> float:
    """Wettest single day in the window, mm."""
    arr = _finite(_clean(daily_mm))
    return float(arr.max()) if arr.size else float("nan")


def rxnday(daily_mm: Sequence[float] | np.ndarray, window: int = 5) -> float:
    """Wettest ``window``-day running total, mm (ETCCDI RX5day at window=5).

    The running total is taken over the window as given, so a caller that wants
    RX5day for a calendar month and cares about spells straddling the month
    boundary must pass a padded series. :func:`monthly_indices_from_daily`
    pads, because a flood that starts on the 29th is not a next-month flood.
    """
    arr = _clean(daily_mm)
    if window < 1:
        raise ValueError("window must be positive")
    if arr.size < window:
        return float("nan")
    filled = np.nan_to_num(arr, nan=0.0)
    kernel = np.ones(window)
    sums = np.convolve(filled, kernel, mode="valid")
    return float(sums.max())


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def consecutive_wet_days(
    daily_mm: Sequence[float] | np.ndarray, threshold: float = WET_DAY_MM
) -> int:
    """Longest run of consecutive days with >= ``threshold`` mm (ETCCDI CWD).

    NaN days break a run rather than extending it, which is the conservative
    choice: an unobserved day is not evidence of rain.
    """
    arr = _clean(daily_mm)
    return _longest_run(np.where(np.isnan(arr), False, arr >= threshold))


def consecutive_dry_days(
    daily_mm: Sequence[float] | np.ndarray, threshold: float = WET_DAY_MM
) -> int:
    """Longest run of consecutive days with < ``threshold`` mm (ETCCDI CDD).

    NaN days break the run, symmetrically with :func:`consecutive_wet_days`.
    """
    arr = _clean(daily_mm)
    return _longest_run(np.where(np.isnan(arr), False, arr < threshold))


def sdii(daily_mm: Sequence[float] | np.ndarray, threshold: float = WET_DAY_MM) -> float:
    """Simple daily intensity index: mean precipitation on wet days, mm/day.

    Separates "it rained a lot" from "it rained hard", which monthly totals
    cannot. Undefined (NaN) in a month with no wet day.
    """
    arr = _finite(_clean(daily_mm))
    wet = arr[arr >= threshold]
    return float(wet.mean()) if wet.size else float("nan")


def mean_ignoring_nan(daily: Sequence[float] | np.ndarray) -> float:
    arr = _finite(_clean(daily))
    return float(arr.mean()) if arr.size else float("nan")


def max_ignoring_nan(daily: Sequence[float] | np.ndarray) -> float:
    arr = _finite(_clean(daily))
    return float(arr.max()) if arr.size else float("nan")


def min_ignoring_nan(daily: Sequence[float] | np.ndarray) -> float:
    arr = _finite(_clean(daily))
    return float(arr.min()) if arr.size else float("nan")


# --------------------------------------------------------------------------
# Standardised anomalies and SPI
# --------------------------------------------------------------------------


def standardised_anomaly(
    value: float, climatology: Sequence[float] | np.ndarray, *, ddof: int = 1
) -> float:
    """``(value - mean) / sd`` against a same-calendar-month climatology.

    The climatology must be *the same calendar month* across years. Pooling all
    twelve months makes a wet-season month look like a permanent +2 sigma
    anomaly and a dry-season month like a permanent -1, which is a seasonal
    cycle wearing an anomaly's clothes.
    """
    ref = _finite(_clean(climatology))
    if ref.size < 10:
        raise IndexError_(
            f"climatology has {ref.size} years; a standardised anomaly on fewer "
            "than 10 is not interpretable. WMO normals use 30."
        )
    sd = float(ref.std(ddof=ddof))
    if sd == 0:
        return float("nan")
    return float((value - float(ref.mean())) / sd)


def spi(
    value: float, climatology: Sequence[float] | np.ndarray, *, ddof: int = 1
) -> float:
    """Standardised Precipitation Index for one accumulation.

    Fits a two-parameter gamma to the non-zero climatology by the Thom (1958)
    approximate maximum-likelihood estimator, mixes in the empirical
    probability of zero, and maps the resulting non-exceedance probability
    through the standard normal quantile. This is the McKee et al. (1993)
    definition as operationalised by WMO, *not* a z-score of raw totals:
    monthly rainfall is strongly right-skewed and a z-score of it systematically
    understates drought and overstates flood.

    ``spi(x, monthly_totals_for_that_calendar_month)`` gives SPI-1. Pass
    3-month rolling totals for SPI-3 -- :func:`spi_series` does the bookkeeping.
    """
    from scipy import stats

    ref = _finite(_clean(climatology))
    if ref.size < 20:
        raise IndexError_(
            f"SPI fitted on {ref.size} years; the gamma fit is not stable below "
            "20 and WMO recommends 30. Use standardised_anomaly() if the record "
            "is genuinely short, and say so."
        )
    if (ref < 0).any() or (np.isfinite(value) and value < 0):
        raise ValueError("SPI is defined on non-negative accumulations")
    if not np.isfinite(value):
        return float("nan")

    n = ref.size
    positive = ref[ref > 0]
    q = (n - positive.size) / n  # empirical P(zero)
    if positive.size < 10:
        raise IndexError_("fewer than 10 non-zero years; gamma fit is meaningless")

    mean = float(positive.mean())
    big_a = math.log(mean) - float(np.log(positive).mean())
    if big_a <= 0:
        return float("nan")
    shape = (1.0 + math.sqrt(1.0 + 4.0 * big_a / 3.0)) / (4.0 * big_a)
    scale = mean / shape

    if value <= 0:
        prob = q
    else:
        prob = q + (1.0 - q) * float(stats.gamma.cdf(value, a=shape, scale=scale))
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return float(stats.norm.ppf(prob))


def spi_series(
    totals: Sequence[float] | np.ndarray,
    months: Sequence[int],
    *,
    accumulation: int = 1,
    base_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """SPI-``accumulation`` for a single municipality's monthly total series.

    ``totals`` and ``months`` are a contiguous monthly series and its calendar
    months. The gamma is fitted separately for each calendar month, on the
    subset selected by ``base_mask`` (the climatology window; default all).
    The first ``accumulation - 1`` values are NaN because their accumulation
    window is incomplete -- not zero, and not back-filled.
    """
    values = _clean(totals)
    month_arr = np.asarray(months, dtype=int)
    if values.size != month_arr.size:
        raise ValueError("totals and months differ in length")
    if accumulation < 1:
        raise ValueError("accumulation must be positive")

    if accumulation == 1:
        acc = values.copy()
    else:
        acc = np.full(values.size, np.nan)
        cumulative = np.cumsum(np.nan_to_num(values, nan=0.0))
        acc[accumulation - 1 :] = cumulative[accumulation - 1 :] - np.concatenate(
            ([0.0], cumulative[: -accumulation])
        )
        nan_mask = np.isnan(values)
        for i in range(accumulation - 1, values.size):
            if nan_mask[i - accumulation + 1 : i + 1].any():
                acc[i] = np.nan

    mask = (
        np.ones(values.size, dtype=bool)
        if base_mask is None
        else np.asarray(base_mask, dtype=bool)
    )
    out = np.full(values.size, np.nan)
    for m in range(1, 13):
        same = month_arr == m
        ref = acc[same & mask]
        ref = ref[np.isfinite(ref)]
        if ref.size < 20:
            continue
        for i in np.flatnonzero(same):
            if np.isfinite(acc[i]):
                out[i] = spi(float(acc[i]), ref)
    return out


# --------------------------------------------------------------------------
# Frame-level assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSpec:
    """One monthly index and the daily variable it consumes."""

    name: str
    variable: str
    daily_only: bool


INDEX_SPECS: tuple[IndexSpec, ...] = (
    IndexSpec("pr_total", "pr", False),
    IndexSpec("r1", "pr", True),
    IndexSpec("r10", "pr", True),
    IndexSpec("r20", "pr", True),
    IndexSpec("r50", "pr", True),
    IndexSpec("rx1day", "pr", True),
    IndexSpec("rx5day", "pr", True),
    IndexSpec("cwd", "pr", True),
    IndexSpec("cdd", "pr", True),
    IndexSpec("sdii", "pr", True),
    IndexSpec("tmax_mean", "tmax", False),
    IndexSpec("tmax_max", "tmax", True),
    IndexSpec("tmin_mean", "tmin", False),
    IndexSpec("tmin_min", "tmin", True),
)


def monthly_indices_from_daily(
    daily: pl.DataFrame,
    *,
    variables: Sequence[str] = ("pr", "tmax", "tmin"),
    stat: str = "mean",
    require_complete_months: bool = True,
) -> pl.DataFrame:
    """Reduce a long daily frame to the monthly exposure indices.

    ``daily`` is what :func:`brepi.sources.climate.brdwgd.chained_daily`
    returns: ``munic_code``, ``date``, ``variable``, ``stat``, ``value``,
    ``product``. Returns one row per ``munic_code`` x ``period``.

    RX5day is computed on a series padded with the last four days of the
    previous month, so a five-day deluge spanning a month boundary is credited
    to the month it ends in rather than being cut in half. ``n_days`` and
    ``n_days_expected`` are carried so a short month is visible; with
    ``require_complete_months`` a month missing days raises, because a monthly
    total computed over 24 days is not a monthly total.
    """
    required = {"munic_code", "date", "variable", "stat", "value"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame lacks {sorted(missing)}")
    if daily.is_empty():
        raise ValueError("daily frame is empty")

    frame = daily.filter(pl.col("stat") == stat)
    if frame.is_empty():
        raise ValueError(f"no rows with stat={stat!r}")

    products = (
        frame.with_columns(pl.col("date").dt.truncate("1mo").alias("period"))
        .group_by(["munic_code", "period"])
        .agg(pl.col("product").unique().sort().str.join("+").alias("product"))
        if "product" in frame.columns
        else None
    )

    wide = frame.pivot(
        on="variable", index=["munic_code", "date"], values="value", aggregate_function="first"
    ).sort(["munic_code", "date"])

    out_rows: list[dict[str, object]] = []
    for (code,), block in wide.group_by(["munic_code"], maintain_order=True):
        dates = block["date"].to_list()
        by_var = {v: block[v].to_numpy() for v in variables if v in block.columns}
        periods = sorted({date(d.year, d.month, 1) for d in dates})
        index_of = {d: i for i, d in enumerate(dates)}
        for period in periods:
            sel = [i for i, d in enumerate(dates) if d.year == period.year and d.month == period.month]
            first = sel[0]
            row: dict[str, object] = {
                "munic_code": code,
                "period": period,
                "n_days": len(sel),
                "n_days_expected": _days_in_month(period),
            }
            if "pr" in by_var:
                pr = by_var["pr"][sel]
                pad_from = max(0, first - 4)
                pr_padded = by_var["pr"][pad_from : sel[-1] + 1]
                row.update(
                    pr_total=total_precipitation(pr),
                    r1=wet_days(pr, 1.0),
                    r10=wet_days(pr, 10.0),
                    r20=wet_days(pr, 20.0),
                    r50=wet_days(pr, 50.0),
                    rx1day=rx1day(pr),
                    rx5day=rxnday(pr_padded, 5),
                    cwd=consecutive_wet_days(pr),
                    cdd=consecutive_dry_days(pr),
                    sdii=sdii(pr),
                )
            if "tmax" in by_var:
                tx = by_var["tmax"][sel]
                row.update(tmax_mean=mean_ignoring_nan(tx), tmax_max=max_ignoring_nan(tx))
            if "tmin" in by_var:
                tn = by_var["tmin"][sel]
                row.update(tmin_mean=mean_ignoring_nan(tn), tmin_min=min_ignoring_nan(tn))
            if "tmax" in by_var and "tmin" in by_var:
                row["dtr"] = mean_ignoring_nan(by_var["tmax"][sel] - by_var["tmin"][sel])
            out_rows.append(row)

    result = pl.DataFrame(out_rows).sort(["munic_code", "period"])
    if require_complete_months:
        short = result.filter(pl.col("n_days") < pl.col("n_days_expected"))
        if short.height:
            raise ValueError(
                f"{short.height} municipality-months are incomplete, e.g. "
                f"{short.head(3).select('munic_code', 'period', 'n_days', 'n_days_expected').to_dicts()}. "
                "A monthly total over a partial month is not a monthly total. "
                "Pass require_complete_months=False only if you will carry n_days "
                "into the model."
            )
    if products is not None:
        result = result.join(products, on=["munic_code", "period"], how="left")
    _ = index_of  # kept for readability of the padding logic above
    return result


def _days_in_month(period: date) -> int:
    import calendar

    return calendar.monthrange(period.year, period.month)[1]


def monthly_indices_from_monthly(
    monthly: pl.DataFrame,
    *,
    requested: Iterable[str] = ("pr_total",),
) -> pl.DataFrame:
    """Pass through the indices a monthly product can legitimately supply.

    Raises on any member of :data:`DAILY_ONLY_INDICES`. This function exists
    precisely so that the refusal is explicit and greppable rather than being
    an omission somebody later "fixes" by fabricating an estimate.
    """
    requested = list(requested)
    impossible = [name for name in requested if name in DAILY_ONLY_INDICES]
    if impossible:
        raise IndexError_(
            f"{impossible} cannot be computed from monthly aggregates. The same "
            "monthly total is produced by one 200 mm day and by twenty 10 mm "
            "days, and for leptospirosis those are opposite exposures. Use a "
            "daily product (brdwgd or era5land) or drop the index; do not "
            "approximate it."
        )
    unknown = [n for n in requested if n not in MONTHLY_SAFE_INDICES]
    if unknown:
        raise IndexError_(f"unknown index/indices {unknown}")
    keep = [c for c in monthly.columns if c in {"munic_code", "period", *requested}]
    return monthly.select(keep)


def attach_anomalies(
    monthly: pl.DataFrame,
    *,
    value_col: str = "pr_total",
    base: tuple[int, int] = CLIMATOLOGY_BASE,
    out_prefix: str | None = None,
    min_base_years: int = 20,
) -> pl.DataFrame:
    """Attach a municipal, calendar-month standardised anomaly.

    Adds ``{prefix}_clim_mean``, ``{prefix}_clim_sd``, ``{prefix}_anom_z`` and
    ``{prefix}_base_years``. The climatology is computed per municipality per
    calendar month over ``base`` and applied to the whole series, so an anomaly
    in 2024 is measured against 1991-2020 and not against a window that
    includes 2024.

    Municipalities with fewer than ``min_base_years`` base years get null
    anomalies rather than a z-score computed on a handful of years.
    """
    prefix = out_prefix or value_col
    with_month = monthly.with_columns(
        pl.col("period").dt.year().alias("_y"), pl.col("period").dt.month().alias("_m")
    )
    clim = (
        with_month.filter(pl.col("_y").is_between(base[0], base[1]))
        .group_by(["munic_code", "_m"])
        .agg(
            pl.col(value_col).mean().alias(f"{prefix}_clim_mean"),
            pl.col(value_col).std(ddof=1).alias(f"{prefix}_clim_sd"),
            pl.col(value_col).count().alias(f"{prefix}_base_years"),
        )
    )
    joined = with_month.join(clim, on=["munic_code", "_m"], how="left")
    return joined.with_columns(
        pl.when(
            (pl.col(f"{prefix}_base_years") >= min_base_years)
            & (pl.col(f"{prefix}_clim_sd") > 0)
        )
        .then(
            (pl.col(value_col) - pl.col(f"{prefix}_clim_mean"))
            / pl.col(f"{prefix}_clim_sd")
        )
        .otherwise(None)
        .alias(f"{prefix}_anom_z")
    ).drop("_y", "_m")


def attach_spi(
    monthly: pl.DataFrame,
    *,
    value_col: str = "pr_total",
    accumulations: Sequence[int] = (1, 3),
    base: tuple[int, int] = CLIMATOLOGY_BASE,
) -> pl.DataFrame:
    """Attach SPI-n columns, fitted per municipality per calendar month on ``base``.

    Requires a contiguous monthly series per municipality; a gap would make the
    n-month accumulation silently span the wrong months, so it raises.
    """
    frame = monthly.sort(["munic_code", "period"])
    results: dict[int, list[float]] = {n: [] for n in accumulations}
    order: list[tuple[str, date]] = []
    for (code,), block in frame.group_by(["munic_code"], maintain_order=True):
        periods = block["period"].to_list()
        _assert_contiguous_months(code, periods)
        totals = block[value_col].to_numpy()
        months = [p.month for p in periods]
        years = np.array([p.year for p in periods])
        mask = (years >= base[0]) & (years <= base[1])
        for n in accumulations:
            results[n].extend(spi_series(totals, months, accumulation=n, base_mask=mask).tolist())
        order.extend((code, p) for p in periods)

    add = pl.DataFrame(
        {
            "munic_code": [c for c, _ in order],
            "period": [p for _, p in order],
            **{f"spi{n}": results[n] for n in accumulations},
        }
    )
    return frame.join(add, on=["munic_code", "period"], how="left")


def _assert_contiguous_months(code: str, periods: Sequence[date]) -> None:
    for previous, current in zip(periods, periods[1:]):
        expected = (
            date(previous.year + 1, 1, 1)
            if previous.month == 12
            else date(previous.year, previous.month + 1, 1)
        )
        if current != expected:
            raise ValueError(
                f"municipality {code}: monthly series jumps {previous} -> {current}. "
                "SPI accumulations over a gap span the wrong months."
            )


__all__ = [
    "CLIMATOLOGY_BASE",
    "DAILY_ONLY_INDICES",
    "INDEX_SPECS",
    "MONTHLY_SAFE_INDICES",
    "PRECIP_DAY_THRESHOLDS",
    "WET_DAY_MM",
    "IndexSpec",
    "IndexError_",
    "attach_anomalies",
    "attach_spi",
    "consecutive_dry_days",
    "consecutive_wet_days",
    "max_ignoring_nan",
    "mean_ignoring_nan",
    "min_ignoring_nan",
    "monthly_indices_from_daily",
    "monthly_indices_from_monthly",
    "rx1day",
    "rxnday",
    "sdii",
    "spi",
    "spi_series",
    "standardised_anomaly",
    "total_precipitation",
    "wet_days",
]
