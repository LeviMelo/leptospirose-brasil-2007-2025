"""ENSO state as a national temporal covariate: ONI and monthly Nino 3.4.

ENSO enters a Brazilian epidemiological panel differently from every other
covariate here: it varies in time and not in space. It cannot identify anything
on its own in a model with month effects, and putting it in alongside a
flexible temporal term is a collinearity trap. Its uses are (i) as an
*interaction* with municipality, since El Nino dries the North and Northeast
while wetting the South -- opposite signs in the same year, which is exactly
the structure a national time term cannot absorb; and (ii) as an instrument-like
external forcing in the multi-scale strand.

Two series, and they are not interchangeable:

``oni``
    The Oceanic Nino Index. A **three-month running mean** of Nino 3.4 SST
    anomaly against *centred 30-year* base periods, so the base shifts every
    five years. This is the operational definition of an El Nino event
    (>= +0.5 for five overlapping seasons) and is what event-classification
    papers mean by ENSO phase. Its smoothing means it is not a monthly series
    dressed up as one -- consecutive values share two months of data.

``nino34``
    Genuinely monthly Nino 3.4 SST and anomaly on a fixed 1991-2020 base. Use
    this when the lag structure matters, because the ONI's running mean
    pre-smooths the exposure and biases an estimated lag toward zero.

Both come from NOAA CPC as fixed-width ASCII and are updated monthly.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import polars as pl

from brepi.config import (
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    NOAA_NINO_MONTHLY_URL,
    NOAA_ONI_URL,
)
from brepi.io import cache

TOOL_VERSION = "brepi.sources.climate.enso/1"
SOURCE_ID = "noaa.cpc.enso"

#: ONI season codes to the *centre* month of the three-month window. DJF is
#: labelled with January: the season straddles the year boundary and NOAA's
#: ``YR`` column is the year of the centre month, not of December.
SEASON_CENTRE_MONTH: dict[str, int] = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}

#: NOAA's operational El Nino / La Nina thresholds on the ONI, in degC.
EVENT_THRESHOLD = 0.5
#: Consecutive overlapping seasons required before NOAA declares an event.
EVENT_MIN_SEASONS = 5


class EnsoError(RuntimeError):
    """The ENSO series could not be parsed into the expected shape."""


def _http_get(url: str) -> tuple[bytes, dict[str, Any]]:
    import time

    import httpx

    last: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response.content, {
                    "remote_modified": response.headers.get("last-modified"),
                    "params": {"url": str(response.url), "attempt": attempt},
                }
            if response.status_code < 500 and response.status_code != 429:
                raise EnsoError(f"{url} returned HTTP {response.status_code}")
            last = EnsoError(f"{url} returned HTTP {response.status_code}")
        if attempt < HTTP_MAX_RETRIES:
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise EnsoError(f"giving up on {url}") from last


def _load(url: str, key: str, *, refresh: bool = False) -> list[str]:
    path, _prov = cache.fetch(
        key,
        lambda: _http_get(url),
        source=SOURCE_ID,
        uri=url,
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )
    return path.read_text(encoding="ascii", errors="replace").splitlines()


def oni(*, refresh: bool = False) -> pl.DataFrame:
    """The Oceanic Nino Index as a monthly frame.

    Columns: ``period`` (first of the season's *centre* month), ``season``,
    ``sst`` (absolute Nino 3.4 SST, degC), ``oni`` (the anomaly), and
    ``phase`` in ``{"el_nino", "la_nina", "neutral"}`` by the +/-0.5 threshold
    applied to the single season. Note that ``phase`` here is the per-season
    threshold crossing; NOAA only declares an *event* after five consecutive
    such seasons, which :func:`events` applies.
    """
    lines = _load(NOAA_ONI_URL, "climate/enso/oni.ascii.txt", refresh=refresh)
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SEASON_CENTRE_MONTH:
            continue
        season, year, total, anom = parts
        rows.append(
            {
                "period": date(int(year), SEASON_CENTRE_MONTH[season], 1),
                "season": season,
                "sst": float(total),
                "oni": float(anom),
            }
        )
    if not rows:
        raise EnsoError(f"parsed no ONI rows from {NOAA_ONI_URL}; the layout changed")

    frame = pl.DataFrame(rows).unique(subset=["period"], keep="last").sort("period")
    _assert_contiguous(frame)
    return frame.with_columns(
        pl.when(pl.col("oni") >= EVENT_THRESHOLD)
        .then(pl.lit("el_nino"))
        .when(pl.col("oni") <= -EVENT_THRESHOLD)
        .then(pl.lit("la_nina"))
        .otherwise(pl.lit("neutral"))
        .alias("phase")
    )


def nino34(*, refresh: bool = False) -> pl.DataFrame:
    """Genuinely monthly Nino region SSTs and anomalies (1991-2020 base).

    Columns ``period``, ``nino34_sst``, ``nino34_anom``, plus Nino 1+2, 3 and 4
    where the file supplies them. Prefer this over :func:`oni` whenever a lag
    is being estimated.
    """
    lines = _load(NOAA_NINO_MONTHLY_URL, "climate/enso/nino34_monthly.ascii", refresh=refresh)
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if header is None:
            if parts[0].upper().startswith("YR"):
                header = [p.upper() for p in parts]
            continue
        if not parts[0].isdigit():
            continue
        values = dict(zip(header, parts))
        try:
            year, month = int(values["YR"]), int(values["MON"])
        except (KeyError, ValueError):
            continue
        row: dict[str, Any] = {"period": date(year, month, 1)}
        for label, column in (
            ("NINO1+2", "nino12"),
            ("ANOM", None),
            ("NINO3", "nino3"),
            ("NINO4", "nino4"),
            ("NINO3.4", "nino34"),
        ):
            if label in values and column:
                row[f"{column}_sst"] = float(values[label])
        # The CPC layout repeats an unqualified ANOM column after each region;
        # split() collapses them, so recover anomalies positionally instead.
        numeric = [p for p in parts[2:]]
        if len(numeric) >= 8:
            row["nino12_anom"] = float(numeric[1])
            row["nino3_anom"] = float(numeric[3])
            row["nino4_anom"] = float(numeric[5])
            row["nino34_sst"] = float(numeric[6])
            row["nino34_anom"] = float(numeric[7])
        rows.append(row)
    if not rows:
        raise EnsoError(f"parsed no rows from {NOAA_NINO_MONTHLY_URL}; the layout changed")
    frame = pl.DataFrame(rows).unique(subset=["period"], keep="last").sort("period")
    _assert_contiguous(frame)
    return frame


def _assert_contiguous(frame: pl.DataFrame) -> None:
    periods = frame["period"].to_list()
    for previous, current in zip(periods, periods[1:]):
        expected = (
            date(previous.year + 1, 1, 1)
            if previous.month == 12
            else date(previous.year, previous.month + 1, 1)
        )
        if current != expected:
            raise EnsoError(
                f"ENSO series jumps {previous} -> {current}; a gap in a national "
                "temporal covariate would silently misalign every lag."
            )


def events(series: pl.DataFrame | None = None) -> pl.DataFrame:
    """NOAA-definition El Nino / La Nina events.

    An event is at least :data:`EVENT_MIN_SEASONS` consecutive overlapping
    seasons past +/-0.5. Returns ``start``, ``end``, ``phase``, ``n_seasons``,
    ``peak``. This is the classification to use when a paper says "El Nino
    years", because a single season past the threshold is not one.
    """
    frame = series if series is not None else oni()
    rows = frame.select("period", "oni", "phase").to_dicts()
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []

    def flush() -> None:
        if len(run) >= EVENT_MIN_SEASONS and run[0]["phase"] != "neutral":
            peak = max(run, key=lambda r: abs(r["oni"]))
            out.append(
                {
                    "phase": run[0]["phase"],
                    "start": run[0]["period"],
                    "end": run[-1]["period"],
                    "n_seasons": len(run),
                    "peak": peak["oni"],
                    "peak_period": peak["period"],
                }
            )

    for row in rows:
        if run and row["phase"] == run[-1]["phase"]:
            run.append(row)
            continue
        flush()
        run = [row]
    flush()
    return pl.DataFrame(out) if out else pl.DataFrame(
        schema={
            "phase": pl.Utf8,
            "start": pl.Date,
            "end": pl.Date,
            "n_seasons": pl.Int64,
            "peak": pl.Float64,
            "peak_period": pl.Date,
        }
    )


def monthly_covariate(
    start: date,
    end: date,
    *,
    kind: Literal["oni", "nino34"] = "oni",
    lags: tuple[int, ...] = (0, 3, 6),
    refresh: bool = False,
) -> pl.DataFrame:
    """A ``period``-keyed ENSO covariate table with lags, for the panel spine.

    Join on ``period`` alone: this varies in time only. Lags are in months and
    are *backward* (``oni_lag3`` at 2024-05 is the value at 2024-02), which is
    the sign convention the DLNM crossbasis expects.
    """
    frame = oni(refresh=refresh) if kind == "oni" else nino34(refresh=refresh)
    value_col = "oni" if kind == "oni" else "nino34_anom"
    if value_col not in frame.columns:
        raise EnsoError(f"{kind} series lacks {value_col}")

    base = frame.select("period", pl.col(value_col).alias(kind))
    for lag in lags:
        if lag == 0:
            continue
        base = base.with_columns(pl.col(kind).shift(lag).alias(f"{kind}_lag{lag}"))
    out = base.filter(pl.col("period").is_between(start, end))
    expected = (end.year - start.year) * 12 + (end.month - start.month) + 1
    if out.height != expected:
        raise EnsoError(
            f"{kind} covers {out.height} of the {expected} months in "
            f"{start}..{end}; the upstream series has not been updated that far."
        )
    return out


__all__ = [
    "EVENT_MIN_SEASONS",
    "EVENT_THRESHOLD",
    "SEASON_CENTRE_MONTH",
    "EnsoError",
    "events",
    "monthly_covariate",
    "nino34",
    "oni",
]
