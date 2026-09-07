"""Municipality-level daily weather: acquisition and monthly aggregation.

What this module buys, and what it refuses to buy
-------------------------------------------------

The expensive part of a climate exposure layer is *zonal statistics*: taking a
0.1-degree daily grid and reducing it, for every one of 5570 municipal
polygons and every one of ~24000 days, to a per-municipality daily value. Doing
that from the raw BR-DWGD NetCDF is ~10^11 cell-polygon evaluations and a
multi-day job. It has already been done, published, peer-reviewed
(Saldanha et al., *Environmental Data Science* 2024) and put on a CDN as
parquet. This module therefore **consumes precomputed municipal zonal
statistics** and does not reimplement them.

The catch, established 2026-07-30 and the single most important fact here, is
that no *one* published municipal product spans 2007-2025:

===================  ======  ==========================  =================
product              grain   VERIFIED coverage           route
===================  ======  ==========================  =================
brdwgd_cdn           daily   1961-01-01 .. 2020-07-31    DigitalOcean CDN
brdwgd               daily   1961-01-01 .. 2024-03-20    Zenodo 13906834
terraclimate_cdn     month   1958-01-01 .. 2021-12-01    DigitalOcean CDN
era5land             daily   1950-01-01 .. 2025-12-31    Zenodo (4 records)
===================  ======  ==========================  =================

So the panel is necessarily *spliced*: BR-DWGD (a gauge-interpolated
observational product, and the right answer for Brazilian precipitation)
through 2024-03-20, ERA5-Land (a reanalysis, wetter and smoother, but the only
thing that reaches 2025) after that. Every row this module emits carries a
``product`` column naming which one it came from, because a distributed-lag
model fitted across an unlabelled product change will happily attribute the
discontinuity to the exposure.

:func:`bias_adjustment` estimates the correction that makes ERA5-Land
comparable to BR-DWGD, per municipality and per calendar month, from the
seventeen-year overlap the two products share. Use it, or control for
``product``; do not silently concatenate.

Routes
------

The CDN mirror tolerates parquet predicate pushdown over HTTP: filtering
``pr.parquet`` to one municipality touches a handful of row groups and returns
in about nine seconds against a 605-million-row file. Zenodo does not -- it
answers HTTP 429 after a few dozen range requests -- so Zenodo artefacts are
downloaded whole into the cache and queried locally. That asymmetry is a
property of the hosts, not a preference, and it is why :data:`PRODUCTS` records
a route per product.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import polars as pl

from brepi.config import (
    BRCLIMR_BRDWGD_CDN_BASE,
    BRCLIMR_TERRACLIMATE_CDN_BASE,
    BRDWGD_ZONAL_ZENODO_RECORD,
    ERA5LAND_ZONAL_ZENODO_RECORDS,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    zenodo_file_uri,
)
from brepi.io import cache

TOOL_VERSION = "brepi.sources.climate.brdwgd/1"

Route = Literal["cdn", "zenodo"]
Grain = Literal["daily", "monthly"]

#: Canonical variable names used throughout brepi. Product-specific file and
#: statistic names are translated onto these; nothing downstream should ever
#: see ``total_precipitation_sum`` or ``Tmax_3.2.3``.
CANONICAL_VARIABLES = ("pr", "tmax", "tmin", "tmean", "rh", "rs", "u2", "eto")

#: Canonical zonal statistics. ``sum`` is the areal *total over the polygon's
#: cells* only for accumulating variables; for precipitation the statistic a
#: municipality-level exposure wants is ``mean`` (the areal mean daily depth in
#: mm), not ``sum`` (which scales with the polygon's cell count and therefore
#: with municipal area, and is the classic way to make Amazonian municipalities
#: look ten times wetter than they are).
CANONICAL_STATS = ("mean", "max", "min", "sd", "sum")

DEFAULT_PRECIP_STAT = "mean"


class ClimateSourceError(RuntimeError):
    """A climate product was asked for something it does not contain."""


@dataclass(frozen=True)
class ClimateProduct:
    """One published municipal zonal-statistics product.

    ``scale``/``offset`` convert the stored value to brepi units (mm/day for
    ``pr``, degrees Celsius for temperature). They are declared per product
    rather than inferred, because inferring a unit from the magnitude of the
    data is how a reanalysis precipitation field in metres silently becomes a
    millimetre field a thousand times too dry.
    """

    id: str
    label: str
    grain: Grain
    route: Route
    start: date
    end: date
    #: canonical variable -> the product's file basename
    files: Mapping[str, str]
    #: canonical variable -> the product's prefix in the ``name`` column
    name_prefix: Mapping[str, str]
    #: canonical variable -> (scale, offset) applied as ``value * scale + offset``
    units: Mapping[str, tuple[float, float]]
    stats: tuple[str, ...]
    citation: str
    #: zenodo record id, or per-year mapping for products published annually
    record: str | Mapping[str, str] | None = None
    base: str | None = None
    notes: str = ""

    def covers(self, start: date, end: date) -> bool:
        return self.start <= start and end <= self.end

    def clip(self, start: date, end: date) -> tuple[date, date] | None:
        lo, hi = max(start, self.start), min(end, self.end)
        return (lo, hi) if lo <= hi else None


_BRDWGD_UNITS = {
    "pr": (1.0, 0.0),
    "tmax": (1.0, 0.0),
    "tmin": (1.0, 0.0),
    "rh": (1.0, 0.0),
    "rs": (1.0, 0.0),
    "u2": (1.0, 0.0),
    "eto": (1.0, 0.0),
}

#: ERA5-Land stores accumulated precipitation in metres and temperature in
#: kelvin. VERIFIED against Porto Alegre 2024 by :func:`self_check`.
_ERA5_UNITS = {
    "pr": (1000.0, 0.0),
    "tmax": (1.0, -273.15),
    "tmin": (1.0, -273.15),
    "tmean": (1.0, -273.15),
}

PRODUCTS: dict[str, ClimateProduct] = {
    "brdwgd": ClimateProduct(
        id="brdwgd",
        label="BR-DWGD v3.2.3 municipal zonal statistics (Zenodo)",
        grain="daily",
        route="zenodo",
        start=date(1961, 1, 1),
        end=date(2024, 3, 20),
        files={
            "pr": "pr_3.2.3.parquet",
            "tmax": "Tmax_3.2.3.parquet",
            "tmin": "Tmin_3.2.3.parquet",
            "rh": "RH_3.2.3.parquet",
            "rs": "Rs_3.2.3.parquet",
            "u2": "u2_3.2.3.parquet",
            "eto": "ETo_3.2.3.parquet",
        },
        name_prefix={
            # The Zenodo v3.2.3 artefacts retain the version in every
            # statistic name (for example ``pr_3.2.3_mean``).  This is not the
            # same schema as the older CDN mirror, whose names are ``pr_mean``.
            "pr": "pr_3.2.3",
            "tmax": "Tmax_3.2.3",
            "tmin": "Tmin_3.2.3",
            "rh": "RH_3.2.3",
            "rs": "Rs_3.2.3",
            "u2": "u2_3.2.3",
            "eto": "ETo_3.2.3",
        },
        units=_BRDWGD_UNITS,
        stats=("mean", "max", "min", "sd", "sum"),
        record=BRDWGD_ZONAL_ZENODO_RECORD,
        citation="Xavier et al. 2022 (10.1002/joc.7731); Saldanha et al. 2024 zonal stats, 10.5281/zenodo.13906834",
        notes=(
            "Gauge-interpolated observations at 0.1 deg. The authoritative "
            "precipitation product for Brazil, but stops 2024-03-20 -- it does "
            "NOT contain the May 2024 Rio Grande do Sul flood."
        ),
    ),
    "brdwgd_cdn": ClimateProduct(
        id="brdwgd_cdn",
        label="BR-DWGD municipal zonal statistics (brclimr CDN mirror)",
        grain="daily",
        route="cdn",
        start=date(1961, 1, 1),
        end=date(2020, 7, 31),
        files={
            "pr": "pr.parquet",
            "tmax": "tmax.parquet",
            "tmin": "tmin.parquet",
            "rh": "rh.parquet",
            "rs": "rs.parquet",
            "u2": "u2.parquet",
            "eto": "eto.parquet",
        },
        name_prefix={v: v for v in ("pr", "tmax", "tmin", "rh", "rs", "u2", "eto")},
        units=_BRDWGD_UNITS,
        stats=("mean", "max", "min", "sd", "sum"),
        base=BRCLIMR_BRDWGD_CDN_BASE,
        citation="brclimr (Saldanha) mirror of BR-DWGD; objects last modified 2023-03-03",
        notes=(
            "An older BR-DWGD vintage than the Zenodo record and it stops "
            "2020-07-31, but it needs no download and supports predicate "
            "pushdown, which makes it the cheap route to a 1991-2020 climatology."
        ),
    ),
    "era5land": ClimateProduct(
        id="era5land",
        label="ERA5-Land municipal zonal statistics (Zenodo, annual records)",
        grain="daily",
        route="zenodo",
        start=date(1950, 1, 1),
        end=date(2025, 12, 31),
        files={
            "pr": "total_precipitation_sum.parquet",
            "tmax": "2m_temperature_max.parquet",
            "tmin": "2m_temperature_min.parquet",
            "tmean": "2m_temperature_mean.parquet",
        },
        name_prefix={
            "pr": "total_precipitation_sum",
            "tmax": "2m_temperature_max",
            "tmin": "2m_temperature_min",
            "tmean": "2m_temperature_mean",
        },
        units=_ERA5_UNITS,
        stats=("mean", "max", "min", "sd"),
        record=ERA5LAND_ZONAL_ZENODO_RECORDS,
        citation="Munoz-Sabater et al. ERA5-Land; Saldanha zonal stats, 10.5281/zenodo.10036211 + annual updates",
        notes=(
            "Reanalysis, not observation. The only municipal product that "
            "reaches 2025, so it is what closes the panel; expect a wet bias "
            "against BR-DWGD and correct for it with bias_adjustment()."
        ),
    ),
    "terraclimate_cdn": ClimateProduct(
        id="terraclimate_cdn",
        label="TerraClimate municipal zonal statistics (brclimr CDN mirror)",
        grain="monthly",
        route="cdn",
        start=date(1958, 1, 1),
        end=date(2021, 12, 1),
        files={"pr": "ppt.parquet", "tmax": "tmax.parquet", "tmin": "tmin.parquet"},
        name_prefix={"pr": "ppt", "tmax": "tmax", "tmin": "tmin"},
        units={"pr": (1.0, 0.0), "tmax": (1.0, 0.0), "tmin": (1.0, 0.0)},
        stats=("mean", "max", "min", "sd", "sum"),
        base=BRCLIMR_TERRACLIMATE_CDN_BASE,
        citation="Abatzoglou et al. TerraClimate; zonal stats 10.5281/zenodo.7825777",
        notes=(
            "MONTHLY. Cannot support RX1day, CWD, CDD or any wet-day count. "
            "Kept only as a coarse cross-check; stops 2021-12."
        ),
    ),
}

#: The default splice. BR-DWGD wherever it exists (observational, gauge-based),
#: ERA5-Land for the tail it cannot reach. Order matters: earlier entries win.
DEFAULT_CHAIN: tuple[str, ...] = ("brdwgd", "era5land")


def available_products() -> pl.DataFrame:
    """One row per product with its verified coverage. Print this before asking
    a product for a window it does not have."""
    return pl.DataFrame(
        [
            {
                "product": p.id,
                "grain": p.grain,
                "route": p.route,
                "start": p.start,
                "end": p.end,
                "variables": ",".join(sorted(p.files)),
                "label": p.label,
            }
            for p in PRODUCTS.values()
        ]
    ).sort("product")


# --------------------------------------------------------------------------
# Artefact location
# --------------------------------------------------------------------------


def _zenodo_record_for(product: ClimateProduct, year: int) -> str:
    rec = product.record
    if isinstance(rec, str):
        return rec
    if not isinstance(rec, Mapping):
        raise ClimateSourceError(f"{product.id} has no Zenodo record configured")
    for key, value in rec.items():
        if "-" in key:
            lo, hi = (int(x) for x in key.split("-"))
            if lo <= year <= hi:
                return value
        elif int(key) == year:
            return value
    raise ClimateSourceError(
        f"{product.id} publishes no Zenodo record covering {year}; "
        f"available: {sorted(rec)}"
    )


def _artefacts(
    product: ClimateProduct, variable: str, start: date, end: date
) -> list[tuple[str, str]]:
    """``[(cache_key, uri)]`` for the artefacts spanning ``start``..``end``."""
    if variable not in product.files:
        raise ClimateSourceError(
            f"{product.id} does not publish {variable!r}; it has "
            f"{sorted(product.files)}"
        )
    filename = product.files[variable]
    if product.route == "cdn":
        return [(f"climate/{product.id}/{filename}", f"{product.base}/{filename}")]

    seen: dict[str, str] = {}
    for year in range(start.year, end.year + 1):
        record = _zenodo_record_for(product, year)
        key = f"climate/{product.id}/{record}/{filename}"
        seen[key] = zenodo_file_uri(record, filename)
    return sorted(seen.items())


# --------------------------------------------------------------------------
# Download (streaming, provenance-tracked)
# --------------------------------------------------------------------------


def _stream_into_cache(key: str, uri: str, *, source: str) -> cache.Provenance:
    """Stream a large remote artefact into the cache with a provenance sidecar.

    ``cache.store`` takes an in-memory payload, which is the right shape for a
    5 MB DBC and the wrong shape for a 3 GB parquet. This writes through to
    the cache path and then constructs the same sidecar, so the artefact is
    indistinguishable from a stored one to ``verify_snapshot``.
    """
    import httpx

    path = cache.cache_path(key)
    tmp = path.with_name(path.name + ".part")
    remote_modified: str | None = None
    last: Exception | None = None

    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            written = 0
            with httpx.stream(
                "GET", uri, timeout=HTTP_TIMEOUT, follow_redirects=True
            ) as response:
                if response.status_code != 200:
                    raise ClimateSourceError(f"{uri} returned HTTP {response.status_code}")
                remote_modified = response.headers.get("last-modified")
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(1 << 22):
                        fh.write(chunk)
                        written += len(chunk)
            tmp.replace(path)
            break
        except Exception as exc:  # noqa: BLE001 - transient network only
            last = exc
            tmp.unlink(missing_ok=True)
            if isinstance(exc, ClimateSourceError) and "HTTP 4" in str(exc) and "429" not in str(exc):
                raise
            if attempt == HTTP_MAX_RETRIES:
                raise ClimateSourceError(f"giving up on {uri}") from last
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    prov = cache.Provenance(
        key=key,
        source=source,
        uri=uri,
        fetched_at=cache._now(),  # noqa: SLF001 - the module's own clock, by design
        bytes=path.stat().st_size,
        sha256=cache.sha256_file(path),
        remote_modified=remote_modified,
        params={"streamed": True},
        tool_version=TOOL_VERSION,
    )
    cache.sidecar_path(key).write_text(prov.to_json(), encoding="utf-8")
    return prov


def ensure_local(
    product: str | ClimateProduct,
    variables: Sequence[str],
    start: date,
    end: date,
    *,
    refresh: bool = False,
) -> list[Path]:
    """Download every artefact needed for ``variables`` over ``start``..``end``.

    Zenodo-routed products must be local before they can be queried; CDN-routed
    products are queried in place and this is a no-op for them.
    """
    prod = product if isinstance(product, ClimateProduct) else _product(product)
    if prod.route == "cdn":
        return []
    out: list[Path] = []
    for variable in variables:
        for key, uri in _artefacts(prod, variable, start, end):
            path = cache.cache_path(key)
            if path.exists() and cache.read_provenance(key) is not None and not refresh:
                out.append(path)
                continue
            _stream_into_cache(key, uri, source=f"climate.{prod.id}")
            out.append(path)
    return out


def cache_keys(
    product: str | ClimateProduct,
    variables: Sequence[str],
    start: date,
    end: date,
) -> list[str]:
    """The cache keys a query would touch. Feed these to ``write_manifest``."""
    prod = product if isinstance(product, ClimateProduct) else _product(product)
    keys: list[str] = []
    for variable in variables:
        keys.extend(k for k, _ in _artefacts(prod, variable, start, end))
    return sorted(set(keys))


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


def _product(name: str) -> ClimateProduct:
    try:
        return PRODUCTS[name]
    except KeyError:
        raise ClimateSourceError(
            f"unknown climate product {name!r}; have {sorted(PRODUCTS)}"
        ) from None


def _connect():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false")
    return con


def _sources_for(
    prod: ClimateProduct, variable: str, start: date, end: date, *, download: bool
) -> list[str]:
    if prod.route == "cdn":
        return [uri for _key, uri in _artefacts(prod, variable, start, end)]
    if download:
        ensure_local(prod, [variable], start, end)
    paths: list[str] = []
    for key, uri in _artefacts(prod, variable, start, end):
        path = cache.cache_path(key)
        if not path.exists():
            raise ClimateSourceError(
                f"{prod.id}/{variable} artefact {key} is not cached and download=False. "
                f"Run ensure_local(); Zenodo ({uri}) rate-limits range reads, so "
                "querying it in place is not an option."
            )
        paths.append(str(path).replace("\\", "/"))
    return paths


def daily_municipal(
    munic_codes: Sequence[str] | None,
    variables: Sequence[str] = ("pr", "tmax", "tmin"),
    *,
    start: date,
    end: date,
    product: str = "brdwgd",
    stats: Sequence[str] = ("mean",),
    download: bool = True,
) -> pl.DataFrame:
    """Daily municipal zonal statistics as a long frame.

    Columns: ``munic_code`` (7-digit string), ``date``, ``variable``, ``stat``,
    ``value`` (brepi units), ``product``.

    ``munic_codes`` of ``None`` means all 5570, which for a Zenodo-routed
    product is a local scan of a multi-gigabyte file and for a CDN-routed one
    is a full download over HTTP. Pass a subset while developing.
    """
    prod = _product(product)
    if prod.grain != "daily":
        raise ClimateSourceError(
            f"{prod.id} is {prod.grain}, not daily. Daily-resolution indices "
            "(RX1day, CWD, CDD, R10/R20/R50) cannot be reconstructed from it; "
            "see brepi.sources.climate.indices.DAILY_ONLY_INDICES."
        )
    if start > end:
        raise ValueError("start after end")
    window = prod.clip(start, end)
    if window is None or window != (start, end):
        raise ClimateSourceError(
            f"{prod.id} covers {prod.start}..{prod.end}; asked for {start}..{end}. "
            "Use chained_daily() to splice products, or narrow the request. "
            "Silently returning the intersection would put a hole in the panel."
        )

    bad_stats = [s for s in stats if s not in prod.stats]
    if bad_stats:
        raise ClimateSourceError(f"{prod.id} has no statistic(s) {bad_stats}; has {prod.stats}")

    codes = _normalise_codes(munic_codes)
    con = _connect()
    frames: list[pl.DataFrame] = []
    for variable in variables:
        prefix = prod.name_prefix[variable]
        wanted = {f"{prefix}_{s}": s for s in stats}
        sources = _sources_for(prod, variable, start, end, download=download)
        scale, offset = prod.units[variable]
        for src in sources:
            clauses = [
                f"date BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'",
                "name IN (" + ", ".join(f"'{n}'" for n in wanted) + ")",
            ]
            if codes is not None:
                clauses.append("code_muni IN (" + ", ".join(codes) + ")")
            sql = (
                "SELECT CAST(code_muni AS VARCHAR) AS munic_code, date, name, "
                f"CAST(value AS DOUBLE) * {scale} + {offset} AS value "
                f"FROM read_parquet('{src}') WHERE " + " AND ".join(clauses)
            )
            part = con.execute(sql).pl()
            if part.height:
                frames.append(
                    part.with_columns(
                        pl.lit(variable).alias("variable"),
                        pl.col("name").replace_strict(wanted).alias("stat"),
                        pl.lit(prod.id).alias("product"),
                    ).drop("name")
                )
    if not frames:
        return pl.DataFrame(
            schema={
                "munic_code": pl.Utf8,
                "date": pl.Date,
                "value": pl.Float64,
                "variable": pl.Utf8,
                "stat": pl.Utf8,
                "product": pl.Utf8,
            }
        )
    out = pl.concat(frames, how="vertical_relaxed")
    return out.select("munic_code", "date", "variable", "stat", "value", "product").sort(
        ["munic_code", "variable", "stat", "date"]
    )


def _normalise_codes(munic_codes: Sequence[str] | None) -> list[str] | None:
    if munic_codes is None:
        return None
    codes = [str(c).strip() for c in munic_codes]
    bad = [c for c in codes if not (c.isdigit() and len(c) == 7)]
    if bad:
        raise ValueError(
            f"{len(bad)} municipality codes are not 7-digit IBGE codes, e.g. {bad[:3]}. "
            "Convert with brepi.geo.lattice.code6_to_code7 first."
        )
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate municipality codes")
    return codes


def chained_daily(
    munic_codes: Sequence[str] | None,
    variables: Sequence[str] = ("pr", "tmax", "tmin"),
    *,
    start: date,
    end: date,
    chain: Sequence[str] = DEFAULT_CHAIN,
    stats: Sequence[str] = ("mean",),
    download: bool = True,
) -> pl.DataFrame:
    """Daily series over ``start``..``end``, spliced across ``chain``.

    Earlier products in ``chain`` win where they have coverage; each row keeps
    its ``product``. Raises if the chain leaves a hole, because a hole in a
    distributed-lag exposure is not a missing covariate, it is a silently
    shortened lag window.
    """
    remaining: list[tuple[date, date]] = [(start, end)]
    frames: list[pl.DataFrame] = []
    for name in chain:
        prod = _product(name)
        still: list[tuple[date, date]] = []
        for lo, hi in remaining:
            window = prod.clip(lo, hi)
            if window is None:
                still.append((lo, hi))
                continue
            wlo, whi = window
            vars_here = [v for v in variables if v in prod.files]
            if vars_here:
                frames.append(
                    daily_municipal(
                        munic_codes,
                        vars_here,
                        start=wlo,
                        end=whi,
                        product=name,
                        stats=stats,
                        download=download,
                    )
                )
            if lo < wlo:
                still.append((lo, _day_before(wlo)))
            if whi < hi:
                still.append((_day_after(whi), hi))
        remaining = still
    if remaining:
        raise ClimateSourceError(
            f"chain {list(chain)} leaves {remaining} uncovered. "
            f"Coverage:\n{available_products()}"
        )
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["munic_code", "variable", "stat", "date"]
    )


def _day_before(d: date) -> date:
    from datetime import timedelta

    return d - timedelta(days=1)


def _day_after(d: date) -> date:
    from datetime import timedelta

    return d + timedelta(days=1)


# --------------------------------------------------------------------------
# Monthly product
# --------------------------------------------------------------------------


def municipal_climate(
    years: Iterable[int],
    variables: Sequence[str] = ("pr", "tmax", "tmin"),
    stats: Sequence[str] = ("mean",),
    *,
    munic_codes: Sequence[str] | None = None,
    chain: Sequence[str] = DEFAULT_CHAIN,
    download: bool = True,
) -> pl.DataFrame:
    """The monthly municipal exposure frame, ready to join onto the spine.

    Returns ``munic_code`` (7-digit), ``period`` (first of month), the derived
    indices from :mod:`brepi.sources.climate.indices`, and ``product`` /
    ``n_days`` provenance columns.

    Whole calendar years only: a partial month at either end produces monthly
    totals that are not totals, which is the kind of error that survives review
    because the number looks plausible.
    """
    from brepi.sources.climate import indices

    years = sorted(set(int(y) for y in years))
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("years must be contiguous")
    daily = chained_daily(
        munic_codes,
        variables,
        start=date(years[0], 1, 1),
        end=date(years[-1], 12, 31),
        chain=chain,
        stats=stats,
        download=download,
    )
    return indices.monthly_indices_from_daily(daily, variables=variables)


def monthly_municipal(
    years: Iterable[int],
    variables: Sequence[str] = ("pr",),
    *,
    munic_codes: Sequence[str] | None = None,
    chain: Sequence[str] = DEFAULT_CHAIN,
    download: bool = True,
    require_complete_months: bool = True,
) -> pl.DataFrame:
    """Aggregate daily products at source, without materialising the daily panel.

    This is the production path for national panels.  A 19-year Brazil-wide
    daily frame has roughly 39 million municipality-days *per variable*; the
    older :func:`municipal_climate` path intentionally remains useful for small
    subsets and daily-only indices, but is an unsafe implementation for that
    scale.  Here DuckDB pushes date, statistic and municipality predicates into
    parquet and returns only municipality-month summaries.

    The returned columns follow the public monthly climate contract:
    ``munic_code``, ``period``, ``pr_total``, wet-day counts, ``rx1day``,
    temperature summaries, ``n_days``, ``n_days_expected``,
    ``coverage_fraction`` and ``product``.  Daily-sequence indices such as
    RX5day/CWD/CDD are deliberately absent: computing them correctly across
    month and product boundaries requires daily state and belongs in
    :func:`municipal_climate`.

    Earlier products in ``chain`` win exactly as in :func:`chained_daily`.
    Product boundaries may split a month; partial aggregates are recombined and
    the contributing products remain visible (for example
    ``"brdwgd+era5land"``).  Incomplete source months fail by default rather
    than being silently treated as monthly totals.
    """
    import calendar

    years = sorted(set(int(y) for y in years))
    if not years:
        raise ValueError("years is empty")
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("years must be contiguous")
    unknown = sorted(set(variables) - set(CANONICAL_VARIABLES))
    if unknown:
        raise ValueError(f"unknown canonical climate variables {unknown}")

    codes = _normalise_codes(munic_codes)
    start, end = date(years[0], 1, 1), date(years[-1], 12, 31)
    windows = _chain_windows(start, end, chain)
    con = _connect()
    pieces: list[pl.DataFrame] = []

    for product_name, lo, hi in windows:
        prod = _product(product_name)
        for variable in variables:
            if variable not in prod.files:
                continue
            sources = _sources_for(prod, variable, lo, hi, download=download)
            prefix = prod.name_prefix[variable]
            scale, offset = prod.units[variable]
            value = f"(CAST(value AS DOUBLE) * {scale} + {offset})"
            clauses = [
                f"date BETWEEN DATE '{lo.isoformat()}' AND DATE '{hi.isoformat()}'",
                f"name = '{prefix}_mean'",
            ]
            if codes is not None:
                clauses.append("code_muni IN (" + ", ".join(codes) + ")")
            where = " AND ".join(clauses)
            for source in sources:
                quoted = source.replace("'", "''")
                extras = ""
                if variable == "pr":
                    extras = f""",
                        SUM(CASE WHEN {value} >= 1.0 THEN 1 ELSE 0 END) AS r1,
                        SUM(CASE WHEN {value} >= 10.0 THEN 1 ELSE 0 END) AS r10,
                        SUM(CASE WHEN {value} >= 20.0 THEN 1 ELSE 0 END) AS r20,
                        SUM(CASE WHEN {value} >= 50.0 THEN 1 ELSE 0 END) AS r50"""
                sql = f"""
                    SELECT CAST(code_muni AS VARCHAR) AS munic_code,
                           CAST(date_trunc('month', date) AS DATE) AS period,
                           '{variable}' AS variable,
                           '{prod.id}' AS product,
                           COUNT(DISTINCT date) AS n_days,
                           SUM({value}) AS value_sum,
                           MAX({value}) AS value_max,
                           MIN({value}) AS value_min
                           {extras}
                    FROM read_parquet('{quoted}')
                    WHERE {where}
                    GROUP BY code_muni, period
                """
                part = con.execute(sql).pl()
                if part.height:
                    pieces.append(part)

    if not pieces:
        raise ClimateSourceError(
            f"no monthly climate rows for {years[0]}..{years[-1]}, "
            f"variables={list(variables)}, chain={list(chain)}"
        )

    long = pl.concat(pieces, how="diagonal_relaxed")
    products = (
        long.group_by(["munic_code", "period"])
        .agg(pl.col("product").unique().sort().str.join("+").alias("product"))
    )
    blocks: list[pl.DataFrame] = []
    for variable in variables:
        block = long.filter(pl.col("variable") == variable)
        if block.is_empty():
            continue
        expressions: list[pl.Expr] = [
            pl.col("n_days").sum().alias(f"n_days_{variable}"),
            pl.col("value_sum").sum().alias("_sum"),
            pl.col("value_max").max().alias("_max"),
            pl.col("value_min").min().alias("_min"),
        ]
        if variable == "pr":
            expressions.extend(pl.col(name).sum().alias(name) for name in ("r1", "r10", "r20", "r50"))
        block = block.group_by(["munic_code", "period"]).agg(expressions)
        if variable == "pr":
            block = block.rename({"_sum": "pr_total", "_max": "rx1day"}).drop("_min")
        elif variable == "tmax":
            block = block.with_columns(
                (pl.col("_sum") / pl.col("n_days_tmax")).alias("tmax_mean")
            ).rename({"_max": "tmax_max"}).drop("_sum", "_min")
        elif variable == "tmin":
            block = block.with_columns(
                (pl.col("_sum") / pl.col("n_days_tmin")).alias("tmin_mean")
            ).rename({"_min": "tmin_min"}).drop("_sum", "_max")
        else:
            block = block.with_columns(
                (pl.col("_sum") / pl.col(f"n_days_{variable}")).alias(f"{variable}_mean")
            ).drop("_sum", "_max", "_min")
        blocks.append(block)

    result = products
    for block in blocks:
        result = result.join(block, on=["munic_code", "period"], how="left")
    day_columns = [c for c in result.columns if c.startswith("n_days_")]
    if "n_days_pr" in day_columns:
        result = result.with_columns(pl.col("n_days_pr").alias("n_days"))
    elif day_columns:
        result = result.with_columns(pl.col(day_columns[0]).alias("n_days"))
    result = result.with_columns(
        pl.struct("period")
        .map_elements(
            lambda row: calendar.monthrange(row["period"].year, row["period"].month)[1],
            return_dtype=pl.Int16,
        )
        .alias("n_days_expected")
    ).with_columns(
        (pl.col("n_days") / pl.col("n_days_expected")).alias("coverage_fraction")
    ).sort(["munic_code", "period"])

    if require_complete_months:
        short = result.filter(pl.col("n_days") != pl.col("n_days_expected"))
        if short.height:
            examples = short.select(
                "munic_code", "period", "n_days", "n_days_expected", "product"
            ).head(5).to_dicts()
            raise ClimateSourceError(
                f"{short.height} municipality-months are incomplete, e.g. {examples}. "
                "Pass require_complete_months=False only to retain explicit coverage "
                "columns and exclude or model incomplete months downstream."
            )
    return result


def _chain_windows(
    start: date, end: date, chain: Sequence[str]
) -> list[tuple[str, date, date]]:
    """Resolve precedence in a product chain into non-overlapping date windows."""
    remaining: list[tuple[date, date]] = [(start, end)]
    windows: list[tuple[str, date, date]] = []
    for name in chain:
        prod = _product(name)
        still: list[tuple[date, date]] = []
        for lo, hi in remaining:
            window = prod.clip(lo, hi)
            if window is None:
                still.append((lo, hi))
                continue
            wlo, whi = window
            windows.append((name, wlo, whi))
            if lo < wlo:
                still.append((lo, _day_before(wlo)))
            if whi < hi:
                still.append((_day_after(whi), hi))
        remaining = still
    if remaining:
        raise ClimateSourceError(
            f"chain {list(chain)} leaves {remaining} uncovered. "
            f"Coverage:\n{available_products()}"
        )
    return windows


# --------------------------------------------------------------------------
# Splice correction
# --------------------------------------------------------------------------


def bias_adjustment(
    munic_codes: Sequence[str] | None,
    *,
    reference: str = "brdwgd",
    target: str = "era5land",
    variable: str = "pr",
    overlap_start: date = date(2007, 1, 1),
    overlap_end: date = date(2023, 12, 31),
    download: bool = True,
) -> pl.DataFrame:
    """Per-municipality, per-calendar-month factor mapping ``target`` onto ``reference``.

    Returns ``munic_code``, ``month``, ``ref_mean``, ``tgt_mean``, ``factor``
    (multiplicative, for precipitation) and ``delta`` (additive, for
    temperature), estimated on the seventeen years the two products share.

    Multiplicative for rainfall because reanalysis rainfall error is
    proportional (a wet bias in a wet place is larger in mm than the same bias
    in a dry place); additive for temperature because reanalysis temperature
    error is an offset. Applying the wrong one is worse than applying none.

    A factor is only estimated where both products have at least ten overlap
    years for that calendar month; elsewhere it is null and the caller must
    decide, rather than being handed a 1.0 that looks like agreement.
    """
    ref = _monthly_means(
        munic_codes, reference, variable, overlap_start, overlap_end, download
    ).rename({"monthly_total": "ref_total"})
    tgt = _monthly_means(
        munic_codes, target, variable, overlap_start, overlap_end, download
    ).rename({"monthly_total": "tgt_total"})
    joined = ref.join(tgt, on=["munic_code", "period"], how="inner")
    if joined.is_empty():
        raise ClimateSourceError(
            f"{reference} and {target} do not overlap on {overlap_start}..{overlap_end}"
        )
    agg = (
        joined.with_columns(pl.col("period").dt.month().alias("month"))
        .group_by(["munic_code", "month"])
        .agg(
            pl.col("ref_total").mean().alias("ref_mean"),
            pl.col("tgt_total").mean().alias("tgt_mean"),
            pl.len().alias("n_years"),
        )
    )
    return agg.with_columns(
        pl.when((pl.col("n_years") >= 10) & (pl.col("tgt_mean") > 0))
        .then(pl.col("ref_mean") / pl.col("tgt_mean"))
        .otherwise(None)
        .alias("factor"),
        pl.when(pl.col("n_years") >= 10)
        .then(pl.col("ref_mean") - pl.col("tgt_mean"))
        .otherwise(None)
        .alias("delta"),
    ).sort(["munic_code", "month"])


def _monthly_means(
    munic_codes: Sequence[str] | None,
    product: str,
    variable: str,
    start: date,
    end: date,
    download: bool,
) -> pl.DataFrame:
    prod = _product(product)
    window = prod.clip(start, end)
    if window is None:
        raise ClimateSourceError(f"{product} does not cover {start}..{end}")
    daily = daily_municipal(
        munic_codes,
        [variable],
        start=window[0],
        end=window[1],
        product=product,
        stats=("mean",),
        download=download,
    )
    return (
        daily.with_columns(pl.col("date").dt.truncate("1mo").alias("period"))
        .group_by(["munic_code", "period"])
        .agg(pl.col("value").sum().alias("monthly_total"))
        .sort(["munic_code", "period"])
    )


__all__ = [
    "CANONICAL_STATS",
    "CANONICAL_VARIABLES",
    "DEFAULT_CHAIN",
    "monthly_municipal",
    "PRODUCTS",
    "ClimateProduct",
    "ClimateSourceError",
    "available_products",
    "bias_adjustment",
    "cache_keys",
    "chained_daily",
    "daily_municipal",
    "ensure_local",
    "municipal_climate",
]
