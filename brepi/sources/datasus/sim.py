"""SIM mortality: individual death records, CID-10 era (1996-).

``/SIM/CID10/DORES`` holds the consolidated series as ``DO{UF}{YYYY}.dbc``
(four-digit year, unlike SINAN and SIH) plus a national ``DOBR{YYYY}``
roll-up. ``/SIM/PRELIM/DORES`` holds the preliminary 2025 and 2026 files,
which are regenerated daily and are *not* a stable base for published counts.

Traps.

*The final series ends 2024.* Anything after that is preliminary by
construction: undercounted at the tail, revised upward for months, and
revised again at consolidation. :func:`list_available` labels the release and
:func:`fetch_year` refuses to silently substitute one for the other.

*Case.* Several state-years -- notably a run of ``DOSP``/``DOAC``/``DOAL``
files around 2007 and 2010-2012 -- are published with an uppercase ``.DBC``
while their siblings are lowercase. Matching is done through
:func:`brepi.io.datasus_ftp.resolve`, which is case-insensitive; a
case-sensitive fetcher 404s on exactly those years and produces a series with
holes that look like genuine zero-mortality.

*National vs state files.* ``DOBR{YYYY}`` is the same data aggregated, not an
extra stratum. Fetching both and concatenating double-counts every death.

*Cause fields.* ``CAUSABAS`` is the underlying cause: one four-character
CID-10 code, and the only field admissible for cause-specific mortality
rates. The associated-cause fields ``LINHAA``..``LINHAD``, ``LINHAII`` and
``CAUSABAS_O`` pack *several* codes into one string with no separator, so
they are matched by substring, not by prefix, and any count derived from them
is a mention count, not a death count.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import polars as pl
from loguru import logger

from brepi.io import cache
from brepi.io.cache import Provenance
from brepi.io.dbc import read_dbc
from brepi.io import datasus_ftp
from brepi.io.datasus_ftp import Backend, RemoteFile, get, resolve

__all__ = [
    "UFS",
    "SimFile",
    "FINAL_DIR",
    "PRELIM_DIR",
    "ASSOCIATED_CAUSE_FIELDS",
    "list_available",
    "fetch_year",
    "fetch_icd_deaths",
    "SOURCE_ID",
]

SOURCE_ID = "datasus.sim"

FINAL_DIR = "/SIM/CID10/DORES"
PRELIM_DIR = "/SIM/PRELIM/DORES"

Release = Literal["final", "prelim"]

UFS: tuple[str, ...] = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
)

_DO_RE = re.compile(r"^DO(?P<uf>[A-Z]{2})(?P<year>\d{4})\.dbc$", re.IGNORECASE)

#: Multi-code fields. Matched by substring; see the module docstring.
ASSOCIATED_CAUSE_FIELDS: tuple[str, ...] = (
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "CAUSABAS_O",
)

DEFAULT_FIELDS: tuple[str, ...] = (
    "NUMERODO",
    "DTOBITO",
    "DTNASC",
    "IDADE",
    "SEXO",
    "RACACOR",
    "ESC",
    "CODMUNRES",
    "CODMUNOCOR",
    "LOCOCOR",
    "CAUSABAS",
    "CAUSABAS_O",
    "CIRCOBITO",
    "ASSISTMED",
    "TIPOBITO",
)


@dataclass(frozen=True)
class SimFile(RemoteFile):
    """One DO file with its parsed scope, year and release channel."""

    uf: str = ""
    year: int = 0
    release: Release = "final"


def _cache_key(filename: str) -> str:
    return f"datasus/sim/do/{filename.upper()}"


def list_available(
    *,
    ufs: Sequence[str] | None = None,
    include_prelim: bool = True,
    include_national: bool = False,
    backend: Backend = "ftp",
) -> list[SimFile]:
    """List DO files, sorted by ``(year, uf)``.

    ``DOBR`` national roll-ups are excluded unless ``include_national`` is
    set, because mixing them with state files double-counts.
    """
    wanted = {u.upper() for u in ufs} if ufs else None
    out: list[SimFile] = []
    directories: list[tuple[str, Release]] = [(FINAL_DIR, "final")]
    if include_prelim:
        directories.append((PRELIM_DIR, "prelim"))
    for directory, release in directories:
        for rf in resolve(directory, r"DO[A-Z]{2}\d{4}\.dbc$", backend=backend):
            m = _DO_RE.match(rf.name)
            if not m:
                continue
            uf = m["uf"].upper()
            if uf == "BR" and not include_national:
                continue
            if wanted is not None and uf not in wanted:
                continue
            out.append(
                SimFile(
                    path=rf.path,
                    name=rf.name,
                    bytes=rf.bytes,
                    modified=rf.modified,
                    uf=uf,
                    year=int(m["year"]),
                    release=release,
                )
            )
    out.sort(key=lambda f: (f.year, f.uf, 0 if f.release == "final" else 1))
    return out


def _find(listing: Sequence[SimFile], uf: str, year: int, prefer: Release) -> SimFile:
    uf = uf.upper()
    candidates = [f for f in listing if f.uf == uf and f.year == year]
    if not candidates:
        raise FileNotFoundError(f"no SIM DO file for {uf} {year}")
    for release in (prefer, "prelim" if prefer == "final" else "final"):
        for f in candidates:
            if f.release == release:
                return f
    return candidates[0]


def fetch_year(
    uf: str,
    year: int,
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
    listing: Sequence[SimFile] | None = None,
) -> tuple[pl.DataFrame, Provenance]:
    """Fetch and decode one state-year of DO (``uf="BR"`` for the roll-up).

    The release actually used is recorded in the provenance ``params``; when
    a year exists only as preliminary this is the only place that fact is
    written down, so propagate it into any table you publish.
    """
    files = (
        list(listing)
        if listing is not None
        else list_available(
            ufs=None if uf.upper() == "BR" else [uf],
            include_national=uf.upper() == "BR",
            backend=backend,
        )
    )
    target = _find(files, uf, year, prefer)
    key = _cache_key(target.name)
    uri = target.uri_ftp if backend == "ftp" else target.uri_mirror

    def _loader() -> tuple[bytes, dict[str, Any]]:
        payload = get(target.path, backend=backend)
        if len(payload) != target.bytes:
            raise ValueError(
                f"{target.name}: expected {target.bytes} bytes, received {len(payload)}"
            )
        return payload, {
            "remote_modified": target.modified,
            "params": {
                "backend": backend,
                "uf": target.uf,
                "year": target.year,
                "release": target.release,
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(key, _loader, source=SOURCE_ID, uri=uri, refresh=refresh)
    if target.release == "prelim":
        logger.warning(
            "SIM {} {} served from PRELIM; counts will be revised upward", target.uf, year
        )
    return read_dbc(path), prov


def _cause_filter(
    prefixes: Sequence[str],
    columns: Sequence[str],
    *,
    include_associated: bool,
) -> pl.Expr:
    cleaned = [p.strip().upper().replace(".", "") for p in prefixes if p.strip()]
    if not cleaned:
        raise ValueError("icd_prefixes is empty")
    alt = "|".join(re.escape(p) for p in cleaned)
    if "CAUSABAS" not in columns:
        raise KeyError(
            f"CAUSABAS absent from this DO layout; columns are {list(columns)[:20]}"
        )
    expr = pl.col("CAUSABAS").str.contains("^(" + alt + ")")
    if include_associated:
        for field in ASSOCIATED_CAUSE_FIELDS:
            if field in columns:
                # unanchored: these fields concatenate 4-character codes
                expr = expr | pl.col(field).str.contains(alt)
    return expr.fill_null(False)


def _decode_filter_one(job: tuple) -> tuple[str, bytes | None, str | None]:
    """Decode one cached DO file, keep matching deaths, return Arrow IPC bytes.

    Runs in a worker *process*: decoding a DO file is a pure-Python row walk
    through ``dbfread`` and is therefore GIL-bound, not I/O-bound, so a thread
    pool buys nothing. The worker returns a serialised frame rather than a
    ``DataFrame`` because only the tiny A27 subset crosses the process
    boundary; the ~1 million-row decoded state-year never leaves the child.
    """
    import io

    tag, path, prefixes, include_associated, projection, uf, year, release, sha = job
    try:
        frame = read_dbc(Path(path))
        hit = frame.filter(
            _cause_filter(prefixes, frame.columns, include_associated=include_associated)
        )
        del frame
        if hit.is_empty():
            return tag, None, None
        keep = [c for c in projection if c in hit.columns]
        hit = hit.select(keep).with_columns(
            pl.lit(uf, dtype=pl.Utf8).alias("_src_uf"),
            pl.lit(year, dtype=pl.Int32).alias("_src_year"),
            pl.lit(release, dtype=pl.Utf8).alias("_src_release"),
            pl.lit(sha, dtype=pl.Utf8).alias("_sha256"),
        )
        buf = io.BytesIO()
        hit.write_ipc(buf)
        return tag, buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001 - reported per file, batch survives
        return tag, None, f"{type(exc).__name__}: {exc}"


def fetch_icd_deaths(
    icd_prefixes: list[str],
    years: Iterable[int],
    ufs: Sequence[str] | None = None,
    *,
    include_associated: bool = False,
    fields: Sequence[str] | None = None,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
    skip_missing: bool = True,
) -> pl.DataFrame:
    """Stream DO files and keep deaths coded to given CID-10 prefixes.

    By default only ``CAUSABAS`` is matched, anchored, which is the definition
    of a cause-specific death. ``include_associated=True`` widens the match to
    the multi-code line fields; the result is then a set of death certificates
    *mentioning* the cause and must be reported as such.

    One state-year is decoded at a time. ``_src_uf``, ``_src_year``,
    ``_src_release`` and ``_sha256`` are attached.
    """
    wanted_ufs = [u.upper() for u in (ufs or UFS)]
    wanted_years = sorted(set(years))
    listing = list_available(ufs=wanted_ufs, backend=backend)
    index: dict[tuple[str, int], list[SimFile]] = {}
    for f in listing:
        index.setdefault((f.uf, f.year), []).append(f)
    projection = list(fields) if fields is not None else list(DEFAULT_FIELDS)
    if include_associated:
        projection = projection + [
            f for f in ASSOCIATED_CAUSE_FIELDS if f not in projection
        ]

    # Download everything first (thread pool), then decode (process pool).
    #
    # DATASUS FTP is latency-bound: a measured serial pass over 50 SIM files
    # (52.6 MB) took 212.8 s against 17.0 s with sixteen connections -- a
    # 12.5-fold difference that is entirely per-transfer handshake cost, not
    # bandwidth.
    #
    # Decoding is the opposite: ``dbfread`` walks every record in Python, so it
    # is CPU- and GIL-bound at roughly 11 s per state-year, which over 486 files
    # dominates the whole extraction. It is moved to a ProcessPoolExecutor;
    # each child holds exactly one decoded state-year and returns only the
    # matched A27 rows as Arrow IPC bytes, so the parent's memory profile stays
    # flat and the peak is (workers x one state-year) rather than the series.
    wanted: list[SimFile] = []
    for year in wanted_years:
        for uf in wanted_ufs:
            available = index.get((uf, year))
            if available:
                wanted.append(_find(available, uf, year, prefer))
    if wanted:
        report = datasus_ftp.fetch_many(
            wanted, lambda f: _cache_key(f.name),
            source=SOURCE_ID, backend=backend, refresh=refresh,
            max_workers=int(os.environ.get("BREPI_FTP_WORKERS", "12")),
            on_progress=lambda p: logger.info("SIM download {}", p.line()),
        )
        if report.failures:
            logger.warning("SIM: %d file(s) failed to download: %s",
                           len(report.failures),
                           list(report.failures)[:5])

    jobs: list[tuple] = []
    for year in wanted_years:
        for uf in wanted_ufs:
            available = index.get((uf, year))
            if not available:
                if skip_missing:
                    continue
                raise FileNotFoundError(f"SIM DO missing for {uf} {year}")
            target = _find(available, uf, year, prefer)
            key = _cache_key(target.name)
            prov = cache.read_provenance(key)
            path = cache.cache_path(key)
            if prov is None or not path.exists():
                if skip_missing:
                    logger.warning("SIM {} {}: not in cache after download", uf, year)
                    continue
                raise FileNotFoundError(f"SIM DO not cached for {uf} {year}")
            jobs.append((
                f"{uf}{year}", str(path), list(icd_prefixes), include_associated,
                list(projection), uf, year, target.release, prov.sha256,
            ))

    parts: list[pl.DataFrame] = []
    workers = max(1, min(int(os.environ.get("BREPI_DECODE_WORKERS", "8")), len(jobs) or 1))
    if jobs:
        import io as _io

        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for tag, payload, err in pool.map(_decode_filter_one, jobs, chunksize=1):
                done += 1
                if err is not None:
                    logger.warning("SIM decode failed for {}: {}", tag, err)
                elif payload is not None:
                    parts.append(pl.read_ipc(_io.BytesIO(payload)))
                if done % 25 == 0 or done == len(jobs):
                    logger.info("SIM decode {}/{} state-years", done, len(jobs))
    if not parts:
        logger.warning("no SIM deaths matched {} in the requested window", icd_prefixes)
        return pl.DataFrame(
            schema={
                **{c: pl.Utf8 for c in projection},
                "_src_uf": pl.Utf8,
                "_src_year": pl.Int32,
                "_src_release": pl.Utf8,
                "_sha256": pl.Utf8,
            }
        )
    out = pl.concat(parts, how="diagonal")
    logger.info("SIM {}: {} deaths over {} state-years", icd_prefixes, out.height, len(parts))
    return out
