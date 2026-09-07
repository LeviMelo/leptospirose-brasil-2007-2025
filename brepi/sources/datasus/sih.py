"""SIH-SUS reduced AIH files (RD): public-hospital admissions.

``/SIHSUS/200801_/Dados`` holds the current series as ``RD{UF}{YY}{MM}.dbc``,
one file per state per competence month, from 200801 to the present. The
pre-2008 layout lives in ``/SIHSUS/199201_200712/Dados`` with the same naming
but a materially different record layout (fewer fields, different code
systems); it is listed by :func:`list_available` when asked for but is not
schema-compatible with the current series and must not be concatenated with
it without an explicit harmonisation step.

Traps.

*Not every file in that directory is per-UF.* ``CH``/``CM`` (and ``SP``'s
national aggregates) use ``CHBR{YY}{MM}``/``CMBR{YY}{MM}`` -- ``BR`` is the
whole country, not a state. A regex that treats characters 3-4 as a UF will
happily "find" a state named BR and produce national totals labelled as one
state. Only ``RD`` prefixed files are matched here.

*Two-digit years wrap.* The legacy tree runs 92..99 then 00..07, so a
lexical sort puts 2000 before 1992 and an "earliest file" heuristic returns
the wrong decade. Years are always parsed and windowed (>=92 -> 1900s) before
any ordering.

*Volume.* One national year of RD is on the order of 10^7 rows. The
extraction helper :func:`fetch_icd_admissions` therefore decodes one
UF-month at a time, filters, projects, and discards; peak memory is one month
of one state, not the series.

Rows are all ``Utf8``. Numeric AIH fields (``VAL_TOT``, ``DIAS_PERM``) must be
cast explicitly by the caller.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl
from loguru import logger

from brepi.io import cache
from brepi.io import datasus_ftp
from brepi.io.cache import Provenance
from brepi.io.dbc import read_dbc, read_dbc_where
from brepi.io.datasus_ftp import Backend, RemoteFile, get, resolve

__all__ = [
    "UFS",
    "SihFile",
    "CURRENT_DIR",
    "LEGACY_DIR",
    "list_available",
    "fetch_month",
    "fetch_icd_admissions",
    "SOURCE_ID",
]

SOURCE_ID = "datasus.sih"

CURRENT_DIR = "/SIHSUS/200801_/Dados"
LEGACY_DIR = "/SIHSUS/199201_200712/Dados"

UFS: tuple[str, ...] = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
)

_RD_RE = re.compile(r"^RD(?P<uf>[A-Z]{2})(?P<yy>\d{2})(?P<mm>\d{2})\.dbc$", re.IGNORECASE)

#: Columns that make an admission row usable without pulling the full ~110
#: field record. Overridable per call.
DEFAULT_FIELDS: tuple[str, ...] = (
    "N_AIH",
    "ANO_CMPT",
    "MES_CMPT",
    "UF_ZI",
    "MUNIC_RES",
    "MUNIC_MOV",
    "NASC",
    "SEXO",
    "IDADE",
    "COD_IDADE",
    "DT_INTER",
    "DT_SAIDA",
    "DIAG_PRINC",
    "DIAG_SECUN",
    "MORTE",
    "DIAS_PERM",
    "UTI_MES_TO",
    "VAL_TOT",
    "CAR_INT",
    "CNES",
)

_DIAG_FIELDS: tuple[str, ...] = ("DIAG_PRINC", "DIAG_SECUN")


@dataclass(frozen=True)
class SihFile(RemoteFile):
    """One RD file with its parsed competence and scope."""

    uf: str = ""
    year: int = 0
    month: int = 0
    series: str = "current"  # "current" (2008-) or "legacy" (1992-2007)


def _window_year(yy: str) -> int:
    """Window a SIH two-digit year: >=92 is the 1990s, everything else 2000s."""
    n = int(yy)
    return 1900 + n if n >= 92 else 2000 + n


def _cache_key(filename: str) -> str:
    return f"datasus/sih/rd/{filename.upper()}"


def list_available(
    *,
    ufs: Sequence[str] | None = None,
    include_legacy: bool = False,
    backend: Backend = "ftp",
) -> list[SihFile]:
    """List RD files, sorted by ``(year, month, uf)``.

    Only ``RD*`` is matched, which excludes the national ``CHBR``/``CMBR``
    files that share the directory. Sorting is on the parsed integers, never
    on the file name.
    """
    wanted = {u.upper() for u in ufs} if ufs else None
    out: list[SihFile] = []
    directories = [(CURRENT_DIR, "current")]
    if include_legacy:
        directories.append((LEGACY_DIR, "legacy"))
    for directory, series in directories:
        for rf in resolve(directory, r"RD[A-Z]{2}\d{4}\.dbc$", backend=backend):
            m = _RD_RE.match(rf.name)
            if not m:
                continue
            uf = m["uf"].upper()
            if uf == "BR" or (wanted is not None and uf not in wanted):
                continue
            month = int(m["mm"])
            if not 1 <= month <= 12:
                continue
            out.append(
                SihFile(
                    path=rf.path,
                    name=rf.name,
                    bytes=rf.bytes,
                    modified=rf.modified,
                    uf=uf,
                    year=_window_year(m["yy"]),
                    month=month,
                    series=series,
                )
            )
    out.sort(key=lambda f: (f.year, f.month, f.uf))
    return out


def _find(
    listing: Sequence[SihFile], uf: str, year: int, month: int
) -> SihFile:
    uf = uf.upper()
    for f in listing:
        if f.uf == uf and f.year == year and f.month == month:
            return f
    raise FileNotFoundError(f"no SIH RD file for {uf} {year}-{month:02d}")


def fetch_month(
    uf: str,
    year: int,
    month: int,
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    listing: Sequence[SihFile] | None = None,
) -> tuple[pl.DataFrame, Provenance]:
    """Fetch and decode one UF-month of RD.

    ``listing`` accepts a pre-computed directory listing; the SIH directory
    has >20k entries, so re-listing per month is the dominant cost of a
    multi-year extraction.
    """
    files = list(listing) if listing is not None else list_available(
        ufs=[uf], include_legacy=year < 2008, backend=backend
    )
    target = _find(files, uf, year, month)
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
                "month": target.month,
                "series": target.series,
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(key, _loader, source=SOURCE_ID, uri=uri, refresh=refresh)
    return read_dbc(path), prov


def _icd_filter(prefixes: Sequence[str], columns: Sequence[str]) -> pl.Expr:
    """``any DIAG_* starts with any prefix``, as one regex per column."""
    cleaned = [p.strip().upper() for p in prefixes if p.strip()]
    if not cleaned:
        raise ValueError("icd_prefixes is empty")
    rx = "^(" + "|".join(re.escape(p) for p in cleaned) + ")"
    present = [c for c in _DIAG_FIELDS if c in columns]
    if not present:
        raise KeyError(
            f"none of {_DIAG_FIELDS} present in this RD layout; columns are {list(columns)[:20]}"
        )
    expr = pl.col(present[0]).str.contains(rx)
    for c in present[1:]:
        expr = expr | pl.col(c).str.contains(rx)
    return expr.fill_null(False)


def _decode_filter_one(job: tuple) -> tuple[str, bytes | None, str | None]:
    """Decode one cached RD file down to the admissions matching ``prefixes``.

    Runs in a worker process and returns Arrow IPC bytes, so the ~10^5-10^6-row
    UF-month never crosses the process boundary -- only the tens of A27 rows.
    Selection uses :func:`brepi.io.dbc.read_dbc_where`, which evaluates the
    diagnosis predicate on the raw fixed-width bytes.
    """
    import io

    tag, path, prefixes, projection, uf, year, month, sha = job
    try:
        hit = read_dbc_where(
            Path(path),
            any_prefix={c: list(prefixes) for c in _DIAG_FIELDS},
            fields=projection,
        )
        if hit.is_empty():
            return tag, None, None
        hit = hit.with_columns(
            pl.lit(uf, dtype=pl.Utf8).alias("_src_uf"),
            pl.lit(year, dtype=pl.Int32).alias("_src_year"),
            pl.lit(month, dtype=pl.Int32).alias("_src_month"),
            pl.lit(sha, dtype=pl.Utf8).alias("_sha256"),
        )
        buf = io.BytesIO()
        hit.write_ipc(buf)
        return tag, buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001 - reported per file, batch survives
        return tag, None, f"{type(exc).__name__}: {exc}"


def fetch_icd_admissions(
    icd_prefixes: list[str],
    years: Iterable[int],
    ufs: Sequence[str] | None = None,
    *,
    fields: Sequence[str] | None = None,
    months: Sequence[int] | None = None,
    backend: Backend = "ftp",
    refresh: bool = False,
    skip_missing: bool = True,
) -> pl.DataFrame:
    """Stream RD files and keep only admissions coded to given CID-10 prefixes.

    ``icd_prefixes`` are matched as anchored prefixes against ``DIAG_PRINC``
    and ``DIAG_SECUN`` -- ``["A27"]`` catches ``A270``, ``A278``, ``A279``.
    Both columns are checked because leptospirosis is frequently a secondary
    code on an admission coded principally to renal failure or to an
    unspecified febrile illness.

    Memory is bounded by one UF-month: each file is decoded, filtered,
    projected onto ``fields`` (default :data:`DEFAULT_FIELDS`) and appended;
    the undecoded remainder is never retained. ``_src_uf``, ``_src_year``,
    ``_src_month`` and ``_sha256`` are added for traceability.

    ``skip_missing`` tolerates UF-months that were never published (a real
    occurrence in early 2008 and in the current open month); set it False to
    make an incomplete window a hard error.
    """
    wanted_ufs = [u.upper() for u in (ufs or UFS)]
    wanted_years = sorted(set(years))
    wanted_months = sorted(set(months)) if months else list(range(1, 13))
    listing = list_available(
        ufs=wanted_ufs,
        include_legacy=any(y < 2008 for y in wanted_years),
        backend=backend,
    )
    index = {(f.uf, f.year, f.month): f for f in listing}
    projection = list(fields) if fields is not None else list(DEFAULT_FIELDS)

    # One calendar year at a time: download that year's UF-months concurrently
    # (FTP is latency-bound), then decode them concurrently (byte-level
    # predicate on a fixed-width block, one process per file). Interleaving by
    # year rather than staging all ~6,000 files first means a run that is
    # stopped early still leaves a complete set of whole years, which is the
    # only kind of partial SIH coverage that can be honestly reported.
    ftp_workers = int(os.environ.get("BREPI_FTP_WORKERS", "12"))
    decode_workers = int(os.environ.get("BREPI_DECODE_WORKERS", "8"))
    parts: list[pl.DataFrame] = []
    covered: list[tuple[int, int]] = []
    for year in wanted_years:
        targets: list[SihFile] = []
        for month in wanted_months:
            for uf in wanted_ufs:
                target = index.get((uf, year, month))
                if target is None:
                    if skip_missing:
                        continue
                    raise FileNotFoundError(f"SIH RD missing for {uf} {year}-{month:02d}")
                targets.append(target)
        if not targets:
            continue
        report = datasus_ftp.fetch_many(
            targets, lambda f: _cache_key(f.name),
            source=SOURCE_ID, backend=backend, refresh=refresh,
            max_workers=ftp_workers,
            on_progress=lambda p: logger.info("SIH {} download {}", year, p.line()),
        )
        if report.failures:
            logger.warning(
                "SIH {}: {} file(s) failed to download: {}",
                year, len(report.failures), list(report.failures)[:5],
            )

        jobs: list[tuple] = []
        for t in targets:
            key = _cache_key(t.name)
            prov = cache.read_provenance(key)
            path = cache.cache_path(key)
            if prov is None or not path.exists():
                continue
            jobs.append((
                t.name, str(path), list(icd_prefixes), list(projection),
                t.uf, t.year, t.month, prov.sha256,
            ))
        if not jobs:
            continue
        n_hit = 0
        with ProcessPoolExecutor(
            max_workers=max(1, min(decode_workers, len(jobs)))
        ) as pool:
            import io as _io

            for tag, payload, err in pool.map(_decode_filter_one, jobs, chunksize=1):
                if err is not None:
                    logger.warning("SIH decode failed for {}: {}", tag, err)
                elif payload is not None:
                    frame = pl.read_ipc(_io.BytesIO(payload))
                    n_hit += frame.height
                    parts.append(frame)
        covered.extend((t.year, t.month) for t in targets)
        logger.info(
            "SIH {}: {} UF-months decoded, {} matching admissions", year, len(jobs), n_hit
        )
    if covered:
        months_seen = sorted(set(covered))
        logger.info(
            "SIH coverage: {} competence months, {}-{} to {}-{}",
            len(months_seen), months_seen[0][0], months_seen[0][1],
            months_seen[-1][0], months_seen[-1][1],
        )
    if not parts:
        logger.warning("no SIH admissions matched {} in the requested window", icd_prefixes)
        return pl.DataFrame(
            schema={
                **{c: pl.Utf8 for c in projection},
                "_src_uf": pl.Utf8,
                "_src_year": pl.Int32,
                "_src_month": pl.Int32,
                "_sha256": pl.Utf8,
            }
        )
    out = pl.concat(parts, how="diagonal")
    logger.info(
        "SIH {}: {} admissions over {} UF-months",
        icd_prefixes,
        out.height,
        len(parts),
    )
    return out
