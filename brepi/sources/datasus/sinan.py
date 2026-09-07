"""SINAN notifiable-disease case records, national annual files.

Coverage is agravo-by-agravo, not uniform. The tree holds two directories with
the same file convention ``{PREFIX}BR{YY}.dbc`` and national scope, one file
per notification year:

``/SINAN/DADOS/FINAIS``
    Closed years. Most agravos start in 2000 (a few in 2001 or 2007); the
    final series currently ends 2024.
``/SINAN/DADOS/PRELIM``
    Open years, presently 2023-2026 depending on agravo, rewritten
    continuously. A year can appear in *both* directories during the handover
    window; :func:`fetch_year` prefers ``final`` and records which it used.

Traps this module encodes.

*"Final" is not immutable.* DATASUS republishes closed year-files without
notice or version marker — the whole leptospirosis 2000-2017 block was
rewritten on 2026-07-15. Nothing but a hashed snapshot detects this, so every
read goes through :func:`brepi.io.cache.fetch` and :func:`snapshot` exists to
freeze a manifest alongside a paper.

*Schema drift.* The record layout changes between years: fields are added,
dropped, and resized. A naive vertical concat either raises or silently
aligns on position. :func:`fetch_range` takes the column union, fills the
gaps with null, and hands back a drift report so the analyst decides which
variables are usable over the whole window rather than discovering a
half-populated column at modelling time.

*Two-digit years.* File names carry ``YY``. The window used here is 00-30 ->
2000s, 31-99 -> 1900s, which is correct for SINAN (no pre-2000 national file
exists) and must not be copied verbatim to SIH, whose legacy tree really does
contain 1992.

All columns are returned as ``Utf8``; see :mod:`brepi.io.dbc` for why.
"""

from __future__ import annotations

import io
from functools import lru_cache
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import polars as pl
from loguru import logger

from brepi.config import DATASUS_ENCODING
from brepi.io import cache
from brepi.io.cache import Provenance
from brepi.io.dbc import read_dbc
from brepi.io import datasus_ftp
from brepi.io.datasus_ftp import Backend, RemoteFile, get, resolve

__all__ = [
    "AGRAVOS",
    "SinanFile",
    "Release",
    "list_available",
    "fetch_year",
    "fetch_range",
    "fetch_year_opendata",
    "snapshot",
    "SOURCE_ID",
]

SOURCE_ID = "datasus.sinan"

FINAIS_DIR = "/SINAN/DADOS/FINAIS"
PRELIM_DIR = "/SINAN/DADOS/PRELIM"
OPENDATA_DIR = "/Dados_Abertos/SINAN"

Release = Literal["final", "prelim"]

#: Short code -> (file prefix, human label). The short code equals the file
#: prefix for every agravo currently published; it is kept as a separate key
#: so that a future prefix change is one edit here and not a rename across
#: every study script.
AGRAVOS: dict[str, tuple[str, str]] = {
    "LEPT": ("LEPT", "Leptospirose"),
    "DENG": ("DENG", "Dengue"),
    "CHIK": ("CHIK", "Chikungunya"),
    "ZIKA": ("ZIKA", "Zika"),
    "MALA": ("MALA", "Malaria"),
    "ESQU": ("ESQU", "Esquistossomose"),
    "HANT": ("HANT", "Hantavirose"),
    "FMAC": ("FMAC", "Febre maculosa"),
    "LTAN": ("LTAN", "Leishmaniose tegumentar"),
    "LEIV": ("LEIV", "Leishmaniose visceral"),
    "CHAG": ("CHAG", "Chagas"),
    "TUBE": ("TUBE", "Tuberculose"),
    "MENI": ("MENI", "Meningite"),
    "ANIM": ("ANIM", "Acidente por animais peconhentos"),
    "IEXO": ("IEXO", "Intoxicacao exogena"),
    "VIOL": ("VIOL", "Violencia interpessoal/autoprovocada"),
    "HANS": ("HANS", "Hanseniase"),
    "ACBI": ("ACBI", "Acidente com material biologico"),
    "ACGR": ("ACGR", "Acidente de trabalho grave"),
    "ANTR": ("ANTR", "Atendimento antirrabico"),
    "BOTU": ("BOTU", "Botulismo"),
    "COLE": ("COLE", "Colera"),
    "COQU": ("COQU", "Coqueluche"),
    "DIFT": ("DIFT", "Difteria"),
    "FTIF": ("FTIF", "Febre tifoide"),
    "PEST": ("PEST", "Peste"),
    "RAIV": ("RAIV", "Raiva humana"),
    "ROTA": ("ROTA", "Rotavirus"),
    "TETA": ("TETA", "Tetano acidental"),
    "TETN": ("TETN", "Tetano neonatal"),
    "TOXC": ("TOXC", "Toxoplasmose congenita"),
    "TOXG": ("TOXG", "Toxoplasmose gestacional"),
    "TRAC": ("TRAC", "Tracoma"),
}

#: Directory names under ``/Dados_Abertos/SINAN``. They are hand-made and do
#: not derive from the prefix -- note ``Leishmaniose_viceral`` (sic, upstream
#: typo) and ``Zikavirus``. Only agravos actually mirrored there appear.
_OPENDATA_DIRS: dict[str, str] = {
    "LEPT": "Leptospirose",
    "DENG": "Dengue",
    "CHIK": "Chikungunya",
    "ZIKA": "Zikavirus",
    "MALA": "Malaria",
    "ESQU": "Esquistossomose",
    "HANT": "Hantavirose",
    "FMAC": "Febre_maculosa",
    "LTAN": "Leishmaniose_tegumentar",
    "LEIV": "Leishmaniose_viceral",
    "CHAG": "Chagas_aguda",
    "TUBE": "Tuberculose",
    "MENI": "Meningite",
    "ANIM": "Acidente_anim_peconhentos",
    "IEXO": "Intoxicacao_exogena",
    "ACBI": "Acidente_tbr_mat_biologico",
    "ACGR": "Acidente_trabalho",
    "ANTR": "Atendim_antirrabico_humano",
    "BOTU": "Botulismo",
    "COLE": "Colera",
    "COQU": "Coqueluche",
    "DIFT": "Difteria",
    "FTIF": "Febre_tifoide",
    "PEST": "Peste",
    "RAIV": "Raiva_humana",
    "ROTA": "Rotavirus",
    "TETA": "Tetano_acidental",
    "TETN": "Tetano_neonatal",
    "TOXC": "Toxoplasmose_congenita",
    "TOXG": "Toxoplasmose_gestacional",
    "TRAC": "Tracoma",
}


@dataclass(frozen=True)
class SinanFile(RemoteFile):
    """A remote SINAN year-file with its parsed year and release channel."""

    year: int = 0
    release: Release = "final"
    agravo: str = ""


def _prefix(agravo: str) -> str:
    key = agravo.upper()
    if key not in AGRAVOS:
        raise KeyError(
            f"unknown agravo {agravo!r}; known: {', '.join(sorted(AGRAVOS))}"
        )
    return AGRAVOS[key][0]


def _year_from_yy(yy: str) -> int:
    """Window a two-digit SINAN year. 00-30 -> 2000s, else 1900s."""
    n = int(yy)
    return 2000 + n if n <= 30 else 1900 + n


def _cache_key(agravo: str, filename: str) -> str:
    return f"datasus/sinan/{agravo.upper()}/{filename}"


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _list_available_cached(agravo: str, backend: Backend) -> tuple[SinanFile, ...]:
    return tuple(_list_available_uncached(agravo, backend=backend))


def list_available(agravo: str, *, backend: Backend = "ftp") -> list[SinanFile]:
    """Listing for one agravo. Memoised per process: the directory does not
    change mid-run, and an FTP LIST costs ~2 s, which dominates a cache hit.
    """
    return list(_list_available_cached(agravo, backend))


def _list_available_uncached(agravo: str, *, backend: Backend = "ftp") -> list[SinanFile]:
    """List every national year-file for an agravo, both release channels.

    Returned sorted by ``(year, release)`` with ``final`` before ``prelim``.
    A year present in both directories yields two entries; that overlap is
    real and callers should not assume uniqueness on ``year`` alone.
    """
    prefix = _prefix(agravo)
    pattern = rf"{prefix}BR(\d{{2}})\.dbc$"
    out: list[SinanFile] = []
    for directory, release in ((FINAIS_DIR, "final"), (PRELIM_DIR, "prelim")):
        for rf in resolve(directory, pattern, backend=backend):
            yy = rf.name[len(prefix) + 2 : len(prefix) + 4]
            if not yy.isdigit():
                continue
            out.append(
                SinanFile(
                    path=rf.path,
                    name=rf.name,
                    bytes=rf.bytes,
                    modified=rf.modified,
                    year=_year_from_yy(yy),
                    release=release,  # type: ignore[arg-type]
                    agravo=agravo.upper(),
                )
            )
    out.sort(key=lambda f: (f.year, 0 if f.release == "final" else 1))
    return out


def _pick(
    files: Sequence[SinanFile], year: int, prefer: Release
) -> SinanFile:
    candidates = [f for f in files if f.year == year]
    if not candidates:
        raise FileNotFoundError(f"no SINAN file for year {year}")
    for release in (prefer, "prelim" if prefer == "final" else "final"):
        for f in candidates:
            if f.release == release:
                return f
    return candidates[0]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def fetch_year(
    agravo: str,
    year: int,
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
    files: Sequence[SinanFile] | None = None,
) -> tuple[pl.DataFrame, Provenance]:
    """Fetch and decode one agravo-year.

    ``files`` accepts a pre-computed listing so that a range fetch performs a
    single directory listing instead of one per year. Bytes always land in the
    cache first, so a second call with ``refresh=False`` is provably offline.
    """
    listing = list(files) if files is not None else list_available(agravo, backend=backend)
    target = _pick(listing, year, prefer)
    key = _cache_key(agravo, target.name)
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
                "agravo": agravo.upper(),
                "year": year,
                "release": target.release,
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(
        key, _loader, source=SOURCE_ID, uri=uri, refresh=refresh
    )
    frame = read_dbc(path)
    logger.info(
        "SINAN {} {} ({}) -> {} rows x {} cols",
        agravo.upper(),
        year,
        target.release,
        frame.height,
        frame.width,
    )
    return frame, prov


def _drift_report(per_year: dict[int, list[str]]) -> pl.DataFrame:
    """Which column appears in which year."""
    years = sorted(per_year)
    all_cols: list[str] = []
    for y in years:
        for c in per_year[y]:
            if c not in all_cols:
                all_cols.append(c)
    rows = []
    for col in all_cols:
        present = [y for y in years if col in per_year[y]]
        missing = [y for y in years if col not in per_year[y]]
        rows.append(
            {
                "column": col,
                "n_years": len(present),
                "n_missing": len(missing),
                "first_year": present[0] if present else None,
                "last_year": present[-1] if present else None,
                "stable": len(missing) == 0,
                "present_in": ",".join(str(y) for y in present),
                "missing_in": ",".join(str(y) for y in missing),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "column": pl.Utf8,
            "n_years": pl.Int32,
            "n_missing": pl.Int32,
            "first_year": pl.Int32,
            "last_year": pl.Int32,
            "stable": pl.Boolean,
            "present_in": pl.Utf8,
            "missing_in": pl.Utf8,
        },
    )


def fetch_range(
    agravo: str,
    years: Iterable[int],
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
    columns: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch several years and stack them, tolerating schema drift.

    Returns ``(frame, drift)``. ``frame`` carries the union of all columns,
    nulls where a year did not have one, plus ``_src_year``, ``_release`` and
    ``_sha256`` so every row can be traced to the exact artefact it came from.
    ``drift`` is the per-column presence table from which years a column
    exists in; inspect it before treating any variable as a covariate.

    ``columns`` restricts the projection *after* the drift audit, so a column
    that is missing in some years is still reported as missing rather than
    quietly absent.
    """
    listing = list_available(agravo, backend=backend)
    frames: list[pl.DataFrame] = []
    per_year: dict[int, list[str]] = {}
    for year in sorted(set(years)):
        frame, prov = fetch_year(
            agravo,
            year,
            backend=backend,
            refresh=refresh,
            prefer=prefer,
            files=listing,
        )
        per_year[year] = list(frame.columns)
        release = (prov.params or {}).get("release", prefer)
        frames.append(
            frame.with_columns(
                pl.lit(year, dtype=pl.Int32).alias("_src_year"),
                pl.lit(str(release), dtype=pl.Utf8).alias("_release"),
                pl.lit(prov.sha256, dtype=pl.Utf8).alias("_sha256"),
            )
        )
    if not frames:
        raise ValueError("no years requested")

    drift = _drift_report(per_year)
    union: list[str] = []
    for f in frames:
        for c in f.columns:
            if c not in union:
                union.append(c)
    aligned = [
        f.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias(c) for c in union if c not in f.columns]
        ).select(union)
        for f in frames
    ]
    out = pl.concat(aligned, how="vertical")
    if columns is not None:
        keep = [c for c in columns if c in out.columns]
        missing = sorted(set(columns) - set(keep))
        if missing:
            raise KeyError(
                f"requested columns absent from every year of {agravo}: {missing}"
            )
        out = out.select([*keep, "_src_year", "_release", "_sha256"])
    logger.info(
        "SINAN {} {}-{}: {} rows, {} columns, {} unstable",
        agravo.upper(),
        min(per_year),
        max(per_year),
        out.height,
        out.width,
        int(drift.filter(~pl.col("stable")).height),
    )
    return out, drift


def snapshot(
    agravo: str,
    years: Iterable[int],
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
    name: str | None = None,
) -> Path:
    """Freeze an agravo-year set into a dated cache manifest.

    The manifest is named ``sinan_{agravo}_{YYYYMMDD}`` and is the analytic
    dataset's identity: archive it with the paper and re-run
    :func:`brepi.io.cache.verify_snapshot` before every modelling session.
    """
    listing = list_available(agravo, backend=backend)
    keys: list[str] = []
    for year in sorted(set(years)):
        target = _pick(listing, year, prefer)
        fetch_year(
            agravo, year, backend=backend, refresh=refresh, prefer=prefer, files=listing
        )
        keys.append(_cache_key(agravo, target.name))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    manifest = name or f"sinan_{agravo.upper()}_{stamp}"
    path = cache.write_manifest(manifest, keys)
    logger.info("wrote manifest {} with {} artefacts", manifest, len(keys))
    return path


# --------------------------------------------------------------------------
# Open-data mirror (alternative transport)
# --------------------------------------------------------------------------


def fetch_year_opendata(
    agravo: str,
    year: int,
    *,
    fmt: Literal["csv", "json", "xml"] = "csv",
    backend: Backend = "ftp",
    refresh: bool = False,
) -> tuple[pl.DataFrame | bytes, Provenance]:
    """Fetch one agravo-year from ``/Dados_Abertos/SINAN`` instead of the DBC.

    The open-data tree republishes the same year-files as
    ``{Nome}/{csv,json,xml}/{PREFIX}BR{YY}.csv.zip``. It exists for two
    situations: environments with no ``.dbc`` decompressor, and
    cross-validation of the decoder against an independently produced text
    rendering of the same extraction.

    It is **not** the authoritative series. The open-data copies are refreshed
    far less often than the ``.dbc`` -- as of 2026-07-30 they were roughly six
    months behind -- and the lag is neither documented nor uniform across
    agravos. Any counts that go into a paper must come from
    :func:`fetch_year`; use this only to check the ones that do.

    ``csv`` is decoded to a polars frame (all ``Utf8``, latin-1, ``;``- or
    ``,``-separated as sniffed). ``json`` and ``xml`` are returned as the raw
    decompressed bytes because their schemas are not stable enough to parse
    blindly.
    """
    key_agravo = agravo.upper()
    if key_agravo not in _OPENDATA_DIRS:
        raise KeyError(f"{key_agravo} is not mirrored under {OPENDATA_DIR}")
    prefix = _prefix(key_agravo)
    directory = f"{OPENDATA_DIR}/{_OPENDATA_DIRS[key_agravo]}/{fmt}"
    yy = f"{year % 100:02d}"
    filename = f"{prefix}BR{yy}.{fmt}.zip"
    matches = resolve(directory, rf"{prefix}BR{yy}\.{fmt}\.zip$", backend=backend)
    if not matches:
        raise FileNotFoundError(f"{directory}/{filename} not published")
    target = matches[0]
    ckey = f"datasus/sinan_opendata/{key_agravo}/{fmt}/{target.name}"

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
                "agravo": key_agravo,
                "year": year,
                "format": fmt,
                "transport": "opendata",
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(
        ckey, _loader, source=f"{SOURCE_ID}.opendata", uri=target.uri_ftp, refresh=refresh
    )
    members = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    names = [n for n in members.namelist() if not n.endswith("/")]
    if not names:
        raise ValueError(f"{target.name} is an empty archive")
    logger.info("opendata {} {} archive members: {}", key_agravo, year, names)
    raw = members.read(names[0])
    if fmt != "csv":
        return raw, prov

    text = raw.decode(DATASUS_ENCODING, errors="replace")
    head = text.split("\n", 1)[0]
    sep = ";" if head.count(";") >= head.count(",") else ","
    frame = pl.read_csv(
        text.encode("utf-8"),
        separator=sep,
        infer_schema_length=0,  # everything Utf8; see brepi.io.dbc
        truncate_ragged_lines=True,
    )
    frame = frame.rename({c: c.strip() for c in frame.columns})
    return frame, prov


# ---------------------------------------------------------------------------
# Materialised (decoded + normalised) access
# ---------------------------------------------------------------------------

#: Bump when the decode or normalisation logic changes in a way that alters
#: the produced frame. Everything derived under an older version becomes
#: unreachable rather than stale.
NORMALISE_VERSION = "1"


def fetch_year_normalised(
    agravo: str,
    year: int,
    *,
    system: str | None = None,
    columns: Sequence[str] | None = None,
    backend: Backend = "ftp",
    refresh: bool = False,
    prefer: Release = "final",
):
    """One agravo-year, decoded *and* code-translated, cached as parquet.

    Decoding a ``.dbc`` and translating its categoricals costs seconds per
    file and is deterministic given the bytes and the codebook version, so it
    is cached exactly like the download. Repeat analyses read parquet instead
    of re-running blast decompression and forty-odd category joins.

    ``system`` names the codebook binding set (default ``SINAN-<AGRAVO>``);
    pass ``system=""`` to skip translation and materialise the raw decode only.
    """
    from brepi.codebook import codebook as _cb
    from brepi.io import materialise as _mat

    listing = list_available(agravo, backend=backend)
    target = _pick(listing, year, prefer)
    key = _cache_key(agravo, target.name)
    binding = f"SINAN-{agravo.upper()}" if system is None else system

    def _build():
        frame, _ = fetch_year(agravo, year, backend=backend, refresh=refresh,
                              prefer=prefer, files=listing)
        if binding:
            try:
                frame = _cb.decode_frame(frame, binding)
            except _cb.CodebookError:
                logger.warning(
                    "no codebook bindings for {}; materialising the raw decode only",
                    binding)
        if columns:
            keep = [c for c in frame.columns if c in set(columns)]
            frame = frame.select(keep)
        return frame

    # Ensure the source *bytes* are cached so a content hash exists -- but do
    # not decode here. Calling fetch_year() eagerly would run the full blast
    # decompression on every cache hit, which is exactly the cost this
    # function exists to avoid.
    if cache.read_provenance(key) is None or refresh:
        datasus_ftp.fetch_many(
            [target], lambda f: key, source=SOURCE_ID, backend=backend,
            refresh=refresh, max_workers=1,
        ).raise_if_failed()

    return _mat.materialise(
        f"sinan/{agravo.lower()}",
        [key],
        _build,
        transform="decode+codebook" if binding else "decode",
        # The reference tables (municipality, UF) change what decode_frame()
        # emits just as much as the concepts do, so their version belongs in
        # the cache key. Omitting it left a decode cached across a fix and
        # mixed two region vocabularies in one column.
        transform_version=(
            f"{NORMALISE_VERSION}"
            f".{_cb.load_codebook().get('version', 0)}"
            f".{_cb.load_codebook().get('references_version', 0)}"
        ),
        options={"columns": sorted(columns) if columns else None, "binding": binding},
        refresh=refresh,
    )
