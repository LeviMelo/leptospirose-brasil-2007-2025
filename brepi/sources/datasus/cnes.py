"""CNES establishment register, monthly competence files.

``/CNES/200508_/Dados/{GROUP}/{GROUP}{UF}{YY}{MM}.dbc``, one file per group
per state per competence month from 200508 onward. The groups are separate
tables about the same establishments, keyed on ``CNES``:

======  ====================================================================
``DC``  detail: complementary equipment/services declared
``EE``  teaching establishments
``EF``  philanthropic establishments
``EP``  professional-level equipment
``EQ``  equipment inventory
``GM``  management/goal contracts
``HB``  habilitations (accredited high-complexity services)
``IN``  incentives
``LT``  beds, by bed type: the bed-count table
``PF``  individual professionals
``RC``  regulation contracts
``SR``  specialised services
``ST``  establishment master record: one row per CNES per competence
======  ====================================================================

Used here as a *detection-capacity* covariate rather than as a health-system
description: whether a leptospirosis case is notified at all depends on
whether the municipality has a hospital that can admit and investigate it.

Traps.

*The register is a monthly snapshot, not a time series.* An establishment
that closes simply stops appearing. Differencing two competences is the only
way to see change, and a municipality with zero rows in a month is
indistinguishable from a transmission failure at the state level -- check row
counts per UF before believing a zero.

*Bed counting.* ``LT`` has one row per (CNES, TP_LEITO, CODLEITO) with
``QT_EXIST`` (installed) and ``QT_SUS`` (of those, available to SUS).
``QT_EXIST`` is the physical capacity; ``QT_SUS`` is the capacity a public
patient can actually reach, and for a notification-capacity covariate it is
usually the right one. Summing rows without grouping double-counts nothing,
but summing ``QT_EXIST + QT_SUS`` counts the same beds twice.

*ICU beds.* ``TP_LEITO == "3"`` is the "complementar" group, which contains
both intensive care and lower-acuity intermediate units. Strict ICU is
``CODLEITO`` 74-83 (adult I/II/III, paediatric, neonatal, burns, coronary);
85-89 are intermediate units and 92-96 are the newer intermediate/neonatal
care codes. :data:`ICU_BED_CODES` and :data:`INTERMEDIATE_BED_CODES` keep the
two separable, because papers that conflate them overstate ICU supply by
roughly a third.

All values are ``Utf8``; quantity columns are cast explicitly where summed.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import polars as pl
from loguru import logger

from brepi.io import cache
from brepi.io.cache import Provenance
from brepi.io.dbc import read_dbc
from brepi.io.datasus_ftp import Backend, RemoteFile, get, resolve

__all__ = [
    "GROUPS",
    "UFS",
    "CnesFile",
    "CnesInventory",
    "BASE_DIR",
    "ICU_BED_CODES",
    "INTERMEDIATE_BED_CODES",
    "list_available",
    "fetch_month",
    "capacity_by_municipality",
    "capacity_series",
    "resolve_inventory",
    "SOURCE_ID",
]

SOURCE_ID = "datasus.cnes"

BASE_DIR = "/CNES/200508_/Dados"

GROUPS: tuple[str, ...] = (
    "DC", "EE", "EF", "EP", "EQ", "GM", "HB", "IN", "LT", "PF", "RC", "SR", "ST",
)

UFS: tuple[str, ...] = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
)

#: ``LT.CODLEITO`` values that are intensive care proper.
ICU_BED_CODES: tuple[str, ...] = (
    "74", "75", "76", "77", "78", "79", "80", "81", "82", "83",
)

#: Intermediate / lower-acuity complementary beds. Not ICU.
INTERMEDIATE_BED_CODES: tuple[str, ...] = (
    "64", "65", "66", "85", "86", "87", "88", "89", "92", "93", "94", "95", "96",
)

#: CNES changed the specialised-service taxonomy between competence 2007-12
#: and 2008-12. The legacy table calls clinical laboratory service ``013``;
#: the replacement calls it ``145``. Official legacy lookup:
#: https://tabnet.datasus.gov.br/cgi/cnes/servi%C3%A7o_classifica%C3%A7%C3%A3o.htm
LABORATORY_SERVICE_CODES_BY_ERA: tuple[tuple[int, str], ...] = (
    (2007, "013"),
    (9999, "145"),
)

_FILE_RE = re.compile(
    r"^(?P<grp>[A-Z]{2})(?P<uf>[A-Z]{2})(?P<yy>\d{2})(?P<mm>\d{2})\.dbc$", re.IGNORECASE
)


@dataclass(frozen=True)
class CnesFile(RemoteFile):
    """One CNES group-UF-competence file."""

    group: str = ""
    uf: str = ""
    year: int = 0
    month: int = 0

    @property
    def competence(self) -> str:
        return f"{self.year:04d}{self.month:02d}"


@dataclass(frozen=True)
class CnesInventory:
    """Frozen complete set of files requested for a CNES stock series."""

    groups: tuple[str, ...]
    ufs: tuple[str, ...]
    years: tuple[int, ...]
    month: int
    files: tuple[CnesFile, ...]

    @property
    def by_group(self) -> dict[str, list[CnesFile]]:
        return {
            group: [item for item in self.files if item.group == group]
            for group in self.groups
        }

    @property
    def cache_keys(self) -> list[str]:
        return [_cache_key(item.group, item.name) for item in self.files]


def _check_group(group: str) -> str:
    g = group.upper()
    if g not in GROUPS:
        raise KeyError(f"unknown CNES group {group!r}; known: {', '.join(GROUPS)}")
    return g


def _cache_key(group: str, filename: str) -> str:
    return f"datasus/cnes/{group.upper()}/{filename.upper()}"


def list_available(
    group: str,
    *,
    ufs: Sequence[str] | None = None,
    backend: Backend = "ftp",
) -> list[CnesFile]:
    """List one group's files, sorted by ``(year, month, uf)``.

    The series starts at competence 200508, so the two-digit year is
    unambiguously 2000s; no windowing is needed or applied.
    """
    g = _check_group(group)
    wanted = {u.upper() for u in ufs} if ufs else None
    out: list[CnesFile] = []
    for rf in resolve(f"{BASE_DIR}/{g}", rf"{g}[A-Z]{{2}}\d{{4}}\.dbc$", backend=backend):
        m = _FILE_RE.match(rf.name)
        if not m:
            continue
        uf = m["uf"].upper()
        month = int(m["mm"])
        if uf not in UFS or not 1 <= month <= 12:
            continue
        if wanted is not None and uf not in wanted:
            continue
        out.append(
            CnesFile(
                path=rf.path,
                name=rf.name,
                bytes=rf.bytes,
                modified=rf.modified,
                group=g,
                uf=uf,
                year=2000 + int(m["yy"]),
                month=month,
            )
        )
    out.sort(key=lambda f: (f.year, f.month, f.uf))
    return out


def _find(listing: Sequence[CnesFile], uf: str, year: int, month: int) -> CnesFile:
    uf = uf.upper()
    for f in listing:
        if f.uf == uf and f.year == year and f.month == month:
            return f
    raise FileNotFoundError(f"no CNES file for {uf} {year}-{month:02d}")


def fetch_month(
    group: str,
    uf: str,
    year: int,
    month: int,
    *,
    backend: Backend = "ftp",
    refresh: bool = False,
    listing: Sequence[CnesFile] | None = None,
) -> tuple[pl.DataFrame, Provenance]:
    """Fetch and decode one group-UF-competence file."""
    g = _check_group(group)
    files = list(listing) if listing is not None else list_available(
        g, ufs=[uf], backend=backend
    )
    target = _find(files, uf, year, month)
    key = _cache_key(g, target.name)
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
                "group": g,
                "uf": target.uf,
                "year": target.year,
                "month": target.month,
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(key, _loader, source=SOURCE_ID, uri=uri, refresh=refresh)
    return read_dbc(path), prov


def _q(name: str) -> pl.Expr:
    """Cast a CNES quantity column to Int64, treating blanks as zero."""
    return pl.col(name).cast(pl.Int64, strict=False).fill_null(0)


def _laboratory_service_code(year: int) -> str:
    return next(
        code
        for last_year, code in LABORATORY_SERVICE_CODES_BY_ERA
        if year <= last_year
    )


def _normalise_cnes_municipality(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse CNES' Distrito Federal RA codes to Brasília's IBGE code.

    Historical CNES snapshots use ``530020`` ... ``530180`` for DF
    administrative regions. They are not IBGE municipalities. Distrito Federal
    has one municipality-equivalent (Brasília, DATASUS code ``530010``), so
    this collapse is exact at the municipal level and preserves capacity mass.
    """
    clean = pl.col("CODUFMUN").cast(pl.Utf8).str.strip_chars().str.zfill(6)
    return frame.with_columns(
        pl.when(clean.str.starts_with("53"))
        .then(pl.lit("530010"))
        .otherwise(clean)
        .alias("CODUFMUN")
    )


def _laboratories_by_municipality(
    sr: pl.DataFrame,
    *,
    service_code: str = "145",
) -> pl.DataFrame:
    """Count establishments declaring the era-appropriate lab service.

    Legacy service 013 and modern service 145 both denote clinical laboratory
    service. Counting distinct CNES identifiers avoids multiplying an
    establishment by service classifications. This is laboratory availability,
    not test volume, quality, or leptospirosis-specific capacity.
    """
    required = {"CNES", "CODUFMUN", "SERV_ESP"}
    missing = required - set(sr.columns)
    if missing:
        raise KeyError(f"CNES SR lacks {sorted(missing)}")
    return (
        sr.filter(
            pl.col("SERV_ESP").cast(pl.Utf8).str.strip_chars()
            == service_code
        )
        .group_by("CODUFMUN")
        .agg(pl.col("CNES").n_unique().alias("n_labs"))
    )


def capacity_by_municipality(
    year: int,
    month: int,
    *,
    groups: Sequence[str] = ("ST", "LT", "SR"),
    ufs: Sequence[str] | None = None,
    backend: Backend = "ftp",
    refresh: bool = False,
    skip_missing: bool = True,
    listings: Mapping[str, Sequence[CnesFile]] | None = None,
    workers: int = 1,
) -> pl.DataFrame:
    """Municipality-level service-capacity covariates for one competence.

    Fields used, explicitly:

    ``ST.CODUFMUN``
        6-digit IBGE municipality code of the establishment (not of the
        patient). One row per establishment, so ``n_estab`` is a distinct
        count of ``ST.CNES``.
    ``ST.TP_UNID``
        establishment type; ``"05"`` and ``"07"`` are general and specialised
        hospitals, counted separately as ``n_hospitals`` because a
        municipality of 40 clinics and no hospital cannot investigate a
        severe case.
    ``LT.QT_EXIST`` / ``LT.QT_SUS``
        installed and SUS-available beds on the (CNES, TP_LEITO, CODLEITO)
        row, summed to the municipality.
    ``LT.TP_LEITO`` / ``LT.CODLEITO``
        bed classification; ICU is ``CODLEITO`` in :data:`ICU_BED_CODES`,
        intermediate units are reported separately.

    Returns one row per municipality present in either group, with
    ``munic_code`` (6-digit), ``year``, ``month``, ``n_estab``,
    ``n_hospitals``, ``beds_total``, ``beds_sus``, ``icu_beds``,
    ``icu_beds_sus``, ``intermediate_beds``, and ``n_labs`` (distinct
    establishments declaring the era-appropriate clinical-laboratory service).
    Municipalities with establishments but no beds get zeros, not nulls; a municipality absent
    from CNES entirely is absent from the frame, which is the honest encoding
    of "no registered establishment".

    Each UF-group file is decoded independently. ``workers`` may be increased
    for national extracts; the transport maintains one stateful FTP connection
    per worker rather than sharing one unsafe connection between threads.
    """
    if int(workers) < 1:
        raise ValueError("workers must be at least 1")
    wanted_ufs = [u.upper() for u in (ufs or UFS)]
    requested = [_check_group(g) for g in groups]
    st_parts: list[pl.DataFrame] = []
    lt_parts: list[pl.DataFrame] = []
    sr_parts: list[pl.DataFrame] = []
    jobs: list[tuple[str, str, CnesFile]] = []
    for group in requested:
        listing = (
            list(listings[group])
            if listings is not None and group in listings
            else list_available(group, ufs=wanted_ufs, backend=backend)
        )
        index = {(f.uf, f.year, f.month): f for f in listing}
        for uf in wanted_ufs:
            target = index.get((uf, year, month))
            if target is None:
                if skip_missing:
                    logger.warning("CNES {} {} {}-{:02d} not published", group, uf, year, month)
                    continue
                raise FileNotFoundError(f"CNES {group} missing for {uf} {year}-{month:02d}")
            jobs.append((group, uf, target))

    if not jobs:
        raise FileNotFoundError(
            f"no CNES data retrieved for {year}-{month:02d} in groups {requested}"
        )

    def _fetch_partition(job: tuple[str, str, CnesFile]) -> tuple[str, str, pl.DataFrame]:
        group, uf, target = job
        frame, _prov = fetch_month(
            group,
            uf,
            year,
            month,
            backend=backend,
            refresh=refresh,
            listing=[target],
        )
        frame = _normalise_cnes_municipality(frame)
        return group, uf, frame

    if int(workers) == 1:
        partitions = map(_fetch_partition, jobs)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(int(workers), len(jobs)),
            thread_name_prefix="cnes",
        )
        partitions = executor.map(_fetch_partition, jobs)

    try:
        for group, uf, frame in partitions:
            if group == "ST":
                missing = {"CNES", "CODUFMUN"} - set(frame.columns)
                if missing:
                    raise KeyError(
                        f"CNES ST {uf} {year}-{month:02d} lacks {sorted(missing)}"
                    )
                st_parts.append(
                    frame.select(
                        [
                            c
                            for c in ("CNES", "CODUFMUN", "TP_UNID")
                            if c in frame.columns
                        ]
                    )
                )
            elif group == "LT":
                missing = {"CODUFMUN", "QT_EXIST"} - set(frame.columns)
                if missing:
                    raise KeyError(
                        f"CNES LT {uf} {year}-{month:02d} lacks {sorted(missing)}"
                    )
                lt_parts.append(
                    frame.select(
                        [
                            c
                            for c in (
                                "CNES",
                                "CODUFMUN",
                                "TP_LEITO",
                                "CODLEITO",
                                "QT_EXIST",
                                "QT_SUS",
                            )
                            if c in frame.columns
                        ]
                    )
                )
            elif group == "SR":
                missing = {"CNES", "CODUFMUN", "SERV_ESP"} - set(frame.columns)
                if missing:
                    raise KeyError(
                        f"CNES SR {uf} {year}-{month:02d} lacks "
                        f"{sorted(missing)}"
                    )
                sr_parts.append(frame.select("CNES", "CODUFMUN", "SERV_ESP"))
            del frame
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    est = None
    if st_parts:
        st = pl.concat(st_parts, how="diagonal")
        hospital = (
            pl.col("TP_UNID").is_in(["05", "07"])
            if "TP_UNID" in st.columns
            else pl.lit(False)
        )
        est = (
            st.with_columns(hospital.alias("_is_hosp"))
            .group_by("CODUFMUN")
            .agg(
                pl.col("CNES").n_unique().alias("n_estab"),
                pl.col("CNES").filter(pl.col("_is_hosp")).n_unique().alias("n_hospitals"),
            )
        )

    beds = None
    if lt_parts:
        lt = pl.concat(lt_parts, how="diagonal")
        if "QT_SUS" not in lt.columns:
            lt = lt.with_columns(pl.lit("0").alias("QT_SUS"))
        if "CODLEITO" not in lt.columns:
            lt = lt.with_columns(pl.lit(None, dtype=pl.Utf8).alias("CODLEITO"))
        is_icu = pl.col("CODLEITO").is_in(list(ICU_BED_CODES))
        is_int = pl.col("CODLEITO").is_in(list(INTERMEDIATE_BED_CODES))
        beds = lt.group_by("CODUFMUN").agg(
            _q("QT_EXIST").sum().alias("beds_total"),
            _q("QT_SUS").sum().alias("beds_sus"),
            _q("QT_EXIST").filter(is_icu).sum().alias("icu_beds"),
            _q("QT_SUS").filter(is_icu).sum().alias("icu_beds_sus"),
            _q("QT_EXIST").filter(is_int).sum().alias("intermediate_beds"),
        )

    labs = (
        _laboratories_by_municipality(
            pl.concat(sr_parts, how="diagonal"),
            service_code=_laboratory_service_code(year),
        )
        if sr_parts
        else None
    )
    out = None
    for block in (est, beds, labs):
        if block is None:
            continue
        out = (
            block
            if out is None
            else out.join(block, on="CODUFMUN", how="full", coalesce=True)
        )
    assert out is not None

    for col, default in (
        ("n_estab", 0),
        ("n_hospitals", 0),
        ("beds_total", 0),
        ("beds_sus", 0),
        ("icu_beds", 0),
        ("icu_beds_sus", 0),
        ("intermediate_beds", 0),
        ("n_labs", 0),
    ):
        if col not in out.columns:
            out = out.with_columns(pl.lit(default, dtype=pl.Int64).alias(col))
        else:
            out = out.with_columns(pl.col(col).cast(pl.Int64).fill_null(default))

    out = (
        out.rename({"CODUFMUN": "munic_code"})
        .with_columns(
            pl.col("munic_code").cast(pl.Utf8).str.strip_chars().str.zfill(6),
            pl.lit(year, dtype=pl.Int32).alias("year"),
            pl.lit(month, dtype=pl.Int32).alias("month"),
        )
        .select(
            "munic_code",
            "year",
            "month",
            "n_estab",
            "n_hospitals",
            "beds_total",
            "beds_sus",
            "icu_beds",
            "icu_beds_sus",
            "intermediate_beds",
            "n_labs",
        )
        .sort("munic_code")
    )
    logger.info(
        "CNES capacity {}-{:02d}: {} municipalities, {} beds, {} ICU",
        year,
        month,
        out.height,
        int(out["beds_total"].sum()),
        int(out["icu_beds"].sum()),
    )
    return out


def resolve_inventory(
    years: Sequence[int],
    *,
    month: int = 12,
    groups: Sequence[str] = ("ST", "LT", "SR"),
    ufs: Sequence[str] | None = None,
    backend: Backend = "ftp",
    require_complete: bool = True,
) -> CnesInventory:
    """Resolve and preflight a complete annual CNES source inventory.

    The returned immutable object is both the retrieval plan and the
    provenance basis for a cache manifest. With ``require_complete=True``,
    failure occurs before any DBC is downloaded.
    """
    requested_years = sorted({int(year) for year in years})
    if not requested_years:
        raise ValueError("years must not be empty")
    if not 1 <= int(month) <= 12:
        raise ValueError("month must be between 1 and 12")
    wanted_ufs = [uf.upper() for uf in (ufs or UFS)]
    requested_groups = [_check_group(group) for group in groups]
    listings = {
        group: list_available(group, ufs=wanted_ufs, backend=backend)
        for group in requested_groups
    }
    available = {
        group: {(item.uf, item.year, item.month) for item in listing}
        for group, listing in listings.items()
    }
    missing = [
        (group, uf, year, int(month))
        for group in requested_groups
        for uf in wanted_ufs
        for year in requested_years
        if (uf, year, int(month)) not in available[group]
    ]
    if missing and require_complete:
        preview = ", ".join(
            f"{group}-{uf}-{year}{month:02d}"
            for group, uf, year, month in missing[:20]
        )
        raise FileNotFoundError(
            f"CNES series is missing {len(missing)} requested files: {preview}"
        )
    selected = tuple(
        item
        for group in requested_groups
        for item in listings[group]
        if item.uf in wanted_ufs
        and item.year in requested_years
        and item.month == int(month)
    )
    return CnesInventory(
        groups=tuple(requested_groups),
        ufs=tuple(wanted_ufs),
        years=tuple(requested_years),
        month=int(month),
        files=selected,
    )


def capacity_series(
    years: Sequence[int],
    *,
    month: int = 12,
    groups: Sequence[str] = ("ST", "LT", "SR"),
    ufs: Sequence[str] | None = None,
    backend: Backend = "ftp",
    refresh: bool = False,
    require_complete: bool = True,
    workers: int = 6,
    inventory: CnesInventory | None = None,
) -> pl.DataFrame:
    """Retrieve annual CNES stocks from one preflighted source inventory."""
    source_inventory = inventory or resolve_inventory(
        years,
        month=month,
        groups=groups,
        ufs=ufs,
        backend=backend,
        require_complete=require_complete,
    )
    requested_years = sorted({int(year) for year in years})
    wanted_ufs = [uf.upper() for uf in (ufs or source_inventory.ufs)]
    requested_groups = [_check_group(group) for group in groups]
    contract = (
        tuple(requested_groups),
        tuple(wanted_ufs),
        tuple(requested_years),
        int(month),
    )
    inventory_contract = (
        source_inventory.groups,
        source_inventory.ufs,
        source_inventory.years,
        source_inventory.month,
    )
    if contract != inventory_contract:
        raise ValueError(
            "CNES inventory contract differs from requested groups/UFs/years/month"
        )
    listings = source_inventory.by_group
    blocks = [
        capacity_by_municipality(
            year,
            int(month),
            groups=requested_groups,
            ufs=wanted_ufs,
            backend=backend,
            refresh=refresh,
            skip_missing=not require_complete,
            listings=listings,
            workers=workers,
        )
        for year in requested_years
    ]
    return pl.concat(blocks, how="vertical_relaxed").sort(
        ["year", "munic_code"]
    )
