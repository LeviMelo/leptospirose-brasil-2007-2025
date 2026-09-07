"""Municipal population denominators, as redistributed by DATASUS.

IBGE publishes the estimates; DATASUS republishes them on the FTP in the
shape its own tabulators consume. Two families, and they are not
interchangeable.

``/IBGE/POPSVS`` -- ``POPSBR{YY}.zip``
    Municipality x sex x single-year-of-age, 2000-2025. Ages run ``000`` to
    ``080``, where ``080`` is 80-and-over, not "exactly 80". ``SEXO`` is
    ``1``=male, ``2``=female. ``COD_MUN`` is the **7-digit** IBGE code
    including the check digit. This is the family to use: it is current, and
    it is the only one that supports age- and sex-standardised rates.

``/IBGE/POPTCU`` -- ``POPTBR{YY}.zip``
    Municipal totals only, the TCU series used for fund transfers. The FTP
    copy stops around 2014 even though the archive names suggest a longer
    run; treat it as legacy and use it only to reproduce an older analysis
    that was built on it. ``MUNIC_RES`` here is the **6-digit** code.

Traps.

*Code width.* POPSVS carries 7 digits, POPTCU and every DATASUS event system
(SIH ``MUNIC_RES``, SIM ``CODMUNRES``, SINAN ``ID_MN_RESI``) carry 6. Joining
a 7-digit denominator to a 6-digit numerator silently yields zero matches and
therefore infinite rates. :func:`municipal_population` always returns the
6-digit form.

*Archive contents vary.* The member names are inconsistent across years --
``POP24.dbf`` in one year, ``pop00.dbf`` in another, and POPTCU ships
``.DBF``, ``.csv`` and ``.xml`` copies of the same table. Nothing here
hardcodes a member name; the archive is inspected at runtime and what was
found is logged.

*Denominator drift.* Estimates for a given year are revised between vintages,
and the municipality set itself changes (splits, mergers). A rate series
built from mixed vintages moves for reasons that have nothing to do with
disease.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import polars as pl
from loguru import logger

from brepi.config import DATASUS_ENCODING
from brepi.io import cache
from brepi.io.cache import Provenance
from brepi.io.dbc import read_dbc
from brepi.io.datasus_ftp import Backend, RemoteFile, get, resolve

__all__ = [
    "Family",
    "PopFile",
    "POPSVS_DIR",
    "POPTCU_DIR",
    "AGE_GROUP_EDGES",
    "list_available",
    "fetch_year",
    "municipal_population",
    "SOURCE_ID",
]

SOURCE_ID = "datasus.ibge_pop"

POPSVS_DIR = "/IBGE/POPSVS"
POPTCU_DIR = "/IBGE/POPTCU"

Family = Literal["popsvs", "poptcu"]

_FAMILY = {
    "popsvs": (POPSVS_DIR, "POPSBR"),
    "poptcu": (POPTCU_DIR, "POPTBR"),
}

#: Lower bounds of the standard five-year groups; the last one is open-ended.
AGE_GROUP_EDGES: tuple[int, ...] = (
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
)

_SEX_LABEL = {"1": "male", "2": "female", "M": "male", "F": "female"}


@dataclass(frozen=True)
class PopFile(RemoteFile):
    """One population archive with its parsed reference year."""

    year: int = 0
    family: Family = "popsvs"


def _year_from_yy(yy: str) -> int:
    """Window a two-digit population year. The series starts in 1980."""
    n = int(yy)
    return 2000 + n if n <= 40 else 1900 + n


def _cache_key(family: Family, filename: str) -> str:
    return f"datasus/ibge_pop/{family}/{filename.upper()}"


def list_available(
    *, family: Family = "popsvs", backend: Backend = "ftp"
) -> list[PopFile]:
    """List population archives for one family, sorted by year."""
    if family not in _FAMILY:
        raise KeyError(f"unknown family {family!r}; known: {', '.join(_FAMILY)}")
    directory, prefix = _FAMILY[family]
    out: list[PopFile] = []
    for rf in resolve(directory, rf"{prefix}(\d{{2}})\.zip$", backend=backend):
        yy = rf.name[len(prefix) : len(prefix) + 2]
        if not yy.isdigit():
            continue
        out.append(
            PopFile(
                path=rf.path,
                name=rf.name,
                bytes=rf.bytes,
                modified=rf.modified,
                year=_year_from_yy(yy),
                family=family,
            )
        )
    out.sort(key=lambda f: f.year)
    return out


def _read_member(archive: zipfile.ZipFile, name: str, tmpdir: Path) -> pl.DataFrame:
    """Decode one archive member, dBase or CSV, all columns as Utf8."""
    payload = archive.read(name)
    lower = name.lower()
    if lower.endswith((".dbf", ".dbc")):
        target = tmpdir / Path(name).name
        target.write_bytes(payload)
        return read_dbc(target)
    if lower.endswith(".csv"):
        text = payload.decode(DATASUS_ENCODING, errors="replace")
        head = text.split("\n", 1)[0]
        sep = ";" if head.count(";") >= head.count(",") else ","
        frame = pl.read_csv(
            text.encode("utf-8"),
            separator=sep,
            infer_schema_length=0,
            truncate_ragged_lines=True,
        )
        return frame.rename({c: c.strip() for c in frame.columns})
    raise ValueError(f"unsupported archive member type: {name}")


def fetch_year(
    year: int,
    *,
    family: Family = "popsvs",
    backend: Backend = "ftp",
    refresh: bool = False,
    listing: Sequence[PopFile] | None = None,
) -> tuple[pl.DataFrame, Provenance]:
    """Fetch one population year and decode the table inside its ZIP.

    The archive's member list is discovered at runtime and logged; a ``.dbf``
    member is preferred over a ``.csv`` of the same table because the CSV
    rendering in POPTCU quotes inconsistently across years.
    """
    files = list(listing) if listing is not None else list_available(
        family=family, backend=backend
    )
    matches = [f for f in files if f.year == year]
    if not matches:
        raise FileNotFoundError(f"no {family} population archive for {year}")
    target = matches[0]
    key = _cache_key(family, target.name)
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
                "family": family,
                "year": year,
                "remote_path": target.path,
            },
        }

    path, prov = cache.fetch(key, _loader, source=SOURCE_ID, uri=uri, refresh=refresh)
    archive = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    members = [n for n in archive.namelist() if not n.endswith("/")]
    logger.info("{} {}: archive members {}", target.name, year, members)
    if not members:
        raise ValueError(f"{target.name} is an empty archive")
    preferred = [n for n in members if n.lower().endswith((".dbf", ".dbc"))]
    chosen = preferred[0] if preferred else members[0]
    logger.info("{}: decoding member {}", target.name, chosen)
    with tempfile.TemporaryDirectory(prefix="brepi_pop_") as tmp:
        frame = _read_member(archive, chosen, Path(tmp))
    return frame, prov


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand in upper:
            return upper[cand]
    return None


def _age_group_expr(col: str) -> pl.Expr:
    """Map single-year age to a five-year label, ``80+`` open-ended."""
    age = pl.col(col).cast(pl.Int32, strict=False)
    expr = pl.when(age.is_null()).then(pl.lit(None, dtype=pl.Utf8))
    for lo in AGE_GROUP_EDGES[:-1]:
        expr = expr.when(age < lo + 5).then(pl.lit(f"{lo:02d}-{lo + 4:02d}"))
    return expr.otherwise(pl.lit("80+")).alias("age_group")


def municipal_population(
    years: Iterable[int],
    by: tuple[str, ...] | None = ("sex", "age"),
    *,
    family: Family = "popsvs",
    backend: Backend = "ftp",
    refresh: bool = False,
) -> pl.DataFrame:
    """Assemble a tidy denominator table over several years.

    Returns ``munic_code`` (6-digit, check digit dropped), ``year``, ``sex``,
    ``age_group``, ``population``.

    ``by`` selects the stratification: ``("sex", "age")`` keeps both,
    ``("sex",)`` or ``("age",)`` marginalises the other, and ``None``
    collapses to the municipal total. Marginalised dimensions carry the
    literal ``"all"`` rather than null, so the key stays complete and a join
    against a stratified numerator fails loudly instead of fanning out.

    ``population`` is summed as Int64. POPTCU has no sex or age dimension at
    all; requesting one from it raises.
    """
    wanted = tuple(b.lower() for b in by) if by else ()
    unknown = set(wanted) - {"sex", "age"}
    if unknown:
        raise ValueError(f"unsupported stratification {sorted(unknown)}")
    if family == "poptcu" and wanted:
        raise ValueError("POPTCU carries municipal totals only; use by=None or family='popsvs'")

    listing = list_available(family=family, backend=backend)
    parts: list[pl.DataFrame] = []
    for year in sorted(set(years)):
        raw, _prov = fetch_year(
            year, family=family, backend=backend, refresh=refresh, listing=listing
        )
        cols = raw.columns
        mun = _first_present(cols, ("COD_MUN", "MUNIC_RES", "CODMUN", "MUNICIPIO"))
        pop = _first_present(cols, ("POP", "POPULACAO", "POPULACAO_TOTAL"))
        if mun is None or pop is None:
            raise KeyError(
                f"population {family} {year}: expected municipality and population "
                f"columns, found {cols}"
            )
        sex_col = _first_present(cols, ("SEXO",))
        age_col = _first_present(cols, ("IDADE", "FXETARIA", "FAIXA_ETARIA"))

        frame = raw.with_columns(
            pl.col(mun).cast(pl.Utf8).str.strip_chars().str.slice(0, 6).alias("munic_code"),
            pl.lit(year, dtype=pl.Int32).alias("year"),
            pl.col(pop).cast(pl.Int64, strict=False).fill_null(0).alias("population"),
        )
        if "sex" in wanted:
            if sex_col is None:
                raise KeyError(f"population {family} {year} has no SEXO column")
            frame = frame.with_columns(
                pl.col(sex_col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .replace_strict(_SEX_LABEL, default="unknown")
                .alias("sex")
            )
        else:
            frame = frame.with_columns(pl.lit("all", dtype=pl.Utf8).alias("sex"))
        if "age" in wanted:
            if age_col is None:
                raise KeyError(f"population {family} {year} has no age column")
            frame = frame.with_columns(_age_group_expr(age_col))
        else:
            frame = frame.with_columns(pl.lit("all", dtype=pl.Utf8).alias("age_group"))

        parts.append(
            frame.group_by(["munic_code", "year", "sex", "age_group"])
            .agg(pl.col("population").sum())
        )

    if not parts:
        raise ValueError("no years requested")
    out = pl.concat(parts, how="vertical").sort(
        ["year", "munic_code", "sex", "age_group"]
    )
    logger.info(
        "population {} {}: {} rows, {} municipalities, total {}",
        family,
        sorted({int(y) for y in out["year"].unique().to_list()}),
        out.height,
        out["munic_code"].n_unique(),
        int(out["population"].sum()),
    )
    return out
