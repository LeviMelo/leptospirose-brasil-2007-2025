"""Validated municipality-to-health-region crosswalks.

Brazil's ``Região de Saúde`` is a SUS administrative geography, revised by
state CIB/CIR resolutions rather than by IBGE.  It must therefore never be
derived from municipality names, centroid overlays, or a stale hard-coded
lookup.  This module turns a dated official extract into the single mapping
accepted by the multiscale analysis and rejects incomplete versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import httpx
import polars as pl

from brepi.config import OPEN_DATASUS_API_BASE
from brepi.geo import lattice
from brepi.io import cache

TOOL_VERSION = "brepi.geo.health_regions/2"
OPEN_DATASUS_ENDPOINT = (
    f"{OPEN_DATASUS_API_BASE}/macrorregiao-e-regiao-de-saude/municipio"
)


class HealthRegionError(ValueError):
    """An official health-region extract cannot support a complete mapping."""


@dataclass(frozen=True)
class HealthRegionReport:
    """Coverage evidence returned with a normalised mapping."""

    source_path: str
    vintage: str
    n_rows_input: int
    n_rows_output: int
    n_municipalities: int
    missing_municipalities: tuple[str, ...]


_MUNICIPAL_COLUMNS = (
    "code7", "codigo_municipio", "cod_municipio", "co_municipio", "cod_mun", "co_mun",
    "cod_ibge", "codigo_ibge", "ibge", "municipio_codigo",
)
_REGION_COLUMNS = (
    "health_region", "co_regiao_saude", "cod_regiao_saude",
    "codigo_regiao_saude", "id_regiao_saude", "regiao_saude", "regiao_de_saude",
)
_REGION_NAME_COLUMNS = (
    "health_region_name", "nome_regiao_saude", "nm_regiao_saude",
    "regiao_saude_nome", "nome_regiao", "regiao_saude",
)


def _normalise_header(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def _first(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {_normalise_header(c): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def _read(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".json":
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document.get("macrorregiao_regiao_saude_municipios", document)
        if not isinstance(rows, list):
            raise HealthRegionError("health-region JSON does not contain a row array")
        return pl.DataFrame(rows)
    if path.suffix.lower() not in {".csv", ".txt"}:
        raise HealthRegionError("health-region extract must be CSV, TXT, JSON, or Parquet")
    head = path.read_bytes()[:4096].decode("utf-8-sig", errors="replace")
    separator = ";" if head.count(";") > head.count(",") else ","
    return pl.read_csv(
        path, separator=separator, infer_schema_length=0, truncate_ragged_lines=True
    )


def fetch_open_datasus(
    *,
    snapshot_date: date,
    refresh: bool = False,
    page_size: int = 860,
) -> tuple[Path, cache.Provenance]:
    """Freeze the official OpenDataSUS municipality/health-region membership.

    The endpoint exposes retrieval time but no ``valid_from`` date for the
    administrative membership. The snapshot date is therefore bitemporal
    *publication/retrieval* metadata, not a claim that all memberships became
    valid on that day.
    """
    if page_size < 1 or page_size > 860:
        raise ValueError("OpenDataSUS page_size must be in 1..860")
    key = f"geo/health_regions/open_datasus_{snapshot_date:%Y%m%d}.json"

    def loader() -> tuple[bytes, dict[str, object]]:
        import json

        rows: list[dict[str, object]] = []
        remote_modified: str | None = None
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            for page in range(100):
                response = client.get(
                    OPEN_DATASUS_ENDPOINT,
                    # Despite calling offset a page number in its Swagger
                    # description, the live API interprets it as a zero-based
                    # row offset (verified 2026-07-30).
                    params={"limit": page_size, "offset": page * page_size},
                )
                response.raise_for_status()
                remote_modified = response.headers.get("last-modified") or remote_modified
                document = response.json()
                batch = document.get("macrorregiao_regiao_saude_municipios", [])
                if not isinstance(batch, list):
                    raise HealthRegionError(
                        "OpenDataSUS response has no municipality row array"
                    )
                rows.extend(batch)
                if len(batch) < page_size:
                    break
            else:
                raise HealthRegionError("OpenDataSUS pagination exceeded 100 pages")
        codes = [str(row.get("codigo_municipio", "")) for row in rows]
        if len(codes) != len(set(codes)):
            raise HealthRegionError(
                "OpenDataSUS pagination returned duplicate municipality codes"
            )
        payload = json.dumps(
            {"macrorregiao_regiao_saude_municipios": rows},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return payload, {
            "remote_modified": remote_modified,
            "params": {
                "snapshot_date": snapshot_date.isoformat(),
                "page_size": page_size,
                "offset_semantics": "row offset (live API; Swagger says page)",
                "pages": page + 1,
                "rows": len(rows),
                "valid_time_disclosed": False,
            },
        }

    return cache.fetch(
        key,
        loader,
        source="opendatasus.health_regions",
        uri=OPEN_DATASUS_ENDPOINT,
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )


def load_official_crosswalk(
    path: str | Path,
    *,
    vintage: str,
    lattice_year: int = lattice.DEFAULT_YEAR,
    require_complete: bool = True,
) -> tuple[pl.DataFrame, HealthRegionReport]:
    """Normalise a dated official municipal health-region extract.

    The input must identify municipality and region. Municipality values may be
    DATASUS six-digit or IBGE seven-digit codes; names are deliberately not
    accepted as keys.  The returned frame has exactly ``code7``,
    ``health_region`` and ``health_region_name`` and is suitable for
    :func:`brepi.geo.lattice.attach_health_regions`.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not vintage.strip():
        raise HealthRegionError("vintage is required (e.g. 'DATASUS 2026-07')")

    raw = _read(source)
    mun = _first(raw.columns, _MUNICIPAL_COLUMNS)
    region = _first(raw.columns, _REGION_COLUMNS)
    name = _first(raw.columns, _REGION_NAME_COLUMNS)
    if mun is None or region is None:
        raise HealthRegionError(
            "could not identify municipality and health-region columns; "
            f"found {raw.columns}"
        )

    code = pl.col(mun).cast(pl.Utf8).str.strip_chars().str.replace_all(r"\\.0$", "")
    width = code.str.len_chars()
    out = raw.with_columns(
        pl.when(width == 6).then(lattice.code6_to_code7_expr(code))
        .when(width == 7).then(code)
        .otherwise(None)
        .alias("code7"),
        pl.col(region).cast(pl.Utf8).str.strip_chars().alias("health_region"),
        (pl.col(name).cast(pl.Utf8).str.strip_chars() if name else pl.lit(None, dtype=pl.Utf8))
        .alias("health_region_name"),
    ).select("code7", "health_region", "health_region_name")

    out = out.filter(pl.col("code7").is_not_null() & pl.col("health_region").is_not_null())
    if out.height != out.select("code7").unique().height:
        duplicated = out.group_by("code7").len().filter(pl.col("len") > 1)
        raise HealthRegionError(
            f"crosswalk assigns {duplicated.height} municipality code(s) more than once"
        )

    target = lattice.load_municipalities(lattice_year).select("code7")
    extras = out.join(target, on="code7", how="anti")
    if extras.height:
        raise HealthRegionError(
            f"crosswalk has {extras.height} code(s) outside the {lattice_year} lattice"
        )
    missing = target.join(out.select("code7"), on="code7", how="anti")["code7"].to_list()
    if require_complete and missing:
        raise HealthRegionError(
            f"crosswalk covers {out.height}/{target.height} municipalities; "
            f"{len(missing)} missing (e.g. {missing[:5]})"
        )
    out = out.sort("code7")
    return out, HealthRegionReport(
        source_path=str(source.resolve()), vintage=vintage,
        n_rows_input=raw.height, n_rows_output=out.height,
        n_municipalities=out["code7"].n_unique(),
        missing_municipalities=tuple(missing),
    )


__all__ = [
    "HealthRegionError",
    "HealthRegionReport",
    "fetch_open_datasus",
    "load_official_crosswalk",
]
