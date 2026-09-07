"""Versioned official IBGE territorial meshes.

Geometry is an analytic input, not decoration.  This adapter pins the vintage,
streams the official archive into the provenance cache, validates it against
the declared municipality lattice, and materialises the small stable schema
consumed by R.  It never substitutes the current mesh for a requested year.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from brepi.config import (
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    IBGE_MUNICIPAL_MESH_BASE,
    PATHS,
)
from brepi.geo.lattice import load_municipalities
from brepi.io.cache import Provenance, fetch_stream

SOURCE_ID = "ibge.malha_municipal"
TOOL_VERSION = "brepi.geo.meshes/1"


class MeshError(RuntimeError):
    """A mesh archive or geometry violates its declared contract."""


@dataclass(frozen=True)
class MeshReport:
    year: int
    source_rows: int
    lattice_rows: int
    output_rows: int
    excluded_non_lattice: tuple[str, ...]
    invalid_before_repair: int
    invalid_after_repair: int
    crs: str
    source_sha256: str
    output: str


def municipality_mesh_url(year: int) -> str:
    """Official unified-Brazil municipal mesh archive for ``year``."""
    if not 2000 <= int(year) <= 2100:
        raise ValueError(f"implausible mesh year {year!r}")
    filename = f"BR_Municipios_{int(year)}.zip"
    return (
        f"{IBGE_MUNICIPAL_MESH_BASE}/municipio_{int(year)}/"
        f"Brasil/BR/{filename}"
    )


def _download(url: str, destination: Path) -> dict[str, Any]:
    import time

    last: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.stream(
                "GET", url, timeout=HTTP_TIMEOUT, follow_redirects=True
            ) as response:
                if response.status_code == 200:
                    with destination.open("wb") as fh:
                        for block in response.iter_bytes(chunk_size=1 << 20):
                            fh.write(block)
                    return {
                        "remote_modified": response.headers.get("last-modified"),
                        "params": {
                            "attempt": attempt,
                            "content_length": response.headers.get("content-length"),
                            "resolved_url": str(response.url),
                        },
                    }
                if response.status_code < 500 and response.status_code != 429:
                    raise MeshError(f"{url} returned HTTP {response.status_code}")
                last = MeshError(f"{url} returned HTTP {response.status_code}")
        except httpx.HTTPError as exc:
            last = exc
        if attempt < HTTP_MAX_RETRIES:
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise MeshError(f"giving up on {url}") from last


def fetch_municipality_mesh(
    year: int = 2022, *, refresh: bool = False
) -> tuple[Path, Provenance]:
    """Fetch a pinned official municipal mesh archive by streaming to disk."""
    url = municipality_mesh_url(year)
    key = f"ibge/meshes/{year}/BR_Municipios_{year}.zip"
    return fetch_stream(
        key,
        lambda destination: _download(url, destination),
        source=SOURCE_ID,
        uri=url,
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a trusted-format ZIP while refusing path traversal."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise MeshError(f"unsafe path in mesh archive: {member.filename!r}")
        zf.extractall(destination)


def materialize_municipality_mesh(
    year: int = 2022,
    *,
    output: Path | None = None,
    refresh: bool = False,
) -> tuple[Path, MeshReport]:
    """Validate and write the municipality lattice as a GeoPackage.

    The output contains exactly the municipality codes in
    :func:`load_municipalities(year)`.  Operational water polygons or any other
    official-mesh extras are reported and excluded; missing lattice units,
    duplicate codes and invalid geometries after repair are blocking errors.
    """
    import geopandas as gpd

    archive, provenance = fetch_municipality_mesh(year, refresh=refresh)
    output = output or (PATHS.panel / "geo_municipality.gpkg")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="brepi-mesh-", dir=PATHS.interim) as tmp_name:
        tmp = Path(tmp_name)
        _safe_extract(archive, tmp)
        candidates = sorted(tmp.rglob("*.shp"))
        if len(candidates) != 1:
            raise MeshError(
                f"expected one shapefile in {archive.name}, found {len(candidates)}"
            )
        gdf = gpd.read_file(candidates[0])

    source_rows = len(gdf)
    code_source = next(
        (name for name in ("CD_MUN", "CD_GEOCMU", "CD_MUNICIP") if name in gdf.columns),
        None,
    )
    if code_source is None:
        raise MeshError(f"mesh has no recognised municipality code: {list(gdf.columns)}")
    name_source = next(
        (name for name in ("NM_MUN", "NM_MUNICIP") if name in gdf.columns),
        None,
    )
    gdf["munic_code"] = gdf[code_source].astype("string").str.strip().str.zfill(7)
    if gdf["munic_code"].duplicated().any():
        dup = sorted(gdf.loc[gdf["munic_code"].duplicated(False), "munic_code"].unique())
        raise MeshError(f"duplicate municipality geometries: {dup[:10]}")

    lattice = load_municipalities(year)
    expected = set(lattice["code7"].to_list())
    present = set(gdf["munic_code"])
    missing = sorted(expected - present)
    if missing:
        raise MeshError(f"official mesh is missing {len(missing)} lattice units: {missing[:10]}")
    excluded = tuple(sorted(present - expected))
    gdf = gdf[gdf["munic_code"].isin(expected)].copy()

    invalid_before = int((~gdf.geometry.is_valid).sum())
    if invalid_before:
        gdf.geometry = gdf.geometry.make_valid()
    invalid_after = int((~gdf.geometry.is_valid).sum())
    if invalid_after:
        raise MeshError(f"{invalid_after} geometries remain invalid after repair")
    if gdf.geometry.is_empty.any() or gdf.geometry.isna().any():
        raise MeshError("mesh contains empty/null municipality geometry")

    columns = ["munic_code"]
    if name_source is not None:
        gdf["municipality_name"] = gdf[name_source].astype("string")
        columns.append("municipality_name")
    if "SIGLA_UF" in gdf.columns:
        gdf["uf_abbr"] = gdf["SIGLA_UF"].astype("string")
        columns.append("uf_abbr")
    if "AREA_KM2" in gdf.columns:
        gdf["area_km2"] = gdf["AREA_KM2"]
        columns.append("area_km2")
    columns.append(gdf.geometry.name)
    gdf = gdf[columns].sort_values("munic_code")

    fd, tmp_output_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".gpkg", dir=output.parent
    )
    os.close(fd)
    tmp_output = Path(tmp_output_name)
    tmp_output.unlink(missing_ok=True)
    try:
        gdf.to_file(tmp_output, layer="municipality", driver="GPKG")
        os.replace(tmp_output, output)
    finally:
        tmp_output.unlink(missing_ok=True)

    report = MeshReport(
        year=year,
        source_rows=source_rows,
        lattice_rows=len(expected),
        output_rows=len(gdf),
        excluded_non_lattice=excluded,
        invalid_before_repair=invalid_before,
        invalid_after_repair=invalid_after,
        crs=str(gdf.crs),
        source_sha256=provenance.sha256,
        output=str(output),
    )
    report_path = PATHS.reports / f"municipality_mesh_{year}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, report


__all__ = [
    "MeshError",
    "MeshReport",
    "municipality_mesh_url",
    "fetch_municipality_mesh",
    "materialize_municipality_mesh",
]
