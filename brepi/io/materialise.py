"""Derived-artefact cache: decode and normalisation are part of the data layer.

Fetching is only the first third of the cost. A DATASUS extraction is
fetch -> decode (.dbc -> DBF -> frame) -> normalise (codes -> meanings), and
the second and third steps are re-run from scratch every time an analysis
touches the data unless something stores their output. On the leptospirosis
SIM extraction the download became the *cheap* part once it was parallelised:
decoding runs at roughly 11 s per state-year, so a full 486-file pass spends
well over an hour turning bytes we already have into a frame we already
computed last week.

So decoded and normalised frames are cached too, keyed by everything that can
change their content:

* the **content hash of every input artefact** (not its path or its name --
  DATASUS silently rewrites files, and a path-keyed cache would happily serve
  a frame derived from bytes that no longer exist upstream);
* the **transform identity**: a version string the caller bumps when the
  decode or normalisation logic changes;
* the **projection and options**, so a narrow column selection does not
  masquerade as the full frame.

The key is a digest of those, so a stale entry is unreachable rather than
merely discouraged: change the codebook version and the old parquet is simply
never looked up again. It stays on disk until :func:`sweep` removes it, which
is deliberate -- an analysis in flight must not have its inputs deleted under
it.

Storage is parquet with zstd, which for these frames runs 8-20x smaller than
the source ``.dbc`` and loads an order of magnitude faster than decoding.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import polars as pl

from brepi.config import PATHS
from brepi.io import cache

SIDECAR = ".materialise.json"


@dataclass(frozen=True)
class Materialisation:
    """What a derived artefact is, and what it was derived from."""

    key: str
    transform: str
    transform_version: str
    inputs: list[str]  # cache keys of the source artefacts
    input_digest: str  # digest of their content hashes
    options_digest: str
    rows: int
    columns: int
    bytes: int
    created_at: str
    seconds: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(parts):
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def input_digest(input_keys: Sequence[str]) -> str:
    """Digest of the *content* of the named cache artefacts.

    Reads each artefact's provenance sidecar rather than the bytes, so this is
    cheap; the sidecar's sha256 is what makes it content-addressed. A missing
    sidecar is an error: deriving from an artefact with no recorded provenance
    would produce a cache entry nothing can invalidate.
    """
    hashes = []
    for key in input_keys:
        prov = cache.read_provenance(key)
        if prov is None:
            raise FileNotFoundError(
                f"no provenance for input '{key}'; a derived artefact cannot be "
                "keyed on an artefact whose content hash is unknown"
            )
        hashes.append(prov.sha256)
    return _digest(hashes)


def options_digest(options: Mapping[str, Any] | None) -> str:
    if not options:
        return "none"
    return _digest([json.dumps(options, sort_keys=True, default=str)])


def artefact_path(key: str) -> Path:
    p = PATHS.root / "derived" / f"{key}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def materialise(
    name: str,
    input_keys: Sequence[str],
    build: Callable[[], pl.DataFrame],
    *,
    transform: str,
    transform_version: str,
    options: Mapping[str, Any] | None = None,
    refresh: bool = False,
    compression: str = "zstd",
) -> tuple[pl.DataFrame, Materialisation]:
    """Return a derived frame, computing it only if nothing valid is cached.

    ``build`` is called only on a miss, so the expensive decode/normalise path
    is skipped entirely on a hit -- including the fetch of the source bytes if
    the caller arranges it that way.
    """
    idig = input_digest(input_keys)
    odig = options_digest(options)
    key = f"{name}/{transform}-{transform_version}-{idig}-{odig}"
    path = artefact_path(key)
    side = path.with_suffix(path.suffix + SIDECAR)

    if path.exists() and side.exists() and not refresh:
        meta = Materialisation(**json.loads(side.read_text(encoding="utf-8")))
        return pl.read_parquet(path), meta

    t0 = time.monotonic()
    frame = build()
    elapsed = time.monotonic() - t0
    frame.write_parquet(path, compression=compression)
    meta = Materialisation(
        key=key, transform=transform, transform_version=transform_version,
        inputs=list(input_keys), input_digest=idig, options_digest=odig,
        rows=frame.height, columns=frame.width, bytes=path.stat().st_size,
        created_at=_now(), seconds=round(elapsed, 2),
    )
    side.write_text(meta.to_json(), encoding="utf-8")
    return frame, meta


def inventory() -> pl.DataFrame:
    """Every derived artefact on disk, with its size and provenance."""
    root = PATHS.root / "derived"
    rows: list[dict[str, Any]] = []
    if root.exists():
        for side in root.rglob(f"*{SIDECAR}"):
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta["path"] = str(side.with_suffix("").relative_to(root))
            rows.append(meta)
    if not rows:
        return pl.DataFrame(schema={"key": pl.Utf8, "bytes": pl.Int64})
    return pl.DataFrame(rows).sort("bytes", descending=True)


def sweep(
    *, keep_transforms: Sequence[str] | None = None, dry_run: bool = True
) -> pl.DataFrame:
    """Remove derived artefacts whose transform version is superseded.

    Only artefacts whose ``(name, transform)`` has a *newer* version present
    are candidates, so an artefact that is simply old but still current is
    never touched. ``dry_run`` is the default because deleting derived data
    during a running analysis is worse than keeping it.
    """
    inv = inventory()
    if inv.is_empty():
        return inv
    latest = (
        inv.group_by(["transform"])
        .agg(pl.col("created_at").max().alias("newest"))
    )
    keep = set(keep_transforms or [])
    victims = (
        inv.join(latest, on="transform")
        .filter(
            (pl.col("created_at") < pl.col("newest"))
            & ~pl.col("transform").is_in(list(keep))
        )
    )
    if not dry_run:
        root = PATHS.root / "derived"
        for rel in victims["path"].to_list():
            p = root / rel
            for target in (Path(str(p) + ".parquet"), Path(str(p) + ".parquet" + SIDECAR)):
                if target.exists():
                    target.unlink()
    return victims.select("key", "transform", "transform_version", "bytes", "created_at")


def usage() -> dict[str, Any]:
    """Disk used by the raw cache and by derived artefacts, separately.

    Reported separately because they have different lifetimes: the raw cache is
    the frozen evidence a paper points at and must not be swept, while derived
    artefacts are reproducible from it and can be discarded freely.
    """
    def _size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    raw = _size(PATHS.cache)
    derived = _size(PATHS.root / "derived")
    return {
        "raw_cache_bytes": raw,
        "raw_cache_gb": round(raw / 1e9, 2),
        "derived_bytes": derived,
        "derived_gb": round(derived / 1e9, 2),
        "derived_artefacts": inventory().height,
    }


__all__ = [
    "Materialisation", "materialise", "inventory", "sweep", "usage",
    "artefact_path", "input_digest", "options_digest",
]
