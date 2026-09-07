"""Logical-key cache with mandatory content hashes and provenance sidecars.

Every remote byte that enters an analysis passes through :func:`store` and
acquires a sidecar recording where it came from, when, and what it hashed to.
Downstream code may then assert that a frozen snapshot is intact
(:func:`verify_snapshot`) rather than trusting that a re-download reproduced it.

The payload path is chosen by a human-readable logical key, not by the digest,
so this is deliberately *not* a content-addressed object store.  SHA-256
provides identity and integrity; the key provides discoverability.  A future
object-store backend may deduplicate by digest without changing this contract.

This matters concretely for DATASUS: SINAN "FINAIS" files are silently
rewritten. Eight leptospirosis year-files that had been final for up to
seventeen years were republished on 2026-07-15. Without a pinned, hashed
snapshot, a rerun three months later will not reproduce the paper's counts,
and nothing in the toolchain will say so.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from brepi.config import PATHS

SIDECAR_SUFFIX = ".provenance.json"


@dataclass(frozen=True)
class Provenance:
    """What a cached artefact is and where it came from."""

    key: str
    source: str  # logical source id, e.g. "datasus.sinan"
    uri: str  # fully-qualified retrieval URI
    fetched_at: str  # ISO-8601, UTC
    bytes: int
    sha256: str
    #: Server-declared last-modification, when the protocol exposes one. For
    #: DATASUS this is the single most useful revision signal.
    remote_modified: str | None = None
    #: Free-form retrieval parameters (SIDRA query terms, FTP host, etc.).
    params: dict[str, Any] | None = None
    tool_version: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def cache_path(key: str) -> Path:
    """Resolve a cache key to a path, creating parent directories.

    Keys are slash-delimited logical paths (``"sinan/lept/LEPTBR24.dbc"``) and
    map directly onto the cache directory so a human can find things.
    """
    path = PATHS.cache / key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sidecar_path(key: str) -> Path:
    p = cache_path(key)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def read_provenance(key: str) -> Provenance | None:
    side = sidecar_path(key)
    if not side.exists():
        return None
    return Provenance(**json.loads(side.read_text(encoding="utf-8")))


def store(
    key: str,
    payload: bytes,
    *,
    source: str,
    uri: str,
    remote_modified: str | None = None,
    params: dict[str, Any] | None = None,
    tool_version: str | None = None,
) -> Provenance:
    """Write ``payload`` to the cache and emit its provenance sidecar."""
    path = cache_path(key)
    path.write_bytes(payload)
    prov = Provenance(
        key=key,
        source=source,
        uri=uri,
        fetched_at=_now(),
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        remote_modified=remote_modified,
        params=params,
        tool_version=tool_version,
    )
    sidecar_path(key).write_text(prov.to_json(), encoding="utf-8")
    return prov


def fetch(
    key: str,
    loader: Callable[[], tuple[bytes, dict[str, Any]]],
    *,
    source: str,
    uri: str,
    refresh: bool = False,
    tool_version: str | None = None,
) -> tuple[Path, Provenance]:
    """Return a cached artefact, retrieving it via ``loader`` on a miss.

    ``loader`` returns ``(payload, metadata)``; recognised metadata keys are
    ``remote_modified`` and ``params``. A cache hit never touches the network,
    which is the point: an analysis rerun against a frozen snapshot must be
    provably offline.
    """
    path = cache_path(key)
    prov = read_provenance(key)
    if path.exists() and prov is not None and not refresh:
        return path, prov

    payload, meta = loader()
    prov = store(
        key,
        payload,
        source=source,
        uri=uri,
        remote_modified=meta.get("remote_modified"),
        params=meta.get("params"),
        tool_version=tool_version,
    )
    return path, prov


def fetch_stream(
    key: str,
    loader: Callable[[Path], dict[str, Any]],
    *,
    source: str,
    uri: str,
    refresh: bool = False,
    tool_version: str | None = None,
) -> tuple[Path, Provenance]:
    """Retrieve a large artefact directly to an atomic cache file.

    ``loader`` receives a temporary path and must stream bytes into it, then
    return the same optional metadata accepted by :func:`fetch`.  Hashing is
    incremental from disk, so a multi-gigabyte raster or territorial mesh is
    never duplicated in RAM.  The destination becomes visible only after the
    loader succeeds.
    """
    path = cache_path(key)
    prov = read_provenance(key)
    if path.exists() and prov is not None and not refresh:
        return path, prov

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        meta = loader(tmp) or {}
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise ValueError(f"stream loader produced no bytes for cache key {key!r}")
        new_prov = Provenance(
            key=key,
            source=source,
            uri=uri,
            fetched_at=_now(),
            bytes=tmp.stat().st_size,
            sha256=sha256_file(tmp),
            remote_modified=meta.get("remote_modified"),
            params=meta.get("params"),
            tool_version=tool_version,
        )
        os.replace(tmp, path)
        sidecar_path(key).write_text(new_prov.to_json(), encoding="utf-8")
        return path, new_prov
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Snapshot manifests
# --------------------------------------------------------------------------


def write_manifest(name: str, keys: Iterable[str]) -> Path:
    """Freeze a named set of cache keys into a manifest.

    The manifest is the analytic dataset's identity. Archive it with the paper.
    """
    entries = []
    for key in sorted(keys):
        prov = read_provenance(key)
        if prov is None:
            raise FileNotFoundError(f"no provenance for cache key {key!r}")
        entries.append(asdict(prov))
    doc = {
        "manifest": name,
        "created_at": _now(),
        "n_artefacts": len(entries),
        "artefacts": entries,
    }
    out = PATHS.cache / f"_manifest_{name}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def verify_snapshot(name: str) -> list[dict[str, Any]]:
    """Re-hash every artefact in a manifest and report drift.

    Returns one record per artefact whose on-disk bytes no longer match the
    manifest, or which has gone missing. An empty list means the snapshot is
    intact. Run this before every modelling session.
    """
    path = PATHS.cache / f"_manifest_{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    drift: list[dict[str, Any]] = []
    for entry in doc["artefacts"]:
        p = cache_path(entry["key"])
        if not p.exists():
            drift.append({"key": entry["key"], "problem": "missing"})
            continue
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            drift.append(
                {
                    "key": entry["key"],
                    "problem": "hash_mismatch",
                    "expected": entry["sha256"],
                    "actual": actual,
                }
            )
    return drift


def diff_against_remote(
    name: str,
    lister: Callable[[], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compare a frozen manifest to the live remote listing.

    ``lister`` returns ``{uri: {"bytes": int, "remote_modified": str}}``. Use
    before submission: any artefact whose remote size or timestamp has moved
    since the snapshot is a candidate robustness check, and — for DATASUS —
    reporting that diff is itself a methodological contribution, since the
    mutability of "final" files is undocumented and widely ignored.
    """
    doc = json.loads((PATHS.cache / f"_manifest_{name}.json").read_text(encoding="utf-8"))
    remote = lister()
    out: list[dict[str, Any]] = []
    for entry in doc["artefacts"]:
        r = remote.get(entry["uri"])
        if r is None:
            out.append({"key": entry["key"], "problem": "gone_from_remote"})
            continue
        if r.get("bytes") != entry["bytes"] or (
            entry["remote_modified"] and r.get("remote_modified") != entry["remote_modified"]
        ):
            out.append(
                {
                    "key": entry["key"],
                    "problem": "revised_upstream",
                    "snapshot": {"bytes": entry["bytes"], "modified": entry["remote_modified"]},
                    "remote": r,
                }
            )
    return out


__all__ = [
    "Provenance",
    "cache_path",
    "sidecar_path",
    "read_provenance",
    "store",
    "fetch",
    "fetch_stream",
    "write_manifest",
    "verify_snapshot",
    "diff_against_remote",
    "sha256_bytes",
    "sha256_file",
]
