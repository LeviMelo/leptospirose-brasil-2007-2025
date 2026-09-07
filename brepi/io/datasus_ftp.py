"""Transport layer for the DATASUS public tree.

Two interchangeable backends serve the same logical paths:

``ftp``
    ``ftp.datasus.gov.br`` over plain FTP. Canonical and always current, but
    egress is geo-restricted to Brazil. From elsewhere the connection is
    *refused* at TCP rather than denied at the application layer, so a failure
    looks like an outage. Verified reachable from Brazil on 2026-07-30.

``mirror``
    An unauthenticated S3 bucket rclone-synced from the same tree daily,
    reachable worldwide. Its object keys drop the ``/dissemin/publicos``
    prefix. Lags the FTP by up to a day.

Both expose size and modification time, which is what makes revision
detection possible.
"""

from __future__ import annotations

import ftplib
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

import httpx

from brepi.config import (
    DATASUS_FTP_HOST,
    DATASUS_FTP_ROOT,
    DATASUS_MIRROR_BASE,
    FTP_TIMEOUT,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
)

Backend = Literal["ftp", "mirror"]

_LIST_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}[AP]M)\s+(?P<size>\d+)\s+(?P<name>.+?)\s*$"
)


@dataclass(frozen=True)
class RemoteFile:
    """One file in the DATASUS tree, as the server describes it."""

    path: str  # logical path under /dissemin/publicos, leading slash
    name: str
    bytes: int
    modified: str | None  # ISO-8601

    @property
    def uri_ftp(self) -> str:
        return f"ftp://{DATASUS_FTP_HOST}{DATASUS_FTP_ROOT}{self.path}"

    @property
    def uri_mirror(self) -> str:
        return f"{DATASUS_MIRROR_BASE}{self.path}"


def _parse_iis_datetime(date: str, tm: str) -> str | None:
    """Parse the Microsoft FTP ``MM-DD-YY hh:mmAM`` listing stamp.

    The two-digit year is windowed at 1990 because the tree contains genuine
    1992 files; a naive ``%y`` would place them in 2092.
    """
    try:
        dt = datetime.strptime(f"{date} {tm}", "%m-%d-%y %I:%M%p")
    except ValueError:
        return None
    if dt.year > datetime.now().year + 1:
        dt = dt.replace(year=dt.year - 100)
    return dt.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# FTP backend
# --------------------------------------------------------------------------


class _FTP:
    """Lazily-opened, reused anonymous FTP connection."""

    def __init__(self) -> None:
        self._conn: ftplib.FTP | None = None

    def conn(self) -> ftplib.FTP:
        if self._conn is None:
            socket.setdefaulttimeout(FTP_TIMEOUT)
            c = ftplib.FTP(DATASUS_FTP_HOST, timeout=FTP_TIMEOUT)
            c.login()
            c.set_pasv(True)
            self._conn = c
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.quit()
            except Exception:
                self._conn.close()
            self._conn = None

    def _retry(self, fn, *a, **k):
        last: Exception | None = None
        for attempt in range(HTTP_MAX_RETRIES):
            try:
                return fn(*a, **k)
            except (ftplib.error_temp, ftplib.error_proto, OSError, EOFError) as exc:
                last = exc
                self.close()
                time.sleep(HTTP_BACKOFF_SECONDS * (2**attempt))
        raise RuntimeError(
            "DATASUS FTP unreachable after retries. If running outside Brazil this is "
            "expected: the host refuses non-Brazilian egress at TCP level. "
            "Use backend='mirror'."
        ) from last

    def listdir(self, directory: str) -> list[RemoteFile]:
        def _do() -> list[RemoteFile]:
            c = self.conn()
            c.cwd(DATASUS_FTP_ROOT + directory)
            lines: list[str] = []
            c.retrlines("LIST", lines.append)
            out: list[RemoteFile] = []
            for line in lines:
                m = _LIST_RE.match(line)
                if not m:
                    continue  # directory entries carry <DIR> instead of a size
                out.append(
                    RemoteFile(
                        path=f"{directory.rstrip('/')}/{m['name']}",
                        name=m["name"],
                        bytes=int(m["size"]),
                        modified=_parse_iis_datetime(m["date"], m["time"]),
                    )
                )
            return out

        return self._retry(_do)

    def get(self, path: str) -> bytes:
        def _do() -> bytes:
            c = self.conn()
            chunks: list[bytes] = []
            c.retrbinary(f"RETR {DATASUS_FTP_ROOT}{path}", chunks.append)
            return b"".join(chunks)

        return self._retry(_do)


_FTP_LOCAL = threading.local()


def _ftp_for_thread() -> _FTP:
    """Return one reusable FTP connection manager per worker thread.

    ``ftplib.FTP`` connections are stateful (``cwd`` and data sockets) and
    therefore cannot be shared safely by concurrent retrievals. Thread-local
    managers preserve connection reuse without serialising independent files
    or risking commands crossing between workers.
    """
    manager = getattr(_FTP_LOCAL, "manager", None)
    if manager is None:
        manager = _FTP()
        _FTP_LOCAL.manager = manager
    return manager


# --------------------------------------------------------------------------
# Mirror backend
# --------------------------------------------------------------------------

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _mirror_list(prefix: str) -> list[RemoteFile]:
    """List mirror objects under ``prefix`` via the S3 v2 listing API."""
    out: list[RemoteFile] = []
    token: str | None = None
    key_prefix = prefix.lstrip("/")
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        while True:
            params = {"list-type": "2", "prefix": key_prefix, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            r = client.get(DATASUS_MIRROR_BASE, params=params)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for c in root.findall(f"{_S3_NS}Contents"):
                key = c.findtext(f"{_S3_NS}Key") or ""
                size = int(c.findtext(f"{_S3_NS}Size") or 0)
                mod = c.findtext(f"{_S3_NS}LastModified")
                if size == 0:
                    continue
                out.append(
                    RemoteFile(
                        path="/" + key,
                        name=key.rsplit("/", 1)[-1],
                        bytes=size,
                        modified=mod,
                    )
                )
            if (root.findtext(f"{_S3_NS}IsTruncated") or "false") != "true":
                break
            token = root.findtext(f"{_S3_NS}NextContinuationToken")
    return out


def _mirror_get(path: str) -> bytes:
    url = f"{DATASUS_MIRROR_BASE}{path}"
    last: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.content
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(HTTP_BACKOFF_SECONDS * (2**attempt))
    raise RuntimeError(f"mirror fetch failed for {url}") from last


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


def listdir(directory: str, *, backend: Backend = "ftp") -> list[RemoteFile]:
    """List one directory of the DATASUS tree.

    ``directory`` is relative to ``/dissemin/publicos`` and begins with a slash,
    e.g. ``"/SINAN/DADOS/FINAIS"``.
    """
    if backend == "ftp":
        return _ftp_for_thread().listdir(directory)
    return _mirror_list(directory)


def get(path: str, *, backend: Backend = "ftp") -> bytes:
    """Retrieve one file's bytes. ``path`` is relative to the public root."""
    if backend == "ftp":
        return _ftp_for_thread().get(path)
    return _mirror_get(path)


def resolve(directory: str, pattern: str, *, backend: Backend = "ftp") -> list[RemoteFile]:
    """List a directory and keep entries whose name matches a regex.

    Case-insensitive, because the tree is genuinely inconsistent: several SIM
    files carry an uppercase ``.DBC`` while their siblings are lowercase, and a
    case-sensitive fetcher silently 404s on exactly those years.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    return [f for f in listdir(directory, backend=backend) if rx.match(f.name)]


def close() -> None:
    """Close the current thread's FTP connection, if one was opened."""
    manager = getattr(_FTP_LOCAL, "manager", None)
    if manager is not None:
        manager.close()
        del _FTP_LOCAL.manager


# --------------------------------------------------------------------------
# Batch retrieval
# --------------------------------------------------------------------------
#
# DATASUS FTP is latency-bound, not bandwidth-bound. A measured single-file
# retrieval of a 0.3 MB SIM file takes ~1.9 s -- roughly 0.2 MB/s -- because
# each transfer pays a PASV handshake and a RETR setup, not because the link is
# slow. Fetching a full system serially therefore spends most of its time
# waiting: SIM DO alone is 27 UFs x 18 years = 486 files.
#
# The host imposes no documented rate limit and tolerates concurrent anonymous
# sessions, so the fix is a small pool of connections. Connections are already
# thread-local (see ``_ftp_for_thread``), so a thread pool is safe; the work is
# I/O-bound, so the GIL is not the constraint either.


@dataclass
class FetchProgress:
    """Snapshot of a batch retrieval, for progress reporting."""

    total: int
    done: int
    cached: int
    failed: int
    bytes_done: int
    bytes_total: int
    elapsed_s: float

    @property
    def pct(self) -> float:
        return 100.0 * self.done / self.total if self.total else 100.0

    @property
    def rate_mb_s(self) -> float:
        return (self.bytes_done / 1e6 / self.elapsed_s) if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float:
        """Seconds remaining, extrapolated from bytes rather than file count.

        File sizes in the DATASUS tree span three orders of magnitude (a 0.3 MB
        DOAC against a 137 MB DOBR), so a count-based ETA is meaningless.
        """
        remaining = max(self.bytes_total - self.bytes_done, 0)
        rate = self.bytes_done / self.elapsed_s if self.elapsed_s > 0 else 0.0
        return remaining / rate if rate > 0 else float("nan")

    def line(self) -> str:
        return (
            f"{self.done}/{self.total} files ({self.pct:.0f}%) "
            f"{self.bytes_done / 1e6:.0f}/{self.bytes_total / 1e6:.0f} MB "
            f"{self.rate_mb_s:.1f} MB/s eta {self.eta_s:.0f}s "
            f"[{self.cached} cached, {self.failed} failed]"
        )


@dataclass
class FetchReport:
    """Outcome of a batch retrieval."""

    paths: dict[str, Path]  # cache key -> local path
    failures: dict[str, str]  # cache key -> error text
    cached: int
    downloaded: int
    bytes_downloaded: int
    elapsed_s: float

    def raise_if_failed(self, tolerate: int = 0) -> "FetchReport":
        if len(self.failures) > tolerate:
            first = list(self.failures.items())[:5]
            raise RuntimeError(
                f"{len(self.failures)} of {len(self.paths) + len(self.failures)} "
                f"retrievals failed; first: {first}"
            )
        return self


def fetch_many(
    files: Sequence[RemoteFile],
    key_for: Callable[[RemoteFile], str],
    *,
    source: str,
    backend: Backend = "ftp",
    refresh: bool = False,
    max_workers: int = 8,
    tool_version: str | None = None,
    on_progress: Callable[[FetchProgress], None] | None = None,
    progress_every: float = 5.0,
) -> FetchReport:
    """Retrieve many remote files concurrently, into the provenance cache.

    Already-cached entries are resolved without opening a connection, so a
    re-run of a partially completed batch costs nothing for what it already
    has. Per-file failures are collected rather than aborting the batch: one
    missing UF-year should not destroy an otherwise complete extraction, and
    the caller decides what is tolerable via
    :meth:`FetchReport.raise_if_failed`.

    ``max_workers`` above ~16 stops helping and starts provoking transient
    refusals from the host.
    """
    from brepi.io import cache  # local import: cache imports config, not this

    started = time.monotonic()
    todo: list[RemoteFile] = []
    paths: dict[str, Path] = {}
    n_cached = 0

    for f in files:
        key = key_for(f)
        if not refresh and cache.read_provenance(key) is not None:
            p = cache.cache_path(key)
            if p.exists():
                paths[key] = p
                n_cached += 1
                continue
        todo.append(f)

    bytes_total = sum(f.bytes for f in todo)
    state = {"done": n_cached, "bytes": 0, "failed": 0, "last": 0.0}
    lock = threading.Lock()
    failures: dict[str, str] = {}

    def _emit(force: bool = False) -> None:
        now = time.monotonic()
        if on_progress is None:
            return
        if not force and (now - state["last"]) < progress_every:
            return
        state["last"] = now
        on_progress(FetchProgress(
            total=len(files), done=state["done"], cached=n_cached,
            failed=state["failed"], bytes_done=state["bytes"],
            bytes_total=bytes_total, elapsed_s=now - started,
        ))

    def _one(f: RemoteFile) -> tuple[str, Path | None, str | None]:
        key = key_for(f)
        try:
            uri = f.uri_ftp if backend == "ftp" else f.uri_mirror
            path, _ = cache.fetch(
                key,
                lambda: (get(f.path, backend=backend),
                         {"remote_modified": f.modified,
                          "params": {"backend": backend, "bytes": f.bytes}}),
                source=source, uri=uri, refresh=refresh, tool_version=tool_version,
            )
            return key, path, None
        except Exception as exc:  # noqa: BLE001 - collected, not swallowed
            return key, None, f"{type(exc).__name__}: {exc}"

    if todo:
        workers = max(1, min(max_workers, len(todo)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, f): f for f in todo}
            for fut in as_completed(futures):
                f = futures[fut]
                key, path, err = fut.result()
                with lock:
                    state["done"] += 1
                    state["bytes"] += f.bytes
                    if err is None and path is not None:
                        paths[key] = path
                    else:
                        failures[key] = err or "unknown error"
                        state["failed"] += 1
                    _emit()
    _emit(force=True)

    return FetchReport(
        paths=paths, failures=failures, cached=n_cached,
        downloaded=len(todo) - len(failures),
        bytes_downloaded=state["bytes"], elapsed_s=time.monotonic() - started,
    )


__all__ = [
    "RemoteFile", "Backend", "listdir", "get", "resolve", "close",
    "fetch_many", "FetchReport", "FetchProgress",
]
