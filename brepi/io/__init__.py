"""Transport, caching and decoding primitives.

Nothing in :mod:`brepi.sources` touches the network except through
:func:`brepi.io.cache.fetch`, so every byte in an analysis has a provenance
sidecar behind it.
"""

from __future__ import annotations

from brepi.io.cache import (
    Provenance,
    diff_against_remote,
    fetch,
    read_provenance,
    store,
    verify_snapshot,
    write_manifest,
)
from brepi.io.datasus_ftp import Backend, RemoteFile, listdir, resolve
from brepi.io.dbc import DbcDecodeError, dbf_header, read_dbc, read_dbc_bytes

__all__ = [
    "Provenance",
    "fetch",
    "store",
    "read_provenance",
    "write_manifest",
    "verify_snapshot",
    "diff_against_remote",
    "RemoteFile",
    "Backend",
    "listdir",
    "resolve",
    "read_dbc",
    "read_dbc_bytes",
    "dbf_header",
    "DbcDecodeError",
]
