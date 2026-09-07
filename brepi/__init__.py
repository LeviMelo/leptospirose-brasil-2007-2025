"""brepi -- reproducible Brazilian epidemiological data assembly.

The package is organised around one invariant: every remote byte enters
through :func:`brepi.io.cache.fetch` and leaves a provenance sidecar, so an
analysis can be re-run from a frozen manifest and the re-run can be *proved*
to be offline. DATASUS rewrites "final" files without notice; a snapshot plus
:func:`brepi.io.cache.verify_snapshot` is the only defence.

Entry points are resolved lazily. Importing ``brepi`` costs a ``config``
import and nothing else, which keeps the circular-import surface at zero and
avoids paying for polars/geopandas in a script that only needs ``PATHS``.
"""

from __future__ import annotations

import importlib
from typing import Any

from brepi.config import PATHS, Paths

__version__ = "0.1.0"

#: Attribute name -> (module, symbol). ``None`` means the module itself.
_LAZY: dict[str, tuple[str, str | None]] = {
    "io": ("brepi.io", None),
    "sources": ("brepi.sources", None),
    "sinan": ("brepi.sources.datasus.sinan", None),
    "sih": ("brepi.sources.datasus.sih", None),
    "sim": ("brepi.sources.datasus.sim", None),
    "cnes": ("brepi.sources.datasus.cnes", None),
    "population": ("brepi.sources.datasus.population", None),
    "fetch": ("brepi.io.cache", "fetch"),
    "store": ("brepi.io.cache", "store"),
    "Provenance": ("brepi.io.cache", "Provenance"),
    "write_manifest": ("brepi.io.cache", "write_manifest"),
    "verify_snapshot": ("brepi.io.cache", "verify_snapshot"),
    "diff_against_remote": ("brepi.io.cache", "diff_against_remote"),
    "read_dbc": ("brepi.io.dbc", "read_dbc"),
    "read_dbc_bytes": ("brepi.io.dbc", "read_dbc_bytes"),
    "dbf_header": ("brepi.io.dbc", "dbf_header"),
    "listdir": ("brepi.io.datasus_ftp", "listdir"),
    "resolve": ("brepi.io.datasus_ftp", "resolve"),
    "AGRAVOS": ("brepi.sources.datasus.sinan", "AGRAVOS"),
}

__all__ = ["PATHS", "Paths", "__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = target
    module = importlib.import_module(module_name)
    value = module if symbol is None else getattr(module, symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
