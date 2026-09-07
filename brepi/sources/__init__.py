"""Source adapters, one subpackage per data provider.

Subpackages are resolved lazily so that importing ``brepi.sources`` does not
drag in the optional heavy dependencies of providers a given study never
touches (geospatial stacks for ``landuse``, for instance).
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBPACKAGES = (
    "datasus",
    "climate",
    "disasters",
    "landuse",
    "sanitation",
    "sidra",
)

__all__ = list(_SUBPACKAGES)


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBPACKAGES))
