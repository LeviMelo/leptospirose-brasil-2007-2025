"""Geographic lattice, code systems and crosswalks.

Every join in the project lands on the municipality lattice defined here.
"""

from __future__ import annotations

from .health_regions import (
    HealthRegionError,
    HealthRegionReport,
    fetch_open_datasus,
    load_official_crosswalk,
)
from .meshes import (
    MeshError,
    MeshReport,
    fetch_municipality_mesh,
    materialize_municipality_mesh,
    municipality_mesh_url,
)
from .transfer import (
    TransferDiagnostics,
    TransferError,
    VariableKind,
    correspondence_from_support,
    transfer_extensive,
    transfer_intensive,
    transfer_rate,
)

__all__ = [
    "lattice",
    "HealthRegionError",
    "HealthRegionReport",
    "fetch_open_datasus",
    "load_official_crosswalk",
    "MeshError",
    "MeshReport",
    "municipality_mesh_url",
    "fetch_municipality_mesh",
    "materialize_municipality_mesh",
    "TransferDiagnostics",
    "TransferError",
    "VariableKind",
    "correspondence_from_support",
    "transfer_extensive",
    "transfer_intensive",
    "transfer_rate",
]
