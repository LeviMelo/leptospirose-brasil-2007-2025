"""IBGE SIDRA aggregate-cube source layer.

Layering, per SIDRA_DESC: nothing outside this package constructs a SIDRA URL.
``api`` retrieves and normalises, ``plan`` sizes and splits requests, ``registry``
holds live-rebuilt table metadata, ``extract`` runs the plan and lands facts.
"""

from __future__ import annotations

__all__ = ["api", "plan", "registry", "extract"]
