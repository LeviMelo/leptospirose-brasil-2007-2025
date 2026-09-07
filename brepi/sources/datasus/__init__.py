"""DATASUS source adapters.

One module per system. Each exposes the same three-step surface -- list what
exists, fetch one unit through the provenance cache, and assemble a filtered
extract -- so a study script reads the same way whether it is pulling
notifications, admissions, deaths, establishments or denominators.

===============  ==========================================================
:mod:`sinan`     notifiable diseases, national annual files, all agravos
:mod:`sih`       SIH-SUS RD hospitalisations, per-UF monthly
:mod:`sim`       SIM deaths, per-UF annual, CID-10 era
:mod:`cnes`      establishment register, per-UF monthly, capacity covariates
:mod:`population`  IBGE municipal denominators as redistributed by DATASUS
===============  ==========================================================
"""

from __future__ import annotations

from brepi.sources.datasus import cnes, population, sih, sim, sinan
from brepi.sources.datasus.cnes import (
    capacity_by_municipality,
    capacity_series,
    resolve_inventory,
)
from brepi.sources.datasus.population import municipal_population
from brepi.sources.datasus.sih import fetch_icd_admissions
from brepi.sources.datasus.sim import fetch_icd_deaths
from brepi.sources.datasus.sinan import AGRAVOS, fetch_range, fetch_year, snapshot

__all__ = [
    "sinan",
    "sih",
    "sim",
    "cnes",
    "population",
    "AGRAVOS",
    "fetch_year",
    "fetch_range",
    "snapshot",
    "fetch_icd_admissions",
    "fetch_icd_deaths",
    "capacity_by_municipality",
    "capacity_series",
    "resolve_inventory",
    "municipal_population",
]
