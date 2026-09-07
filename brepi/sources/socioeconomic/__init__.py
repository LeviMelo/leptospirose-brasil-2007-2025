"""Reusable census and administrative socioeconomic indicators."""

from .census import (
    SocioeconomicError,
    build_urban_anchors,
    urban_selection_2010,
    urban_selection_2022,
)
from .economy import (
    build_real_gdp_per_capita,
    gdp_deflator_selection,
    municipal_gdp_selection,
)

__all__ = [
    "SocioeconomicError",
    "build_urban_anchors",
    "urban_selection_2010",
    "urban_selection_2022",
    "build_real_gdp_per_capita",
    "gdp_deflator_selection",
    "municipal_gdp_selection",
]
