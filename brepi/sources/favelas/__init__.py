"""Censo 2022 *Favelas e Comunidades Urbanas*.

Published only after the 2022 Census. Group 15 of the SIDRA compendium.
"""

from .census2022 import (
    FavelaError,
    SEWAGE_CATEGORIES,
    UNSAFE_SEWAGE,
    build_favela_profile,
    build_inside_outside_sewage,
    favela_count_selection,
    favela_population_selection,
    favela_sewage_selection,
    inside_outside_sewage_selection,
)

__all__ = [
    "FavelaError",
    "SEWAGE_CATEGORIES",
    "UNSAFE_SEWAGE",
    "build_favela_profile",
    "build_inside_outside_sewage",
    "favela_count_selection",
    "favela_population_selection",
    "favela_sewage_selection",
    "inside_outside_sewage_selection",
]
