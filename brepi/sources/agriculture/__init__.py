"""Municipal agricultural and livestock series (PAM, PPM).

These are the only municipal socioeconomic families in the SIDRA compendium
observed *every year* rather than at census anchors, which makes them the only
ones that can enter a model as genuinely time-varying covariates instead of as
interpolated constants.
"""

from .pam_ppm import (
    AgricultureError,
    CROP_TEMPORARY,
    LIVESTOCK_SPECIES,
    build_crop_area,
    build_livestock_density,
    crop_area_selection,
    livestock_selection,
)

__all__ = [
    "AgricultureError",
    "CROP_TEMPORARY",
    "LIVESTOCK_SPECIES",
    "build_crop_area",
    "build_livestock_density",
    "crop_area_selection",
    "livestock_selection",
]
