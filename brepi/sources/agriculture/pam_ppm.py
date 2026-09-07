"""PAM crop area and PPM herd size as municipality-year covariates.

Two SIDRA families, both annual since 1974 at municipality grain:

``3939`` (Pesquisa da Pecuária Municipal)
    Herd size in head, by species.
``1612`` (Produção Agrícola Municipal)
    Planted and harvested area in hectares, by temporary crop.

Traps these functions encode, all recorded in ``docs/sources/SIDRA_COMPENDIUM.md``
under Group 14.

*Table 3939 has no ``Total`` category.* Unlike almost every other classified
table in the registry, there is no category to ask for when you want all
livestock. The total is the analyst's to build -- and building it naively
double-counts, because ``32794`` (Suíno - total) contains ``32795`` (matrizes)
and ``32796`` (Galináceos - total) contains ``32793`` (galinhas). :data:`OVERLAPS`
records the containments and :func:`livestock_selection` refuses a request that
would sum a parent with its own child.

*Head counts are not density.* A municipality with 200,000 cattle over
30,000 km2 is not the exposure a municipality with 200,000 cattle over 300 km2
is. :func:`build_livestock_density` requires land area and refuses to guess.

*Tables 1612 and 5457 use different category ids for the same crop.* Rice is
``2692`` in 1612 and ``40102`` in 5457. Nothing in either response says so.
Only 1612 is exposed here; use of 5457 should go through its own constants
rather than by translating names.

*Value of production spans five currencies.* Variable ``215`` runs Cruzeiro ->
Cruzado -> Cruzado Novo -> Cruzeiro Real -> Real across the series, and the unit
string is the only warning SIDRA gives. Area and head counts carry no such
hazard, so those are what this module returns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from brepi.sources.sidra.extract import Selection

__all__ = [
    "AgricultureError",
    "CROP_TEMPORARY",
    "LIVESTOCK_SPECIES",
    "OVERLAPS",
    "build_crop_area",
    "build_livestock_density",
    "crop_area_selection",
    "livestock_selection",
]

PPM_HERDS = 3939
PAM_TEMPORARY = 1612

#: Herd-size variable of table 3939, in head.
VAR_HERD = "105"
#: Planted area of table 1612, in hectares.
VAR_PLANTED = "109"
#: Harvested area of table 1612, in hectares.
VAR_HARVESTED = "216"

#: Species name -> category id in ``Clsf 79``. Names are the stable key here;
#: the ids are what SIDRA wants.
LIVESTOCK_SPECIES: dict[str, str] = {
    "cattle": "2670",
    "buffalo": "2675",
    "equine": "2672",
    "swine": "32794",
    "swine_breeding_females": "32795",
    "goat": "2681",
    "sheep": "2677",
    "poultry": "32796",
    "hens": "32793",
    "quail": "2680",
}

#: Crop name -> category id in ``Clsf 81`` of table 1612. Only the crops with a
#: standing epidemiological or land-use rationale are named; the table has 34.
CROP_TEMPORARY: dict[str, str] = {
    "rice": "2692",
    "sugarcane": "2696",
    "maize": "2711",
    "soy": "2713",
    "cassava": "2708",
    "beans": "2702",
    "wheat": "2716",
    "cotton": "2689",
    "tobacco": "2703",
}

#: parent -> members contained within it. Summing both sides double-counts.
OVERLAPS: dict[str, frozenset[str]] = {
    "swine": frozenset({"swine_breeding_females"}),
    "poultry": frozenset({"hens"}),
}


class AgricultureError(RuntimeError):
    """Raised when a PAM/PPM request or frame is internally inconsistent."""


def _resolve(names: Sequence[str], lookup: dict[str, str], kind: str) -> list[str]:
    unknown = [n for n in names if n not in lookup]
    if unknown:
        raise AgricultureError(
            f"unknown {kind}: {sorted(unknown)}. Known: {sorted(lookup)}"
        )
    if len(set(names)) != len(names):
        raise AgricultureError(f"duplicate {kind} requested: {sorted(names)}")
    return [lookup[n] for n in names]


def _check_overlaps(names: Sequence[str]) -> None:
    requested = set(names)
    for parent, children in OVERLAPS.items():
        if parent in requested:
            both = children & requested
            if both:
                raise AgricultureError(
                    f"{parent!r} already contains {sorted(both)}; requesting both "
                    "and summing them double-counts. Ask for the parent or the "
                    "children, not both. Table 3939 has no Total category, so "
                    "nothing downstream will catch this for you."
                )


def livestock_selection(
    species: Sequence[str],
    years: Iterable[int],
    *,
    label: str | None = None,
) -> Selection:
    """Herd size in head for ``species`` over ``years``.

    Refuses a request that pairs a species with one of its own subsets, since
    table 3939 offers no total against which such a sum could be checked.
    """
    species = list(species)
    if not species:
        raise AgricultureError("no species requested")
    _check_overlaps(species)
    categories = _resolve(species, LIVESTOCK_SPECIES, "species")
    periods = tuple(str(y) for y in years)
    if not periods:
        raise AgricultureError("no years requested")
    return Selection(
        agregado=PPM_HERDS,
        periods=periods,
        variables=(VAR_HERD,),
        classifications={"79": tuple(categories)},
        label=label or f"ppm_herds_{periods[0]}_{periods[-1]}",
    )


def crop_area_selection(
    crops: Sequence[str],
    years: Iterable[int],
    *,
    measure: str = "planted",
    label: str | None = None,
) -> Selection:
    """Planted or harvested area in hectares for ``crops`` over ``years``.

    ``measure`` is ``"planted"`` or ``"harvested"``. They differ by loss, and
    for an exposure argument planted area is usually what is wanted: a flooded
    rice field exposes workers whether or not it is ultimately harvested.
    """
    variable = {"planted": VAR_PLANTED, "harvested": VAR_HARVESTED}.get(measure)
    if variable is None:
        raise AgricultureError(
            f"measure must be 'planted' or 'harvested', got {measure!r}"
        )
    crops = list(crops)
    if not crops:
        raise AgricultureError("no crops requested")
    categories = _resolve(crops, CROP_TEMPORARY, "crop")
    periods = tuple(str(y) for y in years)
    if not periods:
        raise AgricultureError("no years requested")
    return Selection(
        agregado=PAM_TEMPORARY,
        periods=periods,
        variables=(variable,),
        classifications={"81": tuple(categories)},
        label=label or f"pam_{measure}_{periods[0]}_{periods[-1]}",
    )


def _pivot(facts: pl.DataFrame, lookup: dict[str, str], prefix: str) -> pl.DataFrame:
    """One row per municipality-year, one column per requested category."""
    required = {"locality_id", "period", "category_ids", "value_numeric"}
    missing = required - set(facts.columns)
    if missing:
        raise AgricultureError(f"facts frame missing columns: {sorted(missing)}")
    by_id = {cid: name for name, cid in lookup.items()}
    tidy = (
        facts.with_columns(
            pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code"),
            pl.col("period").cast(pl.Int32).alias("year"),
            pl.col("category_ids").list.first().cast(pl.Utf8).alias("_cid"),
        )
        .with_columns(
            pl.col("_cid").replace_strict(by_id, default=None).alias("_name")
        )
    )
    unmapped = tidy.filter(pl.col("_name").is_null())["_cid"].unique().to_list()
    if unmapped:
        raise AgricultureError(
            f"response carries categories that were not requested: {sorted(unmapped)}"
        )
    wide = tidy.pivot(
        on="_name", index=["munic_code", "year"], values="value_numeric",
        aggregate_function="sum",
    )
    return wide.rename(
        {c: f"{prefix}{c}" for c in wide.columns if c not in ("munic_code", "year")}
    )


def build_livestock_density(
    facts: pl.DataFrame,
    land_area: pl.DataFrame,
    *,
    population: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Head counts plus head per km2, and head per 1,000 people when possible.

    ``land_area`` must carry ``munic_code`` and ``area_km2`` (SIDRA table 4714,
    variable ``6318``). Density is what carries the exposure argument; the raw
    count is kept alongside it so that the ratio can always be audited back to
    its numerator.

    Municipalities absent from ``land_area`` get null density rather than a
    dropped row -- losing a municipality silently is the failure this returns
    instead of.
    """
    if "area_km2" not in land_area.columns or "munic_code" not in land_area.columns:
        raise AgricultureError(
            "land_area needs columns 'munic_code' and 'area_km2' (SIDRA 4714 "
            "variable 6318); head counts are not an exposure without it"
        )
    wide = _pivot(facts, LIVESTOCK_SPECIES, "herd_")
    area = land_area.select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7),
        pl.col("area_km2").cast(pl.Float64),
    ).unique(subset="munic_code")
    out = wide.join(area, on="munic_code", how="left")
    safe_area = (
        pl.when(pl.col("area_km2") > 0).then(pl.col("area_km2")).otherwise(None)
    )
    species_cols = [c for c in wide.columns if c.startswith("herd_")]
    out = out.with_columns(
        [(pl.col(c) / safe_area).alias(c.replace("herd_", "herd_per_km2_"))
         for c in species_cols]
    )
    if population is not None:
        if not {"munic_code", "year", "population"} <= set(population.columns):
            raise AgricultureError(
                "population needs columns 'munic_code', 'year', 'population'"
            )
        pop = population.select(
            pl.col("munic_code").cast(pl.Utf8).str.zfill(7),
            pl.col("year").cast(pl.Int32),
            pl.col("population").cast(pl.Float64),
        )
        out = out.join(pop, on=["munic_code", "year"], how="left")
        safe_pop = (
            pl.when(pl.col("population") > 0).then(pl.col("population")).otherwise(None)
        )
        out = out.with_columns(
            [(pl.col(c) / safe_pop * 1000).alias(c.replace("herd_", "herd_per_1k_"))
             for c in species_cols]
        )
    return out.sort(["munic_code", "year"])


def build_crop_area(
    facts: pl.DataFrame,
    land_area: pl.DataFrame,
    *,
    measure: str = "planted",
) -> pl.DataFrame:
    """Crop area in hectares plus the share of municipal land it occupies.

    The share is the comparable quantity across municipalities of wildly
    different size. 1 km2 is 100 ha, and the conversion is applied here rather
    than left to the caller precisely because getting it wrong by two orders of
    magnitude produces a plausible-looking number.
    """
    if "area_km2" not in land_area.columns or "munic_code" not in land_area.columns:
        raise AgricultureError(
            "land_area needs columns 'munic_code' and 'area_km2' (SIDRA 4714)"
        )
    prefix = "planted_ha_" if measure == "planted" else "harvested_ha_"
    wide = _pivot(facts, CROP_TEMPORARY, prefix)
    area = land_area.select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7),
        (pl.col("area_km2").cast(pl.Float64) * 100).alias("area_ha"),
    ).unique(subset="munic_code")
    out = wide.join(area, on="munic_code", how="left")
    safe_area = pl.when(pl.col("area_ha") > 0).then(pl.col("area_ha")).otherwise(None)
    crop_cols = [c for c in wide.columns if c.startswith(prefix)]
    out = out.with_columns(
        [(pl.col(c) / safe_area).alias(c.replace(prefix, "share_land_"))
         for c in crop_cols]
    )
    return out.sort(["munic_code", "year"])
