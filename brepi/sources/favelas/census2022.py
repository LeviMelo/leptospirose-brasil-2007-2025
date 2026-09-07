"""Favela counts, population and sanitation from the 2022 Census.

Four tables, three of which are ordinary and one of which is not:

``9883``  count of settlements per municipality
``9887``  households and resident population in settlements
``9888``  population, land area and density in settlements
``9892``  settlement households by bathroom availability and sewage type
``10344`` sewage type **inside versus outside** settlements, same municipality

The last is the reason this module exists. Every other structural covariate in
the panel compares municipalities to each other, so any effect attributed to
sanitation can be attributed instead to whatever else differs between rich and
poor municipalities. Table 10344 reports the same sewage classification on both
sides of a boundary drawn *within* one municipality, where municipal income,
surveillance capacity and climate are constant by construction.

Three traps.

*Absent is not zero.* These tables cover municipalities that have settlements.
A municipality with none is missing from the response, and filling it with zero
households-in-favela is correct while filling it with zero *share* is not --
the share is undefined. :func:`build_favela_profile` returns an explicit
``has_favela`` flag so the distinction survives the join.

*Table 10344 has a restricted universe.* Its counts cover only census tracts
sampled for the Pesquisa Urbanística do Entorno dos Domicílios, not all tracts.
Its totals are therefore not municipal totals, and the returned frame carries
``universe`` saying so rather than trusting a reader to remember.

*The sewage classification is hierarchical.* ``46290`` is a parent of ``72110``
and ``72111``; summing them with their parent double-counts. Only leaf
categories appear in :data:`UNSAFE_SEWAGE`.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from brepi.sources.sidra.extract import Selection

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

TAB_COUNT = 9883
TAB_POPULATION = 9887
TAB_DENSITY = 9888
TAB_SEWAGE = 9892
TAB_INSIDE_OUTSIDE = 10344

#: ``Clsf 11558``, shared verbatim with tables 6805, 6806, 9860 and 9892.
#: Parents are marked; only leaves may be summed.
SEWAGE_CATEGORIES: dict[str, str] = {
    "total": "46292",                     # parent of everything
    "network_or_septic_connected": "46290",  # parent of the two below
    "general_or_storm_network": "72110",
    "septic_connected": "72111",
    "septic_unconnected": "72112",
    "rudimentary_pit": "72113",
    "ditch": "92858",
    "river_lake_sea": "72114",
    "other": "72115",
    "none": "92861",
}

#: Leaf categories that constitute leptospirosis-relevant exposure: standing or
#: open waste water in contact with the household. Deliberately excludes
#: ``septic_unconnected``, which is contained but unsewered, and is a judgment
#: call that belongs in the open rather than buried in a filter expression.
UNSAFE_SEWAGE: tuple[str, ...] = (
    "rudimentary_pit",
    "ditch",
    "river_lake_sea",
    "none",
)

#: Occupied private permanent dwellings -- the denominator matching table 9892's
#: variable 9913. Using ``total`` instead mixes in vacant and collective units.
SPECIES_OCCUPIED = "59998"

VAR_SETTLEMENT_COUNT = "9910"
VAR_FAVELA_HOUSEHOLDS = "9909"
VAR_FAVELA_POPULATION = "9612"
VAR_FAVELA_AREA = "9911"
VAR_SEWAGE_HOUSEHOLDS = "9913"
VAR_INSIDE_HOUSEHOLDS = "13548"
VAR_OUTSIDE_HOUSEHOLDS = "13549"


class FavelaError(RuntimeError):
    """Raised when a favela request or frame is internally inconsistent."""


def _categories(names: Sequence[str]) -> tuple[str, ...]:
    unknown = [n for n in names if n not in SEWAGE_CATEGORIES]
    if unknown:
        raise FavelaError(
            f"unknown sewage category: {sorted(unknown)}. "
            f"Known: {sorted(SEWAGE_CATEGORIES)}"
        )
    if "total" in names and len(names) > 1:
        raise FavelaError(
            "'total' is the parent of every other category; requesting it "
            "alongside its children and summing double-counts"
        )
    if "network_or_septic_connected" in names and (
        {"general_or_storm_network", "septic_connected"} & set(names)
    ):
        raise FavelaError(
            "'network_or_septic_connected' contains 'general_or_storm_network' "
            "and 'septic_connected'; do not request a parent with its children"
        )
    return tuple(SEWAGE_CATEGORIES[n] for n in names)


def favela_count_selection() -> Selection:
    """Number of settlements per municipality."""
    return Selection(
        agregado=TAB_COUNT, periods=("2022",), variables=(VAR_SETTLEMENT_COUNT,),
        label="favela_count_2022",
    )


def favela_population_selection() -> Selection:
    """Households and residents in settlements, occupied private permanent."""
    return Selection(
        agregado=TAB_POPULATION, periods=("2022",),
        variables=(VAR_FAVELA_HOUSEHOLDS, VAR_FAVELA_POPULATION),
        classifications={"3": (SPECIES_OCCUPIED,)},
        label="favela_population_2022",
    )


def favela_sewage_selection(categories: Sequence[str] = UNSAFE_SEWAGE) -> Selection:
    """Settlement households by sewage type.

    Requests the bathroom classification's total so the sewage margin is not
    silently conditioned on having a bathroom.
    """
    return Selection(
        agregado=TAB_SEWAGE, periods=("2022",), variables=(VAR_SEWAGE_HOUSEHOLDS,),
        classifications={"11558": _categories(categories), "458": ("72117",)},
        label="favela_sewage_2022",
    )


def inside_outside_sewage_selection(
    categories: Sequence[str] = UNSAFE_SEWAGE,
) -> Selection:
    """Households by sewage type, inside and outside settlements."""
    return Selection(
        agregado=TAB_INSIDE_OUTSIDE, periods=("2022",),
        variables=(VAR_INSIDE_HOUSEHOLDS, VAR_OUTSIDE_HOUSEHOLDS),
        classifications={"11558": _categories(categories)},
        label="favela_inside_outside_sewage_2022",
    )


def _municipality(facts: pl.DataFrame) -> pl.DataFrame:
    required = {"locality_id", "variable_id", "value_numeric"}
    missing = required - set(facts.columns)
    if missing:
        raise FavelaError(f"facts frame missing columns: {sorted(missing)}")
    return facts.with_columns(
        pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code")
    )


def build_favela_profile(
    count_facts: pl.DataFrame,
    population_facts: pl.DataFrame,
    spine: pl.DataFrame,
) -> pl.DataFrame:
    """Settlement count, households and population, completed against ``spine``.

    ``spine`` must carry every municipality in the study lattice. Municipalities
    with no settlement are returned with zero counts and ``has_favela`` false --
    which is a statement about the settlement universe, not a measurement, and
    is why the flag exists rather than leaving a reader to infer it from a zero.
    """
    if "munic_code" not in spine.columns:
        raise FavelaError("spine needs a 'munic_code' column")
    counts = (
        _municipality(count_facts)
        .filter(pl.col("variable_id") == VAR_SETTLEMENT_COUNT)
        .group_by("munic_code")
        .agg(pl.col("value_numeric").sum().alias("favela_count"))
    )
    pop = _municipality(population_facts)
    households = (
        pop.filter(pl.col("variable_id") == VAR_FAVELA_HOUSEHOLDS)
        .group_by("munic_code")
        .agg(pl.col("value_numeric").sum().alias("favela_households"))
    )
    residents = (
        pop.filter(pl.col("variable_id") == VAR_FAVELA_POPULATION)
        .group_by("munic_code")
        .agg(pl.col("value_numeric").sum().alias("favela_population"))
    )
    out = (
        spine.select(pl.col("munic_code").cast(pl.Utf8).str.zfill(7))
        .unique()
        .join(counts, on="munic_code", how="left")
        .join(households, on="munic_code", how="left")
        .join(residents, on="munic_code", how="left")
    )
    return out.with_columns(
        pl.col("favela_count").fill_null(0.0),
        pl.col("favela_households").fill_null(0.0),
        pl.col("favela_population").fill_null(0.0),
        (pl.col("favela_count").fill_null(0.0) > 0).alias("has_favela"),
    ).sort("munic_code")


def build_inside_outside_sewage(facts: pl.DataFrame) -> pl.DataFrame:
    """Unsafe-sewage household counts inside and outside settlements.

    Returns one row per municipality with both sides and their difference in
    percentage points, plus a ``universe`` column recording the sampling
    restriction. Shares are null, never zero, where a side has no households:
    a municipality with no sampled tracts outside settlements has an undefined
    outside share, not a clean one.
    """
    tidy = _municipality(facts)
    inside = (
        tidy.filter(pl.col("variable_id") == VAR_INSIDE_HOUSEHOLDS)
        .group_by("munic_code")
        .agg(pl.col("value_numeric").sum().alias("inside_unsafe"))
    )
    outside = (
        tidy.filter(pl.col("variable_id") == VAR_OUTSIDE_HOUSEHOLDS)
        .group_by("munic_code")
        .agg(pl.col("value_numeric").sum().alias("outside_unsafe"))
    )
    return (
        inside.join(outside, on="munic_code", how="full", coalesce=True)
        .with_columns(
            pl.lit(
                "census tracts sampled for the Pesquisa Urbanistica do Entorno "
                "dos Domicilios, in municipalities with settlements; not "
                "municipal totals"
            ).alias("universe")
        )
        .sort("munic_code")
    )
