"""PAM/PPM selections and derived densities.

The interesting assertions here are the refusals. Table 3939 publishes no
``Total`` category, so a sum that double-counts Suíno-total against
Suíno-matrizes produces a larger number and no error anywhere downstream. These
tests pin the point at which that is caught.
"""

from __future__ import annotations

import polars as pl
import pytest

from brepi.sources.agriculture import (
    AgricultureError,
    build_crop_area,
    build_livestock_density,
    crop_area_selection,
    livestock_selection,
)
from brepi.sources.agriculture.pam_ppm import LIVESTOCK_SPECIES, CROP_TEMPORARY


def _facts(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "locality_id": pl.Utf8,
            "period": pl.Utf8,
            "category_ids": pl.List(pl.Utf8),
            "value_numeric": pl.Float64,
        },
    )


def _herd_facts() -> pl.DataFrame:
    return _facts(
        [
            {"locality_id": "2704302", "period": "2010",
             "category_ids": [LIVESTOCK_SPECIES["cattle"]], "value_numeric": 1000.0},
            {"locality_id": "2704302", "period": "2010",
             "category_ids": [LIVESTOCK_SPECIES["swine"]], "value_numeric": 200.0},
            {"locality_id": "2704302", "period": "2011",
             "category_ids": [LIVESTOCK_SPECIES["cattle"]], "value_numeric": 1200.0},
            {"locality_id": "2704302", "period": "2011",
             "category_ids": [LIVESTOCK_SPECIES["swine"]], "value_numeric": 100.0},
        ]
    )


AREA = pl.DataFrame({"munic_code": ["2704302"], "area_km2": [500.0]})


# --------------------------------------------------------------- selections --

def test_livestock_selection_shape() -> None:
    sel = livestock_selection(["cattle", "swine"], range(2007, 2010))
    assert sel.agregado == 3939
    assert sel.periods == ("2007", "2008", "2009")
    assert sel.variables == ("105",)
    assert sel.classifications == {"79": ("2670", "32794")}


def test_livestock_refuses_parent_with_its_own_child() -> None:
    with pytest.raises(AgricultureError, match="double-counts"):
        livestock_selection(["swine", "swine_breeding_females"], [2010])
    with pytest.raises(AgricultureError, match="double-counts"):
        livestock_selection(["poultry", "hens"], [2010])


def test_livestock_allows_child_alone() -> None:
    """The subset on its own is a coherent request; only the pair is not."""
    sel = livestock_selection(["swine_breeding_females"], [2010])
    assert sel.classifications == {"79": ("32795",)}


def test_unknown_species_names_the_known_ones() -> None:
    with pytest.raises(AgricultureError, match="unknown species"):
        livestock_selection(["cattle", "llama"], [2010])


def test_empty_requests_refused() -> None:
    with pytest.raises(AgricultureError, match="no species"):
        livestock_selection([], [2010])
    with pytest.raises(AgricultureError, match="no years"):
        livestock_selection(["cattle"], [])
    with pytest.raises(AgricultureError, match="no crops"):
        crop_area_selection([], [2010])


def test_duplicate_species_refused() -> None:
    with pytest.raises(AgricultureError, match="duplicate"):
        livestock_selection(["cattle", "cattle"], [2010])


def test_crop_selection_measure_switches_variable() -> None:
    planted = crop_area_selection(["rice"], [2015])
    harvested = crop_area_selection(["rice"], [2015], measure="harvested")
    assert planted.variables == ("109",)
    assert harvested.variables == ("216",)
    assert planted.classifications == {"81": ("2692",)}


def test_crop_selection_rejects_bad_measure() -> None:
    with pytest.raises(AgricultureError, match="planted"):
        crop_area_selection(["rice"], [2015], measure="value")


def test_rice_ids_differ_between_1612_and_5457() -> None:
    """Guards the compendium's Group 14 note against a silent edit.

    5457 calls rice 40102. If anyone ever 'harmonises' these constants by name
    the two tables become silently interchangeable, which they are not.
    """
    assert CROP_TEMPORARY["rice"] == "2692"


# ------------------------------------------------------------------ derived --

def test_livestock_density_divides_by_area() -> None:
    out = build_livestock_density(_herd_facts(), AREA)
    row = out.filter(pl.col("year") == 2010).to_dicts()[0]
    assert row["herd_cattle"] == 1000.0
    assert row["herd_per_km2_cattle"] == pytest.approx(2.0)
    assert row["herd_per_km2_swine"] == pytest.approx(0.4)


def test_livestock_density_keeps_row_when_area_unknown() -> None:
    """A missing area must null the ratio, never drop the municipality."""
    out = build_livestock_density(
        _herd_facts(), pl.DataFrame({"munic_code": ["9999999"], "area_km2": [10.0]})
    )
    assert out.height == 2
    assert out["herd_per_km2_cattle"].null_count() == 2
    assert out["herd_cattle"].to_list() == [1000.0, 1200.0]


def test_zero_area_gives_null_not_infinity() -> None:
    out = build_livestock_density(
        _herd_facts(), pl.DataFrame({"munic_code": ["2704302"], "area_km2": [0.0]})
    )
    assert out["herd_per_km2_cattle"].null_count() == 2


def test_livestock_density_per_capita_when_population_given() -> None:
    pop = pl.DataFrame(
        {"munic_code": ["2704302", "2704302"], "year": [2010, 2011],
         "population": [500.0, 0.0]}
    )
    out = build_livestock_density(_herd_facts(), AREA, population=pop)
    rows = {r["year"]: r for r in out.to_dicts()}
    assert rows[2010]["herd_per_1k_cattle"] == pytest.approx(2000.0)
    assert rows[2011]["herd_per_1k_cattle"] is None  # zero population, not inf


def test_density_requires_land_area() -> None:
    with pytest.raises(AgricultureError, match="area_km2"):
        build_livestock_density(_herd_facts(), pl.DataFrame({"munic_code": ["1"]}))


def test_unrequested_category_in_response_is_an_error() -> None:
    """A response carrying a category nobody asked for means the request and
    the frame disagree; silently pivoting it would invent a column."""
    rogue = _facts(
        [{"locality_id": "2704302", "period": "2010",
          "category_ids": ["999999"], "value_numeric": 1.0}]
    )
    with pytest.raises(AgricultureError, match="not requested"):
        build_livestock_density(rogue, AREA)


def test_crop_share_converts_km2_to_hectares() -> None:
    facts = _facts(
        [{"locality_id": "2704302", "period": "2010",
          "category_ids": [CROP_TEMPORARY["rice"]], "value_numeric": 5000.0}]
    )
    out = build_crop_area(facts, AREA)
    row = out.to_dicts()[0]
    assert row["planted_ha_rice"] == 5000.0
    # 500 km2 == 50,000 ha, so 5,000 ha of rice is a tenth of the municipality
    assert row["share_land_rice"] == pytest.approx(0.1)


def test_crop_area_missing_columns_refused() -> None:
    facts = _facts([])
    with pytest.raises(AgricultureError, match="area_km2"):
        build_crop_area(facts, pl.DataFrame({"munic_code": []}))
