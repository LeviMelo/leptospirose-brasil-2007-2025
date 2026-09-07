"""PNSB statements and the Censo 2022 favela block.

The assertions worth reading are the ones about absence. A municipality with no
favela is missing from table 9883, not present with a zero, and a municipality
outside the Pesquisa Urbanística sample is missing from 10344 for a completely
different reason. Both look like "no row" and neither means "no exposure".
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from brepi.sources.favelas import (
    FavelaError,
    build_favela_profile,
    build_inside_outside_sewage,
    favela_sewage_selection,
    inside_outside_sewage_selection,
)
from brepi.sources.sanitation import (
    PNSBError,
    build_disease_flag,
    build_service_flags,
    compare_with_notifications,
    disease_occurrence_selection,
    service_existence_selection,
)


def _facts(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "locality_id": pl.Utf8,
            "variable_id": pl.Utf8,
            "period": pl.Utf8,
            "category_ids": pl.List(pl.Utf8),
            "value_numeric": pl.Float64,
        },
    )


# ------------------------------------------------------------------- PNSB ---

def test_disease_selection_pins_one_category() -> None:
    sel = disease_occurrence_selection("leptospirosis")
    assert sel.agregado == 354
    assert sel.classifications == {"12963": ("120933",)}


def test_unknown_disease_refused() -> None:
    with pytest.raises(PNSBError, match="unknown disease"):
        disease_occurrence_selection("plague")


def test_unknown_service_refused() -> None:
    with pytest.raises(PNSBError, match="unknown service"):
        service_existence_selection(("teleportation",))


def test_disease_flag_is_boolean_and_keeps_nulls() -> None:
    facts = _facts([
        {"locality_id": "2704302", "variable_id": "2597", "period": "2008",
         "category_ids": ["120933"], "value_numeric": 1.0},
        {"locality_id": "2704303", "variable_id": "2597", "period": "2008",
         "category_ids": ["120933"], "value_numeric": 0.0},
        {"locality_id": "2704304", "variable_id": "2597", "period": "2008",
         "category_ids": ["120933"], "value_numeric": None},
    ])
    out = build_disease_flag(facts).sort("munic_code")
    assert out["pnsb_leptospirosis_2008"].to_list() == [True, False, None]


def test_counts_above_one_mean_the_wrong_request() -> None:
    """At N6 the value is a 0/1 flag. A 27 means an aggregate slipped in."""
    facts = _facts([
        {"locality_id": "2704302", "variable_id": "2597", "period": "2008",
         "category_ids": ["120933"], "value_numeric": 27.0},
    ])
    with pytest.raises(PNSBError, match="must be 0 or 1"):
        build_disease_flag(facts)


def test_service_flags_pivot_by_year() -> None:
    facts = _facts([
        {"locality_id": "2704302", "variable_id": "2613", "period": "2000",
         "category_ids": ["98367"], "value_numeric": 0.0},
        {"locality_id": "2704302", "variable_id": "2613", "period": "2008",
         "category_ids": ["98367"], "value_numeric": 1.0},
    ])
    out = build_service_flags(facts).sort("year")
    assert out["pnsb_has_sewer_network"].to_list() == [False, True]
    assert out["year"].to_list() == [2000, 2008]


def test_comparison_reports_both_directions() -> None:
    flags = pl.DataFrame({
        "munic_code": ["0000001", "0000002", "0000003", "0000004"],
        "pnsb_leptospirosis_2008": [True, True, False, False],
    })
    notifications = pl.DataFrame({
        "munic_code": ["0000001", "0000003"],
        "cases": [5, 3],
    })
    got = compare_with_notifications(flags, notifications)
    assert got["declared"] == 2
    assert got["declared_with_no_notification"] == 1        # 0000002
    assert got["declared_silent_pct"] == pytest.approx(50.0)
    assert got["denied"] == 2
    assert got["denied_with_notification"] == 1             # 0000003
    assert got["denied_notified_pct"] == pytest.approx(50.0)


def test_comparison_drops_null_flags_not_rows_with_no_cases() -> None:
    """A municipality absent from the notification frame notified zero; a
    municipality with a null PNSB answer said nothing and cannot be counted."""
    flags = pl.DataFrame({
        "munic_code": ["0000001", "0000002"],
        "pnsb_leptospirosis_2008": [True, None],
    })
    got = compare_with_notifications(flags, pl.DataFrame({"munic_code": [], "cases": []}))
    assert got["declared"] == 1
    assert got["declared_with_no_notification"] == 1
    assert got["denied"] == 0
    assert math.isnan(got["denied_notified_pct"])


# ---------------------------------------------------------------- favelas ---

def test_sewage_selection_refuses_parent_with_children() -> None:
    with pytest.raises(FavelaError, match="do not request a parent"):
        favela_sewage_selection(
            ["network_or_septic_connected", "general_or_storm_network"]
        )


def test_sewage_selection_refuses_total_with_others() -> None:
    with pytest.raises(FavelaError, match="parent of every other"):
        favela_sewage_selection(["total", "rudimentary_pit"])


def test_default_unsafe_categories_are_leaves() -> None:
    sel = favela_sewage_selection()
    assert sel.classifications["11558"] == ("72113", "92858", "72114", "92861")
    # bathroom margin requested so sewage is not conditioned on having one
    assert sel.classifications["458"] == ("72117",)


def test_inside_outside_selection_asks_both_sides() -> None:
    sel = inside_outside_sewage_selection()
    assert sel.agregado == 10344
    assert sel.variables == ("13548", "13549")


def test_profile_completes_spine_and_flags_absence() -> None:
    counts = _facts([
        {"locality_id": "2704302", "variable_id": "9910", "period": "2022",
         "category_ids": [], "value_numeric": 3.0},
    ])
    pop = _facts([
        {"locality_id": "2704302", "variable_id": "9909", "period": "2022",
         "category_ids": ["59998"], "value_numeric": 900.0},
        {"locality_id": "2704302", "variable_id": "9612", "period": "2022",
         "category_ids": ["59998"], "value_numeric": 3000.0},
    ])
    spine = pl.DataFrame({"munic_code": ["2704302", "3550308"]})
    out = build_favela_profile(counts, pop, spine).sort("munic_code")
    assert out.height == 2
    assert out["has_favela"].to_list() == [True, False]
    # absent municipality gets zero counts, and the flag says why
    assert out["favela_households"].to_list() == [900.0, 0.0]


def test_profile_requires_spine_key() -> None:
    with pytest.raises(FavelaError, match="munic_code"):
        build_favela_profile(_facts([]), _facts([]), pl.DataFrame({"x": [1]}))


def test_inside_outside_carries_its_universe() -> None:
    facts = _facts([
        {"locality_id": "2704302", "variable_id": "13548", "period": "2022",
         "category_ids": ["72113"], "value_numeric": 400.0},
        {"locality_id": "2704302", "variable_id": "13549", "period": "2022",
         "category_ids": ["72113"], "value_numeric": 100.0},
    ])
    out = build_inside_outside_sewage(facts)
    assert out["inside_unsafe"].to_list() == [400.0]
    assert out["outside_unsafe"].to_list() == [100.0]
    assert "not municipal totals" in out["universe"][0]
