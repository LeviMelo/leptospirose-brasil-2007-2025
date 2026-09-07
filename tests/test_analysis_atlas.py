"""Tests for brepi.analysis.atlas.

Every case here is a failure that occurred, or nearly occurred, while assembling
this study's unit-level tables. They are silent failures: none of them raises in
plain polars, and each produces a table that looks fine.
"""
from __future__ import annotations

import polars as pl
import pytest

from brepi.analysis.atlas import Layer, assemble_atlas, normalise_key


@pytest.fixture()
def spine():
    return pl.DataFrame({"munic_code": ["0100015", "0100023", "0100031"],
                         "name": ["A", "B", "C"]})


def test_integer_and_padded_string_keys_are_reconciled(spine):
    # The canonical panel stores a zero-padded string; a CSV that round-tripped
    # through schema inference stores an integer that lost the leading zero.
    # Joined as they stand, these match nothing at all.
    layer = pl.DataFrame({"munic_code": [100015, 100023], "cases": [4, 9]})
    atlas, rep = assemble_atlas(spine, [Layer("l", layer)],
                                key="munic_code", key_width=7)
    assert rep.layers[0].matched == 2
    assert atlas["cases"].to_list() == [4, 9, None]


def test_without_key_width_the_same_join_silently_finds_nothing(spine):
    # The negative control: this is what the caller gets for free, and why
    # key_width exists.
    layer = pl.DataFrame({"munic_code": [100015, 100023], "cases": [4, 9]})
    _, rep = assemble_atlas(spine, [Layer("l", layer)], key="munic_code")
    assert rep.layers[0].matched == 0
    assert rep.layers[0].layer_only == 2


def test_layer_rows_not_in_the_spine_are_reported(spine):
    layer = pl.DataFrame({"munic_code": ["0100015", "9999999"], "v": [1, 2]})
    _, rep = assemble_atlas(spine, [Layer("l", layer)], key="munic_code")
    assert rep.layers[0].layer_only == 1
    assert rep.layers[0].layer_only_examples == ["9999999"]
    assert "NOT IN SPINE" in rep.format()


def test_column_collision_raises_rather_than_producing_name_right(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "name": ["different"]})
    with pytest.raises(ValueError, match="overwrite"):
        assemble_atlas(spine, [Layer("l", layer)], key="munic_code")


def test_prefix_resolves_a_collision(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "name": ["different"]})
    atlas, _ = assemble_atlas(spine, [Layer("l", layer, prefix="rq3_")],
                              key="munic_code")
    assert atlas["name"].to_list() == ["A", "B", "C"]
    assert atlas["rq3_name"].to_list() == ["different", None, None]


def test_duplicate_keys_in_a_layer_raise_instead_of_multiplying_the_spine(spine):
    layer = pl.DataFrame({"munic_code": ["0100015", "0100015"], "v": [1, 2]})
    with pytest.raises(ValueError, match="duplicated key"):
        assemble_atlas(spine, [Layer("l", layer)], key="munic_code")


def test_non_unique_spine_is_rejected():
    bad = pl.DataFrame({"k": ["a", "a"], "v": [1, 2]})
    with pytest.raises(ValueError, match="not unique"):
        assemble_atlas(bad, [], key="k")


def test_complete_layer_that_misses_spine_rows_raises(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "v": [1]})
    with pytest.raises(ValueError, match="declared complete"):
        assemble_atlas(spine, [Layer("l", layer, complete=True)], key="munic_code")


def test_incomplete_layer_is_fine_when_not_declared_complete(spine):
    # RQ5 assigns a regime to 497 of 5,570 municipalities. That is the finding,
    # not a defect, and must not read as one.
    layer = pl.DataFrame({"munic_code": ["0100015"], "regime": [3]})
    atlas, rep = assemble_atlas(spine, [Layer("l", layer)], key="munic_code")
    assert rep.passed
    assert atlas["regime"].to_list() == [3, None, None]


def test_spine_row_set_and_order_are_preserved(spine):
    layer = pl.DataFrame({"munic_code": ["0100031", "0100015"], "v": [9, 8]})
    atlas, _ = assemble_atlas(spine, [Layer("l", layer)], key="munic_code")
    assert atlas["munic_code"].to_list() == spine["munic_code"].to_list()
    assert atlas["v"].to_list() == [8, None, 9]


def test_columns_selects_and_rename_applies(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "a": [1], "b": [2]})
    atlas, _ = assemble_atlas(spine, [Layer("l", layer, columns=["a"],
                                            rename={"a": "alpha"})],
                              key="munic_code")
    assert "b" not in atlas.columns
    assert atlas["alpha"].to_list() == [1, None, None]


def test_missing_requested_column_raises(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "a": [1]})
    with pytest.raises(KeyError, match="lacks column"):
        assemble_atlas(spine, [Layer("l", layer, columns=["nope"])],
                       key="munic_code")


def test_strict_false_records_problems_instead_of_raising(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "name": ["x"]})
    atlas, rep = assemble_atlas(spine, [Layer("l", layer)], key="munic_code",
                                strict=False)
    assert not rep.passed
    assert atlas["name"].to_list() == ["A", "B", "C"]


def test_normalise_key_pads_and_reports_a_missing_key():
    df = pl.DataFrame({"k": [15, 23]})
    assert normalise_key(df, "k", width=7)["k"].to_list() == ["0000015", "0000023"]
    with pytest.raises(KeyError):
        normalise_key(df, "absent", width=None)


def test_report_serialises(spine):
    layer = pl.DataFrame({"munic_code": ["0100015"], "v": [1]})
    _, rep = assemble_atlas(spine, [Layer("l", layer)], key="munic_code")
    d = rep.to_dict()
    assert d["spine_rows"] == 3 and d["passed"] is True
    assert d["layers"][0]["matched_keys"] == 1
