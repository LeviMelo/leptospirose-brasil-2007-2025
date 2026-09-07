"""Tests for brepi.qa.vocabulary.

The failure this module exists to catch is silent by construction: two files
name the same concept in two vocabularies, both individually correct, and a
join between them returns nothing. Nothing errors. A figure comes out empty or,
worse, half-populated from a partial match.

It happened here. The canonical municipality-month panel carried IBGE's
Portuguese macro-regions while the descriptive tables carried English ones, and
fourteen result files split into two camps that could not be joined.
"""
from __future__ import annotations

import polars as pl
import pytest

from brepi.qa import vocabulary as voc


@pytest.fixture()
def results(tmp_path):
    root = tmp_path / "results"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    return root


def _csv(path, **cols):
    pl.DataFrame(cols).write_csv(path)


def test_agreeing_files_pass(results):
    _csv(results / "a" / "x.csv", region=["Norte", "Sul"], n=[1, 2])
    _csv(results / "b" / "y.csv", region=["Norte", "Sul"], v=[3.0, 4.0])
    rep = voc.check_vocabularies(results, ["region"])["region"]
    assert rep.n_files == 2
    assert rep.passed
    assert not rep.disjoint_pairs


def test_disjoint_vocabularies_fail_and_are_named(results):
    # The real bug: both correct, no shared value, join returns nothing.
    _csv(results / "a" / "panel.csv", region=["Norte", "Sudeste"], n=[1, 2])
    _csv(results / "b" / "table.csv", region=["North", "Southeast"], v=[3.0, 4.0])
    rep = voc.check_vocabularies(results, ["region"])["region"]
    assert not rep.passed
    assert len(rep.disjoint_pairs) == 1
    a, b = rep.disjoint_pairs[0]
    assert {a, b} == {"a\\panel.csv", "b\\table.csv"} or {a, b} == {
        "a/panel.csv", "b/table.csv"}
    text = voc.format_report({"region": rep})
    assert "DISJOINT" in text
    assert "Norte" in text and "North" in text


def test_partial_overlap_is_reported_but_does_not_fail(results):
    # A legitimate subset -- not every state has a detection-silent municipality
    # -- must not read as a failure.
    _csv(results / "a" / "all.csv", uf=["SP", "RJ", "MG"], n=[1, 2, 3])
    _csv(results / "b" / "subset.csv", uf=["SP", "RJ"], v=[1.0, 2.0])
    rep = voc.check_vocabularies(results, ["uf"])["uf"]
    assert rep.passed
    assert len(rep.partial_pairs) == 1
    assert "partial overlap" in voc.format_report({"uf": rep})


def test_identifier_columns_are_not_treated_as_vocabularies(results):
    # A column with hundreds of distinct values is a key, not a codelist.
    # Comparing key sets across files is meaningless and would flag every
    # legitimately-different subset.
    _csv(results / "a" / "one.csv", code=[str(i) for i in range(100)])
    _csv(results / "b" / "two.csv", code=[str(i) for i in range(200, 300)])
    rep = voc.check_vocabularies(results, ["code"], max_distinct=60)["code"]
    assert rep.n_files == 0
    assert rep.passed


def test_absent_column_is_simply_skipped(results):
    _csv(results / "a" / "x.csv", region=["Norte"], n=[1])
    _csv(results / "b" / "y.csv", other=["z"], n=[1])
    rep = voc.check_vocabularies(results, ["region"])["region"]
    assert rep.n_files == 1
    assert rep.passed


def test_reads_csvs_whose_dtype_widens_late(results, tmp_path):
    """Result CSVs often start integer-looking and later contain floats.

    Default schema inference samples the head and then raises on the first
    float. A checker that throws on a real result file is a checker nobody
    runs, so inference reads the whole file.
    """
    p = results / "a" / "widening.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = ["region,value"] + [f"Norte,{i}" for i in range(200)] + ["Sul,1.5"]
    p.write_text("\n".join(rows), encoding="utf-8")
    rep = voc.check_vocabularies(results, ["region"])["region"]
    assert rep.n_files == 1
    assert rep.all_values == {"Norte", "Sul"}


def test_report_serialises_for_the_audit(results):
    _csv(results / "a" / "x.csv", region=["Norte", "Sul"])
    _csv(results / "b" / "y.csv", region=["North", "South"])
    d = voc.check_vocabularies(results, ["region"])["region"].to_dict()
    assert d["column"] == "region"
    assert d["passed"] is False
    assert d["files"] == 2
    assert set(d["distinct_values_overall"]) == {"Norte", "Sul", "North", "South"}
    assert len(d["disjoint_pairs"]) == 1
