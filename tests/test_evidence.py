from pathlib import Path

import polars as pl

from brepi.qa.evidence import evaluate_assertion, evaluate_assertions


def test_table_assertions_are_value_checking(tmp_path: Path) -> None:
    pl.DataFrame({"group": ["a", "b"], "x": [1.0, 2.0], "y": [2.0, 4.0]}).write_csv(
        tmp_path / "result.csv")
    specs = [
        {"id": "n", "path": "result.csv", "kind": "row_count", "expected": 2},
        {"id": "width", "path": "result.csv", "kind": "column_count",
         "expected": 3},
        {"id": "cell", "path": "result.csv", "kind": "cell",
         "where": {"group": "b"}, "column": "x", "expected": 2.001,
         "atol": 0.01},
        {"id": "r", "path": "result.csv", "kind": "correlation",
         "columns": ["x", "y"], "expected": 1.0, "atol": 1e-12},
    ]
    audit = evaluate_assertions(tmp_path, specs)
    assert audit["passed"]
    assert audit["assertions_checked"] == 4


def test_column_sum_reconciles_a_derived_table_against_its_source(
        tmp_path: Path) -> None:
    """The assertion that catches a lossy join.

    An assembled table can lose rows to a key mismatch without any other symptom
    -- the file exists, the schema is right, the map merely has holes. A total
    that must equal the source's total is the cheapest way to see it.
    """
    pl.DataFrame({"unit": ["a", "b", "c"], "cases": [10, 0, 32]}).write_csv(
        tmp_path / "atlas.csv")
    ok = evaluate_assertion(tmp_path, {
        "id": "sum", "path": "atlas.csv", "kind": "column_sum",
        "column": "cases", "expected": 42})
    lossy = evaluate_assertion(tmp_path, {
        "id": "lossy", "path": "atlas.csv", "kind": "column_sum",
        "column": "cases", "expected": 74})
    assert ok["passed"] and ok["actual"] == 42
    assert not lossy["passed"]


def test_assertions_fail_closed_on_stale_or_ambiguous_values(tmp_path: Path) -> None:
    pl.DataFrame({"group": ["a", "a"], "value": [1, 2]}).write_csv(
        tmp_path / "result.csv")
    stale = evaluate_assertion(tmp_path, {
        "id": "stale", "path": "result.csv", "kind": "row_count", "expected": 3})
    ambiguous = evaluate_assertion(tmp_path, {
        "id": "ambiguous", "path": "result.csv", "kind": "cell",
        "where": {"group": "a"}, "column": "value", "expected": 1})
    missing = evaluate_assertion(tmp_path, {
        "id": "missing", "path": "gone.csv", "kind": "row_count", "expected": 0})
    assert not stale["passed"]
    assert not ambiguous["passed"] and "selected 2 rows" in ambiguous["error"]
    assert not missing["passed"]
