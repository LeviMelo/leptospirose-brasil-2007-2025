"""Tests for brepi.panel.aggregate.

The central case is the RQ2 defect: a monthly ``asinh(cases)`` summed to
quarters. The sum of a transform is not the transform of a sum, and no error
was raised at the time -- the estimate simply meant nothing. These tests fix
that the refusal now happens.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from brepi.panel.aggregate import aggregate_panel, is_non_additive


@pytest.fixture()
def monthly():
    return pl.DataFrame({
        "unit": ["a"] * 6 + ["b"] * 6,
        "quarter": [1, 1, 1, 2, 2, 2] * 2,
        "cases": [1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 0, 0],
        "person_months": [100.0] * 12,
        "region": ["Norte"] * 6 + ["Sul"] * 6,
        "cases_asinh": [math.asinh(x) for x in
                        [1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 0, 0]],
    })


def test_summing_an_asinh_outcome_is_refused(monthly):
    with pytest.raises(ValueError, match="non-additive"):
        aggregate_panel(monthly, keys=["unit", "quarter"],
                        sums=["cases", "cases_asinh"])


def test_the_correct_route_sums_counts_then_transforms(monthly):
    out, rep = aggregate_panel(
        monthly, keys=["unit", "quarter"],
        sums=["cases", "person_months"], constants=["region"],
        derived={"cases_asinh": pl.col("cases").cast(pl.Float64).arcsinh(),
                 "rate": pl.col("cases") / pl.col("person_months")})
    assert rep.passed
    assert out.height == 4
    a1 = out.filter((pl.col("unit") == "a") & (pl.col("quarter") == 1))
    assert a1["cases"][0] == 6
    # asinh(6), not asinh(1)+asinh(2)+asinh(3) == 3.72
    assert a1["cases_asinh"][0] == pytest.approx(math.asinh(6))
    assert a1["cases_asinh"][0] != pytest.approx(
        sum(math.asinh(x) for x in (1, 2, 3)))


def test_various_non_additive_names_are_recognised():
    for n in ["cfr_share", "incidence_per_100k", "gdp_per_capita_asinh",
              "completeness_z", "log_pop", "urban_fraction", "hosp_rate",
              "median_delay", "sanitation_sewer_share"]:
        assert is_non_additive(n), n
    for n in ["cases", "deaths", "person_months", "hospitalised", "n_events"]:
        assert not is_non_additive(n), n


def test_the_override_is_explicit_and_works(monthly):
    df = monthly.rename({"cases_asinh": "n_log_entries"})
    out, _ = aggregate_panel(df, keys=["unit"], sums=["n_log_entries"],
                             allow_non_additive_sums=["n_log_entries"])
    assert out.height == 2


def test_a_constant_that_is_not_constant_is_reported(monthly):
    bad = monthly.with_columns(
        pl.Series("region", ["Norte", "Sul"] * 6))
    _, rep = aggregate_panel(bad, keys=["unit"], sums=["cases"],
                             constants=["region"])
    assert not rep.passed
    assert rep.varying_constants["region"] == 2
    assert "declared constant but varies" in rep.format()


def test_a_genuine_constant_passes_and_the_check_column_is_removed(monthly):
    out, rep = aggregate_panel(monthly, keys=["unit"], sums=["cases"],
                               constants=["region"])
    assert rep.passed
    assert not [c for c in out.columns if c.startswith("__nu_")]
    assert out.sort("unit")["region"].to_list() == ["Norte", "Sul"]


def test_weighted_mean_requires_and_uses_its_weight():
    df = pl.DataFrame({"u": ["a", "a"], "share": [0.0, 1.0],
                       "pop": [1.0, 9.0]})
    out, _ = aggregate_panel(df, keys=["u"], weighted_means={"share": "pop"})
    assert out["share"][0] == pytest.approx(0.9)


def test_maxima_and_minima_are_suffixed(monthly):
    out, _ = aggregate_panel(monthly, keys=["unit"], sums=["cases"],
                             maxima=["cases"], minima=["cases"])
    assert "cases_max" in out.columns and "cases_min" in out.columns
    assert out.sort("unit")["cases_max"].to_list() == [6, 2]


def test_absent_column_raises_by_name(monthly):
    with pytest.raises(KeyError, match="nope"):
        aggregate_panel(monthly, keys=["unit"], sums=["nope"])


def test_lazyframe_input_is_accepted(monthly):
    out, rep = aggregate_panel(monthly.lazy(), keys=["unit"], sums=["cases"])
    assert rep.rows_in == 12 and out.height == 2


def test_extra_expressions_run_inside_the_group(monthly):
    out, _ = aggregate_panel(
        monthly, keys=["unit"], sums=["cases"],
        extra={"months_with_cases": (pl.col("cases") > 0).sum()})
    assert out.sort("unit")["months_with_cases"].to_list() == [6, 2]


def test_report_serialises(monthly):
    _, rep = aggregate_panel(monthly, keys=["unit"], sums=["cases"],
                             constants=["region"],
                             derived={"x": pl.col("cases") * 2})
    d = rep.to_dict()
    assert d["rows_in"] == 12 and d["rows_out"] == 2
    assert d["derived"] == ["x"] and d["passed"] is True
