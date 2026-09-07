"""Tests for the denominator tensor math.

Run: C:/Users/Galaxy/miniconda3/envs/pegasus/python.exe -m pytest brepi/tests -q
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from brepi.denominators.raking import (
    alr,
    alr_inv,
    cohort_shift_index,
    interpolate_composition,
    rake,
)
from brepi.denominators.rates import (
    RACE_MEASUREMENT_CAVEAT,
    age_standardised_rate,
    crude_rate,
    stratified_rate,
)
from brepi.denominators.tensor import TensorSpec, build_tensor


# ---------------------------------------------------------------- raking ---


def test_rake_recovers_known_table():
    """IPF must return the original table when raked to its own margins."""
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 200, size=(4, 5)).astype(float)
    res = rake(truth.copy(), {(0,): truth.sum(1), (1,): truth.sum(0)})
    assert res.converged
    np.testing.assert_allclose(res.table, truth, rtol=1e-8)
    assert res.kl_from_seed == pytest.approx(0.0, abs=1e-8)


def test_rake_preserves_odds_ratios():
    """The defining property: raking changes margins, never interaction."""
    seed = np.array([[10.0, 20.0], [30.0, 15.0]])
    or_seed = (seed[0, 0] * seed[1, 1]) / (seed[0, 1] * seed[1, 0])
    res = rake(seed, {(0,): np.array([100.0, 400.0]), (1,): np.array([200.0, 300.0])})
    t = res.table
    or_fit = (t[0, 0] * t[1, 1]) / (t[0, 1] * t[1, 0])
    assert or_fit == pytest.approx(or_seed, rel=1e-9)
    np.testing.assert_allclose(t.sum(1), [100.0, 400.0], rtol=1e-9)
    np.testing.assert_allclose(t.sum(0), [200.0, 300.0], rtol=1e-9)


def test_rake_keeps_structural_zeros():
    seed = np.array([[5.0, 0.0], [3.0, 2.0]])
    res = rake(seed, {(0,): np.array([50.0, 50.0])})
    assert res.table[0, 1] == 0.0


def test_rake_rejects_inconsistent_margins():
    seed = np.ones((2, 2))
    with pytest.raises(ValueError, match="inconsistent margins"):
        rake(seed, {(0,): np.array([10.0, 10.0]), (1,): np.array([5.0, 5.0])})


def test_rake_rejects_margin_needing_structural_zero():
    seed = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="structurally zero"):
        rake(seed, {(0,): np.array([1.0, 1.0])})


# ------------------------------------------------------- compositional ---


def test_alr_roundtrip():
    s = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(alr_inv(alr(s)), s, rtol=1e-10)


def test_interpolation_endpoints_and_simplex():
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.3, 0.6]])
    np.testing.assert_allclose(interpolate_composition(a, b, 0.0), a, rtol=1e-9)
    np.testing.assert_allclose(interpolate_composition(a, b, 1.0), b, rtol=1e-9)
    mid = interpolate_composition(a, b, 0.5)
    assert mid.sum() == pytest.approx(1.0)
    assert (mid > 0).all()


def test_interpolation_is_scale_invariant_in_ratios():
    """Geometric, not arithmetic: the midpoint of 0.001 and 0.1 is ~0.01."""
    a = np.array([[0.001, 0.999]])
    b = np.array([[0.100, 0.900]])
    mid = interpolate_composition(a, b, 0.5)
    assert 0.008 < mid[0, 0] < 0.012  # arithmetic would give ~0.0505


def test_cohort_shift():
    idx = cohort_shift_index(n_ages=5, years=10, width=5)
    # A 10-year shift moves each cohort two 5-year groups up.
    assert idx.tolist() == [0, 0, 0, 1, 2]


# ---------------------------------------------------------------- tensor ---


def _fixtures():
    muns = ("3550308", "1200401")
    years = (2010, 2016, 2022)
    ages = ("0-4", "5-9", "10-14")
    spec = TensorSpec(
        years=years, municipalities=muns, sexes=("M", "F"), ages=ages, colours=None
    )
    totals = pl.DataFrame(
        {
            "munic_code": [m for m in muns for _ in years],
            "year": list(years) * 2,
            "population": [1000.0, 1200.0, 1500.0, 400.0, 450.0, 500.0],
        }
    )
    rows = []
    for m in muns:
        for cy, skew in ((2010, 1.0), (2022, 2.0)):
            for si, s in enumerate(("M", "F")):
                for ai, a in enumerate(ages):
                    rows.append(
                        {
                            "munic_code": m,
                            "census_year": cy,
                            "sex": s,
                            "age_group": a,
                            "count": 100.0 * (1 + ai * skew) * (1.1 if si == 0 else 1.0),
                        }
                    )
    return spec, totals, pl.DataFrame(rows)


def test_tensor_matches_official_totals_exactly():
    spec, totals, comp = _fixtures()
    t = build_tensor(spec, totals=totals, census_composition=comp)
    for m in spec.municipalities:
        for y in spec.years:
            expected = totals.filter(
                (pl.col("munic_code") == m) & (pl.col("year") == y)
            )["population"][0]
            assert t.total(municipality=m, year=y) == pytest.approx(expected, rel=1e-9)


def test_tensor_flags_interpolation_method_and_distance():
    spec, totals, comp = _fixtures()
    t = build_tensor(spec, totals=totals, census_composition=comp)
    long = t.to_long()
    got = {
        (r["year"], r["method"], r["years_from_census"])
        for r in long.select("year", "method", "years_from_census").unique().iter_rows(named=True)
    }
    assert (2010, "census_observed", 0) in got
    assert (2022, "census_observed", 0) in got
    assert any(y == 2016 and m == "cohort_interpolated" for y, m, _ in got)


def test_tensor_rejects_missing_total():
    spec, totals, comp = _fixtures()
    with pytest.raises(ValueError, match="no official total"):
        build_tensor(spec, totals=totals.head(3), census_composition=comp)


def test_tensor_long_roundtrip_sums():
    spec, totals, comp = _fixtures()
    t = build_tensor(spec, totals=totals, census_composition=comp)
    assert t.to_long()["population"].sum() == pytest.approx(totals["population"].sum())
    assert t.collapse(["municipality", "year"]).height == len(spec.municipalities) * len(spec.years)


# ----------------------------------------------------------------- rates ---


def test_crude_rate_keeps_structural_zeros():
    den = pl.DataFrame({"munic_code": ["a", "b"], "population": [1000.0, 2000.0]})
    num = pl.DataFrame({"munic_code": ["a"], "cases": [5]})
    out = crude_rate(num, den, by=["munic_code"])
    assert out.height == 2
    assert out.filter(pl.col("munic_code") == "b")["cases"][0] == 0
    assert out.filter(pl.col("munic_code") == "a")["rate"][0] == pytest.approx(500.0)


def test_age_standardised_rate_equals_crude_when_structure_matches_standard():
    from brepi.denominators.rates import WHO_WORLD_STANDARD

    ages = list(WHO_WORLD_STANDARD)
    den = pl.DataFrame(
        {
            "grp": ["x"] * len(ages),
            "age_group": ages,
            "population": [WHO_WORLD_STANDARD[a] * 1_000_000 for a in ages],
        }
    )
    num = pl.DataFrame(
        {"grp": ["x"] * len(ages), "age_group": ages, "cases": [10] * len(ages)}
    )
    asr = age_standardised_rate(num, den, by=["grp"])["asr"][0]
    crude = crude_rate(num, den, by=["grp"])["rate"][0]
    assert asr == pytest.approx(crude, rel=0.02)


def test_colour_stratification_requires_acknowledgement():
    den = pl.DataFrame({"colour": ["branca"], "population": [100.0]})
    num = pl.DataFrame({"colour": ["branca"], "cases": [1]})
    with pytest.raises(ValueError, match="acknowledge_race_measurement_gap"):
        stratified_rate(num, den, by=["colour"])
    res = stratified_rate(num, den, by=["colour"], acknowledge_race_measurement_gap=True)
    assert RACE_MEASUREMENT_CAVEAT in res.caveats
    assert "measurement_caveat" in res.frame.columns
