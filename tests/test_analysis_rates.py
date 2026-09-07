"""Tests for brepi.analysis.rates.

The values checked against are the standard exact intervals, not this module's
own output: an interval routine that agrees only with itself is a tautology.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from brepi.analysis import rates


def test_exact_poisson_matches_textbook_values():
    r, lo, hi = rates.poisson_ci([10], [1.0], scale=1.0)
    assert r[0] == pytest.approx(10.0)
    assert lo[0] == pytest.approx(4.795, abs=1e-3)
    assert hi[0] == pytest.approx(18.390, abs=1e-3)


def test_zero_count_gives_a_zero_lower_bound_not_nan():
    # gamma.ppf(a, 0) is undefined; a naive implementation propagates NaN into
    # every zero-count row, which in this study is 2,106 municipalities.
    r, lo, hi = rates.poisson_ci([0], [1.0], scale=1.0)
    assert r[0] == 0.0
    assert lo[0] == 0.0
    assert hi[0] == pytest.approx(-math.log(0.025), abs=1e-6)


def test_zero_person_time_is_nan_not_infinity():
    r, lo, hi = rates.poisson_ci([3], [0.0])
    assert np.isnan(r[0]) and np.isnan(lo[0]) and np.isnan(hi[0])


def test_scale_multiplies_the_whole_interval():
    a = rates.poisson_ci([7], [1000.0], scale=1.0)
    b = rates.poisson_ci([7], [1000.0], scale=1e5)
    for x, y in zip(a, b):
        assert y[0] == pytest.approx(x[0] * 1e5)


def test_person_time_of_length_one_is_recycled():
    r, _, _ = rates.poisson_ci([1, 2, 3], [10.0])
    assert r.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="person_time"):
        rates.poisson_ci([1, 2, 3], [1.0, 2.0])


def test_clopper_pearson_endpoints():
    p, lo, hi = rates.binom_ci([0, 10], [10, 10])
    assert lo[0] == 0.0 and hi[0] == pytest.approx(0.30850, abs=1e-4)
    assert lo[1] == pytest.approx(0.69150, abs=1e-4) and hi[1] == 1.0


def test_clopper_pearson_covers_the_point_estimate():
    p, lo, hi = rates.binom_ci([3], [17])
    assert lo[0] < p[0] < hi[0]
    assert p[0] == pytest.approx(3 / 17)


def test_zero_denominator_is_nan_not_a_proportion_of_zero():
    # "no cases with a known outcome" and "a case fatality of zero" are
    # different findings and must not share an encoding.
    p, lo, hi = rates.binom_ci([0], [0])
    assert np.isnan(p[0]) and np.isnan(lo[0]) and np.isnan(hi[0])


def test_count_exceeding_total_raises():
    with pytest.raises(ValueError, match="exceeds"):
        rates.binom_ci([5], [3])


def test_rate_table_attaches_three_columns():
    df = pl.DataFrame({"cases": [0, 5], "py": [1000.0, 1000.0]})
    out = rates.rate_table(df, count="cases", person_time="py", prefix="inc")
    assert out.columns == ["cases", "py", "inc", "inc_lo", "inc_hi"]
    assert out["inc"].to_list() == pytest.approx([0.0, 500.0])


def test_proportion_table_attaches_three_columns():
    df = pl.DataFrame({"d": [1], "n": [4]})
    out = rates.proportion_table(df, count="d", total="n", prefix="cfr")
    assert out["cfr"][0] == pytest.approx(0.25)
    assert out["cfr_lo"][0] < 0.25 < out["cfr_hi"][0]
