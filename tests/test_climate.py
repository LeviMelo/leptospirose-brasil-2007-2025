"""Unit tests for the pure climate-index functions.

Every expected value below is computed by hand from the synthetic series, not
by running the implementation. A test that asserts the code agrees with itself
is worth nothing, and index definitions (RX5day's window, CDD's treatment of
missing days, SPI's zero-inflation term) are exactly where an implementation
drifts from the standard without anybody noticing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from brepi.sources.climate import brdwgd, indices
from brepi.sources.climate.brdwgd import (
    PRODUCTS,
    ClimateProduct,
    ClimateSourceError,
    available_products,
    daily_municipal,
    monthly_municipal,
)

# A 31-day January designed so that every index has a different answer.
# index:      0  1   2   3  4  5  6   7   8   9 10 11 12 13 14 15
JAN = [
    0.0, 0.0, 0.0, 0.0, 0.0,      # 5 dry days
    2.0, 12.0, 25.0, 60.0, 30.0,  # a 5-day wet spell, 129 mm, peak 60
    0.0, 0.0,                     # 2 dry
    0.5, 0.9, 0.0,                # sub-threshold drizzle: dry by ETCCDI
    15.0,                         # 1 wet day
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 7 dry days -- the longest dry run
    1.0, 1.0, 1.0,                # 3-day wet spell at exactly the threshold
    0.0,
    55.0, 5.0, 0.0, 0.0,          # a late extreme day
]
assert len(JAN) == 31


class TestPrecipitationIndices:
    def test_total(self):
        # 2+12+25+60+30 = 129; +0.5+0.9 = 130.4; +15 = 145.4; +3 = 148.4;
        # +55+5 = 208.4
        assert indices.total_precipitation(JAN) == pytest.approx(208.4)

    def test_total_rejects_negative(self):
        with pytest.raises(ValueError):
            indices.total_precipitation([1.0, -0.1, 2.0])

    def test_total_skips_nan_rather_than_zero_filling(self):
        assert indices.total_precipitation([1.0, np.nan, 2.0]) == pytest.approx(3.0)

    def test_wet_day_counts(self):
        # >=1mm: 2,12,25,60,30,15,1,1,1,55,5  -> 11.  0.5 and 0.9 are NOT wet.
        assert indices.wet_days(JAN, 1.0) == 11
        # >=10mm: 12,25,60,30,15,55 -> 6
        assert indices.wet_days(JAN, 10.0) == 6
        # >=20mm: 25,60,30,55 -> 4
        assert indices.wet_days(JAN, 20.0) == 4
        # >=50mm: 60,55 -> 2
        assert indices.wet_days(JAN, 50.0) == 2

    def test_sub_threshold_drizzle_is_not_a_wet_day(self):
        assert indices.wet_days([0.5, 0.9, 0.99], 1.0) == 0
        assert indices.wet_days([1.0], 1.0) == 1  # threshold is inclusive

    def test_rx1day(self):
        assert indices.rx1day(JAN) == pytest.approx(60.0)

    def test_rx5day(self):
        # The wet spell at indices 5..9 sums to 2+12+25+60+30 = 129, and no
        # other 5-day window beats it (the late spell is 55+5 = 60).
        assert indices.rxnday(JAN, 5) == pytest.approx(129.0)

    def test_rx5day_window_is_honoured(self):
        assert indices.rxnday([1, 2, 3, 4, 5, 6], 3) == pytest.approx(15.0)  # 4+5+6
        assert indices.rxnday([1, 2, 3, 4, 5, 6], 1) == pytest.approx(6.0)

    def test_rx5day_undefined_on_short_series(self):
        assert np.isnan(indices.rxnday([1.0, 2.0], 5))

    def test_cwd(self):
        # Longest wet run is indices 5..9 -> 5 days. The 1,1,1 run is 3.
        assert indices.consecutive_wet_days(JAN) == 5

    def test_cdd(self):
        # Longest dry run: indices 17..23 is 7 days, but 12,13,14 (0.5, 0.9,
        # 0.0) are all sub-threshold and follow 10,11 -> that run is
        # 10,11,12,13,14 = 5. The 7-day run wins.
        assert indices.consecutive_dry_days(JAN) == 7

    def test_cwd_cdd_partition_a_simple_series(self):
        series = [0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0]
        assert indices.consecutive_wet_days(series) == 2
        assert indices.consecutive_dry_days(series) == 3

    def test_nan_breaks_a_run_rather_than_extending_it(self):
        # An unobserved day is not evidence of rain, nor of drought.
        assert indices.consecutive_wet_days([5.0, 5.0, np.nan, 5.0, 5.0]) == 2
        assert indices.consecutive_dry_days([0.0, 0.0, np.nan, 0.0, 0.0]) == 2

    def test_sdii(self):
        # 208.4 total minus the 1.4 of sub-threshold drizzle = 207.0 on 11 wet days
        assert indices.sdii(JAN) == pytest.approx(207.0 / 11)

    def test_sdii_undefined_without_wet_days(self):
        assert np.isnan(indices.sdii([0.0, 0.2, 0.0]))

    def test_all_dry_month(self):
        dry = [0.0] * 30
        assert indices.total_precipitation(dry) == 0.0
        assert indices.wet_days(dry) == 0
        assert indices.consecutive_dry_days(dry) == 30
        assert indices.consecutive_wet_days(dry) == 0
        assert indices.rx1day(dry) == 0.0


class TestTemperatureIndices:
    def test_mean_max_min(self):
        tmax = [30.0, 32.0, 28.0, np.nan, 34.0]
        assert indices.mean_ignoring_nan(tmax) == pytest.approx(31.0)
        assert indices.max_ignoring_nan(tmax) == pytest.approx(34.0)
        assert indices.min_ignoring_nan(tmax) == pytest.approx(28.0)

    def test_all_nan_is_nan_not_zero(self):
        assert np.isnan(indices.mean_ignoring_nan([np.nan, np.nan]))


class TestStandardisedAnomaly:
    def test_known_z_score(self):
        clim = [10.0] * 5 + [20.0] * 5  # mean 15, sd (ddof=1) = 5.2705...
        sd = float(np.std(clim, ddof=1))
        assert indices.standardised_anomaly(15.0 + 2 * sd, clim) == pytest.approx(2.0)
        assert indices.standardised_anomaly(15.0, clim) == pytest.approx(0.0)

    def test_refuses_a_short_climatology(self):
        with pytest.raises(indices.IndexError_):
            indices.standardised_anomaly(100.0, [1.0, 2.0, 3.0])

    def test_zero_variance_is_nan_not_infinity(self):
        assert np.isnan(indices.standardised_anomaly(5.0, [3.0] * 30))


class TestSPI:
    #: A long synthetic monthly total, right-skewed like real rainfall. Longer
    #: than a real climatology on purpose: this fixture tests the estimator,
    #: not the sampling error of a 30-year record.
    CLIM = list(np.random.default_rng(11).gamma(shape=2.0, scale=60.0, size=4000))

    def test_median_maps_near_zero(self):
        # SPI is the normal quantile of the non-exceedance probability, so the
        # median of the fitted distribution must map to 0.
        median = float(np.median(self.CLIM))
        assert abs(indices.spi(median, self.CLIM)) < 0.1

    def test_monotone_in_the_value(self):
        values = [10.0, 50.0, 100.0, 200.0, 400.0]
        out = [indices.spi(v, self.CLIM) for v in values]
        assert out == sorted(out)

    def test_extreme_wet_is_strongly_positive(self):
        assert indices.spi(max(self.CLIM) * 3, self.CLIM) > 2.0

    def test_spi_is_not_a_z_score_of_raw_totals(self):
        # On a skewed distribution the gamma-based SPI and the naive z-score
        # must differ; if they agree the gamma fit has been short-circuited.
        value = float(np.percentile(self.CLIM, 95))
        z = indices.standardised_anomaly(value, self.CLIM)
        assert abs(indices.spi(value, self.CLIM) - z) > 0.1

    def test_refuses_a_short_record(self):
        with pytest.raises(indices.IndexError_):
            indices.spi(100.0, self.CLIM[:15])

    def test_rejects_negative_precipitation(self):
        with pytest.raises(ValueError):
            indices.spi(-1.0, self.CLIM)

    def test_spi_series_leaves_incomplete_accumulations_null(self):
        rng = np.random.default_rng(3)
        totals = rng.gamma(2.0, 60.0, size=12 * 40)
        months = [(i % 12) + 1 for i in range(totals.size)]
        out = indices.spi_series(totals, months, accumulation=3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert np.isfinite(out[2])
        assert out.size == totals.size

    def test_spi3_differs_from_spi1(self):
        rng = np.random.default_rng(4)
        totals = rng.gamma(2.0, 60.0, size=12 * 40)
        months = [(i % 12) + 1 for i in range(totals.size)]
        one = indices.spi_series(totals, months, accumulation=1)
        three = indices.spi_series(totals, months, accumulation=3)
        assert not np.allclose(one[5:], three[5:], equal_nan=True)


class TestDailyOnlyRefusal:
    """The module must not fabricate daily-resolution indices."""

    @pytest.mark.parametrize(
        "name", ["rx1day", "rx5day", "cwd", "cdd", "r1", "r10", "r20", "r50", "sdii"]
    )
    def test_named_as_impossible_from_monthly(self, name):
        assert name in indices.DAILY_ONLY_INDICES

    @pytest.mark.parametrize("name", ["rx1day", "cwd", "cdd", "r10"])
    def test_monthly_path_raises(self, name):
        monthly = pl.DataFrame(
            {"munic_code": ["3550308"], "period": [date(2024, 1, 1)], "pr_total": [200.0]}
        )
        with pytest.raises(indices.IndexError_, match="cannot be computed"):
            indices.monthly_indices_from_monthly(monthly, requested=[name])

    def test_monthly_path_allows_what_is_legitimate(self):
        monthly = pl.DataFrame(
            {"munic_code": ["3550308"], "period": [date(2024, 1, 1)], "pr_total": [200.0]}
        )
        out = indices.monthly_indices_from_monthly(monthly, requested=["pr_total"])
        assert out.columns == ["munic_code", "period", "pr_total"]

    def test_monthly_product_cannot_be_asked_for_daily_series(self):
        with pytest.raises(ClimateSourceError, match="not daily"):
            daily_municipal(
                ["3550308"],
                ["pr"],
                start=date(2015, 1, 1),
                end=date(2015, 1, 31),
                product="terraclimate_cdn",
            )

    def test_the_two_index_sets_are_disjoint(self):
        assert not (indices.DAILY_ONLY_INDICES & indices.MONTHLY_SAFE_INDICES)


class TestMonthlyAssembly:
    def _daily_frame(self, values, code="3550308", year=2024, month=1):
        n = len(values)
        return pl.DataFrame(
            {
                "munic_code": [code] * n,
                "date": [date(year, month, d + 1) for d in range(n)],
                "variable": ["pr"] * n,
                "stat": ["mean"] * n,
                "value": [float(v) for v in values],
                "product": ["synthetic"] * n,
            }
        )

    def test_indices_match_the_pure_functions(self):
        out = indices.monthly_indices_from_daily(self._daily_frame(JAN), variables=["pr"])
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["pr_total"] == pytest.approx(208.4)
        assert row["r1"] == 11
        assert row["r10"] == 6
        assert row["r20"] == 4
        assert row["r50"] == 2
        assert row["rx1day"] == pytest.approx(60.0)
        assert row["rx5day"] == pytest.approx(129.0)
        assert row["cwd"] == 5
        assert row["cdd"] == 7
        assert row["n_days"] == 31
        assert row["n_days_expected"] == 31

    def test_incomplete_month_raises(self):
        with pytest.raises(ValueError, match="incomplete"):
            indices.monthly_indices_from_daily(
                self._daily_frame(JAN[:20]), variables=["pr"]
            )

    def test_rx5day_spans_the_month_boundary(self):
        # 40 mm on each of Jan 30, 31 and Feb 1, 2, 3: a 200 mm five-day event
        # that is credited to February, not split 80/120.
        jan = [0.0] * 29 + [40.0, 40.0]
        feb = [40.0, 40.0, 40.0] + [0.0] * 26
        frame = pl.concat(
            [self._daily_frame(jan, month=1), self._daily_frame(feb, month=2)]
        )
        out = indices.monthly_indices_from_daily(frame, variables=["pr"]).sort("period")
        feb_row = out.filter(pl.col("period") == date(2024, 2, 1)).row(0, named=True)
        assert feb_row["rx5day"] == pytest.approx(200.0)

    def test_anomaly_uses_same_calendar_month_climatology(self):
        rows = []
        for year in range(1991, 2025):
            for month, value in ((1, 300.0), (7, 30.0)):
                rows.append(
                    {
                        "munic_code": "3550308",
                        "period": date(year, month, 1),
                        "pr_total": value + (10.0 if year % 2 else -10.0),
                    }
                )
        rows.append({"munic_code": "3550308", "period": date(2024, 5, 1), "pr_total": 300.0})
        monthly = pl.DataFrame(rows).sort("period")
        out = indices.attach_anomalies(monthly)
        # January 2024 sits at its own January mean, not at the pooled mean of
        # January and July -- a pooled climatology would score it near +1.4.
        jan24 = out.filter(pl.col("period") == date(2024, 1, 1)).row(0, named=True)
        assert abs(jan24["pr_total_anom_z"]) == pytest.approx(1.0, abs=0.05)

    def test_anomaly_is_null_on_a_short_base(self):
        monthly = pl.DataFrame(
            {
                "munic_code": ["1200401"] * 5,
                "period": [date(y, 1, 1) for y in range(2019, 2024)],
                "pr_total": [100.0, 120.0, 90.0, 110.0, 105.0],
            }
        )
        out = indices.attach_anomalies(monthly)
        assert out["pr_total_anom_z"].null_count() == 5

    def test_spi_refuses_a_gapped_series(self):
        monthly = pl.DataFrame(
            {
                "munic_code": ["3550308"] * 3,
                "period": [date(2024, 1, 1), date(2024, 3, 1), date(2024, 4, 1)],
                "pr_total": [100.0, 120.0, 90.0],
            }
        )
        with pytest.raises(ValueError, match="jumps"):
            indices.attach_spi(monthly)


class TestProductRegistry:
    def test_coverage_windows_are_declared_and_ordered(self):
        for product in PRODUCTS.values():
            assert product.start < product.end

    def test_no_single_product_spans_the_study_window(self):
        # If this ever fails, a product has been extended and the splice in
        # DEFAULT_CHAIN can be simplified. That is a good failure.
        window = (date(2007, 1, 1), date(2025, 12, 31))
        spanning = [p.id for p in PRODUCTS.values() if p.covers(*window) and p.grain == "daily"]
        assert spanning == ["era5land"], spanning

    def test_asking_a_product_beyond_its_coverage_raises(self):
        with pytest.raises(ClimateSourceError, match="covers"):
            daily_municipal(
                ["3550308"],
                ["pr"],
                start=date(2024, 1, 1),
                end=date(2025, 12, 31),
                product="brdwgd",
                download=False,
            )

    def test_available_products_lists_every_product(self):
        assert set(available_products()["product"]) == set(PRODUCTS)

    def test_six_digit_codes_are_rejected(self):
        with pytest.raises(ValueError, match="7-digit"):
            daily_municipal(
                ["355030"],
                ["pr"],
                start=date(2015, 1, 1),
                end=date(2015, 1, 31),
                product="brdwgd_cdn",
                download=False,
            )


class TestPushdownMonthlyAggregation:
    def test_aggregates_at_source_and_exposes_incomplete_months(
        self, tmp_path, monkeypatch
    ):
        import duckdb

        days = pl.date_range(
            date(2024, 1, 1), date(2024, 12, 31), interval="1d", eager=True
        )
        # Deliberately omit 31 January. The production path must neither impute
        # it nor call the resulting 30-day sum a complete monthly exposure.
        fixture = pl.DataFrame(
            {
                "code_muni": [3550308] * (len(days) - 1),
                "date": days.filter(days != date(2024, 1, 31)),
                "name": ["pr_mean"] * (len(days) - 1),
                "value": [2.0] * (len(days) - 1),
            }
        )
        path = tmp_path / "daily.parquet"
        fixture.write_parquet(path)
        product = ClimateProduct(
            id="fixture",
            label="fixture",
            grain="daily",
            route="zenodo",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            files={"pr": "daily.parquet"},
            name_prefix={"pr": "pr"},
            units={"pr": (1.0, 0.0)},
            stats=("mean",),
            citation="test fixture",
        )
        monkeypatch.setitem(PRODUCTS, "fixture", product)
        monkeypatch.setattr(
            brdwgd, "_sources_for", lambda *_args, **_kwargs: [str(path)]
        )
        monkeypatch.setattr(brdwgd, "_connect", duckdb.connect)

        with pytest.raises(ClimateSourceError, match="incomplete"):
            monthly_municipal(
                [2024], munic_codes=["3550308"], chain=("fixture",), download=False
            )

        out = monthly_municipal(
            [2024],
            munic_codes=["3550308"],
            chain=("fixture",),
            download=False,
            require_complete_months=False,
        )
        jan = out.filter(pl.col("period") == date(2024, 1, 1)).row(0, named=True)
        assert jan["pr_total"] == pytest.approx(60.0)
        assert jan["rx1day"] == pytest.approx(2.0)
        assert jan["r1"] == 30
        assert jan["n_days"] == 30
        assert jan["n_days_expected"] == 31
        assert jan["coverage_fraction"] == pytest.approx(30 / 31)
