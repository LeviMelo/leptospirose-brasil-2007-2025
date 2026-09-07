from __future__ import annotations

import polars as pl
import pytest

from brepi.sources.socioeconomic import (
    build_real_gdp_per_capita,
    gdp_deflator_selection,
    municipal_gdp_selection,
)


def test_gdp_selections_pin_series_and_price_basis():
    gdp = municipal_gdp_selection()
    deflator = gdp_deflator_selection()
    assert gdp.agregado == 5938
    assert gdp.variables == ("37",)
    assert gdp.periods[0] == "2007"
    assert gdp.periods[-1] == "2023"
    assert deflator.agregado == 6784
    assert deflator.variables == ("9811",)
    assert deflator.level == "N1"


def test_real_gdp_per_capita_chains_deflator_to_2023_prices():
    gdp = pl.DataFrame(
        {
            "variable_id": ["37", "37"],
            "locality_id": ["1", "1"],
            "period": ["2022", "2023"],
            "value_numeric": [100.0, 120.0],
        }
    )
    deflator = pl.DataFrame(
        {
            "variable_id": ["9811"] * 17,
            "period": [str(year) for year in range(2007, 2024)],
            "value_numeric": [0.0] * 16 + [20.0],
        }
    )
    population = pl.DataFrame(
        {
            "munic_code": ["1", "1"],
            "year": [2022, 2023],
            "population": [100, 100],
        }
    )
    out = build_real_gdp_per_capita(gdp, deflator, population)
    assert out["gdp_per_capita"].to_list() == pytest.approx([1200.0, 1200.0])
