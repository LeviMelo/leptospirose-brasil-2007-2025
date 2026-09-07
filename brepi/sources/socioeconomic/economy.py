"""Municipal GDP with an explicit national-accounts price basis."""

from __future__ import annotations

import math

import polars as pl

from brepi.sources.sidra.extract import Selection


def municipal_gdp_selection() -> Selection:
    """Municipal nominal GDP, 2007-2023, in thousands of BRL."""
    return Selection(
        agregado=5938,
        periods=tuple(str(year) for year in range(2007, 2024)),
        variables=("37",),
        label="municipal_gdp_2007_2023",
    )


def gdp_deflator_selection() -> Selection:
    """Annual Brazilian GDP-deflator variation from national accounts."""
    return Selection(
        agregado=6784,
        periods=tuple(str(year) for year in range(2007, 2024)),
        variables=("9811",),
        level="N1",
        localities=("1",),
        label="national_gdp_deflator_2007_2023",
    )


def _price_index_2023(deflator_facts: pl.DataFrame) -> dict[int, float]:
    rates = {
        int(row["period"]): float(row["value_numeric"]) / 100
        for row in deflator_facts.filter(
            (pl.col("variable_id") == "9811")
            & pl.col("value_numeric").is_not_null()
        ).iter_rows(named=True)
    }
    missing = sorted(set(range(2008, 2024)) - set(rates))
    if missing:
        raise ValueError(f"GDP deflator lacks years needed for chaining: {missing}")
    index = {2023: 1.0}
    for year in range(2022, 2006, -1):
        denominator = 1 + rates[year + 1]
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"invalid GDP deflator rate for {year + 1}")
        index[year] = index[year + 1] / denominator
    return index


def build_real_gdp_per_capita(
    gdp_facts: pl.DataFrame,
    deflator_facts: pl.DataFrame,
    population: pl.DataFrame,
) -> pl.DataFrame:
    """Deflate municipal GDP to 2023 BRL and divide by annual population."""
    required_population = {"munic_code", "year", "population"}
    missing = required_population - set(population.columns)
    if missing:
        raise ValueError(f"population frame missing columns: {sorted(missing)}")
    price_index = _price_index_2023(deflator_facts)
    deflators = pl.DataFrame(
        {
            "year": list(price_index),
            "_price_index_2023": list(price_index.values()),
        }
    )
    observed = (
        gdp_facts.filter(
            (pl.col("variable_id") == "37")
            & pl.col("value_numeric").is_not_null()
        )
        .select(
            pl.col("locality_id").cast(pl.Utf8).alias("munic_code"),
            pl.col("period").cast(pl.Int32).alias("year"),
            pl.col("value_numeric").alias("_gdp_thousand_brl"),
        )
        .join(
            population.select("munic_code", "year", "population"),
            on=["munic_code", "year"],
            how="inner",
            validate="1:1",
        )
        .join(deflators, on="year", how="left", validate="m:1")
        .with_columns(
            (
                pl.col("_gdp_thousand_brl")
                * 1000
                / pl.col("_price_index_2023")
                / pl.col("population")
            ).alias("gdp_per_capita")
        )
    )
    invalid = observed.filter(
        pl.col("population").is_null()
        | (pl.col("population") <= 0)
        | pl.col("gdp_per_capita").is_null()
        | ~pl.col("gdp_per_capita").is_finite()
    )
    if invalid.height:
        examples = invalid.select(
            "munic_code",
            "year",
            "_gdp_thousand_brl",
            "_price_index_2023",
            "population",
            "gdp_per_capita",
        ).head(10).to_dicts()
        raise ValueError(
            f"{invalid.height} invalid GDP-per-capita observations: {examples}"
        )
    return observed.select(
        "munic_code",
        "year",
        "gdp_per_capita",
        "population",
    ).sort(["munic_code", "year"])
