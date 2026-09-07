"""WP2 - build the population denominator for the leptospirosis panel.

Two products:

``population_municipal_year.parquet``
    Municipal totals by year, 2007-2025, from POPSVS (IBGE estimates as
    redistributed by DATASUS). These are the Ministry of Health's own
    denominators, so incidence computed on them is numerically comparable to a
    published Boletim Epidemiologico figure.

``population_tensor_long.parquet``
    The stratified sex x age tensor, needed for age-standardisation and for the
    age-sex profile that every leptospirosis study reports. Colour is not
    included: POPSVS does not carry it, and the census-based colour dimension
    is deferred to the SIDRA layer where the measurement caveat in
    ``brepi.denominators.rates`` applies.

Run:
    python studies/leptospirosis/02_denominators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice
from brepi.sources.datasus import population

YEARS = range(2007, 2026)


def main() -> int:
    PATHS.ensure()
    out = PATHS.interim

    print("[1/4] municipality lattice (2022)")
    muns = lattice.load_municipalities(year=2022)
    print(f"      {muns.height} municipalities, {muns['uf_code'].n_unique()} UFs")
    assert muns.height == 5570, f"expected 5570 municipalities, got {muns.height}"

    print("[2/4] POPSVS municipal population by sex and age")
    pop = population.municipal_population(YEARS, by=("sex", "age"))
    print(f"      {pop.height:,} rows, years {pop['year'].min()}-{pop['year'].max()}")

    # POPSVS keys on the 6-digit DATASUS code; the panel keys on 7-digit IBGE.
    # Going through the codec (rather than string-appending a digit) is what
    # keeps the nine hand-allocated 1995-97 codes from silently vanishing.
    pop = pop.with_columns(lattice.code6_to_code7_expr("munic_code").alias("munic_code7"))
    unmatched = pop.join(muns.select("code7"), left_on="munic_code7", right_on="code7", how="anti")
    if unmatched.height:
        bad = unmatched.select("munic_code", "munic_code7").unique()
        print(f"      WARNING: {bad.height} population codes absent from the 2022 lattice")
        print(bad.head(10))

    print("[3/4] municipal totals")
    totals = (
        pop.group_by("munic_code7", "year")
        .agg(pl.col("population").sum())
        .rename({"munic_code7": "munic_code"})
        .sort("munic_code", "year")
    )
    nat = totals.group_by("year").agg(pl.col("population").sum()).sort("year")
    print(nat.to_pandas().to_string(index=False))

    # Sanity gate: the 2022 census counted 203,080,756 residents. The estimate
    # series for that year should land within a couple of percent of it.
    p2022 = float(nat.filter(pl.col("year") == 2022)["population"][0])
    delta = abs(p2022 - 203_080_756) / 203_080_756
    print(f"      2022 estimate vs Censo 2022 (203,080,756): {delta:.2%}")
    if delta > 0.05:
        print("      GATE FAIL: population series inconsistent with the census")
        return 1
    print("      GATE PASS")

    totals.write_parquet(out / "population_municipal_year.parquet")

    print("[4/4] stratified tensor")
    strat = (
        pop.select(
            pl.col("munic_code7").alias("munic_code"),
            "year", "sex", "age_group", "population",
        )
        .sort("munic_code", "year", "sex", "age_group")
    )
    strat.write_parquet(out / "population_tensor_long.parquet")
    print(f"      {strat.height:,} rows; "
          f"{strat['age_group'].n_unique()} age groups, {strat['sex'].n_unique()} sexes")
    print(strat.group_by("age_group").agg(pl.col("population").sum())
          .sort("age_group").to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
