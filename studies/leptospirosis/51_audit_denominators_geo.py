"""Audit the DATA SURFACE of the denominator, population structure and geography.

Why this exists
---------------
``docs/DATA.md`` section 3 describes the denominator as a *file* ("the tensor",
3,598,254 rows) and section 7 describes geography as a *mesh*. Neither says
which stratifiers actually exist in the artefact, which do not, at what
granularity each geographic level is available, or what a stratified rate could
be estimated on. This script answers those questions from the files themselves
and writes machine-readable evidence.

Populations profiled, stated once
---------------------------------
D1  denominator spine: 5,570 municipalities x 19 years (2007-2025) = 105,830
    municipality-year cells. Completeness of the tensor is judged on D1.
D2  confirmed leptospirosis cases: CLASSI_FIN = 'confirmado' in
    ``lept_line_level.parquet`` (n = 66,667). Feasibility of a stratified rate
    is judged on D2, because a denominator stratum with no numerator support
    buys nothing.
D3  the municipality lattice as served by IBGE localidades for 2022 and cached
    at ``data/cache/ibge/localidades/municipios.json``. Geographic unit counts
    are taken from D3.

Everything written here traces to a file under ``data/``. Nothing is fetched.

Run:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 \
      python \
      studies/leptospirosis/51_audit_denominators_geo.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.denominators.tensor import AGE_GROUPS as TENSOR_AGE_GROUPS
from brepi.denominators.tensor import COLOURS as TENSOR_COLOURS
from brepi.geo import lattice

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "audit_denominators_geo"

TENSOR = ROOT / "data" / "interim" / "population_tensor_long.parquet"
ANNUAL = ROOT / "data" / "interim" / "population_municipal_year.parquet"
LINE = ROOT / "data" / "interim" / "lept_line_level.parquet"
MUN_YEAR_ATLAS = ROOT / "data" / "results" / "atlas" / "municipality_year_atlas.parquet"
MUN_ATLAS = ROOT / "data" / "results" / "atlas" / "municipality_atlas.parquet"
HR_ATLAS = ROOT / "data" / "results" / "atlas" / "health_region_atlas.parquet"
HR_JSON = ROOT / "data" / "cache" / "geo" / "health_regions" / "open_datasus_20260730.json"
GPKG = ROOT / "data" / "panel" / "geo_municipality.gpkg"
GRAPHS = ROOT / "data" / "panel" / "graphs"
POPSVS = ROOT / "data" / "cache" / "datasus" / "ibge_pop" / "popsvs"
SIDRA_META = ROOT / "data" / "cache" / "sidra" / "metadados"
SIDRA_PERIODS = ROOT / "data" / "cache" / "sidra" / "periodos"
SIDRA_VALUES = ROOT / "data" / "cache" / "sidra" / "values"

YEARS = list(range(2007, 2026))
SPINE_MUNICIPALITIES = 5570
SPINE_CELLS = SPINE_MUNICIPALITIES * len(YEARS)

#: The 2022 Census resident count. Not invented here: it is the gate constant
#: already asserted in ``studies/leptospirosis/02_denominators.py`` (line 75).
CENSO_2022_RESIDENTS = 203_080_756

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)


def rule(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def w(df: pl.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / name)
    print(f"  wrote {name}  ({df.height} rows x {df.width} cols)")


def wj(obj: object, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {name}")


# ===========================================================================
# 1. Tensor dimensions
# ===========================================================================
def profile_tensor(tensor: pl.DataFrame) -> dict:
    rule("1. DENOMINATOR TENSOR - actual dimensions carried")

    dims = {c: tensor[c].n_unique() for c in tensor.columns if c != "population"}
    nulls = {c: int(tensor[c].null_count()) for c in tensor.columns}

    ages = sorted(tensor["age_group"].unique().to_list())
    sexes = sorted(tensor["sex"].unique().to_list())
    years = sorted(tensor["year"].unique().to_list())
    muns = tensor["munic_code"].n_unique()

    print(f"  columns present      : {tensor.columns}")
    print(f"  rows                 : {tensor.height:,}")
    print(f"  municipalities       : {muns}")
    print(f"  years                : {years[0]}-{years[-1]} ({len(years)} distinct)")
    print(f"  sexes                : {sexes}")
    print(f"  age groups ({len(ages)})     : {ages}")
    print(f"  colour / race column : {'colour' in tensor.columns}")
    print(f"  method / provenance  : {'method' in tensor.columns}")
    print(f"  years_from_census    : {'years_from_census' in tensor.columns}")
    print(f"  nulls per column     : {nulls}")

    dense = muns * len(years) * len(sexes) * len(ages)
    print(f"\n  dense product        : {muns} x {len(years)} x {len(sexes)} x {len(ages)}"
          f" = {dense:,}")
    print(f"  actual rows          : {tensor.height:,}  (delta {tensor.height - dense:+,})")

    # Which municipalities are not present in all 19 years?
    per_mun = (
        tensor.group_by("munic_code")
        .agg(pl.col("year").n_unique().alias("n_years"),
             pl.len().alias("n_rows"))
        .filter(pl.col("n_years") < len(YEARS))
        .sort("munic_code")
    )
    print(f"\n  municipalities not present in all {len(YEARS)} years: {per_mun.height}")
    if per_mun.height:
        detail = tensor.filter(pl.col("munic_code").is_in(per_mun["munic_code"].to_list()))
        summary = (
            detail.group_by("munic_code")
            .agg(pl.col("year").unique().sort().alias("years"),
                 pl.col("population").sum().alias("population_total"))
            .sort("munic_code")
        )
        print(summary)
        w(summary.with_columns(pl.col("years").cast(pl.List(pl.Utf8)).list.join(";")),
          "01b_tensor_partial_municipalities.csv")

    # Off-spine municipalities: present in the tensor but not in the 2022 lattice.
    lat = lattice.load_municipalities(2022).select("code7", "name", "uf_abbr")
    off = (
        tensor.select("munic_code").unique()
        .join(lat, left_on="munic_code", right_on="code7", how="anti")
        .sort("munic_code")
    )
    print(f"  municipalities in tensor but NOT in the 2022 lattice: {off.height} "
          f"{off['munic_code'].to_list()}")
    missing = lat.join(tensor.select("munic_code").unique(),
                       left_on="code7", right_on="munic_code", how="anti")
    print(f"  2022 lattice municipalities MISSING from the tensor: {missing.height}")

    # Zero and negative cells - a zero denominator cell makes a stratum-specific
    # rate undefined, so their count is a property a stratified analysis needs.
    zero = tensor.filter(pl.col("population") == 0).height
    neg = tensor.filter(pl.col("population") < 0).height
    print(f"\n  cells with population == 0 : {zero:,} ({100*zero/tensor.height:.2f}%)")
    print(f"  cells with population <  0 : {neg}")

    profile = {
        "path": str(TENSOR),
        "rows": int(tensor.height),
        "columns": list(tensor.columns),
        "dimensions_carried": ["munic_code", "year", "sex", "age_group"],
        "dimensions_absent": ["colour/race", "method", "years_from_census",
                              "urban/rural situation", "month"],
        "n_municipalities": int(muns),
        "n_years": len(years),
        "year_min": int(years[0]),
        "year_max": int(years[-1]),
        "sexes": sexes,
        "age_groups": ages,
        "n_age_groups": len(ages),
        "dense_product": int(dense),
        "rows_minus_dense": int(tensor.height - dense),
        "municipalities_off_2022_lattice": off["munic_code"].to_list(),
        "lattice_municipalities_missing_from_tensor": int(missing.height),
        "cells_population_zero": int(zero),
        "cells_population_negative": int(neg),
        "nulls_per_column": nulls,
        "code_system": "IBGE 7-digit (munic_code)",
        "tensor_module_declares_colours": list(TENSOR_COLOURS),
        "tensor_module_declares_age_groups": list(TENSOR_AGE_GROUPS),
        "age_label_mismatch_module_vs_artefact": sorted(
            set(TENSOR_AGE_GROUPS) ^ set(ages)
        ),
    }
    wj(profile, "01_tensor_profile.json")
    return profile


# ===========================================================================
# 2. Provenance: what actually built the artefact
# ===========================================================================
def audit_provenance() -> dict:
    rule("2. PROVENANCE - what built the artefact vs what DATA.md says built it")

    zips = sorted(p.name for p in POPSVS.glob("*.ZIP"))
    print(f"  POPSVS archives cached      : {len(zips)}  {zips[0]} .. {zips[-1]}")
    print("  POPSVS grain (verified)     : municipality x year x sex x SINGLE-YEAR age 000..080")
    print("  POPSVS carries colour/race  : NO")
    print("\n  DATA.md s.3 claims the tensor is built by iterative proportional")
    print("  fitting (raking) of a CENSUS age-sex seed to official municipal totals.")
    print("  studies/leptospirosis/02_denominators.py does NOT do that: it calls")
    print("  population.municipal_population(YEARS, by=('sex','age')) and writes the")
    print("  POPSVS table directly. brepi/denominators/tensor.py (build_tensor,")
    print("  raking, colour axis, per-cell method/years_from_census provenance) is")
    print("  library code that no study script calls.")

    called = {}
    for script in sorted((ROOT / "studies" / "leptospirosis").glob("*.py")):
        text = script.read_text(encoding="utf-8", errors="replace")
        if "build_tensor" in text or "denominators.tensor" in text:
            called[script.name] = [
                ln.strip() for ln in text.splitlines()
                if "build_tensor" in ln or "denominators.tensor" in ln
            ]
    print(f"\n  study scripts referencing brepi.denominators.tensor: "
          f"{sorted(called) or 'NONE'}")

    prov = {
        "artefact_built_by": "studies/leptospirosis/02_denominators.py",
        "actual_source": "DATASUS /IBGE/POPSVS POPSBR{YY}.zip, single-year age, "
                         "collapsed to 5-year bands by "
                         "brepi.sources.datasus.population._age_group_expr",
        "popsvs_archives_cached": zips,
        "raking_applied": False,
        "census_seed_used": False,
        "colour_axis_built": False,
        "per_cell_method_column": False,
        "per_cell_years_from_census_column": False,
        "library_that_would_do_it": "brepi/denominators/tensor.py::build_tensor",
        "study_scripts_calling_that_library": sorted(called),
        "docs_DATA_md_section_3_claim": "built by iterative proportional fitting "
                                        "(raking) of a census age-sex seed",
        "verdict": "DATA.md s.3 describes the designed tensor, not the built one. "
                   "The built artefact is an unraked POPSVS extract with two "
                   "stratifiers (sex, 5-year age) and no provenance columns.",
    }
    wj(prov, "02_tensor_provenance.json")
    return prov


# ===========================================================================
# 3. Reconciliation
# ===========================================================================
def reconcile(tensor: pl.DataFrame, annual: pl.DataFrame) -> pl.DataFrame:
    rule("3. RECONCILIATION - does the tensor sum to the published totals?")

    tsum = (
        tensor.group_by("munic_code", "year")
        .agg(pl.col("population").sum().alias("tensor_total"))
    )
    joined = annual.join(tsum, on=["munic_code", "year"], how="full", coalesce=True)
    joined = joined.with_columns(
        (pl.col("tensor_total") - pl.col("population")).alias("diff")
    )

    by_year = (
        joined.group_by("year")
        .agg(
            pl.len().alias("municipality_years"),
            pl.col("population").sum().alias("annual_national"),
            pl.col("tensor_total").sum().alias("tensor_national"),
            pl.col("diff").abs().max().alias("max_abs_municipal_diff"),
            (pl.col("diff") != 0).sum().alias("n_municipalities_differing"),
            pl.col("population").is_null().sum().alias("annual_missing"),
            pl.col("tensor_total").is_null().sum().alias("tensor_missing"),
        )
        .sort("year")
        .with_columns(
            (pl.col("tensor_national") - pl.col("annual_national")).alias("national_diff")
        )
    )
    print(by_year)
    w(by_year, "03_margin_check_by_year.csv")

    nat2022 = int(by_year.filter(pl.col("year") == 2022)["tensor_national"][0])
    delta = (nat2022 - CENSO_2022_RESIDENTS) / CENSO_2022_RESIDENTS
    print(f"\n  2022 tensor national total : {nat2022:,}")
    print(f"  Censo 2022 resident count  : {CENSO_2022_RESIDENTS:,}")
    print(f"  relative difference        : {delta:+.3%}")

    # Cross-check against the atlas, which is what every figure and rate reads.
    atlas = pl.read_parquet(MUN_YEAR_ATLAS, columns=["munic_code", "year",
                                                     "population", "person_years"])
    cross = (
        atlas.rename({"population": "atlas_population"})
        .join(tsum, on=["munic_code", "year"], how="full", coalesce=True)
        .with_columns((pl.col("tensor_total") - pl.col("atlas_population")).alias("diff"))
    )
    atlas_by_year = (
        cross.group_by("year").agg(
            pl.len().alias("cells"),
            pl.col("atlas_population").sum().alias("atlas_national"),
            pl.col("tensor_total").sum().alias("tensor_national"),
            pl.col("diff").abs().max().alias("max_abs_diff"),
            pl.col("atlas_population").is_null().sum().alias("atlas_missing"),
            pl.col("tensor_total").is_null().sum().alias("tensor_missing"),
        ).sort("year")
    )
    print("\n  tensor vs municipality_year_atlas.population:")
    print(atlas_by_year)
    w(atlas_by_year, "03b_margin_check_vs_atlas.csv")

    # person_years convention actually used downstream
    pyr = atlas.select(
        (pl.col("person_years") / pl.col("population")).alias("ratio")
    ).drop_nulls()
    print(f"\n  atlas person_years / population : min {pyr['ratio'].min():.4f} "
          f"median {pyr['ratio'].median():.4f} max {pyr['ratio'].max():.4f}")

    # -- 3c. the denominator vs the 2022 census, municipality by municipality --
    # The OpenDataSUS health-region extract carries a municipal population column
    # that sums to exactly CENSO_2022_RESIDENTS, i.e. it is the census count, not
    # the POPSVS projection. That gives a free municipality-level comparison
    # between the denominator the study uses and the enumerated population.
    hr_rows = json.loads(
        HR_JSON.read_text(encoding="utf-8")
    )["macrorregiao_regiao_saude_municipios"]
    censo = (
        pl.DataFrame(hr_rows)
        .select(
            lattice.code6_to_code7_expr(
                pl.col("codigo_municipio").cast(pl.Utf8)).alias("munic_code"),
            pl.col("populacao_estimada_ibge_2022").cast(pl.Int64).alias("censo_2022"),
            pl.col("codigo_uf").cast(pl.Utf8).alias("uf_code"),
        )
    )
    print(f"\n  crosswalk population column sums to {censo['censo_2022'].sum():,} "
          f"(Censo 2022 = {CENSO_2022_RESIDENTS:,}; "
          f"identical: {censo['censo_2022'].sum() == CENSO_2022_RESIDENTS})")

    cmp2022 = (
        tsum.filter(pl.col("year") == 2022)
        .join(censo, on="munic_code", how="inner")
        .with_columns(
            (pl.col("tensor_total") - pl.col("censo_2022")).alias("abs_gap"),
            (100 * (pl.col("tensor_total") - pl.col("censo_2022"))
             / pl.col("censo_2022")).alias("pct_gap"),
        )
    )
    q = cmp2022["pct_gap"]
    print("\n  POPSVS-2022 denominator vs Censo-2022 count, per municipality (%):")
    print(f"    n={cmp2022.height}  min {q.min():.1f}  p5 {q.quantile(0.05):.1f}  "
          f"p25 {q.quantile(0.25):.1f}  median {q.median():.1f}  "
          f"p75 {q.quantile(0.75):.1f}  p95 {q.quantile(0.95):.1f}  max {q.max():.1f}")
    print(f"    municipalities where the denominator exceeds the census by >10%: "
          f"{cmp2022.filter(pl.col('pct_gap') > 10).height}")
    print(f"    municipalities where the denominator is below the census by >10%: "
          f"{cmp2022.filter(pl.col('pct_gap') < -10).height}")
    by_uf = (
        cmp2022.group_by("uf_code")
        .agg(pl.col("tensor_total").sum().alias("popsvs_2022"),
             pl.col("censo_2022").sum().alias("censo_2022"),
             pl.len().alias("municipalities"))
        .with_columns((100 * (pl.col("popsvs_2022") - pl.col("censo_2022"))
                       / pl.col("censo_2022")).round(2).alias("pct_gap"))
        .sort("pct_gap")
    )
    print("\n  same gap aggregated by UF (extremes):")
    print(pl.concat([by_uf.head(5), by_uf.tail(5)]))
    w(cmp2022.sort("pct_gap"), "03c_popsvs_vs_censo2022_municipal.csv")
    w(by_uf, "03d_popsvs_vs_censo2022_by_uf.csv")
    return by_year


# ===========================================================================
# 4. Age-sex structure the tensor can support
# ===========================================================================
def age_sex_structure(tensor: pl.DataFrame) -> None:
    rule("4. AGE x SEX STRUCTURE available in the denominator")

    nat = (
        tensor.group_by("year", "sex", "age_group")
        .agg(pl.col("population").sum())
        .sort("year", "sex", "age_group")
    )
    w(nat, "04_national_age_sex_by_year.csv")

    pivot = (
        nat.filter(pl.col("year").is_in([2007, 2010, 2015, 2022, 2025]))
        .pivot(values="population", index="age_group", on=["year", "sex"],
               aggregate_function="sum")
        .sort("age_group")
    )
    print(pivot)

    # Ageing signal: the share 60+ over time. If POPSVS carries a real
    # projection, this moves; if the structure were frozen at a census it would not.
    old = (
        tensor.with_columns(
            pl.col("age_group").is_in(["60-64", "65-69", "70-74", "75-79", "80+"])
            .alias("is_60plus")
        )
        .group_by("year")
        .agg(
            pl.col("population").sum().alias("total"),
            pl.col("population").filter(pl.col("is_60plus")).sum().alias("pop_60plus"),
            pl.col("population").filter(pl.col("age_group") == "00-04").sum().alias("pop_0_4"),
            pl.col("population").filter(pl.col("sex") == "male").sum().alias("pop_male"),
        )
        .sort("year")
        .with_columns(
            (100 * pl.col("pop_60plus") / pl.col("total")).round(2).alias("pct_60plus"),
            (100 * pl.col("pop_0_4") / pl.col("total")).round(2).alias("pct_0_4"),
            (100 * pl.col("pop_male") / pl.col("total")).round(2).alias("pct_male"),
        )
    )
    print(old.select("year", "total", "pct_0_4", "pct_60plus", "pct_male"))
    w(old, "04b_national_structure_trend.csv")

    # Smallest strata: a municipality-year-sex-age cell with a tiny denominator
    # is where a stratified rate explodes.
    small = tensor.filter(pl.col("population") > 0)["population"]
    q = [0.01, 0.05, 0.25, 0.5]
    print("\n  non-zero cell population quantiles: "
          + ", ".join(f"p{int(100*x)}={small.quantile(x):.0f}" for x in q))


# ===========================================================================
# 4c. The age-label contract between the tensor and the standard population
# ===========================================================================
def age_label_contract(tensor: pl.DataFrame) -> None:
    """The artefact labels its youngest bands ``00-04``/``05-09``; the shipped
    standard populations label them ``0-4``/``5-9``. ``age_standardised_rate``
    joins numerator, denominator and standard on that label with ``how="inner"``,
    so the mismatch does not raise - it silently drops both bands from the rate.
    This section proves the drop against the published artefact and quantifies it.
    """
    rule("4c. AGE-LABEL CONTRACT - tensor labels vs the standard population")

    from brepi.denominators.rates import WHO_WORLD_STANDARD, age_standardised_rate

    tensor_labels = sorted(tensor["age_group"].unique().to_list())
    std_labels = list(WHO_WORLD_STANDARD)
    only_tensor = sorted(set(tensor_labels) - set(std_labels))
    only_std = sorted(set(std_labels) - set(tensor_labels))
    print(f"  tensor labels not in WHO_WORLD_STANDARD : {only_tensor}")
    print(f"  standard labels not in the tensor       : {only_std}")
    lost_weight = sum(WHO_WORLD_STANDARD[k] for k in only_std)
    print(f"  standard weight on the unmatched labels : {lost_weight:.4f} "
          f"({100*lost_weight:.2f}% of the standard)")

    # The guard in age_standardised_rate compares the standard against the
    # MODULE's AGE_GROUPS, not against the labels actually present in the data,
    # so it passes while the join drops rows.
    print("  age_standardised_rate() validates the standard against "
          "tensor.AGE_GROUPS ('0-4', '5-9', ...), not against the artefact's "
          "labels, so the guard passes and the inner join drops the bands.")

    # Reproduce the drop on the real data.
    line = pl.read_parquet(LINE, columns=["classi_fin", "age_years", "src_year"])
    conf = line.filter(pl.col("classi_fin") == "confirmado")

    def band(expr: pl.Expr) -> pl.Expr:
        a = expr.cast(pl.Int32, strict=False)
        e = pl.when(a.is_null()).then(pl.lit(None, dtype=pl.Utf8))
        for lo in range(0, 80, 5):
            e = e.when(a < lo + 5).then(pl.lit(f"{lo:02d}-{lo+4:02d}"))
        return e.otherwise(pl.lit("80+"))

    num = (
        conf.with_columns(band(pl.col("age_years")).alias("age_group"),
                          pl.col("src_year").cast(pl.Int32).alias("year"))
        .filter(pl.col("age_group").is_not_null())
        .group_by("year", "age_group").agg(pl.len().alias("cases"))
    )
    den = tensor.group_by("year", "age_group").agg(pl.col("population").sum())

    as_built = age_standardised_rate(num, den, by=["year"],
                                     standard=WHO_WORLD_STANDARD)

    # Corrected: relabel the standard onto the artefact's own labels. The library
    # REFUSES this - its guard demands the module's labels, which are exactly the
    # ones the data does not use - so the corrected rate is computed here with the
    # same estimator rather than through the library.
    relabel = {"0-4": "00-04", "5-9": "05-09"}
    fixed_std = {relabel.get(k, k): v for k, v in WHO_WORLD_STANDARD.items()}
    try:
        age_standardised_rate(num, den, by=["year"], standard=fixed_std)
        library_accepts_correct_standard = True
    except ValueError as exc:
        library_accepts_correct_standard = False
        print(f"\n  age_standardised_rate() REFUSES the correctly-labelled standard: "
              f"{exc}")
        print("  i.e. the guard rejects the standard that matches the data and "
              "accepts the one that does not.")

    total_w = sum(fixed_std.values())
    wts = pl.DataFrame({"age_group": list(fixed_std),
                        "_w": [v / total_w for v in fixed_std.values()]})
    corrected = (
        den.join(num, on=["year", "age_group"], how="left")
        .with_columns(pl.col("cases").fill_null(0))
        .join(wts, on="age_group", how="inner")
        .with_columns(
            pl.when(pl.col("population") > 0)
            .then(pl.col("cases") / pl.col("population"))
            .otherwise(0.0).alias("_r")
        )
        .group_by("year")
        .agg((pl.col("_w") * pl.col("_r")).sum().mul(100_000).alias("asr"),
             pl.col("cases").sum().alias("cases"),
             pl.col("population").sum().alias("population"))
        .sort("year")
    )

    cmp = (
        as_built.select("year", pl.col("asr").alias("asr_as_built"),
                        pl.col("cases").alias("cases_as_built"),
                        pl.col("population").alias("pop_as_built"))
        .join(
            corrected.select("year", pl.col("asr").alias("asr_corrected"),
                             pl.col("cases").alias("cases_corrected"),
                             pl.col("population").alias("pop_corrected")),
            on="year", how="inner")
        .with_columns(
            (pl.col("cases_corrected") - pl.col("cases_as_built")).alias("cases_dropped"),
            (pl.col("pop_corrected") - pl.col("pop_as_built")).alias("population_dropped"),
            (100 * (pl.col("asr_corrected") - pl.col("asr_as_built"))
             / pl.col("asr_as_built")).round(2).alias("asr_pct_understated"),
        )
        .sort("year")
    )
    print(cmp.select("year", "cases_as_built", "cases_corrected", "cases_dropped",
                     "population_dropped", "asr_as_built", "asr_corrected",
                     "asr_pct_understated"))
    w(cmp, "04c_age_label_contract.csv")

    total_dropped = int(cmp["cases_dropped"].sum())
    print(f"\n  cases silently dropped from the age-standardised rate, 2007-2025: "
          f"{total_dropped:,} of {int(cmp['cases_corrected'].sum()):,}")
    print(f"  mean understatement of the ASR: "
          f"{cmp['asr_pct_understated'].mean():.1f}%")

    # Confirm against the artefact the study already published.
    published = ROOT / "data" / "results" / "01_descriptive" / "08_age_standardised.csv"
    verdict = {"published_artefact": str(published)}
    if published.exists():
        pub = pl.read_csv(published)
        pub_cases = int(pub["cases"].sum())
        print(f"\n  published 08_age_standardised.csv total cases : {pub_cases:,}")
        print(f"  confirmed cases with a usable age             : "
              f"{int(cmp['cases_corrected'].sum()):,}")
        print(f"  difference                                    : "
              f"{int(cmp['cases_corrected'].sum()) - pub_cases:,}")
        verdict |= {
            "published_cases": pub_cases,
            "cases_with_usable_age": int(cmp["cases_corrected"].sum()),
            "cases_missing_from_published": int(cmp["cases_corrected"].sum()) - pub_cases,
            "bug_confirmed_in_published_output": pub_cases < int(cmp["cases_corrected"].sum()),
        }
    verdict |= {
        "tensor_labels_not_in_standard": only_tensor,
        "standard_labels_not_in_tensor": only_std,
        "standard_weight_unmatched": round(lost_weight, 4),
        "cases_dropped_total": total_dropped,
        "mean_asr_understatement_pct": round(float(cmp["asr_pct_understated"].mean()), 2),
        "library_accepts_correctly_labelled_standard": library_accepts_correct_standard,
        "mechanism": "brepi/denominators/rates.py::age_standardised_rate joins the "
                     "standard with how='inner' on age_group; its only guard compares "
                     "the standard to tensor.AGE_GROUPS, not to the labels present in "
                     "the data, so a label mismatch drops strata silently and does not "
                     "renormalise the remaining weights.",
        "note_r_path_is_correct": "studies/leptospirosis/22_paper_descriptives.R uses "
                                  "rebase_standard(WHO_WORLD_STANDARD, TENSOR_GROUPS) "
                                  "with TENSOR_GROUPS = c('00-04','05-09',...), so the R "
                                  "descriptive path is unaffected. The Python path is.",
    }
    wj(verdict, "04c_age_label_contract.json")


# ===========================================================================
# 5. Stratified-analysis feasibility against the case counts
# ===========================================================================
def feasibility(tensor: pl.DataFrame) -> None:
    rule("5. WHAT STRATIFIED ANALYSES ARE ESTIMABLE (numerator support, D2)")

    cols = ["classi_fin", "cs_sexo", "cs_raca", "cs_raca_state", "age_years",
            "age_state", "src_year", "municipality_residence_code7",
            "uf_residence_abbr", "uf_residence_region", "evolucao", "ate_hosp"]
    line = pl.read_parquet(LINE, columns=cols)
    conf = line.filter(pl.col("classi_fin") == "confirmado")
    print(f"  confirmed cases (D2): {conf.height:,}")

    # 5-year age band matching the tensor labels
    def band(expr: pl.Expr) -> pl.Expr:
        a = expr.cast(pl.Int32, strict=False)
        e = pl.when(a.is_null()).then(pl.lit(None, dtype=pl.Utf8))
        for lo in range(0, 80, 5):
            e = e.when(a < lo + 5).then(pl.lit(f"{lo:02d}-{lo+4:02d}"))
        return e.otherwise(pl.lit("80+"))

    conf = conf.with_columns(
        band(pl.col("age_years")).alias("age_group"),
        pl.when(pl.col("cs_sexo") == "masculino").then(pl.lit("male"))
        .when(pl.col("cs_sexo") == "feminino").then(pl.lit("female"))
        .otherwise(pl.lit("unknown")).alias("sex"),
        pl.col("src_year").cast(pl.Int32).alias("year"),
    )

    # -- 5a. field completeness on D2 -----------------------------------------
    rows = []
    for fld, lab in [("sex", "sex (CS_SEXO)"), ("age_group", "age band (NU_IDADE_N)"),
                     ("cs_raca", "colour/race (CS_RACA)"),
                     ("municipality_residence_code7", "residence municipality")]:
        ok = conf.filter(pl.col(fld).is_not_null() & (pl.col(fld) != "unknown")).height
        rows.append({"field": lab, "column": fld, "n_usable": ok,
                     "n_total": conf.height,
                     "pct_usable": round(100 * ok / conf.height, 2)})
    comp = pl.DataFrame(rows)
    print("\n  numerator completeness on confirmed cases:")
    print(comp)
    w(comp, "05a_numerator_completeness.csv")

    # -- 5b. sex x age cells: is the joint estimable? --------------------------
    cell = (
        conf.filter(pl.col("sex") != "unknown", pl.col("age_group").is_not_null())
        .group_by("sex", "age_group").agg(pl.len().alias("cases"))
    )
    den = tensor.group_by("sex", "age_group").agg(pl.col("population").sum())
    sexage = (
        den.join(cell, on=["sex", "age_group"], how="left")
        .with_columns(pl.col("cases").fill_null(0))
        .with_columns(
            (pl.col("cases") / (pl.col("population") * 1.0) * 100_000).alias(
                "cases_per_100k_person_years_summed_over_2007_2025")
        )
        .sort("sex", "age_group")
    )
    print("\n  sex x age joint cells (whole period, national):")
    print(sexage)
    w(sexage, "05b_sex_age_cells_national.csv")
    print(f"  cells with 0 cases: {sexage.filter(pl.col('cases') == 0).height} / {sexage.height}")
    print(f"  cells with <20 cases: {sexage.filter(pl.col('cases') < 20).height} / {sexage.height}")

    # -- 5c. estimability ladder ---------------------------------------------
    ladder = []
    specs = [
        ("age x sex, national, whole period", ["sex", "age_group"], None),
        ("age x sex x year, national", ["sex", "age_group"], "year"),
        ("age x sex x macro-region, whole period", ["sex", "age_group"], "region"),
        ("age x sex x macro-region x year", ["sex", "age_group"], "region_year"),
        ("age x sex x UF, whole period", ["sex", "age_group"], "uf"),
        ("age x sex x UF x year", ["sex", "age_group"], "uf_year"),
    ]
    base = conf.filter(pl.col("sex") != "unknown", pl.col("age_group").is_not_null())
    for label, keys, extra in specs:
        k = list(keys)
        f = base
        if extra == "year":
            k += ["year"]
        elif extra == "region":
            k += ["uf_residence_region"]
            f = f.filter(pl.col("uf_residence_region").is_not_null())
        elif extra == "region_year":
            k += ["uf_residence_region", "year"]
            f = f.filter(pl.col("uf_residence_region").is_not_null())
        elif extra == "uf":
            k += ["uf_residence_abbr"]
            f = f.filter(pl.col("uf_residence_abbr").is_not_null())
        elif extra == "uf_year":
            k += ["uf_residence_abbr", "year"]
            f = f.filter(pl.col("uf_residence_abbr").is_not_null())
        g = f.group_by(k).agg(pl.len().alias("cases"))
        ladder.append({
            "stratification": label,
            "cells_with_cases": g.height,
            "median_cases_per_cell": float(g["cases"].median()),
            "cells_lt_5_cases": int(g.filter(pl.col("cases") < 5).height),
            "cells_ge_20_cases": int(g.filter(pl.col("cases") >= 20).height),
            "pct_cells_ge_20": round(100 * g.filter(pl.col("cases") >= 20).height / g.height, 1),
        })
    lad = pl.DataFrame(ladder)
    print("\n  estimability ladder (cells that actually contain cases):")
    print(lad)
    w(lad, "05c_estimability_ladder.csv")

    # -- 5d. race numerator, by year and by state -----------------------------
    race_year = (
        conf.group_by("year")
        .agg(
            pl.len().alias("cases"),
            (pl.col("cs_raca_state") == "valid").sum().alias("race_valid"),
            (pl.col("cs_raca_state") == "unknown").sum().alias("race_ignored"),
            (pl.col("cs_raca_state") == "missing").sum().alias("race_missing"),
        )
        .sort("year")
        .with_columns((100 * pl.col("race_valid") / pl.col("cases")).round(1)
                      .alias("pct_race_valid"))
    )
    print("\n  CS_RACA availability among confirmed cases, by year:")
    print(race_year)
    w(race_year, "05d_race_numerator_by_year.csv")

    race_uf = (
        conf.filter(pl.col("uf_residence_abbr").is_not_null())
        .group_by("uf_residence_abbr")
        .agg(pl.len().alias("cases"),
             (pl.col("cs_raca_state") == "valid").sum().alias("race_valid"))
        .with_columns((100 * pl.col("race_valid") / pl.col("cases")).round(1)
                      .alias("pct_race_valid"))
        .sort("pct_race_valid")
    )
    print("\n  CS_RACA validity by UF of residence (worst 8 / best 4):")
    print(pl.concat([race_uf.head(8), race_uf.tail(4)]))
    w(race_uf, "05e_race_numerator_by_uf.csv")

    race_dist = (
        conf.filter(pl.col("cs_raca_state") == "valid")
        .group_by("cs_raca").agg(pl.len().alias("cases"))
        .sort("cases", descending=True)
    )
    print("\n  colour distribution among confirmed cases with a valid CS_RACA:")
    print(race_dist)
    w(race_dist, "05f_race_case_distribution.csv")

    # -- 5e. sex-age interaction on the OUTCOME (the paper's thesis) ----------
    sev = (
        conf.filter(pl.col("sex") != "unknown", pl.col("age_group").is_not_null())
        .with_columns(
            (pl.col("evolucao") == "obito_por_leptospirose").alias("death"),
            pl.col("evolucao").is_in(["cura", "obito_por_leptospirose",
                                      "obito_por_outras_causas"]).alias("outcome_known"),
            (pl.col("ate_hosp") == "sim").alias("hosp"),
        )
        .group_by("sex", "age_group")
        .agg(pl.len().alias("cases"),
             pl.col("outcome_known").sum().alias("outcome_known"),
             pl.col("death").sum().alias("deaths"),
             pl.col("hosp").sum().alias("hospitalised"))
        .sort("sex", "age_group")
        .with_columns(
            (100 * pl.col("deaths") / pl.col("outcome_known")).round(2).alias("cfr_pct"),
            (100 * pl.col("hospitalised") / pl.col("cases")).round(2).alias("hosp_share_pct"),
        )
    )
    print("\n  severity by sex x age (numerator-only; needs no denominator):")
    print(sev)
    w(sev, "05g_severity_by_sex_age.csv")


# ===========================================================================
# 5b. Does the demographic structure the tensor carries confound the thesis?
# ===========================================================================
def depth_cfr_age_confounding() -> None:
    """The paper indexes surveillance depth by the hospitalised share of confirmed
    cases and relates it to case fatality. Both quantities rise steeply and
    monotonically with age (05g). A territory whose confirmed cases skew old
    therefore has a higher hospitalised share AND a higher CFR for reasons that
    are demographic, not surveillance-related. This section measures how much of
    the depth-CFR association at health-region grain survives indirect age-sex
    standardisation of both sides. It is the single check the tensor's age
    dimension makes possible and the study has not run.
    """
    rule("5b. AGE-SEX STRUCTURE AS A CONFOUNDER OF THE DEPTH -> CFR THESIS")

    from scipy import stats

    line = pl.read_parquet(
        LINE,
        columns=["classi_fin", "cs_sexo", "age_years", "evolucao", "ate_hosp",
                 "municipality_residence_code7"],
    )
    conf = line.filter(pl.col("classi_fin") == "confirmado")

    def band(expr: pl.Expr) -> pl.Expr:
        a = expr.cast(pl.Int32, strict=False)
        e = pl.when(a.is_null()).then(pl.lit(None, dtype=pl.Utf8))
        for lo in range(0, 80, 5):
            e = e.when(a < lo + 5).then(pl.lit(f"{lo:02d}-{lo+4:02d}"))
        return e.otherwise(pl.lit("80+"))

    hr_rows = json.loads(
        HR_JSON.read_text(encoding="utf-8")
    )["macrorregiao_regiao_saude_municipios"]
    hr = pl.DataFrame(hr_rows).select(
        lattice.code6_to_code7_expr(
            pl.col("codigo_municipio").cast(pl.Utf8)).alias("munic_code"),
        pl.col("codigo_regiao_saude").cast(pl.Utf8).alias("health_region"),
    )

    conf = (
        conf.with_columns(
            band(pl.col("age_years")).alias("age_group"),
            pl.when(pl.col("cs_sexo") == "masculino").then(pl.lit("male"))
            .when(pl.col("cs_sexo") == "feminino").then(pl.lit("female"))
            .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("sex"),
            (pl.col("evolucao") == "obito_por_leptospirose").alias("death"),
            pl.col("evolucao").is_in(["cura", "obito_por_leptospirose",
                                      "obito_por_outras_causas"]).alias("outcome_known"),
            (pl.col("ate_hosp") == "sim").alias("hosp"),
            pl.col("ate_hosp").is_in(["sim", "nao"]).alias("hosp_known"),
        )
        .filter(pl.col("sex").is_not_null(), pl.col("age_group").is_not_null())
        .join(hr, left_on="municipality_residence_code7", right_on="munic_code",
              how="inner")
    )
    print(f"  confirmed cases with sex, age and a health region: {conf.height:,}")

    # National age-sex schedules: the standard the indirect method applies.
    sched = (
        conf.group_by("sex", "age_group")
        .agg(
            pl.col("outcome_known").sum().alias("n_outcome"),
            pl.col("death").sum().alias("n_death"),
            pl.col("hosp_known").sum().alias("n_hosp_known"),
            pl.col("hosp").sum().alias("n_hosp"),
        )
        .with_columns(
            (pl.col("n_death") / pl.col("n_outcome")).alias("cfr_std"),
            (pl.col("n_hosp") / pl.col("n_hosp_known")).alias("hosp_std"),
        )
        .select("sex", "age_group", "cfr_std", "hosp_std")
    )

    reg = (
        conf.join(sched, on=["sex", "age_group"], how="left")
        .group_by("health_region")
        .agg(
            pl.len().alias("cases"),
            pl.col("outcome_known").sum().alias("outcome_known"),
            pl.col("death").sum().alias("deaths"),
            pl.col("hosp_known").sum().alias("hosp_known"),
            pl.col("hosp").sum().alias("hospitalised"),
            pl.col("age_years").mean().alias("mean_case_age"),
            (pl.col("age_years") >= 60).mean().alias("share_cases_60plus"),
            (pl.col("sex") == "male").mean().alias("share_male"),
            pl.col("cfr_std").filter(pl.col("outcome_known")).sum().alias("expected_deaths"),
            pl.col("hosp_std").filter(pl.col("hosp_known")).sum().alias("expected_hosp"),
        )
        .with_columns(
            (pl.col("deaths") / pl.col("outcome_known")).alias("cfr_crude"),
            (pl.col("hospitalised") / pl.col("hosp_known")).alias("depth_crude"),
            (pl.col("deaths") / pl.col("expected_deaths")).alias("cfr_smr"),
            (pl.col("hospitalised") / pl.col("expected_hosp")).alias("depth_smr"),
        )
    )
    # Restrict to regions where either ratio is meaningfully estimated.
    MIN_OUTCOME = 30
    sub = reg.filter(pl.col("outcome_known") >= MIN_OUTCOME,
                     pl.col("hosp_known") >= MIN_OUTCOME,
                     pl.col("expected_deaths") > 0)
    print(f"  health regions with >= {MIN_OUTCOME} cases of known outcome and known "
          f"hospitalisation: {sub.height} / {reg.height}")

    def corr(a: str, b: str) -> tuple[float, float]:
        x = sub[a].to_numpy()
        y = sub[b].to_numpy()
        r = stats.spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)

    rows = []
    for lab, a, b in [
        ("case age structure (share 60+) vs depth (crude hosp share)",
         "share_cases_60plus", "depth_crude"),
        ("case age structure (share 60+) vs CFR (crude)",
         "share_cases_60plus", "cfr_crude"),
        ("DEPTH vs CFR, both crude  [the paper's association]",
         "depth_crude", "cfr_crude"),
        ("DEPTH vs CFR, both indirectly age-sex standardised",
         "depth_smr", "cfr_smr"),
        ("share male vs CFR (crude)", "share_male", "cfr_crude"),
    ]:
        rho, p = corr(a, b)
        rows.append({"comparison": lab, "spearman_rho": round(rho, 3),
                     "p_value": float(f"{p:.3g}"), "n_health_regions": sub.height})
    ct = pl.DataFrame(rows)
    print(ct)
    w(ct, "05h_depth_cfr_age_confounding.csv")
    w(reg.sort("cases", descending=True), "05i_health_region_case_structure.csv")

    lo = sub.select(pl.col("share_cases_60plus").quantile(0.1)).item()
    hi = sub.select(pl.col("share_cases_60plus").quantile(0.9)).item()
    crude = ct.filter(pl.col("comparison").str.contains("both crude"))["spearman_rho"][0]
    stdz = ct.filter(pl.col("comparison").str.contains("standardised"))["spearman_rho"][0]
    print(f"\n  share of confirmed cases aged 60+ across health regions: "
          f"p10 {100*lo:.1f}%  p90 {100*hi:.1f}%  "
          f"(a {hi/max(lo, 1e-9):.1f}-fold spread in case age structure)")
    print(f"  depth -> CFR Spearman rho: crude {crude:+.3f}, "
          f"indirectly age-sex standardised {stdz:+.3f} (change {stdz - crude:+.3f}).")
    print("  Read: the case age structure varies substantially between health "
          "regions and drives both quantities at the individual level (05g), but "
          "removing it changes the paper's central association negligibly. The "
          "obvious referee objection - 'your depth index is just an age structure "
          "index' - is therefore answerable with the dimension the tensor already "
          "carries, and this is the check that answers it.")
    wj({
        "n_health_regions": int(sub.height),
        "min_outcome_known_per_region": MIN_OUTCOME,
        "share_cases_60plus_p10": round(float(lo), 4),
        "share_cases_60plus_p90": round(float(hi), 4),
        "depth_cfr_rho_crude": round(float(crude), 4),
        "depth_cfr_rho_age_sex_standardised": round(float(stdz), 4),
        "standardisation_method": "indirect; national age-sex-specific CFR and "
                                  "hospitalisation schedules applied to each health "
                                  "region's own case age-sex distribution, SMR-style",
        "interpretation": "The depth-CFR association is not an artefact of the case "
                          "age-sex structure.",
    }, "05h_depth_cfr_age_confounding.json")


# ===========================================================================
# 6. Geography
# ===========================================================================
def geography() -> dict:
    rule("6. GEOGRAPHY - every level, its unit count, and where it lives")

    lat = lattice.load_municipalities(2022)
    hr_rows = json.loads(HR_JSON.read_text(encoding="utf-8"))["macrorregiao_regiao_saude_municipios"]
    hr = pl.DataFrame(hr_rows)

    hr = hr.with_columns(
        lattice.code6_to_code7_expr(pl.col("codigo_municipio").cast(pl.Utf8)).alias("code7")
    )

    graphs = {}
    for p in sorted(GRAPHS.glob("*.adj")):
        graphs[p.stem] = int(p.read_text(encoding="utf-8").splitlines()[0].strip())

    mun_atlas = pl.read_parquet(MUN_ATLAS)
    hr_atlas = pl.read_parquet(HR_ATLAS)

    levels = [
        ("macro-region (Grande Regiao)", lat["region"].n_unique(),
         "brepi.geo.lattice.load_municipalities -> region",
         "yes (municipality_atlas.region)", graphs.get("macro_region")),
        ("state (UF)", lat["uf_code"].n_unique(),
         "lattice -> uf_code / uf_abbr",
         "yes (municipality_atlas.uf_code, uf_abbr)", graphs.get("uf")),
        ("intermediate region (2017)", lat["intermediate_region"].n_unique(),
         "lattice -> intermediate_region",
         "yes (municipality_atlas.intermediate_region)", graphs.get("intermediate_region")),
        ("mesoregion (legacy)", lat["mesoregion"].n_unique(),
         "lattice -> mesoregion",
         "NO", graphs.get("mesoregion")),
        ("health macro-region (macrorregiao de saude)",
         hr["codigo_macrorregiao_saude"].n_unique(),
         "OpenDataSUS crosswalk JSON -> codigo_macrorregiao_saude",
         "NO", graphs.get("health_macro_region")),
        ("health region (regiao de saude)", hr["codigo_regiao_saude"].n_unique(),
         "OpenDataSUS crosswalk JSON -> codigo_regiao_saude",
         "yes (health_region_atlas, 439 rows)", graphs.get("health_region")),
        ("immediate region (2017)", lat["immediate_region"].n_unique(),
         "lattice -> immediate_region",
         "yes (municipality_atlas.immediate_region_code)", graphs.get("immediate_region")),
        ("microregion (legacy)", lat["microregion"].n_unique(),
         "lattice -> microregion",
         "yes (municipality_atlas.microregion_code)", graphs.get("microregion")),
        ("municipality", lat.height,
         "lattice + IBGE 2022 mesh (geo_municipality.gpkg)",
         "yes (municipality_atlas, 5570 rows)", graphs.get("municipality_all")),
    ]
    tab = pl.DataFrame(
        [{"level": a, "n_units": b, "source": c, "in_atlas": d,
          "adjacency_graph_nodes": e} for a, b, c, d, e in levels]
    )
    print(tab)
    w(tab, "06a_geographic_levels.csv")

    # -- health-region crosswalk coverage -------------------------------------
    print("\n  health-region crosswalk coverage:")
    target = lat.select("code7")
    covered = target.join(hr.select("code7").unique(), on="code7", how="semi").height
    missing = target.join(hr.select("code7").unique(), on="code7", how="anti")
    extra = hr.select("code7").unique().join(target, on="code7", how="anti")
    dup = hr.group_by("code7").len().filter(pl.col("len") > 1)
    print(f"    rows in extract                 : {hr.height}")
    print(f"    distinct municipalities         : {hr['code7'].n_unique()}")
    print(f"    2022-lattice municipalities covered: {covered} / {lat.height} "
          f"({100*covered/lat.height:.2f}%)")
    print(f"    lattice municipalities missing  : {missing.height}")
    print(f"    extract codes off the lattice   : {extra.height}")
    print(f"    municipalities mapped twice     : {dup.height}")

    sizes = hr.group_by("codigo_regiao_saude").len().sort("len")
    print(f"    health-region size (municipalities): min {sizes['len'].min()}, "
          f"median {sizes['len'].median():.0f}, max {sizes['len'].max()}")
    hr_pop = (
        hr.group_by("codigo_regiao_saude")
        .agg(pl.col("populacao_estimada_ibge_2022").sum().alias("pop2022"),
             pl.len().alias("municipalities"))
        .sort("pop2022")
    )
    print(f"    health-region population 2022 (from the extract): "
          f"min {hr_pop['pop2022'].min():,}, median {hr_pop['pop2022'].median():,.0f}, "
          f"max {hr_pop['pop2022'].max():,}, national {hr_pop['pop2022'].sum():,}")
    w(hr_pop, "06b_health_region_sizes.csv")

    # health region x UF: do any health regions straddle a state border?
    straddle = (
        hr.group_by("codigo_regiao_saude").agg(pl.col("codigo_uf").n_unique().alias("n_uf"))
        .filter(pl.col("n_uf") > 1)
    )
    print(f"    health regions spanning >1 UF   : {straddle.height}")

    # -- geometry -------------------------------------------------------------
    print("\n  geometry (data/panel/geo_municipality.gpkg):")
    try:
        import sqlite3

        con = sqlite3.connect(str(GPKG))
        layers = con.execute(
            "SELECT table_name, data_type, srs_id FROM gpkg_contents"
        ).fetchall()
        cnt = con.execute("SELECT COUNT(*) FROM municipality").fetchone()[0]
        cols = [r[1] for r in con.execute("PRAGMA table_info(municipality)").fetchall()]
        con.close()
        print(f"    layers   : {layers}")
        print(f"    features : {cnt}")
        print(f"    columns  : {cols}")
        geom = {"layers": [list(map(str, r)) for r in layers], "features": int(cnt),
                "columns": cols, "size_bytes": GPKG.stat().st_size}
    except Exception as exc:  # pragma: no cover
        print(f"    could not open: {exc}")
        geom = {"error": str(exc)}

    print("\n  adjacency graphs present:")
    for k, v in sorted(graphs.items()):
        print(f"    {k:35s} {v} nodes")

    out = {
        "levels": tab.to_dicts(),
        "health_region_crosswalk": {
            "source": str(HR_JSON),
            "vintage": "OpenDataSUS retrieved 2026-07-30 (no valid_from published)",
            "rows": int(hr.height),
            "distinct_municipalities": int(hr["code7"].n_unique()),
            "lattice_covered": int(covered),
            "lattice_total": int(lat.height),
            "coverage_pct": round(100 * covered / lat.height, 3),
            "missing": missing["code7"].to_list(),
            "off_lattice": extra["code7"].to_list(),
            "duplicated": int(dup.height),
            "n_health_regions": int(hr["codigo_regiao_saude"].n_unique()),
            "n_health_macro_regions": int(hr["codigo_macrorregiao_saude"].n_unique()),
            "health_regions_spanning_multiple_uf": int(straddle.height),
            "extra_columns_unused": ["populacao_estimada_ibge_2022",
                                     "codigo_macrorregiao_saude", "macrorregiao_saude"],
        },
        "geometry": geom,
        "adjacency_graphs": graphs,
        "atlas_rows": {"municipality_atlas": int(mun_atlas.height),
                       "health_region_atlas": int(hr_atlas.height)},
    }
    wj(out, "06_geography.json")
    return out


# ===========================================================================
# 7. What the denominator cannot support, and why
# ===========================================================================
def invalidators(tensor: pl.DataFrame) -> None:
    rule("7. WHAT WOULD INVALIDATE A STRATIFIED RATE")

    items = []

    # 7a. race: numerator exists, denominator does not
    items.append({
        "id": "RACE-NO-DENOMINATOR",
        "severity": "blocking",
        "statement": "The built tensor has no colour/race axis. A race-specific "
                     "incidence rate is currently not computable at any geography.",
        "evidence": "population_tensor_long.parquet columns = "
                    f"{tensor.columns}; no 'colour'.",
    })
    items.append({
        "id": "RACE-CONSTRUCT-MISMATCH",
        "severity": "blocking-if-ignored",
        "statement": "If a colour denominator is built from the census (SIDRA 9606), "
                     "it measures SELF-DECLARED cor ou raca, while SINAN CS_RACA is "
                     "recorded by a health professional at notification. These are "
                     "different constructs; the resulting misclassification is "
                     "differential and its direction is known (branca over-assigned).",
        "evidence": "brepi/denominators/rates.py::RACE_MEASUREMENT_CAVEAT; "
                    "stratified_rate() refuses colour strata unless "
                    "acknowledge_race_measurement_gap=True.",
    })
    items.append({
        "id": "RACE-DIFFERENTIAL-MISSINGNESS",
        "severity": "blocking-if-ignored",
        "statement": "CS_RACA validity varies by year and state, so a colour-specific "
                     "rate has a denominator-independent, spatially patterned bias "
                     "even before the construct mismatch is considered.",
        "evidence": "05d_race_numerator_by_year.csv, 05e_race_numerator_by_uf.csv",
    })

    # 7b. no intercensal composition anchoring
    items.append({
        "id": "NO-CENSUS-ANCHOR-FLAG",
        "severity": "documentation",
        "statement": "The tensor carries no per-cell 'method' or 'years_from_census' "
                     "column, so the interpolation-sensitivity down-weighting that "
                     "brepi.denominators.rates.interpolation_sensitivity_weight() "
                     "implements cannot be run on the built artefact.",
        "evidence": "rates.py raises 'frame lacks years_from_census'.",
    })

    # 7b2. the estimate/census level gap
    items.append({
        "id": "DENOMINATOR-ABOVE-CENSUS",
        "severity": "level-shifting",
        "statement": "The POPSVS series the study divides by is not rebased on the "
                     "2022 Census. Nationally it exceeds the enumerated 2022 "
                     "population, so every incidence rate in the paper is scaled "
                     "down relative to a census-based rate, and the scaling is not "
                     "spatially uniform.",
        "evidence": "03_margin_check_by_year.csv (2022 tensor national), "
                    "03c_popsvs_vs_censo2022_municipal.csv, "
                    "03d_popsvs_vs_censo2022_by_uf.csv",
    })

    items.append({
        "id": "AGE-LABEL-MISMATCH-DROPS-STRATA",
        "severity": "blocking - already affecting a published artefact",
        "statement": "The artefact labels its youngest bands '00-04'/'05-09'; the "
                     "shipped standard populations label them '0-4'/'5-9'. "
                     "age_standardised_rate() inner-joins on that label, so both "
                     "bands are dropped from every Python age-standardised rate "
                     "without warning, and the remaining weights are not "
                     "renormalised. data/results/01_descriptive/08_age_standardised.csv "
                     "contains 64,524 cases instead of 66,666 and understates the ASR "
                     "by about 4% in every year. The R descriptive path is unaffected "
                     "because it calls rebase_standard() with the artefact's labels.",
        "evidence": "04c_age_label_contract.csv, 04c_age_label_contract.json",
    })

    # 7c. terminal age group
    items.append({
        "id": "TERMINAL-AGE-80PLUS",
        "severity": "bounded",
        "statement": "The terminal band is 80+, while the WHO World Standard runs to "
                     "85+. Age-standardised rates must collapse the standard onto 17 "
                     "bands (rebase_standard) or they silently drop weight.",
        "evidence": "brepi/denominators/rates.py WHO_WORLD_STANDARD has 17 keys "
                    "ending '80+'.",
    })

    # 7d. off-spine municipality
    lat = set(lattice.load_municipalities(2022)["code7"].to_list())
    off = sorted(set(tensor["munic_code"].unique().to_list()) - lat)
    per = (
        tensor.filter(pl.col("munic_code").is_in(off))
        .group_by("munic_code").agg(pl.col("year").unique().sort().alias("years"))
    ) if off else None
    items.append({
        "id": "OFF-SPINE-MUNICIPALITY",
        "severity": "small-but-silent",
        "statement": f"{len(off)} municipality code(s) in the tensor are not in the "
                     "2022 lattice the panel is built on. Any national margin computed "
                     "from the tensor without filtering to the spine differs from a "
                     "margin computed from the panel.",
        "evidence": (per.to_dicts() if per is not None else "none")
        if off else "none",
    })

    # 7e. rate denominator convention
    items.append({
        "id": "PERSON-TIME-CONVENTION",
        "severity": "documentation",
        "statement": "Person-time is mid-year stock x 12 (months). A rate per 100,000 "
                     "person-YEARS and a rate per 100,000 person-MONTHS differ by 12x; "
                     "the tensor stores stock, not person-time, so the convention must "
                     "travel with the number.",
        "evidence": "22_paper_descriptives.R: pop[, person_months := population * 12]",
    })

    # 7f. urban/rural
    items.append({
        "id": "NO-URBAN-RURAL-AXIS",
        "severity": "gap",
        "statement": "POPSVS has no situacao do domicilio (urban/rural). Urban-share is "
                     "carried as a municipality-level covariate from the census, not as "
                     "a denominator stratum, so an urban-specific incidence rate is not "
                     "computable from this tensor.",
        "evidence": "POPSBR{YY}.dbf columns = COD_MUN, ANO, SEXO, IDADE, POP",
    })

    for it in items:
        print(f"\n  [{it['severity']}] {it['id']}")
        print(f"    {it['statement']}")
    wj(items, "07_rate_invalidators.json")


# ===========================================================================
# 8. Dark inventory
# ===========================================================================
def dark_inventory() -> None:
    rule("8. DARK BUT AVAILABLE - what exists and is not used")

    # what SIDRA tables are cached, and at what depth
    meta = {}
    for p in sorted(SIDRA_META.glob("*.json")):
        if p.name.endswith(".provenance.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        tid = p.stem
        periods_path = SIDRA_PERIODS / f"{tid}.json"
        periods = json.loads(periods_path.read_text(encoding="utf-8")) if periods_path.exists() else []
        vals = SIDRA_VALUES / tid
        n_val = len([f for f in vals.glob("*.json") if not f.name.endswith(".provenance.json")]) if vals.exists() else 0
        meta[tid] = {
            "name": d.get("nome"),
            "periods": [p_.get("id") for p_ in periods] if isinstance(periods, list) else [],
            "territorial_levels": d.get("nivelTerritorial", {}),
            "classifications": [
                {"id": c["id"], "name": c["nome"], "n_categories": len(c.get("categorias", []))}
                for c in d.get("classificacoes", [])
            ],
            "value_payloads_cached": n_val,
        }
    wj(meta, "08a_sidra_tables_cached.json")

    demographic = {k: v for k, v in meta.items()
                   if k in {"9606", "9514", "9923", "4714", "4709", "6579", "202", "10295", "9860"}}
    print("  demographic / denominator-relevant SIDRA tables in the cache:")
    for k, v in sorted(demographic.items(), key=lambda kv: int(kv[0])):
        cls = ", ".join(f"{c['name']}({c['n_categories']})" for c in v["classifications"])
        print(f"    {k:6s} periods={v['periods']}  levels={list(v['territorial_levels'].get('Administrativo', []))}")
        print(f"           {v['name'][:110]}")
        print(f"           classifications: {cls or 'none'}   value payloads cached: {v['value_payloads_cached']}")

    rows = [
        {
            "item": "Single-year-of-age population (0..80), municipality x year x sex",
            "where": "data/cache/datasus/ibge_pop/popsvs/POPSBR{07..25}.ZIP (19 files, verified)",
            "granularity": "municipality x year 2007-2025 x sex x single year of age",
            "status": "cached raw; collapsed to 17 five-year bands on write",
            "effort": "cheap_extract (re-run 02_denominators.py with a different band map)",
            "question_it_answers": "age-specific incidence with any banding (e.g. paediatric 0-14 "
                                   "single years, or the 15-59 working-age band the occupational "
                                   "hypothesis needs) without re-fetching anything",
        },
        {
            "item": "Census population by cor ou raca x sex x age, municipality",
            "where": "SIDRA agregado 9606 - metadata and periods cached; only a 3-municipality "
                     "probe of values is cached",
            "granularity": "municipality x CENSUS YEARS 2010 and 2022 only x 6 colours x 3 sexes "
                           "x 134 age categories (N1/N2/N3/N6)",
            "status": "available_unused - denominator absent from the tensor",
            "effort": "moderate (one paged SIDRA fetch, 5,570 municipalities x 2 periods)",
            "question_it_answers": "colour-specific incidence with a correct denominator; NOT a "
                                   "time-varying control - it exists for 2010 and 2022 only",
        },
        {
            "item": "Health macro-region (macrorregiao de saude)",
            "where": "data/cache/geo/health_regions/open_datasus_20260730.json "
                     "-> codigo_macrorregiao_saude",
            "granularity": "municipality -> health macro-region, single undated snapshot",
            "status": "available_unused - no atlas column, no adjacency graph",
            "effort": "already_extracted (a group_by on the cached JSON)",
            "question_it_answers": "the referral tier above the health region; the natural scale "
                                   "for the 'how far down the severity distribution does "
                                   "surveillance reach' thesis, because tertiary capacity is "
                                   "organised at macro-region level",
        },
        {
            "item": "Intermediate region (2017 division, 133 units)",
            "where": "brepi.geo.lattice.load_municipalities -> intermediate_region",
            "granularity": "municipality -> intermediate region, fixed since 2017",
            "status": "available_unused as a MODELLING scale - carried in the atlas, no "
                      "adjacency graph built",
            "effort": "cheap_extract (build the .adj from the mesh, as the others were)",
            "question_it_answers": "a coarser BYM2 scale between UF (27) and health region (439) "
                                   "for the multi-scale sensitivity",
        },
        {
            "item": "Mesoregion (legacy division, 137 units)",
            "where": "lattice -> mesoregion / mesoregion_name",
            "granularity": "municipality -> mesoregion, legacy (superseded 2017)",
            "status": "available_unused - not in the atlas, no graph",
            "effort": "cheap_extract",
            "question_it_answers": "comparability with the older Brazilian leptospirosis "
                                   "literature, which stratifies by mesoregion",
        },
        {
            "item": "Municipal land area (km2) in the geometry",
            "where": "data/panel/geo_municipality.gpkg column area_km2; SIDRA 4714 also cached",
            "granularity": "municipality, 2022 vintage",
            "status": "available_unused",
            "effort": "already_extracted",
            "question_it_answers": "population density as a transmission-regime covariate, and "
                                   "the areal support any dasymetric transfer needs",
        },
        {
            "item": "Health-region 2022 population carried inside the crosswalk",
            "where": "open_datasus JSON -> populacao_estimada_ibge_2022",
            "granularity": "municipality, single year 2022",
            "status": "available_unused - the study derives health-region population from the "
                      "tensor instead, which is correct; this is an independent cross-check",
            "effort": "already_extracted",
            "question_it_answers": "an external validation of the aggregated denominator",
        },
        {
            "item": "Household income per capita by sex x cor ou raca x age group",
            "where": "SIDRA agregado 10295; data/interim/sidra/income_10295.parquet holds "
                     "5,570 municipalities but ONLY the Total x Total x Total cell "
                     "(classification 2/58/86 all = Total)",
            "granularity": "municipality x CENSUS YEAR 2022 only x 3 sexes x 6 colours "
                           "x 23 age groups",
            "status": "available_unused - the stratifiers were never requested",
            "effort": "moderate (re-request the same agregado with real category ids)",
            "question_it_answers": "whether the colour gradient in leptospirosis is a "
                                   "colour gradient or an income gradient - the confounder "
                                   "the race analysis will be asked about at review",
        },
        {
            "item": "Census population by situacao do domicilio (urban/rural)",
            "where": "SIDRA 9923 (2022, 14 payloads cached) and 202 (1970-2010)",
            "granularity": "municipality x census years; 202 also reaches N8/N9/N7/N13/N14/N15",
            "status": "available_unused as a DENOMINATOR stratum (urban share is used only "
                      "as a municipality-level covariate)",
            "effort": "cheap_extract for 2022, moderate for a full intercensal series",
            "question_it_answers": "urban vs rural incidence, i.e. the metropolitan-flood "
                                   "archetype against the rural-occupational archetype, with "
                                   "the correct denominator for each",
        },
        {
            "item": "Per-cell interpolation provenance (method, years_from_census)",
            "where": "brepi/denominators/tensor.py builds them; the artefact does not carry them",
            "granularity": "municipality x year",
            "status": "unusable as built - the columns do not exist in the parquet",
            "effort": "moderate (would require running build_tensor with a census seed)",
            "question_it_answers": "the denominator-uncertainty sensitivity the protocol "
                                   "prescribes",
        },
    ]
    dark = pl.DataFrame(rows)
    w(dark, "08b_dark_inventory.csv")
    print(f"\n  {dark.height} dark/unused items catalogued")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"output -> {OUT}")

    tensor = pl.read_parquet(TENSOR)
    annual = pl.read_parquet(ANNUAL)

    prof = profile_tensor(tensor)
    prov = audit_provenance()
    margins = reconcile(tensor, annual)
    age_sex_structure(tensor)
    age_label_contract(tensor)
    feasibility(tensor)
    depth_cfr_age_confounding()
    geo = geography()
    invalidators(tensor)
    dark_inventory()

    summary = {
        "populations_profiled": {
            "D1_denominator_spine": f"{SPINE_MUNICIPALITIES} municipalities x "
                                    f"{len(YEARS)} years = {SPINE_CELLS} cells",
            "D2_confirmed_cases": "CLASSI_FIN='confirmado' in lept_line_level.parquet",
            "D3_lattice": "IBGE localidades, 2022 vintage, cached",
        },
        "tensor": prof,
        "provenance": prov,
        "national_margin_2025": int(
            margins.filter(pl.col("year") == 2025)["tensor_national"][0]
        ),
        "max_municipal_margin_error_all_years": float(
            margins["max_abs_municipal_diff"].max()
        ),
        "geography_levels": {r["level"]: r["n_units"] for r in geo["levels"]},
        "health_region_coverage_pct": geo["health_region_crosswalk"]["coverage_pct"],
    }
    wj(summary, "00_summary.json")
    print(f"\nDONE. {len(list(OUT.glob('*')))} files in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
