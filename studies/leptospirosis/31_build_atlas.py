"""Analytic atlases - every answer, joined onto one spine per geographic grain.

The study's findings finish scattered across the stage directories: incidence in
``01_descriptive``, a surveillance classification in ``rq3_ascertainment``, a
regime label in ``rq5_regimes``, a latent spatial field in
``rq1_exposure_response``, structural covariates only in the panel. Each keys on
the same unit. Anyone making a map, a scatter, or a unit-level table currently
has to perform four to six joins by hand, and each of those joins is an
opportunity to lose rows without noticing -- the municipality code is a
zero-padded string in the panel and an integer in every CSV that has
round-tripped through schema inference.

This script performs those joins once, through :mod:`brepi.analysis.atlas`,
which normalises the key and reports what each join actually matched.

Three atlases, because the study has three natural grains and collapsing them
would either fabricate detail or destroy it:

``municipality_atlas``
    5,570 rows, one per municipality, whole-period summaries. The spine is the
    complete municipal universe, so municipalities with no cases at all are
    present with explicit zeros -- they are the subject of RQ3, not an absence.

``municipality_year_atlas``
    5,570 x 19 rows. Everything that varies in time at municipal grain.

``health_region_atlas``
    One row per health region, carrying RQ1's latent spatial field alongside the
    burden and severity it is meant to explain. This is the only grain at which
    the RQ1 field exists; interpolating it to municipalities would invent
    within-region variation the model never estimated.

Rates are computed **after** aggregation, never averaged from monthly rates, and
carry exact intervals (Poisson for incidence, Clopper-Pearson for proportions)
because the modal municipality here has single-digit counts.

Run:
    python studies/leptospirosis/31_build_atlas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi import paths
from brepi.analysis.atlas import Layer, assemble_atlas
from brepi.analysis.rates import proportion_table, rate_table
from brepi.panel.aggregate import aggregate_panel

OUT = paths.stage("atlas")
RES = paths.RESULTS

GEO_COLS = ["name", "uf_code", "uf_abbr", "region", "microregion_code",
            "immediate_region_code", "intermediate_region",
            "health_region_code", "health_region_name"]


def write(frame: pl.DataFrame, stem: str) -> None:
    """Both formats, deliberately.

    Parquet preserves dtypes for anything that joins downstream; CSV is what a
    person opens. Writing only one of them has, in this project, produced both
    a key that lost its leading zero and a table nobody could read.
    """
    frame.write_parquet(OUT / f"{stem}.parquet")
    frame.write_csv(OUT / f"{stem}.csv")
    print(f"  wrote {stem}: {frame.height} x {frame.width}")


# --------------------------------------------------------------------------
# Municipality-year: the primitives, summed. Everything else derives from here.
# --------------------------------------------------------------------------

def municipality_year(panel: pl.LazyFrame) -> pl.DataFrame:
    """Collapse the month panel to municipality-year by summing primitives.

    Counts and person-time sum; shares and rates do not. Averaging a monthly
    incidence rate over twelve months weights a January of 30,000 people the
    same as a July of 3,000,000 and answers a question no one asked. The
    structural covariates are constant within a municipality-year by
    construction (they are annual anchors broadcast to months), so ``first`` is
    exact rather than a summary.
    """
    out, rep = aggregate_panel(
        panel,
        keys=["munic_code", "year"],
        # person_months holds population/12 per month, so its annual sum is
        # person-years exactly.
        sums=["cases", "deaths", "hospitalised", "criterion_epi",
              "flood_contact", "person_months", "precip_mm", "r50", "r20"],
        # Annual anchors broadcast to months, and geography. Declaring them
        # constant makes the aggregation verify it rather than assume it: a
        # municipality that changed health region mid-year would otherwise be
        # collapsed to whichever row happened to sort first.
        constants=["population", "sanitation_sewer_share", "urban_share",
                   "gdp_per_capita", *GEO_COLS],
        maxima=["rx1day"],
        extra={"months_with_cases": pl.col("any_case").sum(),
               "climate_coverage_mean": pl.col("climate_coverage").mean()},
    )
    if not rep.passed:
        print(rep.format())
    return out.rename({"person_months": "person_years",
                       "precip_mm": "precip_mm_total",
                       "r50": "r50_days", "r20": "r20_days"})


def add_rates(df: pl.DataFrame) -> pl.DataFrame:
    df = rate_table(df, count="cases", person_time="person_years",
                    prefix="incidence_per_100k", scale=1e5)
    df = proportion_table(df, count="deaths", total="cases", prefix="cfr")
    df = proportion_table(df, count="hospitalised", total="cases",
                          prefix="hospitalisation_share")
    return df


# --------------------------------------------------------------------------
# Whole-period municipal summaries
# --------------------------------------------------------------------------

def municipality_totals(my: pl.DataFrame) -> pl.DataFrame:
    tot = (
        my.group_by("munic_code")
        .agg([
            pl.col("cases").sum(),
            pl.col("deaths").sum(),
            pl.col("hospitalised").sum(),
            pl.col("criterion_epi").sum(),
            pl.col("flood_contact").sum(),
            pl.col("person_years").sum(),
            pl.col("population").mean().alias("mean_population"),
            pl.col("population").last().alias("population_final_year"),
            pl.col("precip_mm_total").mean().alias("precip_mm_annual_mean"),
            pl.col("rx1day_max").mean().alias("rx1day_annual_mean"),
            pl.col("r50_days").mean().alias("r50_days_annual_mean"),
            pl.col("r20_days").mean().alias("r20_days_annual_mean"),
            pl.col("sanitation_sewer_share").mean().alias(
                "sanitation_sewer_share_mean"),
            pl.col("sanitation_sewer_share").last().alias(
                "sanitation_sewer_share_final"),
            pl.col("urban_share").mean().alias("urban_share_mean"),
            pl.col("urban_share").last().alias("urban_share_final"),
            pl.col("gdp_per_capita").mean().alias("gdp_per_capita_mean"),
            pl.col("months_with_cases").sum(),
            (pl.col("cases") > 0).sum().alias("years_with_cases"),
            pl.col("year").filter(pl.col("cases") > 0).min().alias("first_case_year"),
            pl.col("year").filter(pl.col("cases") > 0).max().alias("last_case_year"),
            pl.col("year").sort_by("cases", descending=True).first().alias("peak_year"),
            pl.col("cases").max().alias("peak_year_cases"),
            *[pl.col(c).first() for c in GEO_COLS],
        ])
        .sort("munic_code")
    )
    return add_rates(tot)


def seasonal_shape(panel: pl.LazyFrame) -> pl.DataFrame:
    """Per-municipality month-of-year profile, collapsed to two numbers.

    ``peak_month`` is the calendar month carrying most cases; ``seasonal_ratio``
    is peak-month cases over the mean month. Both are undefined for a
    municipality with no cases, and are left null rather than being given the
    value a formula happens to produce at zero.
    """
    m = (panel.group_by(["munic_code", "month"]).agg(pl.col("cases").sum())
         .collect())
    total = m.group_by("munic_code").agg(
        pl.col("cases").sum().alias("total"),
        pl.col("cases").max().alias("peak"))
    peak = (m.sort(["munic_code", "cases", "month"], descending=[False, True, False])
            .group_by("munic_code").first()
            .select(["munic_code", pl.col("month").alias("peak_month")]))
    out = total.join(peak, on="munic_code", how="left")
    return out.with_columns([
        pl.when(pl.col("total") > 0).then(pl.col("peak_month")).alias("peak_month"),
        pl.when(pl.col("total") > 0)
          .then(pl.col("peak") / (pl.col("total") / 12.0))
          .alias("seasonal_ratio"),
    ]).select(["munic_code", "peak_month", "seasonal_ratio"])


def serogroup_layers(line: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Serological composition by municipality of residence.

    MAT typing reaches roughly three per cent of notifications, so a municipal
    composition is meaningful only where several isolates exist. The long table
    keeps every non-zero cell with its denominator so a reader can apply their
    own threshold; the wide summary carries the modal serogroup and how thin the
    evidence for it is, which is the number that decides whether it may be
    plotted at all.
    """
    typed = (
        line.select([
            pl.col("municipality_residence_code7").alias("munic_code"),
            pl.col("serovar_s1_first_serogroup").alias("serogroup"),
            pl.col("serovar_s1_first_reservoir").alias("reservoir"),
        ])
        .filter(pl.col("munic_code").is_not_null()
                & pl.col("serogroup").is_not_null())
        .collect()
    )
    long = (typed.group_by(["munic_code", "serogroup"]).len().rename({"len": "n"})
            .join(typed.group_by("munic_code").len().rename({"len": "typed_total"}),
                  on="munic_code", how="left")
            .with_columns((pl.col("n") / pl.col("typed_total")).alias("share"))
            .sort(["munic_code", "n"], descending=[False, True]))
    top = (long.group_by("munic_code").first()
           .select([
               "munic_code",
               pl.col("serogroup").alias("serogroup_modal"),
               pl.col("n").alias("serogroup_modal_n"),
               pl.col("share").alias("serogroup_modal_share"),
               pl.col("typed_total").alias("serotyped_isolates"),
           ]))
    res = (typed.filter(pl.col("reservoir").is_not_null())
           .group_by(["munic_code", "reservoir"]).len()
           .sort(["munic_code", "len"], descending=[False, True])
           .group_by("munic_code").first()
           .select(["munic_code", pl.col("reservoir").alias("reservoir_modal")]))
    return long, top.join(res, on="munic_code", how="left")


def disaster_layer() -> pl.DataFrame:
    fl = pl.read_parquet(paths.FLOOD_EVENTS)
    dr = pl.read_parquet(paths.DROUGHT_EVENTS)
    g = pl.read_parquet(paths.FLOOD_FIRST_TREATMENT)
    f = fl.group_by("munic_code").agg([
        pl.len().alias("flood_declarations"),
        pl.col("recognised").sum().alias("flood_declarations_recognised"),
        pl.col("affected_total").sum().alias("flood_affected_total"),
    ])
    d = dr.group_by("munic_code").agg(pl.len().alias("drought_declarations"))
    return (f.join(d, on="munic_code", how="full", coalesce=True)
            .join(g.select(["munic_code", "ever_treated", "g_year", "n_episodes"])
                  .rename({"g_year": "flood_first_treatment_year",
                           "n_episodes": "flood_episodes",
                           "ever_treated": "rq2_ever_treated"}),
                  on="munic_code", how="full", coalesce=True))


# --------------------------------------------------------------------------
# Health region
# --------------------------------------------------------------------------

def health_region_atlas(my: pl.DataFrame) -> pl.DataFrame:
    hr = (
        my.group_by("health_region_code")
        .agg([
            pl.col("health_region_name").first(),
            pl.col("uf_abbr").first(),
            pl.col("region").first(),
            pl.col("munic_code").n_unique().alias("municipalities"),
            pl.col("cases").sum(),
            pl.col("deaths").sum(),
            pl.col("hospitalised").sum(),
            pl.col("person_years").sum(),
            pl.col("population").mean().alias("mean_population"),
            pl.col("sanitation_sewer_share").mean().alias(
                "sanitation_sewer_share_mean"),
            pl.col("urban_share").mean().alias("urban_share_mean"),
            pl.col("gdp_per_capita").mean().alias("gdp_per_capita_mean"),
            pl.col("precip_mm_total").mean().alias("precip_mm_annual_mean"),
        ])
        .sort("health_region_code")
    )
    hr = add_rates(hr)

    layers: list[Layer] = []
    field = RES / "rq1_exposure_response/health_region_confirmatory/spatial_field.csv"
    if field.exists():
        f = pl.read_csv(field, infer_schema_length=None)
        # The exported field is already the combined BYM2 effect -- the thing to
        # map. Filtering anyway, so that adding the structured component to the
        # export later cannot silently double this layer's rows.
        if "component" in f.columns:
            f = f.filter(pl.col("component") == "combined")
        keep = [c for c in ("effect", "effect_lo", "effect_hi", "mean", "sd")
                if c in f.columns]
        layers.append(Layer(
            "rq1_spatial_field",
            f.select(["health_region_code", *keep]),
            rename={c: f"rq1_spatial_{c}" for c in keep}))

    leth = RES / "rq4_lethality/rq4_lethality_panel.csv"
    if leth.exists():
        p = pl.read_csv(leth, infer_schema_length=None)
        agg = p.group_by("health_region_code").agg([
            pl.col("outcome_known").sum().alias("rq4_outcome_known"),
            pl.col("lab_share").mean().alias("rq4_lab_share_mean"),
            pl.col("median_delay").median().alias("rq4_median_delay"),
            pl.col("completeness").mean().alias("rq4_completeness_mean"),
        ])
        layers.append(Layer("rq4_lethality", agg))

    atlas, rep = assemble_atlas(hr, layers, key="health_region_code")
    print(rep.format())
    return atlas


# --------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pl.scan_parquet(paths.PANEL_ANALYSIS)
    line = pl.scan_parquet(paths.LINE_LEVEL)

    print("municipality-year")
    my = municipality_year(panel)
    my_rated = add_rates(my)

    tri = paths.PANEL / "triangulation_municipality_year.parquet"
    my_layers: list[Layer] = []
    if tri.exists():
        t = pl.read_parquet(tri).with_columns(
            (pl.col("munic_code") + "|" + pl.col("year").cast(pl.Utf8)).alias("_k"))
        my_rated = my_rated.with_columns(
            (pl.col("munic_code") + "|" + pl.col("year").cast(pl.Utf8)).alias("_k"))
        my_layers.append(Layer(
            "triangulation",
            t.select(["_k", "sim_a27_deaths", "sih_a27_admissions",
                      "sih_a27_principal", "sih_a27_deaths_in_hospital",
                      "sinan_covered", "sim_covered", "sih_covered"])))
        my_atlas, rep = assemble_atlas(my_rated, my_layers, key="_k")
        print(rep.format())
        my_atlas = my_atlas.drop("_k")
    else:
        my_atlas = my_rated
    write(my_atlas, "municipality_year_atlas")

    print("\nmunicipality")
    tot = municipality_totals(my)
    long, sero = serogroup_layers(line)

    layers = [
        Layer("seasonality", seasonal_shape(panel)),
        Layer("serology", sero),
        Layer("disasters", disaster_layer()),
    ]
    surv = RES / "rq3_ascertainment/16_surveillance_completeness_map.csv"
    if surv.exists():
        s = pl.read_csv(surv, infer_schema_length=None)
        layers.append(Layer(
            "rq3_surveillance",
            s.select(["munic_code", "sim_a27_deaths", "sih_a27_admissions",
                      "sinan_positive", "other_positive", "surveillance_class",
                      "sinan_per_independent_signal"]),
            rename={"sim_a27_deaths": "sim_a27_deaths_total",
                    "sih_a27_admissions": "sih_a27_admissions_total"}))
    reg = RES / "rq5_regimes/regime_assignment.csv"
    if reg.exists():
        r = pl.read_csv(reg, infer_schema_length=None)
        layers.append(Layer("rq5_regime", r.select(["munic_code", "regime"]),
                            rename={"regime": "rq5_regime"}))

    atlas, rep = assemble_atlas(tot, layers, key="munic_code", key_width=7)
    print(rep.format())

    atlas = atlas.with_columns([
        pl.col("rq5_regime").is_not_null().alias("rq5_eligible"),
        pl.col("rq2_ever_treated").fill_null(False),
        pl.col("flood_declarations").fill_null(0),
        pl.col("drought_declarations").fill_null(0),
        pl.col("serotyped_isolates").fill_null(0),
    ])
    write(atlas, "municipality_atlas")
    write(long.with_columns(pl.col("munic_code").str.zfill(7)),
          "municipality_serogroup_long")

    print("\nhealth region")
    write(health_region_atlas(my), "health_region_atlas")

    manifest = {
        "municipality_atlas": {"grain": "municipality", "key": ["munic_code"],
                               "rows": atlas.height, "columns": atlas.width},
        "municipality_year_atlas": {"grain": "municipality_year",
                                    "key": ["munic_code", "year"],
                                    "rows": my_atlas.height,
                                    "columns": my_atlas.width},
        "municipality_serogroup_long": {"grain": "municipality_serogroup",
                                        "key": ["munic_code", "serogroup"],
                                        "rows": long.height},
        "health_region_atlas": {"grain": "health_region",
                                "key": ["health_region_code"]},
    }
    (OUT / "atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
