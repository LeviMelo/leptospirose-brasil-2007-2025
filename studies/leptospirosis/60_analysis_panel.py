"""The analysis panel: health region by year, with every field rule made explicit.

This replaces the cross-sectional collapse the first draft relied on. The panel
is the object every downstream analysis reads, so the operational definitions
live here in one place rather than being re-derived per script.

**The construct, stated once and not reversed.** The exposure is

    H = hospitalised confirmed cases / confirmed cases

which is an **inverse** indicator of how broadly surveillance samples the
clinical spectrum. A territory recording nine in ten confirmed cases as
hospitalised is finding almost only severe illness; one recording half is also
finding mild illness. Higher H therefore means NARROWER ascertainment. The
first draft called H "profundidade de detecção" and then described a high value
as *menor* profundidade, which is self-contradictory. The word is abandoned.
Where a breadth-oriented scale is wanted, use B = 1 - H, reported alongside.

**Every field rule.** Four decode states exist upstream (valid / unknown /
missing / invalid) and are not interchangeable. The rules applied here:

| Quantity | Field | Accepted | Denominator |
|---|---|---|---|
| Confirmed case | `classi_fin` | `confirmado` | all notifications |
| Window | `epiweek_onset_year` | 2007–2025, from **symptom onset** | confirmed |
| Place | `municipality_residence_code7` | residence, not notification or hospital | confirmed |
| Death | `evolucao` | `obito_por_leptospirose` only | confirmed with `evolucao_state == valid` |
| Hospitalised | `ate_hosp` | `sim` | confirmed with `ate_hosp_state == valid` |
| Laboratory | `criterio` | `clinico_laboratorial` | confirmed with `criterio_state == valid` |
| Severe phenotype | `cli_icteri`, `cli_renal`, `cli_hemorr` | any `sim` | confirmed with all three decoded |

`obito_por_outras_causas` is **not** a leptospirosis death and is counted as a
known outcome that is not a case fatality. A blank is never read as a negative:
every proportion above is computed on its own valid denominator, and those
denominators are carried in the panel so completeness is auditable per cell.

Outputs to ``data/results/analysis_panel/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS

OUT = PATHS.results / "analysis_panel"
YEAR_MIN, YEAR_MAX = 2007, 2025

#: Age cut for the case-mix covariate. Leptospirosis case fatality rises
#: steeply with age; a difference in median age of two years can hide a large
#: difference in the tail that actually carries the deaths, which is why the
#: first draft's "median age varies little" argument did not address the
#: confounder it claimed to address.
AGE_CUT = 60


def _study_flow() -> pl.DataFrame:
    """The canonical cascade, reconciling to the last record.

    The first draft reported 66,516 confirmed cases in the analytic window,
    from a single exclusion row of 151. That conflated two different
    exclusions and got neither: 155 confirmed records carry an **invalid**
    symptom-onset date and so cannot be placed in any year, and a further 105
    have a valid onset year outside 2007–2025. Treating the undatable records
    as in-window inflated the denominator by 109. Both exclusions are now
    counted separately and the identities are asserted below.
    """
    src = pl.scan_parquet(PATHS.interim / "lept_line_level.parquet").select(
        "classi_fin", "epiweek_onset_year", "epiweek_onset_state",
        "municipality_residence_code7",
    ).collect()

    n_total = src.height
    cls = src["classi_fin"]
    n_disc = int((cls == "descartado").sum())
    n_inc = int((cls == "inconclusivo").sum())
    n_none = int(cls.null_count())
    conf = src.filter(pl.col("classi_fin") == "confirmado")
    n_conf = conf.height
    assert n_disc + n_inc + n_none + n_conf == n_total

    n_undatable = int(conf["epiweek_onset_year"].null_count())
    dated = conf.filter(pl.col("epiweek_onset_year").is_not_null())
    n_outside = int((~dated["epiweek_onset_year"].is_between(YEAR_MIN, YEAR_MAX)).sum())
    inwin = dated.filter(pl.col("epiweek_onset_year").is_between(YEAR_MIN, YEAR_MAX))
    n_inwin = inwin.height
    assert n_undatable + n_outside + n_inwin == n_conf

    n_nogeo = int(inwin["municipality_residence_code7"].null_count())
    n_final = n_inwin - n_nogeo

    rows = [
        ("notifications in the SINAN files", n_total),
        ("discarded", n_disc),
        ("inconclusive", n_inc),
        ("no final classification", n_none),
        ("confirmed (CLASSI_FIN = confirmado)", n_conf),
        ("confirmed, excluded: symptom-onset date invalid", n_undatable),
        ("confirmed, excluded: onset outside 2007-2025", n_outside),
        ("confirmed, in the analytic window", n_inwin),
        ("in window, excluded: no residence municipality", n_nogeo),
        ("ANALYTIC POPULATION", n_final),
    ]
    return pl.DataFrame(
        {"step": [r[0] for r in rows], "n": [r[1] for r in rows]}
    ).with_columns((100 * pl.col("n") / n_total).round(2).alias("pct_of_notifications"))


def _confirmed() -> pl.DataFrame:
    cols = [
        "classi_fin", "epiweek_onset_year", "municipality_residence_code7",
        "evolucao", "evolucao_state", "ate_hosp", "ate_hosp_state",
        "criterio", "criterio_state", "age_years", "age_state", "cs_sexo",
        "cli_icteri", "cli_icteri_state", "cli_renal", "cli_renal_state",
        "cli_hemorr", "cli_hemorr_state",
    ]
    return (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .select(cols)
        .filter(
            (pl.col("classi_fin") == "confirmado")
            & pl.col("epiweek_onset_year").is_between(YEAR_MIN, YEAR_MAX)
            & pl.col("municipality_residence_code7").is_not_null()
        )
        .collect()
    )


def _flag(field: str, state: str, value: str) -> pl.Expr:
    return ((pl.col(state) == "valid") & (pl.col(field) == value)).cast(pl.Int32)


def _valid(state: str) -> pl.Expr:
    return (pl.col(state) == "valid").cast(pl.Int32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flow = _study_flow()
    flow.write_csv(OUT / "study_flow.csv")
    print(flow)
    line = _confirmed()

    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    geo = atlas.select(
        pl.col("munic_code").alias("municipality_residence_code7"),
        "health_region_code", "health_region_name", "uf_abbr", "region",
    )
    line = line.join(geo, on="municipality_residence_code7", how="left")
    unmatched = line.filter(pl.col("health_region_code").is_null()).height
    assert unmatched / line.height < 0.01, f"{unmatched} cases without a health region"
    line = line.filter(pl.col("health_region_code").is_not_null())

    severe_decoded = (
        (pl.col("cli_icteri_state") == "valid")
        & (pl.col("cli_renal_state") == "valid")
        & (pl.col("cli_hemorr_state") == "valid")
    )
    line = line.with_columns(
        _flag("evolucao", "evolucao_state", "obito_por_leptospirose").alias("death"),
        _valid("evolucao_state").alias("outcome_known"),
        _flag("ate_hosp", "ate_hosp_state", "sim").alias("hosp"),
        _valid("ate_hosp_state").alias("hosp_known"),
        _flag("criterio", "criterio_state", "clinico_laboratorial").alias("lab"),
        _valid("criterio_state").alias("crit_known"),
        severe_decoded.cast(pl.Int32).alias("sev_known"),
        (
            severe_decoded
            & (
                (pl.col("cli_icteri") == "sim")
                | (pl.col("cli_renal") == "sim")
                | (pl.col("cli_hemorr") == "sim")
            )
        ).cast(pl.Int32).alias("severe"),
        ((pl.col("age_state") == "valid") & (pl.col("age_years") >= AGE_CUT))
        .cast(pl.Int32).alias("age60"),
        _valid("age_state").alias("age_known"),
        (pl.col("cs_sexo") == "masculino").cast(pl.Int32).alias("male"),
    )
    # Outcome split by hospitalisation status. This is what makes the
    # shared-denominator problem testable: if the case-fatality gradient is a
    # denominator effect, case fatality WITHIN the hospitalised stratum should
    # be far flatter than the overall gradient, because that stratum is not
    # diluted by the mild cases whose detection varies.
    both = (pl.col("evolucao_state") == "valid") & (pl.col("ate_hosp_state") == "valid")
    line = line.with_columns(
        (both & (pl.col("ate_hosp") == "sim")).cast(pl.Int32).alias("hk_hosp"),
        (both & (pl.col("ate_hosp") == "nao")).cast(pl.Int32).alias("hk_nonhosp"),
        (both & (pl.col("ate_hosp") == "sim")
         & (pl.col("evolucao") == "obito_por_leptospirose")).cast(pl.Int32).alias("d_hosp"),
        (both & (pl.col("ate_hosp") == "nao")
         & (pl.col("evolucao") == "obito_por_leptospirose")).cast(pl.Int32).alias("d_nonhosp"),
    )

    keys = ["health_region_code", "health_region_name", "uf_abbr", "region",
            "epiweek_onset_year"]
    panel = line.group_by(keys).agg(
        pl.len().alias("cases"),
        pl.col("death").sum().alias("deaths"),
        pl.col("outcome_known").sum().alias("outcome_known"),
        pl.col("hosp").sum().alias("hospitalised"),
        pl.col("hosp_known").sum().alias("hosp_known"),
        pl.col("lab").sum().alias("lab_confirmed"),
        pl.col("crit_known").sum().alias("crit_known"),
        pl.col("severe").sum().alias("severe"),
        pl.col("sev_known").sum().alias("sev_known"),
        pl.col("age60").sum().alias("age60"),
        pl.col("age_known").sum().alias("age_known"),
        pl.col("male").sum().alias("male"),
        pl.col("hk_hosp").sum().alias("hk_hosp"),
        pl.col("hk_nonhosp").sum().alias("hk_nonhosp"),
        pl.col("d_hosp").sum().alias("d_hosp"),
        pl.col("d_nonhosp").sum().alias("d_nonhosp"),
    ).rename({"epiweek_onset_year": "year"})

    # Person-time for the same region-year.
    #
    # The panel MUST be a complete region x year grid, not the set of cells that
    # happen to contain a case. Aggregating only the cells with cases silently
    # drops the person-time of every quiet year, which understates the
    # denominator and inflates every incidence and mortality rate — and it does
    # so unevenly, because territories with narrow ascertainment have more
    # zero-case years. Building the grid first and filling counts with zero is
    # the difference between a rate and a ratio of convenience.
    pop = (
        pl.read_parquet(PATHS.interim / "population_municipal_year.parquet")
        .join(
            atlas.select(pl.col("munic_code"), "health_region_code"),
            on="munic_code", how="inner",
        )
        .filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        .group_by(["health_region_code", "year"])
        .agg(pl.col("population").sum().alias("person_years"))
    )

    labels = panel.select(
        "health_region_code", "health_region_name", "uf_abbr", "region"
    ).unique(subset="health_region_code")
    grid = pop.join(labels, on="health_region_code", how="inner")

    count_cols = [c for c in panel.columns if c not in
                  ("health_region_code", "health_region_name", "uf_abbr", "region", "year")]
    panel = (
        grid.join(
            panel.select(["health_region_code", "year", *count_cols]),
            on=["health_region_code", "year"], how="left",
        )
        .with_columns([pl.col(c).fill_null(0) for c in count_cols])
    )

    panel = panel.with_columns(
        # The exposure. Higher = narrower ascertainment.
        (pl.col("hospitalised") / pl.col("hosp_known")).alias("H"),
        (1 - pl.col("hospitalised") / pl.col("hosp_known")).alias("B"),
        (pl.col("deaths") / pl.col("outcome_known")).alias("cfr"),
        (1e5 * pl.col("cases") / pl.col("person_years")).alias("incidence_per_100k"),
        (pl.col("outcome_known") / pl.col("cases")).alias("outcome_completeness"),
        (pl.col("hosp_known") / pl.col("cases")).alias("hosp_completeness"),
        (pl.col("sev_known") / pl.col("cases")).alias("severity_completeness"),
        (pl.col("crit_known") / pl.col("cases")).alias("criterion_completeness"),
        (pl.col("age60") / pl.col("age_known")).alias("share_age60"),
        (pl.col("male") / pl.col("cases")).alias("share_male"),
        (pl.col("lab_confirmed") / pl.col("crit_known")).alias("share_lab"),
        (pl.col("severe") / pl.col("sev_known")).alias("share_severe"),
    ).sort(["health_region_code", "year"])

    panel.write_parquet(OUT / "region_year_panel.parquet")

    # Region totals over the whole window, for the cross-sectional displays that
    # remain (maps, descriptive strata). Ratios are formed after summing counts.
    reg = panel.group_by(["health_region_code", "health_region_name", "uf_abbr",
                          "region"]).agg(
        pl.col(["cases", "deaths", "outcome_known", "hospitalised", "hosp_known",
                "lab_confirmed", "crit_known", "severe", "sev_known", "age60",
                "age_known", "male", "person_years", "hk_hosp", "hk_nonhosp",
                "d_hosp", "d_nonhosp"]).sum(),
        pl.len().alias("years_observed"),
    ).with_columns(
        (pl.col("d_hosp") / pl.col("hk_hosp")).alias("cfr_hosp"),
        (pl.col("d_nonhosp") / pl.col("hk_nonhosp")).alias("cfr_nonhosp"),
    ).with_columns(
        (pl.col("hospitalised") / pl.col("hosp_known")).alias("H"),
        (pl.col("deaths") / pl.col("outcome_known")).alias("cfr"),
        (1e5 * pl.col("cases") / pl.col("person_years")).alias("incidence_per_100k"),
        (pl.col("outcome_known") / pl.col("cases")).alias("outcome_completeness"),
        (pl.col("age60") / pl.col("age_known")).alias("share_age60"),
        (pl.col("male") / pl.col("cases")).alias("share_male"),
        (pl.col("lab_confirmed") / pl.col("crit_known")).alias("share_lab"),
        (pl.col("severe") / pl.col("sev_known")).alias("share_severe"),
    ).sort("health_region_code")
    reg.write_parquet(OUT / "region_totals.parquet")

    summary = {
        "window": [YEAR_MIN, YEAR_MAX],
        "age_cut": AGE_CUT,
        "confirmed_cases": int(panel["cases"].sum()),
        "region_years": panel.height,
        "regions": reg.height,
        "regions_with_any_case": int((reg["cases"] > 0).sum()),
        "national": {
            "H": float(panel["hospitalised"].sum() / panel["hosp_known"].sum()),
            "cfr": float(panel["deaths"].sum() / panel["outcome_known"].sum()),
            "outcome_completeness": float(panel["outcome_known"].sum() / panel["cases"].sum()),
            "hosp_completeness": float(panel["hosp_known"].sum() / panel["cases"].sum()),
            "severity_completeness": float(panel["sev_known"].sum() / panel["cases"].sum()),
            "criterion_completeness": float(panel["crit_known"].sum() / panel["cases"].sum()),
            "share_lab": float(panel["lab_confirmed"].sum() / panel["crit_known"].sum()),
            "share_severe": float(panel["severe"].sum() / panel["sev_known"].sum()),
            "share_age60": float(panel["age60"].sum() / panel["age_known"].sum()),
        },
    }
    (OUT / "panel_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
