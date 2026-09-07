"""Three systems, one common window: does population mortality track reported case fatality?

This is the analysis that separates a denominator effect from a real difference
in lethality, and the first draft did not run it. SINAN case fatality is
conditional on being in the SINAN denominator. SIM measures deaths in the
population and does not know what SINAN recorded; SIH measures admissions and
does not know either. If reported case fatality rises six-fold across the
exposure while population mortality from the same cause is flat, the movement
is in the denominator. If population mortality rises with it, territories with
narrow ascertainment really are losing more people.

Everything here is computed on the **common window 2008–2024**, because that is
where all three systems have complete years. The first draft compared a
2007–2025 SINAN gradient against a 2008–2024 SIH gradient and called the ratio
"the excess attributable to notification"; both the window mismatch and the
causal reading are corrected here — this is triangulation between systems with
different failure modes, not a decomposition with an identifying assumption.

Outputs to ``data/results/triangulation/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "triangulation"
PANEL = PATHS.results / "analysis_panel"

#: All three systems have complete national coverage over these years.
WIN_MIN, WIN_MAX = 2008, 2024
MIN_HOSP_KNOWN = 30
N_BINS = 5


def _region_map() -> pl.DataFrame:
    return pl.read_parquet(
        PATHS.results / "atlas" / "municipality_atlas.parquet"
    ).select(pl.col("munic_code"), "health_region_code")


def _sim_by_region() -> pl.DataFrame:
    m = _region_map()
    return (
        pl.scan_parquet(PATHS.interim / "sim_a27_deaths.parquet")
        .select(
            pl.col("CODMUNRES").cast(pl.Utf8).alias("munic6"),
            pl.col("DTOBITO").cast(pl.Utf8).str.slice(4, 4).cast(pl.Int32).alias("year"),
        )
        .filter(pl.col("year").is_between(WIN_MIN, WIN_MAX))
        .collect()
        .join(
            m.with_columns(pl.col("munic_code").str.slice(0, 6).alias("munic6")),
            on="munic6", how="inner",
        )
        .group_by("health_region_code")
        .agg(pl.len().alias("sim_deaths"))
    )


def _sih_by_region() -> pl.DataFrame:
    m = _region_map()
    return (
        pl.scan_parquet(PATHS.interim / "sih_a27_admissions.parquet")
        .select(
            pl.col("MUNIC_RES").cast(pl.Utf8).alias("munic6"),
            pl.col("DT_INTER").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("year"),
        )
        .filter(pl.col("year").is_between(WIN_MIN, WIN_MAX))
        .collect()
        .join(
            m.with_columns(pl.col("munic_code").str.slice(0, 6).alias("munic6")),
            on="munic6", how="inner",
        )
        .group_by("health_region_code")
        .agg(pl.len().alias("sih_admissions"))
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    panel = pl.read_parquet(PANEL / "region_year_panel.parquet")
    win = panel.filter(pl.col("year").is_between(WIN_MIN, WIN_MAX))
    reg = win.group_by("health_region_code").agg(
        pl.col(["cases", "deaths", "outcome_known", "hospitalised", "hosp_known",
                "person_years", "hk_hosp", "d_hosp"]).sum()
    ).with_columns(
        (pl.col("hospitalised") / pl.col("hosp_known")).alias("H")
    ).filter(pl.col("hosp_known") > 0, pl.col("outcome_known") > 0)

    reg = (
        reg.join(_sim_by_region(), on="health_region_code", how="left")
           .join(_sih_by_region(), on="health_region_code", how="left")
           .with_columns(pl.col(["sim_deaths", "sih_admissions"]).fill_null(0))
    )
    reg.write_parquet(OUT / "region_three_systems.parquet")

    elig = reg.filter(pl.col("hk_hosp") >= MIN_HOSP_KNOWN)
    d = elig.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("bin")
    )

    rows = []
    for b, g in d.group_by("bin", maintain_order=True):
        b = int(b[0])
        py = float(g["person_years"].sum())
        cfr, cfr_lo, cfr_hi = binom_ci(int(g["deaths"].sum()), int(g["outcome_known"].sum()))
        inc, inc_lo, inc_hi = poisson_ci(int(g["cases"].sum()), py, scale=1e5)
        adm, adm_lo, adm_hi = poisson_ci(int(g["sih_admissions"].sum()), py, scale=1e5)
        mort, mort_lo, mort_hi = poisson_ci(int(g["sim_deaths"].sum()), py, scale=1e6)
        rows.append({
            "bin": b, "regions": g.height, "cases": int(g["cases"].sum()),
            "person_years": py, "H_median": float(g["H"].median()),
            "cfr": float(cfr), "cfr_lo": float(cfr_lo), "cfr_hi": float(cfr_hi),
            "sinan_incidence_per_100k": float(inc), "sinan_lo": float(inc_lo), "sinan_hi": float(inc_hi),
            "sih_admission_per_100k": float(adm), "sih_lo": float(adm_lo), "sih_hi": float(adm_hi),
            "sim_mortality_per_1m": float(mort), "sim_lo": float(mort_lo), "sim_hi": float(mort_hi),
        })
    t = pl.DataFrame(rows)
    t.write_csv(OUT / "three_systems_by_stratum.csv")

    def grad(col: str) -> float:
        return rows[0][col] / rows[-1][col]

    # The flat-mortality result is the load-bearing one, so it is reported at
    # every eligibility threshold including none at all. If it only held on the
    # regions large enough to estimate a within-stratum case fatality, it would
    # be a selection artefact rather than a finding.
    sweep = []
    for thr in (0, 10, 20, 30, 50):
        s = reg.filter(pl.col("hk_hosp") >= thr)
        b = s.sort("H").with_columns(
            (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
        ).with_columns(
            (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("bin")
        ).group_by("bin").agg(
            pl.col(["cases", "deaths", "outcome_known", "sim_deaths", "person_years"]).sum()
        ).sort("bin")
        cfr_q = (b["deaths"] / b["outcome_known"]).to_list()
        sim_q = (1e6 * b["sim_deaths"] / b["person_years"]).to_list()
        inc_q = (1e5 * b["cases"] / b["person_years"]).to_list()
        sweep.append({
            "min_hosp_known": thr, "regions": s.height, "cases": int(s["cases"].sum()),
            "cfr_Q1": cfr_q[0], "cfr_Q5": cfr_q[-1], "cfr_ratio_Q5_Q1": cfr_q[-1] / cfr_q[0],
            "incidence_Q1": inc_q[0], "incidence_Q5": inc_q[-1],
            "incidence_ratio_Q1_Q5": inc_q[0] / inc_q[-1],
            "sim_Q1_per_1m": sim_q[0], "sim_Q5_per_1m": sim_q[-1],
            "sim_ratio_Q1_Q5": sim_q[0] / sim_q[-1],
            "sim_min_per_1m": min(sim_q), "sim_max_per_1m": max(sim_q),
        })
    pl.DataFrame(sweep).write_csv(OUT / "flat_mortality_threshold_sweep.csv")

    # The strongest single objection to the flat-mortality finding is that SIM
    # might miss A27 deaths in exactly the territories whose surveillance is
    # narrow, which would flatten a real gradient. Two checks discriminate.
    #
    #   (i) The ratio of SIM deaths to SINAN-recorded leptospirosis deaths. If
    #       SIM were differentially blind, that ratio would fall as H rises.
    #  (ii) Deaths per head of population computed from SINAN's OWN numerator.
    #       This shares no denominator with the case count either, and it does
    #       not depend on SIM at all.
    #
    # If both are flat across the exposure, the case-fatality gradient is a
    # denominator phenomenon in two systems at once.
    nb = reg.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("bin")
    ).group_by("bin").agg(
        pl.col(["deaths", "sim_deaths", "person_years", "cases"]).sum()
    ).sort("bin").with_columns(
        (pl.col("sim_deaths") / pl.col("deaths")).alias("sim_per_sinan_death"),
        (1e6 * pl.col("deaths") / pl.col("person_years")).alias("sinan_death_rate_per_1m"),
        (1e6 * pl.col("sim_deaths") / pl.col("person_years")).alias("sim_death_rate_per_1m"),
    )
    nb.write_csv(OUT / "numerator_invariance_by_stratum.csv")

    agree = reg.filter(pl.col("deaths") >= 10)
    numerator = {
        "sim_per_sinan_death_national": float(reg["sim_deaths"].sum() / reg["deaths"].sum()),
        "sim_per_sinan_death_by_stratum": [float(v) for v in nb["sim_per_sinan_death"]],
        "spearman_H_vs_sim_per_sinan_ratio": float(
            stats.spearmanr(agree["H"], agree["sim_deaths"] / agree["deaths"]).statistic),
        "n_regions_for_ratio": agree.height,
        "spearman_H_vs_sinan_death_rate_per_capita": float(
            stats.spearmanr(reg["H"], reg["deaths"] / reg["person_years"]).statistic),
        "spearman_H_vs_sim_death_rate_per_capita": float(
            stats.spearmanr(reg["H"], reg["sim_deaths"] / reg["person_years"]).statistic),
    }

    report = {
        "numerator_invariance": numerator,
        "threshold_sweep": sweep,
        "window": [WIN_MIN, WIN_MAX],
        "regions_eligible": elig.height,
        "cases": int(elig["cases"].sum()),
        "cfr_gradient_Q5_over_Q1": rows[-1]["cfr"] / rows[0]["cfr"],
        "sinan_incidence_gradient_Q1_over_Q5": grad("sinan_incidence_per_100k"),
        "sih_admission_gradient_Q1_over_Q5": grad("sih_admission_per_100k"),
        "sim_mortality_gradient_Q1_over_Q5": grad("sim_mortality_per_1m"),
        "spearman_H_vs_sim_mortality": float(
            stats.spearmanr(
                elig["H"], 1e6 * elig["sim_deaths"] / elig["person_years"]
            ).statistic
        ),
        "national_sim_deaths": int(reg["sim_deaths"].sum()),
        "national_sinan_deaths_in_window": int(reg["deaths"].sum()),
    }
    (OUT / "triangulation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(t.select("bin", "regions", "H_median", "cfr", "sinan_incidence_per_100k",
                   "sih_admission_per_100k", "sim_mortality_per_1m")
           .to_pandas().to_string(index=False))
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
