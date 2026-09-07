"""Rio Grande do Sul 2024: the ascertainment-depth mechanism, within one state.

The paper's central result is cross-sectional — health regions that hospitalise
a larger share of their confirmed cases report higher case fatality, because
their surveillance reaches a shallower way into the severity distribution. A
cross-sectional gradient always invites the reply that the territories simply
differ, in their patients or their pathogens or their hospitals.

April-May 2024 supplies the counterfactual. Catastrophic flooding in Rio Grande
do Sul triggered mass case-finding in a single state, over a few months, holding
the population, the health system, the pathogen and the notification form fixed.
If hospitalisation share indexes how deep surveillance reaches, then during the
surge it must FALL — the same system suddenly finding milder illness — while
incidence rises and case fatality falls. If instead it indexes genuine severity,
a flood exposing people to contaminated water should if anything raise it.

Three named threats are tested rather than asserted away:

*Recording artefact.* `ATE_HOSP` validity among confirmed RS cases drops to
86.6% in 2024 from ~97-99% in adjacent years, and a case with no answer is not
counted as hospitalised. That alone depresses the share. Every figure here is
therefore computed on the **answered denominator** — cases whose hospitalisation
field carries a valid response — and the worst-case bound (treating every
unanswered case as hospitalised) is reported beside it.

*National artefact.* If 2024 depressed hospitalisation share everywhere, the
event is not about the flood. Compared against every other state with at least
150 confirmed cases in 2024.

*Right-edge artefact.* 2025 is preliminary. Reported, and the argument does not
depend on it because 2023 is the pre-period.

Outputs to ``data/results/rs2024/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "rs2024"
LINE = PATHS.interim / "lept_line_level.parquet"
STATE = "RS"
FLOOD_YEAR = 2024
#: A state needs enough 2024 cases for its hospitalisation share to be estimable
#: against RS. Stated rather than buried.
MIN_STATE_CASES_2024 = 150


def _line(uf: str | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(LINE).filter(pl.col("classi_fin") == "confirmado")
    if uf is not None:
        lf = lf.filter(pl.col("uf_residence_abbr") == uf)
    return lf


def state_series() -> pl.DataFrame:
    """RS by onset year: incidence, hospitalisation share, case fatality.

    Hospitalisation share uses the answered denominator. Case fatality uses the
    known-outcome denominator, for the same reason: a blank field is not a
    negative answer.
    """
    d = _line(STATE).select(
        "epiweek_onset_year", "ate_hosp", "ate_hosp_state", "evolucao", "evolucao_state",
        "criterio",
    ).collect()
    g = d.group_by("epiweek_onset_year").agg(
        pl.len().alias("confirmed"),
        (pl.col("ate_hosp_state") == "valid").sum().alias("hosp_answered"),
        ((pl.col("ate_hosp_state") == "valid") & (pl.col("ate_hosp") == "sim")).sum().alias("hospitalised"),
        (pl.col("evolucao_state") == "valid").sum().alias("outcome_known"),
        # Third independent surveillance marker. During a mass-detection event
        # laboratory capacity saturates and confirmation shifts toward the
        # clinical-epidemiological criterion; if the 2024 signal is really a
        # surveillance stretch and not a change in disease, this must move too.
        pl.col("criterio").is_not_null().sum().alias("criterion_stated"),
        (pl.col("criterio") == "clinico_laboratorial").sum().alias("lab_confirmed"),
    ).sort("epiweek_onset_year").rename({"epiweek_onset_year": "year"})

    deaths = (
        _line(STATE).filter(pl.col("evolucao_state") == "valid")
        .group_by("epiweek_onset_year")
        .agg((pl.col("evolucao") == "obito_por_leptospirose").sum().alias("deaths"))
        .collect().rename({"epiweek_onset_year": "year"})
    )
    if int(deaths["deaths"].sum()) == 0:  # label drift guard
        vals = _line(STATE).select("evolucao").collect()["evolucao"].value_counts()
        raise AssertionError(
            "no deaths matched the evolucao death label; observed values: "
            f"{vals.to_dicts()[:8]}"
        )
    g = g.join(deaths, on="year", how="left").with_columns(pl.col("deaths").fill_null(0))

    py = (
        pl.read_parquet(PATHS.results / "atlas" / "municipality_year_atlas.parquet")
        .filter(pl.col("uf_abbr") == STATE)
        .group_by("year").agg(pl.col("person_years").sum().alias("person_years"))
    )
    g = g.join(py, on="year", how="left")

    inc, ilo, ihi = poisson_ci(g["confirmed"].to_numpy(), g["person_years"].to_numpy(), scale=1e5)
    hs, hlo, hhi = binom_ci(g["hospitalised"].to_numpy(), g["hosp_answered"].to_numpy().astype(float))
    cf, clo, chi = binom_ci(g["deaths"].to_numpy(), g["outcome_known"].to_numpy().astype(float))
    lab, llo, lhi = binom_ci(
        g["lab_confirmed"].to_numpy(), g["criterion_stated"].to_numpy().astype(float)
    )
    unanswered = g["confirmed"] - g["hosp_answered"]
    bound = (g["hospitalised"] + unanswered) / g["confirmed"]
    return g.with_columns(
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", ilo), pl.Series("incidence_hi", ihi),
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("hosp_share_lo", 100 * hlo), pl.Series("hosp_share_hi", 100 * hhi),
        (100 * pl.col("hosp_answered") / pl.col("confirmed")).alias("field_validity_pct"),
        pl.Series("hosp_share_worst_case_pct", 100 * bound),
        pl.Series("cfr_pct", 100 * cf),
        pl.Series("cfr_lo", 100 * clo), pl.Series("cfr_hi", 100 * chi),
        pl.Series("lab_share_pct", 100 * lab),
        pl.Series("lab_share_lo", 100 * llo), pl.Series("lab_share_hi", 100 * lhi),
    )


def other_states_2024() -> pl.DataFrame:
    """Was 2024 a national dip, or is it RS alone?"""
    d = (
        _line().filter(pl.col("epiweek_onset_year") == FLOOD_YEAR)
        .group_by("uf_residence_abbr").agg(
            pl.len().alias("confirmed"),
            (pl.col("ate_hosp_state") == "valid").sum().alias("answered"),
            ((pl.col("ate_hosp_state") == "valid") & (pl.col("ate_hosp") == "sim")).sum().alias("hospitalised"),
        ).collect()
        .filter(pl.col("confirmed") >= MIN_STATE_CASES_2024)
        .sort("confirmed", descending=True)
    )
    hs, lo, hi = binom_ci(d["hospitalised"].to_numpy(), d["answered"].to_numpy().astype(float))
    return d.with_columns(
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("hosp_share_lo", 100 * lo), pl.Series("hosp_share_hi", 100 * hi),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = state_series()
    series.write_parquet(OUT / "rs_annual_series.parquet")
    others = other_states_2024()
    others.write_parquet(OUT / "states_2024_hosp_share.parquet")

    win = series.filter(pl.col("year").is_between(2023, 2025)).sort("year")
    pre, flood, post = (win.row(i, named=True) for i in range(3))

    print("=== Rio Grande do Sul, confirmed cases by onset year ===")
    print(f"{'yr':>5}{'conf':>6}{'valid%':>8}{'incid':>8}{'hosp share % (answered)':>28}{'lab %':>8}{'CFR %':>20}")
    for r in series.filter(pl.col("year") >= 2019).iter_rows(named=True):
        mark = "  <-- flood" if r["year"] == FLOOD_YEAR else ""
        print(f"{r['year']:>5}{r['confirmed']:>6}{r['field_validity_pct']:>8.1f}"
              f"{r['incidence_per_100k']:>8.2f}"
              f"{r['hosp_share_pct']:>12.1f} ({r['hosp_share_lo']:.1f}-{r['hosp_share_hi']:.1f})"
              f"{r['lab_share_pct']:>8.1f}"
              f"{r['cfr_pct']:>10.2f} ({r['cfr_lo']:.1f}-{r['cfr_hi']:.1f}){mark}")

    print(f"\n=== the event, 2023 -> 2024 -> 2025 ===")
    print(f"  incidence rate /100k py : {pre['incidence_per_100k']:.2f} -> "
          f"{flood['incidence_per_100k']:.2f} -> {post['incidence_per_100k']:.2f}  "
          f"({flood['incidence_per_100k']/pre['incidence_per_100k']:.1f}x)")
    print(f"  hospitalisation share % : {pre['hosp_share_pct']:.1f} -> "
          f"{flood['hosp_share_pct']:.1f} -> {post['hosp_share_pct']:.1f}  "
          f"({flood['hosp_share_pct']-pre['hosp_share_pct']:+.1f} pp, then "
          f"{post['hosp_share_pct']-flood['hosp_share_pct']:+.1f} pp)")
    print(f"  case fatality %         : {pre['cfr_pct']:.2f} -> "
          f"{flood['cfr_pct']:.2f} -> {post['cfr_pct']:.2f}")
    print(f"  laboratory confirm. %   : {pre['lab_share_pct']:.1f} -> "
          f"{flood['lab_share_pct']:.1f} -> {post['lab_share_pct']:.1f}  "
          f"({flood['lab_share_pct']-pre['lab_share_pct']:+.1f} pp, then "
          f"{post['lab_share_pct']-flood['lab_share_pct']:+.1f} pp)")

    disjoint = flood["hosp_share_hi"] < pre["hosp_share_lo"] and flood["hosp_share_hi"] < post["hosp_share_lo"]
    print(f"  2024 hospitalisation-share interval disjoint from both neighbours: {disjoint}")

    print(f"\n=== threat: recording artefact ===")
    print(f"  field validity {pre['field_validity_pct']:.1f}% -> "
          f"{flood['field_validity_pct']:.1f}% -> {post['field_validity_pct']:.1f}%")
    print(f"  worst case (every unanswered case counted as hospitalised): "
          f"{flood['hosp_share_worst_case_pct']:.1f}% in 2024, still "
          f"{pre['hosp_share_pct']-flood['hosp_share_worst_case_pct']:.1f} pp below 2023")

    print(f"\n=== threat: national artefact — 2024 hospitalisation share by state ===")
    for r in others.iter_rows(named=True):
        mark = "  <-- flood state" if r["uf_residence_abbr"] == STATE else ""
        print(f"  {r['uf_residence_abbr']}  confirmed={r['confirmed']:>5}  "
              f"share {r['hosp_share_pct']:>5.1f}% "
              f"({r['hosp_share_lo']:.1f}-{r['hosp_share_hi']:.1f}){mark}")
    non_rs = others.filter(pl.col("uf_residence_abbr") != STATE)
    print(f"  every other state in range "
          f"{non_rs['hosp_share_pct'].min():.1f}-{non_rs['hosp_share_pct'].max():.1f}%; "
          f"RS at {flood['hosp_share_pct']:.1f}%")

    report = {
        "window": {"pre": pre, "flood": flood, "post": post},
        "incidence_fold_change": float(flood["incidence_per_100k"] / pre["incidence_per_100k"]),
        "hosp_share_change_pp": float(flood["hosp_share_pct"] - pre["hosp_share_pct"]),
        "hosp_share_recovery_pp": float(post["hosp_share_pct"] - flood["hosp_share_pct"]),
        "lab_share_change_pp": float(flood["lab_share_pct"] - pre["lab_share_pct"]),
        "lab_share_recovery_pp": float(post["lab_share_pct"] - flood["lab_share_pct"]),
        "interval_disjoint_from_both_neighbours": bool(disjoint),
        "worst_case_2024_hosp_share_pct": float(flood["hosp_share_worst_case_pct"]),
        "other_states_2024_min_pct": float(non_rs["hosp_share_pct"].min()),
        "other_states_2024_max_pct": float(non_rs["hosp_share_pct"].max()),
        "min_state_cases_2024": MIN_STATE_CASES_2024,
    }
    (OUT / "rs2024_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
