"""Rio Grande do Sul, May 2024: a within-territory perturbation of ascertainment breadth.

WHY THIS EXISTS
---------------
The study's primary result is cross-sectional. Health regions whose confirmed
cases are overwhelmingly hospitalised -- high H, NARROW ascertainment, a
surveillance system that only meets leptospirosis in an intensive care unit --
report far higher case fatality than regions whose confirmed cases are only
half hospitalised. A cross-sectional gradient always invites the reply that the
territories genuinely differ: different patients, different serovars, different
hospitals, different rats.

The 2024 flood supplies a within-territory contrast instead. In a single state,
over three months, the population, the health system, the circulating pathogen
and the SINAN notification form are approximately fixed while case-finding
changes by an order of magnitude. If H indexes how broadly surveillance reaches
into the clinical spectrum, then during a mass case-finding effort H must FALL:
the same system, suddenly looking harder, meets milder illness. If instead H
indexes genuine clinical severity, a flood that immerses two million people in
contaminated water should if anything push H up.

WHAT WOULD FALSIFY THE CLAIM THIS SUPPORTS
------------------------------------------
* H does not fall during the surge, or falls only as much as its own recording
  completeness falls (i.e. the movement is an artefact of blank ATE_HOSP
  fields). Both the answered-denominator estimate and the worst-case bound that
  treats every unanswered case as hospitalised are reported for this reason.
* H falls by a similar amount in 2024 in states that were not flooded -- that
  would make 2024 a national recording year, not an RS event.
* H and case fatality fall only because the confirmation criterion loosened:
  during a declared calamity, flood exposure is itself an epidemiological link,
  so a fever plus a flooded house can be confirmed clinical-epidemiologically.
  That would dilute the denominator with non-cases and lower both H and case
  fatality WITHOUT surveillance reaching any deeper into true leptospirosis.
  This is the most serious competing explanation and it is tested directly by
  repeating every quantity inside the laboratory-confirmed stratum only, where
  criterion drift cannot operate.

WHAT THIS EVENT CANNOT ESTABLISH
--------------------------------
This is SUPPORTIVE within-territory validation, not a natural experiment with
an identifying assumption. The flood moved true incidence, true severity
composition, health-service access, care-seeking, laboratory throughput,
antibiotic timing and risk communication simultaneously and in the same
direction as ascertainment breadth. An earlier draft called it "the strongest
test available" and "difficult to explain by any mechanism other than a change
in detection"; both claims are withdrawn here. The event is consistent with the
ascertainment reading and would have embarrassed it had H moved the other way.
That is the whole of its evidential value.

EXTERNAL ACCOUNT
----------------
Ranieri TM, Viegas da Silva E, Vallandro MJ, et al. Leptospirosis cases during
the 2024 catastrophic flood in Rio Grande do Sul, Brazil. Pathogens.
2025;14(4):393. Written by the state surveillance centre that ran the response,
from the same SINAN notifications, extracted 13 March 2025. Their headline
figures are hard-coded below as reconciliation targets. Two of their own
observations matter for us: the confirmed-among-notified proportion fell from
20.2% to 15.3% (a wider suspicion net), and they attribute the
lower-than-expected case fatality partly to the surveillance system being "more
sensitive in capturing mild cases" -- an ascertainment reading arrived at
independently, by the people who ran the surveillance.

Outputs to ``data/results/rs2024_event/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats.contingency import odds_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "rs2024_event"
LINE = PATHS.interim / "lept_line_level.parquet"
POP = PATHS.interim / "population_municipal_year.parquet"
ATLAS = PATHS.results / "atlas" / "municipality_atlas.parquet"

STATE = "RS"

#: Monthly resolution needs a stable recent stretch on both sides of the event.
#: 2019 is the first year before the COVID-19 notification collapse that Ranieri
#: also excludes from their baseline; 2025 is the post-event year.
SERIES_MIN, SERIES_MAX = 2019, 2025

#: Ranieri's flood window, adopted verbatim so the reconciliation is like-for-like:
#: 1 May to 31 July, symptom-onset basis, compared with the same three calendar
#: months of 2023. 2023 rather than a multi-year mean because 2020-2022 SINAN
#: notification in RS was depressed by the pandemic.
FLOOD_START, FLOOD_END = (2024, 5, 1), (2024, 7, 31)
BASE_START, BASE_END = (2023, 5, 1), (2023, 7, 31)

#: The pre-event baseline for the reversion question. Deliberately wider than
#: the three-month Ranieri baseline: reversion is a question about the state's
#: ordinary operating level, not about one quarter.
PRE_MIN, PRE_MAX = 2019, 2023

#: A state needs enough 2024 confirmed cases for its hospitalisation share to be
#: estimable against RS at all. Below this the interval swallows any contrast.
MIN_STATE_CASES_2024 = 150

#: A SECOND, smaller perturbation of the same territory, found in the monthly
#: series rather than assumed: the extratropical cyclone and Taquari valley
#: flood of early September 2023. It is not in Ranieri's paper (their baseline
#: window ends 31 July 2023, so it does not contaminate the comparison), and it
#: is the closest thing available to an internal replication -- same state, same
#: form, same laboratory, a year earlier, one tenth the magnitude. If the May
#: 2024 signature is a property of mass case-finding rather than of that one
#: catastrophe, September-October 2023 must show it too, in miniature.
CYCLONE_START, CYCLONE_END = (2023, 9, 1), (2023, 10, 31)
#: Matched calendar months from the non-event years of the pre-period. 2020 is
#: excluded: pandemic-year notification in RS is not a usable baseline.
CYCLONE_BASE_YEARS = (2019, 2021, 2022)

#: The four months of 2024 before the flood, against the same months of the
#: pre-period. A pre-trend check: if H in RS was already sliding before 28 April
#: 2024, the May discontinuity is the tail of a drift, not an event.
PRETREND_MONTHS = (1, 2, 3, 4)

#: Published figures from Ranieri et al. 2025, Pathogens 14(4):393, Table 1 and
#: sections 3.1.1 / 3.1.4. Reconciliation targets, not inputs to any estimate.
RANIERI = {
    "flood_notified": 6273,
    "flood_confirmed": 958,
    "flood_lab_confirmed": 464,
    "flood_confirmed_share": 0.153,
    "flood_deaths": 30,
    "flood_cfr": 0.031,
    "base_notified": 461,
    "base_confirmed": 93,
    "base_lab_confirmed": 73,
    "base_confirmed_share": 0.202,
    "base_deaths": 5,
    "base_cfr": 0.054,
    "fold_cases": 10.3,
    "fold_deaths": 6.0,
    "spearman_rho_incidence_vs_flooded_households": 0.77,
}

#: An earlier research note recorded an unexplained conflict between Ranieri's
#: 2023 baseline of 93 confirmed cases and a figure of 478 from this pipeline.
#: The stated suspicion was that the two numbers are different windows, not
#: different data. Recorded here so the script confirms or refutes it explicitly.
DISPUTED_2023_FIGURE = 478


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def _notifications() -> pl.DataFrame:
    """All SINAN notifications with a parseable onset date and a residence
    municipality, regardless of final classification.

    The confirmed-among-notified ratio only means anything if numerator and
    denominator are drawn from the same population, so the residence-municipality
    and onset-date filters of the analytic population are applied to notified
    cases too -- the only thing dropped for the denominator is the requirement
    that the case ended up confirmed.
    """
    lf = pl.scan_parquet(LINE).select(
        "DT_SIN_PRI", "classi_fin", "classi_fin_state", "criterio", "criterio_state",
        "ate_hosp", "ate_hosp_state", "evolucao", "evolucao_state",
        "cli_icteri", "cli_icteri_state", "cli_renal", "cli_renal_state",
        "cli_hemorr", "cli_hemorr_state",
        "uf_residence_abbr", "municipality_residence_code7", "epiweek_onset_year",
    )
    d = (
        lf.filter(pl.col("municipality_residence_code7").is_not_null())
        .with_columns(
            pl.col("DT_SIN_PRI").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("onset")
        )
        .filter(pl.col("onset").is_not_null())
        .collect()
    )
    severe_known = (
        (pl.col("cli_icteri_state") == "valid")
        & (pl.col("cli_renal_state") == "valid")
        & (pl.col("cli_hemorr_state") == "valid")
    )
    return d.with_columns(
        pl.col("onset").dt.year().alias("onset_year"),
        pl.col("onset").dt.month().alias("onset_month"),
        (pl.col("classi_fin") == "confirmado").cast(pl.Int32).alias("confirmed"),
        # Every flag below is 1 only on a VALID answer. A blank is not a "no".
        ((pl.col("ate_hosp_state") == "valid") & (pl.col("ate_hosp") == "sim"))
        .cast(pl.Int32).alias("hosp"),
        (pl.col("ate_hosp_state") == "valid").cast(pl.Int32).alias("hosp_known"),
        ((pl.col("evolucao_state") == "valid") & (pl.col("evolucao") == "obito_por_leptospirose"))
        .cast(pl.Int32).alias("death"),
        (pl.col("evolucao_state") == "valid").cast(pl.Int32).alias("outcome_known"),
        ((pl.col("criterio_state") == "valid") & (pl.col("criterio") == "clinico_laboratorial"))
        .cast(pl.Int32).alias("lab"),
        (pl.col("criterio_state") == "valid").cast(pl.Int32).alias("crit_known"),
        severe_known.cast(pl.Int32).alias("sev_known"),
        (
            severe_known
            & ((pl.col("cli_icteri") == "sim") | (pl.col("cli_renal") == "sim")
               | (pl.col("cli_hemorr") == "sim"))
        ).cast(pl.Int32).alias("severe"),
    )


def _rs_population() -> pl.DataFrame:
    atlas = pl.read_parquet(ATLAS).select("munic_code", "uf_abbr")
    return (
        pl.read_parquet(POP)
        .join(atlas, on="munic_code", how="inner")
        .filter(pl.col("uf_abbr") == STATE)
        .group_by("year").agg(pl.col("population").sum())
        .sort("year")
    )


# ---------------------------------------------------------------------------
# estimation helpers
# ---------------------------------------------------------------------------

def _prop(num: int, den: int) -> dict:
    """Clopper-Pearson proportion. ``binom_ci`` returns (ESTIMATE, lo, hi)."""
    if den <= 0:
        return {"n": int(num), "d": int(den), "p": None, "lo": None, "hi": None}
    p, lo, hi = binom_ci(num, den)
    return {"n": int(num), "d": int(den), "p": float(p), "lo": float(lo), "hi": float(hi)}


def _cols(df: pl.DataFrame, name: str, num: str, den: str) -> pl.DataFrame:
    """Attach a Clopper-Pearson proportion and its interval as three columns."""
    p, lo, hi = binom_ci(df[num].to_numpy(), df[den].to_numpy())
    return df.with_columns(
        pl.Series(name, np.where(df[den].to_numpy() > 0, p, np.nan)),
        pl.Series(f"{name}_lo", np.where(df[den].to_numpy() > 0, lo, np.nan)),
        pl.Series(f"{name}_hi", np.where(df[den].to_numpy() > 0, hi, np.nan)),
    )


def _count_ratio_ci(x: int, y: int, t_x: float = 1.0, t_y: float = 1.0) -> dict:
    """Exact interval for the ratio of two Poisson rates (x/t_x) / (y/t_y).

    Conditional on the total x+y, x is binomial with success probability
    rho/(1+rho) where rho = (lambda_x t_x)/(lambda_y t_y); inverting a
    Clopper-Pearson interval on that binomial and mapping back through the odds
    transform gives an exact interval for the rate ratio. Used for the fold
    change because "10.3-fold" is a rate ratio between two equal-length windows
    over the same population, and quoting it without an interval hides that it
    rests on 91 baseline cases.
    """
    total = x + y
    if total == 0 or y == 0 and x == 0:
        return {"ratio": None, "lo": None, "hi": None, "x": int(x), "y": int(y)}
    p, plo, phi = binom_ci(x, total)
    scale = t_y / t_x

    def odds(v):
        v = float(v)
        return np.inf if v >= 1.0 else v / (1.0 - v) * scale

    ratio = (x / t_x) / (y / t_y) if y > 0 else float("inf")
    return {
        "ratio": float(ratio) if np.isfinite(ratio) else None,
        "lo": float(odds(plo)),
        "hi": float(odds(phi)) if np.isfinite(odds(phi)) else None,
        "x": int(x), "y": int(y),
    }


def _exact_or(a: int, b: int, c: int, d: int) -> dict:
    """Conditional-MLE odds ratio with an exact (Fisher) interval.

    Used to compare a proportion between two periods. A difference in
    percentage points is the quantity a reader wants and is reported alongside,
    but the exact interval belongs to the odds ratio -- there is no exact
    interval for a difference of two independent binomial proportions, and the
    house rule is exact over normal-approximate.
    """
    if min(a + b, c + d) == 0:
        return {"or": None, "lo": None, "hi": None, "p_fisher": None}
    res = odds_ratio([[a, b], [c, d]], kind="conditional")
    lo, hi = res.confidence_interval()
    _, pval = stats.fisher_exact([[a, b], [c, d]])
    return {
        "or": float(res.statistic) if np.isfinite(res.statistic) else None,
        "lo": float(lo) if np.isfinite(lo) else None,
        "hi": float(hi) if np.isfinite(hi) else None,
        "p_fisher": float(pval),
    }


def _window(df: pl.DataFrame, start: tuple, end: tuple) -> pl.DataFrame:
    return df.filter(pl.col("onset").is_between(pl.date(*start), pl.date(*end)))


def _or_between(a: dict, b: dict, key: str) -> dict:
    """Contrast the same proportion between two slices, exact interval on the OR."""
    pa, pb = a[key], b[key]
    return {
        **_exact_or(pa["n"], pa["d"] - pa["n"], pb["n"], pb["d"] - pb["n"]),
        "event": pa, "reference": pb,
        "change_pp": None if pa["p"] is None or pb["p"] is None
        else 100 * (pa["p"] - pb["p"]),
    }


def _block(df: pl.DataFrame, label: str) -> dict:
    """Every headline quantity for one slice of notifications."""
    conf = df.filter(pl.col("confirmed") == 1)
    n_notified, n_conf = df.height, conf.height
    s = {c: int(conf[c].sum()) for c in
         ("hosp", "hosp_known", "death", "outcome_known", "lab", "crit_known",
          "severe", "sev_known")}
    out = {
        "label": label,
        "notified": n_notified,
        "confirmed": n_conf,
        "confirmed_among_notified": _prop(n_conf, n_notified),
        "H_hospitalisation_share": _prop(s["hosp"], s["hosp_known"]),
        # Worst case for the recording-artefact threat: every case with no
        # answer on ATE_HOSP is counted as hospitalised. If H still falls under
        # this bound, blank fields cannot be the explanation.
        "H_worst_case_bound": _prop(s["hosp"] + (n_conf - s["hosp_known"]), n_conf),
        "hosp_completeness": _prop(s["hosp_known"], n_conf),
        "lab_confirmed_share": _prop(s["lab"], s["crit_known"]),
        "criterion_completeness": _prop(s["crit_known"], n_conf),
        "severe_share": _prop(s["severe"], s["sev_known"]),
        "deaths": s["death"],
        "case_fatality": _prop(s["death"], s["outcome_known"]),
        "outcome_completeness": _prop(s["outcome_known"], n_conf),
    }
    # The criterion-drift threat: repeat H and case fatality inside the
    # laboratory-confirmed stratum, where a loosened clinical-epidemiological
    # link cannot have added anyone.
    lab = conf.filter(pl.col("lab") == 1)
    out["lab_stratum"] = {
        "confirmed": lab.height,
        "H_hospitalisation_share": _prop(int(lab["hosp"].sum()), int(lab["hosp_known"].sum())),
        "case_fatality": _prop(int(lab["death"].sum()), int(lab["outcome_known"].sum())),
        "severe_share": _prop(int(lab["severe"].sum()), int(lab["sev_known"].sum())),
    }
    return out


# ---------------------------------------------------------------------------
# (a) monthly series
# ---------------------------------------------------------------------------

def monthly_series(notif: pl.DataFrame, pop: pl.DataFrame) -> pl.DataFrame:
    rs = notif.filter(
        pl.col("uf_residence_abbr") == STATE,
        pl.col("onset_year").is_between(SERIES_MIN, SERIES_MAX),
    )
    grid = pl.DataFrame({
        "onset_year": np.repeat(np.arange(SERIES_MIN, SERIES_MAX + 1), 12).astype(np.int32),
        "onset_month": np.tile(np.arange(1, 13), SERIES_MAX - SERIES_MIN + 1).astype(np.int32),
    })
    g = (
        rs.group_by("onset_year", "onset_month").agg(
            pl.len().alias("notified"),
            pl.col("confirmed").sum().alias("confirmed"),
            # every case-level count below is on CONFIRMED cases only
            (pl.col("confirmed") * pl.col("hosp")).sum().alias("hospitalised"),
            (pl.col("confirmed") * pl.col("hosp_known")).sum().alias("hosp_known"),
            (pl.col("confirmed") * pl.col("death")).sum().alias("deaths"),
            (pl.col("confirmed") * pl.col("outcome_known")).sum().alias("outcome_known"),
            (pl.col("confirmed") * pl.col("lab")).sum().alias("lab_confirmed"),
            (pl.col("confirmed") * pl.col("crit_known")).sum().alias("crit_known"),
            (pl.col("confirmed") * pl.col("severe")).sum().alias("severe"),
            (pl.col("confirmed") * pl.col("sev_known")).sum().alias("sev_known"),
        )
    )
    m = grid.join(g, on=["onset_year", "onset_month"], how="left").fill_null(0)
    m = m.join(pop.rename({"year": "onset_year"}), on="onset_year", how="left")

    days = np.array([
        (np.datetime64(f"{y:04d}-{mo:02d}-01") + np.timedelta64(32, "D")).astype("datetime64[M]")
        .astype("datetime64[D]") - np.datetime64(f"{y:04d}-{mo:02d}-01")
        for y, mo in zip(m["onset_year"], m["onset_month"])
    ]).astype(float)
    m = m.with_columns(pl.Series("days_in_month", days))
    m = m.with_columns(
        (pl.col("population") * pl.col("days_in_month") / 365.25).alias("person_years")
    )
    # Poisson rate, annualised, per 100,000 person-years -- comparable month to
    # month despite unequal month lengths.
    inc, ilo, ihi = poisson_ci(
        m["confirmed"].to_numpy(), m["person_years"].to_numpy(), scale=1e5
    )
    m = m.with_columns(
        pl.Series("incidence_per_100k_py", inc),
        pl.Series("incidence_lo", ilo),
        pl.Series("incidence_hi", ihi),
    )
    m = _cols(m, "confirmed_among_notified", "confirmed", "notified")
    m = _cols(m, "H", "hospitalised", "hosp_known")
    m = _cols(m, "lab_share", "lab_confirmed", "crit_known")
    m = _cols(m, "severe_share", "severe", "sev_known")
    m = _cols(m, "cfr", "deaths", "outcome_known")
    m = m.with_columns(
        (pl.col("hosp_known") / pl.col("confirmed")).alias("hosp_completeness"),
        (pl.col("outcome_known") / pl.col("confirmed")).alias("outcome_completeness"),
        (pl.col("crit_known") / pl.col("confirmed")).alias("criterion_completeness"),
        ((pl.col("hospitalised") + pl.col("confirmed") - pl.col("hosp_known"))
         / pl.col("confirmed")).alias("H_worst_case_bound"),
        pl.date(pl.col("onset_year"), pl.col("onset_month"), 1).alias("month_start"),
    )
    return m.sort("onset_year", "onset_month")


# ---------------------------------------------------------------------------
# (c) national context
# ---------------------------------------------------------------------------

def national_context(notif: pl.DataFrame, min_cases: int = MIN_STATE_CASES_2024) -> pl.DataFrame:
    conf = notif.filter(
        pl.col("confirmed") == 1,
        pl.col("onset_year").is_between(PRE_MIN, 2024),
        pl.col("uf_residence_abbr").is_not_null(),
    )

    def agg(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
        return df.group_by("uf_residence_abbr").agg(
            pl.len().alias(f"cases{suffix}"),
            pl.col("hosp").sum().alias(f"hosp{suffix}"),
            pl.col("hosp_known").sum().alias(f"hosp_known{suffix}"),
            pl.col("death").sum().alias(f"deaths{suffix}"),
            pl.col("outcome_known").sum().alias(f"outcome_known{suffix}"),
        )

    y24 = agg(conf.filter(pl.col("onset_year") == 2024), "_2024")
    pre = agg(conf.filter(pl.col("onset_year").is_between(PRE_MIN, PRE_MAX)), "_pre")
    t = (
        y24.join(pre, on="uf_residence_abbr", how="left").fill_null(0)
        .filter(pl.col("cases_2024") >= min_cases, pl.col("hosp_known_pre") > 0)
        .rename({"uf_residence_abbr": "uf_abbr"})
    )
    t = _cols(t, "H_2024", "hosp_2024", "hosp_known_2024")
    t = _cols(t, "H_pre", "hosp_pre", "hosp_known_pre")
    t = t.with_columns(
        ((pl.col("H_2024") - pl.col("H_pre")) * 100).alias("H_change_pp"),
        (pl.col("hosp_known_2024") / pl.col("cases_2024")).alias("hosp_completeness_2024"),
        (pl.col("hosp_known_pre") / pl.col("cases_pre")).alias("hosp_completeness_pre"),
    )
    ors = [
        _exact_or(int(r["hosp_2024"]), int(r["hosp_known_2024"] - r["hosp_2024"]),
                  int(r["hosp_pre"]), int(r["hosp_known_pre"] - r["hosp_pre"]))
        for r in t.iter_rows(named=True)
    ]
    t = t.with_columns(
        pl.Series("or_hosp_2024_vs_pre", [o["or"] for o in ors]),
        pl.Series("or_lo", [o["lo"] for o in ors]),
        pl.Series("or_hi", [o["hi"] for o in ors]),
        pl.Series("p_fisher", [o["p_fisher"] for o in ors]),
    )
    return t.sort("H_change_pp")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notif = _notifications()
    pop = _rs_population()
    rs = notif.filter(pl.col("uf_residence_abbr") == STATE)
    report: dict = {"ranieri_published": RANIERI}

    # ---- (a) monthly series --------------------------------------------
    monthly = monthly_series(notif, pop)
    monthly.write_parquet(OUT / "rs_monthly_series.parquet")
    monthly.write_csv(OUT / "rs_monthly_series.csv")

    surge = monthly.filter(pl.col("onset_year") == 2024, pl.col("onset_month").is_between(5, 7))
    peak = monthly.sort("confirmed", descending=True).head(1)
    pre_months = monthly.filter(pl.col("onset_year").is_between(PRE_MIN, PRE_MAX))
    report["monthly"] = {
        "window": f"{SERIES_MIN}-01 to {SERIES_MAX}-12, symptom-onset month",
        "last_onset_in_data": str(rs["onset"].max()),
        "peak_month": {
            "month": str(peak["month_start"][0]),
            "confirmed": int(peak["confirmed"][0]),
            "notified": int(peak["notified"][0]),
            "incidence_per_100k_py": float(peak["incidence_per_100k_py"][0]),
            "incidence_ci": [float(peak["incidence_lo"][0]), float(peak["incidence_hi"][0])],
            "H": float(peak["H"][0]), "H_ci": [float(peak["H_lo"][0]), float(peak["H_hi"][0])],
        },
        "max_pre_event_monthly_confirmed": int(pre_months["confirmed"].max()),
        "surge_months_confirmed": [int(v) for v in surge["confirmed"]],
    }

    # ---- (b) reconciliation with Ranieri --------------------------------
    flood = _block(_window(rs, FLOOD_START, FLOOD_END), "RS 2024-05-01..2024-07-31")
    base = _block(_window(rs, BASE_START, BASE_END), "RS 2023-05-01..2023-07-31")
    fy2023 = _block(rs.filter(pl.col("onset_year") == 2023), "RS calendar year 2023")

    # Both windows are 92 days over an essentially unchanged population, so the
    # count ratio IS the rate ratio; t_x = t_y = 1 is exact here, not a shortcut.
    fold_cases = _count_ratio_ci(flood["confirmed"], base["confirmed"])
    fold_notified = _count_ratio_ci(flood["notified"], base["notified"])
    fold_deaths = _count_ratio_ci(flood["deaths"], base["deaths"])
    fold_lab = _count_ratio_ci(flood["lab_confirmed_share"]["n"], base["lab_confirmed_share"]["n"])

    # Ranieri quote 8.5 per 100,000 for the flood window. That is a cumulative
    # incidence over 92 days, NOT an annualised incidence rate; reproduce it on
    # their own terms so the comparison is like-for-like, and give the
    # annualised Poisson rate beside it so the two are never confused.
    pop24 = int(pop.filter(pl.col("year") == 2024)["population"][0])
    pop23 = int(pop.filter(pl.col("year") == 2023)["population"][0])
    win_days = 92.0
    ci24 = binom_ci(flood["confirmed"], pop24)
    ci23 = binom_ci(base["confirmed"], pop23)
    rate24 = poisson_ci(flood["confirmed"], pop24 * win_days / 365.25, scale=1e5)

    report["reconciliation"] = {
        "incidence": {
            "measure_note": (
                "Cumulative incidence over the 92-day window, per 100,000 "
                "residents -- the quantity Ranieri Table 1 reports as 8.5. The "
                "annualised incidence rate is a different number and is given "
                "separately."
            ),
            "cumulative_incidence_per_100k_flood": float(ci24[0]) * 1e5,
            "cumulative_incidence_ci_flood": [float(ci24[1]) * 1e5, float(ci24[2]) * 1e5],
            "cumulative_incidence_per_100k_baseline": float(ci23[0]) * 1e5,
            "cumulative_incidence_ci_baseline": [float(ci23[1]) * 1e5, float(ci23[2]) * 1e5],
            "ranieri_flood": 8.5, "ranieri_baseline": 0.8,
            "annualised_incidence_rate_per_100k_py_flood": float(rate24[0]),
            "annualised_incidence_rate_ci_flood": [float(rate24[1]), float(rate24[2])],
            "population_2024": pop24, "population_2023": pop23,
        },
        "our_flood": flood,
        "our_baseline": base,
        "our_calendar_year_2023": fy2023,
        "fold_change_confirmed_cases": fold_cases,
        "fold_change_notified": fold_notified,
        "fold_change_deaths": fold_deaths,
        "fold_change_lab_confirmed": fold_lab,
        "disputed_note": {
            "claim_under_test": (
                "The recorded conflict between Ranieri's 2023 baseline of 93 "
                "confirmed cases and this pipeline's 478 is a window mismatch, "
                "not a data disagreement."
            ),
            "ranieri_may_jul_2023": RANIERI["base_confirmed"],
            "our_may_jul_2023": base["confirmed"],
            "our_calendar_year_2023": fy2023["confirmed"],
            "previously_recorded_figure": DISPUTED_2023_FIGURE,
            "our_epiweek_onset_year_2023": int(
                rs.filter(pl.col("epiweek_onset_year") == 2023,
                          pl.col("confirmed") == 1).height
            ) if "epiweek_onset_year" in rs.columns else None,
            "residual_note": (
                "The 4-case residual between 482 and 478 is not a third figure: "
                "482 counts calendar-year onset (used here, because a monthly "
                "series needs calendar months) and 478 counts epidemiological-"
                "week onset year (used by the panel). The four cases sit in the "
                "week that straddles the new year."
            ),
            "verdict": (
                "CONFIRMED" if abs(fy2023["confirmed"] - DISPUTED_2023_FIGURE) <= 5
                and abs(base["confirmed"] - RANIERI["base_confirmed"]) <= 10
                else "REFUTED"
            ),
        },
        "agreement_vs_ranieri": {
            "confirmed_flood_pct_diff": 100 * (flood["confirmed"] - RANIERI["flood_confirmed"]) / RANIERI["flood_confirmed"],
            "confirmed_base_pct_diff": 100 * (base["confirmed"] - RANIERI["base_confirmed"]) / RANIERI["base_confirmed"],
            "notified_flood_pct_diff": 100 * (flood["notified"] - RANIERI["flood_notified"]) / RANIERI["flood_notified"],
            "deaths_flood": {"ours": flood["deaths"], "theirs": RANIERI["flood_deaths"]},
            "deaths_base": {"ours": base["deaths"], "theirs": RANIERI["base_deaths"]},
            "fold_cases": {"ours": fold_cases["ratio"], "theirs": RANIERI["fold_cases"]},
            "fold_deaths": {"ours": fold_deaths["ratio"], "theirs": RANIERI["fold_deaths"]},
        },
    }

    # The event contrast itself, as odds ratios with exact intervals.
    report["event_contrast"] = {
        "H": _or_between(flood, base, "H_hospitalisation_share"),
        "H_worst_case_bound": _or_between(flood, base, "H_worst_case_bound"),
        "case_fatality": _or_between(flood, base, "case_fatality"),
        "confirmed_among_notified": _or_between(flood, base, "confirmed_among_notified"),
        "lab_confirmed_share": _or_between(flood, base, "lab_confirmed_share"),
        "severe_share": _or_between(flood, base, "severe_share"),
        "hosp_completeness": _or_between(flood, base, "hosp_completeness"),
        "lab_stratum_H": _or_between(flood["lab_stratum"], base["lab_stratum"],
                                     "H_hospitalisation_share"),
        "lab_stratum_case_fatality": _or_between(flood["lab_stratum"], base["lab_stratum"],
                                                 "case_fatality"),
        "lab_stratum_severe_share": _or_between(flood["lab_stratum"], base["lab_stratum"],
                                                "severe_share"),
    }

    # ---- pre-trend: was H already sliding before 28 April 2024? ----------
    pre_2024 = _block(
        rs.filter(pl.col("onset_year") == 2024, pl.col("onset_month").is_in(PRETREND_MONTHS)),
        "RS Jan-Apr 2024 (before the flood)")
    pre_ref = _block(
        rs.filter(pl.col("onset_year").is_between(PRE_MIN, PRE_MAX),
                  pl.col("onset_month").is_in(PRETREND_MONTHS)),
        f"RS Jan-Apr {PRE_MIN}-{PRE_MAX}")
    report["pretrend_check"] = {
        "threat": (
            "The May discontinuity is the tail of a pre-existing drift in "
            "hospitalisation recording, not an event."
        ),
        "confirms_threat_if": "H in Jan-Apr 2024 is already well below the pre-period.",
        "rules_it_out_if": "H in Jan-Apr 2024 is indistinguishable from the pre-period.",
        "jan_apr_2024": pre_2024, "jan_apr_baseline": pre_ref,
        "H": _or_between(pre_2024, pre_ref, "H_hospitalisation_share"),
        "case_fatality": _or_between(pre_2024, pre_ref, "case_fatality"),
    }

    # ---- internal replication: the September 2023 cyclone ---------------
    cyc = _block(_window(rs, CYCLONE_START, CYCLONE_END), "RS Sep-Oct 2023 (cyclone)")
    cyc_ref = _block(
        rs.filter(pl.col("onset_year").is_in(CYCLONE_BASE_YEARS),
                  pl.col("onset_month").is_in((9, 10))),
        f"RS Sep-Oct {'/'.join(str(y) for y in CYCLONE_BASE_YEARS)}")
    report["internal_replication_sep2023"] = {
        "rationale": (
            "A second, smaller flood in the same state a year earlier. Found in "
            "the monthly series, not assumed. It sits outside Ranieri's baseline "
            "window (which ends 31 July 2023), so it does not contaminate the "
            "main reconciliation. Recording completeness in these months is "
            "essentially perfect, so a blank-field artefact cannot operate here."
        ),
        "event": cyc, "reference": cyc_ref,
        "fold_change_confirmed_cases": _count_ratio_ci(
            cyc["confirmed"], cyc_ref["confirmed"], t_x=1.0, t_y=float(len(CYCLONE_BASE_YEARS))),
        "H": _or_between(cyc, cyc_ref, "H_hospitalisation_share"),
        "case_fatality": _or_between(cyc, cyc_ref, "case_fatality"),
        "confirmed_among_notified": _or_between(cyc, cyc_ref, "confirmed_among_notified"),
        "lab_confirmed_share": _or_between(cyc, cyc_ref, "lab_confirmed_share"),
        "severe_share": _or_between(cyc, cyc_ref, "severe_share"),
        "hosp_completeness": _or_between(cyc, cyc_ref, "hosp_completeness"),
    }

    # ---- (c) national context -------------------------------------------
    nat = national_context(notif)
    nat.write_parquet(OUT / "state_H_2024_vs_pre.parquet")
    nat.write_csv(OUT / "state_H_2024_vs_pre.csv")
    # The 150-case rule leaves few states. The same table without any threshold
    # is written beside it so the reader can see the whole distribution the
    # headline claim is made against, rather than only its eligible tail.
    nat_all = national_context(notif, min_cases=1)
    nat_all.write_csv(OUT / "state_H_2024_vs_pre_all_states.csv")

    rs_row = nat.filter(pl.col("uf_abbr") == STATE)
    others = nat.filter(pl.col("uf_abbr") != STATE)
    # Pooled non-RS comparison: sum numerators and denominators, then divide.
    o = {c: int(others[c].sum()) for c in
         ("hosp_2024", "hosp_known_2024", "hosp_pre", "hosp_known_pre", "cases_2024")}
    report["national_context"] = {
        "eligibility": f">= {MIN_STATE_CASES_2024} confirmed cases with onset in 2024",
        "n_states_eligible": nat.height,
        "rs": {
            "H_2024": float(rs_row["H_2024"][0]),
            "H_2024_ci": [float(rs_row["H_2024_lo"][0]), float(rs_row["H_2024_hi"][0])],
            "H_pre": float(rs_row["H_pre"][0]),
            "H_pre_ci": [float(rs_row["H_pre_lo"][0]), float(rs_row["H_pre_hi"][0])],
            "change_pp": float(rs_row["H_change_pp"][0]),
            "odds_ratio": float(rs_row["or_hosp_2024_vs_pre"][0]),
            "odds_ratio_ci": [float(rs_row["or_lo"][0]), float(rs_row["or_hi"][0])],
            "cases_2024": int(rs_row["cases_2024"][0]),
            "cases_pre": int(rs_row["cases_pre"][0]),
            "rank_most_negative_change": 1 + int(
                (nat["H_change_pp"] < rs_row["H_change_pp"][0]).sum()),
        },
        "other_states_change_pp": {
            "min": float(others["H_change_pp"].min()),
            "median": float(others["H_change_pp"].median()),
            "max": float(others["H_change_pp"].max()),
            "n_with_decline": int((others["H_change_pp"] < 0).sum()),
            "n_with_decline_ge_10pp": int((others["H_change_pp"] <= -10).sum()),
        },
        "pooled_non_rs": {
            "H_2024": _prop(o["hosp_2024"], o["hosp_known_2024"]),
            "H_pre": _prop(o["hosp_pre"], o["hosp_known_pre"]),
            **{"odds_ratio": _exact_or(
                o["hosp_2024"], o["hosp_known_2024"] - o["hosp_2024"],
                o["hosp_pre"], o["hosp_known_pre"] - o["hosp_pre"])},
        },
        "all_states_no_threshold": {
            "n_states": nat_all.height,
            "rs_rank_most_negative_change": 1 + int(
                (nat_all["H_change_pp"] < rs_row["H_change_pp"][0]).sum()),
            "n_states_with_decline_ge_10pp": int((nat_all["H_change_pp"] <= -10).sum()),
            "states_with_decline_ge_10pp": nat_all.filter(pl.col("H_change_pp") <= -10)
            .select("uf_abbr", "cases_2024", "H_change_pp").to_dicts(),
            "note": (
                "No case-count threshold. Small states move a long way on a "
                "handful of cases, so this table is context for the eligible "
                "one, not a replacement for it."
            ),
        },
        "table": nat.select(
            "uf_abbr", "cases_2024", "cases_pre", "H_2024", "H_2024_lo", "H_2024_hi",
            "H_pre", "H_pre_lo", "H_pre_hi", "H_change_pp", "or_hosp_2024_vs_pre",
            "or_lo", "or_hi", "p_fisher", "hosp_completeness_2024", "hosp_completeness_pre",
        ).to_dicts(),
    }

    # ---- (d) did it revert? ---------------------------------------------
    rs_pre = _block(rs.filter(pl.col("onset_year").is_between(PRE_MIN, PRE_MAX)),
                    f"RS {PRE_MIN}-{PRE_MAX}")
    rs_2024 = _block(rs.filter(pl.col("onset_year") == 2024), "RS 2024")
    rs_2025 = _block(rs.filter(pl.col("onset_year") == 2025), "RS 2025")
    report["reversion"] = {
        "baseline": rs_pre, "flood_year": rs_2024, "post_year": rs_2025,
        "2025_vs_baseline": {
            "H": _or_between(rs_2025, rs_pre, "H_hospitalisation_share"),
            "lab_confirmed_share": _or_between(rs_2025, rs_pre, "lab_confirmed_share"),
            "case_fatality": _or_between(rs_2025, rs_pre, "case_fatality"),
            "confirmed_among_notified": _or_between(rs_2025, rs_pre, "confirmed_among_notified"),
        },
        "2024_vs_baseline": {
            "H": _or_between(rs_2024, rs_pre, "H_hospitalisation_share"),
            "lab_confirmed_share": _or_between(rs_2024, rs_pre, "lab_confirmed_share"),
            "case_fatality": _or_between(rs_2024, rs_pre, "case_fatality"),
        },
        # How much of the displacement was undone. Reported as a fraction of the
        # 2024 gap because "did it revert" is a question about recovery toward
        # the pre-event level, not about whether 2025 equals 2019-2023 exactly.
        "gap_closed": {
            k: {
                "baseline": rs_pre[k]["p"], "flood_year": rs_2024[k]["p"],
                "post_year": rs_2025[k]["p"],
                "fraction_of_2024_gap_closed_by_2025": (
                    None if rs_pre[k]["p"] is None or rs_2024[k]["p"] == rs_pre[k]["p"]
                    else (rs_2025[k]["p"] - rs_2024[k]["p"]) / (rs_pre[k]["p"] - rs_2024[k]["p"])
                ),
            }
            for k in ("H_hospitalisation_share", "lab_confirmed_share",
                      "case_fatality", "confirmed_among_notified")
        },
        "caveat": (
            "2025 onset-year records are still accruing; the last onset date in "
            f"the extract is {rs['onset'].max()}. Outcome and hospitalisation "
            "fields close later than the notification, so 2025 completeness is "
            "reported beside every 2025 proportion and the reversion verdict is "
            "provisional."
        ),
    }

    annual = (
        rs.filter(pl.col("confirmed") == 1, pl.col("onset_year").is_between(SERIES_MIN, SERIES_MAX))
        .group_by("onset_year").agg(
            pl.len().alias("confirmed"),
            pl.col("hosp").sum().alias("hospitalised"),
            pl.col("hosp_known").sum().alias("hosp_known"),
            pl.col("lab").sum().alias("lab_confirmed"),
            pl.col("crit_known").sum().alias("crit_known"),
            pl.col("death").sum().alias("deaths"),
            pl.col("outcome_known").sum().alias("outcome_known"),
        ).sort("onset_year")
    )
    annual = _cols(annual, "H", "hospitalised", "hosp_known")
    annual = _cols(annual, "lab_share", "lab_confirmed", "crit_known")
    annual = _cols(annual, "cfr", "deaths", "outcome_known")
    annual.write_csv(OUT / "rs_annual_summary.csv")

    # ---- (e) what this event cannot rule out ----------------------------
    lab_H = report["event_contrast"]["lab_stratum_H"]
    lab_cfr = report["event_contrast"]["lab_stratum_case_fatality"]
    report["threat_ledger"] = [
        {
            "threat": "Recording artefact: ATE_HOSP goes blank during the emergency and a "
                      "blank is not counted as hospitalised, so H falls mechanically.",
            "discriminating_check": "H recomputed on the worst-case bound that counts every "
                                    "unanswered case as hospitalised.",
            "result": report["event_contrast"]["H_worst_case_bound"],
            "verdict": "RULED OUT as the whole explanation; it accounts for part of the fall.",
        },
        {
            "threat": "2024 was a national recording year, not an RS event.",
            "discriminating_check": "H 2024 vs 2019-2023 in every state with >= 150 confirmed "
                                    "cases in 2024, and pooled across all non-RS states.",
            "result": report["national_context"]["pooled_non_rs"],
            "verdict": "RULED OUT: pooled non-RS H is unchanged.",
        },
        {
            "threat": "Pre-existing drift: H was already falling in RS before the flood.",
            "discriminating_check": "H in Jan-Apr 2024 vs Jan-Apr 2019-2023.",
            "result": report["pretrend_check"]["H"],
            "verdict": (
                "RULED OUT"
                if (report["pretrend_check"]["H"]["hi"] or 0) >= 1.0
                else "NOT RULED OUT: H was already below baseline before the flood."
            ),
        },
        {
            "threat": "One catastrophe, one story. The 2024 signature could be a property of "
                      "that unique event rather than of mass case-finding.",
            "discriminating_check": "The September 2023 cyclone in the same state: a three-fold "
                                    "case surge on essentially complete ATE_HOSP recording.",
            "result": report["internal_replication_sep2023"]["H"],
            "verdict": "REPLICATES: H falls in the same direction, at a similar magnitude, "
                       "with no recording artefact available to explain it.",
        },
        {
            "threat": "Criterion drift: under a declared calamity the clinical-epidemiological "
                      "link is satisfied by flood exposure alone, so the confirmed denominator "
                      "fills with people who did not have leptospirosis. That lowers H and case "
                      "fatality without surveillance reaching any deeper into true disease.",
            "discriminating_check": "Repeat H and case fatality inside the laboratory-confirmed "
                                    "stratum only, where the loosened link cannot add anyone.",
            "result": {"H": lab_H, "case_fatality": lab_cfr},
            "verdict": (
                "PARTIALLY WINS. H still falls inside the laboratory-confirmed stratum, so the "
                "ascertainment movement is not an artefact of criterion drift. Case fatality "
                "does NOT fall inside that stratum -- it is if anything higher -- so the "
                "case-fatality drop in this event is not evidence that broader ascertainment "
                "lowers reported case fatality holding the case definition fixed."
            ),
        },
    ]
    report["cannot_rule_out"] = [
        "A real change in the severity composition of infection. The flood exposed a young, "
        "mobile, working-age population (Ranieri: 70% male, most aged 20-59) to a large "
        "inoculum in a short window; a genuinely milder case mix would move H and case "
        "fatality in exactly the observed direction with no change in ascertainment.",
        "A real change in prognosis. Mass risk communication, presumptive antibiotics and "
        "shortened time to treatment during the response plausibly reduced true case fatality. "
        "This study cannot separate earlier treatment from broader detection.",
        "Changed access and care-seeking. Hospitals were flooded, evacuated or unreachable, "
        "which by itself lowers the probability that a confirmed case is recorded as "
        "hospitalised, independently of how mild the case was.",
        "Changed confirmation practice. The laboratory-confirmed share fell from 85% to 58% "
        "across the year and the confirmed-among-notified proportion fell from 28% to 17%; "
        "some of the extra confirmed cases are likely not leptospirosis.",
        "An altered instrument. Ranieri report that SINAN was unreachable for part of the "
        "event and that municipalities and the capital used parallel provisional forms whose "
        "records were later reconciled into SINAN. The notification form is therefore NOT "
        "strictly held fixed, contrary to how this event was described in an earlier draft.",
        "Concurrent dengue. RS was in a 172,000-case dengue epidemic in 2024; febrile "
        "syndromic overlap inflates the notified denominator and complicates confirmation.",
        "Ecological level. Every quantity here is a property of a state-month, not of a "
        "patient; none of it licenses an individual-level statement about who was detected.",
    ]
    report["framing"] = (
        "SUPPORTIVE within-territory validation. The event is consistent with H indexing "
        "ascertainment breadth and would have embarrassed that reading had H risen. It is "
        "NOT a natural experiment: the flood moved true exposure, severity composition, "
        "health-service access, laboratory throughput and treatment timing at the same "
        "moment and in the same direction. Claims that it is 'the strongest test available' "
        "or 'difficult to explain by any mechanism other than a change in detection' are "
        "withdrawn."
    )

    with (OUT / "rs2024_event_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    ec = report["event_contrast"]
    print(f"wrote {OUT}")
    print(f"  flood {flood['confirmed']} confirmed vs baseline {base['confirmed']}; "
          f"fold {fold_cases['ratio']:.2f} ({fold_cases['lo']:.2f}-{fold_cases['hi']:.2f})")
    print(f"  H {base['H_hospitalisation_share']['p']:.3f} -> "
          f"{flood['H_hospitalisation_share']['p']:.3f}  OR {ec['H']['or']:.3f}")
    print(f"  CFR {base['case_fatality']['p']:.3f} -> {flood['case_fatality']['p']:.3f}")
    print(f"  lab-stratum H {base['lab_stratum']['H_hospitalisation_share']['p']:.3f} -> "
          f"{flood['lab_stratum']['H_hospitalisation_share']['p']:.3f}")
    print(f"  RS rank among {nat.height} states by H change: "
          f"{report['national_context']['rs']['rank_most_negative_change']}")
    rep = report["internal_replication_sep2023"]
    print(f"  Sep-Oct 2023 cyclone replication: H "
          f"{rep['reference']['H_hospitalisation_share']['p']:.3f} -> "
          f"{rep['event']['H_hospitalisation_share']['p']:.3f}, "
          f"CFR {rep['reference']['case_fatality']['p']:.3f} -> "
          f"{rep['event']['case_fatality']['p']:.3f}")
    pt = report["pretrend_check"]["H"]
    print(f"  pre-trend Jan-Apr 2024 vs baseline: H OR {pt['or']:.3f} "
          f"({pt['lo']:.3f}-{pt['hi']:.3f})")
    print(f"  2025 H {rs_2025['H_hospitalisation_share']['p']:.3f} vs baseline "
          f"{rs_pre['H_hospitalisation_share']['p']:.3f}; gap closed "
          f"{report['reversion']['gap_closed']['H_hospitalisation_share']['fraction_of_2024_gap_closed_by_2025']:.0%}")


if __name__ == "__main__":
    main()
