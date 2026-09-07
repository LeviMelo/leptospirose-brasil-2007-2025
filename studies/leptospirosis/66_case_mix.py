"""Does demographic case mix explain the case-fatality gradient across H?

**Why this exists.** The case-fatality gradient across the ascertainment-breadth
exposure H (hospitalised confirmed cases / confirmed cases with a valid
hospitalisation field; higher H = *narrower* ascertainment) runs roughly 2.7% to
17% from the broadest to the narrowest quintile. The obvious hostile reading is
that this is not surveillance at all but case mix: narrow-ascertainment
territories might simply notify older patients, and leptospirosis case fatality
rises steeply with age.

The earlier draft dismissed that reading with the wrong evidence. It
age-standardised **incidence** — which says nothing about confounding of a
*conditional-on-being-a-case* quantity — and compared the **median** age of
hospitalised patients, which is blind to precisely the tail that carries the
deaths. Two case series with the same median age can differ several-fold in the
share aged 70+, and that share is where the case fatality is 25%.

**What would falsify the claim this analysis supports.** The claim is that the
gradient is not a composition effect. It would be falsified if (i) the age-sex
distribution of confirmed cases shifted materially older across H quintiles AND
(ii) directly standardising case fatality to a common national age-sex
distribution collapsed the Q5/Q1 ratio toward 1, AND (iii) the gradient
disappeared inside broad age strata. Any one of those three coming out the wrong
way weakens the link; all three would kill it. All three are run here, and (iii)
is the decisive one, because a within-stratum gradient cannot be produced by
between-stratum composition however the weights are chosen.

**The counterfactual the objection actually asserts.** Alongside the direct
standardisation, each quintile's own case mix is run through the *national*
age-sex case-fatality schedule. The resulting expected case fatality is the
whole of what composition can generate on its own, so the Q5/Q1 ratio of the
expected values is a hard ceiling on the confounder's contribution — no
weighting scheme can extract more gradient from case mix than case mix
contains. The observed-over-expected ratio is a standardised case-fatality
ratio (indirect standardisation), reported with an exact Poisson interval on
the observed count, and its Q5/Q1 ratio should agree with the directly
standardised one; the two are computed independently precisely so that
disagreement would show.

**A note on the interval for a standardised quantity.** Clopper-Pearson is exact
for a crude proportion and is used for every crude number here. A directly
standardised proportion is a weighted sum of proportions and has no exact
interval; the gamma method of Fay & Feuer (1997) is used instead, and is
cross-checked against a parametric bootstrap that resamples deaths within each
age-sex stratum. If the two disagree the gamma interval is not to be trusted,
so the comparison is reported rather than assumed.

Outputs to ``data/results/case_mix/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "case_mix"
PANEL = PATHS.results / "analysis_panel"

YEAR_MIN, YEAR_MAX = 2007, 2025
N_BINS = 5

#: Eligibility for entering the H quintiles. H must be measured on enough
#: records to be an estimate rather than a coin flip, and the case fatality it
#: is compared against needs a denominator that is not dominated by sampling
#: error. These are the same judgements the canonical gradient analysis makes
#: (``40_ascertainment_depth.py``), restated on the panel's field rules: H's
#: denominator is ``hosp_known`` (valid ATE_HOSP), not all cases.
MIN_HOSP_KNOWN = 30
MIN_OUTCOME_KNOWN = 20

#: Age bands. Cut at 60 and again at 70 because leptospirosis case fatality
#: keeps rising past 60 — collapsing 60+ into one band would hide the stratum
#: with the highest fatality and the smallest count, which is exactly the tail
#: the median-age argument missed.
AGE_BANDS = ["0-14", "15-29", "30-44", "45-59", "60-69", "70+"]
AGE_CUTS = [15, 30, 45, 60, 70]

#: Three broad strata for the stratified check. Wide enough that every H
#: quintile has a usable death count inside each one; the six-band scheme above
#: would leave the 70+ x Q1 cell too thin to read.
BROAD_STRATA = ["<30", "30-59", "60+"]

BOOT_REPS = 4000
SEED = 20260814


# --------------------------------------------------------------------------
# small statistical helpers
# --------------------------------------------------------------------------

def _katz_ratio(x1: int, n1: int, x0: int, n0: int) -> tuple[float, float, float]:
    """Ratio of two independent proportions with a log-scale (Katz) interval.

    Returns ``(ratio, lower, upper)`` -- estimate FIRST, matching the house
    convention of ``binom_ci``/``poisson_ci``. Every ratio reported here rests
    on hundreds of deaths per arm, which is where the log-scale interval is
    well behaved; it is not used anywhere a cell is thin.
    """
    if x1 == 0 or x0 == 0 or n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), float("nan")
    p1, p0 = x1 / n1, x0 / n0
    r = p1 / p0
    se = np.sqrt(1 / x1 - 1 / n1 + 1 / x0 - 1 / n0)
    z = stats.norm.ppf(0.975)
    return float(r), float(r * np.exp(-z * se)), float(r * np.exp(z * se))


def _direct_standardise(
    deaths: np.ndarray, known: np.ndarray, weight: np.ndarray, conf: float = 0.95
) -> dict:
    """Directly standardised proportion with a Fay-Feuer gamma interval.

    ``weight`` is the standard population's share in each stratum; strata the
    index population cannot estimate (``known == 0``) are dropped and the
    remaining weights renormalised, with the dropped mass reported as
    ``weight_covered`` so a reader can see how much of the standard was
    actually reproduced. Silently treating an empty stratum as zero fatality
    would bias the standardised value downward exactly where counts are thin.
    """
    d = np.asarray(deaths, dtype=float)
    n = np.asarray(known, dtype=float)
    w = np.asarray(weight, dtype=float)

    usable = n > 0
    covered = float(w[usable].sum() / w.sum())
    w = np.where(usable, w, 0.0)
    w = w / w.sum()

    p = np.where(usable, d / np.maximum(n, 1), 0.0)
    dsr = float((w * p).sum())
    # Binomial variance of each stratum-specific proportion. The Fay-Feuer
    # construction was derived for Poisson counts; using the (smaller)
    # binomial variance makes the interval, if anything, conservative here.
    var = float((w**2 * np.where(usable, p * (1 - p) / np.maximum(n, 1), 0.0)).sum())
    wm = float(np.max(np.where(usable, w / np.maximum(n, 1), 0.0)))

    a = 1.0 - conf
    if dsr <= 0 or var <= 0:
        return {"dsr": dsr, "lo": 0.0, "hi": float("nan"),
                "weight_covered": covered, "strata_used": int(usable.sum())}
    lo = var / (2 * dsr) * stats.chi2.ppf(a / 2, 2 * dsr**2 / var)
    hi = ((var + wm**2) / (2 * (dsr + wm))
          * stats.chi2.ppf(1 - a / 2, 2 * (dsr + wm) ** 2 / (var + wm**2)))
    return {"dsr": dsr, "lo": float(lo), "hi": float(hi),
            "weight_covered": covered, "strata_used": int(usable.sum())}


def _boot_dsr(
    deaths: np.ndarray, known: np.ndarray, weight: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Parametric bootstrap draws of a directly standardised proportion.

    Deaths in each stratum are redrawn Binomial(n_i, p_hat_i). This is the
    crosscheck on the gamma interval, and it also supplies an interval for the
    Q5/Q1 ratio, for which no closed form of the right shape exists.
    """
    n = np.asarray(known, dtype=float)
    w = np.asarray(weight, dtype=float)
    usable = n > 0
    w = np.where(usable, w, 0.0)
    w = w / w.sum()
    p = np.where(usable, np.asarray(deaths, float) / np.maximum(n, 1), 0.0)
    draws = rng.binomial(n.astype(int), p, size=(BOOT_REPS, n.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = np.where(usable, draws / np.maximum(n, 1), 0.0)
    return ps @ w


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _line() -> pl.DataFrame:
    """Analytic population, with the demographic and outcome flags this needs.

    Field rules are the panel's, restated rather than imported so that a defect
    here cannot be blamed on a silent upstream change: death is
    ``evolucao == obito_por_leptospirose`` on ``evolucao_state == valid`` only,
    and ``obito_por_outras_causas`` is a known outcome that is not a case
    fatality.
    """
    cols = ["classi_fin", "epiweek_onset_year", "municipality_residence_code7",
            "evolucao", "evolucao_state", "ate_hosp", "ate_hosp_state",
            "age_years", "age_state", "cs_sexo"]
    line = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .select(cols)
        .filter(
            (pl.col("classi_fin") == "confirmado")
            & pl.col("epiweek_onset_year").is_between(YEAR_MIN, YEAR_MAX)
            & pl.col("municipality_residence_code7").is_not_null()
        )
        .collect()
    )
    assert line.height == 66358, f"analytic population is {line.height}, expected 66358"

    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    geo = atlas.select(
        pl.col("munic_code").alias("municipality_residence_code7"),
        "health_region_code", "uf_abbr", "region",
    )
    line = line.join(geo, on="municipality_residence_code7", how="left")
    unmatched = line.filter(pl.col("health_region_code").is_null()).height
    assert unmatched / line.height < 0.01, f"{unmatched} cases without a health region"
    line = line.filter(pl.col("health_region_code").is_not_null())

    band = pl.when(pl.col("age_state") != "valid").then(None)
    for cut, label in zip(AGE_CUTS, AGE_BANDS[:-1]):
        band = band.when(pl.col("age_years") < cut).then(pl.lit(label))
    band = band.otherwise(pl.lit(AGE_BANDS[-1]))

    broad = (
        pl.when(pl.col("age_state") != "valid").then(None)
        .when(pl.col("age_years") < 30).then(pl.lit("<30"))
        .when(pl.col("age_years") < 60).then(pl.lit("30-59"))
        .otherwise(pl.lit("60+"))
    )

    return line.with_columns(
        band.alias("age_band"),
        broad.alias("age_broad"),
        # Sex is coded, not stated-unknown: cs_sexo is null for 4 records and
        # never 'ignorado' in this extract. Null is excluded from the sex
        # denominator rather than folded into 'female'.
        pl.col("cs_sexo").alias("sex"),
        ((pl.col("evolucao_state") == "valid")
         & (pl.col("evolucao") == "obito_por_leptospirose")).cast(pl.Int32).alias("death"),
        (pl.col("evolucao_state") == "valid").cast(pl.Int32).alias("outcome_known"),
        ((pl.col("ate_hosp_state") == "valid")
         & (pl.col("ate_hosp") == "sim")).cast(pl.Int32).alias("hospitalised"),
        (pl.col("ate_hosp_state") == "valid").cast(pl.Int32).alias("hosp_known"),
    )


def _quintiles(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Case-weighted H quintiles over health regions, 2007-2025.

    Bands hold equal case mass rather than equal region counts, so each band's
    case fatality is estimated with comparable precision; equal-count bands
    would put most of the national case load into one band.
    """
    reg = line.group_by(["health_region_code", "uf_abbr", "region"]).agg(
        pl.len().alias("cases"),
        pl.col("death").sum().alias("deaths"),
        pl.col("outcome_known").sum().alias("outcome_known"),
        pl.col("hospitalised").sum().alias("hospitalised"),
        pl.col("hosp_known").sum().alias("hosp_known"),
    ).with_columns((pl.col("hospitalised") / pl.col("hosp_known")).alias("H"))

    # Consistency gate: H recomputed here must equal the panel's H region by
    # region. A mismatch means the field rules drifted, not that a number moved.
    totals = pl.read_parquet(PANEL / "region_totals.parquet").select(
        "health_region_code", "cases", "hospitalised", "hosp_known", "deaths")
    chk = reg.join(totals, on="health_region_code", how="inner", suffix="_panel")
    assert chk.height == reg.height, "region set differs from the panel"
    for c in ("cases", "hospitalised", "hosp_known", "deaths"):
        bad = chk.filter(pl.col(c) != pl.col(f"{c}_panel")).height
        assert bad == 0, f"{bad} regions disagree with the panel on {c}"

    elig = reg.filter(
        (pl.col("hosp_known") >= MIN_HOSP_KNOWN)
        & (pl.col("outcome_known") >= MIN_OUTCOME_KNOWN)
    )
    q = elig.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("H_quintile")
    ).drop("_cw")

    meta = {
        "regions_total": reg.height,
        "regions_eligible": q.height,
        "cases_total": int(reg["cases"].sum()),
        "cases_eligible": int(q["cases"].sum()),
        "min_hosp_known": MIN_HOSP_KNOWN,
        "min_outcome_known": MIN_OUTCOME_KNOWN,
    }
    return q, meta


# --------------------------------------------------------------------------
# (a) the age-fatality relationship itself
# --------------------------------------------------------------------------

def age_fatality(line: pl.DataFrame) -> pl.DataFrame:
    """National case fatality by age band, exact intervals, own denominators.

    Establishing the magnitude of the confounder before adjusting for it: if
    age barely moved case fatality there would be nothing to adjust away and
    the whole objection would be empty.
    """
    rows = []
    grp = line.filter(pl.col("age_band").is_not_null())
    tot = grp.group_by("age_band").agg(
        pl.len().alias("cases"),
        pl.col("outcome_known").sum().alias("outcome_known"),
        pl.col("death").sum().alias("deaths"),
    )
    order = {b: i for i, b in enumerate(AGE_BANDS)}
    tot = tot.sort(pl.col("age_band").replace_strict(order, return_dtype=pl.Int32))

    ref = tot.filter(pl.col("age_band") == "15-29").row(0, named=True)
    for r in tot.iter_rows(named=True):
        cfr, lo, hi = binom_ci(int(r["deaths"]), int(r["outcome_known"]))
        rr, rlo, rhi = _katz_ratio(int(r["deaths"]), int(r["outcome_known"]),
                                   int(ref["deaths"]), int(ref["outcome_known"]))
        rows.append({
            "age_band": r["age_band"], "cases": int(r["cases"]),
            "outcome_known": int(r["outcome_known"]), "deaths": int(r["deaths"]),
            "outcome_completeness": float(r["outcome_known"] / r["cases"]),
            "cfr_pct": 100 * float(cfr), "cfr_lo_pct": 100 * float(lo),
            "cfr_hi_pct": 100 * float(hi),
            "cfp_ratio_vs_15_29": rr, "cfp_ratio_lo": rlo, "cfp_ratio_hi": rhi,
        })
    return pl.DataFrame(rows)


def age_sex_fatality(line: pl.DataFrame) -> pl.DataFrame:
    """The same, split by sex -- the twelve strata the standardisation uses."""
    d = line.filter(pl.col("age_band").is_not_null() & pl.col("sex").is_not_null())
    g = d.group_by(["age_band", "sex"]).agg(
        pl.len().alias("cases"),
        pl.col("outcome_known").sum().alias("outcome_known"),
        pl.col("death").sum().alias("deaths"),
    )
    cfr, lo, hi = binom_ci(g["deaths"].to_numpy(), g["outcome_known"].to_numpy())
    order = {b: i for i, b in enumerate(AGE_BANDS)}
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cfr), pl.Series("cfr_lo_pct", 100 * lo),
        pl.Series("cfr_hi_pct", 100 * hi),
    ).sort([pl.col("age_band").replace_strict(order, return_dtype=pl.Int32), "sex"])


# --------------------------------------------------------------------------
# (b) composition across the exposure
# --------------------------------------------------------------------------

def composition(line_q: pl.DataFrame) -> pl.DataFrame:
    """Full age distribution, proportion male and proportion 60+ per quintile.

    The proportion in *each* band, not the median: the median is insensitive to
    the tail that carries the deaths, which is the reason the earlier draft's
    demography argument did not address the confounder it claimed to.
    """
    rows = []
    for qv in range(1, N_BINS + 1):
        g = line_q.filter(pl.col("H_quintile") == qv)
        n_age = g.filter(pl.col("age_band").is_not_null()).height
        n_sex = g.filter(pl.col("sex").is_not_null()).height
        rec = {"H_quintile": qv, "cases": g.height, "age_known": n_age,
               "sex_known": n_sex,
               "H": float(g["hospitalised"].sum() / g["hosp_known"].sum()),
               # Carried here because it is the *other* thing that differs
               # across the exposure and could bias the comparison. It is not
               # this script's threat to close, but a reader comparing case
               # fatality across quintiles must be able to see it.
               "outcome_completeness": float(g["outcome_known"].sum() / g.height)}
        for b in AGE_BANDS:
            k = int((g["age_band"] == b).sum())
            p, lo, hi = binom_ci(k, n_age)
            rec[f"n_{b}"] = k
            rec[f"pct_{b}"] = 100 * float(p)
            rec[f"pct_{b}_lo"] = 100 * float(lo)
            rec[f"pct_{b}_hi"] = 100 * float(hi)
        k60 = int(g.filter(pl.col("age_band").is_in(["60-69", "70+"])).height)
        p, lo, hi = binom_ci(k60, n_age)
        rec.update({"n_60plus": k60, "pct_60plus": 100 * float(p),
                    "pct_60plus_lo": 100 * float(lo), "pct_60plus_hi": 100 * float(hi)})
        km = int((g["sex"] == "masculino").sum())
        p, lo, hi = binom_ci(km, n_sex)
        rec.update({"n_male": km, "pct_male": 100 * float(p),
                    "pct_male_lo": 100 * float(lo), "pct_male_hi": 100 * float(hi)})
        ages = g.filter(pl.col("age_band").is_not_null())["age_years"]
        rec.update({"age_mean": float(ages.mean()), "age_median": float(ages.median()),
                    "age_p90": float(ages.quantile(0.90)),
                    "age_p97_5": float(ages.quantile(0.975))})
        rows.append(rec)
    return pl.DataFrame(rows)


def composition_correlation(line_q: pl.DataFrame) -> dict:
    """Does case mix track H across regions, or only across five bins?

    Five quintile means can look like a trend by accident. The question "do
    narrow-ascertainment territories have older confirmed cases" is properly a
    rank-correlation question over the 208 eligible health regions, and a
    confounder that is not monotone in the exposure cannot generate a monotone
    gradient in the outcome. Reported as Spearman rho, which is a statement
    about ordering only and not an effect size.
    """
    reg = line_q.group_by("health_region_code").agg(
        pl.len().alias("cases"),
        pl.col("hospitalised").sum(), pl.col("hosp_known").sum(),
        pl.col("age_band").is_not_null().sum().alias("age_known"),
        pl.col("age_band").is_in(["60-69", "70+"]).sum().alias("n60"),
        pl.col("sex").is_not_null().sum().alias("sex_known"),
        (pl.col("sex") == "masculino").sum().alias("male"),
        pl.col("age_years").mean().alias("age_mean"),
        pl.col("death").sum().alias("deaths"),
        pl.col("outcome_known").sum().alias("outcome_known"),
    ).with_columns(
        (pl.col("hospitalised") / pl.col("hosp_known")).alias("H"),
        (pl.col("n60") / pl.col("age_known")).alias("share_60plus"),
        (pl.col("male") / pl.col("sex_known")).alias("share_male"),
        (pl.col("deaths") / pl.col("outcome_known")).alias("cfr"),
    )
    h = reg["H"].to_numpy()

    def rho(col: str) -> dict:
        r = stats.spearmanr(h, reg[col].to_numpy())
        return {"rho": float(r.statistic), "p": float(r.pvalue)}

    return {"n_regions": reg.height,
            "H_vs_share_60plus": rho("share_60plus"),
            "H_vs_mean_age": rho("age_mean"),
            "H_vs_share_male": rho("share_male"),
            "H_vs_cfr": rho("cfr")}


# --------------------------------------------------------------------------
# (c) direct age-sex standardisation
# --------------------------------------------------------------------------

def standardise(line_q: pl.DataFrame, line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Crude and directly standardised case fatality per H quintile.

    The standard is the age-sex distribution of **all** confirmed cases
    nationally (all 432 regions, not only the eligible ones), so the same fixed
    external weights apply to every quintile and the standardised values are
    mutually comparable by construction.

    Crude here is computed on the same subset the standardised value uses
    (cases with a valid age, a coded sex and a known outcome) so that the two
    differ only by the reweighting and not by the denominator. The one case
    with an invalid age and the four with no coded sex make this a distinction
    of about a thousandth of a percentage point, which is asserted below rather
    than assumed.
    """
    nat = line.filter(pl.col("age_band").is_not_null() & pl.col("sex").is_not_null())
    std = nat.group_by(["age_band", "sex"]).agg(
        pl.len().alias("std_cases"),
        pl.col("outcome_known").sum().alias("nat_known"),
        pl.col("death").sum().alias("nat_deaths"),
    )
    std = std.with_columns(
        (pl.col("std_cases") / pl.col("std_cases").sum()).alias("w"),
        # The national age-sex case-fatality schedule, used below for the
        # counterfactual "what gradient could case mix alone produce".
        (pl.col("nat_deaths") / pl.col("nat_known")).alias("p_nat"),
    )
    strata = std.select("age_band", "sex", "std_cases", "w", "p_nat").sort(
        ["age_band", "sex"])
    assert abs(float(strata["w"].sum()) - 1.0) < 1e-12

    rng = np.random.default_rng(SEED)
    rows, boots = [], {}
    for qv in range(1, N_BINS + 1):
        g = line_q.filter(
            (pl.col("H_quintile") == qv)
            & pl.col("age_band").is_not_null() & pl.col("sex").is_not_null()
        )
        cells = g.group_by(["age_band", "sex"]).agg(
            pl.col("outcome_known").sum().alias("known"),
            pl.col("death").sum().alias("deaths"),
        )
        m = strata.join(cells, on=["age_band", "sex"], how="left").with_columns(
            pl.col(["known", "deaths"]).fill_null(0)
        ).sort(["age_band", "sex"])

        d_tot, n_tot = int(m["deaths"].sum()), int(m["known"].sum())
        crude, clo, chi = binom_ci(d_tot, n_tot)

        # Full-denominator crude, for continuity with the published gradient.
        gall = line_q.filter(pl.col("H_quintile") == qv)
        d_all, n_all = int(gall["death"].sum()), int(gall["outcome_known"].sum())
        crude_all, _, _ = binom_ci(d_all, n_all)
        assert abs(float(crude) - float(crude_all)) < 5e-4, (
            f"Q{qv}: restricting to coded age and sex moved crude case fatality by "
            f"{100 * abs(float(crude) - float(crude_all)):.3f} points -- demographic "
            "coding is not near-complete after all, and the comparison is unsafe")

        st = _direct_standardise(m["deaths"].to_numpy(), m["known"].to_numpy(),
                                 m["w"].to_numpy())
        boots[qv] = _boot_dsr(m["deaths"].to_numpy(), m["known"].to_numpy(),
                              m["w"].to_numpy(), rng)

        # Age-only standardisation, to attribute whatever movement occurs to
        # the age margin or the sex margin rather than to "demography".
        ma = m.group_by("age_band").agg(
            pl.col(["std_cases", "known", "deaths"]).sum()
        ).with_columns((pl.col("std_cases") / pl.col("std_cases").sum()).alias("w"))
        sta = _direct_standardise(ma["deaths"].to_numpy(), ma["known"].to_numpy(),
                                  ma["w"].to_numpy())

        # The counterfactual the objection actually asserts: apply the NATIONAL
        # age-sex case-fatality schedule to THIS quintile's case mix. The
        # resulting expected case fatality is the entire gradient that
        # composition on its own is capable of generating. Its Q5/Q1 ratio is
        # therefore the ceiling on the confounder's contribution, and the
        # observed-over-expected ratio is a standardised case-fatality ratio
        # (indirect standardisation) with an exact Poisson interval on the
        # observed count.
        expected_deaths = float((m["known"].to_numpy() * m["p_nat"].to_numpy()).sum())
        expected_cfr = expected_deaths / n_tot
        sc, sc_lo, sc_hi = poisson_ci(d_tot, expected_deaths, scale=1.0)

        rows.append({
            "H_quintile": qv, "regions": int(g["health_region_code"].n_unique()),
            "cases": g.height, "outcome_known": n_tot, "deaths": d_tot,
            "H": float(gall["hospitalised"].sum() / gall["hosp_known"].sum()),
            "crude_cfr_pct": 100 * float(crude),
            "crude_lo_pct": 100 * float(clo), "crude_hi_pct": 100 * float(chi),
            "std_cfr_pct": 100 * st["dsr"],
            "std_lo_pct": 100 * st["lo"], "std_hi_pct": 100 * st["hi"],
            "std_boot_lo_pct": 100 * float(np.percentile(boots[qv], 2.5)),
            "std_boot_hi_pct": 100 * float(np.percentile(boots[qv], 97.5)),
            "std_age_only_pct": 100 * sta["dsr"],
            "std_age_only_lo_pct": 100 * sta["lo"],
            "std_age_only_hi_pct": 100 * sta["hi"],
            "expected_cfr_pct": 100 * expected_cfr,
            "expected_deaths": expected_deaths,
            "scfr_obs_over_exp": float(sc),
            "scfr_lo": float(sc_lo), "scfr_hi": float(sc_hi),
            "strata_used": st["strata_used"], "weight_covered": st["weight_covered"],
        })

    t = pl.DataFrame(rows)
    q1, q5 = rows[0], rows[-1]

    crude_r, crude_rlo, crude_rhi = _katz_ratio(
        q5["deaths"], q5["outcome_known"], q1["deaths"], q1["outcome_known"])
    br = boots[N_BINS] / boots[1]
    std_r = q5["std_cfr_pct"] / q1["std_cfr_pct"]

    summary = {
        "standard_population": "all confirmed cases nationally with coded age and sex",
        "standard_n": int(strata["std_cases"].sum()),
        "strata": int(strata.height),
        "crude_Q1_pct": q1["crude_cfr_pct"], "crude_Q5_pct": q5["crude_cfr_pct"],
        "std_Q1_pct": q1["std_cfr_pct"], "std_Q5_pct": q5["std_cfr_pct"],
        "crude_Q5_over_Q1": crude_r,
        "crude_Q5_over_Q1_lo": crude_rlo, "crude_Q5_over_Q1_hi": crude_rhi,
        "std_Q5_over_Q1": std_r,
        "std_Q5_over_Q1_lo": float(np.percentile(br, 2.5)),
        "std_Q5_over_Q1_hi": float(np.percentile(br, 97.5)),
        "gradient_retained_pct": 100 * (std_r - 1) / (crude_r - 1),
        "age_only_Q5_over_Q1": q5["std_age_only_pct"] / q1["std_age_only_pct"],
        # The ceiling on the confounder: the gradient case mix alone generates.
        "expected_Q1_pct": q1["expected_cfr_pct"], "expected_Q5_pct": q5["expected_cfr_pct"],
        "expected_Q5_over_Q1": q5["expected_cfr_pct"] / q1["expected_cfr_pct"],
        "scfr_Q1": q1["scfr_obs_over_exp"], "scfr_Q5": q5["scfr_obs_over_exp"],
        "scfr_Q5_over_Q1": q5["scfr_obs_over_exp"] / q1["scfr_obs_over_exp"],
        "crude_monotone_in_H": bool(all(
            a["crude_cfr_pct"] < b["crude_cfr_pct"] for a, b in zip(rows, rows[1:]))),
        "std_monotone_in_H": bool(all(
            a["std_cfr_pct"] < b["std_cfr_pct"] for a, b in zip(rows, rows[1:]))),
        "bootstrap_reps": BOOT_REPS,
        # The crosscheck on the gamma interval: largest absolute discrepancy
        # between the gamma and bootstrap bounds, in percentage points.
        "max_gamma_vs_bootstrap_gap_pp": float(max(
            max(abs(r["std_lo_pct"] - r["std_boot_lo_pct"]),
                abs(r["std_hi_pct"] - r["std_boot_hi_pct"])) for r in rows)),
    }
    return t, summary


# --------------------------------------------------------------------------
# (d) stratified check
# --------------------------------------------------------------------------

def stratified(line_q: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Case fatality across H quintiles *within* three broad age strata.

    A composition effect cannot survive this: if narrow-ascertainment
    territories had higher case fatality only because their cases are older,
    then holding age broadly fixed must flatten the gradient. If it does not
    flatten inside every stratum, the gradient is not composition.
    """
    rows = []
    for s in BROAD_STRATA:
        d = line_q.filter(pl.col("age_broad") == s)
        for qv in range(1, N_BINS + 1):
            g = d.filter(pl.col("H_quintile") == qv)
            n = int(g["outcome_known"].sum())
            k = int(g["death"].sum())
            cfr, lo, hi = binom_ci(k, n)
            rows.append({
                "age_stratum": s, "H_quintile": qv, "cases": g.height,
                "outcome_known": n, "deaths": k,
                "cfr_pct": 100 * float(cfr), "cfr_lo_pct": 100 * float(lo),
                "cfr_hi_pct": 100 * float(hi),
            })
    t = pl.DataFrame(rows)

    ratios = {}
    for s in BROAD_STRATA:
        d = t.filter(pl.col("age_stratum") == s).sort("H_quintile")
        a = d.row(0, named=True)
        b = d.row(N_BINS - 1, named=True)
        r, lo, hi = _katz_ratio(b["deaths"], b["outcome_known"],
                                a["deaths"], a["outcome_known"])
        ratios[s] = {
            "Q1_cfr_pct": a["cfr_pct"], "Q5_cfr_pct": b["cfr_pct"],
            "Q5_over_Q1": r, "Q5_over_Q1_lo": lo, "Q5_over_Q1_hi": hi,
            "Q1_outcome_known": a["outcome_known"], "Q5_outcome_known": b["outcome_known"],
            "monotone": bool(
                all(x < y for x, y in zip(d["cfr_pct"].to_list(),
                                          d["cfr_pct"].to_list()[1:]))),
        }
    return t, ratios


# --------------------------------------------------------------------------
# (e) the model covariate file
# --------------------------------------------------------------------------

def region_year_agesex(line: pl.DataFrame) -> pl.DataFrame:
    """Per health region and year: counts, outcome denominator and case mix.

    Counts are carried alongside proportions so the spatial model can weight
    them, and so any proportion in the file is auditable against its own
    denominator. Built over all 432 regions, not only the quintile-eligible
    ones: eligibility is a property of *this* analysis, not of the covariate.

    The row set is taken from the analysis panel rather than from the cases,
    so the file is one row per panel row and a join cannot silently drop or
    duplicate a cell. Region-years with no confirmed case therefore appear with
    zero counts and a **null** case mix — not a zero one. "No cases, so no age
    distribution" and "cases, none of them aged 60+" are different statements
    and a model that cannot tell them apart will fit the difference as signal.
    """
    d = line.with_columns(
        pl.col("age_band").is_not_null().cast(pl.Int32).alias("age_known_i"),
        pl.col("sex").is_not_null().cast(pl.Int32).alias("sex_known_i"),
        (pl.col("sex") == "masculino").cast(pl.Int32).alias("male_i"),
    )
    aggs = [
        pl.len().alias("cases_x"),
        pl.col("death").sum().alias("deaths_x"),
        pl.col("outcome_known").sum().alias("outcome_known_x"),
        pl.col("age_known_i").sum().alias("age_known"),
        pl.col("sex_known_i").sum().alias("sex_known"),
        pl.col("male_i").sum().alias("male"),
    ]
    for b in AGE_BANDS:
        aggs.append((pl.col("age_band") == b).sum().alias(f"n_age_{_slug(b)}"))

    g = (
        d.group_by(["health_region_code", "epiweek_onset_year"])
        .agg(aggs)
        .rename({"epiweek_onset_year": "year"})
    )

    panel = pl.read_parquet(PANEL / "region_year_panel.parquet").select(
        "health_region_code", "health_region_name", "uf_abbr", "region", "year",
        "cases", "deaths", "outcome_known")
    out = panel.join(g, on=["health_region_code", "year"], how="left")
    assert out.height == panel.height, "the case-mix join changed the panel row count"

    count_cols = ["age_known", "sex_known", "male"] + [
        f"n_age_{_slug(b)}" for b in AGE_BANDS]
    out = out.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int64) for c in count_cols]
        + [pl.col(c).fill_null(0) for c in ("cases_x", "deaths_x", "outcome_known_x")]
    )

    # Cell-for-cell agreement with the panel on the three quantities both files
    # carry. A disagreement means the field rules drifted between the two
    # scripts, which is a defect, not a discrepancy to be reconciled downstream.
    for c in ("cases", "deaths", "outcome_known"):
        bad = out.filter(pl.col(c) != pl.col(f"{c}_x")).height
        assert bad == 0, f"{bad} region-years disagree with the panel on {c}"
    out = out.drop("cases_x", "deaths_x", "outcome_known_x")

    def _prop(num: str, den: str) -> pl.Expr:
        return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)

    return out.with_columns(
        [_prop(f"n_age_{_slug(b)}", "age_known").alias(f"p_age_{_slug(b)}")
         for b in AGE_BANDS]
        + [_prop("male", "sex_known").alias("p_male")]
    ).with_columns(
        (pl.col("p_age_60_69") + pl.col("p_age_70p")).alias("p_age_60plus")
    ).sort(["health_region_code", "year"])


def _slug(band: str) -> str:
    return band.replace("-", "_").replace("+", "p")


# --------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = _line()
    q, meta = _quintiles(line)
    line_q = line.join(q.select("health_region_code", "H_quintile"),
                       on="health_region_code", how="inner")
    q.sort(["H_quintile", "H"]).write_csv(OUT / "H_quintile_assignment.csv")

    # (a)
    age = age_fatality(line)
    age.write_csv(OUT / "age_fatality_national.csv")
    agesex = age_sex_fatality(line)
    agesex.write_csv(OUT / "age_sex_fatality_national.csv")

    # (b)
    comp = composition(line_q)
    comp.write_csv(OUT / "composition_by_H_quintile.csv")
    comp_rho = composition_correlation(line_q)

    # (c)
    std, std_sum = standardise(line_q, line)
    std.write_csv(OUT / "standardised_cfr_by_H_quintile.csv")

    # (d)
    strat, strat_sum = stratified(line_q)
    strat.write_csv(OUT / "cfr_by_H_quintile_within_age_stratum.csv")

    # (e)
    cov = region_year_agesex(line)
    cov.write_csv(OUT / "region_year_agesex.csv")

    a_lo = age.row(0, named=True)
    a_hi = age.filter(pl.col("age_band") == "70+").row(0, named=True)
    report = {
        "window": [YEAR_MIN, YEAR_MAX],
        "eligibility": meta,
        "age_fatality": {
            "cfr_0_14_pct": a_lo["cfr_pct"], "cfr_70plus_pct": a_hi["cfr_pct"],
            "cfp_ratio_70plus_vs_0_14": a_hi["cfr_pct"] / a_lo["cfr_pct"],
            "cfp_ratio_70plus_vs_15_29": a_hi["cfp_ratio_vs_15_29"],
            "cfp_ratio_70plus_vs_15_29_lo": a_hi["cfp_ratio_lo"],
            "cfp_ratio_70plus_vs_15_29_hi": a_hi["cfp_ratio_hi"],
            "age_coding_completeness": float(
                line.filter(pl.col("age_band").is_not_null()).height / line.height),
            "sex_coding_completeness": float(
                line.filter(pl.col("sex").is_not_null()).height / line.height),
        },
        "composition": {
            "pct_60plus_Q1": comp.row(0, named=True)["pct_60plus"],
            "pct_60plus_Q5": comp.row(N_BINS - 1, named=True)["pct_60plus"],
            "pct_male_Q1": comp.row(0, named=True)["pct_male"],
            "pct_male_Q5": comp.row(N_BINS - 1, named=True)["pct_male"],
            "age_p90_Q1": comp.row(0, named=True)["age_p90"],
            "age_p90_Q5": comp.row(N_BINS - 1, named=True)["age_p90"],
            "age_mean_Q1": comp.row(0, named=True)["age_mean"],
            "age_mean_Q5": comp.row(N_BINS - 1, named=True)["age_mean"],
            "pct_60plus_max_quintile": int(
                comp.sort("pct_60plus", descending=True).row(0, named=True)["H_quintile"]),
            "pct_60plus_monotone_in_H": bool(all(
                x < y for x, y in zip(comp["pct_60plus"].to_list(),
                                      comp["pct_60plus"].to_list()[1:]))),
            "region_level": comp_rho,
        },
        "standardisation": std_sum,
        "stratified": strat_sum,
        "covariate_file": {
            "path": "data/results/case_mix/region_year_agesex.csv",
            "region_years": cov.height,
            "region_years_with_a_case": int((cov["cases"] > 0).sum()),
            "regions": int(cov["health_region_code"].n_unique()),
        },
    }
    (OUT / "case_mix_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    pd_opts = dict(index=False)
    print("=== (a) national case fatality by age band ===")
    print(age.select("age_band", "cases", "outcome_known", "deaths",
                     "cfr_pct", "cfr_lo_pct", "cfr_hi_pct",
                     "cfp_ratio_vs_15_29").to_pandas().to_string(**pd_opts))
    print("\n=== (b) composition across H quintiles ===")
    print(comp.select(["H_quintile", "H", "cases"]
                      + [f"pct_{b}" for b in AGE_BANDS]
                      + ["pct_60plus", "pct_male", "age_mean", "age_median",
                         "age_p90", "outcome_completeness"]
                      ).to_pandas().round(3).to_string(**pd_opts))
    print(json.dumps(comp_rho, indent=2))
    print("\n=== (c) crude and age-sex standardised case fatality ===")
    print(std.select("H_quintile", "H", "outcome_known", "deaths",
                     "crude_cfr_pct", "crude_lo_pct", "crude_hi_pct",
                     "std_cfr_pct", "std_lo_pct", "std_hi_pct",
                     "std_boot_lo_pct", "std_boot_hi_pct",
                     "std_age_only_pct", "expected_cfr_pct",
                     "scfr_obs_over_exp", "scfr_lo",
                     "scfr_hi").to_pandas().round(3).to_string(**pd_opts))
    print(json.dumps(std_sum, indent=2))
    print("\n=== (d) within broad age strata ===")
    print(strat.to_pandas().round(3).to_string(**pd_opts))
    print(json.dumps(strat_sum, indent=2))
    print("\n=== (e) covariate file ===")
    print(cov.head(3).to_pandas().to_string(**pd_opts))
    print(f"{cov.height} region-years written")


if __name__ == "__main__":
    main()
