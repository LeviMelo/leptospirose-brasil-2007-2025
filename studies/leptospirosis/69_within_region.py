"""Does reported case fatality follow ascertainment *inside* a single territory?

The paper's primary claim is cross-sectional: health regions whose confirmed
cases are almost all hospitalised (H high, ascertainment NARROW) report far
higher case fatality than regions that also record mild illness (H low,
ascertainment BROAD). The strongest objection to that claim needs no
sophistication at all — territories differ. Amazonian health regions and
metropolitan Southeastern ones differ in serovar ecology, exposure intensity,
distance to care, admission thresholds, laboratory availability, coding
culture and age structure, and every one of those differences is a candidate
explanation for a between-territory gradient that has nothing to do with how
far surveillance reaches into the clinical spectrum.

A region fixed effect removes **every time-invariant regional characteristic
by construction**, named or unnamed, measured or unmeasured. What is left is
the question this script asks: when one health region's own ascertainment
moves from one year to the next, does its own reported case fatality move with
it, in the predicted direction, and by how much?

This is deliberately a different framework from the Bayesian spatial model
fitted elsewhere in this study. That model borrows strength across neighbours
and puts structure on space; this one throws all cross-region information away
and uses only the time dimension. They share the panel and the construct, and
nothing else — so a common failure mode would have to live in the panel
itself.

**What would falsify the claim tested here.** The within-region odds ratio for
a 10-percentage-point rise in H is at or below 1.0; or it is far weaker than
the pooled cross-sectional estimate on the same sample *and* the gap survives
the measurement-error correction in §T1; or lead values of H predict case
fatality as strongly as contemporaneous ones, which would say the association
is carried by something slow-moving in the region rather than by ascertainment.
Any of those materially narrows what the paper may claim, and is reported as
the finding rather than softened.

**Threat ledger** (each row has a directed check below, not a free-floating
robustness sweep):

T1  H is measured on a finite number of cases, so its year-to-year wobble is
    partly binomial noise. Classical measurement error attenuates the within
    estimate specifically (region means average the noise away, deviations do
    not), which would manufacture exactly the within<between pattern that
    would otherwise damage the paper. Check: reliability ratio and an
    errors-in-variables correction, plus the estimate across four minimum
    denominators.
T2  The association could be arithmetic rather than epidemiological: deaths
    are concentrated among hospitalised cases, so a year with a higher
    hospitalised share mechanically has a higher death share. Check: refit
    within the hospitalised stratum alone and the non-hospitalised stratum
    alone. Under a pure composition (denominator) mechanism the stratum-
    specific case fatalities do not move with H; under a "sicker patients"
    mechanism they do. Note that composition is the paper's thesis, not a
    threat to it — this check identifies which mechanism is operating.
T3  Outcome recording is incomplete and varies within a region over time.
    Unrecorded outcomes deflate reported case fatality; if completeness moves
    with H, the association is a recording artefact. Check: adjust for
    outcome and hospitalisation-field completeness.
T4  Case mix moves within a region over time — an ageing case series would
    raise case fatality on its own. Check: adjust for the share aged 60+, the
    male share, the laboratory-confirmed share and the severe-phenotype share.
T5  A shared secular trend (national case definitions, SINAN form revisions)
    could drive H and case fatality together. Check: replace the linear year
    term with year fixed effects, and run the first-difference view in (c),
    which is invariant to any region-specific level.
T6  Reverse or spurious timing. Check: lag 0 / +1 / -1 in (d).

Outputs to ``data/results/within_region/``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci  # noqa: E402
from brepi.config import PATHS  # noqa: E402

OUT = PATHS.results / "within_region"
PANEL = PATHS.results / "analysis_panel"

#: Minimum count required in BOTH denominators for a region-year to enter the
#: estimation sample: ``outcome_known`` (the binomial denominator of the
#: outcome) and ``hosp_known`` (the denominator of the exposure). The median
#: region-year in the full panel has three confirmed cases; at that size H is
#: a coin flip and its year-to-year movement is almost entirely noise, which
#: attenuates a within-region estimate towards the null and would understate
#: the very thing this script is measuring. Ten is the primary threshold
#: because it is the smallest denominator at which the binomial standard error
#: of H (at most 0.16) is comfortably below the observed median within-region
#: standard deviation of H, i.e. the signal exceeds the noise. The whole grid
#: is reported so the reader can see the threshold is not doing the work.
MIN_N_PRIMARY = 10
MIN_N_GRID = (5, 10, 20, 30)

#: A region needs at least two eligible years to contribute anything at all to
#: a within estimate; with one year the fixed effect fits it exactly and it
#: drops out of the score for H. Made explicit so the pooled comparison in (b)
#: runs on the identical sample rather than on a larger one.
MIN_YEARS_FE = 2

#: First differences need a run of years to have a meaningful rank
#: correlation. Eight eligible years gives at least a handful of consecutive
#: pairs per region.
MIN_YEARS_FD = 8

#: The exposure is reported per 10 percentage points of H, which is roughly
#: one within-region standard deviation and is a movement that actually
#: happens in these data. A per-unit (0 to 1) odds ratio would be an
#: extrapolation across the whole range in a single number.
H_SCALE = 10.0

#: Year is centred so the intercept is interpretable and the FE design matrix
#: is better conditioned. Midpoint of 2007-2025.
YEAR_CENTRE = 2016

BOOT = 2000
RNG_SEED = 20260814


# --------------------------------------------------------------------------
# estimation helpers
# --------------------------------------------------------------------------

def _fit(
    df: pd.DataFrame,
    terms: list[str],
    *,
    region_fe: bool,
    year_fe: bool = False,
    deaths: str = "deaths",
    denom: str = "outcome_known",
) -> dict:
    """Binomial GLM, logit link, cluster-robust by health region.

    ``terms`` are continuous columns of ``df``. The endogenous variable is the
    grouped two-column (successes, failures) form, so each region-year carries
    its own denominator rather than being collapsed to a proportion — an
    unweighted regression on proportions would let a region-year with ten
    cases speak as loudly as one with nine hundred.
    """
    d = df[df[denom] > 0].copy()
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    for t in terms:
        X[t] = d[t].astype(float).to_numpy()
    if year_fe:
        yd = pd.get_dummies(d["year"].astype(int), prefix="yr", drop_first=True)
        X = pd.concat([X, yd.astype(float)], axis=1)
    if region_fe:
        rd = pd.get_dummies(d["health_region_code"].astype(str),
                            prefix="rg", drop_first=True)
        X = pd.concat([X, rd.astype(float)], axis=1)

    y = np.column_stack([
        d[deaths].to_numpy(float),
        (d[denom] - d[deaths]).to_numpy(float),
    ])
    groups = d["health_region_code"].astype(str).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X.to_numpy(float), family=sm.families.Binomial())
        res = model.fit(
            maxiter=300,
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
    names = list(X.columns)
    out = {
        "n_obs": int(d.shape[0]),
        "n_regions": int(pd.unique(groups).size),
        "n_deaths": int(d[deaths].sum()),
        "n_denominator": int(d[denom].sum()),
        "converged": bool(res.converged),
        "params": {},
    }
    ci = res.conf_int()
    for t in terms:
        i = names.index(t)
        out["params"][t] = {
            "beta": float(res.params[i]),
            "se": float(res.bse[i]),
            "or": float(np.exp(res.params[i])),
            "or_lo": float(np.exp(ci[i, 0])),
            "or_hi": float(np.exp(ci[i, 1])),
            "p": float(res.pvalues[i]),
        }
    out["_res"] = res
    out["_names"] = names
    return out


def _wald_difference(res, names: list[str], a: str, b: str) -> dict:
    """Test coefficient(a) - coefficient(b) = 0 under the cluster covariance.

    This is what turns "the between estimate is bigger than the within one"
    from an eyeball comparison into a statement with a p-value: the two
    coefficients come from the same fit and their covariance is known.
    """
    ia, ib = names.index(a), names.index(b)
    c = np.zeros(len(names))
    c[ia], c[ib] = 1.0, -1.0
    diff = float(c @ res.params)
    var = float(c @ res.cov_params() @ c)
    se = float(np.sqrt(var))
    z = diff / se
    return {
        "difference_in_log_odds": diff,
        "se": se,
        "ratio_of_odds_ratios": float(np.exp(diff)),
        "ratio_lo": float(np.exp(diff - 1.96 * se)),
        "ratio_hi": float(np.exp(diff + 1.96 * se)),
        "z": float(z),
        "p": float(2 * stats.norm.sf(abs(z))),
    }


def _eligible(panel: pl.DataFrame, min_n: int) -> pl.DataFrame:
    """Region-years with adequate denominators, keeping only regions that can
    contribute within-region information and are not perfectly separated.

    A region whose eligible years contain no death at all, or nothing but
    deaths, has a fixed effect at +/-infinity: it is fitted exactly and
    contributes nothing to the slope on H, but it does break the IRLS. Those
    regions are dropped explicitly and counted, rather than being allowed to
    fail silently.
    """
    q = panel.filter(
        (pl.col("outcome_known") >= min_n)
        & (pl.col("hosp_known") >= min_n)
    )
    g = q.group_by("health_region_code").agg(
        pl.len().alias("years_eligible"),
        pl.col("deaths").sum().alias("_d"),
        pl.col("outcome_known").sum().alias("_ok"),
    )
    keep = g.filter(
        (pl.col("years_eligible") >= MIN_YEARS_FE)
        & (pl.col("_d") > 0)
        & (pl.col("_d") < pl.col("_ok"))
    )
    return q.join(keep.select("health_region_code", "years_eligible"),
                  on="health_region_code", how="inner")


def _prepare(q: pl.DataFrame) -> pd.DataFrame:
    """Add the modelling columns: scaled exposure, centred year, region-mean
    exposure and its deviation (the Mundlak pair)."""
    # The region mean of H is formed by summing numerators and denominators
    # first, never by averaging the yearly ratios.
    means = q.group_by("health_region_code").agg(
        (pl.col("hospitalised").sum() / pl.col("hosp_known").sum()).alias("H_region")
    )
    q = q.join(means, on="health_region_code", how="left").with_columns(
        (pl.col("H") * H_SCALE).alias("Hx"),
        (pl.col("H_region") * H_SCALE).alias("Hx_region"),
        ((pl.col("H") - pl.col("H_region")) * H_SCALE).alias("Hx_dev"),
        (pl.col("year") - YEAR_CENTRE).cast(pl.Float64).alias("yearc"),
    )
    return q.to_pandas()


# --------------------------------------------------------------------------
# (a) within-region estimate, (b) pooled contrast
# --------------------------------------------------------------------------

def within_and_between(panel: pl.DataFrame) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    detail: dict = {}
    for min_n in MIN_N_GRID:
        q = _eligible(panel, min_n)
        if q.height == 0:
            continue
        d = _prepare(q)

        fe = _fit(d, ["Hx", "yearc"], region_fe=True)
        pooled = _fit(d, ["Hx", "yearc"], region_fe=False)
        # Mundlak / correlated random effects: splitting H into its region mean
        # and the within-region deviation puts the between and within
        # components in ONE fit, so their difference has a covariance and can
        # be tested rather than merely compared.
        mund = _fit(d, ["Hx_dev", "Hx_region", "yearc"], region_fe=False)
        contrast = _wald_difference(mund["_res"], mund["_names"],
                                    "Hx_region", "Hx_dev")

        for label, fit, term in (
            ("within (region FE)", fe, "Hx"),
            ("pooled (no FE)", pooled, "Hx"),
            ("Mundlak: within component", mund, "Hx_dev"),
            ("Mundlak: between component", mund, "Hx_region"),
        ):
            p = fit["params"][term]
            rows.append({
                "min_denominator": min_n,
                "model": label,
                "or_per_10pp_H": p["or"],
                "or_lo": p["or_lo"],
                "or_hi": p["or_hi"],
                "p_value": p["p"],
                "region_years": fit["n_obs"],
                "regions": fit["n_regions"],
                "deaths": fit["n_deaths"],
                "outcome_known": fit["n_denominator"],
                "converged": fit["converged"],
            })

        detail[min_n] = {
            "within": {k: v for k, v in fe.items() if not k.startswith("_")},
            "pooled": {k: v for k, v in pooled.items() if not k.startswith("_")},
            "mundlak_between_vs_within": contrast,
            "within_share_of_pooled_log_odds": (
                fe["params"]["Hx"]["beta"] / pooled["params"]["Hx"]["beta"]
                if pooled["params"]["Hx"]["beta"] != 0 else float("nan")
            ),
        }
    return rows, detail


def marginal_view(panel: pl.DataFrame, min_n: int) -> dict:
    """Translate the odds ratio into case fatality on the percentage scale.

    An odds ratio is not a magnitude statement a reader can use. This holds
    the fitted region effects at the sample and moves every region-year's H to
    two fixed values, reporting the mean predicted case fatality at each — a
    within-region contrast expressed in percentage points.
    """
    q = _eligible(panel, min_n)
    d = _prepare(q)
    fit = _fit(d, ["Hx", "yearc"], region_fe=True)
    res, names = fit["_res"], fit["_names"]
    dd = d[d["outcome_known"] > 0].copy()

    X = pd.DataFrame(index=dd.index)
    X["const"] = 1.0
    X["Hx"] = dd["Hx"].astype(float).to_numpy()
    X["yearc"] = dd["yearc"].astype(float).to_numpy()
    rd = pd.get_dummies(dd["health_region_code"].astype(str),
                        prefix="rg", drop_first=True).astype(float)
    X = pd.concat([X, rd], axis=1)
    assert list(X.columns) == names

    w = dd["outcome_known"].to_numpy(float)
    out = {}
    for h in (0.50, 0.70, 0.90):
        Xh = X.to_numpy(float).copy()
        Xh[:, names.index("Hx")] = h * H_SCALE
        eta = Xh @ res.params
        p = 1.0 / (1.0 + np.exp(-eta))
        out[f"predicted_cfr_at_H_{int(h * 100)}"] = float(np.average(p, weights=w))
    out["percentage_point_difference_H90_minus_H50"] = (
        out["predicted_cfr_at_H_90"] - out["predicted_cfr_at_H_50"]
    )
    out["observed_mean_H"] = float(dd["hospitalised"].sum() / dd["hosp_known"].sum())
    return out


# --------------------------------------------------------------------------
# T1: how much of H's within-region movement is binomial noise?
# --------------------------------------------------------------------------

def reliability(panel: pl.DataFrame, min_n: int) -> dict:
    """Reliability ratio of the within-region deviation of H, and the implied
    errors-in-variables correction.

    The observed within-region variance of H is the true within variance plus
    binomial sampling variance. The sampling part is computed from the region's
    own mean H and each year's ``hosp_known`` — using the region mean rather
    than the realised year value keeps the noise estimate from being correlated
    with the deviation whose variance it is being subtracted from. The
    correction ``beta / lambda`` is the classical linear one applied to a logit
    coefficient, so it is an approximation and is labelled as such; it bounds
    the size of the attenuation rather than pinning it down.
    """
    q = _eligible(panel, min_n)
    d = _prepare(q)
    w = d["outcome_known"].to_numpy(float)
    dev = d["Hx_dev"].to_numpy(float) / H_SCALE
    hbar = d["H_region"].to_numpy(float)
    nk = d["hosp_known"].to_numpy(float)

    var_obs = float(np.average(dev ** 2, weights=w))
    var_noise = float(np.average(hbar * (1 - hbar) / nk, weights=w))
    lam = (var_obs - var_noise) / var_obs

    fit = _fit(d, ["Hx", "yearc"], region_fe=True)
    b = fit["params"]["Hx"]["beta"]
    se = fit["params"]["Hx"]["se"]
    return {
        "min_denominator": min_n,
        "within_variance_of_H_observed": var_obs,
        "within_variance_of_H_binomial_noise": var_noise,
        "reliability_ratio_lambda": float(lam),
        "observed_or_per_10pp": float(np.exp(b)),
        "attenuation_corrected_or_per_10pp": float(np.exp(b / lam)),
        "corrected_or_lo": float(np.exp((b - 1.96 * se) / lam)),
        "corrected_or_hi": float(np.exp((b + 1.96 * se) / lam)),
    }


# --------------------------------------------------------------------------
# T2-T5: directed adjustment checks, all on the primary sample
# --------------------------------------------------------------------------

def threat_checks(panel: pl.DataFrame, min_n: int) -> list[dict]:
    q = _eligible(panel, min_n)
    d = _prepare(q)
    rows: list[dict] = []

    def add(threat: str, spec: str, fit: dict, term: str = "Hx") -> None:
        p = fit["params"][term]
        rows.append({
            "threat": threat, "specification": spec,
            "or_per_10pp_H": p["or"], "or_lo": p["or_lo"], "or_hi": p["or_hi"],
            "p_value": p["p"], "region_years": fit["n_obs"],
            "regions": fit["n_regions"], "deaths": fit["n_deaths"],
            "converged": fit["converged"],
        })

    base = _fit(d, ["Hx", "yearc"], region_fe=True)
    add("-", "region FE + linear year (primary)", base)

    # T5: a shared secular trend. Year fixed effects absorb any national-level
    # movement of any shape, at the cost of the between-year identification
    # that a linear term borrows.
    add("T5 shared secular trend", "region FE + year FE",
        _fit(d, ["Hx"], region_fe=True, year_fe=True))

    # T3: recording completeness moving with H inside a region.
    dc = d.dropna(subset=["outcome_completeness", "hosp_completeness"])
    add("T3 recording completeness",
        "region FE + year + outcome & hospitalisation completeness",
        _fit(dc, ["Hx", "yearc", "outcome_completeness", "hosp_completeness"],
             region_fe=True))

    # T4: case mix drifting inside a region. The two blocks are kept apart on
    # purpose. Age and sex are genuine confounders: an ageing case series would
    # raise case fatality whatever surveillance did. The laboratory-confirmed
    # and severe-phenotype shares are NOT confounders — they are downstream of
    # ascertainment breadth itself (a region that starts finding mild illness
    # mechanically records a lower severe share), so conditioning on them
    # removes part of the very pathway under study. The over-adjusted fit is
    # reported as a lower bound, labelled as such, not as the preferred
    # estimate.
    demo = ["share_age60", "share_male"]
    mediators = ["share_lab", "share_severe"]
    dm = d.dropna(subset=demo + mediators)
    add("T4 case mix (confounders)", "region FE + year + age60 & male shares",
        _fit(dm, [*["Hx", "yearc"], *demo], region_fe=True))
    add("T4 case mix (OVER-adjusted, lower bound)",
        "region FE + year + age/sex + lab & severe shares (mediators)",
        _fit(dm, [*["Hx", "yearc"], *demo, *mediators], region_fe=True))
    add("T4 case mix (sample-matched baseline)",
        "region FE + year, restricted to the case-mix sample",
        _fit(dm, ["Hx", "yearc"], region_fe=True))

    # T2: is it arithmetic? Refit the same within-region model on the two
    # strata that make up the total, using the subset of cases with BOTH the
    # outcome and the hospitalisation field valid so the strata sum exactly.
    dh = d[d["hk_hosp"] >= min_n]
    add("T2 mechanism", "hospitalised stratum only (d_hosp / hk_hosp)",
        _fit(dh, ["Hx", "yearc"], region_fe=True,
             deaths="d_hosp", denom="hk_hosp"))
    dn = d[d["hk_nonhosp"] >= min_n]
    if dn.shape[0] > 0 and dn["d_nonhosp"].sum() > 0:
        add("T2 mechanism", "non-hospitalised stratum only (d_nonhosp / hk_nonhosp)",
            _fit(dn, ["Hx", "yearc"], region_fe=True,
                 deaths="d_nonhosp", denom="hk_nonhosp"))
    return rows


def composition_counterfactual(panel: pl.DataFrame, min_n: int) -> dict:
    """How much of the within-region movement is pure denominator arithmetic?

    Reported case fatality on the subset with both fields valid is exactly

        cfr = H * cfr_hospitalised + (1 - H) * cfr_non-hospitalised

    so if a region's two stratum-specific case fatalities were frozen in time,
    changing H alone would still move reported case fatality. This builds that
    counterfactual explicitly: each region's stratum-specific case fatalities
    are held at their whole-period values, only the yearly mix H varies, and
    the identical fixed-effect model is refitted on the synthetic deaths. The
    resulting odds ratio is what pure composition predicts.

    Comparing it with the observed one settles what the association IS rather
    than whether it exists. If they match, the within-region movement of
    reported case fatality is a denominator effect -- which is the study's
    thesis, not a threat to it. If the observed one is materially larger,
    something beyond composition (genuinely sicker patients in narrow-
    ascertainment years) is also operating and the thesis is incomplete.
    """
    q = _eligible(panel, min_n).filter(
        (pl.col("hk_hosp") + pl.col("hk_nonhosp")) >= min_n
    )
    # Stratum-specific case fatality per region, over the whole period:
    # numerators and denominators summed first, then the ratio.
    strat = q.group_by("health_region_code").agg(
        pl.col(["d_hosp", "hk_hosp", "d_nonhosp", "hk_nonhosp"]).sum()
    ).with_columns(
        (pl.col("d_hosp") / pl.col("hk_hosp")).alias("cfr_hosp_region"),
        pl.when(pl.col("hk_nonhosp") > 0)
          .then(pl.col("d_nonhosp") / pl.col("hk_nonhosp"))
          .otherwise(0.0).alias("cfr_nonhosp_region"),
    ).select("health_region_code", "cfr_hosp_region", "cfr_nonhosp_region")

    d = _prepare(
        q.join(strat, on="health_region_code", how="left").with_columns(
            (pl.col("hk_hosp") + pl.col("hk_nonhosp")).alias("bk_known"),
            (pl.col("d_hosp") + pl.col("d_nonhosp")).alias("bk_deaths"),
        ).with_columns(
            (pl.col("cfr_hosp_region") * pl.col("hk_hosp")
             + pl.col("cfr_nonhosp_region") * pl.col("hk_nonhosp")
             ).alias("bk_deaths_synth"),
        )
    )
    # The exposure for this comparison is H on the both-fields-valid subset, so
    # the identity above holds exactly rather than approximately.
    d["Hx"] = (d["hk_hosp"] / d["bk_known"]).astype(float) * H_SCALE

    obs = _fit(d, ["Hx", "yearc"], region_fe=True,
               deaths="bk_deaths", denom="bk_known")
    synth = _fit(d, ["Hx", "yearc"], region_fe=True,
                 deaths="bk_deaths_synth", denom="bk_known")
    o = obs["params"]["Hx"]
    s = synth["params"]["Hx"]
    return {
        "min_denominator": min_n,
        "region_years": obs["n_obs"],
        "regions": obs["n_regions"],
        "deaths": obs["n_deaths"],
        "observed_or_per_10pp": o["or"],
        "observed_lo": o["or_lo"], "observed_hi": o["or_hi"],
        "composition_only_or_per_10pp": s["or"],
        "composition_only_lo": s["or_lo"], "composition_only_hi": s["or_hi"],
        "share_of_observed_log_odds_explained_by_composition":
            s["beta"] / o["beta"] if o["beta"] != 0 else float("nan"),
        "national_cfr_hospitalised": float(
            q["d_hosp"].sum() / q["hk_hosp"].sum()),
        "national_cfr_non_hospitalised": float(
            q["d_nonhosp"].sum() / q["hk_nonhosp"].sum()),
    }


# --------------------------------------------------------------------------
# (c) first differences
# --------------------------------------------------------------------------

def first_differences(panel: pl.DataFrame, min_n: int, rng: np.random.Generator) -> dict:
    """Year-on-year change in H against year-on-year change in case fatality.

    This is the most assumption-free view available: a difference removes the
    region level exactly, with no link function and no distributional claim.
    Only consecutive calendar years are differenced — a gap year would make the
    difference span an unknown interval.
    """
    q = _eligible(panel, min_n).filter(pl.col("years_eligible") >= MIN_YEARS_FD)
    d = q.sort(["health_region_code", "year"]).with_columns(
        pl.col("year").shift(1).over("health_region_code").alias("_yprev"),
        pl.col("H").shift(1).over("health_region_code").alias("_Hprev"),
        pl.col("cfr").shift(1).over("health_region_code").alias("_cprev"),
    ).filter(pl.col("_yprev") == pl.col("year") - 1).with_columns(
        (pl.col("H") - pl.col("_Hprev")).alias("dH"),
        (pl.col("cfr") - pl.col("_cprev")).alias("dCFR"),
    )
    dh = d["dH"].to_numpy()
    dc = d["dCFR"].to_numpy()
    reg = d["health_region_code"].to_numpy().astype(str)

    pooled_rho = float(stats.spearmanr(dh, dc).statistic)

    # Cluster bootstrap by region: resample whole regions, not pairs, because
    # the pairs inside a region are not independent of one another.
    uniq = np.unique(reg)
    idx = {r: np.flatnonzero(reg == r) for r in uniq}
    boots = []
    for _ in range(BOOT):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx[r] for r in pick])
        if np.unique(dh[sel]).size < 3:
            continue
        boots.append(stats.spearmanr(dh[sel], dc[sel]).statistic)
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    per_region = []
    for r in uniq:
        i = idx[r]
        if i.size < 4 or np.unique(dh[i]).size < 2 or np.unique(dc[i]).size < 2:
            continue
        rho = stats.spearmanr(dh[i], dc[i]).statistic
        if np.isfinite(rho):
            per_region.append({"health_region_code": r, "n_pairs": int(i.size),
                               "rho": float(rho)})
    rhos = np.array([p["rho"] for p in per_region])
    # Sign test on the per-region rhos: under no association the sign of each
    # region's rho is a fair coin, so the count of positives is Binomial(n, .5).
    n_pos = int((rhos > 0).sum())
    sign_p = float(stats.binomtest(n_pos, len(rhos), 0.5).pvalue)

    return {
        "min_denominator": min_n,
        "min_years": MIN_YEARS_FD,
        "regions": int(uniq.size),
        "consecutive_year_pairs": int(dh.size),
        "pooled_spearman_rho": pooled_rho,
        "pooled_rho_lo": float(lo),
        "pooled_rho_hi": float(hi),
        "bootstrap_replicates": int(boots.size),
        "regions_with_rho_computed": len(per_region),
        "regions_with_positive_rho": n_pos,
        "share_positive": n_pos / len(rhos),
        "sign_test_p": sign_p,
        "median_within_region_rho": float(np.median(rhos)),
        "iqr_within_region_rho": [float(np.percentile(rhos, 25)),
                                  float(np.percentile(rhos, 75))],
        "_per_region": per_region,
        "_pairs": d.select("health_region_code", "year", "dH", "dCFR",
                           "outcome_known", "hosp_known"),
    }


# --------------------------------------------------------------------------
# (d) lag / lead
# --------------------------------------------------------------------------

def lag_check(panel: pl.DataFrame, min_n: int) -> tuple[list[dict], dict]:
    """Does H predict case fatality in its own year, or equally well a year
    early and a year late?

    Ascertainment breadth in year t determines which cases enter year t's
    denominator. It has no route by which it should predict year t-1's
    reported case fatality. If the lead coefficient matches the contemporaneous
    one, the shared driver is something that persists across years in the
    region -- exactly the slow-moving regional characteristic the fixed effect
    was meant to remove but which a trend would not.

    All three models are fitted on the IDENTICAL set of region-years (those
    with an adequately measured H in t-1, t and t+1), because comparing
    coefficients estimated on different samples would confound the timing
    question with a sample composition question.
    """
    q = _eligible(panel, min_n)
    keys = q.select("health_region_code", "year")

    # Neighbouring-year H is taken from the FULL panel but only where that
    # year's own hosp_known clears the same bar, so the lag and lead exposures
    # are measured to the same standard as the contemporaneous one.
    nb = panel.filter(pl.col("hosp_known") >= min_n).select(
        "health_region_code", "year", pl.col("H").alias("_H")
    )
    d = (
        q.join(nb.with_columns((pl.col("year") + 1).alias("year"))
                 .rename({"_H": "H_lag1"}),
               on=["health_region_code", "year"], how="inner")
         .join(nb.with_columns((pl.col("year") - 1).alias("year"))
                 .rename({"_H": "H_lead1"}),
               on=["health_region_code", "year"], how="inner")
    )
    # Re-impose the fixed-effect eligibility on the reduced sample.
    g = d.group_by("health_region_code").agg(
        pl.len().alias("_t"), pl.col("deaths").sum().alias("_d"),
        pl.col("outcome_known").sum().alias("_ok"))
    keep = g.filter((pl.col("_t") >= MIN_YEARS_FE) & (pl.col("_d") > 0)
                    & (pl.col("_d") < pl.col("_ok")))
    d = d.join(keep.select("health_region_code"), on="health_region_code",
               how="inner")

    d = _prepare(d).assign(
        Hx_lag1=lambda x: x["H_lag1"] * H_SCALE,
        Hx_lead1=lambda x: x["H_lead1"] * H_SCALE,
    )

    rows = []
    for label, term in (("lag 0 (contemporaneous)", "Hx"),
                        ("lag +1 (H one year earlier)", "Hx_lag1"),
                        ("lag -1 (H one year later, a lead)", "Hx_lead1")):
        fit = _fit(d, [term, "yearc"], region_fe=True)
        p = fit["params"][term]
        rows.append({
            "term": label, "model": "one at a time",
            "or_per_10pp_H": p["or"], "or_lo": p["or_lo"], "or_hi": p["or_hi"],
            "p_value": p["p"], "region_years": fit["n_obs"],
            "regions": fit["n_regions"], "deaths": fit["n_deaths"],
        })

    joint = _fit(d, ["Hx", "Hx_lag1", "Hx_lead1", "yearc"], region_fe=True)
    for label, term in (("lag 0 (contemporaneous)", "Hx"),
                        ("lag +1 (H one year earlier)", "Hx_lag1"),
                        ("lag -1 (H one year later, a lead)", "Hx_lead1")):
        p = joint["params"][term]
        rows.append({
            "term": label, "model": "all three jointly",
            "or_per_10pp_H": p["or"], "or_lo": p["or_lo"], "or_hi": p["or_hi"],
            "p_value": p["p"], "region_years": joint["n_obs"],
            "regions": joint["n_regions"], "deaths": joint["n_deaths"],
        })

    summary = {
        "sample_region_years": joint["n_obs"],
        "sample_regions": joint["n_regions"],
        "contemporaneous_vs_lead": _wald_difference(
            joint["_res"], joint["_names"], "Hx", "Hx_lead1"),
        "contemporaneous_vs_lag": _wald_difference(
            joint["_res"], joint["_names"], "Hx", "Hx_lag1"),
        "autocorrelation_of_H_lag1": float(
            np.corrcoef(d["Hx"], d["Hx_lag1"])[0, 1]),
    }
    return rows, summary


# --------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    panel = pl.read_parquet(PANEL / "region_year_panel.parquet")

    # Plausibility gate, before anything is estimated: the estimation sample
    # must reproduce a national case fatality in the range the whole study
    # reports (about 10%), and H must lie inside (0, 1).
    prim = _eligible(panel, MIN_N_PRIMARY)
    cfr_p, cfr_lo, cfr_hi = binom_ci(int(prim["deaths"].sum()),
                                     int(prim["outcome_known"].sum()))
    h_p, h_lo, h_hi = binom_ci(int(prim["hospitalised"].sum()),
                               int(prim["hosp_known"].sum()))
    assert 0.03 < float(cfr_p) < 0.25, f"implausible case fatality {cfr_p}"
    assert 0.0 < float(h_p) < 1.0, f"H outside (0,1): {h_p}"

    grid, detail = within_and_between(panel)
    tbl = pl.DataFrame(grid)
    tbl.write_csv(OUT / "within_vs_between_by_threshold.csv")

    marg = marginal_view(panel, MIN_N_PRIMARY)
    rel = [reliability(panel, m) for m in MIN_N_GRID]
    pl.DataFrame(rel).write_csv(OUT / "measurement_error_correction.csv")

    thr = threat_checks(panel, MIN_N_PRIMARY)
    pl.DataFrame(thr).write_csv(OUT / "threat_ledger_checks.csv")

    comp = composition_counterfactual(panel, MIN_N_PRIMARY)
    pl.DataFrame([comp]).write_csv(OUT / "composition_counterfactual.csv")

    fd = first_differences(panel, MIN_N_PRIMARY, rng)
    pl.DataFrame(fd.pop("_per_region")).write_csv(OUT / "first_difference_by_region.csv")
    fd.pop("_pairs").write_csv(OUT / "first_difference_pairs.csv")
    # A looser threshold buys more regions at the price of noisier differences;
    # both are reported so the reader can see which way that trade moves the
    # correlation.
    fd5 = first_differences(panel, 5, rng)
    fd5.pop("_per_region"); fd5.pop("_pairs")

    lagrows, lagsum = lag_check(panel, MIN_N_PRIMARY)
    pl.DataFrame(lagrows).write_csv(OUT / "lag_lead.csv")

    p = detail[MIN_N_PRIMARY]
    between_or = next(r["or_per_10pp_H"] for r in grid
                      if r["min_denominator"] == MIN_N_PRIMARY
                      and r["model"] == "Mundlak: between component")
    report = {
        "question": (
            "When ascertainment breadth changes within a single health region "
            "over time, does reported case fatality move with it?"
        ),
        "exposure": "H = hospitalised / hosp_known, per 10 percentage points; "
                    "higher H = narrower ascertainment",
        "measure": "odds ratio for leptospirosis death among confirmed cases "
                   "with a valid outcome field, binomial GLM with logit link, "
                   "health-region fixed effects, cluster-robust by region",
        "primary_minimum_denominator": MIN_N_PRIMARY,
        "estimation_sample": {
            "region_years": p["within"]["n_obs"],
            "regions": p["within"]["n_regions"],
            "deaths": p["within"]["n_deaths"],
            "outcome_known": p["within"]["n_denominator"],
            "case_fatality": float(cfr_p),
            "case_fatality_ci": [float(cfr_lo), float(cfr_hi)],
            "H": float(h_p), "H_ci": [float(h_lo), float(h_hi)],
        },
        "a_within_region": p["within"]["params"]["Hx"],
        "b_pooled": p["pooled"]["params"]["Hx"],
        "b_between_vs_within": p["mundlak_between_vs_within"],
        "b_within_share_of_pooled_log_odds": p["within_share_of_pooled_log_odds"],
        # The same share once the within estimate is corrected for the
        # binomial noise in H (T1), which attenuates the within component and
        # not the between one. Recorded here rather than recomputed in prose,
        # so the manuscript and this file cannot drift apart.
        "b_within_share_of_pooled_log_odds_noise_corrected": float(
            np.log(rel[MIN_N_GRID.index(MIN_N_PRIMARY)]
                   ["attenuation_corrected_or_per_10pp"])
            / p["pooled"]["params"]["Hx"]["beta"]
        ),
        "b_within_share_of_between_log_odds_noise_corrected": float(
            np.log(rel[MIN_N_GRID.index(MIN_N_PRIMARY)]
                   ["attenuation_corrected_or_per_10pp"])
            / np.log(between_or)
        ),
        "a_marginal_percentage_scale": marg,
        "T1_measurement_error": rel,
        "T2_composition_counterfactual": comp,
        "c_first_differences_primary": fd,
        "c_first_differences_min5": fd5,
        "d_lag_lead": lagsum,
        "sensitivity_by_threshold": {
            str(m): {
                "within_or": detail[m]["within"]["params"]["Hx"]["or"],
                "within_lo": detail[m]["within"]["params"]["Hx"]["or_lo"],
                "within_hi": detail[m]["within"]["params"]["Hx"]["or_hi"],
                "pooled_or": detail[m]["pooled"]["params"]["Hx"]["or"],
                "pooled_lo": detail[m]["pooled"]["params"]["Hx"]["or_lo"],
                "pooled_hi": detail[m]["pooled"]["params"]["Hx"]["or_hi"],
                "regions": detail[m]["within"]["n_regions"],
                "region_years": detail[m]["within"]["n_obs"],
            }
            for m in detail
        },
    }
    (OUT / "within_region_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(tbl.to_pandas().to_string(index=False))
    print()
    print(pl.DataFrame(thr).to_pandas().to_string(index=False))
    print()
    print(pl.DataFrame(lagrows).to_pandas().to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("sensitivity_by_threshold",)}, indent=2))


if __name__ == "__main__":
    main()
