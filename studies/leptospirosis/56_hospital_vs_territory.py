"""Is the surveillance-depth gradient a property of TERRITORIES or of HOSPITALS?

Open threads T-10 and the loose end at the foot of T-12. Every result in the
depth chain bands *health regions*. But SIH names the treating hospital
(``CNES``, 2,055 distinct) and the municipality of treatment (``MUNIC_MOV``)
alongside the municipality of residence (``MUNIC_RES``). If the severity
gradient is really a hospital property -- a handful of referral centres
absorbing the sickest patients -- then "territory surveillance depth" is the
wrong unit of analysis and the thesis needs restating.

Three questions, in order:

1. **Where does the variance live?** In-hospital fatality and ICU use are
   decomposed into a between-hospital and a between-health-region component
   with crossed and nested random-intercept logistic models (lme4, via
   ``56_variance_components.R``). A model-free method-of-moments decomposition
   that subtracts binomial sampling noise is run alongside, so the answer does
   not rest on one optimiser converging.

2. **The decisive test.** Hold the treating hospital fixed. Within one hospital
   admitting patients from more than one depth band, does the patient's *home*
   band still predict death and ICU use? If yes, depth is carried by the
   territory the patient came from; if no, by the hospital they reached.
   Identification comes only from patients treated outside their own health
   region, so the power available is stated up front rather than discovered
   afterwards.

3. **The non-monotone admission rate.** SIH A27 admissions per 100,000
   person-years run 1.71 / 1.85 / 2.00 / 1.09 / 0.86 across the bands -- rising
   across the first three, then falling off a cliff -- while SINAN incidence
   over the same window falls monotonically. The offered candidates are urban /
   rural mixing, hospital supply, and referral flow. The arithmetic is tested
   before any of them.

Window is 2008-2024 (SIH coverage), dated by admission. Geography is by
residence throughout except where treatment geography is explicitly the point.
Bands are the case-weighted quintiles of hospitalisation share among confirmed
cases defined in ``40_ascertainment_depth.py`` and reused in 52/53/54; the
banding code is copied here verbatim rather than re-derived.

Outputs to ``data/results/hospital_vs_territory/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "hospital_vs_territory"
RSCRIPT = r"C:/Program Files/R/R-4.4.1/bin/Rscript.exe"
R_HELPER = Path(__file__).with_name("56_variance_components.R")

MIN_CASES = 30           # health region entry threshold, as in 40/52/53/54
YEARS = (2008, 2024)     # SIH coverage window
ATLAS_YEARS = 19         # person-years in the atlas span 2007-2025

#: Volume threshold for a hospital to enter the within-hospital test. Fifty A27
#: admissions at the national in-hospital fatality of 6.2% is about three
#: expected deaths -- the point below which a stratum contributes essentially
#: nothing to a conditional likelihood while still costing a parameter. The
#: test is repeated at 30 and 100 so the reader can see it is not a knife-edge.
MIN_HOSP_VOLUME = 50
VOLUME_SENSITIVITY = (30, 50, 100)


# --------------------------------------------------------------------------
# banding -- copied from 40_ascertainment_depth.py, not re-derived
# --------------------------------------------------------------------------
def bands() -> pl.DataFrame:
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.filter(pl.col("cases") >= MIN_CASES).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share")
    ).sort("hosp_share")
    hr = hr.with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    )
    return hr.select("health_region_code", "quintile", "hosp_share")


def survivor_depth() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hospitalisation share computed among survivors only.

    Threat to the decisive test: 97.0% of fatal confirmed cases are recorded as
    hospitalised against 68.8% of cured ones, so a territory's hospitalisation
    share is mechanically pushed up by its own deaths. An association between
    that share and in-hospital fatality is then partly arithmetic. Recomputing
    the index after removing every case that died breaks the link -- at the cost
    of dropping the 8.1% of confirmed cases with unknown outcome, which are
    dropped from the denominator rather than assumed to have survived.
    """
    ll = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter((pl.col("classi_fin") == "confirmado") & (pl.col("evolucao") == "cura"))
        .select("municipality_residence_code7", "ate_hosp")
        .collect()
        .with_columns(
            pl.col("municipality_residence_code7").cast(pl.Utf8).str.zfill(7)
            .str.slice(0, 6).alias("m6")
        )
    )
    mun = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7).str.slice(0, 6).alias("m6"),
        "health_region_code",
    )
    ll = ll.join(mun, on="m6", how="inner")
    by_m = ll.group_by("m6").agg(
        pl.len().alias("surv_cases"),
        (pl.col("ate_hosp") == "sim").sum().alias("surv_hosp"),
    ).with_columns(
        (pl.col("surv_hosp") / pl.col("surv_cases")).alias("munic_hosp_share_surv")
    ).select("m6", "munic_hosp_share_surv", "surv_cases")
    by_hr = ll.group_by("health_region_code").agg(
        pl.len().alias("hr_surv_cases"),
        (pl.col("ate_hosp") == "sim").sum().alias("hr_surv_hosp"),
    ).with_columns(
        (pl.col("hr_surv_hosp") / pl.col("hr_surv_cases")).alias("hs_res_surv")
    ).select("health_region_code", "hs_res_surv", "hr_surv_cases")
    return by_m, by_hr


def _munic_lookup() -> pl.DataFrame:
    m = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    return m.select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7).str.slice(0, 6).alias("m6"),
        "health_region_code",
        "region",
        "uf_abbr",
        pl.col("cases").alias("munic_cases"),
        (pl.col("hospitalised") / pl.col("cases")).alias("munic_hosp_share"),
        "urban_share_mean",
    )


# --------------------------------------------------------------------------
# analysis frame
# --------------------------------------------------------------------------
def build_frame() -> pl.DataFrame:
    """One row per A27 admission, carrying home territory and treating hospital."""
    d = pl.read_parquet(PATHS.interim / "sih_a27_admissions.parquet").with_columns(
        pl.col("MUNIC_RES").cast(pl.Utf8).str.zfill(6).alias("m6_res"),
        pl.col("MUNIC_MOV").cast(pl.Utf8).str.zfill(6).alias("m6_mov"),
        pl.col("DT_INTER").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("admit"),
        (pl.col("MORTE") == "1").alias("died"),
        pl.col("UTI_MES_TO").cast(pl.Float64, strict=False).alias("icu_days"),
        pl.col("DIAS_PERM").cast(pl.Float64, strict=False).alias("los"),
        pl.col("IDADE").cast(pl.Float64, strict=False).alias("idade_raw"),
        pl.col("COD_IDADE").alias("age_unit"),
        (pl.col("SEXO") == "3").alias("female"),
    ).with_columns(
        pl.col("admit").dt.year().alias("year"),
        # COD_IDADE 4 = years, 5 = years past 100, 3 = months, 2 = days.
        pl.when(pl.col("age_unit") == "4").then(pl.col("idade_raw"))
        .when(pl.col("age_unit") == "5").then(pl.col("idade_raw") + 100)
        .when(pl.col("age_unit") == "3").then(pl.col("idade_raw") / 12)
        .when(pl.col("age_unit") == "2").then(pl.col("idade_raw") / 365)
        .otherwise(0.0).alias("age_years"),
    ).filter(pl.col("year").is_between(*YEARS))

    mun = _munic_lookup()
    res = mun.rename({
        "m6": "m6_res", "health_region_code": "hr_res", "region": "reg_res",
        "uf_abbr": "uf_res",
    })
    mov = mun.select("m6", "health_region_code").rename(
        {"m6": "m6_mov", "health_region_code": "hr_mov"}
    )
    before = d.height
    d = d.join(res, on="m6_res", how="left").join(mov, on="m6_mov", how="left")
    if d["hr_res"].null_count() > 0.05 * before:
        raise AssertionError("residence geography join lost more than 5% of admissions")

    b = bands()
    d = d.join(b.rename({"health_region_code": "hr_res", "quintile": "band_res",
                         "hosp_share": "hs_res"}), on="hr_res", how="left")
    d = d.join(b.rename({"health_region_code": "hr_mov", "quintile": "band_mov",
                         "hosp_share": "hs_mov"}), on="hr_mov", how="left")

    surv_m, surv_hr = survivor_depth()
    d = d.join(surv_m.rename({"m6": "m6_res"}), on="m6_res", how="left")
    d = d.join(surv_hr.rename({"health_region_code": "hr_res"}), on="hr_res", how="left")

    d = d.with_columns(
        (pl.col("icu_days") > 0).alias("any_icu"),
        (pl.col("m6_res") != pl.col("m6_mov")).alias("referred_munic"),
        (pl.col("hr_res") != pl.col("hr_mov")).alias("referred_hr"),
    )
    # hospital-level descriptors, computed on the full A27 file so a hospital's
    # character does not depend on which patients survive a later filter
    hosp = d.group_by("CNES").agg(
        pl.len().alias("hosp_volume"),
        pl.col("referred_munic").mean().alias("hosp_inflow_share"),
        (pl.col("any_icu").sum() > 0).alias("hosp_ever_icu"),
        pl.col("hr_res").n_unique().alias("hosp_n_home_hr"),
        pl.col("band_res").n_unique().alias("hosp_n_home_band"),
    )
    d = d.join(hosp, on="CNES", how="left")
    d = d.filter(pl.col("band_res").is_not_null() & pl.col("hr_mov").is_not_null())

    # A hospital sits in exactly one place. If CNES codes were reused across
    # regions the nested decomposition below would be meaningless, so check
    # rather than assume.
    multi_site = d.group_by("CNES").agg(pl.col("hr_mov").n_unique().alias("k")) \
        .filter(pl.col("k") > 1)
    if multi_site.height:
        share = d.join(multi_site.select("CNES"), on="CNES").height / d.height
        print(f"  note: {multi_site.height} CNES codes appear in more than one "
              f"health region ({100 * share:.2f}% of admissions)")
    return d


# --------------------------------------------------------------------------
# 1. where does the variance live
# --------------------------------------------------------------------------
def _dl_tau2(x: np.ndarray, n: np.ndarray) -> dict:
    """Between-cluster variance of a proportion, sampling noise removed.

    A DerSimonian-Laird style method of moments on the raw proportion scale.
    Observed between-cluster variance in a binary outcome is inflated by
    binomial sampling: a hospital with twelve admissions and one death reads
    8.3% for reasons that have nothing to do with the hospital. Subtracting the
    expected within-cluster variance leaves the part attributable to the
    clusters themselves. Truncated at zero, because a negative variance
    estimate means "no detectable heterogeneity", not a negative quantity.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    p_bar = x.sum() / n.sum()
    w = n
    obs = float(np.sum(w * (x / n - p_bar) ** 2) / w.sum())
    exp_sampling = float(p_bar * (1 - p_bar) * len(n) / w.sum())
    tau2 = max(obs - exp_sampling, 0.0)
    return {
        "clusters": int(len(n)), "events": int(x.sum()), "n": int(n.sum()),
        "pooled_pct": 100 * p_bar,
        "observed_var": obs, "sampling_var": exp_sampling,
        "tau2": tau2, "tau_pct_points": 100 * float(np.sqrt(tau2)),
    }


def moment_decomposition(d: pl.DataFrame, outcome: str, min_vol: int) -> dict:
    """Between-hospital-within-region against between-region heterogeneity.

    Hospitals sit inside health regions, so the two levels are not exchangeable
    competitors: the question is whether hospitals *within* a region differ from
    one another as much as regions differ from one another. Both components are
    estimated on the same restricted set -- hospitals above the volume
    threshold, in regions holding at least two of them -- so the comparison is
    like for like. The unrestricted between-region figure is reported alongside
    so the cost of the restriction is visible.
    """
    hosp_all = d.group_by(["CNES", "hr_mov"]).agg(
        pl.col(outcome).sum().alias("x"), pl.len().alias("n")
    )
    reg_all = d.group_by("hr_mov").agg(pl.col(outcome).sum().alias("x"), pl.len().alias("n"))

    big = hosp_all.filter(pl.col("n") >= min_vol)
    multi = big.group_by("hr_mov").len().filter(pl.col("len") >= 2)["hr_mov"].to_list()
    big = big.filter(pl.col("hr_mov").is_in(multi))
    sub = d.filter(pl.col("hr_mov").is_in(multi) & (pl.col("hosp_volume") >= min_vol))
    reg_sub = sub.group_by("hr_mov").agg(pl.col(outcome).sum().alias("x"), pl.len().alias("n"))

    parts = []
    for r in multi:
        h = big.filter(pl.col("hr_mov") == r)
        parts.append(_dl_tau2(h["x"].to_numpy(), h["n"].to_numpy()))
    wts = np.array([p["n"] for p in parts], dtype=float)
    tau2_hosp = float(np.sum(wts * [p["tau2"] for p in parts]) / wts.sum())
    within = {
        "regions_assessed": len(parts), "hospitals": big.height,
        "admissions": int(wts.sum()), "tau2": tau2_hosp,
        "tau_pct_points": 100 * float(np.sqrt(tau2_hosp)),
    }
    between = _dl_tau2(reg_sub["x"].to_numpy(), reg_sub["n"].to_numpy())
    return {
        "outcome": outcome, "min_hospital_volume": min_vol,
        "between_region_unrestricted": _dl_tau2(reg_all["x"].to_numpy(),
                                                reg_all["n"].to_numpy()),
        "between_hospital_unrestricted": _dl_tau2(hosp_all["x"].to_numpy(),
                                                  hosp_all["n"].to_numpy()),
        "between_region": between,
        "between_hospital_within_region": within,
        "ratio_within_region_hospital_to_between_region": (
            float(tau2_hosp / between["tau2"]) if between["tau2"] > 0 else None
        ),
    }


# --------------------------------------------------------------------------
# 2. the decisive test -- hold the hospital fixed
# --------------------------------------------------------------------------
def identifying_variation(d: pl.DataFrame, min_vol: int) -> dict:
    """How much within-hospital variation in home band actually exists.

    Stated before the test is run. If a hospital's patients all come from one
    band, that hospital contributes nothing to a hospital-stratified estimate,
    however many admissions it has.
    """
    sub = d.filter((pl.col("hosp_volume") >= min_vol) & (pl.col("hosp_n_home_band") >= 2))
    if sub.is_empty():
        return {"min_hospital_volume": min_vol, "hospitals": 0}
    sub = sub.with_columns(
        pl.col("band_res").mode().first().over("CNES").alias("modal_band"),
        (pl.col("hs_res") - pl.col("hs_res").mean().over("CNES")).alias("hs_dev"),
    )
    off = sub.filter(pl.col("band_res") != pl.col("modal_band"))
    return {
        "min_hospital_volume": min_vol,
        "hospitals": int(sub["CNES"].n_unique()),
        "admissions": sub.height,
        "deaths": int(sub["died"].sum()),
        "off_modal_band_admissions": off.height,
        "off_modal_band_deaths": int(off["died"].sum()),
        "off_modal_band_icu": int(off["any_icu"].sum()),
        "within_hospital_sd_home_hosp_share": float(sub["hs_dev"].std()),
        "national_sd_home_hosp_share": float(d["hs_res"].std()),
        "variance_retained_pct": 100 * float(
            (sub["hs_dev"].std() / d["hs_res"].std()) ** 2
        ),
    }


def mh_odds_ratio(d: pl.DataFrame, outcome: str, exposure: str) -> dict:
    """Mantel-Haenszel odds ratio stratified by treating hospital.

    Model-free: every stratum is one hospital, so nothing about the hospital --
    its ICU, its staffing, its referral role -- can contribute. Only the
    contrast between patients from different home territories treated in the
    same building.
    """
    g = d.group_by("CNES").agg(
        (pl.col(exposure) & pl.col(outcome)).sum().alias("a"),
        (pl.col(exposure) & ~pl.col(outcome)).sum().alias("b"),
        (~pl.col(exposure) & pl.col(outcome)).sum().alias("c"),
        (~pl.col(exposure) & ~pl.col(outcome)).sum().alias("d"),
    )
    a, b, c, dd = (g[k].to_numpy().astype(float) for k in "abcd")
    n = a + b + c + dd
    keep = (n > 0) & ((a + b) > 0) & ((c + dd) > 0)
    a, b, c, dd, n = a[keep], b[keep], c[keep], dd[keep], n[keep]
    num, den = np.sum(a * dd / n), np.sum(b * c / n)
    if den == 0 or num == 0:
        return {"strata": int(keep.sum()), "or_mh": None,
                "note": "no informative stratum"}
    or_mh = num / den
    # Robins-Breslow-Greenland variance of log(OR_MH)
    p, q = (a + dd) / n, (b + c) / n
    r, s = a * dd / n, b * c / n
    var = (np.sum(p * r) / (2 * num ** 2)
           + np.sum(p * s + q * r) / (2 * num * den)
           + np.sum(q * s) / (2 * den ** 2))
    se = float(np.sqrt(var))
    return {
        "strata": int(keep.sum()),
        "exposed_events": int(a.sum()), "exposed_n": int((a + b).sum()),
        "unexposed_events": int(c.sum()), "unexposed_n": int((c + dd).sum()),
        "or_mh": float(or_mh),
        "or_lo": float(np.exp(np.log(or_mh) - 1.96 * se)),
        "or_hi": float(np.exp(np.log(or_mh) + 1.96 * se)),
        "se_log_or": se,
    }


def crude_vs_stratified(d: pl.DataFrame, min_vol: int) -> dict:
    """The crude band contrast, then the same contrast within hospitals.

    Exposure is a *shallow* home band (Q4-Q5) against a deep one (Q1-Q3). The
    split is at the median of the case-weighted band scale, and it is the
    coarsest contrast that still spans the gradient, which is what the thin
    within-hospital variation can support.
    """
    sub = d.filter((pl.col("hosp_volume") >= min_vol) & (pl.col("hosp_n_home_band") >= 2))
    sub = sub.with_columns((pl.col("band_res") >= 4).alias("shallow_home"))
    res = {"min_hospital_volume": min_vol, "admissions": sub.height}
    for outcome in ("died", "any_icu"):
        num = sub.group_by("shallow_home").agg(
            pl.col(outcome).sum().alias("x"), pl.len().alias("n")
        ).sort("shallow_home")
        x = num["x"].to_numpy()
        nn = num["n"].to_numpy().astype(float)
        p, lo, hi = binom_ci(x, nn)
        crude_or = (x[1] / (nn[1] - x[1])) / (x[0] / (nn[0] - x[0]))
        res[outcome] = {
            "deep_home_pct": 100 * float(p[0]), "deep_lo": 100 * float(lo[0]),
            "deep_hi": 100 * float(hi[0]), "deep_n": int(nn[0]),
            "shallow_home_pct": 100 * float(p[1]), "shallow_lo": 100 * float(lo[1]),
            "shallow_hi": 100 * float(hi[1]), "shallow_n": int(nn[1]),
            "crude_or": float(crude_or),
            "hospital_stratified": mh_odds_ratio(sub, outcome, "shallow_home"),
        }
        mh = res[outcome]["hospital_stratified"]
        if mh.get("se_log_or"):
            # what the test could have found: the smallest odds ratio detectable
            # at 80% power and 5% two-sided, given the discordance available.
            res[outcome]["min_detectable_or_80pct_power"] = float(
                np.exp(2.80 * mh["se_log_or"])
            )
    return res


def _assert_inside(est, lo, hi, what: str) -> None:
    est, lo, hi = np.asarray(est), np.asarray(lo), np.asarray(hi)
    ok = np.isnan(est) | ((est >= lo - 1e-12) & (est <= hi + 1e-12))
    if not np.all(ok):
        raise AssertionError(f"{what}: estimate outside its own interval")


def discordant_flow(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Patients treated in a band other than their own: whose band tracks the outcome?

    A direct read of the same contrast without any stratification machinery.
    The summary is an expected-count comparison: apply the home-band fatality
    schedule to these patients, then the treatment-band schedule, and see which
    reproduces the deaths actually observed. Home and treatment are perfectly
    confounded for the 93% of patients who never leave, so only these 1,198
    carry any information about which label is doing the work.
    """
    x = d.filter(pl.col("band_mov").is_not_null() & (pl.col("band_res") != pl.col("band_mov")))
    g = x.group_by(["band_res", "band_mov"]).agg(
        pl.len().alias("n"), pl.col("died").sum().alias("deaths"),
        pl.col("any_icu").sum().alias("icu"),
    ).sort(["band_res", "band_mov"])
    cfr, lo, hi = binom_ci(g["deaths"].to_numpy(), g["n"].to_numpy().astype(float))
    _assert_inside(cfr, lo, hi, "discordant cell fatality")
    tab = g.with_columns(
        pl.Series("cfr_pct", 100 * cfr), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    )

    # marginal fatality schedules, estimated on the concordant patients only, so
    # the discordant ones are not used to build the yardstick they are judged by
    conc = d.filter(pl.col("band_res") == pl.col("band_mov"))
    sched = conc.group_by("band_res").agg(
        (pl.col("died").sum() / pl.len()).alias("p")
    ).sort("band_res")
    p = dict(zip(sched["band_res"].to_list(), sched["p"].to_list()))
    exp_home = float(sum(r["n"] * p[r["band_res"]] for r in tab.iter_rows(named=True)))
    exp_treat = float(sum(r["n"] * p[r["band_mov"]] for r in tab.iter_rows(named=True)))
    obs = int(tab["deaths"].sum())
    summary = {
        "discordant_admissions": int(tab["n"].sum()),
        "observed_deaths": obs,
        "expected_under_home_band_schedule": exp_home,
        "expected_under_treatment_band_schedule": exp_treat,
        "concordant_fatality_schedule_pct": {int(k): 100 * v for k, v in p.items()},
    }
    return tab, summary


def home_vs_treatment_marginals(d: pl.DataFrame) -> pl.DataFrame:
    """In-hospital fatality banded by HOME region against banded by TREATMENT region.

    If depth is a territory property the two should look alike, because most
    patients are treated at home. Divergence localises the gradient.
    """
    rows = []
    for key, label in (("band_res", "home (residence)"), ("band_mov", "treatment")):
        g = d.filter(pl.col(key).is_not_null()).group_by(key).agg(
            pl.len().alias("n"), pl.col("died").sum().alias("deaths"),
            pl.col("any_icu").sum().alias("icu"),
        ).sort(key).rename({key: "band"})
        cfr, lo, hi = binom_ci(g["deaths"].to_numpy(), g["n"].to_numpy().astype(float))
        icu, ilo, ihi = binom_ci(g["icu"].to_numpy(), g["n"].to_numpy().astype(float))
        _assert_inside(cfr, lo, hi, f"in-hospital fatality by {label}")
        _assert_inside(icu, ilo, ihi, f"ICU use by {label}")
        rows.append(g.with_columns(
            pl.lit(label).alias("banded_by"),
            pl.Series("cfr_pct", 100 * cfr), pl.Series("cfr_lo", 100 * lo),
            pl.Series("cfr_hi", 100 * hi),
            pl.Series("icu_pct", 100 * icu), pl.Series("icu_lo", 100 * ilo),
            pl.Series("icu_hi", 100 * ihi),
        ))
    return pl.concat(rows, how="diagonal")


def referral_centre_hypothesis(d: pl.DataFrame) -> dict:
    """'A few referral centres take the sickest' -- stated, then tested.

    Threat: the band gradient in in-hospital fatality is produced by shallow
    bands routing their patients into a small number of high-volume referral
    hospitals whose case mix is severe for reasons unrelated to territory.
    Two checks: how concentrated the bands' admissions are, and whether the
    gradient survives among patients who never left their own municipality --
    i.e. among people who by construction did not reach a referral centre.
    """
    conc = []
    for q in range(1, 6):
        s = d.filter(pl.col("band_res") == q)
        v = s.group_by("CNES").len().sort("len", descending=True)
        tot = v["len"].sum()
        conc.append({
            "band": q, "admissions": int(tot), "hospitals": v.height,
            "top1_share_pct": 100 * float(v["len"][0] / tot),
            "top5_share_pct": 100 * float(v["len"][:5].sum() / tot),
            "top10_share_pct": 100 * float(v["len"][:10].sum() / tot),
            "hhi": float(np.sum((v["len"].to_numpy() / tot) ** 2)),
        })
    strata = {}
    for name, sub in (
        ("all", d),
        ("treated in own municipality", d.filter(~pl.col("referred_munic"))),
        ("referred out of municipality", d.filter(pl.col("referred_munic"))),
        ("hospital volume < 50", d.filter(pl.col("hosp_volume") < MIN_HOSP_VOLUME)),
        ("hospital volume >= 50", d.filter(pl.col("hosp_volume") >= MIN_HOSP_VOLUME)),
    ):
        g = sub.group_by("band_res").agg(
            pl.len().alias("n"), pl.col("died").sum().alias("deaths")
        ).sort("band_res")
        cfr, lo, hi = binom_ci(g["deaths"].to_numpy(), g["n"].to_numpy().astype(float))
        strata[name] = {
            "n": g["n"].to_list(), "deaths": g["deaths"].to_list(),
            "cfr_pct": [100 * float(v) for v in cfr],
            "cfr_lo": [100 * float(v) for v in lo],
            "cfr_hi": [100 * float(v) for v in hi],
            "Q5_over_Q1": float(cfr[4] / cfr[0]) if cfr[0] > 0 else None,
        }
    return {"concentration": conc, "gradient_within_strata": strata}


# --------------------------------------------------------------------------
# 3. the non-monotone admission rate
# --------------------------------------------------------------------------
def sinan_by_band() -> pl.DataFrame:
    """Confirmed cases and recorded hospitalisations by band, on the SIH window."""
    ll = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter(pl.col("classi_fin") == "confirmado")
        .select("municipality_residence_code7", "notification_year", "ate_hosp")
        .collect()
        .with_columns(
            pl.col("municipality_residence_code7").cast(pl.Utf8).str.zfill(7)
            .str.slice(0, 6).alias("m6_res")
        )
        .filter(pl.col("notification_year").is_between(*YEARS))
    )
    mun = _munic_lookup().select("m6", "health_region_code", "region").rename(
        {"m6": "m6_res", "health_region_code": "hr_res", "region": "reg_res"}
    )
    ll = ll.join(mun, on="m6_res", how="inner").join(
        bands().rename({"health_region_code": "hr_res", "quintile": "band"}),
        on="hr_res", how="inner",
    )
    return ll.group_by(["band", "reg_res"]).agg(
        pl.len().alias("sinan_cases"),
        (pl.col("ate_hosp") == "sim").sum().alias("sinan_hospitalised"),
    )


def rate_arithmetic(d: pl.DataFrame) -> dict:
    """Decompose the admission rate before reaching for any explanation.

    A population-based rate of *hospitalised* cases is the product of two things
    the banding already fixes: the notification rate and the hospitalisation
    share. If the product is non-monotone, no fact about hospitals is needed to
    produce a non-monotone admission rate.
    """
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    py = hr.select("health_region_code", "person_years", "region").join(
        bands().rename({"health_region_code": "health_region_code", "quintile": "band"}),
        on="health_region_code", how="inner",
    )
    scale = (YEARS[1] - YEARS[0] + 1) / ATLAS_YEARS
    g_py = py.group_by("band").agg(
        (pl.col("person_years").sum() * scale).alias("person_years")
    ).sort("band")

    sn = sinan_by_band().group_by("band").agg(
        pl.col("sinan_cases").sum(), pl.col("sinan_hospitalised").sum()
    ).sort("band")
    adm = d.group_by("band_res").agg(pl.len().alias("sih_admissions")).sort("band_res") \
        .rename({"band_res": "band"})
    m = g_py.join(sn, on="band").join(adm, on="band")

    py_v = m["person_years"].to_numpy()
    inc, inc_lo, inc_hi = poisson_ci(m["sinan_cases"].to_numpy(), py_v, scale=1e5)
    hos, hos_lo, hos_hi = poisson_ci(m["sinan_hospitalised"].to_numpy(), py_v, scale=1e5)
    sih, sih_lo, sih_hi = poisson_ci(m["sih_admissions"].to_numpy(), py_v, scale=1e5)
    for est, lo, hi in ((inc, inc_lo, inc_hi), (hos, hos_lo, hos_hi), (sih, sih_lo, sih_hi)):
        assert np.all((est >= lo) & (est <= hi)), "estimate outside its own interval"

    share = m["sinan_hospitalised"].to_numpy() / m["sinan_cases"].to_numpy()
    return {
        "person_years": [float(v) for v in py_v],
        "sinan_incidence_per_100k": [float(v) for v in inc],
        "sinan_incidence_lo": [float(v) for v in inc_lo],
        "sinan_incidence_hi": [float(v) for v in inc_hi],
        "sinan_hospitalisation_share": [float(v) for v in share],
        "sinan_hospitalised_rate_per_100k": [float(v) for v in hos],
        "sinan_hospitalised_lo": [float(v) for v in hos_lo],
        "sinan_hospitalised_hi": [float(v) for v in hos_hi],
        "sih_admission_rate_per_100k": [float(v) for v in sih],
        "sih_admission_lo": [float(v) for v in sih_lo],
        "sih_admission_hi": [float(v) for v in sih_hi],
        "sih_over_sinan_hospitalised": [float(a / b) for a, b in zip(sih, hos)],
        "band_to_band_incidence_ratio": [float(inc[i] / inc[i + 1]) for i in range(4)],
        "band_to_band_share_ratio": [float(share[i + 1] / share[i]) for i in range(4)],
        "counts": m.to_dicts(),
    }


def nonmonotonicity_candidates(d: pl.DataFrame) -> dict:
    """The three offered explanations, each tested against the arithmetic."""
    out: dict = {}

    # (a) referral flow. The rate is residence-based, so referral cannot move
    #     it by construction; recomputing on treatment geography shows what
    #     referral would have to do to matter.
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    scale = (YEARS[1] - YEARS[0] + 1) / ATLAS_YEARS
    py = hr.select("health_region_code", "person_years").join(
        bands().rename({"quintile": "band"}), on="health_region_code", how="inner"
    ).group_by("band").agg((pl.col("person_years").sum() * scale).alias("py")).sort("band")
    by_treat = d.filter(pl.col("band_mov").is_not_null()).group_by("band_mov").agg(
        pl.len().alias("n")).sort("band_mov").rename({"band_mov": "band"})
    m = py.join(by_treat, on="band")
    r_t, _, _ = poisson_ci(m["n"].to_numpy(), m["py"].to_numpy(), scale=1e5)
    by_res = d.group_by("band_res").agg(pl.len().alias("n")).sort("band_res")
    r_r, _, _ = poisson_ci(by_res["n"].to_numpy(), py["py"].to_numpy(), scale=1e5)
    out["referral"] = {
        "rate_by_residence": [float(v) for v in r_r],
        "rate_by_treatment_location": [float(v) for v in r_t],
        "net_inflow_pct": [
            100 * float(t / r - 1) for t, r in zip(m["n"].to_numpy(), by_res["n"].to_numpy())
        ],
    }

    # (b) macro-region mixing. If the hump is regional composition it should
    #     vanish inside a macro-region.
    sn = sinan_by_band()
    hr_reg = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet").join(
        bands().rename({"quintile": "band"}), on="health_region_code", how="inner"
    )
    py_reg = hr_reg.group_by(["band", "region"]).agg(
        (pl.col("person_years").sum() * scale).alias("py")
    )
    mm = sn.rename({"reg_res": "region"}).join(py_reg, on=["band", "region"], how="inner")
    rate, lo, hi = poisson_ci(mm["sinan_hospitalised"].to_numpy(), mm["py"].to_numpy(), scale=1e5)
    mm = mm.with_columns(
        pl.Series("hosp_rate_per_100k", rate), pl.Series("lo", lo), pl.Series("hi", hi)
    ).sort(["region", "band"])
    out["within_macro_region"] = mm.to_dicts()

    # (c) hospital supply. Distinct A27-treating hospitals per million
    #     person-years, and the share of admissions in ICU-capable hospitals.
    sup = d.group_by("band_res").agg(
        pl.col("CNES").n_unique().alias("hospitals"),
        pl.len().alias("admissions"),
        pl.col("hosp_ever_icu").mean().alias("share_adm_icu_capable"),
    ).sort("band_res")
    out["hospital_supply"] = {
        "hospitals": sup["hospitals"].to_list(),
        "hospitals_per_million_py": [
            float(h / (p / 1e6)) for h, p in zip(sup["hospitals"].to_list(),
                                                 py["py"].to_list())
        ],
        "share_admissions_in_icu_capable_hospital_pct": [
            100 * float(v) for v in sup["share_adm_icu_capable"].to_list()
        ],
    }

    # (b2) the same claim as a direct standardisation. If the bands' rate shape
    #      is regional composition, removing the composition removes the shape.
    std_w = py_reg.group_by("region").agg(pl.col("py").sum().alias("w"))
    m2 = mm.join(std_w, on="region", how="inner").with_columns(
        (pl.col("sinan_hospitalised") / pl.col("py")).alias("stratum_rate")
    )
    std = m2.group_by("band").agg(
        ((pl.col("stratum_rate") * pl.col("w")).sum() / pl.col("w").sum() * 1e5)
        .alias("std_hosp_rate_per_100k"),
        (pl.col("w").sum()).alias("std_pop_covered"),
    ).sort("band")
    crude = m2.group_by("band").agg(
        (pl.col("sinan_hospitalised").sum() / pl.col("py").sum() * 1e5).alias("crude")
    ).sort("band")
    out["macro_region_standardised"] = {
        "crude_hospitalised_rate": crude["crude"].to_list(),
        "standardised_hospitalised_rate": std["std_hosp_rate_per_100k"].to_list(),
        "note": "standard population = national person-years by macro-region; "
                "bands with no person-years in a macro-region contribute nothing "
                "to that stratum, so coverage is stated alongside",
        "strata_covered_per_band": m2.group_by("band").len().sort("band")["len"].to_list(),
    }

    # (e) the plateau itself. The banding variable is hospitalisation share; if
    #     share and incidence were monotonically linked a plateau could not
    #     occur, so the rank correlation between them is the whole story.
    hrb = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet").filter(
        pl.col("cases") >= MIN_CASES
    ).with_columns((pl.col("hospitalised") / pl.col("cases")).alias("hs"))
    from scipy import stats as _st
    rho = _st.spearmanr(hrb["hs"].to_numpy(), hrb["incidence_per_100k"].to_numpy())
    out["share_vs_incidence_across_health_regions"] = {
        "n_health_regions": hrb.height,
        "spearman": float(rho.statistic), "p": float(rho.pvalue),
    }

    # (d) urbanisation, case-weighted
    mun = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").join(
        bands().rename({"quintile": "band"}), on="health_region_code", how="inner"
    ).filter(pl.col("cases") > 0)
    urb = mun.group_by("band").agg(
        ((pl.col("urban_share_mean") * pl.col("cases")).sum() / pl.col("cases").sum())
        .alias("case_weighted_urban_share")
    ).sort("band")
    out["urbanisation"] = urb.to_dicts()
    return out


def plateau_contributors() -> pl.DataFrame:
    """Which health regions carry the Q2-Q3 notification plateau."""
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet").join(
        bands().rename({"quintile": "band"}), on="health_region_code", how="inner"
    )
    hr = hr.filter(pl.col("band").is_in([2, 3]))
    rate, lo, hi = poisson_ci(hr["cases"].to_numpy(), hr["person_years"].to_numpy(), scale=1e5)
    return hr.select(
        "band", "health_region_name", "uf_abbr", "region", "cases", "person_years"
    ).with_columns(
        pl.Series("incidence_per_100k", rate)
    ).sort("cases", descending=True).head(20)


# --------------------------------------------------------------------------
def run_r(frame_path: Path) -> dict:
    """lme4 variance components and hospital-stratified conditional logistic.

    Run with the working directory outside the repository: the project's
    ``.Rprofile`` activates an ``renv`` library that does not carry lme4, and
    the models want the user library.
    """
    proc = subprocess.run(
        [RSCRIPT, str(R_HELPER), str(frame_path.resolve()), str(OUT.resolve())],
        capture_output=True, text=True, cwd=str(OUT.resolve()),
    )
    print(proc.stdout[-12000:])
    if proc.returncode != 0:
        print("R FAILED:\n" + proc.stderr[-4000:])
        return {"error": proc.stderr[-2000:]}
    p = OUT / "r_models.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_frame()
    print(f"A27 admissions {YEARS[0]}-{YEARS[1]} in banded health regions: {d.height}")
    print(f"  hospitals {d['CNES'].n_unique()}   deaths {int(d['died'].sum())}   "
          f"ICU {int(d['any_icu'].sum())}")
    print(f"  crossed a municipal boundary {int(d['referred_munic'].sum())} "
          f"({100 * d['referred_munic'].mean():.1f}%), a health-region boundary "
          f"{int(d['referred_hr'].sum())} ({100 * d['referred_hr'].mean():.1f}%)")

    frame_path = OUT / "analysis_frame.parquet"
    d.select(
        "CNES", "hr_res", "hr_mov", "band_res", "band_mov", "hs_res", "hs_mov",
        "munic_hosp_share", "munic_cases", "munic_hosp_share_surv", "surv_cases",
        "hs_res_surv", "hr_surv_cases", "died", "any_icu", "los", "age_years",
        "female", "year", "referred_munic", "referred_hr", "hosp_volume",
        "hosp_inflow_share", "hosp_ever_icu", "hosp_n_home_band", "reg_res", "uf_res",
    ).write_parquet(frame_path)

    report: dict = {"admissions": d.height, "window": list(YEARS),
                    "hospitals": int(d["CNES"].n_unique())}

    # ---- 1 -------------------------------------------------------------
    print("\n=== 1. where does the variance live ===")
    mom = {}
    for outcome in ("died", "any_icu"):
        mom[outcome] = moment_decomposition(d, outcome, MIN_HOSP_VOLUME)
        m = mom[outcome]
        br, bh = m["between_region"], m["between_hospital_within_region"]
        print(f"  {outcome:>8}: between-region tau {br['tau_pct_points']:.2f} pp "
              f"({br['clusters']} regions, {br['n']} adm) | between-hospital *within* "
              f"region tau {bh['tau_pct_points']:.2f} pp ({bh['hospitals']} hospitals in "
              f"{bh['regions_assessed']} regions) | variance ratio "
              f"{m['ratio_within_region_hospital_to_between_region']:.2f}")
        u = m["between_region_unrestricted"]
        print(f"            unrestricted between-region tau {u['tau_pct_points']:.2f} pp "
              f"({u['clusters']} regions, pooled {u['pooled_pct']:.2f}%)")
    report["moment_decomposition"] = mom

    # ---- 2 -------------------------------------------------------------
    print("\n=== 2. the decisive test: hold the hospital fixed ===")
    idv = {v: identifying_variation(d, v) for v in VOLUME_SENSITIVITY}
    for v, s in idv.items():
        print(f"  volume>={v}: {s['hospitals']} multi-band hospitals, "
              f"{s['admissions']} admissions, of which {s['off_modal_band_admissions']} "
              f"from a non-modal band ({s['off_modal_band_deaths']} deaths); "
              f"within-hospital SD of home hosp-share {s['within_hospital_sd_home_hosp_share']:.3f} "
              f"vs national {s['national_sd_home_hosp_share']:.3f} "
              f"({s['variance_retained_pct']:.1f}% of variance)")
    report["identifying_variation"] = idv

    strat = {v: crude_vs_stratified(d, v) for v in VOLUME_SENSITIVITY}
    for v, s in strat.items():
        for outcome in ("died", "any_icu"):
            o = s[outcome]
            mh = o["hospital_stratified"]
            print(f"  volume>={v} {outcome:>8}: crude shallow/deep OR {o['crude_or']:.2f}"
                  f"  -> hospital-stratified MH OR "
                  f"{mh['or_mh']:.2f} ({mh['or_lo']:.2f}-{mh['or_hi']:.2f}) "
                  f"on {mh['strata']} hospitals; smallest OR this test could find "
                  f"{o['min_detectable_or_80pct_power']:.2f}")
    report["hospital_stratified"] = strat

    hvt = home_vs_treatment_marginals(d)
    hvt.write_csv(OUT / "home_vs_treatment_bands.csv")
    print("\n  in-hospital fatality banded by home vs by treatment location")
    for r in hvt.iter_rows(named=True):
        print(f"    {r['banded_by']:<18} Q{r['band']} n={r['n']:>6} "
              f"CFR {r['cfr_pct']:>5.2f}% ({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})  "
              f"ICU {r['icu_pct']:>5.1f}%")

    disc, disc_sum = discordant_flow(d)
    disc.write_csv(OUT / "discordant_home_vs_treatment.csv")
    report["discordant_cells"] = disc.to_dicts()
    report["discordant_summary"] = disc_sum
    print(f"\n  {disc_sum['discordant_admissions']} patients treated outside their own "
          f"band: {disc_sum['observed_deaths']} deaths observed, "
          f"{disc_sum['expected_under_home_band_schedule']:.1f} expected on the HOME "
          f"schedule, {disc_sum['expected_under_treatment_band_schedule']:.1f} on the "
          f"TREATMENT schedule")

    ref = referral_centre_hypothesis(d)
    report["referral_centre_hypothesis"] = ref
    print("\n  admission concentration and the gradient within care-pathway strata")
    for c in ref["concentration"]:
        print(f"    Q{c['band']} {c['hospitals']:>4} hospitals, top-5 share "
              f"{c['top5_share_pct']:>5.1f}%, HHI {c['hhi']:.4f}")
    for k, v in ref["gradient_within_strata"].items():
        tail = f"   Q5/Q1 {v['Q5_over_Q1']:.2f}" if v["Q5_over_Q1"] else ""
        print(f"    {k:<30} n=" + " ".join(f"{n:6d}" for n in v["n"]) +
              "  CFR " + " ".join(f"{x:5.2f}" for x in v["cfr_pct"]) + tail)

    # ---- 3 -------------------------------------------------------------
    print("\n=== 3. the non-monotone admission rate ===")
    arith = rate_arithmetic(d)
    report["rate_arithmetic"] = arith
    print(f"  {'band':>5}{'SINAN inc':>11}{'hosp share':>12}{'SINAN hosp rate':>17}"
          f"{'SIH adm rate':>14}{'SIH/SINAN':>11}")
    for i in range(5):
        print(f"  {i+1:>5}{arith['sinan_incidence_per_100k'][i]:>11.2f}"
              f"{arith['sinan_hospitalisation_share'][i]:>12.3f}"
              f"{arith['sinan_hospitalised_rate_per_100k'][i]:>17.2f}"
              f"{arith['sih_admission_rate_per_100k'][i]:>14.2f}"
              f"{arith['sih_over_sinan_hospitalised'][i]:>11.2f}")
    print("  band-to-band incidence ratio Qi/Qi+1: " +
          " ".join(f"{v:.2f}" for v in arith["band_to_band_incidence_ratio"]))
    print("  band-to-band share ratio Qi+1/Qi:    " +
          " ".join(f"{v:.2f}" for v in arith["band_to_band_share_ratio"]))

    cand = nonmonotonicity_candidates(d)
    report["nonmonotonicity_candidates"] = cand
    print("\n  referral: rate by residence " +
          " ".join(f"{v:.2f}" for v in cand["referral"]["rate_by_residence"]))
    print("            rate by treatment " +
          " ".join(f"{v:.2f}" for v in cand["referral"]["rate_by_treatment_location"]))
    print("  hospitals per million py  " +
          " ".join(f"{v:.2f}" for v in cand["hospital_supply"]["hospitals_per_million_py"]))
    print("  case-weighted urban share " +
          " ".join(f"{r['case_weighted_urban_share']:.3f}" for r in cand["urbanisation"]))
    st = cand["macro_region_standardised"]
    print("  SINAN hospitalised rate, crude        " +
          " ".join(f"{v:.2f}" for v in st["crude_hospitalised_rate"]))
    print("  SINAN hospitalised rate, macro-region standardised " +
          " ".join(f"{v:.2f}" for v in st["standardised_hospitalised_rate"]))
    sv = cand["share_vs_incidence_across_health_regions"]
    print(f"  Spearman(hospitalisation share, incidence) across {sv['n_health_regions']} "
          f"health regions = {sv['spearman']:+.3f} (p={sv['p']:.2g})")
    wm = pl.DataFrame(cand["within_macro_region"])
    wm.write_csv(OUT / "hospitalised_rate_within_macro_region.csv")
    print("\n  SINAN hospitalised-case rate per 100k, by band within macro-region")
    for reg in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        s = wm.filter(pl.col("region") == reg).sort("band")
        if s.is_empty():
            continue
        print(f"    {reg:<13} " + "  ".join(
            f"Q{int(r['band'])} {r['hosp_rate_per_100k']:.2f}" for r in s.iter_rows(named=True)))

    plateau = plateau_contributors()
    plateau.write_csv(OUT / "q2_q3_plateau_contributors.csv")

    # ---- R --------------------------------------------------------------
    print("\n=== lme4 variance components and conditional logistic ===")
    report["r_models"] = run_r(frame_path)

    # ---- summary tables a reader will actually open ----------------------
    rows = []
    for outcome in ("died", "any_icu"):
        m, rm = mom[outcome], report["r_models"].get(outcome, {})
        vc = rm.get("crossed_home_region_and_hospital", {})
        rows.append({
            "outcome": outcome,
            "admissions": d.height,
            "events": int(d[outcome].sum()),
            "moment_between_region_tau_pp": m["between_region"]["tau_pct_points"],
            "moment_between_hospital_within_region_tau_pp":
                m["between_hospital_within_region"]["tau_pct_points"],
            "moment_variance_ratio_hospital_over_region":
                m["ratio_within_region_hospital_to_between_region"],
            "glmm_hospital_variance": vc.get("CNES", {}).get("variance"),
            "glmm_hospital_MOR": vc.get("CNES", {}).get("median_odds_ratio"),
            "glmm_home_region_variance": vc.get("hr_res", {}).get("variance"),
            "glmm_home_region_MOR": vc.get("hr_res", {}).get("median_odds_ratio"),
            "glmm_hospital_variance_share_pct":
                rm.get("variance_share_pct", {}).get("hospital"),
            "glmm_home_region_variance_share_pct":
                rm.get("variance_share_pct", {}).get("home_health_region"),
        })
    pl.DataFrame(rows).write_csv(OUT / "variance_decomposition.csv")

    a = report["rate_arithmetic"]
    pl.DataFrame({
        "band": [1, 2, 3, 4, 5],
        "person_years": a["person_years"],
        "sinan_incidence_per_100k": a["sinan_incidence_per_100k"],
        "hospitalisation_share": a["sinan_hospitalisation_share"],
        "sinan_hospitalised_rate_per_100k": a["sinan_hospitalised_rate_per_100k"],
        "sinan_hospitalised_lo": a["sinan_hospitalised_lo"],
        "sinan_hospitalised_hi": a["sinan_hospitalised_hi"],
        "sih_admission_rate_per_100k": a["sih_admission_rate_per_100k"],
        "sih_admission_lo": a["sih_admission_lo"],
        "sih_admission_hi": a["sih_admission_hi"],
        "sih_over_sinan_hospitalised": a["sih_over_sinan_hospitalised"],
        "macro_region_standardised_hospitalised_rate":
            cand["macro_region_standardised"]["standardised_hospitalised_rate"],
        "hospitals_per_million_py": cand["hospital_supply"]["hospitals_per_million_py"],
    }).write_csv(OUT / "admission_rate_arithmetic.csv")

    (OUT / "hospital_vs_territory_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
