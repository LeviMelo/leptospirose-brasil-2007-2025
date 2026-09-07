"""Does hospitalisation share measure surveillance DEPTH or case SEVERITY?

Two linked threads, both aimed at the paper's central result — that the share of
confirmed cases recorded as hospitalised orders health-region case fatality from
3.00% to 17.09% across case-weighted quintiles.

T-11. Depth and severity predict the same correlation. Two families of test
      separate them, and both rest on data the hospitalisation-share index does
      not touch:

      (a) The SINAN clinical block (CLI_ICTERI, CLI_RENAL, CLI_HEMORR,
          CLI_HEMOPU) marks the severe leptospirosis phenotype — Weil's
          syndrome and pulmonary haemorrhage. Splitting incidence into severe
          and non-severe halves asks WHERE the incidence deficit sits. Depth
          predicts the deficit is concentrated in the non-severe half (mild
          illness is what shallow surveillance misses). Genuine severity
          predicts the severe half is LARGER where hospitalisation share is
          high (more people are actually getting sicker).

      (b) SIH is hospital billing. It never sees SINAN. Its in-hospital
          fatality is a severity measurement with no notification step in it.
          Genuine severity predicts SINAN hospitalisation share tracks SIH
          in-hospital fatality. Depth predicts it instead tracks the ratio of
          SINAN notifications to SIH admissions, and leaves SIH in-hospital
          fatality alone.

T-1a. The 2.90 -> 1.81 sewer odds-ratio attenuation between the SINAN case
      fatality model and the SIH in-hospital fatality model was read as
      evidence for case-mix selection. NAMED THREAT: sanitation is a near
      perfect proxy for macro-region, so the attenuation may be nothing but two
      differently-confounded regional labels. Tested by asking whether the SIH
      sewer coefficient is spatially confounded in the same way the SINAN one
      is — nonparametrically, and in a quasi-binomial GLM with and without
      macro-region fixed effects, on common support and a common specification.

The GLMs here are NOT the INLA refits. They are unpenalised health-region-year
quasi-binomial fits whose purpose is the WITHIN-model contrast (with vs without
macro-region), and they are calibrated against the published INLA odds ratios
before that contrast is read.

Outputs to ``data/results/depth_vs_severity/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brepi.analysis.rates import binom_ci, poisson_ci  # noqa: E402
from brepi.config import PATHS  # noqa: E402

OUT = PATHS.results / "depth_vs_severity"
REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

#: Mirrors 40_ascertainment_depth.py so the bands are the published bands.
MIN_CASES = 30
MIN_KNOWN_OUTCOMES = 20
#: A health region contributing eight SIH admissions carries an in-hospital
#: fatality estimate whose sampling error dwarfs any gradient. Same logic as
#: MIN_CASES, applied to the independent system.
MIN_ADMISSIONS = 30

#: The severe leptospirosis phenotype. Jaundice and renal impairment are the
#: two halves of Weil's syndrome; haemorrhage and pulmonary haemorrhage are the
#: severe pulmonary form. Fever, myalgia and headache are deliberately NOT here
#: — they are present in nearly every notification and carry no severity signal.
SEVERE_FIELDS = ["cli_icteri", "cli_renal", "cli_hemorr", "cli_hemopu"]

# Import the banding function from the script that produced the central result,
# so the quintiles here are literally the published quintiles rather than a
# re-implementation that could drift.
_spec = importlib.util.spec_from_file_location(
    "_asc_depth", Path(__file__).with_name("40_ascertainment_depth.py"))
_asc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_asc)
case_weighted_quintiles = _asc._case_weighted_quintiles


# ---------------------------------------------------------------------------
# Interval helpers. The unpack order is (ESTIMATE, lo, hi); getting it wrong
# has silently corrupted results in this repo before, so every call is checked.
# ---------------------------------------------------------------------------
def prop(count, total) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, lo, hi = binom_ci(np.asarray(count, float), np.asarray(total, float))
    ok = np.isfinite(p)
    assert np.all(lo[ok] <= p[ok] + 1e-12) and np.all(p[ok] <= hi[ok] + 1e-12), \
        "binom_ci estimate outside its own interval — unpack order is wrong"
    return p, lo, hi


def rate(count, person_time, scale=1e5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, lo, hi = poisson_ci(np.asarray(count, float), np.asarray(person_time, float),
                           scale=scale)
    ok = np.isfinite(r)
    assert np.all(lo[ok] <= r[ok] + 1e-9) and np.all(r[ok] <= hi[ok] + 1e-9), \
        "poisson_ci estimate outside its own interval — unpack order is wrong"
    return r, lo, hi


def spearman(a, b) -> tuple[float, float, int]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    r = stats.spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_health_region() -> pl.DataFrame:
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    return hr.with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share"),
        (pl.col("deaths") / pl.col("rq4_outcome_known")).alias("cfr"),
    )


def load_clinical() -> pl.DataFrame:
    """Confirmed cases by health region, split on the severe phenotype.

    Residence municipality, matching the geography every other health-region
    quantity in this study uses.
    """
    keep = ["municipality_residence_code7", "src_year", "ate_hosp", "evolucao",
            "evolucao_state"] + SEVERE_FIELDS
    line = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter(pl.col("classi_fin") == "confirmado")
        .select(keep)
        .collect()
    )
    xwalk = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code"), pl.col("health_region_code"), pl.col("region"))
    line = line.join(xwalk, left_on="municipality_residence_code7",
                     right_on="munic_code", how="left")
    unmatched = line["health_region_code"].null_count()
    print(f"  line level: {line.height:,} confirmed cases, "
          f"{unmatched} without a health region "
          f"({100 * unmatched / line.height:.2f}%)")
    line = line.filter(pl.col("health_region_code").is_not_null())

    # "Severity known" means every one of the four markers was recorded as
    # yes/no. A case with a blank block is not a mild case; treating it as one
    # would manufacture exactly the gradient under test.
    line = line.with_columns(
        sev_known=pl.all_horizontal([pl.col(c).is_in(["sim", "nao"])
                                     for c in SEVERE_FIELDS]),
        severe=pl.any_horizontal([pl.col(c) == "sim" for c in SEVERE_FIELDS]),
        died=(pl.col("evolucao") == "obito_por_leptospirose"),
        outcome_ok=(pl.col("evolucao_state") == "valid"),
        hosp=(pl.col("ate_hosp") == "sim"),
    )
    sk, sv = pl.col("sev_known"), pl.col("severe")
    ok = pl.col("outcome_ok")
    return line.group_by("health_region_code").agg(
        pl.len().alias("ll_cases"),
        sk.sum().alias("sev_known"),
        (sk & sv).sum().alias("severe"),
        (sk & ~sv).sum().alias("nonsevere"),
        (sk & sv & ok).sum().alias("severe_outcome_known"),
        (sk & sv & ok & pl.col("died")).sum().alias("severe_deaths"),
        (sk & ~sv & ok).sum().alias("nonsevere_outcome_known"),
        (sk & ~sv & ok & pl.col("died")).sum().alias("nonsevere_deaths"),
        (sk & sv & pl.col("hosp")).sum().alias("severe_hosp"),
        (sk & ~sv & pl.col("hosp")).sum().alias("nonsevere_hosp"),
    )


def load_triangulation_hr() -> pl.DataFrame:
    """SINAN and SIH on the SAME municipality-years, restricted to SIH coverage.

    Restricting both sides to ``sih_covered`` is what makes the SINAN-to-SIH
    ratio a ratio of two things counted over identical territory-time.
    """
    tri = pl.read_parquet(PATHS.panel / "triangulation_municipality_year.parquet")
    cov = tri.filter(pl.col("sih_covered"))
    print(f"  SIH-covered municipality-years: {cov.height:,} "
          f"({cov['year'].min()}-{cov['year'].max()})")
    return cov.group_by("health_region_code").agg(
        pl.col("sinan_confirmed").sum().alias("w_sinan_confirmed"),
        pl.col("sinan_hospitalised").sum().alias("w_sinan_hospitalised"),
        pl.col("sinan_deaths").sum().alias("w_sinan_deaths"),
        pl.col("sih_a27_admissions").sum().alias("sih_adm"),
        pl.col("sih_a27_deaths_in_hospital").sum().alias("sih_deaths"),
        pl.col("population").sum().alias("w_person_years"),
    )


def health_region_year_covariates() -> pl.DataFrame:
    """Population-weighted structural covariates at health region-year.

    Reconstructed from the municipality-year atlas rather than taken from the
    fitted panel, so that the SINAN and SIH models below see one construct.
    Validated against ``rq4_lethality_panel.csv`` on the overlap.
    """
    my = pl.read_parquet(PATHS.results / "atlas" / "municipality_year_atlas.parquet")
    my = my.with_columns(
        gdp_per_capita_asinh=np.arcsinh(pl.col("gdp_per_capita") / 1e4))
    w = pl.col("population")
    return my.group_by(["health_region_code", "year"]).agg(
        ((pl.col("sanitation_sewer_share") * w).sum() / w.sum())
        .alias("sanitation_sewer_share"),
        ((pl.col("urban_share") * w).sum() / w.sum()).alias("urban_share"),
        ((pl.col("gdp_per_capita_asinh") * w).sum() / w.sum())
        .alias("gdp_per_capita_asinh"),
        w.sum().alias("population"),
        pl.col("region").first().alias("region"),
        pl.col("uf_abbr").first().alias("uf_abbr"),
    )


# ---------------------------------------------------------------------------
# Quasi-binomial GLM by IRLS. statsmodels is not in this environment; the model
# is four lines of Fisher scoring and the dispersion correction matters more
# than the solver.
# ---------------------------------------------------------------------------
def binom_glm(y, n, X, names) -> pl.DataFrame:
    y, n, X = np.asarray(y, float), np.asarray(n, float), np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / (n.sum() - y.sum() + 0.5))
    for _ in range(200):
        mu = np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 1e-10, 1 - 1e-10)
        W = n * mu * (1 - mu)
        z = X @ beta + (y - n * mu) / W
        new = np.linalg.solve((X.T * W) @ X, (X.T * W) @ z)
        if np.max(np.abs(new - beta)) < 1e-11:
            beta = new
            break
        beta = new
    mu = np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 1e-10, 1 - 1e-10)
    W = n * mu * (1 - mu)
    cov = np.linalg.inv((X.T * W) @ X)
    dof = len(y) - X.shape[1]
    # Health-region-years are grossly overdispersed relative to binomial; a
    # nominal binomial SE here would be a fiction. Pearson scale correction.
    phi = float(np.sum((y - n * mu) ** 2 / W) / dof)
    se = np.sqrt(np.diag(cov) * max(phi, 1.0))
    # A state dummy covering a stratum with no events is quasi-separated and
    # its interval runs to infinity. That is a property of that nuisance term,
    # not of the covariates being read, so it is clipped rather than allowed to
    # write `inf` into the results file.
    ex = lambda v: np.exp(np.clip(v, -50, 50))
    return pl.DataFrame({
        "term": names,
        "log_or": beta,
        "se": se,
        "or": ex(beta),
        "or_lo": ex(beta - 1.96 * se),
        "or_hi": ex(beta + 1.96 * se),
    }).with_columns(pl.lit(phi).alias("dispersion"), pl.lit(dof).alias("resid_df"))


def design(df: pl.DataFrame, covars: list[str], fe: str | None):
    """`fe` is None, "region" (5 macro-regions) or "uf" (27 states)."""
    X = [np.ones(df.height)] + [df[c].to_numpy().astype(float) for c in covars]
    names = ["(Intercept)"] + list(covars)
    if fe == "region":
        reg = df["region"].to_numpy()
        for r in REGIONS[1:]:                      # Norte is the reference
            X.append((reg == r).astype(float))
            names.append(f"region[{r}]")
    elif fe == "uf":
        uf = df["uf_abbr"].to_numpy()
        for u in sorted(set(uf.tolist()))[1:]:
            X.append((uf == u).astype(float))
            names.append(f"uf[{u}]")
    return np.column_stack(X), names


# ===========================================================================
# T-11
# ===========================================================================
def t11_severity_decomposition(hr: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Where does the incidence deficit sit — in mild illness or in severe?"""
    d = hr.filter(pl.col("cases") >= MIN_CASES)
    q = case_weighted_quintiles(d, "hosp_share")
    g = q.group_by("quintile").agg(
        pl.len().alias("health_regions"),
        pl.col("hosp_share").median().alias("band_median_hosp_share"),
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("rq4_outcome_known").sum().alias("outcome_known"),
        pl.col("person_years").sum().alias("person_years"),
        pl.col("sev_known").sum().alias("sev_known"),
        pl.col("severe").sum().alias("severe"),
        pl.col("nonsevere").sum().alias("nonsevere"),
        pl.col("severe_outcome_known").sum().alias("severe_outcome_known"),
        pl.col("severe_deaths").sum().alias("severe_deaths"),
        pl.col("nonsevere_outcome_known").sum().alias("nonsevere_outcome_known"),
        pl.col("nonsevere_deaths").sum().alias("nonsevere_deaths"),
        pl.col("severe_hosp").sum().alias("severe_hosp"),
        pl.col("nonsevere_hosp").sum().alias("nonsevere_hosp"),
        pl.col("ll_cases").sum().alias("ll_cases"),
    ).sort("quintile")

    py = g["person_years"].to_numpy()
    inc_all, inc_all_lo, inc_all_hi = rate(g["sev_known"], py)
    inc_sev, inc_sev_lo, inc_sev_hi = rate(g["severe"], py)
    inc_mild, inc_mild_lo, inc_mild_hi = rate(g["nonsevere"], py)
    sev_share, sev_lo, sev_hi = prop(g["severe"], g["sev_known"])
    rec, rec_lo, rec_hi = prop(g["sev_known"], g["ll_cases"])
    cfr_s, cfr_s_lo, cfr_s_hi = prop(g["severe_deaths"], g["severe_outcome_known"])
    cfr_m, cfr_m_lo, cfr_m_hi = prop(g["nonsevere_deaths"], g["nonsevere_outcome_known"])
    hs_s, _, _ = prop(g["severe_hosp"], g["severe"])
    hs_m, _, _ = prop(g["nonsevere_hosp"], g["nonsevere"])
    cfr_all, cfr_all_lo, cfr_all_hi = prop(g["deaths"], g["outcome_known"])

    # Guard: these must be the published bands. If the gradient does not
    # reproduce, every decomposition below is of some other quantity.
    assert abs(100 * cfr_all[0] - 3.00) < 0.05 and abs(100 * cfr_all[-1] - 17.09) < 0.05, \
        f"band reproduction failed: Q1={100 * cfr_all[0]:.2f}% Q5={100 * cfr_all[-1]:.2f}%"

    # Direct standardisation on the measured severe/non-severe split, with Q1's
    # severity mix as the standard. The distance between the crude gradient and
    # the standardised one is the part of the case-fatality gradient that the
    # MEASURED clinical case mix accounts for.
    w_s = float(g["severe"][0] / g["sev_known"][0])
    w_n = 1.0 - w_s
    std_cfr = w_s * cfr_s + w_n * cfr_m
    var = (w_s ** 2 * cfr_s * (1 - cfr_s) / g["severe_outcome_known"].to_numpy()
           + w_n ** 2 * cfr_m * (1 - cfr_m) / g["nonsevere_outcome_known"].to_numpy())
    std_lo, std_hi = std_cfr - 1.96 * np.sqrt(var), std_cfr + 1.96 * np.sqrt(var)

    tab = g.with_columns(
        pl.Series("cfr_all_pct", 100 * cfr_all),
        pl.Series("cfr_all_lo", 100 * cfr_all_lo),
        pl.Series("cfr_all_hi", 100 * cfr_all_hi),
        pl.Series("cfr_severity_standardised_pct", 100 * std_cfr),
        pl.Series("cfr_severity_standardised_lo", 100 * std_lo),
        pl.Series("cfr_severity_standardised_hi", 100 * std_hi),
        pl.Series("incidence_severity_known_per_100k", inc_all),
        pl.Series("incidence_severity_known_lo", inc_all_lo),
        pl.Series("incidence_severity_known_hi", inc_all_hi),
        pl.Series("incidence_severe_per_100k", inc_sev),
        pl.Series("incidence_severe_lo", inc_sev_lo),
        pl.Series("incidence_severe_hi", inc_sev_hi),
        pl.Series("incidence_nonsevere_per_100k", inc_mild),
        pl.Series("incidence_nonsevere_lo", inc_mild_lo),
        pl.Series("incidence_nonsevere_hi", inc_mild_hi),
        pl.Series("severe_share_pct", 100 * sev_share),
        pl.Series("severe_share_lo", 100 * sev_lo),
        pl.Series("severe_share_hi", 100 * sev_hi),
        pl.Series("severity_recorded_pct", 100 * rec),
        pl.Series("severity_recorded_lo", 100 * rec_lo),
        pl.Series("severity_recorded_hi", 100 * rec_hi),
        pl.Series("cfr_severe_pct", 100 * cfr_s),
        pl.Series("cfr_severe_lo", 100 * cfr_s_lo),
        pl.Series("cfr_severe_hi", 100 * cfr_s_hi),
        pl.Series("cfr_nonsevere_pct", 100 * cfr_m),
        pl.Series("cfr_nonsevere_lo", 100 * cfr_m_lo),
        pl.Series("cfr_nonsevere_hi", 100 * cfr_m_hi),
        pl.Series("hosp_share_severe_pct", 100 * hs_s),
        pl.Series("hosp_share_nonsevere_pct", 100 * hs_m),
    )

    summary = {
        "fold_drop_incidence_severe_Q1_over_Q5": float(inc_sev[0] / inc_sev[-1]),
        "fold_drop_incidence_nonsevere_Q1_over_Q5": float(inc_mild[0] / inc_mild[-1]),
        "fold_drop_incidence_all_Q1_over_Q5": float(inc_all[0] / inc_all[-1]),
        "nonsevere_to_severe_case_ratio_Q1": float(g["nonsevere"][0] / g["severe"][0]),
        "nonsevere_to_severe_case_ratio_Q5": float(g["nonsevere"][-1] / g["severe"][-1]),
        "severe_share_pct_Q1": float(100 * sev_share[0]),
        "severe_share_pct_Q5": float(100 * sev_share[-1]),
        "cfr_severe_pct_Q1": float(100 * cfr_s[0]),
        "cfr_severe_pct_Q5": float(100 * cfr_s[-1]),
        "cfr_nonsevere_pct_Q1": float(100 * cfr_m[0]),
        "cfr_nonsevere_pct_Q5": float(100 * cfr_m[-1]),
        "severity_recorded_pct_Q1": float(100 * rec[0]),
        "severity_recorded_pct_Q5": float(100 * rec[-1]),
        # The decomposition. Crude gradient = (case-mix part) x (within-stratum part).
        "cfr_gradient_crude_Q5_over_Q1": float(cfr_all[-1] / cfr_all[0]),
        "cfr_gradient_within_severe_Q5_over_Q1": float(cfr_s[-1] / cfr_s[0]),
        "cfr_gradient_within_nonsevere_Q5_over_Q1": float(cfr_m[-1] / cfr_m[0]),
        "cfr_gradient_severity_standardised_Q5_over_Q1": float(std_cfr[-1] / std_cfr[0]),
        "case_mix_share_of_log_gradient": float(
            1 - np.log(std_cfr[-1] / std_cfr[0]) / np.log(cfr_all[-1] / cfr_all[0])),
        "cfr_all_pct_Q1": float(100 * cfr_all[0]),
        "cfr_all_pct_Q5": float(100 * cfr_all[-1]),
        "cfr_standardised_pct_Q1": float(100 * std_cfr[0]),
        "cfr_standardised_pct_Q5": float(100 * std_cfr[-1]),
    }
    return tab, summary


def t11_sih(hr: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """The external test. SIH never sees SINAN."""
    d = hr.filter(pl.col("cases") >= MIN_CASES)
    q = case_weighted_quintiles(d, "hosp_share")
    g = q.group_by("quintile").agg(
        pl.len().alias("health_regions"),
        pl.col("hosp_share").median().alias("band_median_hosp_share"),
        pl.col("w_sinan_confirmed").sum().alias("sinan_confirmed"),
        pl.col("w_sinan_hospitalised").sum().alias("sinan_hospitalised"),
        pl.col("sih_adm").sum().alias("sih_adm"),
        pl.col("sih_deaths").sum().alias("sih_deaths"),
        pl.col("w_person_years").sum().alias("person_years"),
    ).sort("quintile")

    fat, fat_lo, fat_hi = prop(g["sih_deaths"], g["sih_adm"])
    adm, adm_lo, adm_hi = rate(g["sih_adm"], g["person_years"])
    sin, sin_lo, sin_hi = rate(g["sinan_confirmed"], g["person_years"])
    ratio = g["sinan_hospitalised"].to_numpy() / g["sih_adm"].to_numpy()

    tab = g.with_columns(
        pl.Series("sih_inhospital_fatality_pct", 100 * fat),
        pl.Series("sih_inhospital_fatality_lo", 100 * fat_lo),
        pl.Series("sih_inhospital_fatality_hi", 100 * fat_hi),
        pl.Series("sih_admission_rate_per_100k", adm),
        pl.Series("sih_admission_rate_lo", adm_lo),
        pl.Series("sih_admission_rate_hi", adm_hi),
        pl.Series("sinan_incidence_per_100k", sin),
        pl.Series("sinan_incidence_lo", sin_lo),
        pl.Series("sinan_incidence_hi", sin_hi),
        pl.Series("sinan_hosp_to_sih_adm_ratio", ratio),
    )

    # Region-level correlations, on regions with enough of BOTH systems.
    c = d.filter((pl.col("sih_adm") >= MIN_ADMISSIONS)
                 & (pl.col("w_sinan_confirmed") >= MIN_CASES))
    hs = c["hosp_share"].to_numpy()
    sih_fat = (c["sih_deaths"] / c["sih_adm"]).to_numpy()
    cap = (c["w_sinan_hospitalised"] / c["sih_adm"]).to_numpy()
    sih_rate = (c["sih_adm"] / c["w_person_years"]).to_numpy()
    sin_rate = (c["w_sinan_confirmed"] / c["w_person_years"]).to_numpy()

    # Within macro-region: the same test that demoted sanitation. If the SIH
    # signal only exists between macro-regions it is a regional label too.
    within = {}
    for reg in REGIONS:
        s = c.filter(pl.col("region") == reg)
        if s.height < 8:
            within[reg] = {"n": s.height, "note": "too few regions for a rank correlation"}
            continue
        within[reg] = {
            "n": s.height,
            "hosp_share_vs_sih_inhospital_fatality": spearman(
                s["hosp_share"].to_numpy(),
                (s["sih_deaths"] / s["sih_adm"]).to_numpy()),
            "hosp_share_vs_sinan_hosp_to_sih_adm_ratio": spearman(
                s["hosp_share"].to_numpy(),
                (s["w_sinan_hospitalised"] / s["sih_adm"]).to_numpy()),
            "hosp_share_vs_sih_admission_rate": spearman(
                s["hosp_share"].to_numpy(),
                (s["sih_adm"] / s["w_person_years"]).to_numpy()),
        }

    summary = {
        "n_health_regions_both_systems": c.height,
        "min_admissions": MIN_ADMISSIONS,
        "hosp_share_vs_sih_inhospital_fatality": spearman(hs, sih_fat),
        "hosp_share_vs_sinan_hosp_to_sih_adm_ratio": spearman(hs, cap),
        "hosp_share_vs_sih_admission_rate": spearman(hs, sih_rate),
        "hosp_share_vs_sinan_incidence": spearman(hs, sin_rate),
        "sinan_cfr_vs_sih_inhospital_fatality": spearman(
            (c["deaths"] / c["rq4_outcome_known"]).to_numpy(), sih_fat),
        "within_macro_region": within,
        "sih_fatality_pct_Q1": float(100 * fat[0]),
        "sih_fatality_pct_Q5": float(100 * fat[-1]),
        "sih_admission_rate_Q1": float(adm[0]),
        "sih_admission_rate_Q5": float(adm[-1]),
        "sinan_incidence_Q1": float(sin[0]),
        "sinan_incidence_Q5": float(sin[-1]),
        "capture_ratio_Q1": float(ratio[0]),
        "capture_ratio_Q5": float(ratio[-1]),
        "sih_fatality_gradient_Q5_over_Q1": float(fat[-1] / fat[0]),
        "sinan_incidence_gradient_Q1_over_Q5": float(sin[0] / sin[-1]),
        "sih_admission_rate_gradient_Q1_over_Q5": float(adm[0] / adm[-1]),
        # How much of each SINAN gradient does a system with no notification
        # step in it reproduce? Ratios of logs, because gradients multiply.
        "share_of_log_incidence_gradient_reproduced_by_sih": float(
            np.log(adm[0] / adm[-1]) / np.log(sin[0] / sin[-1])),
    }
    return tab, summary


def t11_verdict(sev_sum: dict, sih_sum: dict) -> dict:
    """The two mechanisms, side by side, on the log scale where they compose."""
    crude = sev_sum["cfr_gradient_crude_Q5_over_Q1"]
    sih = sih_sum["sih_fatality_gradient_Q5_over_Q1"]
    return {
        "crude_cfr_gradient_Q5_over_Q1": crude,
        "external_sih_fatality_gradient_Q5_over_Q1": sih,
        "share_of_log_cfr_gradient_reproduced_by_sih": float(np.log(sih) / np.log(crude)),
        "residual_gradient_not_seen_by_sih": float(crude / sih),
        "share_of_log_cfr_gradient_explained_by_measured_case_mix":
            sev_sum["case_mix_share_of_log_gradient"],
        "incidence_deficit_fold_severe": sev_sum["fold_drop_incidence_severe_Q1_over_Q5"],
        "incidence_deficit_fold_nonsevere":
            sev_sum["fold_drop_incidence_nonsevere_Q1_over_Q5"],
        "share_of_log_incidence_gradient_reproduced_by_sih":
            sih_sum["share_of_log_incidence_gradient_reproduced_by_sih"],
    }


def t1a_verdict(sew: pl.DataFrame) -> dict:
    """Attenuation of the sewer coefficient, by model and by spatial control."""
    def get(model: str, adj: str) -> dict:
        r = sew.filter((pl.col("model") == model) & (pl.col("adjustment") == adj))
        return {"or": float(r["or"][0]), "or_lo": float(r["or_lo"][0]),
                "or_hi": float(r["or_hi"][0]), "log_or": float(r["log_or"][0])}

    A = "SINAN case fatality (common support+spec)"
    B = "SIH in-hospital fatality (common support+spec)"
    out = {f"{m}|{a}": get(m, a)
           for m in sew["model"].unique().to_list()
           for a in ["no spatial FE", "macro-region FE", "state (UF) FE"]}
    out["published_inla_sinan_or"] = 2.9023193592139
    out["published_inla_sih_or"] = 1.81328035282044
    out["published_inla_attenuation_ratio"] = 2.9023193592139 / 1.81328035282044
    out["glm_common_support_attenuation_ratio"] = (
        get(A, "no spatial FE")["or"] / get(B, "no spatial FE")["or"])
    out["glm_spatial_absorption_sinan_log_pct"] = 100 * (
        1 - get(A, "state (UF) FE")["log_or"] / get(A, "no spatial FE")["log_or"])
    out["glm_spatial_absorption_sih_log_pct"] = 100 * (
        1 - get(B, "state (UF) FE")["log_or"] / get(B, "no spatial FE")["log_or"])
    return out


# ===========================================================================
# T-1a
# ===========================================================================
def t1a_nonparametric(hry: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Does the SIH sewer gradient vanish within macro-region, as SINAN's does?

    Health-region totals, banded on the population-weighted sewer share. Bands
    are weighted by each outcome's own denominator so every band carries equal
    estimation mass for that outcome.
    """
    def bands(df: pl.DataFrame, weight: str, k: int) -> pl.DataFrame:
        d = df.filter(pl.col(weight) > 0).sort("sewer")
        cw = (pl.col(weight).cum_sum() / pl.col(weight).sum())
        return d.with_columns((cw * k).ceil().clip(1, k).cast(pl.Int32).alias("band"))

    rows = []
    for scope in ["Brazil"] + REGIONS:
        sub = hry if scope == "Brazil" else hry.filter(pl.col("region") == scope)
        k = 5 if scope == "Brazil" else 3
        for label, num, den in [("SINAN case fatality", "deaths", "outcome_known"),
                                ("SIH in-hospital fatality", "sih_deaths", "sih_adm")]:
            b = bands(sub.filter(pl.col(den) >= 1), den, k)
            if b.height == 0:
                continue
            g = b.group_by("band").agg(
                pl.len().alias("health_regions"),
                pl.col("sewer").median().alias("band_median_sewer"),
                pl.col(num).sum().cast(pl.Int64).alias("numerator"),
                pl.col(den).sum().cast(pl.Int64).alias("denominator"),
            ).sort("band")
            p, lo, hi = prop(g["numerator"], g["denominator"])
            rows.append(g.with_columns(
                pl.lit(scope).alias("scope"), pl.lit(label).alias("outcome"),
                pl.Series("pct", 100 * p), pl.Series("pct_lo", 100 * lo),
                pl.Series("pct_hi", 100 * hi),
            ))
    grad = pl.concat(rows, how="diagonal")

    # Composition: is a sewer band a macro-region under another name, when the
    # weights are SIH admissions rather than SINAN cases?
    b = bands(hry.filter(pl.col("sih_adm") >= 1), "sih_adm", 5)
    comp = b.group_by(["band", "region"]).agg(pl.col("sih_adm").sum().alias("sih_adm"))
    comp = comp.with_columns(
        (100 * pl.col("sih_adm") / pl.col("sih_adm").sum().over("band"))
        .alias("pct_of_band")
    ).sort(["band", "pct_of_band"], descending=[False, True])

    # Rank correlations across health regions, mirroring 40_ascertainment_depth.
    corr = {}
    for scope in ["Brazil"] + REGIONS:
        sub = hry if scope == "Brazil" else hry.filter(pl.col("region") == scope)
        s_sin = sub.filter(pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES)
        s_sih = sub.filter(pl.col("sih_adm") >= MIN_ADMISSIONS)
        entry = {}
        if s_sin.height >= 8:
            entry["sewer_vs_sinan_cfr"] = spearman(
                s_sin["sewer"].to_numpy(),
                (s_sin["deaths"] / s_sin["outcome_known"]).to_numpy())
        if s_sih.height >= 8:
            entry["sewer_vs_sih_inhospital_fatality"] = spearman(
                s_sih["sewer"].to_numpy(),
                (s_sih["sih_deaths"] / s_sih["sih_adm"]).to_numpy())
        corr[scope] = entry
    return grad, comp, corr


def t1a_glm(sinan: pl.DataFrame, sih: pl.DataFrame,
            common: pl.DataFrame) -> pl.DataFrame:
    """Sewer odds ratio with and without macro-region, in both systems."""
    COV = ["sanitation_sewer_share", "urban_share", "gdp_per_capita_asinh"]
    out = []
    specs = [
        # (label, frame, numerator, denominator, covariates)
        ("SINAN case fatality (published spec)", sinan, "deaths", "outcome_known",
         COV + ["completeness_z"]),
        ("SIH in-hospital fatality (published spec)", sih, "sih_deaths", "sih_adm", COV),
        ("SINAN case fatality (common support+spec)", common, "deaths",
         "outcome_known", COV),
        ("SIH in-hospital fatality (common support+spec)", common, "sih_deaths",
         "sih_adm", COV),
    ]
    labels = {None: "no spatial FE", "region": "macro-region FE", "uf": "state (UF) FE"}
    for label, df, num, den, covars in specs:
        d = df.filter(pl.col(den) > 0)
        for fe in (None, "region", "uf"):
            X, names = design(d, covars, fe=fe)
            res = binom_glm(d[num].to_numpy(), d[den].to_numpy(), X, names)
            out.append(res.with_columns(
                pl.lit(label).alias("model"),
                pl.lit(labels[fe]).alias("adjustment"),
                pl.lit(d.height).alias("cells"),
                pl.lit(int(d[den].sum())).alias("denominator_total"),
                pl.lit(int(d[num].sum())).alias("numerator_total"),
            ))
    return pl.concat(out, how="diagonal")


# ===========================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print("=" * 74)
    print("loading")
    hr = load_health_region()
    clin = load_clinical()
    tri = load_triangulation_hr()
    hr = hr.join(clin, on="health_region_code", how="left") \
           .join(tri, on="health_region_code", how="left")

    # Guard: the line-level confirmed count must agree with the atlas case
    # count, or the severity split is being applied to a different case series.
    agree = hr.select(
        (pl.col("ll_cases").sum() / pl.col("cases").sum()).alias("r")).item()
    print(f"  line-level cases / atlas cases = {agree:.4f}")
    assert 0.97 < agree < 1.03, "line level and atlas disagree on the case series"

    # ---- T-11 ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("T-11  severity decomposition of the incidence deficit")
    sev_tab, sev_sum = t11_severity_decomposition(hr)
    sev_tab.write_csv(OUT / "t11a_severity_decomposition.csv")
    for r in sev_tab.iter_rows(named=True):
        print(f"  Q{r['quintile']}  hosp_share={r['band_median_hosp_share']:.2f}  "
              f"inc_severe={r['incidence_severe_per_100k']:5.2f} "
              f"({r['incidence_severe_lo']:.2f}-{r['incidence_severe_hi']:.2f})  "
              f"inc_nonsevere={r['incidence_nonsevere_per_100k']:5.2f} "
              f"({r['incidence_nonsevere_lo']:.2f}-{r['incidence_nonsevere_hi']:.2f})  "
              f"severe_share={r['severe_share_pct']:5.1f}%  "
              f"recorded={r['severity_recorded_pct']:5.1f}%")
    print(f"  Q1/Q5 fold drop: severe {sev_sum['fold_drop_incidence_severe_Q1_over_Q5']:.2f}x"
          f"  non-severe {sev_sum['fold_drop_incidence_nonsevere_Q1_over_Q5']:.2f}x")
    print("  case fatality, crude and within clinical stratum:")
    for r in sev_tab.iter_rows(named=True):
        print(f"    Q{r['quintile']}  crude {r['cfr_all_pct']:6.2f}% "
              f"({r['cfr_all_lo']:.2f}-{r['cfr_all_hi']:.2f})   "
              f"severe {r['cfr_severe_pct']:6.2f}% "
              f"({r['cfr_severe_lo']:.2f}-{r['cfr_severe_hi']:.2f})   "
              f"non-severe {r['cfr_nonsevere_pct']:5.2f}% "
              f"({r['cfr_nonsevere_lo']:.2f}-{r['cfr_nonsevere_hi']:.2f})   "
              f"mix-standardised {r['cfr_severity_standardised_pct']:6.2f}% "
              f"({r['cfr_severity_standardised_lo']:.2f}-"
              f"{r['cfr_severity_standardised_hi']:.2f})")
    print(f"  gradient Q5/Q1: crude "
          f"{sev_sum['cfr_gradient_crude_Q5_over_Q1']:.2f}x  ->  "
          f"severity-standardised "
          f"{sev_sum['cfr_gradient_severity_standardised_Q5_over_Q1']:.2f}x  "
          f"(measured case mix accounts for "
          f"{100 * sev_sum['case_mix_share_of_log_gradient']:.0f}% of the log gradient)")
    report["T11_severity_decomposition"] = sev_sum

    print("\n" + "-" * 74)
    print("T-11  the external test: SIH, which never sees SINAN")
    sih_tab, sih_sum = t11_sih(hr)
    sih_tab.write_csv(OUT / "t11b_sih_by_hospshare_band.csv")
    for r in sih_tab.iter_rows(named=True):
        print(f"  Q{r['quintile']}  hosp_share={r['band_median_hosp_share']:.2f}  "
              f"SIH fatality={r['sih_inhospital_fatality_pct']:5.2f}% "
              f"({r['sih_inhospital_fatality_lo']:.2f}-{r['sih_inhospital_fatality_hi']:.2f})  "
              f"SIH adm rate={r['sih_admission_rate_per_100k']:5.2f}  "
              f"SINAN inc={r['sinan_incidence_per_100k']:5.2f}  "
              f"SINAN-hosp/SIH-adm={r['sinan_hosp_to_sih_adm_ratio']:5.2f}")
    for k in ["hosp_share_vs_sih_inhospital_fatality",
              "hosp_share_vs_sinan_hosp_to_sih_adm_ratio",
              "hosp_share_vs_sih_admission_rate",
              "hosp_share_vs_sinan_incidence",
              "sinan_cfr_vs_sih_inhospital_fatality"]:
        r, p, n = sih_sum[k]
        print(f"  spearman {k:<45} {r:+.3f} (p={p:.2g}, n={n})")
    print("  within macro-region (hosp share vs the SIH quantities):")
    for reg, v in sih_sum["within_macro_region"].items():
        if "note" in v:
            print(f"    {reg:<13} n={v['n']:>3}  {v['note']}")
            continue
        print(f"    {reg:<13} n={v['n']:>3}  "
              f"SIH fatality {v['hosp_share_vs_sih_inhospital_fatality'][0]:+.3f}   "
              f"SINAN-hosp/SIH-adm "
              f"{v['hosp_share_vs_sinan_hosp_to_sih_adm_ratio'][0]:+.3f}   "
              f"SIH adm rate {v['hosp_share_vs_sih_admission_rate'][0]:+.3f}")
    report["T11_sih_external"] = sih_sum
    verdict = t11_verdict(sev_sum, sih_sum)
    report["T11_verdict"] = verdict
    print("\n  T-11 verdict, on the log scale where gradients compose:")
    print(f"    crude case-fatality gradient          {verdict['crude_cfr_gradient_Q5_over_Q1']:.2f}x")
    print(f"    reproduced by SIH (no SINAN in it)    "
          f"{verdict['external_sih_fatality_gradient_Q5_over_Q1']:.2f}x  "
          f"= {100 * verdict['share_of_log_cfr_gradient_reproduced_by_sih']:.0f}% of the log gradient")
    print(f"    residual, invisible to SIH            "
          f"{verdict['residual_gradient_not_seen_by_sih']:.2f}x")
    print(f"    SINAN incidence gradient reproduced by SIH admissions: "
          f"{100 * verdict['share_of_log_incidence_gradient_reproduced_by_sih']:.0f}% "
          f"of the log gradient")

    # ---- T-1a ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("T-1a  is the SIH sewer coefficient spatially confounded too?")
    cov = health_region_year_covariates()

    lp = pl.read_csv(PATHS.results / "rq4_lethality" / "rq4_lethality_panel.csv").select(
        pl.col("health_region_code").cast(pl.Utf8), pl.col("year").cast(pl.Int32),
        pl.col("deaths").cast(pl.Int64), pl.col("outcome_known").cast(pl.Int64),
        pl.col("cases").cast(pl.Int64))
    tri_hry = (
        pl.read_parquet(PATHS.panel / "triangulation_municipality_year.parquet")
        .filter(pl.col("sih_covered"))
        .group_by(["health_region_code", "year"])
        .agg(pl.col("sih_a27_admissions").sum().alias("sih_adm"),
             pl.col("sih_a27_deaths_in_hospital").sum().alias("sih_deaths"))
        .with_columns(pl.col("year").cast(pl.Int32))
    )
    cov = cov.with_columns(pl.col("year").cast(pl.Int32))

    # Validate the reconstructed covariates against the ones the fitted model
    # saw. If these disagree, every comparison below is against a straw man.
    chk = lp.join(cov, on=["health_region_code", "year"], how="inner").join(
        pl.read_csv(PATHS.results / "rq4_lethality" / "rq4_lethality_panel.csv").select(
            pl.col("health_region_code").cast(pl.Utf8),
            pl.col("year").cast(pl.Int32),
            pl.col("sanitation_sewer_share").alias("sewer_fitted")),
        on=["health_region_code", "year"], how="inner")
    dif = float(np.max(np.abs(chk["sanitation_sewer_share"].to_numpy()
                              - chk["sewer_fitted"].to_numpy())))
    cr = float(np.corrcoef(chk["sanitation_sewer_share"].to_numpy(),
                           chk["sewer_fitted"].to_numpy())[0, 1])
    print(f"  reconstructed vs fitted sewer share: r={cr:.5f}, max|diff|={dif:.4f}, "
          f"n={chk.height}")
    report["T1a_covariate_reconstruction"] = {
        "pearson_r_vs_fitted_panel": cr, "max_abs_diff": dif, "n_cells": chk.height}

    sinan_hry = lp.join(cov, on=["health_region_code", "year"], how="inner")
    sinan_hry = sinan_hry.with_columns(
        completeness_z=((pl.col("outcome_known") / pl.col("cases"))
                        - (pl.col("outcome_known") / pl.col("cases")).mean())
        / (pl.col("outcome_known") / pl.col("cases")).std())
    sih_hry = tri_hry.join(cov, on=["health_region_code", "year"], how="inner") \
                     .filter(pl.col("sih_adm") > 0)
    common_hry = sinan_hry.join(
        tri_hry, on=["health_region_code", "year"], how="inner").filter(
        (pl.col("sih_adm") > 0) & (pl.col("outcome_known") > 0))
    print(f"  cells: SINAN {sinan_hry.height}, SIH {sih_hry.height}, "
          f"common support {common_hry.height}")

    glm = t1a_glm(sinan_hry, sih_hry, common_hry)
    glm.write_csv(OUT / "t1a_glm_sewer_odds.csv")
    print("\n  sewer-share odds ratio (per full 0->1 range), quasi-binomial GLM:")
    sew = glm.filter(pl.col("term") == "sanitation_sewer_share")
    for r in sew.iter_rows(named=True):
        print(f"    {r['model']:<46} {r['adjustment']:<21} "
              f"OR {r['or']:5.2f} ({r['or_lo']:.2f}-{r['or_hi']:.2f})  "
              f"cells={r['cells']} phi={r['dispersion']:.2f}")
    report["T1a_glm_sewer"] = sew.select(
        ["model", "adjustment", "or", "or_lo", "or_hi", "log_or", "se",
         "cells", "numerator_total", "denominator_total", "dispersion"]).to_dicts()
    tv = t1a_verdict(sew)
    report["T1a_verdict"] = tv
    print(f"\n  published INLA attenuation 2.90 / 1.81 = "
          f"{tv['published_inla_attenuation_ratio']:.2f}x")
    print(f"  same two outcomes, common support, common spec, no spatial field: "
          f"{tv['glm_common_support_attenuation_ratio']:.2f}x")
    print(f"  sewer log-OR absorbed by state fixed effects: SINAN "
          f"{tv['glm_spatial_absorption_sinan_log_pct']:.0f}%, SIH "
          f"{tv['glm_spatial_absorption_sih_log_pct']:.0f}%")

    # Health-region totals for the nonparametric arm.
    hry_tot = (
        sinan_hry.join(tri_hry, on=["health_region_code", "year"],
                       how="full", coalesce=True)
        .join(cov.select(["health_region_code", "year", "region",
                          "sanitation_sewer_share", "population"]),
              on=["health_region_code", "year"], how="left")
        .group_by("health_region_code").agg(
            pl.col("deaths").sum().alias("deaths"),
            pl.col("outcome_known").sum().alias("outcome_known"),
            pl.col("sih_deaths").sum().alias("sih_deaths"),
            pl.col("sih_adm").sum().alias("sih_adm"),
            ((pl.col("sanitation_sewer_share") * pl.col("population")).sum()
             / pl.col("population").sum()).alias("sewer"),
            pl.col("region").drop_nulls().first().alias("region"),
        ).filter(pl.col("sewer").is_not_null() & pl.col("region").is_not_null())
        .fill_null(0)
    )
    grad, comp, corr = t1a_nonparametric(hry_tot)
    grad.write_csv(OUT / "t1a_sewer_gradient_by_scope.csv")
    comp.write_csv(OUT / "t1a_sih_sewer_band_regional_composition.csv")

    print("\n  sewer-band gradients (Brazil quintiles, within-region tertiles):")
    for scope in ["Brazil"] + REGIONS:
        for outcome in ["SINAN case fatality", "SIH in-hospital fatality"]:
            s = grad.filter((pl.col("scope") == scope) & (pl.col("outcome") == outcome))
            if s.height == 0:
                continue
            cells = "  ".join(f"{v:5.2f}%" for v in s["pct"])
            lo, hi = s["pct"][0], s["pct"][-1]
            print(f"    {scope:<13} {outcome:<26} {cells}   "
                  f"top/bottom {hi / lo:.2f}x  (n_hr={s['health_regions'].sum()})")
    print("\n  SIH-admission-weighted sewer quintiles, macro-region composition:")
    for band in range(1, 6):
        s = comp.filter(pl.col("band") == band).head(1)
        print(f"    band {band}: {s['region'][0]} {s['pct_of_band'][0]:.1f}% of admissions")
    print("\n  rank correlations across health regions:")
    for scope, e in corr.items():
        parts = "  ".join(
            f"{k.replace('sewer_vs_', '')} {v[0]:+.3f} (n={v[2]})" for k, v in e.items())
        print(f"    {scope:<13} {parts}")
    report["T1a_within_region_correlations"] = corr
    report["T1a_gradient_summary"] = {
        f"{r['scope']}|{r['outcome']}": {
            "bands_pct": list(grad.filter((pl.col("scope") == r["scope"])
                                          & (pl.col("outcome") == r["outcome"]))["pct"]),
        } for r in grad.unique(subset=["scope", "outcome"]).iter_rows(named=True)
    }
    report["T1a_sih_band_composition"] = comp.head(20).to_dicts()

    (OUT / "depth_vs_severity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
