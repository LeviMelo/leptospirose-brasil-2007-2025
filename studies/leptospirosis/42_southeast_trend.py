"""T-6 - Is the national leptospirosis incidence decline epidemiology or surveillance?

The national AAPC in ``T3_trend.csv`` (-2.24%/yr) is carried entirely by the
Southeast (-3.39%/yr, Mann-Kendall p = 0.005); every other region's interval
covers zero and Centro-Oeste's point estimate is positive. Before that number
can be read as "leptospirosis is receding in Brazil", one threat has to be
named and killed:

    THREAT. The Southeast decline is a decline in case-FINDING, not in disease.

The decisive evidence is triangulation. SINAN needs a clinician to suspect
leptospirosis and somebody to file a notification form. SIM needs a death and
a coder writing CID-10 A27 as underlying cause. SIH needs a public-hospital
admission and a billing code. The three fail independently. If SINAN falls
while SIM and SIH hold, the fall is in the notification apparatus. If all three
fall together, the fall is in the disease.

This script

1.  decomposes the Southeast trend by state and by health region, and asks
    whether it survives removing any single state (leave-one-out refit);
2.  fits the same log-linear quasi-Poisson AAPC to SINAN confirmed cases, SIM
    A27 deaths and SIH A27 admissions inside the window where all three systems
    are published (2008-2024), and then fits the two RATIO models -- confirmed
    cases per A27 death, confirmed cases per A27 admission -- which are the
    direct tests of differential ascertainment;
3.  tracks the surveillance markers that a case-finding contraction would move:
    notification volume, laboratory-confirmation share among confirmed cases,
    hospitalisation share among confirmed cases, outcome completeness.

Geography. The three-system comparison runs on
``data/panel/triangulation_municipality_year.parquet``, where SINAN, SIM and
SIH are all on municipality of RESIDENCE. The analytic exposure panel assigns
SINAN cases to probable municipality of infection; mixing that with
residence-based SIM/SIH would manufacture cross-system discordance. At the
scale of a macro-region the difference is small, but it is free to avoid.

Model. Identical to ``R/11_tables.R::trend_apc``, reimplemented here because
statsmodels is not in this environment: log-linear quasi-Poisson in calendar
year with a log person-time offset, AAPC = 100 * (exp(beta) - 1), Wald interval
on the log scale with a normal quantile, dispersion from the Pearson statistic.
The implementation is validated against R's own ``glm(family=quasipoisson)`` on
the actual series before any result is written; the script aborts if it drifts.

Run:
    PYTHONPATH=. python studies/leptospirosis/42_southeast_trend.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.analysis.rates import binom_ci, poisson_ci

ROOT = Path(__file__).resolve().parents[2]
TRI = ROOT / "data/panel/triangulation_municipality_year.parquet"
LINE = ROOT / "data/interim/lept_line_level.parquet"
ATLAS_MY = ROOT / "data/results/atlas/municipality_year_atlas.parquet"
T3 = ROOT / "data/results/01_descriptive/T3_trend.csv"
T9 = ROOT / "data/results/01_descriptive/T9_confirmation_criterion.csv"
OUT = ROOT / "data/results/southeast_trend"

RSCRIPT = r"C:/Program Files/R/R-4.4.1/bin/Rscript.exe"

# Source windows, restated from 18_build_triangulation_panel.py. Nothing is
# fitted outside the window where its source system is published.
SINAN_YEARS = (2007, 2025)
SIM_YEARS = (2007, 2024)
SIH_YEARS = (2008, 2024)
COMMON = (2008, 2024)          # all three systems published
Z = float(stats.norm.ppf(0.975))

SE_UFS = ["SP", "RJ", "MG", "ES"]


def rule(s: str) -> None:
    print("\n" + "-" * 74 + "\n" + s)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _irls_poisson(y: np.ndarray, X: np.ndarray, offset: np.ndarray,
                  tol: float = 1e-11, maxit: int = 100):
    """Poisson log-link IRLS with an offset. Returns (beta, mu, XtWX)."""
    mu = y + 0.1
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    dev_old = np.inf
    for _ in range(maxit):
        w = mu
        z = eta - offset + (y - mu) / mu
        XtW = X.T * w
        XtWX = XtW @ X
        beta = np.linalg.solve(XtWX, XtW @ z)
        eta = X @ beta + offset
        eta = np.clip(eta, -700, 700)
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        dev = 2.0 * np.sum(term - (y - mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev
    w = mu
    XtWX = (X.T * w) @ X
    return beta, mu, XtWX


def qpois_aapc(years, events, person_time, *, conf: float = 0.95) -> dict:
    """Average annual percent change from a log-linear quasi-Poisson rate model.

    ``events`` are counts; ``person_time`` is the offset denominator (any
    proportional unit -- the slope is invariant to its scale). Returns the
    AAPC with a Wald interval on the log scale, the Pearson dispersion, and the
    Mann-Kendall statistic on the annual rate series.
    """
    yr = np.asarray(years, dtype=float)
    y = np.asarray(events, dtype=float)
    pt = np.asarray(person_time, dtype=float)
    keep = pt > 0
    yr, y, pt = yr[keep], y[keep], pt[keep]
    order = np.argsort(yr)
    yr, y, pt = yr[order], y[order], pt[order]
    n = yr.size
    out = {"n_years": int(n), "events": float(y.sum())}
    if n < 4 or y.sum() == 0:
        out.update(aapc=np.nan, aapc_lo=np.nan, aapc_hi=np.nan,
                   dispersion=np.nan, mk_tau=np.nan, mk_p=np.nan)
        return out
    # Centring the year changes the intercept, never the slope; it keeps
    # X'WX well conditioned when the offset spans eight orders of magnitude.
    X = np.column_stack([np.ones(n), yr - yr.mean()])
    beta, mu, XtWX = _irls_poisson(y, X, np.log(pt))
    resid_p = (y - mu) / np.sqrt(mu)
    dispersion = float(np.sum(resid_p ** 2) / (n - X.shape[1]))
    cov = dispersion * np.linalg.inv(XtWX)
    b = float(beta[1])
    se = float(np.sqrt(cov[1, 1]))
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    mk = mann_kendall(y / pt)
    out.update(aapc=100 * (np.exp(b) - 1),
               aapc_lo=100 * (np.exp(b - z * se) - 1),
               aapc_hi=100 * (np.exp(b + z * se) - 1),
               dispersion=dispersion, mk_tau=mk[0], mk_p=mk[1])
    return out


def mann_kendall(x) -> tuple[float, float]:
    """Mann-Kendall tau and two-sided p with the tie correction (R parity)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan"), float("nan")
    d = x[:, None] - x[None, :]
    s = float(np.sum(np.sign(d[np.tril_indices(n, -1)])))
    _, counts = np.unique(x, return_counts=True)
    vs = (n * (n - 1) * (2 * n + 5)
          - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(vs)
    elif s < 0:
        z = (s + 1) / np.sqrt(vs)
    else:
        z = 0.0
    return s / (0.5 * n * (n - 1)), float(2 * stats.norm.cdf(-abs(z)))


def qbinom_trend(years, success, total, *, conf: float = 0.95) -> dict:
    """Annual odds ratio from a quasi-binomial logistic trend in calendar year.

    A proportion (laboratory share, hospitalisation share, completeness) is
    grouped binomial data; the year-to-year variation is far wider than
    binomial, so the dispersion is estimated rather than fixed at one. The
    reported quantity is an ODDS RATIO per calendar year, not a risk ratio.
    """
    yr = np.asarray(years, dtype=float)
    k = np.asarray(success, dtype=float)
    m = np.asarray(total, dtype=float)
    keep = m > 0
    yr, k, m = yr[keep], k[keep], m[keep]
    order = np.argsort(yr)
    yr, k, m = yr[order], k[order], m[order]
    n = yr.size
    out = {"n_years": int(n), "success": float(k.sum()), "total": float(m.sum())}
    if n < 4:
        out.update(or_year=np.nan, or_lo=np.nan, or_hi=np.nan,
                   dispersion=np.nan, mk_tau=np.nan, mk_p=np.nan)
        return out
    X = np.column_stack([np.ones(n), yr - yr.mean()])
    p = (k + 0.5) / (m + 1.0)
    eta = np.log(p / (1 - p))
    beta = np.zeros(2)
    for _ in range(200):
        w = m * p * (1 - p)
        z = eta + (k / m - p) / (p * (1 - p))
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-12:
            beta = beta_new
            break
        beta = beta_new
        eta = np.clip(X @ beta, -700, 700)
        p = 1.0 / (1.0 + np.exp(-eta))
    w = m * p * (1 - p)
    XtWX = (X.T * w) @ X
    resid_p = (k - m * p) / np.sqrt(w)
    dispersion = float(np.sum(resid_p ** 2) / (n - 2))
    cov = dispersion * np.linalg.inv(XtWX)
    b, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    mk = mann_kendall(k / m)
    out.update(or_year=float(np.exp(b)), or_lo=float(np.exp(b - z * se)),
               or_hi=float(np.exp(b + z * se)), dispersion=dispersion,
               mk_tau=mk[0], mk_p=mk[1])
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def check_ci_helpers() -> None:
    """The (estimate, lo, hi) contract. This repo has been bitten by it once."""
    est, lo, hi = poisson_ci([1817], [83107250.0], scale=1e5)
    assert lo[0] <= est[0] <= hi[0], (est, lo, hi)
    assert abs(est[0] - 1817 / 83107250.0 * 1e5) < 1e-9
    p, plo, phi = binom_ci([300], [1000])
    assert plo[0] <= p[0] <= phi[0] and abs(p[0] - 0.3) < 1e-12
    print("CI helpers return (estimate, lower, upper) - verified")


def validate_against_r(series: pl.DataFrame) -> dict:
    """Refit the Southeast SINAN series in R and demand agreement.

    The AAPC engine above is a reimplementation of the one that produced
    T3_trend. If it disagreed with R by even a percent, every number in this
    file would be uncheckable against the rest of the study.
    """
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "series.csv"
        series.write_csv(csv)
        rs = Path(td) / "fit.R"
        rs.write_text(
            'd <- read.csv("%s")\n'
            'f <- glm(events ~ year + offset(log(person_time)), '
            'family = quasipoisson(), data = d)\n'
            'co <- summary(f)$coefficients["year", ]\n'
            'cat(sprintf("%%.12f %%.12f %%.12f\\n", co[1], co[2], '
            'summary(f)$dispersion))\n' % csv.as_posix(), encoding="utf-8")
        res = subprocess.run([RSCRIPT, "--vanilla", str(rs)],
                             capture_output=True, text=True, cwd=str(ROOT))
        if res.returncode != 0:
            raise RuntimeError("R validation failed:\n" + res.stderr)
        b_r, se_r, disp_r = (float(v) for v in res.stdout.split())
    mine = qpois_aapc(series["year"], series["events"], series["person_time"])
    aapc_r = 100 * (np.exp(b_r) - 1)
    lo_r = 100 * (np.exp(b_r - Z * se_r) - 1)
    hi_r = 100 * (np.exp(b_r + Z * se_r) - 1)
    drift = max(abs(mine["aapc"] - aapc_r), abs(mine["aapc_lo"] - lo_r),
                abs(mine["aapc_hi"] - hi_r),
                abs(mine["dispersion"] - disp_r) / disp_r)
    print(f"R glm(quasipoisson): AAPC {aapc_r:+.4f} ({lo_r:+.4f}, {hi_r:+.4f}) "
          f"dispersion {disp_r:.3f}")
    print(f"python IRLS        : AAPC {mine['aapc']:+.4f} "
          f"({mine['aapc_lo']:+.4f}, {mine['aapc_hi']:+.4f}) "
          f"dispersion {mine['dispersion']:.3f}")
    if drift > 1e-6:
        raise AssertionError(f"python AAPC drifts from R by {drift:.2e}")
    print("python AAPC engine matches R to < 1e-6 - proceeding")
    return {"r_aapc": aapc_r, "r_lo": lo_r, "r_hi": hi_r,
            "python_aapc": mine["aapc"], "max_abs_drift": drift}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_triangulation() -> pl.DataFrame:
    d = pl.read_parquet(TRI)
    # Null out anything the coverage flags say was never published, so a
    # groupby sum cannot turn "system absent" into "zero events".
    return d.with_columns(
        pl.when(pl.col("sim_covered")).then(pl.col("sim_a27_deaths"))
        .otherwise(None).alias("sim_a27_deaths"),
        pl.when(pl.col("sih_covered")).then(pl.col("sih_a27_admissions"))
        .otherwise(None).alias("sih_a27_admissions"),
        pl.when(pl.col("sih_covered")).then(pl.col("sih_a27_principal"))
        .otherwise(None).alias("sih_a27_principal"),
    )


def annual(d: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    keys = (by or []) + ["year"]
    return d.group_by(keys).agg(
        pl.col("sinan_confirmed").sum().alias("sinan"),
        pl.col("sinan_deaths").sum().alias("sinan_deaths"),
        pl.col("sinan_hospitalised").sum().alias("sinan_hosp"),
        pl.col("sim_a27_deaths").sum().alias("sim"),
        pl.col("sih_a27_admissions").sum().alias("sih"),
        pl.col("sih_a27_principal").sum().alias("sih_principal"),
        pl.col("population").sum().alias("population"),
    ).sort(keys)


def load_line() -> pl.DataFrame:
    """Confirmed-case line list with the study's own case definition.

    Mirrors 22_paper_descriptives.R exactly: CLASSI_FIN == confirmado, onset
    inside 2007-2025 with notification date as fallback, region from residence
    with notification UF as fallback.
    """
    cols = ["src_year", "classi_fin", "criterio", "DT_SIN_PRI", "DT_NOTIFIC",
            "municipality_residence_region", "uf_notification_region",
            "ate_hosp", "evolucao", "evolucao_state", "uf_residence_abbr",
            "uf_notification_abbr"]
    d = pl.read_parquet(LINE, columns=cols)
    to_date = lambda c: (pl.col(c).str.strptime(pl.Date, strict=False)
                         if d.schema[c] == pl.Utf8 else pl.col(c).cast(pl.Date))
    d = d.with_columns(to_date("DT_SIN_PRI").alias("onset"),
                       to_date("DT_NOTIFIC").alias("notified"))
    lo, hi = pl.date(2007, 1, 1), pl.date(2025, 12, 31)
    in_window = (
        pl.when(pl.col("onset").is_not_null())
        .then(pl.col("onset").is_between(lo, hi))
        .otherwise(pl.col("notified").is_not_null()
                   & pl.col("notified").is_between(lo, hi)))
    return d.with_columns(
        in_window.alias("in_window"),
        pl.coalesce(["municipality_residence_region",
                     "uf_notification_region"]).alias("region"),
        pl.coalesce(["uf_residence_abbr", "uf_notification_abbr"]).alias("uf"),
        pl.col("src_year").cast(pl.Int32).alias("year"),
    ).with_columns(
        ((pl.col("classi_fin") == "confirmado") & pl.col("in_window"))
        .fill_null(False).alias("is_case"))


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def block_regional(tri: pl.DataFrame) -> pl.DataFrame:
    """Reproduce the regional AAPC on residence geography, then decompose."""
    rows = []
    nat = annual(tri)
    for label, sub in [("national", nat)]:
        rows.append({"scope": label, "window": "2007-2025",
                     **qpois_aapc(sub["year"], sub["sinan"], sub["population"])})
    for reg, sub in annual(tri, ["region"]).group_by("region"):
        rows.append({"scope": f"region: {reg[0]}", "window": "2007-2025",
                     **qpois_aapc(sub["year"], sub["sinan"], sub["population"])})
    # Named threat: SINAN 2025 is the most recent file and may be provisional;
    # if the whole Southeast signal rests on it, it is not a trend.
    for label, yrs in [("2007-2024 (drop provisional 2025)", (2007, 2024)),
                       ("2007-2025 excl. 2020-21 COVID trough", None)]:
        se = annual(tri.filter(pl.col("region") == "Sudeste"))
        se_f = (se.filter(pl.col("year") <= 2024) if yrs
                else se.filter(~pl.col("year").is_in([2020, 2021])))
        rows.append({"scope": "region: Sudeste", "window": label,
                     **qpois_aapc(se_f["year"], se_f["sinan"],
                                  se_f["population"])})
    return pl.DataFrame(rows)


def block_uf(tri: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    se = tri.filter(pl.col("region") == "Sudeste")
    by_uf = annual(se, ["uf_abbr"])
    rows = []
    for uf, sub in by_uf.group_by("uf_abbr"):
        est, lo, hi = poisson_ci([float(sub["sinan"].sum())],
                                 [float(sub["population"].sum())], scale=1e5)
        assert lo[0] <= est[0] <= hi[0]
        rows.append({"uf": uf[0], "cases_2007_2025": int(sub["sinan"].sum()),
                     "incidence_per_100k_py": float(est[0]),
                     "incidence_lo": float(lo[0]), "incidence_hi": float(hi[0]),
                     **qpois_aapc(sub["year"], sub["sinan"], sub["population"])})
    uf_tbl = pl.DataFrame(rows).sort("uf")

    # Leave-one-out: does the Southeast decline survive dropping each state?
    loo = []
    for uf in SE_UFS:
        sub = annual(se.filter(pl.col("uf_abbr") != uf))
        loo.append({"dropped": uf,
                    **qpois_aapc(sub["year"], sub["sinan"], sub["population"])})
    full = annual(se)
    loo.append({"dropped": "(none)",
                **qpois_aapc(full["year"], full["sinan"], full["population"])})
    return uf_tbl, pl.DataFrame(loo)


def block_contribution(tri: pl.DataFrame) -> pl.DataFrame:
    """Where did the missing cases go? Absolute decomposition, first vs last."""
    se = annual(tri.filter(pl.col("region") == "Sudeste"), ["uf_abbr"])
    early = se.filter(pl.col("year").is_between(2007, 2011))
    late = se.filter(pl.col("year").is_between(2020, 2024))
    e = early.group_by("uf_abbr").agg(
        (pl.col("sinan").sum() / 5).alias("mean_cases_2007_2011"),
        (pl.col("population").sum() / 5).alias("pop_early"))
    l = late.group_by("uf_abbr").agg(
        (pl.col("sinan").sum() / 5).alias("mean_cases_2020_2024"),
        (pl.col("population").sum() / 5).alias("pop_late"))
    m = e.join(l, on="uf_abbr").with_columns(
        (pl.col("mean_cases_2020_2024") - pl.col("mean_cases_2007_2011"))
        .alias("delta_cases"))
    total_delta = float(m["delta_cases"].sum())
    return m.with_columns(
        (pl.col("delta_cases") / total_delta).alias("share_of_SE_decline"),
        (1e5 * pl.col("mean_cases_2007_2011") / pl.col("pop_early"))
        .alias("rate_early_per_100k"),
        (1e5 * pl.col("mean_cases_2020_2024") / pl.col("pop_late"))
        .alias("rate_late_per_100k"),
    ).sort("delta_cases")


def block_health_regions(tri: pl.DataFrame) -> pl.DataFrame:
    se = tri.filter(pl.col("region") == "Sudeste")
    hr = se.group_by(["health_region_code", "health_region_name", "uf_abbr",
                      "year"]).agg(pl.col("sinan_confirmed").sum().alias("sinan"),
                                   pl.col("population").sum().alias("population"))
    rows = []
    for key, sub in hr.group_by(["health_region_code", "health_region_name",
                                 "uf_abbr"]):
        n = int(sub["sinan"].sum())
        if n < 100:      # a slope on a handful of cases is noise with a CI
            continue
        rows.append({"health_region_code": key[0], "health_region_name": key[1],
                     "uf": key[2], "cases": n,
                     **qpois_aapc(sub["year"], sub["sinan"], sub["population"])})
    return pl.DataFrame(rows).sort("aapc")


def block_three_systems(tri: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame,
                                                    pl.DataFrame]:
    """THE DECISIVE CHECK. Same territory, same years, three systems."""
    territories = {
        "Brazil": tri,
        "Sudeste": tri.filter(pl.col("region") == "Sudeste"),
        "Brazil excl. Sudeste": tri.filter(pl.col("region") != "Sudeste"),
        **{f"Sudeste/{uf}": tri.filter((pl.col("region") == "Sudeste")
                                       & (pl.col("uf_abbr") == uf))
           for uf in SE_UFS},
    }
    trends, ratios, series = [], [], []
    for name, d in territories.items():
        a = annual(d).filter(pl.col("year").is_between(*COMMON))
        assert a["sim"].null_count() == 0 and a["sih"].null_count() == 0, name
        series.append(a.with_columns(pl.lit(name).alias("territory")))
        for sysname, col, measure in [
            ("SINAN confirmed cases", "sinan",
             "incidence rate per 100k person-years"),
            ("SIM A27 deaths (underlying cause)", "sim",
             "mortality rate per 100k person-years"),
            ("SIH A27 admissions (any diagnosis)", "sih",
             "admission rate per 100k person-years"),
            ("SIH A27 admissions (principal diagnosis)", "sih_principal",
             "admission rate per 100k person-years"),
        ]:
            est, lo, hi = poisson_ci([float(a[col].sum())],
                                     [float(a["population"].sum())], scale=1e5)
            assert lo[0] <= est[0] <= hi[0]
            trends.append({"territory": name, "system": sysname,
                           "measure": measure, "window": f"{COMMON[0]}-{COMMON[1]}",
                           "rate_per_100k_py": float(est[0]),
                           "rate_lo": float(lo[0]), "rate_hi": float(hi[0]),
                           **qpois_aapc(a["year"], a[col], a["population"])})
        # Ratio models. Offsetting SINAN counts by log(SIM deaths) makes the
        # slope the annual change in confirmed cases PER A27 death: the
        # ascertainment depth of the notification system relative to a system
        # that does not depend on notification at all.
        for label, denom, meaning in [
            ("SINAN confirmed per SIM A27 death", "sim",
             "confirmed cases notified per A27 death registered"),
            ("SINAN confirmed per SIH A27 admission", "sih",
             "confirmed cases notified per A27 hospital admission"),
            ("SIH A27 admissions per SIM A27 death", "sih_sim",
             "A27 admissions per A27 death (two independent systems)"),
        ]:
            num = "sih" if denom == "sih_sim" else "sinan"
            den = "sim" if denom == "sih_sim" else denom
            d_ok = a.filter(pl.col(den) > 0)
            r = qpois_aapc(d_ok["year"], d_ok[num], d_ok[den])
            ratios.append({"territory": name, "ratio": label, "meaning": meaning,
                           "window": f"{COMMON[0]}-{COMMON[1]}",
                           "level": float(d_ok[num].sum() / d_ok[den].sum()),
                           **r})
    return (pl.DataFrame(trends), pl.DataFrame(ratios),
            pl.concat(series).sort(["territory", "year"]))


def block_severity_stratum(tri: pl.DataFrame) -> pl.DataFrame:
    """Split SINAN by severity and ask which stratum carries the decline.

    SIM and SIH only see severe disease. If the SINAN decline were purely
    surveillance retreat, the hospitalised stratum of SINAN would ALSO have to
    fall relative to SIH admissions (the notification system losing even severe
    cases). If instead SINAN-hospitalised tracks SIH while non-hospitalised
    SINAN falls faster, the divergence is confined to the mild stratum -- which
    is exactly the stratum the independent systems cannot adjudicate, and the
    honest answer there is "undetermined by triangulation".
    """
    rows = []
    terr = [("Sudeste", tri.filter(pl.col("region") == "Sudeste"))]
    terr += [(f"Sudeste/{uf}", tri.filter((pl.col("region") == "Sudeste")
                                          & (pl.col("uf_abbr") == uf)))
             for uf in SE_UFS]
    terr += [("Brazil excl. Sudeste", tri.filter(pl.col("region") != "Sudeste"))]
    for name, d in terr:
        a = annual(d).filter(pl.col("year").is_between(*COMMON))
        a = a.with_columns(
            (pl.col("sinan") - pl.col("sinan_hosp")).alias("sinan_nonhosp"))
        early = a.filter(pl.col("year").is_between(2008, 2012))
        late = a.filter(pl.col("year").is_between(2020, 2024))
        for col, label in [("sinan", "SINAN confirmed, all"),
                           ("sinan_hosp", "SINAN confirmed, hospitalised"),
                           ("sinan_nonhosp", "SINAN confirmed, not hospitalised"),
                           ("sih", "SIH A27 admissions")]:
            # Absolute contribution: how many cases per year did this stratum
            # actually shed? An AAPC on a small stratum can be steep and still
            # account for very few of the missing notifications.
            rows.append({"territory": name, "stratum": label,
                         "offset": "population (incidence rate per 100k py)",
                         "events": int(a[col].sum()),
                         "mean_per_year_2008_2012": float(early[col].sum() / 5),
                         "mean_per_year_2020_2024": float(late[col].sum() / 5),
                         "delta_cases_per_year":
                             float(late[col].sum() / 5 - early[col].sum() / 5),
                         **qpois_aapc(a["year"], a[col], a["population"])})
        rows.append({"territory": name,
                     "stratum": "SINAN hospitalised per SIH A27 admission",
                     "offset": "SIH A27 admissions (ratio model)",
                     "events": int(a["sinan_hosp"].sum()),
                     "level": float(a["sinan_hosp"].sum() / a["sih"].sum()),
                     **qpois_aapc(a["year"], a["sinan_hosp"], a["sih"])})
        rows.append({"territory": name,
                     "stratum": "SINAN not hospitalised per SIH A27 admission",
                     "offset": "SIH A27 admissions (ratio model)",
                     "events": int(a["sinan_nonhosp"].sum()),
                     "level": float(a["sinan_nonhosp"].sum() / a["sih"].sum()),
                     **qpois_aapc(a["year"], a["sinan_nonhosp"], a["sih"])})
    return pl.DataFrame(rows)


def block_criterion_by_severity(line: pl.DataFrame,
                                tri: pl.DataFrame) -> pl.DataFrame:
    """Cross the confirmation criterion with hospitalisation, in the Southeast.

    THREAT: "the Southeast decline is laboratory testing contracting". The
    laboratory-confirmed count does fall while the clinical-epidemiological
    count is flat -- but the laboratory stratum is ~88% of confirmed cases, so
    that is mostly arithmetic. The discriminating question is WHERE the
    laboratory confirmations were lost. If testing contracted, the laboratory
    share should fall inside BOTH severity strata. If instead the laboratory
    share holds within each stratum and only the mild stratum shrinks, no test
    was withdrawn: there were fewer mild cases to test.
    """
    pop = (tri.filter(pl.col("region") == "Sudeste").group_by("year")
           .agg(pl.col("population").sum().alias("pop")).sort("year"))
    c = line.filter(pl.col("is_case") & (pl.col("region") == "Sudeste")
                    & pl.col("ate_hosp").is_in(["sim", "nao"])
                    & pl.col("criterio").is_not_null())
    y = c.group_by(["year", "ate_hosp"]).agg(
        pl.len().alias("n"),
        (pl.col("criterio") == "clinico_laboratorial").sum().alias("lab"),
    ).sort(["ate_hosp", "year"])
    rows, series = [], []
    for (h,), sub in y.group_by("ate_hosp"):
        lab_name = "hospitalised" if h == "sim" else "not hospitalised"
        s = sub.join(pop, on="year")
        for col, what in [("n", "all confirmed"), ("lab", "laboratory-confirmed")]:
            rows.append({"stratum": lab_name, "series": what,
                         "measure": "AAPC of the incidence rate per 100k "
                                    "person-years (quasi-Poisson)",
                         "events": int(s[col].sum()),
                         **qpois_aapc(s["year"], s[col], s["pop"])})
        # The endpoint shares are reported for orientation only. The mild
        # stratum thins to ~110-190 cases a year and its annual share swings
        # by 10 points either way; the FITTED odds ratio is the estimate, and
        # a first-vs-last difference read off this series would be mostly noise.
        p, lo, hi = binom_ci(s["lab"].to_numpy(), s["n"].to_numpy())
        assert np.all((lo <= p) & (p <= hi))
        rows.append({"stratum": lab_name,
                     "series": "laboratory share within the stratum",
                     "measure": "odds ratio per calendar year "
                                "(quasi-binomial logistic)",
                     "events": int(s["lab"].sum()),
                     "first_year_share": float(s["lab"][0] / s["n"][0]),
                     "last_year_share": float(s["lab"][-1] / s["n"][-1]),
                     **qbinom_trend(s["year"], s["lab"], s["n"])})
        series.append(s.select(
            pl.lit(lab_name).alias("stratum"), "year",
            pl.col("n").alias("confirmed"), pl.col("lab").alias("laboratory"),
            pl.Series("lab_share", p), pl.Series("lab_share_lo", lo),
            pl.Series("lab_share_hi", hi)))
    return pl.DataFrame(rows), pl.concat(series).sort(["stratum", "year"])


def block_markers(line: pl.DataFrame,
                  tri: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Surveillance markers a case-finding contraction would move."""
    conf = line.filter(pl.col("is_case"))
    pop_all = tri.group_by("year").agg(pl.col("population").sum()).sort("year")
    pop_by_scope = {
        "Sudeste": tri.filter(pl.col("region") == "Sudeste")
        .group_by("year").agg(pl.col("population").sum()).sort("year"),
        "Brazil excl. Sudeste": tri.filter(pl.col("region") != "Sudeste")
        .group_by("year").agg(pl.col("population").sum()).sort("year"),
        "Brazil": pop_all,
    }
    out_rows, trend_rows = [], []
    for scope, sel in [("Sudeste", pl.col("region") == "Sudeste"),
                       ("Brazil excl. Sudeste", pl.col("region") != "Sudeste"),
                       ("Brazil", pl.lit(True))]:
        c = conf.filter(sel)
        notif = line.filter(sel & pl.col("in_window"))
        y = c.group_by("year").agg(
            pl.len().alias("confirmed"),
            (pl.col("criterio") == "clinico_laboratorial").sum().alias("lab"),
            pl.col("criterio").is_not_null().sum().alias("criterio_known"),
            (pl.col("ate_hosp") == "sim").sum().alias("hospitalised"),
            (pl.col("ate_hosp").is_in(["sim", "nao"])).sum().alias("hosp_known"),
            (pl.col("criterio") == "clinico_epidemiologico").sum()
            .alias("clin_epi"),
            (pl.col("evolucao_state") == "valid").sum().alias("outcome_known"),
            ((pl.col("evolucao") == "obito_por_leptospirose")
             .fill_null(False)).sum().alias("deaths"),
        ).sort("year")
        nvol = notif.group_by("year").agg(
            pl.len().alias("notifications"),
            pl.col("classi_fin").is_not_null().sum().alias("classified"),
        ).sort("year")
        y = y.join(nvol, on="year", how="left")
        for num, den, name in [("lab", "criterio_known",
                                "laboratory-confirmation share among confirmed"),
                               ("hospitalised", "hosp_known",
                                "hospitalisation share among confirmed"),
                               ("outcome_known", "confirmed",
                                "outcome completeness among confirmed"),
                               ("confirmed", "classified",
                                "confirmed share of classified notifications"),
                               ("deaths", "outcome_known",
                                "case fatality among confirmed with known "
                                "outcome"),
                               # The two denominators the severity markers rest
                               # on. A rising hospitalisation share means
                               # nothing if the field simply started being
                               # filled in.
                               ("hosp_known", "confirmed",
                                "ATE_HOSP answered share among confirmed"),
                               ("criterio_known", "confirmed",
                                "CRITERIO answered share among confirmed")]:
            p, lo, hi = binom_ci(y[num].to_numpy(), y[den].to_numpy())
            assert np.all((lo <= p) & (p <= hi))
            for i, yr in enumerate(y["year"]):
                out_rows.append({"scope": scope, "marker": name, "year": int(yr),
                                 "numerator": int(y[num][i]),
                                 "denominator": int(y[den][i]),
                                 "proportion": float(p[i]),
                                 "lo": float(lo[i]), "hi": float(hi[i])})
            trend_rows.append({"scope": scope, "marker": name,
                               "measure": "odds ratio per calendar year "
                                          "(quasi-binomial logistic)",
                               "window": "2007-2025",
                               **qbinom_trend(y["year"], y[num], y[den])})
        # THREAT: "the decline is falling laboratory testing". If confirmations
        # fell because serology stopped being done, the laboratory-confirmed
        # COUNT must fall much faster than the clinical-epidemiological count.
        # If both fall in step, the pool of true cases fell, not the testing.
        pop = (pop_by_scope[scope].rename({"population": "pop"})
               if scope in pop_by_scope else None)
        if pop is not None:
            y2 = y.join(pop, on="year", how="inner")
            for col, label in [("lab", "confirmed cases, laboratory criterion"),
                               ("clin_epi",
                                "confirmed cases, clinical-epidemiological "
                                "criterion")]:
                trend_rows.append({
                    "scope": scope, "marker": label,
                    "measure": "AAPC of the incidence rate per 100k "
                               "person-years (quasi-Poisson)",
                    "window": "2007-2025",
                    **qpois_aapc(y2["year"], y2[col], y2["pop"])})
        # Notification VOLUME is a count, not a proportion: a contraction in
        # case-finding shows up here first, before any proportion moves.
        trend_rows.append({"scope": scope, "marker": "notification volume (all "
                           "suspected, any classification)",
                           "measure": "AAPC of the notification count "
                                      "(quasi-Poisson, no offset)",
                           "window": "2007-2025",
                           **qpois_aapc(y["year"], y["notifications"],
                                        np.ones(y.height))})
    return pl.DataFrame(out_rows), pl.DataFrame(trend_rows)


# ---------------------------------------------------------------------------
def fmt(r: dict, key: str = "aapc") -> str:
    return (f"{r[key]:+6.2f}%/yr ({r[key + '_lo' if key == 'aapc' else '_lo']:+6.2f}"
            f", {r[key + '_hi' if key == 'aapc' else '_hi']:+6.2f})")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rule("Guards")
    check_ci_helpers()

    tri = load_triangulation()
    se_series = annual(tri.filter(pl.col("region") == "Sudeste")).select(
        pl.col("year"), pl.col("sinan").alias("events"),
        pl.col("population").alias("person_time"))
    r_check = validate_against_r(se_series)

    rule("1. Regional decomposition, SINAN confirmed cases (residence geography)")
    reg = block_regional(tri)
    print(reg.select("scope", "window", "n_years", "events", "aapc", "aapc_lo",
                     "aapc_hi", "mk_p").with_columns(
        pl.col(["aapc", "aapc_lo", "aapc_hi"]).round(2),
        pl.col("mk_p").round(4)))
    reg.write_csv(OUT / "01_regional_aapc.csv")

    t3 = pl.read_csv(T3)
    se_t3 = float(t3.filter(pl.col("scope") == "region: Sudeste")["aapc"][0])
    se_mine = float(reg.filter((pl.col("scope") == "region: Sudeste")
                               & (pl.col("window") == "2007-2025"))["aapc"][0])
    print(f"\nT3_trend Sudeste AAPC (infection geography, person-months): "
          f"{se_t3:+.3f}")
    print(f"this script  Sudeste AAPC (residence geography, person-years): "
          f"{se_mine:+.3f}")

    rule("2. Southeast by state")
    uf_tbl, loo = block_uf(tri)
    print(uf_tbl.select("uf", "cases_2007_2025", "incidence_per_100k_py",
                        "aapc", "aapc_lo", "aapc_hi", "mk_p").with_columns(
        pl.col(["incidence_per_100k_py", "aapc", "aapc_lo", "aapc_hi"]).round(2),
        pl.col("mk_p").round(4)))
    print("\nleave-one-out (Southeast AAPC with each state removed):")
    print(loo.select("dropped", "events", "aapc", "aapc_lo", "aapc_hi",
                     "mk_p").with_columns(
        pl.col(["aapc", "aapc_lo", "aapc_hi"]).round(2), pl.col("mk_p").round(4)))
    uf_tbl.write_csv(OUT / "02_sudeste_by_uf.csv")
    loo.write_csv(OUT / "03_sudeste_leave_one_state_out.csv")

    contrib = block_contribution(tri)
    print("\nabsolute decomposition, mean annual cases 2007-2011 vs 2020-2024:")
    print(contrib.with_columns(
        pl.col(["mean_cases_2007_2011", "mean_cases_2020_2024", "delta_cases",
                "rate_early_per_100k", "rate_late_per_100k"]).round(2),
        pl.col("share_of_SE_decline").round(3)).select(
        "uf_abbr", "mean_cases_2007_2011", "mean_cases_2020_2024", "delta_cases",
        "share_of_SE_decline", "rate_early_per_100k", "rate_late_per_100k"))
    contrib.write_csv(OUT / "04_sudeste_uf_contribution.csv")

    hr = block_health_regions(tri)
    hr.write_csv(OUT / "05_sudeste_health_region_aapc.csv")
    n_neg = int(hr.filter(pl.col("aapc_hi") < 0).height)
    n_pos = int(hr.filter(pl.col("aapc_lo") > 0).height)
    print(f"\nhealth regions in the Southeast with >=100 confirmed cases: "
          f"{hr.height}; interval entirely below zero: {n_neg}; entirely above: "
          f"{n_pos}")
    print(hr.head(10).select("health_region_name", "uf", "cases", "aapc",
                             "aapc_lo", "aapc_hi").with_columns(
        pl.col(["aapc", "aapc_lo", "aapc_hi"]).round(2)))

    rule(f"3. DECISIVE CHECK - three independent systems, {COMMON[0]}-{COMMON[1]}")
    trends, ratios, series = block_three_systems(tri)
    trends.write_csv(OUT / "06_three_system_aapc.csv")
    ratios.write_csv(OUT / "07_ratio_trends.csv")
    series.write_csv(OUT / "08_annual_series_by_territory.csv")
    for terr in ["Sudeste", "Brazil excl. Sudeste"]:
        print(f"\n{terr}:")
        print(trends.filter(pl.col("territory") == terr).select(
            "system", "events", "rate_per_100k_py", "aapc", "aapc_lo", "aapc_hi",
            "dispersion", "mk_p").with_columns(
            pl.col(["rate_per_100k_py", "aapc", "aapc_lo", "aapc_hi",
                    "dispersion"]).round(2), pl.col("mk_p").round(4)))
    print("\nall territories, SINAN vs SIM vs SIH:")
    with pl.Config(tbl_rows=40, fmt_str_lengths=44):
        print(trends.select("territory", "system", "events", "aapc", "aapc_lo",
                            "aapc_hi", "mk_p").with_columns(
            pl.col(["aapc", "aapc_lo", "aapc_hi"]).round(2),
            pl.col("mk_p").round(4)))
    print("\nratio models (annual change in the ratio):")
    with pl.Config(tbl_rows=40, fmt_str_lengths=48):
        print(ratios.select("territory", "ratio", "level", "aapc", "aapc_lo",
                            "aapc_hi", "mk_p").with_columns(
            pl.col(["level", "aapc", "aapc_lo", "aapc_hi"]).round(2),
            pl.col("mk_p").round(4)))

    # Multiplicative decomposition. The SINAN incidence rate is
    #   (SINAN / SIM) * (SIM / population),
    # so on the log scale the SINAN slope is the sum of the ratio slope (the
    # ascertainment-depth component) and the SIM slope (the component anchored
    # to a system that needs no notification). Reported as a coherence check on
    # the three fits, not as a new estimate.
    def g(df, **kw):
        f = df
        for k, v in kw.items():
            f = f.filter(pl.col(k) == v)
        return f.to_dicts()[0]
    se_sinan = g(trends, territory="Sudeste", system="SINAN confirmed cases")
    se_sim = g(trends, territory="Sudeste",
               system="SIM A27 deaths (underlying cause)")
    se_rat = g(ratios, territory="Sudeste",
               ratio="SINAN confirmed per SIM A27 death")
    implied = 100 * ((1 + se_rat["aapc"] / 100) * (1 + se_sim["aapc"] / 100) - 1)
    print(f"\ndecomposition of the Sudeste SINAN AAPC ({se_sinan['aapc']:+.2f}%/yr):"
          f"\n  disease component, anchored to SIM A27 deaths : "
          f"{se_sim['aapc']:+.2f}%/yr"
          f"\n  ascertainment-depth component (SINAN per death): "
          f"{se_rat['aapc']:+.2f}%/yr"
          f"\n  product of the two                            : {implied:+.2f}%/yr")
    decomp = {"sinan_aapc": se_sinan["aapc"], "sim_aapc": se_sim["aapc"],
              "ratio_aapc": se_rat["aapc"], "product": implied}

    rule("3b. Which severity stratum carries the SINAN decline?")
    sev = block_severity_stratum(tri)
    sev.write_csv(OUT / "11_severity_stratum.csv")
    with pl.Config(tbl_rows=60, fmt_str_lengths=48):
        print(sev.select("territory", "stratum", "events", "aapc", "aapc_lo",
                         "aapc_hi", "mk_p").with_columns(
            pl.col(["aapc", "aapc_lo", "aapc_hi"]).round(2),
            pl.col("mk_p").round(4)))
    # Plausibility: SINAN hospitalised should EXCEED SIH A27 admissions, because
    # SIH is the public (SUS) hospital billing system only. A ratio below one
    # would mean the notification system misses admissions it recorded.
    lev = sev.filter(pl.col("stratum")
                     == "SINAN hospitalised per SIH A27 admission")
    print("\nSINAN hospitalised / SIH A27 admissions, level over the window:")
    print(lev.select("territory", "level").with_columns(pl.col("level").round(2)))

    rule("4. Surveillance markers")
    line = load_line()
    markers, mtrends = block_markers(line, tri)
    markers.write_csv(OUT / "09_surveillance_markers_by_year.csv")
    mtrends.write_csv(OUT / "10_surveillance_marker_trends.csv")
    with pl.Config(tbl_rows=40, fmt_str_lengths=56, tbl_width_chars=200):
        print(mtrends.select("scope", "marker", "or_year", "or_lo", "or_hi",
                             "aapc", "aapc_lo", "aapc_hi", "mk_p").with_columns(
            pl.col(["or_year", "or_lo", "or_hi", "aapc", "aapc_lo",
                    "aapc_hi"]).round(3), pl.col("mk_p").round(4)))
    print("\nSudeste markers, first and last year:")
    print(markers.filter((pl.col("scope") == "Sudeste")
                         & pl.col("year").is_in([2007, 2015, 2024, 2025]))
          .with_columns(pl.col(["proportion", "lo", "hi"]).round(4))
          .sort("marker", "year"))

    rule("4b. Confirmation criterion crossed with severity, Sudeste")
    crit, crit_series = block_criterion_by_severity(line, tri)
    crit.write_csv(OUT / "12_criterion_by_severity_sudeste.csv")
    crit_series.write_csv(OUT / "13_lab_share_by_severity_year_sudeste.csv")
    with pl.Config(tbl_rows=20, fmt_str_lengths=42, tbl_width_chars=200):
        print(crit.select("stratum", "series", "events", "aapc", "aapc_lo",
                          "aapc_hi", "or_year", "or_lo", "or_hi",
                          "first_year_share", "last_year_share", "mk_p")
              .with_columns(pl.col(["aapc", "aapc_lo", "aapc_hi", "or_year",
                                    "or_lo", "or_hi", "first_year_share",
                                    "last_year_share"]).round(3),
                            pl.col("mk_p").round(4)))
    ls = crit.filter(pl.col("series") == "laboratory share within the stratum")
    a, b = ls.filter(pl.col("stratum") == "hospitalised").to_dicts()[0], \
        ls.filter(pl.col("stratum") == "not hospitalised").to_dicts()[0]
    overlap = (a["or_lo"] <= b["or_hi"]) and (b["or_lo"] <= a["or_hi"])
    print(f"\nlaboratory share, annual odds ratio: hospitalised "
          f"{a['or_year']:.3f} ({a['or_lo']:.3f}, {a['or_hi']:.3f}); "
          f"not hospitalised {b['or_year']:.3f} ({b['or_lo']:.3f}, "
          f"{b['or_hi']:.3f})\nintervals overlap: {overlap} -> a "
          f"severity-SPECIFIC withdrawal of laboratory confirmation is "
          f"{'not evidenced' if overlap else 'evidenced'}")

    # Cross-check the national laboratory share against T9, which was computed
    # independently in R from the same line list.
    t9 = pl.read_csv(T9)
    mine9 = markers.filter((pl.col("scope") == "Brazil")
                           & (pl.col("marker") ==
                              "laboratory-confirmation share among confirmed"))
    j = t9.join(mine9, on="year").with_columns(
        (pl.col("lab_share") - pl.col("proportion")).abs().alias("d"))
    print(f"\nT9 cross-check: max |lab_share difference| over "
          f"{j.height} years = {float(j['d'].max()):.2e}")

    rule("Verdict inputs")
    se_ratio_sim = ratios.filter((pl.col("territory") == "Sudeste")
                                 & pl.col("ratio").str.contains("SIM A27 death")
                                 & pl.col("ratio").str.starts_with("SINAN"))
    se_ratio_sih = ratios.filter((pl.col("territory") == "Sudeste")
                                 & pl.col("ratio").str.contains("SIH A27 admission"))
    report = {
        "r_validation": r_check,
        "windows": {"sinan": SINAN_YEARS, "sim": SIM_YEARS, "sih": SIH_YEARS,
                    "common": COMMON},
        "sudeste_sinan_aapc_2007_2025": reg.filter(
            (pl.col("scope") == "region: Sudeste")
            & (pl.col("window") == "2007-2025")).to_dicts()[0],
        "sudeste_three_system": trends.filter(
            pl.col("territory") == "Sudeste").to_dicts(),
        "sudeste_ratio_sinan_per_sim_death": se_ratio_sim.to_dicts(),
        "sudeste_ratio_sinan_per_sih_admission": se_ratio_sih.to_dicts(),
        "sudeste_marker_trends": mtrends.filter(
            pl.col("scope") == "Sudeste").to_dicts(),
        "sudeste_aapc_decomposition": decomp,
        "sudeste_severity_stratum": sev.filter(
            pl.col("territory") == "Sudeste").to_dicts(),
        "sudeste_by_uf": uf_tbl.to_dicts(),
        "sudeste_leave_one_state_out": loo.to_dicts(),
        "t9_lab_share_max_abs_difference": float(j["d"].max()),
        "sudeste_criterion_by_severity": crit.to_dicts(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float),
                                     encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
