#!/usr/bin/env python
"""Two temporal robustness threads for the leptospirosis descriptive paper.

T-3  Is the trend stable without the preliminary 2025 edge?
     NAMED THREAT: the reported national and regional AAPCs are artefacts of an
     incomplete final year (2025 is preliminary; later revisions remain
     possible).  The rebuttal is a refit of the SAME estimator on 2007-2024 and
     a side-by-side against the published 2007-2025 values, plus the raw size
     of the dropped edge (2025 vs 2024 counts and incidence rates).

T-4  Does the Northeast's distinct seasonality survive at health-region grain?
     NAMED THREAT: the Northeast June peak is an artefact of macro-region
     aggregation -- one state or a few large-volume health regions carrying a
     macro-region whose remainder follows the national February/March pattern.
     The rebuttal is the same circular statistic recomputed per health region
     and per state, the fraction of Northeast health regions that actually peak
     in May-July, and explicit concentration diagnostics (leave-one-state-out,
     leave-one-health-region-out, cluster bootstrap over health regions).

METHOD PARITY.  Both threads reimplement, in Python, the estimators the paper
already published, so the new numbers are comparable rather than merely
adjacent:

  * AAPC  -- R/11_tables.R::trend_apc.  Log-linear QUASI-Poisson in calendar
    year with log(person-time) offset; AAPC = 100*(exp(beta)-1); Wald interval
    on beta with the quasi-Poisson (Pearson) dispersion.  Mann-Kendall tau on
    the rate series is carried alongside, as there.
  * Circular seasonality -- R/11_tables.R::circular_season.  theta =
    2*pi*(month-0.5)/12, resultant length r = |sum w e^{i theta}| / sum w,
    mean angle -> peak month.  The Rayleigh p-value is NOT computed and NOT
    reported: it treats each case as an independent draw and cases within a
    municipality-month are anything but.

Both reimplementations are GATED: the script refits the published window first
and asserts it reproduces data/results/01_descriptive/T3_trend.csv and
T5_seasonality.csv before any new number is written.  A reimplementation that
cannot reproduce the published value is not evidence about the published value.

Usage:
  PYTHONPATH=. python studies/leptospirosis/43_temporal_threads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from brepi.analysis.rates import binom_ci, poisson_ci

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "temporal_threads"
DESC = ROOT / "data" / "results" / "01_descriptive"

LINE = ROOT / "data" / "interim" / "lept_line_level.parquet"
POP = ROOT / "data" / "interim" / "population_tensor_long.parquet"
PANEL = ROOT / "data" / "panel" / "lept_panel_municipality_month.parquet"

REGION_PT = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul",
             "5": "Centro-Oeste"}

# T-4 minimum unit size.  Stated and justified once, here.
#
# A circular mean is a weighted vector sum: a single municipality-month cluster
# of k cases rotates the resultant by an amount that scales with k / n.  With
# n = 100 and the resultant lengths seen at macro-region grain (r ~ 0.3-0.38),
# the resultant vector R = n*r is ~30-38 cases long, so a 10-case outbreak in
# the "wrong" month moves the mean angle by at most ~15 degrees, i.e. half a
# month; below n = 100 a single outbreak can move it more than a full month and
# the "peak month" stops being a property of the unit's climatology.  100 is
# therefore the smallest unit size at which a *peak month* is worth tabulating.
# Sensitivity of the headline fraction to this choice is reported (T4f) because
# the choice is mine, not the paper's.
MIN_CASES = 100

WINDOW_LO, WINDOW_HI = 2007, 2025


def rule(s: str) -> None:
    print("\n" + "-" * 74 + f"\n{s}\n" + "-" * 74)


def w(df: pl.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / name)
    print(f"  wrote {name} - {df.height} rows")


# ---------------------------------------------------------------------------
# Estimators (parity with R/11_tables.R)
# ---------------------------------------------------------------------------
def quasipoisson_aapc(years: np.ndarray, events: np.ndarray,
                      person_time: np.ndarray, conf: float = 0.95) -> dict:
    """Log-linear quasi-Poisson AAPC with an offset, matching `trend_apc`.

    Fitted by IRLS.  The slope and its standard error are invariant to
    centring the year, so the year is centred for conditioning; R fits the raw
    year and gets the same beta.  Dispersion is the Pearson statistic over the
    residual degrees of freedom, exactly as `summary.glm` computes it for
    family = quasipoisson.
    """
    y = np.asarray(events, dtype=float)
    t = np.asarray(person_time, dtype=float)
    x = np.asarray(years, dtype=float)
    n = y.size
    if n < 4 or y.sum() == 0:
        return dict(aapc=np.nan, aapc_lo=np.nan, aapc_hi=np.nan,
                    dispersion=np.nan, n_years=n)
    xc = x - x.mean()
    X = np.column_stack([np.ones(n), xc])
    off = np.log(t)

    # IRLS with R's glm.fit start (mu = y + 0.1), R's stopping rule
    # (|dev - dev_old| / (|dev| + 0.1) < 1e-8) and R's maxit (25).  Converging
    # *tighter* than R is not harmless here: it shifts the Pearson dispersion
    # in the sixth significant figure and the gate below is tight enough to see
    # it.  Matching the published estimator means matching how it stopped.
    def deviance(mu_: np.ndarray) -> float:
        nz = y > 0
        term = np.zeros_like(y)
        term[nz] = y[nz] * np.log(y[nz] / mu_[nz])
        return float(2 * np.sum(term - (y - mu_)))

    mu = y + 0.1
    eta = np.log(mu)
    b = np.zeros(2)
    dev_old = deviance(mu)
    w_last = mu.copy()
    for _ in range(25):
        # `glm.fit` forms the IRLS weight from the CURRENT mu, solves, and only
        # then updates mu.  The weight vector it returns -- and that
        # `summary.glm` uses for both the unscaled covariance and the
        # dispersion -- is therefore the one built from the PENULTIMATE mu,
        # while the working residual is built from the final mu.  Reproducing
        # the published dispersion requires reproducing that asymmetry.
        w_last = mu.copy()
        z = eta - off + (y - mu) / mu
        XtW = X.T * w_last
        b = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ b + off
        mu = np.exp(eta)
        dev = deviance(mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < 1e-8:
            break
        dev_old = dev

    XtW = X.T * w_last
    cov_unscaled = np.linalg.inv(XtW @ X)
    working_resid = (y - mu) / mu
    dispersion = float(np.sum(w_last * working_resid ** 2) / (n - 2))
    se = float(np.sqrt(cov_unscaled[1, 1] * dispersion))
    beta = float(b[1])
    # Wald z, matching qnorm in trend_apc (not a t quantile).
    from scipy.stats import norm
    z_crit = float(norm.ppf(1 - (1 - conf) / 2))
    return dict(aapc=100 * (np.exp(beta) - 1),
                aapc_lo=100 * (np.exp(beta - z_crit * se) - 1),
                aapc_hi=100 * (np.exp(beta + z_crit * se) - 1),
                dispersion=dispersion, n_years=n)


def mann_kendall(x: np.ndarray) -> dict:
    """Mann-Kendall tau with the tie correction, matching `.mann_kendall`."""
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return dict(mk_tau=np.nan, mk_p=np.nan)
    d = x[:, None] - x[None, :]
    s = float(np.sum(np.sign(d[np.tril_indices(n, -1)])))
    _, counts = np.unique(x, return_counts=True)
    vs = (n * (n - 1) * (2 * n + 5)
          - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    z = (s - 1) / np.sqrt(vs) if s > 0 else ((s + 1) / np.sqrt(vs) if s < 0 else 0.0)
    return dict(mk_tau=s / (0.5 * n * (n - 1)), mk_p=2 * float(norm.cdf(-abs(z))))


def circular_season(months: np.ndarray, weights: np.ndarray) -> dict:
    """Circular summary of monthly case mass, matching `circular_season`.

    Returns peak_month, peak_month_frac, resultant_r, cases.  NO Rayleigh
    p-value: it assumes independent cases, which these are not.
    """
    m = np.asarray(months, dtype=float)
    wgt = np.asarray(weights, dtype=float)
    n = float(wgt.sum())
    if n <= 0:
        return dict(peak_month=None, peak_month_frac=np.nan,
                    resultant_r=np.nan, cases=0.0)
    th = 2 * np.pi * (m - 0.5) / 12.0
    C = float(np.sum(wgt * np.cos(th)))
    S = float(np.sum(wgt * np.sin(th)))
    r = np.hypot(C, S) / n
    ang = np.arctan2(S, C)
    if ang < 0:
        ang += 2 * np.pi
    frac = ang / (2 * np.pi) * 12 + 0.5
    if frac > 12:
        frac -= 12
    pm = int(np.round(frac)) % 12
    if pm == 0:
        pm = 12
    return dict(peak_month=pm, peak_month_frac=float(frac),
                resultant_r=float(r), cases=n)


# ===========================================================================
# THREAD T-3
# ===========================================================================
def thread_t3() -> dict:
    rule("T-3  AAPC with and without the preliminary 2025 edge")

    # --- numerator: the paper's case definition, verbatim ------------------
    line = pl.read_parquet(
        LINE,
        columns=["src_year", "classi_fin", "municipality_residence_region",
                 "uf_notification_region", "DT_SIN_PRI", "DT_NOTIFIC"],
    ).with_columns(
        pl.col("DT_SIN_PRI").str.to_date(strict=False).alias("onset"),
        pl.col("DT_NOTIFIC").str.to_date(strict=False).alias("notified"),
    ).with_columns(
        pl.coalesce(["municipality_residence_region",
                     "uf_notification_region"]).alias("region"),
        pl.col("src_year").cast(pl.Int32).alias("year"),
    )
    lo_d, hi_d = pl.date(2007, 1, 1), pl.date(2025, 12, 31)
    in_window = (
        pl.when(pl.col("onset").is_not_null())
        .then(pl.col("onset").is_between(lo_d, hi_d))
        .otherwise(pl.col("notified").is_not_null()
                   & pl.col("notified").is_between(lo_d, hi_d))
    )
    line = line.with_columns(in_window.fill_null(False).alias("in_window"))
    conf = line.filter((pl.col("classi_fin") == "confirmado")
                       & pl.col("in_window") & pl.col("region").is_not_null())
    print(f"  confirmed cases in window with a known region: {conf.height:,}")

    num = conf.group_by(["region", "year"]).agg(pl.len().alias("cases"))

    # --- denominator: the population tensor, person-months -----------------
    pop = pl.read_parquet(POP, columns=["munic_code", "year", "population"])
    pop = (pop.with_columns(
        pl.col("munic_code").cast(pl.Utf8).str.slice(0, 1)
        .replace_strict(REGION_PT, default=None).alias("region"))
        .filter(pl.col("region").is_not_null()
                & pl.col("year").is_between(WINDOW_LO, WINDOW_HI))
        .group_by(["region", "year"])
        .agg((pl.col("population").sum() * 12).alias("person_months")))

    burden = (pop.join(num, on=["region", "year"], how="left")
              .with_columns(pl.col("cases").fill_null(0))
              .sort(["region", "year"]))

    def fit(scope: str, df: pl.DataFrame, drop_covid: bool = False) -> dict:
        d = df.filter(~pl.col("year").is_in([2020, 2021])) if drop_covid else df
        agg = (d.group_by("year")
               .agg(pl.col("cases").sum().alias("events"),
                    pl.col("person_months").sum().alias("person_time"))
               .sort("year"))
        a = quasipoisson_aapc(agg["year"].to_numpy(), agg["events"].to_numpy(),
                              agg["person_time"].to_numpy())
        mk = mann_kendall(agg["events"].to_numpy() / agg["person_time"].to_numpy())
        return dict(scope=scope, **a, **mk,
                    cases_total=int(agg["events"].sum()))

    def fit_all(burden_df: pl.DataFrame, window: str) -> pl.DataFrame:
        rows = [fit("national", burden_df),
                fit("national, excluding 2020-21", burden_df, drop_covid=True)]
        for reg in sorted(burden_df["region"].unique().to_list()):
            rows.append(fit(f"region: {reg}",
                            burden_df.filter(pl.col("region") == reg)))
        return pl.DataFrame(rows).with_columns(pl.lit(window).alias("window"))

    full = fit_all(burden, "2007-2025")

    # --- GATE: reproduce the published T3_trend.csv ------------------------
    pub = pl.read_csv(DESC / "T3_trend.csv")
    chk = full.join(pub, on="scope", how="inner", suffix="_pub")
    assert chk.height == pub.height, "scope labels do not line up with T3_trend"
    worst = 0.0
    for col in ["aapc", "aapc_lo", "aapc_hi", "dispersion", "mk_tau", "mk_p"]:
        d = float((chk[col] - chk[f"{col}_pub"]).abs().max())
        worst = max(worst, d)
        print(f"  gate |{col} - published| max = {d:.3e}")
    assert worst < 1e-8, (
        f"Python reimplementation does not reproduce T3_trend.csv (max abs "
        f"difference {worst:.3e}); the 2007-2024 refit would not be comparable."
    )
    assert (chk["n_years"] == chk["n_years_pub"]).all()
    print("  GATE PASSED: trend_apc reproduced to <1e-8 on 2007-2025.")

    # --- the actual test: drop the preliminary edge ------------------------
    trunc = fit_all(burden.filter(pl.col("year") <= 2024), "2007-2024")

    cmp = (full.select(["scope", "aapc", "aapc_lo", "aapc_hi", "dispersion",
                        "mk_tau", "mk_p", "n_years", "cases_total"])
           .rename({c: f"{c}_2007_2025" for c in
                    ["aapc", "aapc_lo", "aapc_hi", "dispersion", "mk_tau",
                     "mk_p", "n_years", "cases_total"]})
           .join(trunc.select(["scope", "aapc", "aapc_lo", "aapc_hi",
                               "dispersion", "mk_tau", "mk_p", "n_years",
                               "cases_total"])
                 .rename({c: f"{c}_2007_2024" for c in
                          ["aapc", "aapc_lo", "aapc_hi", "dispersion",
                           "mk_tau", "mk_p", "n_years", "cases_total"]}),
                 on="scope"))
    cmp = cmp.with_columns(
        (pl.col("aapc_2007_2024") - pl.col("aapc_2007_2025"))
        .alias("aapc_shift_pp_per_year"),
        # Does the sign of the conclusion change? "excludes zero" is the
        # conclusion a reader takes from an AAPC interval.
        ((pl.col("aapc_lo_2007_2025") > 0) | (pl.col("aapc_hi_2007_2025") < 0))
        .alias("ci_excludes_zero_2007_2025"),
        ((pl.col("aapc_lo_2007_2024") > 0) | (pl.col("aapc_hi_2007_2024") < 0))
        .alias("ci_excludes_zero_2007_2024"),
    ).with_columns(
        (pl.col("ci_excludes_zero_2007_2025")
         != pl.col("ci_excludes_zero_2007_2024")).alias("conclusion_changes")
    )
    w(cmp, "T3a_aapc_window_comparison.csv")
    print(cmp.select([
        "scope",
        pl.format("{} [{}, {}]", pl.col("aapc_2007_2025").round(3),
                  pl.col("aapc_lo_2007_2025").round(3),
                  pl.col("aapc_hi_2007_2025").round(3)).alias("AAPC 2007-2025"),
        pl.format("{} [{}, {}]", pl.col("aapc_2007_2024").round(3),
                  pl.col("aapc_lo_2007_2024").round(3),
                  pl.col("aapc_hi_2007_2024").round(3)).alias("AAPC 2007-2024"),
        pl.col("aapc_shift_pp_per_year").round(3),
        "conclusion_changes"]))

    # --- the size of the edge: 2025 against 2024 ---------------------------
    def edge(scope: str, df: pl.DataFrame) -> dict:
        agg = (df.group_by("year")
               .agg(pl.col("cases").sum().alias("cases"),
                    (pl.col("person_months").sum() / 12).alias("person_years"))
               .sort("year"))
        row = {"scope": scope}
        for yr in (2024, 2025):
            r = agg.filter(pl.col("year") == yr)
            c = int(r["cases"][0])
            py = float(r["person_years"][0])
            # ESTIMATE FIRST -- poisson_ci returns (rate, lo, hi).
            rate, lo, hi = poisson_ci(c, py, scale=1e5)
            rate, lo, hi = float(rate), float(lo), float(hi)
            assert lo <= rate <= hi, "poisson_ci unpacked in the wrong order"
            row[f"cases_{yr}"] = c
            row[f"person_years_{yr}"] = py
            row[f"rate_per_100k_py_{yr}"] = rate
            row[f"rate_lo_{yr}"] = lo
            row[f"rate_hi_{yr}"] = hi
        c24, c25 = row["cases_2024"], row["cases_2025"]
        t24, t25 = row["person_years_2024"], row["person_years_2025"]
        row["pct_change_cases_2025_vs_2024"] = 100 * (c25 - c24) / c24
        # Exact incidence RATE RATIO (2025 vs 2024) via the conditional
        # binomial: given c24+c25, c25 ~ Bin(n, p) with p/(1-p) = RR * t25/t24.
        # Clopper-Pearson on p therefore gives an exact interval for RR.
        p, plo, phi = binom_ci(c25, c24 + c25)
        p, plo, phi = float(p), float(plo), float(phi)
        assert plo <= p <= phi, "binom_ci unpacked in the wrong order"
        k = t24 / t25
        row["irr_2025_vs_2024"] = (p / (1 - p)) * k
        row["irr_lo"] = (plo / (1 - plo)) * k
        row["irr_hi"] = (phi / (1 - phi)) * k
        return row

    rows = [edge("national", burden)]
    for reg in sorted(burden["region"].unique().to_list()):
        rows.append(edge(f"region: {reg}", burden.filter(pl.col("region") == reg)))
    edge_tbl = pl.DataFrame(rows)
    w(edge_tbl, "T3c_edge_2024_vs_2025.csv")
    print(edge_tbl.select(["scope", "cases_2024", "cases_2025",
                           "pct_change_cases_2025_vs_2024", "irr_2025_vs_2024",
                           "irr_lo", "irr_hi"]))

    # --- the full yearly series, so the edge is visible in context ---------
    series_rows = []
    for scope, df in [("national", burden)] + [
            (f"region: {r}", burden.filter(pl.col("region") == r))
            for r in sorted(burden["region"].unique().to_list())]:
        agg = (df.group_by("year")
               .agg(pl.col("cases").sum().alias("cases"),
                    (pl.col("person_months").sum() / 12).alias("person_years"))
               .sort("year"))
        rate, lo, hi = poisson_ci(agg["cases"].to_numpy(),
                                  agg["person_years"].to_numpy(), scale=1e5)
        assert np.all(lo <= rate) and np.all(rate <= hi)
        series_rows.append(agg.with_columns(
            pl.lit(scope).alias("scope"),
            pl.Series("rate_per_100k_py", rate),
            pl.Series("rate_lo", lo), pl.Series("rate_hi", hi)))
    series = pl.concat(series_rows).select(
        ["scope", "year", "cases", "person_years", "rate_per_100k_py",
         "rate_lo", "rate_hi"])
    w(series, "T3b_yearly_series.csv")

    return {"cmp": cmp, "edge": edge_tbl}


# ===========================================================================
# THREAD T-4
# ===========================================================================
def thread_t4() -> dict:
    rule("T-4  Northeast seasonality at health-region and state grain")

    panel_full = pl.read_parquet(
        PANEL,
        columns=["munic_code", "period", "year", "month", "uf_abbr", "region",
                 "health_region_code", "health_region_name", "cases"])
    assert (panel_full["month"] == panel_full["period"].dt.month()).all(), \
        "panel month is not the month of `period`"
    # The full lattice of health regions, INCLUDING those that never recorded a
    # confirmed case -- a choropleth needs them present and empty, not absent.
    lattice_hr = (panel_full.select(["health_region_code", "health_region_name",
                                     "uf_abbr", "region"]).unique()
                  .unique(subset=["health_region_code"]))
    panel = panel_full.filter(pl.col("cases") > 0)
    print(f"  panel confirmed cases 2007-2025: {int(panel['cases'].sum()):,}")

    # --- GATE: reproduce T5_seasonality at macro-region grain --------------
    # The panel keys geography on municipality of INFECTION with residence as
    # a documented fallback, and drops the handful of cases with no onset date;
    # T5 keys on residence with notification as fallback.  The two are not the
    # same 66,516 cases, so the gate is agreement in PEAK MONTH and closeness
    # in r, not bit equality -- and the gap is reported, not assumed away.
    pub5 = pl.read_csv(DESC / "T5_seasonality.csv")
    val_rows = []
    for scope, df in [("national", panel)] + [
            (r, panel.filter(pl.col("region") == r))
            for r in sorted(panel["region"].unique().to_list())]:
        agg = df.group_by("month").agg(pl.col("cases").sum().alias("n"))
        s = circular_season(agg["month"].to_numpy(), agg["n"].to_numpy())
        p = pub5.filter(pl.col("scope") == scope)
        val_rows.append(dict(
            scope=scope, panel_peak_month=s["peak_month"],
            panel_resultant_r=s["resultant_r"], panel_cases=int(s["cases"]),
            published_peak_month=int(p["peak_month"][0]),
            published_resultant_r=float(p["resultant_r"][0]),
            published_cases=int(p["cases"][0])))
    val = pl.DataFrame(val_rows).with_columns(
        (pl.col("panel_peak_month") == pl.col("published_peak_month"))
        .alias("peak_month_agrees"),
        (pl.col("panel_resultant_r") - pl.col("published_resultant_r")).abs()
        .alias("r_abs_diff"),
        (pl.col("published_cases") - pl.col("panel_cases")).alias("case_gap"))
    w(val, "T4a_macro_region_gate.csv")
    print(val)
    assert val["peak_month_agrees"].all(), (
        "the panel does not reproduce the published macro-region peak months; "
        "health-region results computed on it would not be comparable to T5")
    assert float(val["r_abs_diff"].max()) < 0.01
    print("  GATE PASSED: panel reproduces every published macro-region peak "
          "month, r within 0.01.")

    # --- circular seasonality per unit -------------------------------------
    def per_unit(keys: list[str]) -> pl.DataFrame:
        agg = panel.group_by(keys + ["month"]).agg(
            pl.col("cases").sum().alias("n"))
        rows = []
        for key, g in agg.group_by(keys, maintain_order=True):
            s = circular_season(g["month"].to_numpy(), g["n"].to_numpy())
            rows.append(dict(zip(keys, key), **s))
        out = pl.DataFrame(rows).with_columns(
            pl.col("cases").cast(pl.Int64),
            (pl.col("cases") >= MIN_CASES).alias("meets_threshold"),
            pl.col("peak_month").is_in([5, 6, 7]).alias("peak_may_jul"),
            pl.col("peak_month").is_in([1, 2, 3]).alias("peak_jan_mar"))
        return out.sort("cases", descending=True)

    hr = (lattice_hr.join(per_unit(["health_region_code"]),
                          on="health_region_code", how="left")
          .with_columns(pl.col("cases").fill_null(0),
                        pl.col("meets_threshold").fill_null(False),
                        pl.col("peak_may_jul").fill_null(False),
                        pl.col("peak_jan_mar").fill_null(False))
          .select(["health_region_code", "health_region_name", "uf_abbr",
                   "region", "cases", "peak_month", "peak_month_frac",
                   "resultant_r", "meets_threshold", "peak_may_jul",
                   "peak_jan_mar"])
          .sort(["region", "uf_abbr", "health_region_code"]))
    w(hr, "T4b_health_region_seasonality.csv")
    print(f"  health regions: {hr.height} total, "
          f"{int((hr['cases'] > 0).sum())} with >=1 confirmed case, "
          f"{int(hr['meets_threshold'].sum())} with >={MIN_CASES}")

    uf_meta = panel.select(["uf_abbr", "region"]).unique()
    st = (per_unit(["uf_abbr"]).join(uf_meta, on="uf_abbr", how="left")
          .select(["uf_abbr", "region", "cases", "peak_month",
                   "peak_month_frac", "resultant_r", "meets_threshold",
                   "peak_may_jul", "peak_jan_mar"])
          .sort(["region", "uf_abbr"]))
    w(st, "T4c_state_seasonality.csv")
    print("\n  state-grain seasonality (all 27 UFs):")
    print(st.select(["uf_abbr", "region", "cases", "peak_month",
                     "resultant_r", "meets_threshold"]), )

    # --- is the June peak GENERAL across the Northeast, or CONCENTRATED? ---
    def share_tbl(df: pl.DataFrame, unit: str) -> pl.DataFrame:
        q = df.filter(pl.col("meets_threshold"))
        rows = []
        for reg in sorted(df["region"].unique().to_list()):
            g = q.filter(pl.col("region") == reg)
            all_g = df.filter(pl.col("region") == reg)
            n_units = g.height
            if n_units == 0:
                continue
            k = int(g["peak_may_jul"].sum())
            # Clopper-Pearson on the fraction OF UNITS -- units, not cases, are
            # the sampling frame for "is the pattern general?".
            p, lo, hi = binom_ci(k, n_units)
            p, lo, hi = float(p), float(lo), float(hi)
            assert lo <= p <= hi
            cw = float(g.filter(pl.col("peak_may_jul"))["cases"].sum()
                       / g["cases"].sum())
            rows.append(dict(
                unit=unit, region=reg, units_total=all_g.height,
                units_qualifying=n_units,
                cases_total=int(all_g["cases"].sum()),
                cases_in_qualifying=int(g["cases"].sum()),
                case_coverage_of_qualifying=float(
                    g["cases"].sum() / all_g["cases"].sum()),
                units_peaking_may_jul=k, frac_units_may_jul=p,
                frac_lo=lo, frac_hi=hi,
                case_weighted_share_may_jul=cw,
                units_peaking_jan_mar=int(g["peak_jan_mar"].sum()),
                frac_units_jan_mar=float(g["peak_jan_mar"].mean()),
                median_r=float(g["resultant_r"].median())))
        return pl.DataFrame(rows)

    shares = pl.concat([share_tbl(hr, "health_region"),
                        share_tbl(st, "state")])
    w(shares, "T4d_may_jul_share_by_region.csv")
    print("\n  fraction of units peaking May-July, by macro-region:")
    print(shares)

    # --- peak-month histogram for Northeast health regions -----------------
    ne_hr = hr.filter((pl.col("region") == "Nordeste")
                      & pl.col("meets_threshold"))
    hist = (ne_hr.group_by("peak_month")
            .agg(pl.len().alias("n_health_regions"),
                 pl.col("cases").sum().alias("cases"))
            .sort("peak_month"))
    w(hist, "T4e_northeast_peak_month_histogram.csv")
    print("\n  Northeast health regions by peak month:")
    print(hist)

    # --- CONCENTRATION DIAGNOSTICS ----------------------------------------
    ne = panel.filter(pl.col("region") == "Nordeste")

    def circ_of(df: pl.DataFrame) -> dict:
        a = df.group_by("month").agg(pl.col("cases").sum().alias("n"))
        return circular_season(a["month"].to_numpy(), a["n"].to_numpy())

    base = circ_of(ne)
    conc = [dict(diagnostic="Northeast, all units", dropped="-",
                 cases=int(base["cases"]), peak_month=base["peak_month"],
                 peak_month_frac=base["peak_month_frac"],
                 resultant_r=base["resultant_r"])]
    for uf in sorted(ne["uf_abbr"].unique().to_list()):
        s = circ_of(ne.filter(pl.col("uf_abbr") != uf))
        conc.append(dict(diagnostic="leave-one-state-out", dropped=uf,
                         cases=int(s["cases"]), peak_month=s["peak_month"],
                         peak_month_frac=s["peak_month_frac"],
                         resultant_r=s["resultant_r"]))
    top5 = (ne_hr.sort("cases", descending=True)
            .head(5)["health_region_code"].to_list())
    for code in top5:
        s = circ_of(ne.filter(pl.col("health_region_code") != code))
        nm = hr.filter(pl.col("health_region_code") == code)
        conc.append(dict(
            diagnostic="leave-one-health-region-out",
            dropped=f"{code} {nm['health_region_name'][0]} ({nm['uf_abbr'][0]})",
            cases=int(s["cases"]), peak_month=s["peak_month"],
            peak_month_frac=s["peak_month_frac"],
            resultant_r=s["resultant_r"]))
    # Drop the five largest together -- the sharpest form of the named threat.
    s = circ_of(ne.filter(~pl.col("health_region_code").is_in(top5)))
    conc.append(dict(diagnostic="drop the 5 largest health regions together",
                     dropped=", ".join(top5), cases=int(s["cases"]),
                     peak_month=s["peak_month"],
                     peak_month_frac=s["peak_month_frac"],
                     resultant_r=s["resultant_r"]))
    # The complement of the analysis set: every Northeast health region too
    # small to carry a peak month of its own, POOLED.  This is the direct test
    # of "the rest of the Northeast follows the national February/March
    # pattern" -- the remainder is exactly that rest, and pooling gives it the
    # case mass an individual small region lacks.
    qual = set(ne_hr["health_region_code"].to_list())
    s = circ_of(ne.filter(~pl.col("health_region_code").is_in(list(qual))))
    conc.append(dict(
        diagnostic=f"POOLED remainder: all NE health regions with <{MIN_CASES} "
                   f"cases",
        dropped=f"kept {ne['health_region_code'].n_unique() - len(qual)} "
                f"sub-threshold health regions",
        cases=int(s["cases"]), peak_month=s["peak_month"],
        peak_month_frac=s["peak_month_frac"], resultant_r=s["resultant_r"]))
    # The same pooled remainder taken one state at a time: if June were carried
    # by a single state, that state's small regions would be the only pooled
    # remainder peaking in mid-year.
    for uf in sorted(ne["uf_abbr"].unique().to_list()):
        sub = ne.filter((pl.col("uf_abbr") == uf)
                        & ~pl.col("health_region_code").is_in(list(qual)))
        if sub["cases"].sum() < 50:
            continue
        s = circ_of(sub)
        conc.append(dict(
            diagnostic="POOLED sub-threshold regions, one state",
            dropped=uf, cases=int(s["cases"]), peak_month=s["peak_month"],
            peak_month_frac=s["peak_month_frac"],
            resultant_r=s["resultant_r"]))
    conc_tbl = pl.DataFrame(conc)
    w(conc_tbl, "T4f_northeast_concentration.csv")
    print("\n  Northeast concentration diagnostics:")
    print(conc_tbl)

    # --- cluster bootstrap over health regions ----------------------------
    # Resampling HEALTH REGIONS (not cases) is the resampling unit that
    # respects the clustering the Rayleigh p-value ignores: it asks how much
    # the Northeast peak month depends on which health regions happen to be in
    # the sample.
    rng = np.random.default_rng(20260814)
    ne_by_hr = (ne.group_by(["health_region_code", "month"])
                .agg(pl.col("cases").sum().alias("n")))
    codes = ne_by_hr["health_region_code"].unique().sort().to_list()
    idx = {c: i for i, c in enumerate(codes)}
    mat = np.zeros((len(codes), 12))
    for c, m, n in ne_by_hr.iter_rows():
        mat[idx[c], int(m) - 1] += n
    months = np.arange(1, 13, dtype=float)
    B = 5000
    draws = rng.integers(0, len(codes), size=(B, len(codes)))
    peaks = np.empty(B, dtype=int)
    rs = np.empty(B)
    for b in range(B):
        wts = mat[draws[b]].sum(axis=0)
        s = circular_season(months, wts)
        peaks[b] = s["peak_month"]
        rs[b] = s["resultant_r"]
    boot = (pl.DataFrame({"peak_month": peaks})
            .group_by("peak_month").agg(pl.len().alias("replicates"))
            .with_columns((pl.col("replicates") / B).alias("share"))
            .sort("peak_month"))
    w(boot, "T4g_northeast_cluster_bootstrap.csv")
    print(f"\n  cluster bootstrap over {len(codes)} Northeast health regions, "
          f"B={B}:")
    print(boot)
    boot_may_jul = float(np.mean(np.isin(peaks, [5, 6, 7])))
    print(f"  bootstrap share with an aggregate peak in May-July: "
          f"{boot_may_jul:.4f}")
    print(f"  bootstrap r: median {np.median(rs):.4f} "
          f"[{np.percentile(rs, 2.5):.4f}, {np.percentile(rs, 97.5):.4f}]")

    # --- threshold sensitivity (my choice, so I report its footprint) ------
    # NAMED THREAT: the May-July fraction is an artefact of MIN_CASES = 100.
    rows = []
    for thr in (50, 100, 200, 500):
        g = hr.filter((pl.col("region") == "Nordeste") & (pl.col("cases") >= thr))
        k, n = int(g["peak_may_jul"].sum()), g.height
        p, lo, hi = binom_ci(k, n)
        p, lo, hi = float(p), float(lo), float(hi)
        assert lo <= p <= hi
        rows.append(dict(min_cases=thr, ne_health_regions=n,
                         peaking_may_jul=k, frac=p, frac_lo=lo, frac_hi=hi,
                         cases_covered=int(g["cases"].sum())))
    thr_tbl = pl.DataFrame(rows)
    w(thr_tbl, "T4h_threshold_sensitivity.csv")
    print("\n  threshold sensitivity for the Northeast May-July fraction:")
    print(thr_tbl)

    return {"hr": hr, "st": st, "shares": shares, "conc": conc_tbl,
            "boot": boot, "boot_may_jul": boot_may_jul, "thr": thr_tbl,
            "hist": hist, "val": val}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    thread_t3()
    thread_t4()
    rule("done")
    print(f"  results in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
