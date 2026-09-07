"""What the minimum-case rule removes, and whether the answer depends on it.

The earlier draft dropped health regions with fewer than 30 confirmed cases and
disclosed it in one sentence: "212 of 439 regions, 96.2% of cases". A reviewer
objected that this makes the analysis representative of reported case **mass**
rather than of Brazilian **territory**, and — the sharper point — that a region
may hold few confirmed cases *because* its ascertainment is narrow, which is the
study's own subject matter and not a nuisance to be deleted.

The objection has two halves and they need different answers.

*Half one: how much territory leaves.* The draft never gave a population share,
only a case share, and the two are not close. Case share is 96.4%; the retained
regions hold 76.8% of national person-time. Roughly a quarter of Brazil's
population-time is outside the analysis at the 30-case rule, and the sentence
"96.2% of cases" conceals that entirely.

*Half two: is the excluded tail informative rather than noisy?* It is, and it
points the way the thesis predicts. Excluded regions have a **higher** H (0.784
vs 0.720), a **higher** reported case fatality (10.6% vs 9.7%) and an eight-fold
lower reported incidence rate. Across all regions the rank correlation between
case count and H is negative. Regions with few confirmed cases are, on average,
regions finding only severe illness. Deleting them removes the end of the
exposure distribution where the claim is strongest, so the exclusion is
conservative — but it is conservative by accident, and a reader is entitled to
see the estimate without it.

**Where the person-time comes from, and why it is rebuilt here.** A population
share is only as good as its denominator, and the denominator for this question
has to cover every health region over every year — including the 7 regions that
never recorded a single confirmed case, which by construction cannot appear in a
panel keyed on notifications. Person-time is therefore rebuilt from
``population_municipal_year`` over the full 2007–2025 window for all 439
regions. For the 432 regions that do appear in ``region_totals.parquet`` the
rebuild reconciles to the stored column exactly (asserted below); the remaining
7 contribute 11.4 million person-years, 0.30% of the national total, which the
panel cannot carry and which belong on the excluded side of every threshold.

**What would falsify the claims made here.** (i) If the excluded regions had a
*lower* H and a *lower* case fatality than the retained ones, the exclusion
would be discarding the benign end of the distribution and inflating the
gradient — the opposite of the reading above. (ii) If the case-fatality gradient
across H quintiles or the region-level Spearman rho moved materially with the
threshold, the headline result would be an artefact of the cut. (iii) If the
hierarchical binomial fitted with no exclusion at all returned an odds ratio for
H whose credible interval excluded the >=30-subset estimate on the *low* side,
the deletion would have been inflating the effect.

Model in (c) is a non-spatial partial-pooling device: a region iid intercept, no
neighbourhood structure. The spatial model is another agent's work and this is
deliberately not it.

Outputs to ``data/results/exclusion_sensitivity/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS

OUT = PATHS.results / "exclusion_sensitivity"
PANEL = PATHS.results / "analysis_panel"
#: renv is activated from the repository root, so the R subprocess has to run
#: there or INLA will not be on the library path.
REPO_ROOT = Path(__file__).resolve().parents[2]
YEAR_MIN, YEAR_MAX = 2007, 2025

#: The thresholds the reviewer asked to see side by side. 30 is the value the
#: earlier draft used; 10/20/50 bracket it. A fifth arm — no threshold at all —
#: is carried through every table under the label 0, because "what happens with
#: nothing excluded" is the question, not a footnote to it.
THRESHOLDS = (0, 10, 20, 30, 50)
REFERENCE_THRESHOLD = 30

#: Case-weighted quintiles, as in 40/62: equal case mass per band, so each
#: band's case fatality is estimated with comparable precision. Equal-count
#: bands would concentrate the national case load in one band.
N_BINS = 5

RSCRIPT = os.environ.get("BREPI_RSCRIPT", r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")

MACRO_REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def _geo() -> pl.DataFrame:
    return pl.read_parquet(
        PATHS.results / "atlas" / "municipality_atlas.parquet"
    ).select("munic_code", "health_region_code", "health_region_name",
             "uf_abbr", "region")


def _person_years_full() -> pl.DataFrame:
    """Full-window person-time for all 439 health regions.

    Not taken from ``region_totals.parquet``: that table is keyed on
    notifications and so cannot carry the regions that never notified one. The
    denominator of a territory does not depend on whether a case was reported
    in it.
    """
    geo = _geo()
    return (
        pl.read_parquet(PATHS.interim / "population_municipal_year.parquet")
        .filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        .join(geo, on="munic_code", how="inner")
        .group_by(["health_region_code", "health_region_name", "uf_abbr", "region"])
        .agg(pl.col("population").sum().alias("person_years"))
    )


def _regions() -> tuple[pl.DataFrame, dict]:
    """Every health region, with counts filled to zero where nothing was reported."""
    py = _person_years_full()
    counts = pl.read_parquet(PANEL / "region_totals.parquet").select(
        "health_region_code", "cases", "deaths", "outcome_known", "hospitalised",
        "hosp_known", "hk_hosp", "d_hosp", "years_observed",
        pl.col("person_years").alias("person_years_case_years_only"),
    )
    d = py.join(counts, on="health_region_code", how="left").with_columns(
        pl.col(["cases", "deaths", "outcome_known", "hospitalised", "hosp_known",
                "hk_hosp", "d_hosp", "years_observed",
                "person_years_case_years_only"]).fill_null(0)
    ).with_columns(
        # H is undefined, not zero, where no case has a decoded ate_hosp.
        pl.when(pl.col("hosp_known") > 0)
          .then(pl.col("hospitalised") / pl.col("hosp_known"))
          .otherwise(None).alias("H"),
        pl.when(pl.col("outcome_known") > 0)
          .then(pl.col("deaths") / pl.col("outcome_known"))
          .otherwise(None).alias("cfr"),
        (1e5 * pl.col("cases") / pl.col("person_years")).alias("incidence_per_100k"),
        pl.when(pl.col("cases") > 0)
          .then(pl.col("outcome_known") / pl.col("cases"))
          .otherwise(None).alias("outcome_completeness"),
    ).sort("health_region_code")

    # Reconciliation, asserted rather than assumed: for every region the panel
    # does carry, the independently rebuilt person-time must equal the stored
    # column to the person-year. If it ever stops matching, the two are keyed
    # on different geographies or different windows and every share below is
    # wrong.
    with_cases = d.filter(pl.col("cases") > 0)
    gap = (with_cases["person_years"]
           - with_cases["person_years_case_years_only"]).abs().max()
    assert int(gap) == 0, f"person-time disagrees with region_totals by up to {gap}"

    silent = d.filter(pl.col("cases") == 0)
    recon = {
        "national_person_years_2007_2025": int(d["person_years"].sum()),
        "health_regions": d.height,
        "health_regions_in_region_totals": with_cases.height,
        "max_abs_person_year_gap_vs_region_totals": int(gap),
        "regions_with_no_confirmed_case_ever": silent.height,
        "their_person_years": int(silent["person_years"].sum()),
        "their_share_of_national_person_years_pct": float(
            100 * silent["person_years"].sum() / d["person_years"].sum()),
        "note": (
            "regions that never notified a confirmed case cannot appear in a "
            "notification-keyed panel; they are territory the study covers and "
            "are counted on the excluded side of every threshold here"
        ),
    }
    return d, recon


# --------------------------------------------------------------------------
# (a) What is being excluded
# --------------------------------------------------------------------------

def _stratum_row(g: pl.DataFrame, total: pl.DataFrame, threshold: int,
                 arm: str) -> dict:
    cases = int(g["cases"].sum())
    py = float(g["person_years"].sum())
    deaths = int(g["deaths"].sum())
    ok = int(g["outcome_known"].sum())
    hosp = int(g["hospitalised"].sum())
    hk = int(g["hosp_known"].sum())

    inc, inc_lo, inc_hi = poisson_ci(cases, py, scale=1e5)
    if ok > 0:
        cfr, cfr_lo, cfr_hi = binom_ci(deaths, ok)
    else:
        cfr = cfr_lo = cfr_hi = float("nan")
    if hk > 0:
        h, h_lo, h_hi = binom_ci(hosp, hk)
    else:
        h = h_lo = h_hi = float("nan")

    return {
        "threshold": threshold,
        "arm": arm,
        "health_regions": g.height,
        "pct_of_health_regions": 100 * g.height / total.height,
        "person_years": py,
        "pct_of_person_years": 100 * py / float(total["person_years"].sum()),
        "cases": cases,
        "pct_of_cases": 100 * cases / int(total["cases"].sum()),
        "incidence_per_100k": float(inc),
        "incidence_lo": float(inc_lo),
        "incidence_hi": float(inc_hi),
        "deaths": deaths,
        "outcome_known": ok,
        "regions_with_estimable_cfr": int((g["outcome_known"] > 0).sum()),
        "cfr_pct": 100 * float(cfr),
        "cfr_lo_pct": 100 * float(cfr_lo),
        "cfr_hi_pct": 100 * float(cfr_hi),
        "hosp_known": hk,
        "H": float(h),
        "H_lo": float(h_lo),
        "H_hi": float(h_hi),
        "outcome_completeness_pct": 100 * ok / cases if cases else float("nan"),
    }


def threshold_table(d: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for t in THRESHOLDS:
        ex = d.filter(pl.col("cases") < t)
        ke = d.filter(pl.col("cases") >= t)
        if ex.height:
            rows.append(_stratum_row(ex, d, t, "excluded"))
        rows.append(_stratum_row(ke, d, t, "retained"))
    return pl.DataFrame(rows)


def macro_composition(d: pl.DataFrame) -> pl.DataFrame:
    """Macro-region composition of the excluded set, two ways.

    ``pct_of_excluded_person_years`` says what the excluded block is made of;
    ``pct_of_own_macroregion_person_years`` says how much of each macro-region
    the rule deletes. The second is the one that shows whether the exclusion is
    geographically even, and it is not.
    """
    rows = []
    macro_py = d.group_by("region").agg(
        pl.col("person_years").sum().alias("macro_py"),
        pl.col("cases").sum().alias("macro_cases"),
        pl.len().alias("macro_regions"),
    )
    for t in THRESHOLDS:
        if t == 0:
            continue
        ex = d.filter(pl.col("cases") < t)
        g = ex.group_by("region").agg(
            pl.len().alias("health_regions"),
            pl.col("person_years").sum().alias("person_years"),
            pl.col("cases").sum().alias("cases"),
            pl.col("deaths").sum().alias("deaths"),
            pl.col("outcome_known").sum().alias("outcome_known"),
            pl.col("hospitalised").sum().alias("hospitalised"),
            pl.col("hosp_known").sum().alias("hosp_known"),
        ).join(macro_py, on="region", how="left")
        tot_py = float(ex["person_years"].sum())
        for r in g.iter_rows(named=True):
            rows.append({
                "threshold": t,
                "region": r["region"],
                "health_regions_excluded": r["health_regions"],
                "pct_of_own_macroregion_regions":
                    100 * r["health_regions"] / r["macro_regions"],
                "person_years": float(r["person_years"]),
                "pct_of_excluded_person_years": 100 * r["person_years"] / tot_py,
                "pct_of_own_macroregion_person_years":
                    100 * r["person_years"] / r["macro_py"],
                "cases": r["cases"],
                "pct_of_own_macroregion_cases":
                    100 * r["cases"] / r["macro_cases"] if r["macro_cases"] else None,
                "H": (r["hospitalised"] / r["hosp_known"]) if r["hosp_known"] else None,
                "cfr_pct": (100 * r["deaths"] / r["outcome_known"])
                    if r["outcome_known"] else None,
            })
    return pl.DataFrame(rows).sort(["threshold", "region"])


def informativeness(d: pl.DataFrame) -> dict:
    """Is the excluded tail noise, or is it the narrow-ascertainment end?

    The reviewer's claim in testable form: if few confirmed cases is partly a
    symptom of narrow ascertainment, then case count should rank *negatively*
    against H, and the excluded block should sit at higher H and higher case
    fatality than the retained one. If instead the excluded regions were simply
    small and otherwise ordinary, both correlations would be near zero.
    """
    w = d.filter((pl.col("cases") > 0) & pl.col("H").is_not_null())
    wc = w.filter(pl.col("cfr").is_not_null())
    r_cases = stats.spearmanr(w["cases"].to_numpy(), w["H"].to_numpy())
    r_inc = stats.spearmanr(w["incidence_per_100k"].to_numpy(), w["H"].to_numpy())
    r_comp = stats.spearmanr(w["cases"].to_numpy(),
                             w["outcome_completeness"].to_numpy())
    return {
        "n_regions": w.height,
        "spearman_cases_vs_H": [float(r_cases.statistic), float(r_cases.pvalue)],
        "spearman_incidence_vs_H": [float(r_inc.statistic), float(r_inc.pvalue)],
        "spearman_cases_vs_outcome_completeness": [
            float(r_comp.statistic), float(r_comp.pvalue)],
        "n_regions_with_cfr": wc.height,
        "reading": (
            "case count ranks negatively against H: regions with few confirmed "
            "cases tend to be regions recording a higher hospitalised fraction, "
            "i.e. narrower ascertainment. The excluded tail is the informative "
            "end of the exposure, not statistical noise."
        ),
    }


# --------------------------------------------------------------------------
# (b) Does the answer move?
# --------------------------------------------------------------------------

def gradient_at(d: pl.DataFrame, threshold: int) -> tuple[pl.DataFrame, dict]:
    """Case-fatality by case-weighted quintile of H, on the retained set."""
    elig = d.filter(
        (pl.col("cases") >= max(threshold, 1))
        & pl.col("H").is_not_null()
        & (pl.col("outcome_known") > 0)
    )
    q = elig.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("quintile")
    )
    g = q.group_by("quintile").agg(
        pl.len().alias("health_regions"),
        pl.col("H").median().alias("H_median"),
        pl.col("cases").sum().alias("cases"),
        pl.col("deaths").sum().alias("deaths"),
        pl.col("outcome_known").sum().alias("outcome_known"),
        pl.col("person_years").sum().alias("person_years"),
    ).sort("quintile")

    cfr, lo, hi = binom_ci(g["deaths"].to_numpy(),
                          g["outcome_known"].to_numpy().astype(float))
    inc, ilo, ihi = poisson_ci(g["cases"].to_numpy(),
                              g["person_years"].to_numpy().astype(float), scale=1e5)
    g = g.with_columns(
        pl.lit(threshold).alias("threshold"),
        pl.Series("cfr_pct", 100 * cfr),
        pl.Series("cfr_lo_pct", 100 * lo),
        pl.Series("cfr_hi_pct", 100 * hi),
        pl.Series("incidence_per_100k", inc),
        pl.Series("incidence_lo", ilo),
        pl.Series("incidence_hi", ihi),
    )

    rho = stats.spearmanr(elig["H"].to_numpy(), elig["cfr"].to_numpy())
    row = g.to_dicts()
    summary = {
        "threshold": threshold,
        "health_regions": elig.height,
        "cases": int(elig["cases"].sum()),
        "pct_of_national_person_years": float(
            100 * elig["person_years"].sum() / d["person_years"].sum()),
        "cfr_Q1_pct": row[0]["cfr_pct"],
        "cfr_Q1_lo_pct": row[0]["cfr_lo_pct"],
        "cfr_Q1_hi_pct": row[0]["cfr_hi_pct"],
        "cfr_Q5_pct": row[-1]["cfr_pct"],
        "cfr_Q5_lo_pct": row[-1]["cfr_lo_pct"],
        "cfr_Q5_hi_pct": row[-1]["cfr_hi_pct"],
        "cfr_gradient_Q5_over_Q1": row[-1]["cfr_pct"] / row[0]["cfr_pct"],
        "H_median_Q1": row[0]["H_median"],
        "H_median_Q5": row[-1]["H_median"],
        "spearman_H_vs_cfr": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
    }
    return g, summary


# --------------------------------------------------------------------------
# (c) No exclusion at all: hierarchical binomial with partial pooling
# --------------------------------------------------------------------------

#: Written to the results directory so the fit is auditable alongside its
#: output rather than hidden in a temporary file. R-INLA is used because it is
#: the toolchain already installed here; the effect is an iid region intercept,
#: deliberately NOT the spatial (BYM) model another agent is fitting.
INLA_SCRIPT = r"""
suppressMessages(library(INLA))
suppressMessages(library(jsonlite))

args   <- commandArgs(trailingOnly = TRUE)
f_reg  <- args[1]; f_ry <- args[2]; f_out <- args[3]
reg    <- read.csv(f_reg, stringsAsFactors = FALSE)
ry     <- read.csv(f_ry,  stringsAsFactors = FALSE)

# Vague priors on the fixed effects; the iid precision keeps INLA's default
# log-gamma. Nothing here is trying to be informative -- the point of the model
# is partial pooling of small regions, not prior-driven shrinkage of the slope.
CTRL_FIXED <- list(mean = 0, prec = 0.01, mean.intercept = 0, prec.intercept = 0.01)

qsum <- function(m, name, scale) {
  mg <- m$marginals.fixed[[name]]
  q  <- inla.qmarginal(c(0.025, 0.5, 0.975), mg)
  # exp() is monotone, so posterior quantiles of the odds ratio are the
  # exponentiated quantiles of the log-odds coefficient.
  list(or = exp(scale * q[2]), or_lo = exp(scale * q[1]), or_hi = exp(scale * q[3]))
}

sd_of <- function(m, hyp) {
  mg <- m$marginals.hyperpar[[hyp]]
  q  <- inla.qmarginal(c(0.025, 0.5, 0.975), mg)
  # precision -> sd is monotone decreasing, so the quantiles swap ends.
  list(sd = 1/sqrt(q[2]), sd_lo = 1/sqrt(q[3]), sd_hi = 1/sqrt(q[1]))
}

fit_region <- function(df, label) {
  df$id <- seq_len(nrow(df))
  m <- inla(deaths ~ H + f(id, model = "iid"),
            family = "binomial", Ntrials = df$outcome_known, data = df,
            control.fixed = CTRL_FIXED,
            control.compute = list(dic = TRUE, waic = TRUE))
  per1  <- qsum(m, "H", 1.0)
  per10 <- qsum(m, "H", 0.1)
  s     <- sd_of(m, "Precision for id")
  list(model = "region-level, iid region intercept", subset = label,
       n_rows = nrow(df), n_regions = nrow(df),
       cases = sum(df$cases), deaths = sum(df$deaths),
       outcome_known = sum(df$outcome_known),
       person_years = sum(df$person_years),
       intercept = unname(m$summary.fixed["(Intercept)", "mean"]),
       beta_H = unname(m$summary.fixed["H", "mean"]),
       beta_H_sd = unname(m$summary.fixed["H", "sd"]),
       or_per_unit_H = per1$or, or_lo = per1$or_lo, or_hi = per1$or_hi,
       or_per_10pp_H = per10$or, or_10pp_lo = per10$or_lo, or_10pp_hi = per10$or_hi,
       region_sd = s$sd, region_sd_lo = s$sd_lo, region_sd_hi = s$sd_hi,
       dic = m$dic$dic, waic = m$waic$waic)
}

fit_region_year <- function(df, label, xvar) {
  # One row per region-year. Two exposures are available and they answer
  # different questions:
  #
  #   H_region : the region's exposure over the whole window, constant within
  #              region. Because the region intercept is also constant within
  #              region and the binomial is closed under summation at constant
  #              p, this model is algebraically the region-level model plus a
  #              year effect -- it is run once as a lossless-aggregation check,
  #              not as independent evidence.
  #   H_year   : the region-year exposure. This one does not collapse: the
  #              region intercept is now identified from repeated observations
  #              rather than acting as pure binomial overdispersion, and the
  #              slope is driven by within-region movement in H over time.
  df$rid <- as.integer(factor(df$health_region_code))
  df$yid <- as.integer(factor(df$year))
  df$xx  <- df[[xvar]]
  df     <- df[!is.na(df$xx), ]
  m <- inla(deaths ~ xx + f(rid, model = "iid") + f(yid, model = "iid"),
            family = "binomial", Ntrials = df$outcome_known, data = df,
            control.fixed = CTRL_FIXED,
            control.compute = list(dic = TRUE, waic = TRUE))
  per1  <- qsum(m, "xx", 1.0)
  per10 <- qsum(m, "xx", 0.1)
  s     <- sd_of(m, "Precision for rid")
  list(model = sprintf("region-year (%s), iid region + iid year", xvar),
       subset = label,
       n_rows = nrow(df), n_regions = length(unique(df$health_region_code)),
       cases = sum(df$cases), deaths = sum(df$deaths),
       outcome_known = sum(df$outcome_known),
       person_years = NA,
       intercept = unname(m$summary.fixed["(Intercept)", "mean"]),
       beta_H = unname(m$summary.fixed["xx", "mean"]),
       beta_H_sd = unname(m$summary.fixed["xx", "sd"]),
       or_per_unit_H = per1$or, or_lo = per1$or_lo, or_hi = per1$or_hi,
       or_per_10pp_H = per10$or, or_10pp_lo = per10$or_lo, or_10pp_hi = per10$or_hi,
       region_sd = s$sd, region_sd_lo = s$sd_lo, region_sd_hi = s$sd_hi,
       dic = m$dic$dic, waic = m$waic$waic)
}

fits <- list()
fits[[length(fits) + 1]] <- fit_region(reg, "no exclusion (all regions with >=1 case)")
for (t in c(10, 20, 30, 50)) {
  fits[[length(fits) + 1]] <- fit_region(reg[reg$cases >= t, ],
                                         sprintf("cases >= %d", t))
}
fits[[length(fits) + 1]] <- fit_region_year(
  ry, "no exclusion (all regions with >=1 case)", "H_region")
fits[[length(fits) + 1]] <- fit_region_year(
  ry, "no exclusion (all regions with >=1 case)", "H_year")
fits[[length(fits) + 1]] <- fit_region_year(
  ry[ry$cases >= 30, ], "cases >= 30", "H_year")

write(toJSON(fits, auto_unbox = TRUE, digits = 10, na = "null"), f_out)
cat("fits written:", length(fits), "\n")
"""


def _model_frames(d: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rows the binomial likelihood can actually use.

    A region with no decoded outcome contributes a Binomial(0, p) — no
    information, and INLA will not take a zero-trial row. A region with no
    decoded ate_hosp has no exposure value. Both are counted and reported
    rather than silently dropped; neither is an ascertainment-related
    exclusion, it is the absence of the measurement itself.
    """
    reg = d.filter(
        (pl.col("cases") > 0) & pl.col("H").is_not_null() & (pl.col("outcome_known") > 0)
    ).select("health_region_code", "health_region_name", "uf_abbr", "region",
             "cases", "deaths", "outcome_known", "H", "person_years")

    panel = pl.read_parquet(PANEL / "region_year_panel.parquet").select(
        "health_region_code", "year", "deaths", "outcome_known",
        "hospitalised", "hosp_known",
    ).filter(pl.col("outcome_known") > 0).with_columns(
        pl.when(pl.col("hosp_known") > 0)
          .then(pl.col("hospitalised") / pl.col("hosp_known"))
          .otherwise(None).alias("H_year")
    )
    ry = panel.join(
        reg.select("health_region_code", pl.col("H").alias("H_region"), "cases"),
        on="health_region_code", how="inner",
    ).select("health_region_code", "year", "deaths", "outcome_known", "cases",
             "H_region", "H_year")
    return reg, ry


def fit_hierarchical(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    reg, ry = _model_frames(d)
    f_reg = OUT / "model_frame_region.csv"
    f_ry = OUT / "model_frame_region_year.csv"
    f_r = OUT / "_inla_binomial.R"
    f_out = OUT / "inla_fits.json"
    reg.write_csv(f_reg)
    ry.write_csv(f_ry)
    f_r.write_text(INLA_SCRIPT, encoding="utf-8")

    proc = subprocess.run(
        [RSCRIPT, str(f_r), str(f_reg), str(f_ry), str(f_out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("INLA fit failed")

    fits = pl.DataFrame(json.loads(f_out.read_text(encoding="utf-8")))
    diag = {
        "regions_in_model": reg.height,
        "regions_with_a_case_but_no_decoded_outcome": int(
            d.filter((pl.col("cases") > 0) & (pl.col("outcome_known") == 0)).height),
        "regions_with_a_case_but_no_decoded_hospitalisation": int(
            d.filter((pl.col("cases") > 0) & pl.col("H").is_null()).height),
        "regions_with_no_confirmed_case_at_all": int(d.filter(pl.col("cases") == 0).height),
        "region_years_with_a_decoded_outcome": ry.height,
        "region_years_with_a_decoded_outcome_and_H": int(
            ry.filter(pl.col("H_year").is_not_null()).height),
    }
    return fits, diag


# --------------------------------------------------------------------------
# (d) Map input
# --------------------------------------------------------------------------

def inclusion_table(d: pl.DataFrame) -> pl.DataFrame:
    return d.select(
        "health_region_code", "health_region_name", "uf_abbr", "region",
        "cases", "deaths", "outcome_known", "hosp_known", "person_years",
        "H", "cfr", "incidence_per_100k", "outcome_completeness",
        "years_observed",
        *[(pl.col("cases") < t).alias(f"excluded_at_{t}") for t in (10, 20, 30, 50)],
    ).rename({"excluded_at_30": "excluded_at_30"}).sort("health_region_code")


# --------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, recon = _regions()

    assert d.height == 439, f"expected 439 health regions, got {d.height}"
    assert int(d["cases"].sum()) == 66358, "case total does not reconcile to the panel"

    tt = threshold_table(d)
    tt.write_csv(OUT / "exclusion_by_threshold.csv")
    comp = macro_composition(d)
    comp.write_csv(OUT / "excluded_macroregion_composition.csv")
    info = informativeness(d)

    grads, summaries = [], []
    for t in THRESHOLDS:
        g, s = gradient_at(d, t)
        grads.append(g)
        summaries.append(s)
    pl.concat(grads).write_csv(OUT / "gradient_by_threshold.csv")
    pl.DataFrame(summaries).write_csv(OUT / "gradient_summary_by_threshold.csv")

    fits, diag = fit_hierarchical(d)
    fits.write_csv(OUT / "hierarchical_model_fits.csv")

    incl = inclusion_table(d)
    incl.write_csv(OUT / "region_inclusion.csv")

    # ---- report ----------------------------------------------------------
    ex30 = tt.filter((pl.col("threshold") == 30) & (pl.col("arm") == "excluded")).to_dicts()[0]
    ke30 = tt.filter((pl.col("threshold") == 30) & (pl.col("arm") == "retained")).to_dicts()[0]
    f_all = fits.filter(
        (pl.col("model") == "region-level, iid region intercept")
        & pl.col("subset").str.starts_with("no exclusion")).to_dicts()[0]
    f_30 = fits.filter(
        (pl.col("model") == "region-level, iid region intercept")
        & (pl.col("subset") == "cases >= 30")).to_dicts()[0]
    ry_check = fits.filter(pl.col("model").str.contains("H_region")).to_dicts()[0]
    ry_all = fits.filter(
        pl.col("model").str.contains("H_year")
        & pl.col("subset").str.starts_with("no exclusion")).to_dicts()[0]
    ry_30 = fits.filter(
        pl.col("model").str.contains("H_year")
        & (pl.col("subset") == "cases >= 30")).to_dicts()[0]

    report = {
        "window": [YEAR_MIN, YEAR_MAX],
        "health_regions_total": d.height,
        "person_time_denominator": recon,
        "a_what_is_excluded_at_30": {
            "health_regions_excluded": ex30["health_regions"],
            "health_regions_retained": ke30["health_regions"],
            "excluded_person_year_share_pct": ex30["pct_of_person_years"],
            "excluded_case_share_pct": ex30["pct_of_cases"],
            "excluded_incidence_per_100k": [ex30["incidence_per_100k"],
                                            ex30["incidence_lo"], ex30["incidence_hi"]],
            "retained_incidence_per_100k": [ke30["incidence_per_100k"],
                                            ke30["incidence_lo"], ke30["incidence_hi"]],
            "excluded_cfr_pct": [ex30["cfr_pct"], ex30["cfr_lo_pct"], ex30["cfr_hi_pct"]],
            "retained_cfr_pct": [ke30["cfr_pct"], ke30["cfr_lo_pct"], ke30["cfr_hi_pct"]],
            "excluded_H": [ex30["H"], ex30["H_lo"], ex30["H_hi"]],
            "retained_H": [ke30["H"], ke30["H_lo"], ke30["H_hi"]],
            "excluded_outcome_known": ex30["outcome_known"],
            "excluded_regions_with_estimable_cfr": ex30["regions_with_estimable_cfr"],
        },
        "a_informativeness_of_the_excluded_tail": info,
        "b_gradient_by_threshold": summaries,
        "c_hierarchical_binomial": {
            "engine": "R-INLA, binomial likelihood, iid region intercept",
            "diagnostics": diag,
            "no_exclusion": {
                "or_per_10pp_H": [f_all["or_per_10pp_H"], f_all["or_10pp_lo"],
                                  f_all["or_10pp_hi"]],
                "or_per_unit_H": [f_all["or_per_unit_H"], f_all["or_lo"], f_all["or_hi"]],
                "n_regions": f_all["n_regions"],
                "region_sd": f_all["region_sd"],
            },
            "cases_ge_30": {
                "or_per_10pp_H": [f_30["or_per_10pp_H"], f_30["or_10pp_lo"],
                                  f_30["or_10pp_hi"]],
                "or_per_unit_H": [f_30["or_per_unit_H"], f_30["or_lo"], f_30["or_hi"]],
                "n_regions": f_30["n_regions"],
                "region_sd": f_30["region_sd"],
            },
            "region_year_time_varying_H_no_exclusion_or_per_10pp": [
                ry_all["or_per_10pp_H"], ry_all["or_10pp_lo"], ry_all["or_10pp_hi"]],
            "region_year_time_varying_H_ge_30_or_per_10pp": [
                ry_30["or_per_10pp_H"], ry_30["or_10pp_lo"], ry_30["or_10pp_hi"]],
            "lossless_aggregation_check": {
                "what": (
                    "region-year rows with the time-constant region H must "
                    "reproduce the region-level fit, because the binomial is "
                    "closed under summation at constant p"
                ),
                "region_level_beta_H": f_all["beta_H"],
                "region_year_beta_H_region": ry_check["beta_H"],
            },
        },
        "outputs": sorted(p.name for p in OUT.glob("*")),
    }
    (OUT / "exclusion_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console ---------------------------------------------------------
    print("=== (a) what the minimum-case rule removes ===")
    print(f"  national person-years 2007-2025, all {d.height} health regions: "
          f"{d['person_years'].sum():,}")
    print(f"  reconciles exactly with region_totals for its "
          f"{recon['health_regions_in_region_totals']} regions; the "
          f"{recon['regions_with_no_confirmed_case_ever']} regions that never "
          f"notified a confirmed case add "
          f"{recon['their_share_of_national_person_years_pct']:.2f}% of person-time")
    print()
    hdr = (f"  {'thr':>3} {'arm':<9} {'regions':>7} {'pop%':>6} {'case%':>6} "
           f"{'inc/100k':>9} {'CFR%':>6} {'H':>6}")
    print(hdr)
    for r in tt.iter_rows(named=True):
        print(f"  {r['threshold']:>3} {r['arm']:<9} {r['health_regions']:>7} "
              f"{r['pct_of_person_years']:>6.2f} {r['pct_of_cases']:>6.2f} "
              f"{r['incidence_per_100k']:>9.3f} {r['cfr_pct']:>6.2f} {r['H']:>6.3f}")

    print("\n  macro-region share of own person-time excluded at 30:")
    for r in comp.filter(pl.col("threshold") == 30).iter_rows(named=True):
        print(f"    {r['region']:<13} {r['health_regions_excluded']:>3} regions "
              f"({r['pct_of_own_macroregion_regions']:>5.1f}% of its regions), "
              f"{r['pct_of_own_macroregion_person_years']:>5.1f}% of its person-time, "
              f"H={r['H']:.3f}")

    print(f"\n  Spearman(cases, H) = {info['spearman_cases_vs_H'][0]:+.3f} "
          f"(p={info['spearman_cases_vs_H'][1]:.2g}, n={info['n_regions']}): "
          f"fewer cases go with narrower ascertainment")

    print("\n=== (b) does the answer move? ===")
    print(f"  {'thr':>3} {'regions':>7} {'pop%':>6} {'Q1 CFR%':>8} {'Q5 CFR%':>8} "
          f"{'Q5/Q1':>6} {'rho(H,CFR)':>11}")
    for s in summaries:
        print(f"  {s['threshold']:>3} {s['health_regions']:>7} "
              f"{s['pct_of_national_person_years']:>6.2f} "
              f"{s['cfr_Q1_pct']:>8.2f} {s['cfr_Q5_pct']:>8.2f} "
              f"{s['cfr_gradient_Q5_over_Q1']:>6.2f} "
              f"{s['spearman_H_vs_cfr']:>+11.3f}")

    print("\n=== (c) hierarchical binomial, partial pooling instead of deletion ===")
    for r in fits.iter_rows(named=True):
        print(f"  {r['model']:<38} {r['subset']:<40} n={r['n_rows']:>5} "
              f"OR/10pp H = {r['or_per_10pp_H']:.3f} "
              f"({r['or_10pp_lo']:.3f}-{r['or_10pp_hi']:.3f})  "
              f"sd_region={r['region_sd']:.3f}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
