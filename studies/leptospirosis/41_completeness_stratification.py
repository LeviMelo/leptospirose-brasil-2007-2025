"""Is the case-fatality geography an artefact of where outcomes get recorded?

Thread T-2. Case fatality is a proportion whose denominator is *cases with a
recorded outcome*, and the SINAN outcome field (``EVOLUCAO``) is not recorded
uniformly. Table T7c reports its completeness moving from 45.3% in 2007 to
70.5% in 2025 -- a 52.8 pp swing, the largest drift of any field. The RQ4 model
carries completeness as a covariate and the structural coefficients barely
move, but adjusting for a covariate is not the same as showing that a
missing-data mechanism is ignorable.

THE NAMED THREAT: the health-region case-fatality geography -- and in
particular the 3.0% -> 17.1% gradient across hospitalisation-share quintiles,
now the paper's central result -- is an artefact of *where and when* outcomes
happen to be recorded rather than of where people die.

The test is restriction, not adjustment. If the geography is manufactured by
differential recording, then throwing away every health-region-year whose
outcome recording is incomplete should scramble it. If the geography survives
on the subset where there is almost nothing to be missing about, the mechanism
cannot be doing the work.

One thing has to be settled before any of that, because it changes the size of
the threat: T7c is computed over all 328,984 *notifications*, of which 72% were
discarded. The case-fatality denominator is confirmed cases only. This script
recomputes the completeness of the outcome field on that population, which is
the one the estimand actually lives on.

Cells are rebuilt from the line level rather than lifted from the atlas,
because the atlas's health-region ``cfr`` divides a panel-derived death count
(infection municipality, onset year) by a line-level ``rq4_outcome_known``
(residence municipality, source year). Those are two different attributions of
the same cases; the mixture is fine to about a percent at national scale but
has no year dimension and cannot be restricted. Everything here is built from
one source with one attribution, replicating the RQ4 definitions exactly
(``23_rq4_lethality.R``), and reconciled against the atlas before use.

Outputs to ``data/results/completeness_stratification/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci
from brepi.config import PATHS

OUT = PATHS.results / "completeness_stratification"

#: Same inclusion rule as ``40_ascertainment_depth.py``, so that the restricted
#: estimates are compared against the published ones on like terms.
MIN_CASES = 30
MIN_KNOWN_OUTCOMES = 20

#: Completeness cuts applied at the health-region-YEAR cell. 1.0 is the
#: strongest available cut: a cell in which no confirmed case is missing an
#: outcome, so no missingness mechanism can operate inside it at all.
CUTS = [0.80, 0.90, 1.00]

#: A cell of 1 case with 1 recorded outcome has completeness 1.0 for free.
#: Selecting on completeness therefore selects on smallness unless this is
#: imposed as well; the ``_min10`` variants carry it.
MIN_CELL_CASES = 10

RNG = np.random.default_rng(20260814)
N_PERM = 999


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def cp(count, total) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clopper-Pearson, unpacked in the only correct order, then verified.

    ``binom_ci`` returns (ESTIMATE, lower, upper). Unpacking it as (lo, mid, hi)
    has already corrupted results in this repository once. The assertion makes
    a repeat a crash rather than a plausible-looking table.
    """
    est, lo, hi = binom_ci(np.asarray(count, float), np.asarray(total, float))
    ok = np.isfinite(est)
    if ok.any():
        assert np.all(est[ok] >= lo[ok] - 1e-12), "estimate below its lower bound"
        assert np.all(est[ok] <= hi[ok] + 1e-12), "estimate above its upper bound"
    return est, lo, hi


def spearman(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return {"rho": None, "p": None, "n": int(keep.sum())}
    r = stats.spearmanr(a[keep], b[keep])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(keep.sum())}


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

def build_cells() -> tuple[pl.DataFrame, dict]:
    """Health-region-year cells of confirmed cases, from the line level.

    Replicates ``23_rq4_lethality.R``: confirmed = ``classi_fin == 'confirmado'``,
    year = ``src_year``, geography = residence municipality crosswalked to the
    health region, death = ``evolucao == 'obito_por_leptospirose'``,
    outcome known = ``evolucao_state == 'valid'``, hospitalised =
    ``ate_hosp == 'sim'``.

    Unlike ``severity_panel()`` this keeps cells whose outcome-known count is
    zero. A health region-year that reported cases and recorded no outcome at
    all is precisely the observation this thread is about; dropping it would
    bias every completeness figure upward.
    """
    line = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter(pl.col("classi_fin") == "confirmado")
        .select(["src_year", "ID_MN_RESI", "evolucao", "evolucao_state", "ate_hosp"])
        .collect()
    )
    mun = (
        pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
        .select(["munic_code", "health_region_code", "health_region_name",
                 "uf_abbr", "region"])
        .with_columns(pl.col("munic_code").str.slice(0, 6).alias("munic6"))
        .drop("munic_code")
    )
    assert mun["munic6"].n_unique() == mun.height, "six-digit crosswalk is not unique"

    n_all = line.height
    line = line.join(mun, left_on="ID_MN_RESI", right_on="munic6", how="left")
    unmatched = int(line["health_region_code"].null_count())
    line = line.filter(pl.col("health_region_code").is_not_null())

    cells = (
        line.group_by(["health_region_code", "health_region_name", "uf_abbr",
                       "region", "src_year"])
        .agg(
            pl.len().alias("cases"),
            (pl.col("evolucao_state") == "valid").sum().alias("outcome_known"),
            (pl.col("evolucao") == "obito_por_leptospirose").sum().alias("deaths"),
            (pl.col("ate_hosp") == "sim").sum().alias("hospitalised"),
            pl.col("ate_hosp").is_not_null().sum().alias("hosp_field_known"),
        )
        .rename({"src_year": "year"})
        .sort(["health_region_code", "year"])
    )
    assert (cells["deaths"] <= cells["outcome_known"]).all(), (
        "a death is a known outcome; deaths exceeding outcome_known is a "
        "definition error")
    cells = cells.with_columns(
        (pl.col("outcome_known") / pl.col("cases")).alias("completeness"),
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share"),
    )
    meta = {
        "confirmed_cases": n_all,
        "without_a_health_region": unmatched,
        "pct_without_a_health_region": 100 * unmatched / n_all,
        "cases_in_cells": int(cells["cases"].sum()),
        "cells": int(cells.height),
    }
    return cells, meta


def reconcile(cells: pl.DataFrame) -> dict:
    """Does this rebuild agree with the atlas the published result was cut from?"""
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    mine = cells.group_by("health_region_code").agg(
        pl.col("cases").sum().alias("cases_line"),
        pl.col("deaths").sum().alias("deaths_line"),
        pl.col("hospitalised").sum().alias("hosp_line"),
        pl.col("outcome_known").sum().alias("known_line"),
    )
    j = hr.select(["health_region_code", "cases", "deaths", "hospitalised",
                   "rq4_outcome_known"]).join(mine, on="health_region_code", how="left")
    j = j.fill_null(0)
    out = {}
    for atlas_col, mine_col in [("cases", "cases_line"), ("deaths", "deaths_line"),
                                ("hospitalised", "hosp_line"),
                                ("rq4_outcome_known", "known_line")]:
        a, b = j[atlas_col].to_numpy().astype(float), j[mine_col].to_numpy().astype(float)
        out[atlas_col] = {
            "atlas_total": float(a.sum()), "rebuild_total": float(b.sum()),
            "pct_difference": float(100 * (b.sum() - a.sum()) / a.sum()),
            "spearman_across_health_regions": spearman(a, b),
        }
    return out


# ---------------------------------------------------------------------------
# 1. Where and when is the outcome recorded?
# ---------------------------------------------------------------------------

def completeness_by_year(cells: pl.DataFrame) -> pl.DataFrame:
    g = cells.group_by("year").agg(
        pl.col("cases").sum(), pl.col("outcome_known").sum(),
        pl.col("deaths").sum(), pl.col("hosp_field_known").sum(),
    ).sort("year")
    est, lo, hi = cp(g["outcome_known"], g["cases"])
    hest, hlo, hhi = cp(g["hosp_field_known"], g["cases"])
    cfr, cfr_lo, cfr_hi = cp(g["deaths"], g["outcome_known"])
    return g.with_columns(
        pl.Series("outcome_completeness_pct", 100 * est),
        pl.Series("outcome_completeness_lo", 100 * lo),
        pl.Series("outcome_completeness_hi", 100 * hi),
        pl.Series("ate_hosp_completeness_pct", 100 * hest),
        pl.Series("ate_hosp_completeness_lo", 100 * hlo),
        pl.Series("ate_hosp_completeness_hi", 100 * hhi),
        pl.Series("cfr_pct", 100 * cfr),
        pl.Series("cfr_lo", 100 * cfr_lo),
        pl.Series("cfr_hi", 100 * cfr_hi),
    )


def completeness_by_region(cells: pl.DataFrame) -> pl.DataFrame:
    """Health-region completeness over the whole period, with an exact interval."""
    g = cells.group_by(["health_region_code", "health_region_name", "uf_abbr",
                        "region"]).agg(
        pl.col("cases").sum(), pl.col("outcome_known").sum(),
        pl.col("deaths").sum(), pl.col("hospitalised").sum(),
        pl.len().alias("cells"),
        (pl.col("cases") > 0).sum().alias("cells_with_cases"),
    ).sort("health_region_code")
    est, lo, hi = cp(g["outcome_known"], g["cases"])
    cfr, cfr_lo, cfr_hi = cp(g["deaths"], g["outcome_known"])
    hs, hs_lo, hs_hi = cp(g["hospitalised"], g["cases"])
    return g.with_columns(
        pl.Series("completeness", est),
        pl.Series("completeness_lo", lo), pl.Series("completeness_hi", hi),
        pl.Series("cfr", cfr), pl.Series("cfr_lo", cfr_lo), pl.Series("cfr_hi", cfr_hi),
        pl.Series("hosp_share", hs),
        pl.Series("hosp_share_lo", hs_lo), pl.Series("hosp_share_hi", hs_hi),
    )


def read_adjacency(path: Path) -> tuple[int, list[list[int]]]:
    """Parse an INLA .adj file into zero-based neighbour lists."""
    tokens = path.read_text().split()
    n = int(tokens[0])
    nb: list[list[int]] = [[] for _ in range(n)]
    i = 1
    while i < len(tokens):
        node = int(tokens[i]) - 1
        k = int(tokens[i + 1])
        nb[node] = [int(t) - 1 for t in tokens[i + 2: i + 2 + k] if int(t) > 0]
        i += 2 + k
    return n, nb


def morans_i(values: np.ndarray, nb: list[list[int]]) -> dict:
    """Row-standardised Moran's I with a permutation reference distribution.

    Permutation rather than the normal approximation because the statistic is
    computed on a subset of the lattice with an irregular neighbour count.
    """
    z = values - values.mean()
    denom = float((z ** 2).sum())
    pairs = [(i, j) for i, ns in enumerate(nb) for j in ns]
    if not pairs or denom == 0:
        return {"morans_i": None, "note": "no edges among retained units"}
    wsum = 0.0
    rowdeg = np.array([len(ns) for ns in nb], float)

    def statistic(zv: np.ndarray) -> float:
        num = 0.0
        s = 0.0
        for i, ns in enumerate(nb):
            if not ns:
                continue
            w = 1.0 / len(ns)
            for j in ns:
                num += w * zv[i] * zv[j]
                s += w
        return (len(zv) / s) * (num / float((zv ** 2).sum()))

    obs = statistic(z)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = RNG.permutation(values)
        null[b] = statistic(perm - perm.mean())
    p = (1 + int((null >= obs).sum())) / (N_PERM + 1)
    wsum = float(rowdeg[rowdeg > 0].size)
    return {
        "morans_i": float(obs),
        "permutation_p_one_sided_positive": float(p),
        "permutation_mean": float(null.mean()),
        "permutation_sd": float(null.std(ddof=1)),
        "n_units": int(values.size),
        "units_with_at_least_one_neighbour": int(wsum),
        "n_permutations": N_PERM,
    }


def spatial_structure(cells: pl.DataFrame, reg: pl.DataFrame) -> dict:
    """Is completeness clustered in space, and how does that compare with time?

    Two separate questions and two separate statistics:

    * Moran's I on the health-region completeness surface -- do neighbouring
      regions record outcomes at similar rates?
    * A one-way variance decomposition on health-region-year cells -- does
      knowing the region tell you more about a cell's completeness than knowing
      the year? If the drift T7c reports were the dominant axis, year would
      win. The response is on the empirical logit scale because completeness
      piles up at 1.0 and a raw-scale variance there is not interpretable.
    """
    out: dict = {}

    codes = sorted(
        pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
        ["health_region_code"].unique().to_list()
    )
    n_graph, nb_all = read_adjacency(PATHS.panel / "graphs" / "health_region.adj")
    assert n_graph == len(codes), (
        f"graph has {n_graph} nodes but {len(codes)} health regions; the "
        "node order assumption (sorted health_region_code) is unsafe")

    keep = reg.filter(pl.col("cases") >= MIN_CASES)
    idx = {c: i for i, c in enumerate(codes)}
    sel = [idx[c] for c in keep["health_region_code"].to_list()]
    pos = {g: k for k, g in enumerate(sel)}
    nb_sub = [[pos[j] for j in nb_all[g] if j in pos] for g in sel]
    out["morans_i_completeness"] = morans_i(
        keep["completeness"].to_numpy(), nb_sub)
    out["morans_i_completeness"]["restriction"] = f"health regions with >= {MIN_CASES} cases"
    out["morans_i_case_fatality"] = morans_i(keep["cfr"].to_numpy(), nb_sub)
    out["morans_i_hosp_share"] = morans_i(keep["hosp_share"].to_numpy(), nb_sub)

    # Variance decomposition on cells.
    c = cells.filter(pl.col("cases") >= 20)
    k = c["outcome_known"].to_numpy().astype(float)
    n = c["cases"].to_numpy().astype(float)
    y = np.log((k + 0.5) / (n - k + 0.5))
    total_ss = float(((y - y.mean()) ** 2).sum())

    def between_ss(labels: np.ndarray) -> float:
        s = 0.0
        for lab in np.unique(labels):
            m = labels == lab
            s += m.sum() * (y[m].mean() - y.mean()) ** 2
        return float(s)

    hr_lab = np.array(c["health_region_code"].to_list())
    yr_lab = c["year"].to_numpy()
    out["variance_decomposition_empirical_logit_completeness"] = {
        "cells": int(c.height),
        "restriction": "health-region-years with >= 20 confirmed cases",
        "total_sum_of_squares": total_ss,
        "share_explained_by_health_region": between_ss(hr_lab) / total_ss,
        "share_explained_by_year": between_ss(yr_lab) / total_ss,
        "n_health_regions": int(np.unique(hr_lab).size),
        "n_years": int(np.unique(yr_lab).size),
    }

    # Threat: the space-beats-time split is an artefact of restricting to the
    # 758 largest cells. Repeat once at a looser cell size. Not a sweep -- one
    # named objection, one answer.
    c2 = cells.filter(pl.col("cases") >= 10)
    k2 = c2["outcome_known"].to_numpy().astype(float)
    n2 = c2["cases"].to_numpy().astype(float)
    y2 = np.log((k2 + 0.5) / (n2 - k2 + 0.5))
    tss2 = float(((y2 - y2.mean()) ** 2).sum())

    def between_ss2(labels: np.ndarray) -> float:
        s = 0.0
        for lab in np.unique(labels):
            m = labels == lab
            s += m.sum() * (y2[m].mean() - y2.mean()) ** 2
        return float(s)

    out["variance_decomposition_min10_cells"] = {
        "cells": int(c2.height),
        "restriction": "health-region-years with >= 10 confirmed cases",
        "share_explained_by_health_region": between_ss2(
            np.array(c2["health_region_code"].to_list())) / tss2,
        "share_explained_by_year": between_ss2(c2["year"].to_numpy()) / tss2,
    }
    out["completeness_distribution_across_health_regions"] = {
        "n_health_regions": int(keep.height),
        "min": float(keep["completeness"].min()),
        "p10": float(keep["completeness"].quantile(0.10)),
        "median": float(keep["completeness"].median()),
        "p90": float(keep["completeness"].quantile(0.90)),
        "max": float(keep["completeness"].max()),
        "n_below_0.80": int(keep.filter(pl.col("completeness") < 0.80).height),
        "n_below_0.90": int(keep.filter(pl.col("completeness") < 0.90).height),
    }
    return out


# ---------------------------------------------------------------------------
# 2 & 3. Restriction
# ---------------------------------------------------------------------------

def restrict(cells: pl.DataFrame, cut: float, min_cell_cases: int = 0) -> pl.DataFrame:
    return cells.filter(
        (pl.col("cases") >= max(min_cell_cases, 1))
        & (pl.col("completeness") >= cut - 1e-12)
    )


def region_totals(sub: pl.DataFrame) -> pl.DataFrame:
    g = sub.group_by("health_region_code").agg(
        pl.col("cases").sum(), pl.col("outcome_known").sum(),
        pl.col("deaths").sum(), pl.col("hospitalised").sum(),
        pl.len().alias("cells"),
    )
    est, lo, hi = cp(g["deaths"], g["outcome_known"])
    hs, _, _ = cp(g["hospitalised"], g["cases"])
    return g.with_columns(
        pl.Series("cfr", est), pl.Series("cfr_lo", lo), pl.Series("cfr_hi", hi),
        pl.Series("hosp_share", hs),
    )


def geography_survival(cells: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Does the ranking of health regions by case fatality survive restriction?"""
    full = region_totals(cells).filter(
        (pl.col("cases") >= MIN_CASES) & (pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES))
    rows, summary = [], {"full": {
        "n_health_regions": int(full.height),
        "cases": int(full["cases"].sum()),
        "known_outcomes": int(full["outcome_known"].sum()),
        "deaths": int(full["deaths"].sum()),
    }}
    est, lo, hi = cp([int(full["deaths"].sum())], [int(full["outcome_known"].sum())])
    summary["full"]["pooled_case_fatality_pct"] = [100 * float(est[0]),
                                                   100 * float(lo[0]), 100 * float(hi[0])]

    for cut in CUTS:
        for mcc in (0, MIN_CELL_CASES):
            sub = restrict(cells, cut, mcc)
            r = region_totals(sub).filter(
                (pl.col("cases") >= MIN_CASES)
                & (pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES))
            j = full.select(["health_region_code", "cfr"]).rename({"cfr": "cfr_full"}).join(
                r.select(["health_region_code", "cfr", "cases", "outcome_known",
                          "deaths", "cells"]).rename({"cfr": "cfr_restricted"}),
                on="health_region_code", how="inner")
            sp = spearman(j["cfr_full"].to_numpy(), j["cfr_restricted"].to_numpy())
            pe, ple, phe = cp([int(r["deaths"].sum())], [int(r["outcome_known"].sum())])
            rows.append({
                "cut": cut, "min_cell_cases": mcc,
                "cells_retained": int(sub.height),
                "cells_total": int(cells.filter(pl.col("cases") > 0).height),
                "cases_retained": int(sub["cases"].sum()),
                "cases_total": int(cells["cases"].sum()),
                "case_retention_pct": 100 * float(sub["cases"].sum()) / float(cells["cases"].sum()),
                "health_regions_retained": int(r.height),
                "health_regions_full": int(full.height),
                "health_regions_paired": int(j.height),
                "spearman_rho_full_vs_restricted": sp["rho"],
                "spearman_p": sp["p"],
                "restricted_pooled_cfr_pct": 100 * float(pe[0]),
                "restricted_pooled_cfr_lo": 100 * float(ple[0]),
                "restricted_pooled_cfr_hi": 100 * float(phe[0]),
                "restricted_completeness_pct": 100 * float(sub["outcome_known"].sum())
                                               / float(sub["cases"].sum()),
            })
            summary[f"cut_{cut}_mincell_{mcc}"] = sp

    # Threat: a rank correlation between a full estimate and a SUBSET of itself
    # is mechanically inflated -- the two share most of their deaths. The
    # non-overlapping version splits each region's own cells into the
    # well-recorded and the poorly-recorded half and correlates those two
    # disjoint case-fatality estimates. If differential recording manufactured
    # the geography, the halves would disagree.
    disjoint = []
    for cut in CUTS[:2]:
        hi_ = region_totals(cells.filter(pl.col("completeness") >= cut - 1e-12))
        lo_ = region_totals(cells.filter(pl.col("completeness") < cut - 1e-12))
        j = (hi_.filter(pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES)
             .select(["health_region_code", "cfr", "outcome_known"])
             .rename({"cfr": "cfr_high_completeness_cells",
                      "outcome_known": "known_high"})
             .join(lo_.filter(pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES)
                   .select(["health_region_code", "cfr", "outcome_known"])
                   .rename({"cfr": "cfr_low_completeness_cells",
                            "outcome_known": "known_low"}),
                   on="health_region_code", how="inner"))
        sp = spearman(j["cfr_high_completeness_cells"].to_numpy(),
                      j["cfr_low_completeness_cells"].to_numpy())
        disjoint.append({
            "cut": cut, "health_regions_with_both_halves": int(j.height),
            "spearman_rho": sp["rho"], "spearman_p": sp["p"],
            "known_outcomes_high_half": int(j["known_high"].sum()),
            "known_outcomes_low_half": int(j["known_low"].sum()),
        })
    summary["disjoint_split_no_shared_deaths"] = disjoint
    return pl.DataFrame(rows), summary


def case_weighted_quintiles(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Bands holding equal case mass -- identical rule to 40_ascertainment_depth."""
    d = df.sort(column).with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw"))
    return d.with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")).drop("_cw")


def gradient(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Case fatality by hospitalisation-share quintile, primitives summed first."""
    q = case_weighted_quintiles(df, "hosp_share")
    g = q.group_by("quintile").agg(
        pl.len().alias("health_regions"),
        pl.col("hosp_share").median().alias("band_median_hosp_share"),
        pl.col("cases").sum(), pl.col("deaths").sum(),
        pl.col("hospitalised").sum(), pl.col("outcome_known").sum(),
    ).sort("quintile")
    cfr, lo, hi = cp(g["deaths"], g["outcome_known"])
    hs, hlo, hhi = cp(g["hospitalised"], g["cases"])
    comp, _, _ = cp(g["outcome_known"], g["cases"])
    return g.with_columns(
        pl.lit(label).alias("variant"),
        pl.Series("cfr_pct", 100 * cfr), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("hosp_share_lo", 100 * hlo), pl.Series("hosp_share_hi", 100 * hhi),
        pl.Series("outcome_completeness_pct", 100 * comp),
    )


def gradient_survival(cells: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """The key check: does 3.0% -> 17.1% survive restriction?

    Two variants, because they answer different objections.

    * *fixed bands* keeps the published band definition (each region's
      hospitalisation share over all its years) and only restricts which cells
      contribute deaths and known outcomes. This isolates the denominator
      mechanism: the exposure is held exactly as published.
    * *internal* recomputes the hospitalisation share and the bands inside the
      restricted data. This is the honest self-contained replication, and it
      also guards against the objection that the exposure itself was measured
      on cells that were then thrown away.
    """
    full = region_totals(cells).filter(
        (pl.col("cases") >= MIN_CASES) & (pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES))
    tables = [gradient(full, "full data (no completeness restriction)")]
    summary: dict = {}

    bands = case_weighted_quintiles(full, "hosp_share").select(
        ["health_region_code", "quintile"])

    for cut in CUTS:
        for mcc in (0, MIN_CELL_CASES):
            sub = restrict(cells, cut, mcc)
            r = region_totals(sub)

            fixed = (r.join(bands, on="health_region_code", how="inner")
                     .group_by("quintile").agg(
                         pl.len().alias("health_regions"),
                         pl.col("hosp_share").median().alias("band_median_hosp_share"),
                         pl.col("cases").sum(), pl.col("deaths").sum(),
                         pl.col("hospitalised").sum(),
                         pl.col("outcome_known").sum()).sort("quintile"))
            cfr, lo, hi = cp(fixed["deaths"], fixed["outcome_known"])
            hs, _, _ = cp(fixed["hospitalised"], fixed["cases"])
            comp, _, _ = cp(fixed["outcome_known"], fixed["cases"])
            lab_fixed = f"fixed bands, completeness >= {cut:.2f}, min cell cases {mcc}"
            tables.append(fixed.with_columns(
                pl.lit(lab_fixed).alias("variant"),
                pl.Series("cfr_pct", 100 * cfr), pl.Series("cfr_lo", 100 * lo),
                pl.Series("cfr_hi", 100 * hi),
                pl.Series("hosp_share_pct", 100 * hs),
                pl.Series("outcome_completeness_pct", 100 * comp)))

            internal_src = r.filter(
                (pl.col("cases") >= MIN_CASES)
                & (pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES))
            lab_int = f"internal bands, completeness >= {cut:.2f}, min cell cases {mcc}"
            gi = gradient(internal_src, lab_int)
            tables.append(gi)

            def span(t: pl.DataFrame) -> dict:
                q1, q5 = t.row(0, named=True), t.row(t.height - 1, named=True)
                return {
                    "bands_present": [int(v) for v in t["quintile"].to_list()],
                    "q1_quintile": int(q1["quintile"]), "q5_quintile": int(q5["quintile"]),
                    "q1_cfr_pct": q1["cfr_pct"], "q1_lo": q1["cfr_lo"], "q1_hi": q1["cfr_hi"],
                    "q1_known_outcomes": int(q1["outcome_known"]),
                    "q5_cfr_pct": q5["cfr_pct"], "q5_lo": q5["cfr_lo"], "q5_hi": q5["cfr_hi"],
                    "q5_known_outcomes": int(q5["outcome_known"]),
                    "fold": q5["cfr_pct"] / q1["cfr_pct"] if q1["cfr_pct"] else None,
                    "intervals_disjoint": bool(q5["cfr_lo"] > q1["cfr_hi"]),
                    "health_regions": [int(v) for v in t["health_regions"].to_list()],
                }
            summary[lab_fixed] = span(tables[-2])
            summary[lab_int] = span(gi)

    summary["full data (no completeness restriction)"] = {
        "q1_cfr_pct": tables[0]["cfr_pct"][0], "q5_cfr_pct": tables[0]["cfr_pct"][-1],
        "fold": tables[0]["cfr_pct"][-1] / tables[0]["cfr_pct"][0],
        "intervals_disjoint": bool(tables[0]["cfr_lo"][-1] > tables[0]["cfr_hi"][0]),
    }
    return pl.concat(tables, how="diagonal"), summary


# ---------------------------------------------------------------------------
# 4. The reverse direction
# ---------------------------------------------------------------------------

def completeness_associations(cells: pl.DataFrame, reg: pl.DataFrame) -> dict:
    """Is completeness itself correlated with hospitalisation share or lethality?

    Verified twice: once exactly as ``40_ascertainment_depth.py`` computes it
    off the atlas (so the reported -0.082 can be checked), and once on the
    coherent line-level rebuild.
    """
    out: dict = {}

    atlas = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    a = atlas.with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hs"),
        (pl.col("deaths") / pl.col("rq4_outcome_known")).alias("cfr"),
        (pl.col("rq4_outcome_known") / pl.col("cases")).alias("comp"),
    ).filter((pl.col("cases") >= MIN_CASES)
             & (pl.col("rq4_outcome_known") >= MIN_KNOWN_OUTCOMES))
    out["atlas_replication"] = {
        "n_health_regions": int(a.height),
        "completeness_vs_hosp_share": spearman(a["comp"].to_numpy(), a["hs"].to_numpy()),
        "completeness_vs_cfr": spearman(a["comp"].to_numpy(), a["cfr"].to_numpy()),
        "hosp_share_vs_cfr": spearman(a["hs"].to_numpy(), a["cfr"].to_numpy()),
    }

    r = reg.filter((pl.col("cases") >= MIN_CASES)
                   & (pl.col("outcome_known") >= MIN_KNOWN_OUTCOMES))
    out["line_level_rebuild"] = {
        "n_health_regions": int(r.height),
        "completeness_vs_hosp_share": spearman(r["completeness"].to_numpy(),
                                               r["hosp_share"].to_numpy()),
        "completeness_vs_cfr": spearman(r["completeness"].to_numpy(), r["cfr"].to_numpy()),
        "hosp_share_vs_cfr": spearman(r["hosp_share"].to_numpy(), r["cfr"].to_numpy()),
    }

    c = cells.filter(pl.col("cases") >= 20)
    out["cell_level"] = {
        "n_cells": int(c.height),
        "restriction": "health-region-years with >= 20 confirmed cases",
        "completeness_vs_hosp_share": spearman(c["completeness"].to_numpy(),
                                               c["hosp_share"].to_numpy()),
        "completeness_vs_year": spearman(c["completeness"].to_numpy(),
                                         c["year"].to_numpy().astype(float)),
    }

    # Direction of the bias, stated as a contrast rather than a correlation:
    # what is case fatality among the cases that would be dropped versus kept?
    for cut in CUTS:
        hi_ = restrict(cells, cut, 0)
        lo_ = cells.filter((pl.col("cases") > 0) & (pl.col("completeness") < cut - 1e-12))
        e1, l1, u1 = cp([int(hi_["deaths"].sum())], [int(hi_["outcome_known"].sum())])
        e0, l0, u0 = cp([int(lo_["deaths"].sum())], [int(lo_["outcome_known"].sum())])
        out[f"pooled_cfr_above_vs_below_{cut:.2f}"] = {
            "above_cfr_pct": [100 * float(e1[0]), 100 * float(l1[0]), 100 * float(u1[0])],
            "above_known_outcomes": int(hi_["outcome_known"].sum()),
            "below_cfr_pct": [100 * float(e0[0]), 100 * float(l0[0]), 100 * float(u0[0])],
            "below_known_outcomes": int(lo_["outcome_known"].sum()),
        }
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, meta = build_cells()
    cells.write_parquet(OUT / "health_region_year_cells.parquet")

    rec = reconcile(cells)
    by_year = completeness_by_year(cells)
    by_year.write_csv(OUT / "completeness_by_year.csv")
    reg = completeness_by_region(cells)
    reg.write_csv(OUT / "completeness_by_health_region.csv")

    spatial = spatial_structure(cells, reg)
    surv_tbl, surv = geography_survival(cells)
    surv_tbl.write_csv(OUT / "cfr_rank_survival_by_cut.csv")
    grad_tbl, grad = gradient_survival(cells)
    grad_tbl.write_csv(OUT / "gradient_by_quintile_restricted.csv")
    assoc = completeness_associations(cells, reg)

    report = {
        "line_level_build": meta,
        "reconciliation_against_atlas": rec,
        "national_outcome_completeness_among_confirmed_cases": {
            "cases": int(cells["cases"].sum()),
            "outcome_known": int(cells["outcome_known"].sum()),
            "pct": 100 * float(cells["outcome_known"].sum()) / float(cells["cases"].sum()),
            "first_year_pct": float(by_year["outcome_completeness_pct"][0]),
            "last_year_pct": float(by_year["outcome_completeness_pct"][-1]),
            "min_pct": float(by_year["outcome_completeness_pct"].min()),
            "max_pct": float(by_year["outcome_completeness_pct"].max()),
            "swing_pp": float(by_year["outcome_completeness_pct"].max()
                              - by_year["outcome_completeness_pct"].min()),
            "note": "T7c's 45.3% -> 70.5% is over all 328,984 notifications; "
                    "the case-fatality denominator is confirmed cases only",
        },
        "spatial_and_temporal_structure": spatial,
        "rank_survival": surv,
        "gradient_survival": grad,
        "completeness_associations": assoc,
    }
    (OUT / "completeness_stratification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ---------------- console ----------------
    n = report["national_outcome_completeness_among_confirmed_cases"]
    print("=== outcome completeness on the population the estimand lives on ===")
    print(f"  confirmed cases {n['cases']:,}  outcome recorded {n['outcome_known']:,} "
          f"= {n['pct']:.1f}%")
    print(f"  by year: {n['first_year_pct']:.1f}% (2007) -> {n['last_year_pct']:.1f}% (2025), "
          f"range {n['min_pct']:.1f}-{n['max_pct']:.1f}%, swing {n['swing_pp']:.1f} pp")
    print("  (T7c's 52.8 pp swing is over ALL notifications, 72% of which were discarded)")

    print("\n=== reconciliation, line-level rebuild vs atlas ===")
    for k, v in rec.items():
        print(f"  {k:<20} atlas {v['atlas_total']:>9,.0f}  rebuild {v['rebuild_total']:>9,.0f}"
              f"  ({v['pct_difference']:+.2f}%)  rank rho "
              f"{v['spearman_across_health_regions']['rho']:.4f}")

    m = spatial["morans_i_completeness"]
    v = spatial["variance_decomposition_empirical_logit_completeness"]
    d = spatial["completeness_distribution_across_health_regions"]
    print("\n=== is completeness itself structured? ===")
    print(f"  health regions with >= {MIN_CASES} cases: n={d['n_health_regions']}, "
          f"completeness median {d['median']:.3f} "
          f"(p10 {d['p10']:.3f}, p90 {d['p90']:.3f}, min {d['min']:.3f})")
    print(f"  {d['n_below_0.90']} of {d['n_health_regions']} regions below 90%, "
          f"{d['n_below_0.80']} below 80%")
    print(f"  Moran's I (completeness) {m['morans_i']:+.4f}  perm p={m['permutation_p_one_sided_positive']:.4f}")
    print(f"  Moran's I (case fatality) {spatial['morans_i_case_fatality']['morans_i']:+.4f}"
          f"  Moran's I (hosp share) {spatial['morans_i_hosp_share']['morans_i']:+.4f}")
    print(f"  cell variance of logit completeness: health region explains "
          f"{100*v['share_explained_by_health_region']:.1f}%, year explains "
          f"{100*v['share_explained_by_year']:.1f}%  (n={v['cells']} cells, >=20 cases)")
    v2 = spatial["variance_decomposition_min10_cells"]
    print(f"    same at >=10 cases: region {100*v2['share_explained_by_health_region']:.1f}%, "
          f"year {100*v2['share_explained_by_year']:.1f}%  (n={v2['cells']} cells)")

    print("\n=== does the health-region case-fatality ORDERING survive restriction? ===")
    print(f"  {'cut':>5} {'mincell':>8} {'cells':>7} {'cases kept':>11} {'regions':>8} "
          f"{'paired':>7} {'rho':>7} {'pooled CFR':>12}")
    for r in surv_tbl.iter_rows(named=True):
        print(f"  {r['cut']:>5.2f} {r['min_cell_cases']:>8} {r['cells_retained']:>7} "
              f"{r['case_retention_pct']:>10.1f}% {r['health_regions_retained']:>8} "
              f"{r['health_regions_paired']:>7} {r['spearman_rho_full_vs_restricted']:>7.3f} "
              f"{r['restricted_pooled_cfr_pct']:>7.2f}% ")
    print("  non-overlapping split (well-recorded cells vs the rest, no shared deaths):")
    for d_ in surv["disjoint_split_no_shared_deaths"]:
        print(f"    cut {d_['cut']:.2f}: n={d_['health_regions_with_both_halves']} regions "
              f"with both halves, rho={d_['spearman_rho']:+.3f} (p={d_['spearman_p']:.2g}), "
              f"known outcomes {d_['known_outcomes_high_half']:,} vs "
              f"{d_['known_outcomes_low_half']:,}")

    print("\n=== the key check: case fatality by hospitalisation-share quintile ===")
    for variant in grad_tbl["variant"].unique(maintain_order=True):
        t = grad_tbl.filter(pl.col("variant") == variant)
        cells_s = "  ".join(f"Q{r['quintile']} {r['cfr_pct']:5.2f}%"
                            for r in t.iter_rows(named=True))
        fold = t["cfr_pct"][-1] / t["cfr_pct"][0]
        dis = t["cfr_lo"][-1] > t["cfr_hi"][0]
        print(f"  {variant:<58} {cells_s}   {fold:4.1f}x "
              f"{'disjoint' if dis else 'OVERLAPPING'}")

    a = assoc["atlas_replication"]
    b = assoc["line_level_rebuild"]
    print("\n=== reverse direction: is completeness itself the exposure? ===")
    print(f"  atlas  (n={a['n_health_regions']}): completeness~hosp share "
          f"{a['completeness_vs_hosp_share']['rho']:+.3f} "
          f"(p={a['completeness_vs_hosp_share']['p']:.3f}); "
          f"completeness~CFR {a['completeness_vs_cfr']['rho']:+.3f} "
          f"(p={a['completeness_vs_cfr']['p']:.3f})")
    print(f"  rebuild(n={b['n_health_regions']}): completeness~hosp share "
          f"{b['completeness_vs_hosp_share']['rho']:+.3f} "
          f"(p={b['completeness_vs_hosp_share']['p']:.3f}); "
          f"completeness~CFR {b['completeness_vs_cfr']['rho']:+.3f} "
          f"(p={b['completeness_vs_cfr']['p']:.3f})")
    for cut in CUTS:
        k = assoc[f"pooled_cfr_above_vs_below_{cut:.2f}"]
        print(f"  cells >= {cut:.2f}: CFR {k['above_cfr_pct'][0]:.2f}% "
              f"(n={k['above_known_outcomes']:,})   below: {k['below_cfr_pct'][0]:.2f}% "
              f"(n={k['below_known_outcomes']:,})")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
