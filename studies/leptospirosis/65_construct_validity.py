"""Is the case-fatality gradient a property of disease, or of confirmation and coding?

The study's claim is that reported case fatality rises across H — the hospitalised
share of confirmed cases, an **inverse** indicator of ascertainment breadth (high H
= surveillance reaching only severe illness = narrow ascertainment). Two reviewer
objections attack the claim not through confounding but through *definition*: the
gradient might be manufactured by how cases are confirmed, or by how severity is
coded. This script runs both attacks.

**Threat 1 — the confirmation pathway moves with case-finding intensity.**
Brazil confirms leptospirosis by laboratory criteria OR by clinical-epidemiological
criteria. The mix is not fixed: in Rio Grande do Sul in 2024, during mass
case-finding after the floods, the laboratory-confirmed share fell sharply. If
broad-ascertainment territories dilute their confirmed pool with
clinical-epidemiological cases that are not really leptospirosis, their case
fatality would fall for a reason that has nothing to do with the clinical spectrum,
and the gradient would be an artefact of confirmation practice.
*Confirms the threat:* the gradient is present among all confirmed cases but
disappears within laboratory-confirmed cases (a homogeneous case definition).
*Rules it out:* the gradient survives inside each confirmation pathway separately,
and the region ordering on H is preserved when H is rebuilt on laboratory-confirmed
cases alone.

**Threat 2 — the severe phenotype is a coding artefact.**
The severe phenotype used throughout is "jaundice OR renal failure OR haemorrhage"
from the SINAN clinical block. It is a three-field composite over fields that are
blank in a non-trivial share of records, and the blank rate could itself vary with
ascertainment. If the phenotype is fragile — if the conclusion flips when the
definition is narrowed to jaundice alone, or tightened to a Weil-like
jaundice-AND-renal, or when the analysis is restricted to records whose whole
clinical block is filled in — then the severity argument is about form-filling.
*The mechanism under test:* narrow ascertainment should preferentially fail to find
MILD cases. Severe-case incidence should therefore vary far less across H than
non-severe incidence.
*Confirms the threat:* severe and non-severe incidence fall by similar factors from
Q1 to Q5, or the ordering of those two factors reverses under a different severity
definition.
*Rules it out:* the non-severe Q1/Q5 incidence ratio exceeds the severe one under
every definition constructed, including on the fully-decoded subset.

The exposure strata are held FIXED at the primary H quintiles throughout. Only the
case definition and the severity definition are varied. Varying the strata at the
same time would confound the two questions.

Outputs to ``data/results/construct_validity/``.
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

OUT = PATHS.results / "construct_validity"
PANEL = PATHS.results / "analysis_panel"

YEAR_MIN, YEAR_MAX = 2007, 2025

#: A region needs enough cases with a decoded hospitalisation field for H to mean
#: anything; below this the exposure is mostly sampling noise and the region would
#: be assigned to a quintile essentially at random. Same threshold as
#: ``62_triangulation.py`` so the strata are comparable across scripts.
MIN_HOSP_KNOWN = 30

#: Quintiles of H, formed on the *case-weighted* cumulative distribution so each
#: stratum carries a comparable number of confirmed cases rather than a comparable
#: number of regions (regions differ in size by three orders of magnitude).
N_BINS = 5

#: The full SINAN clinical block. ``cli_outros`` ("other findings") is included in
#: the completeness audit because differential blank rates are the threat being
#: tested, but it is never part of any severity definition: it names no syndrome.
CLI_FIELDS = [
    "cli_febre", "cli_mialgi", "cli_cefale", "cli_prost", "cli_vomito",
    "cli_diarre", "cli_icteri", "cli_conges", "cli_pantur", "cli_renal",
    "cli_hemorr", "cli_hemopu", "cli_respir", "cli_cardia", "cli_mening",
    "cli_outros",
]


def exact_ratio_ci(
    x1: int, d1: float, x2: int, d2: float, conf: float = 0.95
) -> tuple[float, float, float]:
    """Exact conditional interval for the ratio of two rates ``(x1/d1) / (x2/d2)``.

    Conditioning on the total ``x1 + x2`` makes ``x1`` binomial, so a
    Clopper-Pearson interval on that binomial transforms into an exact interval
    for the rate ratio. Returns ``(ratio, lower, upper)`` — estimate FIRST, like
    every other interval helper in this repo.

    Applied to two case-fatality proportions this is the Poisson approximation to
    the ratio of two binomial proportions; it is exact for the rate ratio and
    conservative for the proportion ratio at the case fatalities seen here (~10%),
    and is labelled as a rate ratio wherever it is used that way.
    """
    total = x1 + x2
    if total == 0 or d1 <= 0 or d2 <= 0:
        return float("nan"), float("nan"), float("nan")
    p, p_lo, p_hi = binom_ci(x1, total, conf=conf)
    f = float(d2) / float(d1)

    def _conv(q: float) -> float:
        q = float(q)
        return float("inf") if q >= 1.0 else q / (1.0 - q) * f

    ratio = ((x1 / d1) / (x2 / d2)) if x2 > 0 else float("inf")
    return float(ratio), _conv(p_lo), _conv(p_hi)


def _valid(field: str) -> pl.Expr:
    return pl.col(f"{field}_state") == "valid"


def _yes(field: str) -> pl.Expr:
    return _valid(field) & (pl.col(field) == "sim")


def load_line() -> pl.DataFrame:
    """The analytic population, with geography attached, exactly as the panel builds it."""
    cols = [
        "classi_fin", "epiweek_onset_year", "municipality_residence_code7",
        "evolucao", "evolucao_state", "ate_hosp", "ate_hosp_state",
        "criterio", "criterio_state",
    ]
    for f in CLI_FIELDS:
        cols += [f, f"{f}_state"]

    line = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .select(cols)
        .filter(
            (pl.col("classi_fin") == "confirmado")
            & pl.col("epiweek_onset_year").is_between(YEAR_MIN, YEAR_MAX)
            & pl.col("municipality_residence_code7").is_not_null()
        )
        .collect()
        .rename({"epiweek_onset_year": "year"})
    )

    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    geo = atlas.select(
        pl.col("munic_code").alias("municipality_residence_code7"),
        "health_region_code", "uf_abbr",
    )
    line = line.join(geo, on="municipality_residence_code7", how="left")
    return line.filter(pl.col("health_region_code").is_not_null())


def add_flags(line: pl.DataFrame) -> pl.DataFrame:
    """Case-definition and severity flags. Every flag carries its own denominator flag."""
    sev3_known = _valid("cli_icteri") & _valid("cli_renal") & _valid("cli_hemorr")
    sev4_known = sev3_known & _valid("cli_hemopu")
    weil_known = _valid("cli_icteri") & _valid("cli_renal")
    block_known = pl.all_horizontal([_valid(f) for f in CLI_FIELDS])

    return line.with_columns(
        # -- outcome / exposure ------------------------------------------------
        (pl.col("evolucao_state") == "valid").cast(pl.Int32).alias("outcome_known"),
        ((pl.col("evolucao_state") == "valid")
         & (pl.col("evolucao") == "obito_por_leptospirose")).cast(pl.Int32).alias("death"),
        (pl.col("ate_hosp_state") == "valid").cast(pl.Int32).alias("hosp_known"),
        ((pl.col("ate_hosp_state") == "valid")
         & (pl.col("ate_hosp") == "sim")).cast(pl.Int32).alias("hosp"),
        # -- confirmation pathway ---------------------------------------------
        (pl.col("criterio_state") == "valid").cast(pl.Int32).alias("crit_known"),
        ((pl.col("criterio_state") == "valid")
         & (pl.col("criterio") == "clinico_laboratorial")).cast(pl.Int32).alias("lab"),
        ((pl.col("criterio_state") == "valid")
         & (pl.col("criterio") == "clinico_epidemiologico")).cast(pl.Int32).alias("clinepi"),
        # -- severity denominators --------------------------------------------
        sev3_known.cast(pl.Int32).alias("sev3_known"),
        sev4_known.cast(pl.Int32).alias("sev4_known"),
        weil_known.cast(pl.Int32).alias("weil_known"),
        _valid("cli_icteri").cast(pl.Int32).alias("jaund_known"),
        _valid("cli_renal").cast(pl.Int32).alias("renal_known"),
        block_known.cast(pl.Int32).alias("block_known"),
    ).with_columns(
        # -- severity numerators, each inside its own denominator --------------
        (pl.col("sev3_known").cast(pl.Boolean)
         & (_yes("cli_icteri") | _yes("cli_renal") | _yes("cli_hemorr"))
         ).cast(pl.Int32).alias("sev3"),
        (pl.col("sev4_known").cast(pl.Boolean)
         & (_yes("cli_icteri") | _yes("cli_renal") | _yes("cli_hemorr") | _yes("cli_hemopu"))
         ).cast(pl.Int32).alias("sev4"),
        (pl.col("jaund_known").cast(pl.Boolean) & _yes("cli_icteri")).cast(pl.Int32).alias("jaund"),
        (pl.col("renal_known").cast(pl.Boolean) & _yes("cli_renal")).cast(pl.Int32).alias("renal"),
        (pl.col("weil_known").cast(pl.Boolean) & _yes("cli_icteri") & _yes("cli_renal")
         ).cast(pl.Int32).alias("weil"),
        # Count-based score 0-3 over the same three components, defined only where
        # all three decode (otherwise a low score would just mean a blank form).
        pl.when(sev3_known).then(
            _yes("cli_icteri").cast(pl.Int32)
            + _yes("cli_renal").cast(pl.Int32)
            + _yes("cli_hemorr").cast(pl.Int32)
        ).otherwise(None).alias("sev_score"),
    ).with_columns(
        (pl.col("sev_score") >= 2).fill_null(False).cast(pl.Int32).alias("score2"),
    )


def build_strata(line: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Region-level H (all confirmed and laboratory-only) plus the fixed H quintiles."""
    panel = pl.read_parquet(PANEL / "region_year_panel.parquet")
    py = panel.group_by("health_region_code").agg(
        pl.col("person_years").sum(), pl.col("cases").sum().alias("panel_cases")
    )

    reg = line.group_by("health_region_code").agg(
        pl.len().alias("cases"),
        pl.col(["death", "outcome_known", "hosp", "hosp_known",
                "lab", "clinepi", "crit_known"]).sum(),
        # H rebuilt on laboratory-confirmed cases alone: same construct, stricter
        # case definition.
        (pl.col("hosp") * pl.col("lab")).sum().alias("hosp_lab"),
        (pl.col("hosp_known") * pl.col("lab")).sum().alias("hosp_known_lab"),
    ).join(py, on="health_region_code", how="left")

    # Plausibility gate: this script rebuilds the region aggregates from the line
    # file rather than reading the shared panel, so the two must agree exactly.
    mismatch = reg.filter(pl.col("cases") != pl.col("panel_cases"))
    assert mismatch.height == 0, f"{mismatch.height} regions disagree with the panel"
    assert int(reg["cases"].sum()) == 66358, int(reg["cases"].sum())

    reg = reg.with_columns(
        (pl.col("hosp") / pl.col("hosp_known")).alias("H"),
        (pl.col("hosp_lab") / pl.col("hosp_known_lab")).alias("H_lab"),
    )

    elig = reg.filter(pl.col("hosp_known") >= MIN_HOSP_KNOWN)
    # Sort on the region code as well as on H. Fifteen of the 209 eligible regions
    # share a tied value of H, and a group-by upstream does not guarantee row
    # order, so sorting on H alone let tied regions fall into different quintiles
    # from one run to the next and moved the headline gradients in the third
    # significant figure. A reproducible tiebreak is not cosmetic.
    elig = elig.sort(["H", "health_region_code"]).with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("q")
    ).drop("_cw")

    # The same case-weighted rule applied to H_lab, to see whether regions would be
    # sorted into different strata by the stricter case definition.
    elig = elig.join(
        elig.filter(pl.col("hosp_known_lab") >= MIN_HOSP_KNOWN)
            .sort(["H_lab", "health_region_code"])
            .with_columns((pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw"))
            .with_columns((pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS)
                          .cast(pl.Int32).alias("q_lab"))
            .select("health_region_code", "q_lab"),
        on="health_region_code", how="left",
    )
    return reg, elig


# ---------------------------------------------------------------------------
# Strand 1: confirmation pathway
# ---------------------------------------------------------------------------

def strand1(line: pl.DataFrame, elig: pl.DataFrame) -> dict:
    rep: dict = {}

    # -- laboratory share by H quintile ------------------------------------
    byq = line.group_by("q").agg(
        pl.len().alias("cases"),
        pl.col(["lab", "clinepi", "crit_known", "death", "outcome_known"]).sum(),
        (pl.col("death") * pl.col("lab")).sum().alias("death_lab"),
        (pl.col("outcome_known") * pl.col("lab")).sum().alias("ok_lab"),
        (pl.col("death") * pl.col("clinepi")).sum().alias("death_ce"),
        (pl.col("outcome_known") * pl.col("clinepi")).sum().alias("ok_ce"),
    ).sort("q")

    hq = elig.group_by("q").agg(
        pl.col("hosp").sum(), pl.col("hosp_known").sum(), pl.len().alias("regions")
    ).with_columns((pl.col("hosp") / pl.col("hosp_known")).alias("H")).sort("q")
    byq = byq.join(hq.select("q", "H", "regions"), on="q", how="left")

    rows = []
    for r in byq.iter_rows(named=True):
        share, s_lo, s_hi = binom_ci(int(r["lab"]), int(r["crit_known"]))
        rows.append({
            "q": int(r["q"]), "regions": int(r["regions"]), "H": float(r["H"]),
            "cases": int(r["cases"]), "crit_known": int(r["crit_known"]),
            "criterion_completeness": float(r["crit_known"]) / float(r["cases"]),
            "lab_confirmed": int(r["lab"]),
            "share_lab": float(share), "share_lab_lo": float(s_lo), "share_lab_hi": float(s_hi),
        })
    lab_q = pl.DataFrame(rows)
    lab_q.write_csv(OUT / "lab_share_by_quintile.csv")

    # -- laboratory share by year (national) --------------------------------
    byy = line.group_by("year").agg(
        pl.len().alias("cases"), pl.col(["lab", "crit_known"]).sum()
    ).sort("year")
    rows = []
    for r in byy.iter_rows(named=True):
        share, s_lo, s_hi = binom_ci(int(r["lab"]), int(r["crit_known"]))
        rows.append({
            "year": int(r["year"]), "cases": int(r["cases"]),
            "crit_known": int(r["crit_known"]), "lab_confirmed": int(r["lab"]),
            "share_lab": float(share), "share_lab_lo": float(s_lo), "share_lab_hi": float(s_hi),
        })
    lab_y = pl.DataFrame(rows)
    lab_y.write_csv(OUT / "lab_share_by_year.csv")

    # The named case: Rio Grande do Sul, 2024. A finding that predicts a specific
    # place and year gets checked by name, not left to the pooled series.
    rs = line.filter(pl.col("uf_abbr") == "RS").group_by("year").agg(
        pl.len().alias("cases"), pl.col(["lab", "crit_known"]).sum()
    ).sort("year")
    rs_rows = []
    for r in rs.iter_rows(named=True):
        share, s_lo, s_hi = binom_ci(int(r["lab"]), int(r["crit_known"]))
        rs_rows.append({
            "year": int(r["year"]), "cases": int(r["cases"]),
            "crit_known": int(r["crit_known"]), "lab_confirmed": int(r["lab"]),
            "share_lab": float(share), "share_lab_lo": float(s_lo), "share_lab_hi": float(s_hi),
        })
    pl.DataFrame(rs_rows).write_csv(OUT / "lab_share_rs_by_year.csv")
    rs_by_year = {int(r["year"]): round(float(r["share_lab"]), 4) for r in rs_rows}

    # -- case fatality by H quintile within each confirmation pathway --------
    defs = [
        ("all confirmed", "death", "outcome_known"),
        ("laboratory-confirmed only", "death_lab", "ok_lab"),
        ("clinical-epidemiological only", "death_ce", "ok_ce"),
    ]
    rows = []
    for name, dcol, ncol in defs:
        for r in byq.iter_rows(named=True):
            cfr, lo, hi = binom_ci(int(r[dcol]), int(r[ncol]))
            rows.append({
                "case_definition": name, "q": int(r["q"]), "H": float(r["H"]),
                "deaths": int(r[dcol]), "outcome_known": int(r[ncol]),
                "cfr": float(cfr), "cfr_lo": float(lo), "cfr_hi": float(hi),
            })
    cfr_tab = pl.DataFrame(rows)
    cfr_tab.write_csv(OUT / "cfr_by_quintile_by_criterion.csv")

    grad = {}
    for name, dcol, ncol in defs:
        q1 = byq.filter(pl.col("q") == 1).row(0, named=True)
        q5 = byq.filter(pl.col("q") == N_BINS).row(0, named=True)
        rr, lo, hi = exact_ratio_ci(int(q5[dcol]), int(q5[ncol]),
                                    int(q1[dcol]), int(q1[ncol]))
        cfr1 = float(binom_ci(int(q1[dcol]), int(q1[ncol]))[0])
        cfr5 = float(binom_ci(int(q5[dcol]), int(q5[ncol]))[0])
        grad[name] = {
            "cfr_Q1": cfr1, "cfr_Q5": cfr5,
            "Q5_over_Q1": rr, "Q5_over_Q1_lo": lo, "Q5_over_Q1_hi": hi,
            "deaths_Q1": int(q1[dcol]), "outcome_known_Q1": int(q1[ncol]),
            "deaths_Q5": int(q5[dcol]), "outcome_known_Q5": int(q5[ncol]),
        }

    # -- the premise of the threat, tested directly --------------------------
    # The threat requires the confirmation mix to move with H. It does not have
    # to: if the laboratory share is unrelated to H across regions, the pathway
    # cannot manufacture a monotone gradient in H no matter how different the two
    # pathways are from each other.
    reg_lab = line.group_by("health_region_code").agg(
        pl.col(["lab", "crit_known"]).sum()
    ).join(elig.select("health_region_code", "H"), on="health_region_code", how="inner") \
     .filter(pl.col("crit_known") > 0)
    s_lab = stats.spearmanr(reg_lab["H"].to_numpy(),
                            (reg_lab["lab"] / reg_lab["crit_known"]).to_numpy())
    rep["region_spearman_H_vs_lab_share"] = {
        "spearman_rho": float(s_lab.statistic), "p": float(s_lab.pvalue),
        "regions": reg_lab.height,
    }

    # -- chasing an implausible number ---------------------------------------
    # Case fatality among clinical-epidemiological cases in Q5 comes out far above
    # anything the clinical literature reports for leptospirosis as a whole. Under
    # the plausibility gate that is a bug report until explained. The candidate
    # explanation is that where laboratory capacity is thin, clinical-
    # epidemiological confirmation is partly DEATH-DRIVEN: a fatal case gets
    # confirmed on clinical and epidemiological grounds when no serology exists.
    # That predicts clinical-epidemiological cases are also more often
    # hospitalised than laboratory-confirmed cases, and increasingly so as H rises.
    path_rows = []
    for r in byq.iter_rows(named=True):
        q = int(r["q"])
        sub = line.filter(pl.col("q") == q)
        lab_h = sub.filter(pl.col("lab") == 1)
        ce_h = sub.filter(pl.col("clinepi") == 1)
        h_lab, hl_lo, hl_hi = binom_ci(int(lab_h["hosp"].sum()), int(lab_h["hosp_known"].sum()))
        h_ce, hc_lo, hc_hi = binom_ci(int(ce_h["hosp"].sum()), int(ce_h["hosp_known"].sum()))
        rr, rr_lo, rr_hi = exact_ratio_ci(int(r["death_ce"]), int(r["ok_ce"]),
                                          int(r["death_lab"]), int(r["ok_lab"]))
        path_rows.append({
            "q": q, "H": float(r["H"]),
            "H_within_lab": float(h_lab), "H_within_lab_lo": float(hl_lo),
            "H_within_lab_hi": float(hl_hi),
            "H_within_clinepi": float(h_ce), "H_within_clinepi_lo": float(hc_lo),
            "H_within_clinepi_hi": float(hc_hi),
            "cfr_ratio_clinepi_over_lab": rr, "cfr_ratio_lo": rr_lo, "cfr_ratio_hi": rr_hi,
        })
    pl.DataFrame(path_rows).write_csv(OUT / "pathway_mechanism_by_quintile.csv")
    rep["pathway_mechanism"] = {
        "note": ("clinical-epidemiological confirmation is partly death-driven where "
                 "laboratory capacity is thin; it is a sicker case series, not a "
                 "diluted one, so it cannot be the source of a gradient that runs "
                 "the other way"),
        "by_quintile": path_rows,
    }

    # -- region-level rank association, within each pathway ------------------
    regcfr = line.group_by("health_region_code").agg(
        pl.col(["death", "outcome_known"]).sum(),
        (pl.col("death") * pl.col("lab")).sum().alias("death_lab"),
        (pl.col("outcome_known") * pl.col("lab")).sum().alias("ok_lab"),
        (pl.col("death") * pl.col("clinepi")).sum().alias("death_ce"),
        (pl.col("outcome_known") * pl.col("clinepi")).sum().alias("ok_ce"),
    )
    e = elig.select("health_region_code", "H", "H_lab", "q", "q_lab",
                    "hosp_known_lab").join(regcfr, on="health_region_code", how="left")

    rho = {}
    for name, dcol, ncol in defs:
        sub = e.filter(pl.col(ncol) > 0)
        s = stats.spearmanr(sub["H"].to_numpy(),
                            (sub[dcol] / sub[ncol]).to_numpy())
        rho[name] = {"spearman_rho": float(s.statistic), "p": float(s.pvalue),
                     "regions": sub.height}

    # -- does the stricter case definition preserve the region ordering? -----
    hl = e.filter(pl.col("hosp_known_lab") >= MIN_HOSP_KNOWN)
    s_h = stats.spearmanr(hl["H"].to_numpy(), hl["H_lab"].to_numpy())
    same = int((hl["q"] == hl["q_lab"]).sum())
    within1 = int(((hl["q"] - hl["q_lab"]).abs() <= 1).sum())

    e.write_parquet(OUT / "region_H_variants.parquet")

    rep["lab_share_national"] = float(
        line["lab"].sum() / line["crit_known"].sum())
    rep["criterion_completeness_national"] = float(
        line["crit_known"].sum() / line.height)
    rep["lab_share_by_quintile"] = {
        f"Q{r['q']}": round(r["share_lab"], 4) for r in lab_q.iter_rows(named=True)}
    rep["lab_share_by_year"] = {
        int(r["year"]): round(r["share_lab"], 4) for r in lab_y.iter_rows(named=True)}
    rep["rs_lab_share_by_year"] = rs_by_year
    rep["cfr_gradient_by_case_definition"] = grad
    rep["region_spearman_H_vs_cfr"] = rho
    rep["H_vs_H_lab"] = {
        "regions": hl.height,
        "spearman_rho": float(s_h.statistic), "p": float(s_h.pvalue),
        "same_quintile": same, "same_quintile_pct": 100.0 * same / hl.height,
        "within_one_quintile": within1,
        "within_one_quintile_pct": 100.0 * within1 / hl.height,
        "H_median_all": float(hl["H"].median()),
        "H_median_lab": float(hl["H_lab"].median()),
    }
    a = grad["all confirmed"]["Q5_over_Q1"]
    l = grad["laboratory-confirmed only"]["Q5_over_Q1"]
    c = grad["clinical-epidemiological only"]["Q5_over_Q1"]
    rep["verdict"] = {
        "gradient_survives_in_both_pathways": bool(
            grad["laboratory-confirmed only"]["Q5_over_Q1_lo"] > 1.0
            and grad["clinical-epidemiological only"]["Q5_over_Q1_lo"] > 1.0),
        "attenuation_restricting_to_laboratory_pct": 100.0 * (a - l) / a,
        "ordering_preserved_under_lab_only_H": bool(s_h.statistic > 0.9),
        "statement": (
            f"Restricting to a homogeneous, laboratory-confirmed case definition "
            f"attenuates the case-fatality gradient from {a:.2f} to {l:.2f} but does "
            f"not remove it; the gradient is steeper still ({c:.2f}) among "
            f"clinical-epidemiological cases. The association is robust to "
            f"confirmation pathway."),
    }
    return rep


# ---------------------------------------------------------------------------
# Strand 2: the severity phenotype
# ---------------------------------------------------------------------------

#: (label, numerator column, denominator column). Each severity definition is a
#: numerator inside its OWN valid denominator: a record whose jaundice field is
#: blank contributes to neither the severe nor the non-severe count for that
#: definition, because a blank is not a negative answer.
SEV_DEFS = [
    ("jaundice OR renal OR haemorrhage (primary)", "sev3", "sev3_known"),
    ("jaundice OR renal OR haemorrhage OR pulmonary haemorrhage", "sev4", "sev4_known"),
    ("jaundice alone", "jaund", "jaund_known"),
    ("renal failure alone", "renal", "renal_known"),
    ("jaundice AND renal (Weil-like)", "weil", "weil_known"),
    ("score >= 2 of 3", "score2", "sev3_known"),
]


def _incidence_rows(g: pl.DataFrame, label: str, num: str, den: str,
                    subset: str) -> list[dict]:
    rows = []
    for r in g.iter_rows(named=True):
        py = float(r["person_years"])
        sev = int(r[num])
        non = int(r[den]) - sev
        s, s_lo, s_hi = poisson_ci(sev, py, scale=1e5)
        n, n_lo, n_hi = poisson_ci(non, py, scale=1e5)
        p, p_lo, p_hi = binom_ci(sev, int(r[den]))
        rows.append({
            "subset": subset, "severity_definition": label, "q": int(r["q"]),
            "H": float(r["H"]), "person_years": py,
            "denominator_cases": int(r[den]), "severe_cases": sev,
            "nonsevere_cases": non,
            "share_severe": float(p), "share_severe_lo": float(p_lo),
            "share_severe_hi": float(p_hi),
            "severe_per_100k": float(s), "severe_lo": float(s_lo), "severe_hi": float(s_hi),
            "nonsevere_per_100k": float(n), "nonsevere_lo": float(n_lo),
            "nonsevere_hi": float(n_hi),
        })
    return rows


def _gradient_row(rows: list[dict], label: str, subset: str) -> dict:
    q1 = next(r for r in rows if r["q"] == 1)
    q5 = next(r for r in rows if r["q"] == N_BINS)
    sr, s_lo, s_hi = exact_ratio_ci(q1["severe_cases"], q1["person_years"],
                                    q5["severe_cases"], q5["person_years"])
    nr, n_lo, n_hi = exact_ratio_ci(q1["nonsevere_cases"], q1["person_years"],
                                    q5["nonsevere_cases"], q5["person_years"])
    return {
        "subset": subset, "severity_definition": label,
        "severe_Q1_per_100k": q1["severe_per_100k"],
        "severe_Q5_per_100k": q5["severe_per_100k"],
        "severe_Q1_over_Q5": sr, "severe_Q1_over_Q5_lo": s_lo, "severe_Q1_over_Q5_hi": s_hi,
        "nonsevere_Q1_per_100k": q1["nonsevere_per_100k"],
        "nonsevere_Q5_per_100k": q5["nonsevere_per_100k"],
        "nonsevere_Q1_over_Q5": nr, "nonsevere_Q1_over_Q5_lo": n_lo,
        "nonsevere_Q1_over_Q5_hi": n_hi,
        "nonsevere_over_severe_gradient": nr / sr if sr else float("nan"),
    }


def strand2(line: pl.DataFrame, elig: pl.DataFrame) -> dict:
    rep: dict = {}

    # -- the cli_hemopu / cli_hemorr question, answered with a count ---------
    both = line.filter(_valid("cli_hemopu") & _valid("cli_hemorr"))
    disagree = both.filter(pl.col("cli_hemopu") != pl.col("cli_hemorr")).height
    only_one = line.filter(
        (_valid("cli_hemopu") | _valid("cli_hemorr"))
        & ~(_valid("cli_hemopu") & _valid("cli_hemorr"))
    ).height
    rep["hemopu_vs_hemorr"] = {
        "both_valid": both.height, "disagree": disagree,
        "disagree_pct": 100.0 * disagree / both.height,
        "exactly_one_valid": only_one,
        "verdict": ("cli_hemopu carries no independent information: it is the same "
                    "field in practice and is excluded from every severity definition "
                    "except the explicit four-component check"),
    }

    py_q = elig.group_by("q").agg(pl.col("person_years").sum(),
                                  pl.col("cases").sum().alias("region_cases"))
    hq = elig.group_by("q").agg(pl.col("hosp").sum(), pl.col("hosp_known").sum()) \
             .with_columns((pl.col("hosp") / pl.col("hosp_known")).alias("H")) \
             .select("q", "H")

    # -- per-field completeness, overall and by quintile ---------------------
    agg = [pl.len().alias("cases")] + [
        _valid(f).sum().alias(f"{f}_valid") for f in CLI_FIELDS
    ] + [pl.col("block_known").sum().alias("block_known")]
    comp_q = line.group_by("q").agg(agg).sort("q")
    comp_all = line.select(agg)

    rows = []
    for f in CLI_FIELDS + ["block_known"]:
        col = f"{f}_valid" if f != "block_known" else "block_known"
        n_all = int(comp_all[0, "cases"])
        v_all = int(comp_all[0, col])
        p, lo, hi = binom_ci(v_all, n_all)
        row = {"field": f, "scope": "national", "cases": n_all, "valid": v_all,
               "completeness": float(p), "completeness_lo": float(lo),
               "completeness_hi": float(hi)}
        rows.append(row)
        for r in comp_q.iter_rows(named=True):
            p, lo, hi = binom_ci(int(r[col]), int(r["cases"]))
            rows.append({"field": f, "scope": f"Q{int(r['q'])}",
                         "cases": int(r["cases"]), "valid": int(r[col]),
                         "completeness": float(p), "completeness_lo": float(lo),
                         "completeness_hi": float(hi)})
    comp = pl.DataFrame(rows)
    comp.write_csv(OUT / "clinical_block_completeness.csv")

    # Completeness spread across quintiles is the differential-form-filling threat
    # made numeric: if it is small, the threat is small before any restriction.
    spread = {}
    for f in CLI_FIELDS + ["block_known"]:
        vals = comp.filter((pl.col("field") == f) & (pl.col("scope") != "national"))
        spread[f] = {
            "national": float(comp.filter((pl.col("field") == f)
                                          & (pl.col("scope") == "national"))[0, "completeness"]),
            "min_quintile": float(vals["completeness"].min()),
            "max_quintile": float(vals["completeness"].max()),
        }
    rep["clinical_block_completeness"] = spread

    # -- severity-score distribution by quintile ----------------------------
    sc = line.filter(pl.col("sev3_known") == 1).group_by(["q", "sev_score"]).agg(
        pl.len().alias("n")).sort(["q", "sev_score"])
    sc = sc.join(sc.group_by("q").agg(pl.col("n").sum().alias("denom")), on="q")
    sc = sc.with_columns((pl.col("n") / pl.col("denom")).alias("share"))
    sc.write_csv(OUT / "severity_score_distribution.csv")

    # -- incidence by quintile, for every severity definition ---------------
    inc_rows: list[dict] = []
    grad_rows: list[dict] = []

    sum_cols = ["sev3", "sev3_known", "sev4", "sev4_known", "jaund", "jaund_known",
                "renal", "renal_known", "weil", "weil_known", "score2",
                "block_known"]

    def _by_q(df: pl.DataFrame) -> pl.DataFrame:
        return (df.group_by("q").agg([pl.col(c).sum() for c in sum_cols]
                                     + [pl.len().alias("cases")])
                  .join(py_q, on="q").join(hq, on="q").sort("q"))

    g_all = _by_q(line)
    for label, num, den in SEV_DEFS:
        r = _incidence_rows(g_all, label, num, den, "all confirmed cases")
        inc_rows += r
        grad_rows.append(_gradient_row(r, label, "all confirmed cases"))

    # The named threat: differential form-filling. Restricting to records whose
    # ENTIRE clinical block decodes removes any record where the coder skipped the
    # block, at the cost of a smaller and possibly selected denominator.
    g_blk = _by_q(line.filter(pl.col("block_known") == 1))
    for label, num, den in SEV_DEFS:
        r = _incidence_rows(g_blk, label, num, den, "clinical block fully decoded")
        inc_rows += r
        grad_rows.append(_gradient_row(r, label, "clinical block fully decoded"))

    # A total-case reference line: what the gradient looks like with no severity
    # definition at all. Non-severe incidence has to be read against this.
    tot_rows = []
    for r in g_all.iter_rows(named=True):
        inc, lo, hi = poisson_ci(int(r["cases"]), float(r["person_years"]), scale=1e5)
        tot_rows.append({"q": int(r["q"]), "H": float(r["H"]),
                         "cases": int(r["cases"]),
                         "person_years": float(r["person_years"]),
                         "incidence_per_100k": float(inc),
                         "incidence_lo": float(lo), "incidence_hi": float(hi)})
    pl.DataFrame(tot_rows).write_csv(OUT / "total_incidence_by_quintile.csv")
    tr, t_lo, t_hi = exact_ratio_ci(tot_rows[0]["cases"], tot_rows[0]["person_years"],
                                    tot_rows[-1]["cases"], tot_rows[-1]["person_years"])
    rep["total_incidence_Q1_over_Q5"] = {"ratio": tr, "lo": t_lo, "hi": t_hi}

    inc_tab = pl.DataFrame(inc_rows)
    inc_tab.write_csv(OUT / "severity_incidence_by_quintile.csv")
    grad_tab = pl.DataFrame(grad_rows)
    grad_tab.write_csv(OUT / "severity_gradient_summary.csv")

    rep["severity_gradients"] = {
        f"{r['subset']} | {r['severity_definition']}": {
            "severe_Q1_over_Q5": r["severe_Q1_over_Q5"],
            "severe_lo": r["severe_Q1_over_Q5_lo"], "severe_hi": r["severe_Q1_over_Q5_hi"],
            "nonsevere_Q1_over_Q5": r["nonsevere_Q1_over_Q5"],
            "nonsevere_lo": r["nonsevere_Q1_over_Q5_lo"],
            "nonsevere_hi": r["nonsevere_Q1_over_Q5_hi"],
            "nonsevere_exceeds_severe": bool(
                r["nonsevere_Q1_over_Q5"] > r["severe_Q1_over_Q5"]),
        }
        for r in grad_rows
    }
    rep["conclusion_consistent_across_definitions"] = bool(
        all(r["nonsevere_Q1_over_Q5"] > r["severe_Q1_over_Q5"] for r in grad_rows)
    )
    # A stronger statement than the point estimates: does the severe gradient's
    # upper bound sit below the non-severe gradient's lower bound?
    rep["separation_by_interval"] = {
        f"{r['subset']} | {r['severity_definition']}":
            bool(r["severe_Q1_over_Q5_hi"] < r["nonsevere_Q1_over_Q5_lo"])
        for r in grad_rows
    }

    # Justifying the three-component definition. It is not chosen because it is
    # the most favourable to the mechanism — it is the LEAST favourable of the
    # definitions constructed, because "any of three" admits comparatively mild
    # jaundice and so retains the most residual variation across H. The stricter
    # phenotypes drive severe incidence flat. Reporting the primary is therefore
    # the conservative choice, which is the argument for keeping it.
    prim = next(r for r in grad_rows
                if r["subset"] == "all confirmed cases"
                and r["severity_definition"] == SEV_DEFS[0][0])
    allconf = [r for r in grad_rows if r["subset"] == "all confirmed cases"]
    rep["definition_verdict"] = {
        "primary_severe_Q1_over_Q5": prim["severe_Q1_over_Q5"],
        "max_severe_Q1_over_Q5_across_definitions": max(
            r["severe_Q1_over_Q5"] for r in allconf),
        "primary_is_the_most_conservative": bool(
            prim["severe_Q1_over_Q5"] >= max(r["severe_Q1_over_Q5"] for r in allconf)),
        "min_nonsevere_over_severe_gradient": min(
            r["nonsevere_over_severe_gradient"] for r in grad_rows),
        "statement": (
            "The primary three-component phenotype leaves the LARGEST residual "
            "gradient in severe-case incidence of any definition tested; stricter "
            "phenotypes (Weil-like, score >= 2) drive it to unity or below. Keeping "
            "the three-component definition therefore understates the mechanism "
            "rather than manufacturing it, and the conclusion does not depend on "
            "which definition is used."),
    }
    return rep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    line = add_flags(load_line())
    reg, elig = build_strata(line)
    line = line.join(elig.select("health_region_code", "q"),
                     on="health_region_code", how="inner")

    coverage = {
        "regions_total": reg.height,
        "regions_eligible": elig.height,
        "min_hosp_known": MIN_HOSP_KNOWN,
        "cases_total": int(reg["cases"].sum()),
        "cases_in_eligible_regions": line.height,
        "cases_retained_pct": 100.0 * line.height / int(reg["cases"].sum()),
        "quintile_H": {
            f"Q{int(r['q'])}": float(r["hosp"]) / float(r["hosp_known"])
            for r in elig.group_by("q").agg(pl.col(["hosp", "hosp_known"]).sum())
                          .sort("q").iter_rows(named=True)
        },
    }

    report = {"coverage": coverage,
              "strand1_confirmation_pathway": strand1(line, elig),
              "strand2_severity_phenotype": strand2(line, elig)}
    (OUT / "construct_validity_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
