"""Clinical presentation of Brazilian leptospirosis, and whether it moves with depth.

Open thread T-12. The SINAN clinical block carries 15 signs at 91-98% recording
on confirmed cases and the study has used four of them (jaundice, renal,
haemorrhage, pulmonary haemorrhage) to build a binary severe phenotype. The
other eleven have never been read. Two things follow.

**First, a descriptive core the manuscript does not have.** What does notified
Brazilian leptospirosis actually look like, and where does it depart from the
classic clinical description (fever, myalgia with calf predominance, headache,
conjunctival suffusion, jaundice)?

**Second, a sharper test of the study's central claim.** The depth account says
territories differ in how far down the severity distribution surveillance
reaches, not in the disease they have. That prediction is directional and
falsifiable at the level of individual signs: the *mild* signs that every case
has (fever, myalgia, headache) should be near-flat across bands, while the
*severe* signs should rise sharply as detection gets shallower. If mild signs
move as much as severe ones, the bands hold different disease, not different
depth, and the central claim is in trouble.

**Third, the severity-adjusted version of the central claim.** The binary severe
phenotype throws away the difference between a case with one severe marker and a
case with all four. A count index (0-4) is a graded severity scale available at
no extra cost. If case fatality rises monotonically with it -- and if the
Q5-versus-Q1 case-fatality ratio survives *within* each level of it -- then the
depth gradient is not a case-mix artefact of the severity mix the block can see.

Every proportion is on the **recorded** denominator. A blank sign is not a "no".
Recording completeness itself varies across bands (DATA.md 16.2) and is reported
beside every band comparison, because it bounds all of them.

**The standing trap, restated because it bounds the third analysis.** The
clinical form is completed at notification, before deterioration. Severity
measured here is severity *at presentation*, so any severity adjustment built on
it is a LOWER BOUND on the case-mix contribution, never a point estimate.

Outputs to ``data/results/clinical_profile/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci
from brepi.config import PATHS

OUT = PATHS.results / "clinical_profile"
LINE = PATHS.interim / "lept_line_level.parquet"

#: Mirrors 40_ascertainment_depth.py so the bands are the published bands.
MIN_CASES = 30

#: The 15 signs of the clinical block, with the role each plays in the classic
#: textbook description of leptospirosis. The role strings are qualitative on
#: purpose: this file compares Brazil's notified profile against the *structure*
#: of the classic description, and does not import prevalence figures from
#: clinical series that were measured on hospitalised patients in other
#: countries and would not be a like-for-like denominator.
SIGNS: dict[str, tuple[str, str]] = {
    "cli_febre":  ("fever", "cardinal - near-universal in the classic description"),
    "cli_mialgi": ("myalgia", "cardinal - classic triad with fever and headache"),
    "cli_cefale": ("headache", "cardinal - classic triad"),
    "cli_pantur": ("calf pain", "classic localising sign, gastrocnemius tenderness"),
    "cli_prost":  ("prostration", "common non-specific"),
    "cli_vomito": ("vomiting", "common non-specific"),
    "cli_icteri": ("jaundice", "SEVERE - Weil's syndrome"),
    "cli_diarre": ("diarrhoea", "common non-specific"),
    "cli_respir": ("respiratory involvement", "severe-adjacent, not in current index"),
    "cli_renal":  ("renal impairment", "SEVERE - Weil's syndrome"),
    "cli_conges": ("conjunctival suffusion", "classic hallmark, said to be near-pathognomonic"),
    "cli_hemorr": ("haemorrhage", "SEVERE - haemorrhagic form"),
    "cli_hemopu": ("pulmonary haemorrhage", "SEVERE - pulmonary haemorrhage syndrome"),
    "cli_cardia": ("cardiac involvement", "severe-adjacent, not in current index"),
    "cli_mening": ("meningismus", "classic aseptic-meningitis presentation"),
}

#: The published severe phenotype. Copied, not re-derived (44_depth_vs_severity.py).
SEVERE_FIELDS = ["cli_icteri", "cli_renal", "cli_hemorr", "cli_hemopu"]
#: The signs a depth account predicts are FLAT across bands.
MILD_FIELDS = ["cli_febre", "cli_mialgi", "cli_cefale"]
#: Candidate extension of the severity index, tested rather than assumed.
EXTENDED_FIELDS = SEVERE_FIELDS + ["cli_respir", "cli_cardia"]

#: **Independent** severity axes. `cli_hemorr` and `cli_hemopu` are not two
#: variables: among the 61,219 confirmed cases with both recorded they agree on
#: 61,206, and 5,900 of 5,913 positives are concordant. A count over the four
#: published fields therefore gives haemorrhage double weight, which is what
#: flattens the 4-field count between levels 2 and 3 (checked and reported in
#: `haemorrhage_collinearity`). The binary phenotype is untouched by this --
#: any-of-four and any-of-three differ on 13 cases -- but a graded index must be
#: built on the axes, not the fields.
AXIS_FIELDS: dict[str, list[str]] = {
    "jaundice": ["cli_icteri"],
    "renal": ["cli_renal"],
    "haemorrhage": ["cli_hemorr", "cli_hemopu"],
}
AXIS_EXTRA: dict[str, list[str]] = {
    "respiratory": ["cli_respir"],
    "cardiac": ["cli_cardia"],
}

# Import the banding function from the script that produced the central result,
# so the quintiles here are literally the published quintiles.
_spec = importlib.util.spec_from_file_location(
    "_asc_depth", Path(__file__).with_name("40_ascertainment_depth.py"))
_asc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_asc)
case_weighted_quintiles = _asc._case_weighted_quintiles


# ---------------------------------------------------------------------------
# Interval helpers. Unpack order is (ESTIMATE, lo, hi) and every call asserts it.
# ---------------------------------------------------------------------------
def prop(count, total) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, lo, hi = binom_ci(np.asarray(count, float), np.asarray(total, float))
    ok = np.isfinite(p)
    assert np.all(lo[ok] <= p[ok] + 1e-12) and np.all(p[ok] <= hi[ok] + 1e-12), \
        "binom_ci estimate outside its own interval - unpack order is wrong"
    return p, lo, hi


def katz_rr(a, n1, b, n0) -> tuple[float, float, float]:
    """Risk ratio (a/n1) / (b/n0) with a Katz log interval.

    Returns (ESTIMATE, lo, hi). Undefined cells return NaN rather than a silent
    zero.
    """
    a, n1, b, n0 = float(a), float(n1), float(b), float(n0)
    if min(n1, n0) == 0 or a == 0 or b == 0:
        return (float("nan"), float("nan"), float("nan"))
    rr = (a / n1) / (b / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
    return rr, float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))


def mh_risk_ratio(a, n1, b, n0) -> dict:
    """Mantel-Haenszel risk ratio with a Greenland-Robins interval.

    ``a``/``n1`` are deaths/known-outcomes in the exposed stratum arm (Q5),
    ``b``/``n0`` the unexposed (Q1), one entry per severity stratum. This is the
    adjusted answer to "does the depth gradient survive holding severity fixed".
    """
    a, n1, b, n0 = (np.asarray(x, float) for x in (a, n1, b, n0))
    N = n1 + n0
    keep = N > 0
    a, n1, b, n0, N = a[keep], n1[keep], b[keep], n0[keep], N[keep]
    R = a * n0 / N
    S = b * n1 / N
    if S.sum() == 0 or R.sum() == 0:
        return {"rr_mh": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "strata": int(keep.sum())}
    rr = float(R.sum() / S.sum())
    M1 = a + b
    P = (n1 * n0 * M1 - a * b * N) / N ** 2
    var = float(P.sum() / (R.sum() * S.sum()))
    se = np.sqrt(var)
    return {"rr_mh": rr, "lo": float(rr * np.exp(-1.96 * se)),
            "hi": float(rr * np.exp(1.96 * se)), "strata": int(keep.sum())}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def bands() -> pl.DataFrame:
    """Health-region hospitalisation-share quintiles: the published banding."""
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.filter(pl.col("cases") >= MIN_CASES).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share")
    )
    q = case_weighted_quintiles(hr, "hosp_share")
    return q.select("health_region_code", "quintile", "hosp_share", "region")


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (all confirmed cases, confirmed cases mapped to a banded region).

    The national profile uses every confirmed case; the band comparisons use the
    banded subset. Reporting both keeps the manuscript table on the full 66,667
    while the gradient analysis stays on the published banding.
    """
    keep = (["municipality_residence_code7", "ate_hosp", "evolucao",
             "evolucao_state", "doenca_tra", "con_ambien", "src_year"]
            + list(SIGNS))
    lf = pl.scan_parquet(LINE).filter(pl.col("classi_fin") == "confirmado")
    have = set(lf.collect_schema().names())
    missing = [c for c in keep if c not in have]
    if missing:
        raise AssertionError(f"line level lacks expected columns: {missing}")
    d = lf.select(keep).collect()

    rec = {c: (pl.col(c).is_in(["sim", "nao"])) for c in SIGNS}
    d = d.with_columns(
        [rec[c].alias(f"{c}__rec") for c in SIGNS]
        + [(pl.col(c) == "sim").alias(f"{c}__yes") for c in SIGNS]
        + [
            (pl.col("evolucao") == "obito_por_leptospirose").alias("died"),
            (pl.col("evolucao_state") == "valid").alias("outcome_ok"),
            (pl.col("ate_hosp") == "sim").alias("hosp"),
        ]
    )
    # Severity count: defined only where all four markers are recorded. A blank
    # block is not a count of zero; treating it as one would manufacture the
    # gradient under test.
    d = d.with_columns(
        sev_known=pl.all_horizontal([pl.col(f"{c}__rec") for c in SEVERE_FIELDS]),
        sev_count=pl.sum_horizontal(
            [pl.col(f"{c}__yes").cast(pl.Int32) for c in SEVERE_FIELDS]),
        ext_known=pl.all_horizontal([pl.col(f"{c}__rec") for c in EXTENDED_FIELDS]),
        ext_count=pl.sum_horizontal(
            [pl.col(f"{c}__yes").cast(pl.Int32) for c in EXTENDED_FIELDS]),
        block_complete=pl.all_horizontal([pl.col(f"{c}__rec") for c in SIGNS]),
        n_signs_recorded=pl.sum_horizontal(
            [pl.col(f"{c}__rec").cast(pl.Int32) for c in SIGNS]),
    ).with_columns(
        pl.when(pl.col("sev_known")).then(pl.col("sev_count")).otherwise(None)
          .alias("sev_count"),
        pl.when(pl.col("ext_known")).then(pl.col("ext_count")).otherwise(None)
          .alias("ext_count"),
    )

    # Axis counts: haemorrhage counted once, not twice.
    axis_all = {**AXIS_FIELDS, **AXIS_EXTRA}
    d = d.with_columns([
        pl.any_horizontal([pl.col(f"{f}__yes") for f in fields]).alias(f"ax_{name}")
        for name, fields in axis_all.items()
    ] + [
        pl.any_horizontal([pl.col(f"{f}__rec") for f in fields]).alias(f"axrec_{name}")
        for name, fields in axis_all.items()
    ])
    d = d.with_columns(
        axis_count=pl.when(
            pl.all_horizontal([pl.col(f"axrec_{n}") for n in AXIS_FIELDS])
        ).then(
            pl.sum_horizontal([pl.col(f"ax_{n}").cast(pl.Int32) for n in AXIS_FIELDS])
        ).otherwise(None),
        axis5_count=pl.when(
            pl.all_horizontal([pl.col(f"axrec_{n}") for n in axis_all])
        ).then(
            pl.sum_horizontal([pl.col(f"ax_{n}").cast(pl.Int32) for n in axis_all])
        ).otherwise(None),
    )
    return_binary_check = d.filter(
        pl.col("sev_known")
    ).select(
        ((pl.col("sev_count") > 0) != (pl.col("axis_count") > 0)).sum()
    ).item()
    assert return_binary_check < 50, (
        "the 3-axis and 4-field binary phenotypes disagree on "
        f"{return_binary_check} cases; the axis grouping is wrong")

    xwalk = pl.read_parquet(
        PATHS.results / "atlas" / "municipality_atlas.parquet"
    ).select("munic_code", "health_region_code")
    banded = d.join(xwalk, left_on="municipality_residence_code7",
                    right_on="munic_code", how="left")
    unmatched = int(banded["health_region_code"].null_count())
    if unmatched > 0.01 * banded.height:
        raise AssertionError(
            f"{unmatched} of {banded.height} confirmed cases did not match a "
            "health region; the crosswalk or the key width is wrong")
    banded = banded.join(bands(), on="health_region_code", how="inner")
    return d, banded


# ---------------------------------------------------------------------------
# 1. National clinical presentation profile
# ---------------------------------------------------------------------------
def national_profile(d: pl.DataFrame) -> pl.DataFrame:
    rows = []
    n_total = d.height
    for field, (label, role) in SIGNS.items():
        recorded = int(d[f"{field}__rec"].sum())
        present = int((d[f"{field}__rec"] & d[f"{field}__yes"]).sum())
        p, lo, hi = prop([present], [recorded])
        rows.append({
            "field": field, "sign": label, "classic_role": role,
            "confirmed_cases": n_total, "recorded": recorded,
            "pct_recorded": 100 * recorded / n_total,
            "present": present,
            "prevalence_pct": 100 * float(p[0]),
            "lo": 100 * float(lo[0]), "hi": 100 * float(hi[0]),
        })
    return pl.DataFrame(rows).sort("prevalence_pct", descending=True)


def sign_lethality(d: pl.DataFrame) -> pl.DataFrame:
    """Case fatality with and without each sign; a risk ratio, not a rate ratio.

    This is what decides whether the four-marker index is the right four.
    """
    k = d.filter(pl.col("outcome_ok"))
    rows = []
    for field, (label, role) in SIGNS.items():
        s = k.filter(pl.col(f"{field}__rec"))
        yes = s.filter(pl.col(f"{field}__yes"))
        no = s.filter(~pl.col(f"{field}__yes"))
        a, n1 = int(yes["died"].sum()), yes.height
        b, n0 = int(no["died"].sum()), no.height
        p, lo, hi = prop([a, b], [n1, n0])
        rr, rlo, rhi = katz_rr(a, n1, b, n0)
        rows.append({
            "field": field, "sign": label, "classic_role": role,
            "known_outcome_with_sign": n1, "deaths_with_sign": a,
            "cfr_present_pct": 100 * float(p[0]),
            "cfr_present_lo": 100 * float(lo[0]), "cfr_present_hi": 100 * float(hi[0]),
            "known_outcome_without_sign": n0, "deaths_without_sign": b,
            "cfr_absent_pct": 100 * float(p[1]),
            "cfr_absent_lo": 100 * float(lo[1]), "cfr_absent_hi": 100 * float(hi[1]),
            "risk_ratio": rr, "rr_lo": rlo, "rr_hi": rhi,
        })
    return pl.DataFrame(rows).sort("risk_ratio", descending=True)


# ---------------------------------------------------------------------------
# 2. Does the profile differ across the depth gradient?
# ---------------------------------------------------------------------------
def profile_by_band(b: pl.DataFrame, *, subset: str = "all") -> pl.DataFrame:
    rows = []
    band_n = {r["quintile"]: r["n"] for r in
              b.group_by("quintile").agg(pl.len().alias("n")).iter_rows(named=True)}
    for field, (label, role) in SIGNS.items():
        g = b.group_by("quintile").agg(
            pl.col(f"{field}__rec").sum().alias("recorded"),
            (pl.col(f"{field}__rec") & pl.col(f"{field}__yes")).sum().alias("present"),
        ).sort("quintile")
        p, lo, hi = prop(g["present"].to_numpy(), g["recorded"].to_numpy())
        for i, r in enumerate(g.iter_rows(named=True)):
            rows.append({
                "subset": subset, "field": field, "sign": label,
                "classic_role": role, "quintile": r["quintile"],
                "band_cases": band_n[r["quintile"]],
                "recorded": r["recorded"],
                "pct_recorded": 100 * r["recorded"] / band_n[r["quintile"]],
                "present": r["present"],
                "prevalence_pct": 100 * float(p[i]),
                "lo": 100 * float(lo[i]), "hi": 100 * float(hi[i]),
            })
    return pl.DataFrame(rows)


def band_spread(prof: pl.DataFrame) -> pl.DataFrame:
    """Q5-minus-Q1 in percentage points and the Q5/Q1 prevalence ratio per sign."""
    rows = []
    for field, (label, role) in SIGNS.items():
        s = prof.filter(pl.col("field") == field).sort("quintile").to_dicts()
        q1, q5 = s[0], s[-1]
        rows.append({
            "field": field, "sign": label, "classic_role": role,
            "q1_pct": q1["prevalence_pct"], "q5_pct": q5["prevalence_pct"],
            "q5_minus_q1_pp": q5["prevalence_pct"] - q1["prevalence_pct"],
            "q5_over_q1": (q5["prevalence_pct"] / q1["prevalence_pct"]
                           if q1["prevalence_pct"] else float("nan")),
            "disjoint_intervals": bool(q1["hi"] < q5["lo"] or q5["hi"] < q1["lo"]),
            "group": ("mild triad" if field in MILD_FIELDS else
                      "severe index" if field in SEVERE_FIELDS else "other"),
        })
    return pl.DataFrame(rows).sort("q5_over_q1", descending=True)


# ---------------------------------------------------------------------------
# 3. A graded severity index, and the severity-adjusted central claim
# ---------------------------------------------------------------------------
def cfr_by_severity_count(d: pl.DataFrame, col: str = "sev_count") -> pl.DataFrame:
    k = d.filter(pl.col("outcome_ok") & pl.col(col).is_not_null())
    g = k.group_by(col).agg(
        pl.len().alias("known_outcomes"),
        pl.col("died").sum().alias("deaths"),
        pl.col("hosp").sum().alias("hospitalised"),
    ).sort(col)
    cf, lo, hi = prop(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy())
    hs, hlo, hhi = prop(g["hospitalised"].to_numpy(), g["known_outcomes"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("hosp_share_lo", 100 * hlo), pl.Series("hosp_share_hi", 100 * hhi),
    ).rename({col: "severity_count"})


def severity_mix_by_band(b: pl.DataFrame, col: str = "sev_count") -> pl.DataFrame:
    """Case mix: the share of cases at each severity level, by band."""
    k = b.filter(pl.col(col).is_not_null())
    g = k.group_by(["quintile", col]).agg(pl.len().alias("n"))
    tot = g.group_by("quintile").agg(pl.col("n").sum().alias("total"))
    out = g.join(tot, on="quintile")
    p, lo, hi = prop(out["n"].to_numpy(), out["total"].to_numpy())
    return out.with_columns(
        pl.Series("pct", 100 * p), pl.Series("lo", 100 * lo), pl.Series("hi", 100 * hi)
    ).rename({col: "severity_count"}).sort(["quintile", "severity_count"])


def cfr_by_band_within_count(b: pl.DataFrame, col: str = "sev_count") -> pl.DataFrame:
    """THE test: case fatality by depth band, holding presentation severity fixed."""
    k = b.filter(pl.col("outcome_ok") & pl.col(col).is_not_null())
    g = k.group_by([col, "quintile"]).agg(
        pl.len().alias("known_outcomes"),
        pl.col("died").sum().alias("deaths"),
    ).sort([col, "quintile"])
    cf, lo, hi = prop(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    ).rename({col: "severity_count"})


def haemorrhage_collinearity(d: pl.DataFrame) -> dict:
    """Plausibility gate on the non-monotone 4-field count.

    A count index whose case fatality plateaus between levels 2 and 3 is a bug
    until proven otherwise. It is not a data error: two of the four fields are
    the same variable, so level 3 is usually two real axes wearing three field
    names, exactly like level 2.
    """
    k = d.filter(pl.col("cli_hemorr__rec") & pl.col("cli_hemopu__rec"))
    both = int((k["cli_hemorr__yes"] & k["cli_hemopu__yes"]).sum())
    only_h = int((k["cli_hemorr__yes"] & ~k["cli_hemopu__yes"]).sum())
    only_p = int((~k["cli_hemorr__yes"] & k["cli_hemopu__yes"]).sum())
    return {
        "both_recorded": k.height,
        "concordant": int(k.height - only_h - only_p),
        "pct_concordant": 100 * (k.height - only_h - only_p) / k.height,
        "haemorrhage_only": only_h, "pulmonary_only": only_p,
        "both_present": both,
        "verdict": ("cli_hemorr and cli_hemopu are one variable in practice; a "
                    "count over the four published fields double-weights the "
                    "haemorrhage axis"),
    }


def within_stratum_rr(b: pl.DataFrame, col: str) -> pl.DataFrame:
    """Q5 vs Q1 case-fatality RISK RATIO within each severity stratum, with CIs.

    Reported per stratum rather than only as a pooled figure because the ratios
    are heterogeneous, and a single Mantel-Haenszel number over heterogeneous
    strata hides the shape of the finding.
    """
    k = b.filter(pl.col("outcome_ok") & pl.col(col).is_not_null())
    rows = []
    for lv in sorted(k[col].unique().to_list()):
        q5 = k.filter((pl.col(col) == lv) & (pl.col("quintile") == 5))
        q1 = k.filter((pl.col(col) == lv) & (pl.col("quintile") == 1))
        a, n1 = int(q5["died"].sum()), q5.height
        bd, n0 = int(q1["died"].sum()), q1.height
        rr, lo, hi = katz_rr(a, n1, bd, n0)
        rows.append({
            "index": col, "level": int(lv),
            "q1_known": n0, "q1_deaths": bd, "q5_known": n1, "q5_deaths": a,
            "risk_ratio_q5_vs_q1": rr, "rr_lo": lo, "rr_hi": hi,
            "excludes_null": bool(np.isfinite(lo) and lo > 1),
        })
    return pl.DataFrame(rows)


def mild_signs_within_severity(b: pl.DataFrame) -> pl.DataFrame:
    """The sharpest form of the depth prediction.

    Mild-sign prevalence should be flat across bands *within* a fixed severity
    level. If it is flat overall only because the severity mix happens to
    cancel, that would be luck; if it is flat within the mildest stratum too,
    the bands really are finding the same illness and seeing it the same way.
    """
    rows = []
    for lv, tag in ((0, "no severe axis"), (None, "any severe axis")):
        s = (b.filter(pl.col("axis_count") == 0) if lv == 0
             else b.filter(pl.col("axis_count") > 0))
        for field in MILD_FIELDS + ["cli_pantur", "cli_conges", "cli_prost"]:
            g = s.group_by("quintile").agg(
                pl.col(f"{field}__rec").sum().alias("recorded"),
                (pl.col(f"{field}__rec") & pl.col(f"{field}__yes")).sum().alias("present"),
            ).sort("quintile")
            p, lo, hi = prop(g["present"].to_numpy(), g["recorded"].to_numpy())
            for i, r in enumerate(g.iter_rows(named=True)):
                rows.append({
                    "stratum": tag, "field": field, "sign": SIGNS[field][0],
                    "quintile": r["quintile"], "recorded": r["recorded"],
                    "present": r["present"], "prevalence_pct": 100 * float(p[i]),
                    "lo": 100 * float(lo[i]), "hi": 100 * float(hi[i]),
                })
    return pl.DataFrame(rows)


def residual_heterogeneity(b: pl.DataFrame, col: str = "axis_count") -> pl.DataFrame:
    """Is a severity stratum actually homogeneous across bands?

    Named threat: the strata hold *recorded markers* fixed, not severity. If a
    "no severe axis" case in Q5 is far more often hospitalised than the same
    nominal case in Q1, the stratum still contains a severity gradient, and the
    stratified estimate remains an upper bound on any depth-only effect.
    Hospitalisation is the independent severity read here: it is the field that
    defines the bands, but *within* a band it varies case by case.
    """
    k = b.filter(pl.col(col).is_not_null())
    g = k.group_by([col, "quintile"]).agg(
        pl.len().alias("cases"), pl.col("hosp").sum().alias("hospitalised"),
    ).sort([col, "quintile"])
    p, lo, hi = prop(g["hospitalised"].to_numpy(), g["cases"].to_numpy())
    return g.with_columns(
        pl.Series("hosp_share_pct", 100 * p), pl.Series("lo", 100 * lo),
        pl.Series("hi", 100 * hi),
    ).rename({col: "severity_axes"})


def cfr_band_within_axis_hospitalised(b: pl.DataFrame) -> pl.DataFrame:
    """The tightest severity control the data allow: same axes AND admitted.

    Restricting to admitted cases removes the ambulatory tail whose presence in
    the deep bands is the whole point of the depth account, so this comparison
    is between cases that all cleared the same care threshold and carry the same
    recorded organ involvement.
    """
    k = b.filter(pl.col("outcome_ok") & pl.col("axis_count").is_not_null()
                 & pl.col("hosp"))
    g = k.group_by(["axis_count", "quintile"]).agg(
        pl.len().alias("known_outcomes"), pl.col("died").sum().alias("deaths"),
    ).sort(["axis_count", "quintile"])
    cf, lo, hi = prop(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    )


#: Signs that depend on a patient history being taken, as against signs a chart
#: or an examination yields on its own. The split is stated in advance because
#: the test below is only interpretable if the grouping was not chosen after
#: seeing which signs moved.
HISTORY_SIGNS = ["cli_febre", "cli_mialgi", "cli_cefale", "cli_pantur",
                 "cli_prost", "cli_vomito"]


def history_taking_score(b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Why does headache thin out where surveillance is shallow?

    Candidate mechanism: forms filled from a hospital chart record fewer
    patient-reported symptoms than forms filled at an interview. If that is it,
    the deficit should appear (a) across bands within the mildest severity
    stratum and (b) *within a single band*, between admitted and non-admitted
    cases -- where no territorial difference can be doing the work.
    """
    k = b.filter(
        pl.all_horizontal([pl.col(f"{c}__rec") for c in HISTORY_SIGNS])
        & (pl.col("axis_count") == 0)
    ).with_columns(
        pl.sum_horizontal([pl.col(f"{c}__yes").cast(pl.Int32)
                           for c in HISTORY_SIGNS]).alias("history_score")
    )
    by_band = k.group_by("quintile").agg(
        pl.len().alias("cases"),
        pl.col("history_score").mean().alias("mean_history_signs"),
        pl.col("hosp").mean().alias("hosp_share"),
    ).sort("quintile")
    # `hosp` is null where ATE_HOSP was never filled, and a null admission is
    # not a non-admission; the contrast is only defined on recorded cases.
    by_hosp = k.filter(pl.col("ate_hosp").is_in(["sim", "nao"])).group_by(
        ["quintile", "hosp"]).agg(
        pl.len().alias("cases"),
        pl.col("history_score").mean().alias("mean_history_signs"),
        (pl.col("cli_cefale__yes").mean() * 100).alias("headache_pct"),
        (pl.col("cli_pantur__yes").mean() * 100).alias("calf_pain_pct"),
    ).sort(["quintile", "hosp"])
    return by_band, by_hosp


def standardised_cfr(b: pl.DataFrame, col: str, label: str) -> pl.DataFrame:
    """Direct standardisation of band case fatality to Q1's severity mix.

    Q1 is the standard because it is the deepest-detecting band: the question is
    what each band's case fatality would be if it were finding the same mix of
    presentations that the deepest surveillance finds.
    """
    k = b.filter(pl.col("outcome_ok") & pl.col(col).is_not_null())
    levels = sorted(k[col].unique().to_list())
    # Standard weights: Q1's distribution over severity levels among cases with
    # a known outcome and a defined count.
    q1 = k.filter(pl.col("quintile") == 1)
    w = np.array([q1.filter(pl.col(col) == lv).height for lv in levels], float)
    w = w / w.sum()

    rows = []
    for q in sorted(k["quintile"].unique().to_list()):
        s = k.filter(pl.col("quintile") == q)
        n = np.array([s.filter(pl.col(col) == lv).height for lv in levels], float)
        dth = np.array([s.filter((pl.col(col) == lv) & pl.col("died")).height
                        for lv in levels], float)
        crude, clo, chi = prop([dth.sum()], [n.sum()])
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, dth / n, np.nan)
        ok = n > 0
        wv = w[ok] / w[ok].sum()          # renormalise if a stratum is empty
        adj = float(np.sum(wv * p[ok]))
        var = float(np.sum(wv ** 2 * p[ok] * (1 - p[ok]) / n[ok]))
        se = np.sqrt(var)
        rows.append({
            "index": label, "quintile": q,
            "known_outcomes": int(n.sum()), "deaths": int(dth.sum()),
            "crude_cfr_pct": 100 * float(crude[0]),
            "crude_lo": 100 * float(clo[0]), "crude_hi": 100 * float(chi[0]),
            "adjusted_cfr_pct": 100 * adj,
            "adjusted_lo": 100 * max(adj - 1.96 * se, 0.0),
            "adjusted_hi": 100 * (adj + 1.96 * se),
            "strata_used": int(ok.sum()),
        })
    return pl.DataFrame(rows)


def mh_gradient(b: pl.DataFrame, col: str) -> dict:
    """Q5 vs Q1 case-fatality risk ratio, Mantel-Haenszel over severity strata."""
    k = b.filter(pl.col("outcome_ok") & pl.col(col).is_not_null())
    levels = sorted(k[col].unique().to_list())
    a, n1, bb, n0 = [], [], [], []
    for lv in levels:
        q5 = k.filter((pl.col(col) == lv) & (pl.col("quintile") == 5))
        q1 = k.filter((pl.col(col) == lv) & (pl.col("quintile") == 1))
        a.append(int(q5["died"].sum())); n1.append(q5.height)
        bb.append(int(q1["died"].sum())); n0.append(q1.height)
    crude = katz_rr(sum(a), sum(n1), sum(bb), sum(n0))
    out = mh_risk_ratio(a, n1, bb, n0)
    out["crude_rr"] = crude[0]
    out["crude_lo"], out["crude_hi"] = crude[1], crude[2]
    out["levels"] = [int(x) for x in levels]
    out["q5_deaths"], out["q5_known"] = int(sum(a)), int(sum(n1))
    out["q1_deaths"], out["q1_known"] = int(sum(bb)), int(sum(n0))
    return out


# ---------------------------------------------------------------------------
# 4. Work-relatedness
# ---------------------------------------------------------------------------
def work_by_band(b: pl.DataFrame) -> pl.DataFrame:
    n = b.group_by("quintile").agg(pl.len().alias("band_cases")).sort("quintile")
    g = b.group_by("quintile").agg(
        pl.col("doenca_tra").is_in(["sim", "nao"]).sum().alias("recorded"),
        (pl.col("doenca_tra") == "sim").sum().alias("work_related"),
    ).sort("quintile").join(n, on="quintile")
    p, lo, hi = prop(g["work_related"].to_numpy(), g["recorded"].to_numpy())
    return g.with_columns(
        (100 * pl.col("recorded") / pl.col("band_cases")).alias("pct_recorded"),
        pl.Series("work_pct", 100 * p), pl.Series("work_lo", 100 * lo),
        pl.Series("work_hi", 100 * hi),
    )


def profile_by_work(d: pl.DataFrame) -> pl.DataFrame:
    """Do occupational cases present differently? Sign prevalence by work status."""
    rows = []
    for status in ("sim", "nao"):
        s = d.filter(pl.col("doenca_tra") == status)
        for field, (label, _role) in SIGNS.items():
            rec = int(s[f"{field}__rec"].sum())
            pres = int((s[f"{field}__rec"] & s[f"{field}__yes"]).sum())
            p, lo, hi = prop([pres], [rec])
            rows.append({
                "work_related": status, "field": field, "sign": label,
                "recorded": rec, "present": pres,
                "prevalence_pct": 100 * float(p[0]),
                "lo": 100 * float(lo[0]), "hi": 100 * float(hi[0]),
            })
    return pl.DataFrame(rows)


def work_outcome(d: pl.DataFrame) -> pl.DataFrame:
    k = d.filter(pl.col("outcome_ok") & pl.col("doenca_tra").is_in(["sim", "nao"]))
    g = k.group_by("doenca_tra").agg(
        pl.len().alias("known_outcomes"),
        pl.col("died").sum().alias("deaths"),
        pl.col("hosp").sum().alias("hospitalised"),
        (pl.col("sev_count").is_not_null() & (pl.col("sev_count") > 0)).sum()
            .alias("severe"),
        pl.col("sev_count").is_not_null().sum().alias("sev_known"),
    ).sort("doenca_tra")
    cf, lo, hi = prop(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy())
    hs, _, _ = prop(g["hospitalised"].to_numpy(), g["known_outcomes"].to_numpy())
    sv, slo, shi = prop(g["severe"].to_numpy(), g["sev_known"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
        pl.Series("hosp_share_pct", 100 * hs),
        pl.Series("severe_share_pct", 100 * sv),
        pl.Series("severe_lo", 100 * slo), pl.Series("severe_hi", 100 * shi),
    )


def cfr_by_band_within_work(b: pl.DataFrame) -> pl.DataFrame:
    k = b.filter(pl.col("outcome_ok") & pl.col("doenca_tra").is_in(["sim", "nao"]))
    g = k.group_by(["doenca_tra", "quintile"]).agg(
        pl.len().alias("known_outcomes"), pl.col("died").sum().alias("deaths"),
    ).sort(["doenca_tra", "quintile"])
    cf, lo, hi = prop(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    )


# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, b = load()
    print(f"confirmed cases: {d.height:,}; mapped to a banded health region: "
          f"{b.height:,} ({100 * b.height / d.height:.1f}%)")

    # --- 1. national profile -------------------------------------------------
    nat = national_profile(d)
    nat.write_csv(OUT / "national_sign_prevalence.csv")
    print("\n=== 1. NATIONAL CLINICAL PROFILE, % of confirmed cases with the sign recorded ===")
    print(f"{'sign':<26}{'recorded':>9}{'%rec':>7}{'present':>9}{'prev%':>8}"
          f"{'95% CI':>16}   classic role")
    for r in nat.iter_rows(named=True):
        print(f"{r['sign']:<26}{r['recorded']:>9,}{r['pct_recorded']:>7.1f}"
              f"{r['present']:>9,}{r['prevalence_pct']:>8.1f}"
              f"{r['lo']:>8.1f}-{r['hi']:<7.1f} {r['classic_role']}")

    # Plausibility gate: the catalogue (DATA.md 16.2) was built independently.
    catalogue = {"cli_febre": 58608, "cli_mialgi": 54709, "cli_cefale": 47981,
                 "cli_icteri": 31984, "cli_conges": 11313, "cli_mening": 1693}
    mismatch = {f: (int(nat.filter(pl.col("field") == f)["present"][0]), v)
                for f, v in catalogue.items()
                if int(nat.filter(pl.col("field") == f)["present"][0]) != v}
    print(f"  cross-check against DATA.md 16.2 counts: "
          f"{'MISMATCH ' + str(mismatch) if mismatch else 'all 6 spot checks match'}")

    leth = sign_lethality(d)
    leth.write_csv(OUT / "sign_lethality.csv")
    print("\n=== case fatality by sign at notification (risk ratio present vs absent) ===")
    print(f"{'sign':<26}{'CFR+':>8}{'CFR-':>8}{'RR':>7}{'95% CI':>16}")
    for r in leth.iter_rows(named=True):
        print(f"{r['sign']:<26}{r['cfr_present_pct']:>7.2f}%{r['cfr_absent_pct']:>7.2f}%"
              f"{r['risk_ratio']:>7.2f}{r['rr_lo']:>8.2f}-{r['rr_hi']:<7.2f}")

    # --- 2. profile across the depth gradient --------------------------------
    prof = profile_by_band(b)
    prof.write_csv(OUT / "sign_prevalence_by_band.csv")
    spread = band_spread(prof)
    spread.write_csv(OUT / "sign_band_spread.csv")
    print("\n=== 2. SIGN PREVALENCE BY DEPTH BAND (Q1 deepest -> Q5 shallowest) ===")
    print(f"{'sign':<26}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5':>7}"
          f"{'Q5/Q1':>8}  group")
    for r in spread.iter_rows(named=True):
        v = prof.filter(pl.col("field") == r["field"]).sort("quintile")["prevalence_pct"]
        print(f"{r['sign']:<26}" + "".join(f"{x:>7.1f}" for x in v)
              + f"{r['q5_over_q1']:>8.2f}  {r['group']}")

    comp = prof.group_by("quintile").agg(
        pl.col("pct_recorded").mean().alias("mean_pct_recorded")).sort("quintile")
    print("  clinical-block recording completeness by band: "
          + "  ".join(f"Q{r['quintile']}={r['mean_pct_recorded']:.1f}%"
                      for r in comp.iter_rows(named=True)))

    mild = spread.filter(pl.col("group") == "mild triad")
    sev = spread.filter(pl.col("group") == "severe index")
    print(f"  mild triad  Q5/Q1: "
          + ", ".join(f"{r['sign']} {r['q5_over_q1']:.2f}" for r in mild.iter_rows(named=True)))
    print(f"  severe index Q5/Q1: "
          + ", ".join(f"{r['sign']} {r['q5_over_q1']:.2f}" for r in sev.iter_rows(named=True)))

    # Named threat: differential recording. Shallow bands record the block less
    # often, so a band difference in prevalence could be a difference in WHO gets
    # a completed form rather than in who is sick. Restrict to cases with all 15
    # signs recorded and see whether the pattern holds.
    bc = b.filter(pl.col("block_complete"))
    prof_bc = profile_by_band(bc, subset="complete_block")
    pl.concat([prof, prof_bc]).write_csv(OUT / "sign_prevalence_by_band.csv")
    spread_bc = band_spread(prof_bc)
    spread_bc.write_csv(OUT / "sign_band_spread_complete_block.csv")
    print(f"\n  [threat: differential recording] complete-block cases "
          f"{bc.height:,}/{b.height:,} ({100 * bc.height / b.height:.1f}%); "
          "Q5/Q1 recomputed on them:")
    for r in spread_bc.filter(pl.col("group") != "other").iter_rows(named=True):
        print(f"    {r['sign']:<24} {r['q1_pct']:>6.1f} -> {r['q5_pct']:>6.1f}  "
              f"ratio {r['q5_over_q1']:.2f}")

    # --- 3. graded severity index -------------------------------------------
    print("\n=== 3. GRADED SEVERITY INDEX: count of severe markers at notification ===")
    cnt = cfr_by_severity_count(d, "sev_count")
    cnt.write_csv(OUT / "cfr_by_severity_count.csv")
    for r in cnt.iter_rows(named=True):
        print(f"  count={r['severity_count']}  n={r['known_outcomes']:>6,} "
              f"deaths={r['deaths']:>5,}  CFR {r['cfr_pct']:>6.2f}% "
              f"({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})  hosp {r['hosp_share_pct']:.1f}%")
    monotone = all(x < y for x, y in zip(cnt["cfr_pct"].to_list(),
                                         cnt["cfr_pct"].to_list()[1:]))
    print(f"  monotone in the count: {monotone}")

    # PLAUSIBILITY GATE. A severity count whose case fatality plateaus is a bug
    # until diagnosed. Diagnose it before using the index for anything.
    coll = haemorrhage_collinearity(d)
    print(f"  [gate] cli_hemorr vs cli_hemopu: {coll['pct_concordant']:.2f}% concordant "
          f"({coll['haemorrhage_only']} + {coll['pulmonary_only']} discordant of "
          f"{coll['both_recorded']:,}) -> the 4-field count double-weights haemorrhage")
    axis = cfr_by_severity_count(d, "axis_count")
    axis.write_csv(OUT / "cfr_by_axis_count.csv")
    axis_monotone = all(x < y for x, y in zip(axis["cfr_pct"].to_list(),
                                              axis["cfr_pct"].to_list()[1:]))
    print("  corrected 3-axis index (jaundice / renal / haemorrhage):")
    for r in axis.iter_rows(named=True):
        print(f"    axes={r['severity_count']}  n={r['known_outcomes']:>6,} "
              f"deaths={r['deaths']:>5,}  CFR {r['cfr_pct']:>6.2f}% "
              f"({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})  hosp {r['hosp_share_pct']:.1f}%")
    print(f"    monotone in the axis count: {axis_monotone}")
    axis5 = cfr_by_severity_count(d, "axis5_count")
    axis5.write_csv(OUT / "cfr_by_axis5_count.csv")
    axis5_monotone = all(x < y for x, y in zip(axis5["cfr_pct"].to_list(),
                                               axis5["cfr_pct"].to_list()[1:]))
    print("  5-axis index (adds respiratory, cardiac):")
    for r in axis5.iter_rows(named=True):
        print(f"    axes={r['severity_count']}  n={r['known_outcomes']:>6,}  "
              f"CFR {r['cfr_pct']:>6.2f}% ({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})")
    print(f"    monotone in the 5-axis count: {axis5_monotone}")

    mix = severity_mix_by_band(b, "sev_count")
    mix.write_csv(OUT / "severity_mix_by_band.csv")
    print("\n  case mix by band (% of cases with a defined count):")
    piv = mix.pivot(on="severity_count", index="quintile", values="pct").sort("quintile")
    print(piv)

    mix_axis = severity_mix_by_band(b, "axis_count")
    mix_axis.write_csv(OUT / "axis_severity_mix_by_band.csv")

    within = pl.concat([
        cfr_by_band_within_count(b, "sev_count").with_columns(pl.lit("4-field").alias("index")),
        cfr_by_band_within_count(b, "axis_count").with_columns(pl.lit("3-axis").alias("index")),
    ])
    within.write_csv(OUT / "cfr_by_band_within_severity_count.csv")
    within = within.filter(pl.col("index") == "3-axis")
    print("\n  DECISIVE: case fatality by band, WITHIN each severity level (3-axis index)")
    print(f"{'level':>6}{'Q1':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}{'Q5':>10}{'Q5/Q1':>8}"
          f"{'nQ1':>8}{'nQ5':>8}")
    within_ratio = {}
    for lv in sorted(within["severity_count"].unique().to_list()):
        s = within.filter(pl.col("severity_count") == lv).sort("quintile").to_dicts()
        if len(s) < 5:
            continue
        ratio = s[4]["cfr_pct"] / s[0]["cfr_pct"] if s[0]["cfr_pct"] else float("nan")
        within_ratio[int(lv)] = ratio
        print(f"{lv:>6}" + "".join(f"{x['cfr_pct']:>9.2f}%" for x in s)
              + f"{ratio:>8.2f}{s[0]['known_outcomes']:>8,}{s[4]['known_outcomes']:>8,}")

    wsr = pl.concat([within_stratum_rr(b, "sev_count"),
                     within_stratum_rr(b, "axis_count"),
                     within_stratum_rr(b, "axis5_count")])
    wsr.write_csv(OUT / "within_stratum_risk_ratio.csv")
    print("  Q5 vs Q1 risk ratio within each 3-axis level (heterogeneous by design):")
    for r in wsr.filter(pl.col("index") == "axis_count").iter_rows(named=True):
        print(f"    axes={r['level']}  RR {r['risk_ratio_q5_vs_q1']:.2f} "
              f"({r['rr_lo']:.2f}-{r['rr_hi']:.2f})  excludes 1: {r['excludes_null']}")

    std_bin = standardised_cfr(
        b.with_columns(
            pl.when(pl.col("sev_count").is_not_null())
              .then((pl.col("sev_count") > 0).cast(pl.Int32)).otherwise(None)
              .alias("sev_binary")),
        "sev_binary", "binary severe phenotype")
    std_cnt = standardised_cfr(b, "sev_count", "severe marker count 0-4")
    std_axis = standardised_cfr(b, "axis_count", "severity axis count 0-3")
    std_axis5 = standardised_cfr(b, "axis5_count", "severity axis count 0-5")
    pl.concat([std_bin, std_cnt, std_axis, std_axis5]).write_csv(
        OUT / "standardised_cfr_by_band.csv")
    print("\n  case fatality standardised to Q1's presentation mix:")
    for tab in (std_bin, std_cnt, std_axis, std_axis5):
        rr = tab.sort("quintile").to_dicts()
        crude = rr[4]["crude_cfr_pct"] / rr[0]["crude_cfr_pct"]
        adj = rr[4]["adjusted_cfr_pct"] / rr[0]["adjusted_cfr_pct"]
        print(f"    {rr[0]['index']:<28} crude Q5/Q1 {crude:.2f}x -> "
              f"adjusted {adj:.2f}x   (Q5 {rr[4]['crude_cfr_pct']:.2f}% -> "
              f"{rr[4]['adjusted_cfr_pct']:.2f}%)")

    mh_bin = mh_gradient(
        b.with_columns(
            pl.when(pl.col("sev_count").is_not_null())
              .then((pl.col("sev_count") > 0).cast(pl.Int32)).otherwise(None)
              .alias("sev_binary")), "sev_binary")
    mh_cnt = mh_gradient(b, "sev_count")
    mh_axis = mh_gradient(b, "axis_count")
    mh_axis5 = mh_gradient(b, "axis5_count")
    print("\n  Mantel-Haenszel Q5 vs Q1 case-fatality risk ratio:")
    for name, m in (("binary", mh_bin), ("4-field 0-4", mh_cnt),
                    ("3-axis 0-3", mh_axis), ("5-axis 0-5", mh_axis5)):
        print(f"    {name:<14} crude {m['crude_rr']:.2f} "
              f"({m['crude_lo']:.2f}-{m['crude_hi']:.2f})   MH-adjusted "
              f"{m['rr_mh']:.2f} ({m['lo']:.2f}-{m['hi']:.2f})  strata={m['strata']}")

    # Named threat: the strata are not homogeneous. Level 0 in Q5 is not the
    # same case as level 0 in Q1, and the RR of 8.87 at level 0 says so.
    rh = residual_heterogeneity(b)
    rh.write_csv(OUT / "residual_heterogeneity_within_axis.csv")
    print("\n  [threat: strata not homogeneous] hospitalisation share WITHIN each axis level")
    print(f"{'axes':>6}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}")
    for lv in sorted(rh["severity_axes"].unique().to_list()):
        s = rh.filter(pl.col("severity_axes") == lv).sort("quintile")
        print(f"{lv:>6}" + "".join(f"{x:>8.1f}%" for x in s["hosp_share_pct"]))

    hosp_within = cfr_band_within_axis_hospitalised(b)
    hosp_within.write_csv(OUT / "cfr_by_band_within_axis_hospitalised.csv")
    print("  case fatality by band within axis level, ADMITTED CASES ONLY")
    print(f"{'axes':>6}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}{'Q5/Q1':>8}{'nQ1':>7}")
    hosp_ratio = {}
    for lv in sorted(hosp_within["axis_count"].unique().to_list()):
        s = hosp_within.filter(pl.col("axis_count") == lv).sort("quintile").to_dicts()
        if len(s) < 5:
            continue
        ratio = s[4]["cfr_pct"] / s[0]["cfr_pct"] if s[0]["cfr_pct"] else float("nan")
        hosp_ratio[int(lv)] = ratio
        print(f"{lv:>6}" + "".join(f"{x['cfr_pct']:>8.2f}%" for x in s)
              + f"{ratio:>8.2f}{s[0]['known_outcomes']:>7,}")

    msw = mild_signs_within_severity(b)
    msw.write_csv(OUT / "mild_signs_within_severity.csv")
    print("\n  mild-sign prevalence by band WITHIN the no-severe-axis stratum")
    print(f"{'sign':<24}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5':>7}{'Q5/Q1':>8}")
    for field in MILD_FIELDS + ["cli_pantur", "cli_conges", "cli_prost"]:
        s = msw.filter((pl.col("field") == field)
                       & (pl.col("stratum") == "no severe axis")).sort("quintile")
        v = s["prevalence_pct"].to_list()
        print(f"{SIGNS[field][0]:<24}" + "".join(f"{x:>7.1f}" for x in v)
              + f"{v[4] / v[0]:>8.2f}")

    hs_band, hs_hosp = history_taking_score(b)
    hs_band.write_csv(OUT / "history_taking_score_by_band.csv")
    hs_hosp.write_csv(OUT / "history_taking_score_by_admission.csv")
    print("\n  mechanism: patient-reported signs recorded per case (0-6), "
          "mild stratum only")
    for r in hs_band.iter_rows(named=True):
        print(f"    Q{r['quintile']}  n={r['cases']:>6,}  "
              f"mean history signs {r['mean_history_signs']:.2f}  "
              f"(admitted {100 * r['hosp_share']:.0f}% of them)")
    print("  the same contrast WITHIN a band, admitted vs not:")
    for r in hs_hosp.iter_rows(named=True):
        print(f"    Q{r['quintile']} {'admitted    ' if r['hosp'] else 'not admitted'}"
              f"  n={r['cases']:>6,}  history {r['mean_history_signs']:.2f}  "
              f"headache {r['headache_pct']:.1f}%  calf pain {r['calf_pain_pct']:.1f}%")

    # --- 4. work-relatedness -------------------------------------------------
    print("\n=== 4. WORK-RELATEDNESS ACROSS THE GRADIENT ===")
    wb = work_by_band(b)
    wb.write_csv(OUT / "work_relatedness_by_band.csv")
    for r in wb.iter_rows(named=True):
        print(f"  Q{r['quintile']}  recorded {r['pct_recorded']:>5.1f}%  "
              f"work-related {r['work_pct']:>5.1f}% ({r['work_lo']:.1f}-{r['work_hi']:.1f})"
              f"  n={r['recorded']:,}")
    wo = work_outcome(d)
    wo.write_csv(OUT / "work_outcome.csv")
    for r in wo.iter_rows(named=True):
        print(f"  doenca_tra={r['doenca_tra']:<4} n={r['known_outcomes']:>6,}  "
              f"CFR {r['cfr_pct']:>5.2f}% ({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})  "
              f"severe {r['severe_share_pct']:.1f}%  hosp {r['hosp_share_pct']:.1f}%")
    pw = profile_by_work(d)
    pw.write_csv(OUT / "sign_prevalence_by_work.csv")
    print("  sign prevalence, work-related vs not (pp difference, largest first):")
    diffs = []
    for field, (label, _r) in SIGNS.items():
        y = pw.filter((pl.col("field") == field) & (pl.col("work_related") == "sim"))["prevalence_pct"][0]
        n = pw.filter((pl.col("field") == field) & (pl.col("work_related") == "nao"))["prevalence_pct"][0]
        diffs.append((label, y, n, y - n))
    for label, y, n, dd in sorted(diffs, key=lambda t: -abs(t[3]))[:6]:
        print(f"    {label:<24} work {y:>5.1f}%  non-work {n:>5.1f}%  {dd:+.1f} pp")
    cw = cfr_by_band_within_work(b)
    cw.write_csv(OUT / "cfr_by_band_within_work.csv")
    print("  case fatality by band within work status:")
    work_ratio = {}
    for st in ("sim", "nao"):
        s = cw.filter(pl.col("doenca_tra") == st).sort("quintile").to_dicts()
        ratio = s[4]["cfr_pct"] / s[0]["cfr_pct"] if s[0]["cfr_pct"] else float("nan")
        work_ratio[st] = ratio
        print(f"    {st}: " + " ".join(f"Q{x['quintile']}={x['cfr_pct']:.2f}%" for x in s)
              + f"   Q5/Q1 {ratio:.2f}x")

    # --- report --------------------------------------------------------------
    report = {
        "confirmed_cases": d.height,
        "banded_cases": b.height,
        "catalogue_spot_check_mismatches": mismatch,
        "national_profile": nat.to_dicts(),
        "sign_lethality_top": leth.head(6).to_dicts(),
        "band_spread": spread.to_dicts(),
        "band_spread_complete_block": spread_bc.to_dicts(),
        "clinical_block_recording_by_band": comp.to_dicts(),
        "cfr_by_severity_count_4field": cnt.to_dicts(),
        "cfr_by_severity_count_4field_monotone": bool(monotone),
        "haemorrhage_collinearity": coll,
        "cfr_by_axis_count": axis.to_dicts(),
        "cfr_by_axis_count_monotone": bool(axis_monotone),
        "cfr_by_axis5_count": axis5.to_dicts(),
        "cfr_by_axis5_count_monotone": bool(axis5_monotone),
        "within_count_q5_over_q1_3axis": {str(k): round(v, 2)
                                          for k, v in within_ratio.items()},
        "within_stratum_risk_ratio": wsr.to_dicts(),
        "residual_heterogeneity_within_axis": rh.to_dicts(),
        "cfr_band_within_axis_admitted_only_q5_over_q1": {
            str(k): round(v, 2) for k, v in hosp_ratio.items()},
        "mild_signs_within_severity": msw.to_dicts(),
        "history_taking_score_by_band": hs_band.to_dicts(),
        "history_taking_score_by_admission": hs_hosp.to_dicts(),
        "standardisation": {
            "binary": std_bin.to_dicts(), "count_4field": std_cnt.to_dicts(),
            "axis_0_3": std_axis.to_dicts(), "axis_0_5": std_axis5.to_dicts(),
        },
        "mantel_haenszel_q5_vs_q1": {
            "binary": mh_bin, "count_0_4": mh_cnt, "axis_0_3": mh_axis,
            "axis_0_5": mh_axis5},
        "work_relatedness_by_band": wb.to_dicts(),
        "work_outcome": wo.to_dicts(),
        "work_cfr_band_ratio": {k: round(v, 2) for k, v in work_ratio.items()},
    }
    (OUT / "clinical_profile_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
