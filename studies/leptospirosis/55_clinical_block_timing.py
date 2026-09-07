"""Is the SINAN clinical block a notification-time snapshot, taken before the
patient deteriorates?

`docs/DATA.md` §16.2 states it as fact ("The clinical form is completed **at
notification**, before deterioration") and `RESEARCH_JOURNAL.md` thread T-8
offers it as the likeliest reading of an epidemiologically impossible number:
case fatality in the *non-severe* stratum (no jaundice, no renal impairment, no
haemorrhage, no pulmonary haemorrhage) rises 8.87-fold across the depth bands,
from 0.75% to 6.67%, on 159 deaths in 2,383 known outcomes. If the block is a
snapshot, the non-severe label is a moment in time, not a phenotype, and Link
3d's >=23% case-mix bound is measuring the wrong thing.

Nobody has dated the block. This script does.

THE TEST, as specified. If the block reflects status at notification, then among
cases admitted AFTER notification (the form was filled with no hospital course
in view) the severe-marker prevalence should be LOWER than among those notified
on or after admission (the clinician had already seen the patient in hospital).

THE STRONGER TEST, which the data turn out to support. 5,978 of 5,981 confirmed
leptospirosis deaths carry `DT_OBITO`. For a large minority the notification date
falls ON OR AFTER the date of death: the entire clinical course, up to and
including the patient dying, was over before the form was dated. If the
non-severe label were a pre-deterioration snapshot it must be RARE in that
group. It is not.

Three named threats are carried through the whole script:

  TH-A  Ordering is confounded with surveillance depth. Hospital-based
        notification is the norm where surveillance is shallow, and shallow
        territories have sicker confirmed cases. Every ordering contrast is
        therefore reported within depth quintile as well as pooled.
  TH-B  Ordering is confounded with case severity through *selection*, not
        recording: a patient notified in an outpatient clinic and admitted
        later was well enough to be an outpatient first. This makes the
        severe-prevalence contrast uninterpretable on its own and is why the
        case-fatality and post-death arms carry the argument.
  TH-C  Date fields contain year-typing errors. A notification 729 days after
        admission is a keying error, not an ordering fact. Gates are named at
        the top and the headline is re-run without them.

Outputs to ``data/results/clinical_block_timing/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brepi.analysis.rates import binom_ci  # noqa: E402
from brepi.config import PATHS  # noqa: E402

OUT = PATHS.results / "clinical_block_timing"

#: The severe leptospirosis phenotype, identical to 44_depth_vs_severity.py.
SEVERE_FIELDS = ["cli_icteri", "cli_renal", "cli_hemorr", "cli_hemopu"]
MARKER_LABEL = {
    "cli_icteri": "jaundice",
    "cli_renal": "renal impairment",
    "cli_hemorr": "haemorrhage",
    "cli_hemopu": "pulmonary haemorrhage",
}

#: Mirrors 40_ascertainment_depth.py so the bands are the published bands.
MIN_CASES = 30

#: TH-C gates. An interval this large between two dates on the same form is a
#: year-typing error; it is excluded from the ORDERING classification only, and
#: the headline contrast is repeated ungated in `sensitivity_gate`.
MAX_NOTIF_ADM_DAYS = 60
MAX_NOTIF_OBITO_DAYS = 90

DATE_COLS = ["DT_NOTIFIC", "DT_SIN_PRI", "ATE_DT_INT", "DT_OBITO",
             "DT_INVEST", "DT_ENCERRA"]

#: The eleven signs the severe phenotype ignores. If the non-severe deaths carry
#: these, the four-marker definition is too narrow and the anomaly is a
#: DEFINITION artefact rather than a timing or recording one.
OTHER_SIGNS = ["cli_respir", "cli_cardia", "cli_conges", "cli_mening",
               "cli_diarre", "cli_vomito", "cli_prost", "cli_pantur",
               "cli_febre", "cli_mialgi", "cli_cefale"]

# The published banding, imported rather than re-derived.
_spec = importlib.util.spec_from_file_location(
    "_asc_depth", Path(__file__).with_name("40_ascertainment_depth.py"))
_asc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_asc)
case_weighted_quintiles = _asc._case_weighted_quintiles


# ---------------------------------------------------------------------------
# Interval helper. Unpack order is (ESTIMATE, lo, hi) and is asserted, because
# reading it as (lo, mid, hi) has corrupted results in this repo before.
# ---------------------------------------------------------------------------
def prop(count, total):
    p, lo, hi = binom_ci(np.asarray(count, float), np.asarray(total, float))
    ok = np.isfinite(p)
    assert np.all(lo[ok] <= p[ok] + 1e-12) and np.all(p[ok] <= hi[ok] + 1e-12), \
        "binom_ci estimate outside its own interval — unpack order is wrong"
    return p, lo, hi


def pct_cols(df: pl.DataFrame, count: str, total: str, prefix: str) -> pl.DataFrame:
    p, lo, hi = prop(df[count], df[total])
    return df.with_columns(pl.Series(f"{prefix}_pct", 100 * p),
                           pl.Series(f"{prefix}_lo", 100 * lo),
                           pl.Series(f"{prefix}_hi", 100 * hi))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load() -> tuple[pl.DataFrame, dict]:
    """Confirmed cases with parsed dates, the severity split and the depth band."""
    keep = (DATE_COLS + SEVERE_FIELDS + OTHER_SIGNS
            + ["municipality_residence_code7", "src_year", "ate_hosp",
               "evolucao", "evolucao_state"])
    line = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter(pl.col("classi_fin") == "confirmado")
        .select(keep)
        .with_columns([pl.col(c).str.to_date("%Y-%m-%d", strict=False).alias(c)
                       for c in DATE_COLS])
        .collect()
    )
    n_confirmed = line.height

    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.with_columns((pl.col("hospitalised") / pl.col("cases")).alias("hosp_share"))
    banded = case_weighted_quintiles(hr.filter(pl.col("cases") >= MIN_CASES),
                                     "hosp_share")
    xwalk = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet") \
        .select("munic_code", "health_region_code", "region")

    line = (line.join(xwalk, left_on="municipality_residence_code7",
                      right_on="munic_code", how="left")
                .join(banded.select("health_region_code", "quintile", "hosp_share"),
                      on="health_region_code", how="left"))

    line = line.with_columns(
        sev_known=pl.all_horizontal([pl.col(c).is_in(["sim", "nao"])
                                     for c in SEVERE_FIELDS]),
        severe=pl.any_horizontal([pl.col(c) == "sim" for c in SEVERE_FIELDS]),
        died=(pl.col("evolucao") == "obito_por_leptospirose"),
        outcome_ok=(pl.col("evolucao_state") == "valid"),
        admitted=(pl.col("ate_hosp") == "sim"),
        notif_minus_adm=(pl.col("DT_NOTIFIC") - pl.col("ATE_DT_INT")).dt.total_days(),
        notif_minus_obito=(pl.col("DT_NOTIFIC") - pl.col("DT_OBITO")).dt.total_days(),
        invest_minus_notif=(pl.col("DT_INVEST") - pl.col("DT_NOTIFIC")).dt.total_days(),
        adm_minus_obito=(pl.col("DT_OBITO") - pl.col("ATE_DT_INT")).dt.total_days(),
    )

    # Ordering, gated per TH-C. Null where the gate bites or a date is absent.
    g = pl.col("notif_minus_adm").abs() <= MAX_NOTIF_ADM_DAYS
    line = line.with_columns(
        ordering=pl.when(pl.col("notif_minus_adm").is_null() | ~g)
        .then(None)
        .when(pl.col("notif_minus_adm") < 0).then(pl.lit("notif_before_adm"))
        .when(pl.col("notif_minus_adm") == 0).then(pl.lit("same_day"))
        .otherwise(pl.lit("notif_after_adm"))
    )

    banded_line = line.filter(pl.col("quintile").is_not_null())
    prov = {
        "confirmed_cases": n_confirmed,
        "in_banded_health_regions": banded_line.height,
        "outside_banded_health_regions": n_confirmed - banded_line.height,
        "date_completeness_pct": {
            c: round(100 * line[c].drop_nulls().len() / n_confirmed, 2)
            for c in DATE_COLS},
        "admitted_ate_hosp_sim": int(line["admitted"].sum()),
        "admitted_with_admission_date": int(
            line.filter(pl.col("admitted") & pl.col("ATE_DT_INT").is_not_null()).height),
        "not_admitted_with_admission_date": int(
            line.filter((pl.col("ate_hosp") == "nao")
                        & pl.col("ATE_DT_INT").is_not_null()).height),
        "deaths": int(line["died"].sum()),
        "deaths_with_death_date": int(
            line.filter(pl.col("died") & pl.col("DT_OBITO").is_not_null()).height),
        # DT_INVEST is the natural timestamp for the investigation form that
        # CARRIES the clinical block. It is only an independent timestamp if it
        # differs from DT_NOTIFIC — this is the check that decides whether the
        # investigation-date arm is worth running at all.
        "dt_invest_equals_dt_notific_pct": round(
            100 * (line["DT_INVEST"] == line["DT_NOTIFIC"]).sum()
            / line["DT_INVEST"].drop_nulls().len(), 2),
        "gate_MAX_NOTIF_ADM_DAYS": MAX_NOTIF_ADM_DAYS,
        "gate_MAX_NOTIF_OBITO_DAYS": MAX_NOTIF_OBITO_DAYS,
        "ordering_gated_out": int(
            line.filter(pl.col("notif_minus_adm").is_not_null()
                        & pl.col("ordering").is_null()).height),
    }
    return banded_line, prov


def reproduce_bands(line: pl.DataFrame) -> dict:
    """Guard: the line-level crude gradient must be the published one.

    If it is not, everything below is a decomposition of some other quantity.
    """
    g = (line.group_by("quintile")
         .agg(pl.len().alias("cases"),
              pl.col("outcome_ok").sum().alias("outcome_known"),
              (pl.col("outcome_ok") & pl.col("died")).sum().alias("deaths"))
         .sort("quintile"))
    cfr, lo, hi = prop(g["deaths"], g["outcome_known"])
    assert abs(100 * cfr[0] - 3.00) < 0.20 and abs(100 * cfr[-1] - 17.09) < 0.20, \
        f"band reproduction failed: Q1={100 * cfr[0]:.2f}% Q5={100 * cfr[-1]:.2f}%"
    return {"cfr_pct_by_quintile": [round(100 * v, 3) for v in cfr],
            "deaths": g["deaths"].to_list(),
            "outcome_known": g["outcome_known"].to_list(),
            "crude_gradient_Q5_over_Q1": float(cfr[-1] / cfr[0])}


# ---------------------------------------------------------------------------
# A. When is the form dated, relative to the hospital course?
# ---------------------------------------------------------------------------
def ordering_distribution(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    adm = line.filter(pl.col("admitted") & pl.col("ATE_DT_INT").is_not_null())
    v = adm["notif_minus_adm"].drop_nulls().to_numpy()
    q = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    rows = []
    for scope, sub in [("Brazil", adm)] + [
            (f"Q{k}", adm.filter(pl.col("quintile") == k)) for k in range(1, 6)]:
        s = sub.filter(pl.col("ordering").is_not_null())
        n = s.height
        rows.append({
            "scope": scope, "admitted_with_both_dates": n,
            "notif_before_adm": int((s["ordering"] == "notif_before_adm").sum()),
            "same_day": int((s["ordering"] == "same_day").sum()),
            "notif_after_adm": int((s["ordering"] == "notif_after_adm").sum()),
            "pct_notif_after_adm": round(
                100 * (s["ordering"] == "notif_after_adm").sum() / n, 2),
            "median_notif_minus_adm_days": float(
                np.median(s["notif_minus_adm"].to_numpy())),
        })
    tab = pl.DataFrame(rows)
    summary = {
        "n_admitted_with_admission_date": adm.height,
        "notif_minus_adm_percentiles_1_5_10_25_50_75_90_95_99":
            [float(x) for x in q],
        "pct_notified_strictly_after_admission": round(100 * float((v > 0).mean()), 2),
        "pct_notified_same_day": round(100 * float((v == 0).mean()), 2),
        "pct_notified_strictly_before_admission": round(100 * float((v < 0).mean()), 2),
    }
    return tab, summary


# ---------------------------------------------------------------------------
# B. THE TEST — severe-marker prevalence by ordering
# ---------------------------------------------------------------------------
ORDER = ["notif_before_adm", "same_day", "notif_after_adm"]


def severe_prevalence_by_ordering(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Pooled and within depth quintile (TH-A), and per marker.

    Per marker matters: jaundice is usually present at presentation, whereas
    renal failure and pulmonary haemorrhage are the late events. A pure timing
    artefact should move the late markers much more than jaundice.
    """
    d = line.filter(pl.col("admitted") & pl.col("ordering").is_not_null()
                    & pl.col("sev_known"))
    rows = []
    for scope, sub in [("Brazil", d)] + [
            (f"Q{k}", d.filter(pl.col("quintile") == k)) for k in range(1, 6)]:
        g = (sub.group_by("ordering")
             .agg(pl.len().alias("n"),
                  pl.col("severe").sum().alias("severe"),
                  *[(pl.col(c) == "sim").sum().alias(c) for c in SEVERE_FIELDS])
             .sort("ordering"))
        for r in g.iter_rows(named=True):
            rows.append({"scope": scope, **r})
    tab = pl.DataFrame(rows)
    tab = pct_cols(tab, "severe", "n", "severe_share")
    for c in SEVERE_FIELDS:
        p, _, _ = prop(tab[c], tab["n"])
        tab = tab.with_columns(pl.Series(f"{c}_pct", 100 * p))

    def get(scope, order, col="severe_share_pct"):
        r = tab.filter((pl.col("scope") == scope) & (pl.col("ordering") == order))
        return float(r[col][0]) if r.height else float("nan")

    summary = {
        "prediction": ("if the block is a notification-time snapshot, severe "
                       "prevalence is LOWEST where notification precedes admission"),
        "pooled": {o: {"n": int(tab.filter((pl.col("scope") == "Brazil")
                                           & (pl.col("ordering") == o))["n"][0]),
                       "severe_pct": get("Brazil", o),
                       "severe_lo": get("Brazil", o, "severe_share_lo"),
                       "severe_hi": get("Brazil", o, "severe_share_hi")}
                   for o in ORDER},
        "pooled_pp_after_minus_before": get("Brazil", "notif_after_adm")
            - get("Brazil", "notif_before_adm"),
        "pooled_prevalence_ratio_after_over_before":
            get("Brazil", "notif_after_adm") / get("Brazil", "notif_before_adm"),
        "within_quintile_pp_after_minus_before": {
            f"Q{k}": get(f"Q{k}", "notif_after_adm") - get(f"Q{k}", "notif_before_adm")
            for k in range(1, 6)},
        "by_marker_pooled_pp_after_minus_before": {
            MARKER_LABEL[c]: get("Brazil", "notif_after_adm", f"{c}_pct")
            - get("Brazil", "notif_before_adm", f"{c}_pct") for c in SEVERE_FIELDS},
    }
    return tab, summary


def severe_prevalence_by_lag(line: pl.DataFrame) -> pl.DataFrame:
    """The same test read as a dose-response in days of hospital course seen."""
    d = line.filter(pl.col("admitted") & pl.col("ordering").is_not_null()
                    & pl.col("sev_known"))
    bins = [(-60, -8, "notified 8-60d before admission"),
            (-7, -3, "notified 3-7d before"),
            (-2, -1, "notified 1-2d before"),
            (0, 0, "same day"),
            (1, 2, "notified 1-2d after"),
            (3, 6, "notified 3-6d after"),
            (7, 13, "notified 7-13d after"),
            (14, 60, "notified 14-60d after")]
    rows = []
    for lo, hi, lab in bins:
        s = d.filter(pl.col("notif_minus_adm").is_between(lo, hi))
        rows.append({"lag_band": lab, "lag_lo": lo, "lag_hi": hi, "n": s.height,
                     "severe": int(s["severe"].sum())})
    return pct_cols(pl.DataFrame(rows), "severe", "n", "severe_share")


# ---------------------------------------------------------------------------
# C. Does the non-severe case-fatality gradient survive the restriction?
# ---------------------------------------------------------------------------
def nonsevere_cfr(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Case fatality (deaths / known outcomes), non-severe stratum, by band.

    Strata: the whole confirmed series (the T-8 number), admitted cases only,
    and admitted cases split by ordering. Admitted-only is a DIFFERENT estimand
    from the published one — the banding variable is hospitalisation share, so
    conditioning on admission removes most of Q1's mild cases and almost none of
    Q5's. Reported side by side and labelled, never substituted.
    """
    def block(sub: pl.DataFrame, label: str) -> pl.DataFrame:
        s = sub.filter(pl.col("sev_known") & pl.col("outcome_ok"))
        g = (s.group_by("quintile")
             .agg(pl.len().alias("outcome_known"),
                  pl.col("died").sum().alias("deaths"),
                  (~pl.col("severe")).sum().alias("nonsevere_known"),
                  (~pl.col("severe") & pl.col("died")).sum().alias("nonsevere_deaths"),
                  pl.col("severe").sum().alias("severe_known"),
                  (pl.col("severe") & pl.col("died")).sum().alias("severe_deaths"))
             .sort("quintile"))
        g = pct_cols(g, "nonsevere_deaths", "nonsevere_known", "cfr_nonsevere")
        g = pct_cols(g, "severe_deaths", "severe_known", "cfr_severe")
        g = pct_cols(g, "deaths", "outcome_known", "cfr_all")
        return g.with_columns(pl.lit(label).alias("stratum"))

    adm = line.filter(pl.col("admitted"))
    strata = [
        (line, "all confirmed cases (published T-8 population)"),
        (adm, "admitted only"),
        (adm.filter(pl.col("ordering") == "notif_before_adm"),
         "admitted, notified BEFORE admission"),
        (adm.filter(pl.col("ordering") == "same_day"),
         "admitted, notified same day"),
        (adm.filter(pl.col("ordering") == "notif_after_adm"),
         "admitted, notified AFTER admission"),
        (adm.filter(pl.col("notif_minus_adm").is_between(3, MAX_NOTIF_ADM_DAYS)),
         "admitted, notified >=3d AFTER admission"),
        (adm.filter(pl.col("notif_minus_adm").is_between(7, MAX_NOTIF_ADM_DAYS)),
         "admitted, notified >=7d AFTER admission"),
    ]
    tab = pl.concat([block(s, lab) for s, lab in strata], how="diagonal")

    def grad(label: str) -> dict:
        s = tab.filter(pl.col("stratum") == label).sort("quintile")
        if s.height < 5 or s["nonsevere_known"].min() == 0:
            return {"note": "incomplete band coverage", "n_bands": s.height}
        v = s["cfr_nonsevere_pct"].to_numpy()
        return {
            "Q1_pct": float(v[0]), "Q1_lo": float(s["cfr_nonsevere_lo"][0]),
            "Q1_hi": float(s["cfr_nonsevere_hi"][0]),
            "Q5_pct": float(v[-1]), "Q5_lo": float(s["cfr_nonsevere_lo"][-1]),
            "Q5_hi": float(s["cfr_nonsevere_hi"][-1]),
            "gradient_Q5_over_Q1": float(v[-1] / v[0]) if v[0] > 0 else float("nan"),
            "nonsevere_deaths_total": int(s["nonsevere_deaths"].sum()),
            "nonsevere_known_total": int(s["nonsevere_known"].sum()),
        }

    return tab, {lab: grad(lab) for _, lab in strata}


# ---------------------------------------------------------------------------
# D. The stronger test — the form dated after the patient died
# ---------------------------------------------------------------------------
def deaths_vs_death_date(line: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Among confirmed deaths, was the form dated before or after the death?

    A form dated on or after DT_OBITO was completed with the whole course, up to
    and including the death, already in the past. A pre-deterioration snapshot
    cannot exist there. If the non-severe label is a timing artefact its
    prevalence must collapse in that group.
    """
    d = line.filter(pl.col("died") & pl.col("sev_known")
                    & pl.col("DT_OBITO").is_not_null()
                    & (pl.col("notif_minus_obito").abs() <= MAX_NOTIF_OBITO_DAYS))
    groups = [
        ("notified BEFORE death", pl.col("notif_minus_obito") < 0),
        ("notified ON the death date", pl.col("notif_minus_obito") == 0),
        ("notified 1-2d AFTER death", pl.col("notif_minus_obito").is_between(1, 2)),
        ("notified >=3d AFTER death", pl.col("notif_minus_obito") >= 3),
    ]
    rows = []
    for scope, sub in [("Brazil", d)] + [
            (f"Q{k}", d.filter(pl.col("quintile") == k)) for k in range(1, 6)]:
        for lab, expr in groups:
            s = sub.filter(expr)
            rows.append({"scope": scope, "group": lab, "deaths": s.height,
                         "nonsevere": int((~s["severe"]).sum()),
                         **{f"{c}_sim": int((s[c] == "sim").sum())
                            for c in SEVERE_FIELDS}})
        s = sub.filter(pl.col("notif_minus_obito") >= 0)
        rows.append({"scope": scope, "group": "notified ON/AFTER death (all)",
                     "deaths": s.height, "nonsevere": int((~s["severe"]).sum()),
                     **{f"{c}_sim": int((s[c] == "sim").sum())
                        for c in SEVERE_FIELDS}})
    tab = pct_cols(pl.DataFrame(rows), "nonsevere", "deaths", "nonsevere_share")

    # Recording quality in the same split. If forms dated after the death are
    # filled from an administrative source with no clinical detail, the block
    # should be RECORDED less often there — a different defect from a snapshot,
    # but also not a phenotype.
    allsub = line.filter(pl.col("died") & pl.col("DT_OBITO").is_not_null()
                         & (pl.col("notif_minus_obito").abs() <= MAX_NOTIF_OBITO_DAYS))
    rec_rows = []
    for lab, expr in [("notified BEFORE death", pl.col("notif_minus_obito") < 0),
                      ("notified ON/AFTER death", pl.col("notif_minus_obito") >= 0)]:
        s = allsub.filter(expr)
        rec_rows.append({"group": lab, "deaths": s.height,
                         "sev_known": int(s["sev_known"].sum()),
                         **{f"{c}_ignorado": int((s[c] == "ignorado").sum())
                            for c in SEVERE_FIELDS}})
    rec = pct_cols(pl.DataFrame(rec_rows), "sev_known", "deaths", "block_recorded")

    def get(scope, group, col="nonsevere_share_pct"):
        r = tab.filter((pl.col("scope") == scope) & (pl.col("group") == group))
        return float(r[col][0]) if r.height else float("nan")

    before, after = "notified BEFORE death", "notified ON/AFTER death (all)"
    summary = {
        "prediction_if_snapshot": ("the non-severe share among deaths collapses "
                                   "once the form is dated after the death"),
        "n_deaths_analysed": d.height,
        "nonsevere_share_pct_notified_before_death": get("Brazil", before),
        "nonsevere_share_pct_notified_on_or_after_death": get("Brazil", after),
        "nonsevere_share_pct_notified_3d_or_more_after_death":
            get("Brazil", "notified >=3d AFTER death"),
        "risk_ratio_after_over_before":
            get("Brazil", after) / get("Brazil", before),
        "by_quintile_nonsevere_share_pct_on_or_after_death": {
            f"Q{k}": get(f"Q{k}", after) for k in range(1, 6)},
        "by_quintile_deaths_on_or_after_death": {
            f"Q{k}": int(tab.filter((pl.col("scope") == f"Q{k}")
                                    & (pl.col("group") == after))["deaths"][0])
            for k in range(1, 6)},
        "block_recorded_pct_before_death": float(
            rec.filter(pl.col("group") == "notified BEFORE death")["block_recorded_pct"][0]),
        "block_recorded_pct_on_or_after_death": float(
            rec.filter(pl.col("group") == "notified ON/AFTER death")["block_recorded_pct"][0]),
    }
    return tab, rec, summary


def admission_to_death(line: pl.DataFrame) -> dict:
    """How long is the window in which a snapshot could go stale?"""
    d = line.filter(pl.col("died") & pl.col("DT_OBITO").is_not_null()
                    & pl.col("ATE_DT_INT").is_not_null()
                    & pl.col("adm_minus_obito").is_between(-5, 180))
    v = d["adm_minus_obito"].to_numpy()
    out = {"n": d.height,
           "admission_to_death_days_p10_p25_p50_p75_p90":
               [float(x) for x in np.percentile(v, [10, 25, 50, 75, 90])],
           "pct_dead_within_2_days_of_admission": round(100 * float((v <= 2).mean()), 2)}
    ns = d.filter(pl.col("sev_known") & ~pl.col("severe"))
    sv = d.filter(pl.col("sev_known") & pl.col("severe"))
    out["median_admission_to_death_nonsevere_label"] = float(
        np.median(ns["adm_minus_obito"].to_numpy())) if ns.height else float("nan")
    out["median_admission_to_death_severe_label"] = float(
        np.median(sv["adm_minus_obito"].to_numpy())) if sv.height else float("nan")
    out["n_nonsevere_label"], out["n_severe_label"] = ns.height, sv.height
    return out


# ---------------------------------------------------------------------------
# E. What this does to the >=23% case-mix bound
# ---------------------------------------------------------------------------
def casemix_bound(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Link 3d's decomposition, recomputed under two named restrictions.

    Link 3d standardises case fatality on the measured severe/non-severe split
    with Q1's mix as the standard, and reports
    ``1 - log(std ratio) / log(crude ratio)`` as the share of the log gradient
    the MEASURED case mix accounts for.

    R1 — drop cases whose block is demonstrably premature: confirmed cases
         admitted whose notification date is strictly before the admission date.
         This is a subset of the SAME case series and keeps the mild
         non-admitted cases that make Q1 what it is, so it is comparable.

    R2 — an extreme reclassification bound. NAMED THREAT: every death labelled
         non-severe is a misclassified severe case. Move all of them into the
         severe stratum and re-standardise. This is the largest the measured
         case-mix share could be if the entire non-severe-death anomaly is
         misrecording. It is a bound, not an estimate.

    R3 — the phenotype was too narrow. Respiratory and cardiac involvement are
         recorded on the same form and are severity in any clinical reading;
         section E shows respiratory involvement in 46% of non-severe-labelled
         deaths against 12% of non-severe-labelled survivors. Widen the
         definition to six markers and redo the standardisation. This is not a
         bound but a better measurement of the same construct.
    """
    def decompose(sub: pl.DataFrame, label: str, reclass_ns_deaths: bool = False,
                  markers: list[str] | None = None):
        markers = markers or SEVERE_FIELDS
        known = pl.all_horizontal([pl.col(c).is_in(["sim", "nao"]) for c in markers])
        sev = pl.any_horizontal([pl.col(c) == "sim" for c in markers])
        s = sub.filter(known)
        if reclass_ns_deaths:
            sev = sev | pl.col("died")
        s = s.with_columns(sev.alias("_sev"))
        g = (s.group_by("quintile").agg(
                pl.len().alias("sev_known"),
                pl.col("_sev").sum().alias("severe"),
                (~pl.col("_sev")).sum().alias("nonsevere"),
                (pl.col("_sev") & pl.col("outcome_ok")).sum().alias("severe_ok"),
                (pl.col("_sev") & pl.col("outcome_ok") & pl.col("died")).sum()
                .alias("severe_deaths"),
                ((~pl.col("_sev")) & pl.col("outcome_ok")).sum().alias("nonsevere_ok"),
                ((~pl.col("_sev")) & pl.col("outcome_ok") & pl.col("died")).sum()
                .alias("nonsevere_deaths"),
                pl.col("outcome_ok").sum().alias("outcome_known"),
                (pl.col("outcome_ok") & pl.col("died")).sum().alias("deaths"))
             .sort("quintile"))
        cfr_s, _, _ = prop(g["severe_deaths"], g["severe_ok"])
        cfr_m, _, _ = prop(g["nonsevere_deaths"], g["nonsevere_ok"])
        cfr_a, _, _ = prop(g["deaths"], g["outcome_known"])
        w_s = float(g["severe"][0] / g["sev_known"][0])
        std = w_s * cfr_s + (1 - w_s) * cfr_m
        share = float(1 - np.log(std[-1] / std[0]) / np.log(cfr_a[-1] / cfr_a[0]))
        row = g.with_columns(
            pl.lit(label).alias("restriction"),
            pl.Series("cfr_all_pct", 100 * cfr_a),
            pl.Series("cfr_severe_pct", 100 * cfr_s),
            pl.Series("cfr_nonsevere_pct", 100 * cfr_m),
            pl.Series("cfr_standardised_pct", 100 * std))
        info = {"crude_gradient_Q5_over_Q1": float(cfr_a[-1] / cfr_a[0]),
                "standardised_gradient_Q5_over_Q1": float(std[-1] / std[0]),
                "case_mix_share_of_log_gradient": share,
                "severe_share_Q1": w_s,
                "severe_share_Q5": float(g["severe"][-1] / g["sev_known"][-1]),
                "sev_known_total": int(g["sev_known"].sum())}
        return row, info

    premature = (pl.col("admitted") & (pl.col("notif_minus_adm") < 0)
                 & (pl.col("notif_minus_adm") >= -MAX_NOTIF_ADM_DAYS))
    wide = SEVERE_FIELDS + ["cli_respir", "cli_cardia"]
    specs = [
        (line, "published (no restriction)", False, None),
        (line.filter(~premature.fill_null(False)),
         "R1: drop admitted cases notified before admission", False, None),
        (line, "R2: every non-severe death reclassified severe", True, None),
        (line, "R3: phenotype widened with respiratory + cardiac", False, wide),
    ]
    tabs, info = [], {}
    for sub, lab, rc, mk in specs:
        t, i = decompose(sub, lab, rc, mk)
        tabs.append(t)
        info[lab] = i
    info["R1_cases_dropped"] = int(line.filter(premature.fill_null(False)).height)
    info["R1_pct_of_series_dropped"] = round(
        100 * line.filter(premature.fill_null(False)).height / line.height, 2)
    return pl.concat(tabs, how="diagonal"), info


def deaths_labelled_nonsevere_by_band(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """The anomaly, stated as a labelling rate rather than a case fatality.

    What share of each band's DEATHS carry a block with none of the four markers?
    This is the quantity a misclassification story has to move, and unlike the
    non-severe case fatality it does not depend on the size of the non-severe
    denominator — which is exactly what varies across the bands.
    """
    d = line.filter(pl.col("died") & pl.col("sev_known"))
    g = (d.group_by("quintile").agg(pl.len().alias("deaths_block_recorded"),
                                    (~pl.col("severe")).sum().alias("labelled_nonsevere"))
         .sort("quintile"))
    g = pct_cols(g, "labelled_nonsevere", "deaths_block_recorded", "nonsevere_label")
    v = g["nonsevere_label_pct"].to_numpy()
    return g, {"by_quintile_pct": [float(x) for x in v],
               "Q1_over_Q5": float(v[0] / v[-1]),
               "deaths_with_block_recorded": int(g["deaths_block_recorded"].sum())}


def phenotype_breadth(line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Rival explanation: the four-marker phenotype is simply too narrow.

    Compare the other eleven recorded signs between non-severe-labelled cases
    who died and non-severe-labelled cases who survived. Respiratory and cardiac
    involvement are severity in any clinical reading; if they are concentrated
    in the non-severe deaths, the four-marker phenotype is missing severity that
    the form actually recorded, and the anomaly needs no timing story at all.
    """
    ns = line.filter(pl.col("sev_known") & ~pl.col("severe") & pl.col("outcome_ok"))
    rows = []
    for sign in OTHER_SIGNS:
        s = ns.filter(pl.col(sign).is_in(["sim", "nao"]))
        died = s.filter(pl.col("died"))
        surv = s.filter(~pl.col("died"))
        rows.append({
            "sign": sign, "label": sign.replace("cli_", ""),
            "n_died": died.height, "sim_died": int((died[sign] == "sim").sum()),
            "n_survived": surv.height, "sim_survived": int((surv[sign] == "sim").sum()),
        })
    tab = pl.DataFrame(rows)
    tab = pct_cols(tab, "sim_died", "n_died", "prev_died")
    tab = pct_cols(tab, "sim_survived", "n_survived", "prev_survived")
    tab = tab.with_columns(
        (pl.col("prev_died_pct") - pl.col("prev_survived_pct")).alias("pp_diff"),
        (pl.col("prev_died_pct") / pl.col("prev_survived_pct")).alias("prevalence_ratio"),
    ).sort("pp_diff", descending=True)

    # How much of the anomaly would a broader phenotype absorb? Add respiratory
    # and cardiac involvement to the definition and re-read the non-severe
    # case fatality gradient.
    wide_sev = pl.any_horizontal(
        [pl.col(c) == "sim" for c in SEVERE_FIELDS + ["cli_respir", "cli_cardia"]])
    wide_known = pl.all_horizontal(
        [pl.col(c).is_in(["sim", "nao"])
         for c in SEVERE_FIELDS + ["cli_respir", "cli_cardia"]])
    w = (line.filter(wide_known & pl.col("outcome_ok"))
         .with_columns(wide_sev.alias("_sev"))
         .group_by("quintile").agg(
             ((~pl.col("_sev"))).sum().alias("nonsevere_known"),
             ((~pl.col("_sev")) & pl.col("died")).sum().alias("nonsevere_deaths"),
             pl.len().alias("outcome_known"),
             pl.col("died").sum().alias("deaths"))
         .sort("quintile"))
    w = pct_cols(w, "nonsevere_deaths", "nonsevere_known", "cfr_nonsevere_wide")
    wv = w["cfr_nonsevere_wide_pct"].to_numpy()
    summary = {
        "n_nonsevere_with_known_outcome": ns.height,
        "n_nonsevere_deaths": int(ns["died"].sum()),
        "top_signs_by_pp_difference_died_minus_survived": [
            {"sign": r["label"], "prev_died_pct": r["prev_died_pct"],
             "prev_survived_pct": r["prev_survived_pct"],
             "pp_diff": r["pp_diff"], "prevalence_ratio": r["prevalence_ratio"]}
            for r in tab.head(5).iter_rows(named=True)],
        "widened_phenotype_adds": ["cli_respir", "cli_cardia"],
        "widened_nonsevere_cfr_pct_by_quintile": [float(x) for x in wv],
        "widened_nonsevere_cfr_Q1": float(wv[0]), "widened_nonsevere_cfr_Q5": float(wv[-1]),
        "widened_nonsevere_gradient_Q5_over_Q1": float(wv[-1] / wv[0]),
        "widened_nonsevere_deaths_total": int(w["nonsevere_deaths"].sum()),
        "widened_nonsevere_known_total": int(w["nonsevere_known"].sum()),
    }
    return tab, (w, summary)


def sensitivity_gate(line: pl.DataFrame) -> dict:
    """TH-C: repeat the headline ordering contrast with no date gate at all."""
    d = line.filter(pl.col("admitted") & pl.col("ATE_DT_INT").is_not_null()
                    & pl.col("sev_known"))
    out = {}
    for lab, gate in [("gate +/-60d (headline)", 60), ("gate +/-30d", 30),
                      ("gate +/-14d", 14), ("no gate", None)]:
        s = d if gate is None else d.filter(pl.col("notif_minus_adm").abs() <= gate)
        b = s.filter(pl.col("notif_minus_adm") < 0)
        a = s.filter(pl.col("notif_minus_adm") > 0)
        pb, _, _ = prop([b["severe"].sum()], [b.height])
        pa, _, _ = prop([a["severe"].sum()], [a.height])
        out[lab] = {"n_before": b.height, "severe_pct_before": float(100 * pb[0]),
                    "n_after": a.height, "severe_pct_after": float(100 * pa[0]),
                    "pp_after_minus_before": float(100 * (pa[0] - pb[0]))}
    return out


# ===========================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print("=" * 78)
    print("loading")
    line, prov = load()
    report["provenance"] = prov
    print(f"  confirmed {prov['confirmed_cases']:,}; "
          f"in banded health regions {prov['in_banded_health_regions']:,}")
    print(f"  admission date present for "
          f"{prov['admitted_with_admission_date']:,} of "
          f"{prov['admitted_ate_hosp_sim']:,} admitted cases "
          f"({100 * prov['admitted_with_admission_date'] / prov['admitted_ate_hosp_sim']:.1f}%)")
    print(f"  death date present for {prov['deaths_with_death_date']:,} of "
          f"{prov['deaths']:,} leptospirosis deaths")
    print(f"  DT_INVEST == DT_NOTIFIC in {prov['dt_invest_equals_dt_notific_pct']}% "
          f"of records -> the investigation date is NOT an independent timestamp "
          f"for the block; that arm is dropped")

    report["band_reproduction"] = reproduce_bands(line)
    print(f"  band check: crude case fatality "
          f"{report['band_reproduction']['cfr_pct_by_quintile']} "
          f"({report['band_reproduction']['crude_gradient_Q5_over_Q1']:.2f}x)")

    # ---- A ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("A. When is the notification dated relative to the admission?")
    ord_tab, ord_sum = ordering_distribution(line)
    ord_tab.write_csv(OUT / "ordering_distribution.csv")
    report["A_ordering_distribution"] = ord_sum
    for r in ord_tab.iter_rows(named=True):
        print(f"  {r['scope']:<7} n={r['admitted_with_both_dates']:>6}  "
              f"before {r['notif_before_adm']:>5}  same-day {r['same_day']:>6}  "
              f"after {r['notif_after_adm']:>6}  "
              f"({r['pct_notif_after_adm']:.1f}% after, median "
              f"{r['median_notif_minus_adm_days']:+.0f}d)")

    # ---- B ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B. THE TEST — severe-marker prevalence by notification/admission order")
    sev_tab, sev_sum = severe_prevalence_by_ordering(line)
    sev_tab.write_csv(OUT / "severe_prevalence_by_ordering.csv")
    report["B_severe_prevalence_by_ordering"] = sev_sum
    for scope in ["Brazil"] + [f"Q{k}" for k in range(1, 6)]:
        parts = []
        for o in ORDER:
            r = sev_tab.filter((pl.col("scope") == scope) & (pl.col("ordering") == o))
            if r.height:
                parts.append(f"{o.replace('_adm', ''):<14} "
                             f"{r['severe_share_pct'][0]:5.1f}% "
                             f"({r['severe_share_lo'][0]:.1f}-{r['severe_share_hi'][0]:.1f}, "
                             f"n={r['n'][0]})")
        print(f"  {scope:<7} " + "   ".join(parts))
    print(f"  pooled: after minus before = "
          f"{sev_sum['pooled_pp_after_minus_before']:+.1f} pp "
          f"(prevalence ratio "
          f"{sev_sum['pooled_prevalence_ratio_after_over_before']:.3f})")
    print("  by marker, after minus before (pp): " + "  ".join(
        f"{k} {v:+.1f}" for k, v in
        sev_sum["by_marker_pooled_pp_after_minus_before"].items()))

    lag = severe_prevalence_by_lag(line)
    lag.write_csv(OUT / "severe_prevalence_by_lag.csv")
    print("  dose-response in days of hospital course visible at notification:")
    for r in lag.iter_rows(named=True):
        print(f"    {r['lag_band']:<32} n={r['n']:>6}  severe "
              f"{r['severe_share_pct']:5.1f}% "
              f"({r['severe_share_lo']:.1f}-{r['severe_share_hi']:.1f})")

    report["B_sensitivity_to_date_gate"] = sensitivity_gate(line)
    print("  TH-C, ungated:")
    for k, v in report["B_sensitivity_to_date_gate"].items():
        print(f"    {k:<22} before {v['severe_pct_before']:5.1f}% (n={v['n_before']:>5})"
              f"   after {v['severe_pct_after']:5.1f}% (n={v['n_after']:>6})"
              f"   {v['pp_after_minus_before']:+.1f} pp")

    # ---- C ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("C. Does the non-severe case-fatality gradient survive the restriction?")
    cfr_tab, cfr_sum = nonsevere_cfr(line)
    cfr_tab.write_csv(OUT / "nonsevere_cfr_by_ordering_and_band.csv")
    report["C_nonsevere_case_fatality"] = cfr_sum
    for lab, v in cfr_sum.items():
        if "note" in v:
            print(f"  {lab:<48} {v['note']}")
            continue
        print(f"  {lab:<48} Q1 {v['Q1_pct']:5.2f}% "
              f"({v['Q1_lo']:.2f}-{v['Q1_hi']:.2f})  ->  Q5 {v['Q5_pct']:5.2f}% "
              f"({v['Q5_lo']:.2f}-{v['Q5_hi']:.2f})  "
              f"{v['gradient_Q5_over_Q1']:5.2f}x  "
              f"[{v['nonsevere_deaths_total']} deaths / "
              f"{v['nonsevere_known_total']} known outcomes]")

    # ---- D ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("D. THE STRONGER TEST — the form dated on or after the patient died")
    dd_tab, rec_tab, dd_sum = deaths_vs_death_date(line)
    dd_tab.write_csv(OUT / "deaths_by_notification_vs_death_date.csv")
    rec_tab.write_csv(OUT / "block_recording_by_notification_vs_death_date.csv")
    report["D_deaths_vs_death_date"] = dd_sum
    for r in dd_tab.filter(pl.col("scope") == "Brazil").iter_rows(named=True):
        print(f"  {r['group']:<30} deaths={r['deaths']:>5}  labelled non-severe "
              f"{r['nonsevere']:>4} = {r['nonsevere_share_pct']:5.1f}% "
              f"({r['nonsevere_share_lo']:.1f}-{r['nonsevere_share_hi']:.1f})")
    print(f"  risk ratio (on/after death vs before death) = "
          f"{dd_sum['risk_ratio_after_over_before']:.3f}")
    print("  by depth band, non-severe share among deaths notified on/after death:")
    for k in range(1, 6):
        print(f"    Q{k}  {dd_sum['by_quintile_nonsevere_share_pct_on_or_after_death'][f'Q{k}']:5.1f}%"
              f"  (n={dd_sum['by_quintile_deaths_on_or_after_death'][f'Q{k}']})")
    print(f"  clinical block recorded at all: "
          f"{dd_sum['block_recorded_pct_before_death']:.1f}% before death vs "
          f"{dd_sum['block_recorded_pct_on_or_after_death']:.1f}% on/after death")
    report["D_admission_to_death"] = admission_to_death(line)
    a2d = report["D_admission_to_death"]
    print(f"  admission-to-death days (p10/p25/p50/p75/p90): "
          f"{a2d['admission_to_death_days_p10_p25_p50_p75_p90']}, "
          f"{a2d['pct_dead_within_2_days_of_admission']}% dead within 2 days "
          f"(n={a2d['n']})")
    print(f"  median admission-to-death: non-severe label "
          f"{a2d['median_admission_to_death_nonsevere_label']:.0f}d "
          f"(n={a2d['n_nonsevere_label']}), severe label "
          f"{a2d['median_admission_to_death_severe_label']:.0f}d "
          f"(n={a2d['n_severe_label']})")

    # ---- E ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("E. Rival explanations the timing story was standing in for")
    lab_tab, lab_sum = deaths_labelled_nonsevere_by_band(line)
    lab_tab.write_csv(OUT / "deaths_labelled_nonsevere_by_band.csv")
    report["E_death_labelling_by_band"] = lab_sum
    print("  share of each band's deaths whose block carries none of the four markers:")
    for r in lab_tab.iter_rows(named=True):
        print(f"    Q{r['quintile']}  {r['nonsevere_label_pct']:5.1f}% "
              f"({r['nonsevere_label_lo']:.1f}-{r['nonsevere_label_hi']:.1f})  "
              f"{r['labelled_nonsevere']:>3}/{r['deaths_block_recorded']:>4} deaths")
    print(f"    Q1/Q5 ratio {lab_sum['Q1_over_Q5']:.2f} — the mislabelling of deaths "
          f"is WORST where the anomaly is weakest")

    ph_tab, (wide_tab, ph_sum) = phenotype_breadth(line)
    ph_tab.write_csv(OUT / "nonsevere_deaths_other_signs.csv")
    wide_tab.write_csv(OUT / "widened_phenotype_nonsevere_cfr.csv")
    report["E_phenotype_breadth"] = ph_sum
    print(f"  among the {ph_sum['n_nonsevere_with_known_outcome']:,} non-severe-labelled "
          f"cases with a known outcome ({ph_sum['n_nonsevere_deaths']} deaths), the "
          f"other eleven signs, died vs survived:")
    for r in ph_tab.iter_rows(named=True):
        print(f"    {r['label']:<8} died {r['prev_died_pct']:5.1f}% "
              f"({r['prev_died_lo']:.1f}-{r['prev_died_hi']:.1f})  "
              f"survived {r['prev_survived_pct']:5.1f}%  "
              f"{r['pp_diff']:+5.1f} pp   PR {r['prevalence_ratio']:.2f}")
    print(f"  widening the phenotype with respiratory + cardiac involvement: "
          f"non-severe case fatality "
          f"{ph_sum['widened_nonsevere_cfr_Q1']:.2f}% -> "
          f"{ph_sum['widened_nonsevere_cfr_Q5']:.2f}%, "
          f"{ph_sum['widened_nonsevere_gradient_Q5_over_Q1']:.2f}x "
          f"[{ph_sum['widened_nonsevere_deaths_total']} deaths / "
          f"{ph_sum['widened_nonsevere_known_total']} known outcomes]")

    # ---- F ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("F. What this does to the >=23% case-mix bound")
    cm_tab, cm_sum = casemix_bound(line)
    cm_tab.write_csv(OUT / "casemix_bound_under_restriction.csv")
    report["F_casemix_bound"] = cm_sum
    for lab in ["published (no restriction)",
                "R1: drop admitted cases notified before admission",
                "R2: every non-severe death reclassified severe",
                "R3: phenotype widened with respiratory + cardiac"]:
        v = cm_sum[lab]
        print(f"  {lab:<52} crude {v['crude_gradient_Q5_over_Q1']:.2f}x -> "
              f"standardised {v['standardised_gradient_Q5_over_Q1']:.2f}x  "
              f"= {100 * v['case_mix_share_of_log_gradient']:.0f}% of the log gradient "
              f"(severe share Q1 {100 * v['severe_share_Q1']:.1f}% -> "
              f"Q5 {100 * v['severe_share_Q5']:.1f}%)")
    print(f"  R1 drops {cm_sum['R1_cases_dropped']:,} cases "
          f"({cm_sum['R1_pct_of_series_dropped']}% of the banded series)")

    (OUT / "clinical_block_timing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
