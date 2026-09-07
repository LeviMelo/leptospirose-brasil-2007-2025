#!/usr/bin/env python
"""Bundle every result the manuscript quotes, with provenance.

Analyses accumulate outputs across a dozen directories, written on different
days by different scripts, some superseded. At submission time the question
"which file produced Table 2, and was it written before or after the model was
fixed?" has to be answerable, and a directory listing does not answer it.

This walks the declared result set, copies it into a versioned bundle, and
writes a manifest carrying, per file: a SHA-256, its size and row count, its
modification time, and the analysis stage that produced it. It also records the
git commit and the working-tree state at export, so a bundle produced from a
dirty tree says so instead of implying a clean provenance it does not have.

Files that are declared but absent are listed as MISSING rather than skipped:
a bundle that silently omits a table the manuscript cites is worse than one
that fails loudly.

Usage:
    python studies/leptospirosis/24_export_paper_data.py [--version v1]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results"

# Declared result set. The key is the analysis stage; the value is a list of
# (relative path under data/results, what the manuscript uses it for).
BUNDLE: dict[str, list[tuple[str, str]]] = {
    "00_descriptive": [
        ("01_descriptive/T1_study_flow.csv", "study flow: notifications to confirmed cases"),
        ("01_descriptive/T2a_incidence_by_region.csv", "incidence and rate ratios by macro-region"),
        ("01_descriptive/T2b_incidence_by_year.csv", "annual incidence series"),
        ("01_descriptive/T2c_cfr_by_region.csv", "case fatality by region, both denominators"),
        ("01_descriptive/T2d_hospitalisation_by_region.csv", "hospitalisation share by region"),
        ("01_descriptive/T3_trend.csv", "average annual percent change and Mann-Kendall"),
        ("01_descriptive/T4a_rate_by_age_sex.csv", "age- and sex-specific incidence"),
        ("01_descriptive/T4b_rate_by_sex.csv", "male-to-female rate ratio"),
        ("01_descriptive/T4c_asr_by_year.csv", "age-standardised incidence by year"),
        ("01_descriptive/T4d_asr_by_region.csv", "age-standardised incidence by region"),
        ("01_descriptive/T5_seasonality.csv", "circular seasonality: peak month, resultant length"),
        ("01_descriptive/T6a_concentration.csv", "burden concentration: Gini, top-k, Lorenz points"),
        ("01_descriptive/T6b_lorenz.csv", "Lorenz curve for the concentration figure"),
        ("01_descriptive/T7a_completeness_by_year.csv", "field completeness by year"),
        ("01_descriptive/T7b_completeness_overall.csv", "field completeness over the period"),
        ("01_descriptive/T7c_completeness_drift.csv", "completeness drift, the data-quality result"),
        ("01_descriptive/T8_clinical_exposure_profile.csv", "clinical signs and exposure antecedents"),
        ("01_descriptive/T9_confirmation_criterion.csv", "laboratory vs clinical-epidemiological"),
        ("01_descriptive/T10a_delay_overall.csv", "onset-to-notification delay"),
        ("01_descriptive/T10b_delay_by_year.csv", "delay by year"),
        ("01_descriptive/T10c_digitisation_lag_by_year.csv", "notification-to-digitisation lag"),
        ("01_descriptive/03_breaks.csv", "COVID trough and 2024 flood spike"),
        ("01_descriptive/T11a_serovar_profile.csv", "serovar profile among typed cases"),
        ("01_descriptive/T11b_reservoir_attribution.csv", "presumptive maintenance-host attribution"),
        ("01_descriptive/T11c_reservoir_by_region.csv", "reservoir attribution by macro-region"),
        ("01_descriptive/T11d_rattus_share_by_region.csv", "Rattus-associated share by macro-region"),
        ("01_descriptive/T11e_reservoir_by_year.csv", "reservoir attribution over time"),
        ("00_data_quality/lept_line_level_column_audit.csv", "column availability by dictionary vintage"),
    ],
    "rq1_exposure_response": [
        ("rq1_exposure_response/health_region_confirmatory/cumulative_exposure_response.csv",
         "primary result: cumulative RR across the rainfall anomaly range"),
        ("rq1_exposure_response/health_region_confirmatory/exposure_lag_surface.csv",
         "exposure-lag-response surface, the headline figure"),
        ("rq1_exposure_response/health_region_confirmatory/lag_response_high_exposure.csv",
         "lag curve at the high-exposure percentile"),
        ("rq1_exposure_response/health_region_confirmatory/structural_coefficients.csv",
         "structural block; also the incidence side of the RQ4 comparison"),
        ("rq1_exposure_response/health_region_confirmatory/hyperparameters.csv", "variance components"),
        ("rq1_exposure_response/health_region_confirmatory/fit_metadata.json",
         "execution profile, scores, and scientific status"),
        ("rq1_exposure_response/health_region_confirmatory_h2/", "H2: effect modification by sanitation"),
        ("rq1_exposure_response/health_region_sensitivity/rq1_sensitivity.csv",
         "prespecified robustness battery"),
        ("rq1_exposure_response/health_region_sensitivity/verdict.txt", "the quotable robustness verdict"),
        ("rq1_exposure_response/multiscale/rq1_scale_comparison.csv",
         "RQ1 across analytic scales: the modifiable areal unit result"),
        ("rq1_exposure_response/climate_splice/climate_splice_sensitivity.csv",
         "climate product-splice sensitivity, separated from the 2024 outbreak"),
        ("rq1_exposure_response/climate_splice/verdict.txt", "splice verdict"),
        ("rq1_exposure_response/enso/enso_phase_exposure_response.csv",
         "exposure-response by ENSO phase"),
        ("rq1_exposure_response/enso/enso_interaction_coefficients.csv",
         "ENSO interaction coefficients"),
    ],
    "rq4_lethality": [
        ("rq4_lethality/rq4_primary_odds.csv", "primary lethality odds ratios"),
        ("rq4_lethality/rq4_health_system_odds.csv", "with the health-system block"),
        ("rq4_lethality/rq4_unadjusted_odds.csv", "without the completeness adjustment"),
        ("rq4_lethality/rq4_completeness_adjustment_effect.csv",
         "what the completeness adjustment changes"),
        ("rq4_lethality/rq4_determinant_comparison.csv",
         "the RQ4 result: incidence vs lethality determinants"),
        ("rq4_lethality/rq4_spatial_effects.csv", "residual health-region lethality"),
        ("rq4_lethality/rq4_hyperparameters.csv", "variance components"),
        ("rq4_lethality/rq4_lethality_panel.csv", "the analysis panel, for reproduction"),
        ("rq4_lethality/rq4b_hospitalisation_odds.csv", "hospitalisation risk among confirmed cases"),
        ("rq4_lethality/rq4c_sih_inhospital_fatality_odds.csv",
         "in-hospital fatality from SIH billing records"),
        ("rq4_lethality/rq4d_lethality_system_agreement.csv",
         "SINAN vs SIH lethality agreement across 419 health regions"),
    ],
    "rq2_flood_disasters": [
        ("rq2_flood_disasters/corrected_primary_event_study.csv", "corrected flood event-time ATT"),
        ("rq2_flood_disasters/corrected_primary_overall.csv", "corrected aggregated ATT"),
        ("rq2_flood_disasters/corrected_primary_horizons.csv", "corrected dynamic ATT horizons"),
        ("rq2_flood_disasters/corrected_primary_group_time.csv", "full corrected group-time ATT cells"),
        ("rq2_flood_disasters/corrected_primary_shape.csv", "primary panel dimensions"),
        ("rq2_flood_disasters/corrected_primary_honest_did.csv", "HonestDiD robust bounds"),
        ("rq2_flood_disasters/corrected_primary_status.csv", "primary execution and design audit"),
        ("rq2_flood_disasters/corrected_primary_asinh_event_study.csv", "outcome-scale sensitivity"),
        ("rq2_flood_disasters/corrected_primary_asinh_overall.csv", "outcome-scale aggregated sensitivity"),
        ("rq2_flood_disasters/corrected_primary_asinh_group_time.csv", "outcome-scale group-time ATT cells"),
        ("rq2_flood_disasters/corrected_primary_asinh_horizons.csv", "outcome-scale dynamic horizons"),
        ("rq2_flood_disasters/corrected_primary_asinh_honest_did.csv", "outcome-scale HonestDiD bounds"),
        ("rq2_flood_disasters/corrected_primary_asinh_shape.csv", "outcome-scale panel dimensions"),
        ("rq2_flood_disasters/corrected_primary_asinh_status.csv", "outcome-scale execution status"),
        ("rq2_flood_disasters/corrected_primary_assignment_balance.csv", "baseline treated-control standardised differences"),
        ("rq2_flood_disasters/corrected_primary_assignment_overlap.csv", "baseline propensity-score overlap"),
        ("rq2_flood_disasters/corrected_drought_placebo_event_study.csv", "drought negative-control event study"),
        ("rq2_flood_disasters/corrected_drought_placebo_horizons.csv", "drought negative-control horizons"),
        ("rq2_flood_disasters/corrected_drought_placebo_group_time.csv", "drought group-time ATT cells"),
        ("rq2_flood_disasters/corrected_drought_placebo_honest_did.csv", "drought HonestDiD bounds"),
        ("rq2_flood_disasters/corrected_drought_placebo_overall.csv", "drought aggregated ATT"),
        ("rq2_flood_disasters/corrected_drought_placebo_shape.csv", "drought panel dimensions"),
        ("rq2_flood_disasters/corrected_drought_placebo_status.csv", "drought execution status"),
        ("rq2_flood_disasters/corrected_recurrence_clean_event_study.csv", "recurrence-cleaned bound"),
        ("rq2_flood_disasters/corrected_recurrence_clean_group_time.csv", "recurrence-clean group-time ATT cells"),
        ("rq2_flood_disasters/corrected_recurrence_clean_honest_did.csv", "recurrence-clean HonestDiD bounds"),
        ("rq2_flood_disasters/corrected_recurrence_clean_horizons.csv", "recurrence-clean dynamic horizons"),
        ("rq2_flood_disasters/corrected_recurrence_clean_overall.csv", "recurrence-clean aggregated ATT"),
        ("rq2_flood_disasters/corrected_recurrence_clean_shape.csv", "recurrence-clean panel dimensions"),
        ("rq2_flood_disasters/corrected_recurrence_clean_status.csv", "recurrence-clean execution status"),
        ("rq2_flood_disasters/corrected_nevertreated_event_study.csv", "never-treated controls"),
        ("rq2_flood_disasters/corrected_nevertreated_group_time.csv", "never-treated-control group-time ATT cells"),
        ("rq2_flood_disasters/corrected_nevertreated_honest_did.csv", "never-treated-control HonestDiD bounds"),
        ("rq2_flood_disasters/corrected_nevertreated_horizons.csv", "never-treated-control dynamic horizons"),
        ("rq2_flood_disasters/corrected_nevertreated_overall.csv", "never-treated-control aggregated ATT"),
        ("rq2_flood_disasters/corrected_nevertreated_shape.csv", "never-treated-control panel dimensions"),
        ("rq2_flood_disasters/corrected_nevertreated_status.csv", "never-treated-control execution status"),
        ("rq2_flood_disasters/corrected_flood_monthly_crosschecks.csv", "monthly Poisson flood cross-check with rainfall DLNM"),
        ("rq2_flood_disasters/corrected_flood_monthly_crosschecks_status.csv", "monthly flood execution status"),
        ("rq2_flood_disasters/corrected_drought_monthly_crosschecks.csv", "monthly Poisson drought negative control with rainfall DLNM"),
        ("rq2_flood_disasters/corrected_drought_monthly_crosschecks_status.csv", "monthly drought execution status"),
        ("rq2_flood_disasters/corrected_rq2_summary.csv", "canonical cross-specification estimand table"),
        ("rq2_flood_disasters/corrected_rq2_adjudication.json", "machine-readable causal adjudication"),
        ("rq2_flood_disasters/corrected_orchestration_status.json", "checkpoint orchestration status"),
        ("rq2_flood_disasters/recurrence.csv", "treatment recurrence structure"),
    ],
    "rq5_regimes": [
        ("rq5_regimes/regime_assignment.csv", "municipality to regime"),
        ("rq5_regimes/regime_summary.csv", "regime burden and case fatality"),
        ("rq5_regimes/regime_seasonality.csv", "regime seasonal profiles"),
        ("rq5_regimes/regime_stability.csv", "bootstrap adjusted Rand index"),
        ("rq5_regimes/regime_model_selection.csv", "BIC across K"),
    ],
    "rq3_ascertainment": [
        ("rq3_ascertainment/01_state_surveillance.csv", "state surveillance intensity"),
        ("rq3_ascertainment/02_cfr_by_incidence.csv",
         "the case-fatality gradient across incidence bands"),
        ("rq3_ascertainment/03_backcalculation.csv", "back-calculated burden"),
        ("rq3_ascertainment/03_detection_silent_municipalities.csv", "detection-silent municipalities corroborated by SIM"),
        ("rq3_ascertainment/07_triangulation_summary.json", "three-system triangulation summary"),
        ("rq3_ascertainment/10_cr_deaths_national.csv", "conservative death-list capture-recapture lower bound"),
        ("rq3_ascertainment/10b_cr_deaths_national_loose.csv", "looser overlap sensitivity"),
        ("rq3_ascertainment/11_cr_deaths_by_year.csv", "capture-recapture lower bound by year"),
        ("rq3_ascertainment/12_cr_deaths_by_state.csv", "capture-recapture lower bound by state"),
        ("rq3_ascertainment/13_detection_counterfactual.csv", "surveillance-effort counterfactual range"),
        ("rq3_ascertainment/14_detection_logistic_coefficients.csv", "detection probability model"),
        ("rq3_ascertainment/15_intensity_nb_coefficients.csv", "notification-intensity model"),
        ("rq3_ascertainment/16_surveillance_completeness_map.csv", "municipality surveillance classification"),
    ],
    # The assembled unit-level tables. Not a fifth analysis: every column is
    # carried from a stage above. They are bundled because a figure should be
    # made from one join, and because the CSVs alone lose the zero-padded
    # municipality key that makes that join work.
    "atlas": [
        ("atlas/municipality_atlas.parquet", "every municipality-level answer on one spine"),
        ("atlas/municipality_atlas.csv", "the same table, readable"),
        ("atlas/municipality_year_atlas.parquet", "municipality x year, with exact rates"),
        ("atlas/municipality_serogroup_long.parquet", "non-zero serogroup cells with denominators"),
        ("atlas/health_region_atlas.parquet", "RQ1 spatial field beside burden and severity"),
        ("atlas/health_region_atlas.csv", "the same table, readable"),
        ("atlas/atlas_manifest.json", "grain, key and shape of each atlas"),
    ],
    "data_quality": [
        ("00_data_quality/panel_sparsity.json", "zero structure of the case panel"),
        ("00_data_quality/panel_join_audit.csv", "panel assembly audit"),
        ("00_data_quality/lepto_climate_panel_quality.json", "climate coverage and imputation"),
        ("00_data_quality/lepto_sanitation_panel_quality.json", "sanitation interpolation"),
        ("00_data_quality/lepto_socioeconomic_panel_quality.json", "GDP and urban interpolation"),
        ("00_data_quality/municipality_mesh_2022.json", "geography vintage"),
        ("00_data_quality/rq2_did_feasibility.json", "RQ2 design feasibility"),
        ("00_data_quality/rq2_treatment_cohorts.csv", "flood declaration cohorts"),
    ],
}

# Documentation copied beside, not mixed into, the statistical tables. This
# makes the release self-describing while keeping the datasets neutral with
# respect to later table/plot/figure decisions.
SUPPORTING_DOCS: list[tuple[str, str]] = [
    ("docs/RESEARCH_JOURNAL.md", "scope, argument chain, verified findings, standing constraints"),
    ("docs/research/ANALYTIC_DATASETS.md", "analytic dataset methods and use contract"),
    ("docs/ISSUE_LEDGER.md", "open engineering defects"),
    ("docs/protocol/LEPTO_BR_PROTOCOL_V3.md", "historical protocol under which these results were generated"),
    ("docs/protocol/COVERAGE.md", "protocol-versus-delivered matrix"),
    ("studies/leptospirosis/ANALYTIC_DATASETS.yaml", "machine-readable dataset catalog"),
    ("studies/leptospirosis/CLAIM_LEDGER.yaml", "machine-readable scientific claim ledger"),
    ("studies/leptospirosis/RESULT_ASSERTIONS.yaml", "material numeric assertions"),
]


# What produced each stage. A result written before its producer's latest edit
# deserves review, but mtime alone cannot say whether that edit changed the
# estimand. Explicit supersession is maintained in the research registry.
# Keyed by the LONGEST MATCHING PATH PREFIX, not by stage. A stage-level map
# was tried first and was useless: an edit to the battery's spec file flagged
# every confirmatory output, and 16 of 60 files came back flagged, almost all
# falsely. A check that cries wolf gets ignored, which is worse than not having
# it -- so the mapping has to be as precise as the actual dependency.
PRODUCERS: dict[str, list[str]] = {
    "01_descriptive/": ["studies/leptospirosis/22_paper_descriptives.R",
               "R/11_tables.R", "R/01_descriptive.R"],
    "00_data_quality/lept_line_level_column_audit.csv": [
        "studies/leptospirosis/21_build_line_level.py"],
    # These two stages are written by the Python scripts alone. The R modules of
# the same name are consumed by later R drivers, not by these writers, and
# listing them here produced false review signals.
    "01_descriptive/03_breaks.csv": ["studies/leptospirosis/04_descriptive.py"],
    "rq1_exposure_response/health_region_confirmatory/": [
        "studies/leptospirosis/11_fit_rq1.R", "R/02_crossbasis.R",
        "R/03_inla_spacetime.R"],
    "rq1_exposure_response/health_region_confirmatory_h2/": [
        "studies/leptospirosis/12_fit_rq1_modifier.R", "R/02_crossbasis.R",
        "R/03_inla_spacetime.R"],
    "rq1_exposure_response/health_region_sensitivity/": [
        "studies/leptospirosis/13_rq1_sensitivity.R",
        "studies/leptospirosis/rq1_spec.R", "R/08_sensitivity.R",
        "R/09_exec.R"],
    "rq4_lethality/": ["studies/leptospirosis/23_rq4_lethality.R", "R/12_severity.R"],
    "rq4_lethality/rq4b": ["studies/leptospirosis/30_rq4_hospitalisation.R", "R/12_severity.R"],
    "rq4_lethality/rq4c": ["studies/leptospirosis/30_rq4_hospitalisation.R", "R/12_severity.R"],
    "rq4_lethality/rq4d": ["studies/leptospirosis/30_rq4_hospitalisation.R", "R/12_severity.R"],
    "rq1_exposure_response/multiscale/": [
        "studies/leptospirosis/28_rq1_multiscale.R",
        "studies/leptospirosis/rq1_scale_spec.R"],
    "rq1_exposure_response/climate_splice/": [
        "studies/leptospirosis/29_rq1_climate_splice.R",
        "studies/leptospirosis/rq1_spec.R"],
    "rq1_exposure_response/enso/": ["studies/leptospirosis/27_rq1_enso.R"],
    "rq2_flood_disasters/corrected_flood_monthly": [
                             "studies/leptospirosis/15c_rq2_crosschecks.R",
                             "R/05_did_flood.R"],
    "rq2_flood_disasters/corrected_drought_monthly": [
                             "studies/leptospirosis/15c_rq2_crosschecks.R",
                             "R/05_did_flood.R"],
    "rq2_flood_disasters/corrected_rq2_": [
                             "studies/leptospirosis/25_rq2_adjudication.py"],
    "rq2_flood_disasters/corrected_primary_assignment": [
                             "studies/leptospirosis/15d_rq2_assignment_diagnostics.R",
                             "studies/leptospirosis/rq2_spec.R",
                             "R/05_did_flood.R"],
    "rq2_flood_disasters/corrected_": ["studies/leptospirosis/15b_rq2_one.R",
                             "studies/leptospirosis/rq2_spec.R",
                             "R/05_did_flood.R"],
    "rq2_flood_disasters/recurrence.csv": ["studies/leptospirosis/15_rq2_did.R",
                             "studies/leptospirosis/rq2_spec.R",
                             "R/05_did_flood.R"],
    "rq5_regimes/": ["studies/leptospirosis/18_rq5_regimes.R", "R/10_regimes.R"],
    "rq3_ascertainment/": ["studies/leptospirosis/06_ascertainment.py"],
}


def producer_mtime(rel: str) -> datetime | None:
    """Filesystem mtime of a producer file, or None if it is not there.

    Deliberately NOT the git commit time. Results are produced and only then
    committed, so a commit timestamp is always later than the run it records
    and would mark every fresh result as superseded. The mtime answers the
    question actually being asked: has the code been touched since the result
    was written?

    This is only a review signal. A fresh clone stamps every producer with the
    checkout time, and a non-semantic edit can postdate a valid result.
    """
    f = ROOT / rel
    if not f.is_file():
        f = ROOT.parent / rel
    if not f.is_file():
        return None
    return datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rowcount(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except (UnicodeDecodeError, OSError):
        return None


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    dest = ROOT / "data" / "export" / f"paper_{args.version}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    rows: list[dict] = []
    missing: list[str] = []
    for stage, entries in BUNDLE.items():
        for rel, purpose in entries:
            src = RESULTS / rel
            targets: list[tuple[Path, str]] = []
            if rel.endswith("/"):
                if not src.is_dir():
                    missing.append(rel)
                    continue
                for f in sorted(src.rglob("*")):
                    if f.is_file():
                        targets.append((f, f"{rel}{f.relative_to(src).as_posix()}"))
                if not targets:
                    missing.append(rel)
                    continue
            elif src.is_file():
                targets.append((src, rel))
            else:
                missing.append(rel)
                continue

            for path, relname in targets:
                out = dest / stage / Path(relname).name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
                rows.append({
                    "stage": stage,
                    "file": f"{stage}/{out.name}",
                    "source": f"data/results/{relname}",
                    "purpose": purpose,
                    "bytes": path.stat().st_size,
                    "rows": rowcount(path),
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    "sha256": sha256(path),
                })

    for rel, purpose in SUPPORTING_DOCS:
        src = ROOT / rel
        if not src.is_file():
            missing.append(rel)
            continue
        out = dest / "documentation" / src.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        rows.append({
            "stage": "documentation", "file": f"documentation/{out.name}",
            "source": rel, "purpose": purpose, "bytes": src.stat().st_size,
            "rows": rowcount(src),
            "modified_utc": datetime.fromtimestamp(
                src.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256(src),
        })

    # Review signals. Filesystem mtime is not scientific provenance: a producer
    # can be touched by a comment/path/serialization edit after a valid fit, and
    # a stale result can be copied after a model change. Therefore these fields
    # trigger review but never assert supersession. Explicit supersession is
    # controlled by docs/RESEARCH_JOURNAL.md and by declaring only the current
    # files in BUNDLE.
    newest = max(r["modified_utc"] for r in rows)
    newest_dt = datetime.fromisoformat(newest)
    # A national study is normally assembled over several days. Twelve hours
    # flagged almost the entire valid bundle and created noise; one week still
    # catches accidentally mixed release vintages without treating ordinary
    # staged execution as suspicious.
    stale_hours = 24.0 * 7

    # Most recent edit among the producers of each declared prefix.
    prefix_time: dict[str, tuple[datetime | None, str]] = {}
    for prefix, prods in PRODUCERS.items():
        best: datetime | None = None
        which = ""
        for p in prods:
            t = producer_mtime(p)
            if t and (best is None or t > best):
                best, which = t, p
        prefix_time[prefix] = (best, which)

    def producer_for(source: str) -> tuple[datetime | None, str]:
        rel = source[len("data/results/"):] if source.startswith("data/results/") else source
        match = ""
        for prefix in PRODUCERS:
            if rel.startswith(prefix) and len(prefix) > len(match):
                match = prefix
        return prefix_time.get(match, (None, ""))

    for r in rows:
        mt = datetime.fromisoformat(r["modified_utc"])
        age_h = (newest_dt - mt).total_seconds() / 3600
        r["hours_older_than_newest"] = round(age_h, 1)
        ptime, pwhich = producer_for(r["source"])
        code_newer = ptime is not None and mt < ptime
        r["producer_last_changed"] = ptime.isoformat(timespec="seconds") if ptime else ""
        r["producer"] = pwhich
        r["review_flag"] = ("CODE_NEWER_REVIEW" if code_newer
                            else "AGE_REVIEW" if age_h > stale_hours else "")
    review = [r for r in rows if r["review_flag"]]

    # Write only after review fields have been attached. The earlier exporter
    # wrote MANIFEST.csv before computing them, silently discarding its own QA.
    manifest = dest / "MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    dirty = git("status", "--porcelain")
    prov = {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": args.version,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_tree_clean": not dirty,
        # Split on whitespace rather than slicing a fixed prefix: porcelain
        # status codes are two characters plus a space for most states but not
        # all, and a fixed slice silently ate the first character of the path.
        "git_dirty_files": [ln.split(maxsplit=1)[1] for ln in dirty.splitlines()
                            if len(ln.split(maxsplit=1)) > 1] if dirty else [],
        "files_exported": len(rows),
        "files_declared_but_missing": missing,
        "newest_result_utc": newest,
        "staleness_threshold_hours": stale_hours,
        "files_flagged_for_review": [r["source"] for r in review],
        "python": sys.version.split()[0],
    }
    (dest / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print(f"exported {len(rows)} files to {dest}")
    print(f"  commit {prov['git_commit'][:8]} on {prov['git_branch']}"
          f"  tree {'clean' if prov['git_tree_clean'] else 'DIRTY'}")
    by_stage: dict[str, int] = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    for stage, n in sorted(by_stage.items()):
        print(f"  {stage:<24} {n:>3} files")
    code_review = [r for r in review if r["review_flag"] == "CODE_NEWER_REVIEW"]
    if code_review:
        print(f"\nCODE NEWER THAN RESULT -- REVIEW, NOT PROOF OF SUPERSESSION "
              f"({len(code_review)}):")
        for r in sorted(code_review, key=lambda x: x["source"]):
            print(f"  {r['source']}")
            print(f"      written {r['modified_utc']}  but "
                  f"{r['producer']} was last edited {r['producer_last_changed']}")
        print("\nModification time alone cannot identify a scientific change; "
              "consult the result registry, claim assertions, and code diff.")
    age_review = [r for r in review if r["review_flag"] == "AGE_REVIEW"]
    if age_review:
        print(f"\nOLDER THAN THE REST OF THE BUNDLE ({len(age_review)}), "
              f"newest is {newest}:")
        for r in sorted(age_review, key=lambda x: -x["hours_older_than_newest"]):
            print(f"  {r['hours_older_than_newest']:>7.1f}h  {r['source']}")
    if missing:
        print(f"\nDECLARED BUT MISSING ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        print("\nThese are cited by the manuscript plan and are not on disk. "
              "Either the stage has not been run or the path has changed.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
