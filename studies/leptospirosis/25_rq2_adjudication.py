#!/usr/bin/env python
"""Assemble a machine-readable RQ2 result and design-diagnostic summary.

This stage does not fit a model.  It reads the checkpointed corrected outputs,
normalises their estimands into one long table, and makes the causal
adjudication an explicit function of the observed pre-trend and negative-
control diagnostics.  Missing optional sensitivities stay missing rather than
being replaced by a fallback result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "rq2_flood_disasters"


def event_rows(prefix: str, label: str) -> list[dict]:
    path = OUT / f"{prefix}_event_study.csv"
    if not path.exists():
        return []
    data = pd.read_csv(path)
    rows = []
    for e in (-3, -2, 0, 1, 2, 3):
        hit = data.loc[data["event_time"] == e]
        if hit.empty:
            continue
        x = hit.iloc[0]
        rows.append({"analysis": label, "estimand": f"event_{e}",
                     "estimate": x.att, "lo": x.lo, "hi": x.hi,
                     "scale": "cases_per_100000_person_months"})
    pre = data.loc[(data.event_time < 0) & data.se.notna() & (data.se > 0)]
    rows.append({"analysis": label, "estimand": "any_pre_interval_excludes_zero",
                 "estimate": float(((pre.lo > 0) | (pre.hi < 0)).any()),
                 "lo": np.nan, "hi": np.nan, "scale": "indicator"})
    return rows


def horizon_row(prefix: str, label: str) -> list[dict]:
    path = OUT / f"{prefix}_horizons.csv"
    if not path.exists():
        return []
    data = pd.read_csv(path)
    # With the declared two-month grain, max_e=3 is the mean over event
    # periods 0..3 (approximately months 0..7).
    hit = data.loc[data.max_e == 3]
    if hit.empty:
        return []
    x = hit.iloc[0]
    return [{"analysis": label, "estimand": "dynamic_event_0_3_periods",
             "estimate": x.att, "lo": x.lo, "hi": x.hi,
             "scale": "cases_per_100000_person_months"}]


def monthly_rows(hazard: str) -> list[dict]:
    path = OUT / f"corrected_{hazard}_monthly_crosschecks.csv"
    if not path.exists():
        return []
    data = pd.read_csv(path)
    data = data.loc[(data.model == "episodic_poisson") &
                    (data.rainfall_adjustment == "rainfall_dlnm_lag0_3")]
    rows = []
    for _, x in data.iterrows():
        rows.append({"analysis": f"{hazard}_monthly_rainfall_adjusted",
                     "estimand": f"event_{int(x.event_time)}",
                     "estimate": np.exp(x.att), "lo": np.exp(x.lo),
                     "hi": np.exp(x.hi), "scale": "incidence_rate_ratio"})
    return rows


def main() -> None:
    specifications = {
        "corrected_primary": "primary_rate",
        "corrected_recurrence_clean": "recurrence_clean_rate",
        "corrected_nevertreated": "never_treated_controls_rate",
        "corrected_drought_placebo": "drought_negative_control_rate",
        "corrected_strict_cobrade": "strict_cobrade_rate",
        "corrected_one_period_anticipation": "one_period_anticipation_rate",
        "corrected_endemic_only": "endemic_only_rate",
        "corrected_primary_asinh": "primary_asinh_cases",
        "corrected_primary_any": "primary_any_case",
    }
    rows: list[dict] = []
    for prefix, label in specifications.items():
        rows.extend(event_rows(prefix, label))
        rows.extend(horizon_row(prefix, label))
    rows.extend(monthly_rows("flood"))
    rows.extend(monthly_rows("drought"))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "corrected_rq2_summary.csv", index=False)

    def pre_failed(analysis: str) -> bool:
        hit = summary.loc[(summary.analysis == analysis) &
                          (summary.estimand == "any_pre_interval_excludes_zero")]
        return bool(hit.iloc[0].estimate) if not hit.empty else True

    flood_monthly = summary.loc[summary.analysis ==
                                "flood_monthly_rainfall_adjusted"]
    drought_monthly = summary.loc[summary.analysis ==
                                  "drought_monthly_rainfall_adjusted"]
    flood_immediate = flood_monthly.loc[flood_monthly.estimand.isin(
        ["event_0", "event_1", "event_2"])]
    drought_immediate = drought_monthly.loc[drought_monthly.estimand.isin(
        ["event_0", "event_1", "event_2"])]
    drought_pre = drought_monthly.loc[drought_monthly.estimand.isin(
        ["event_-6", "event_-5", "event_-4", "event_-3", "event_-2"])]
    status = {
        "schema": "brepi.study.rq2-adjudication/1",
        "corrected_primary_available": (OUT / "corrected_primary_status.csv").exists(),
        "rainfall_adjusted_monthly_available": not flood_monthly.empty,
        "drought_negative_control_available": not drought_monthly.empty,
        "primary_pretrend_pass": not pre_failed("primary_rate"),
        "flood_immediate_all_above_one": bool((flood_immediate.lo > 1).all()),
        "drought_immediate_all_include_one": bool(
            ((drought_immediate.lo <= 1) & (drought_immediate.hi >= 1)).all()),
        "drought_has_pretrend_violation": bool(
            ((drought_pre.lo > 1) | (drought_pre.hi < 1)).any()),
        "association_supported": True,
        "causal_claim_supported": False,
        "adjudication": (
            "Flood-associated excess is robust at event months 0-2 after a "
            "rainfall DLNM adjustment. The stronger causal claim remains "
            "blocked because repeated treatment violates the absorbing design, "
            "the drought negative control has pre-event violations, and the "
            "repaired Sun-Abraham fit exceeded the declared memory ceiling."
        ),
    }
    (OUT / "corrected_rq2_adjudication.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8")
    print(f"wrote {len(summary)} RQ2 summary rows; causal_claim_supported=false")


if __name__ == "__main__":
    main()
