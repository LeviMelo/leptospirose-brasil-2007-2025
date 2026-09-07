"""The shared-denominator problem, and the test that discriminates.

Reported case fatality and the hospitalisation proportion are

    CFR = D / N_known        H = Hosp / N_hosp-known

Both are ratios over the detected case count and both numerators rise with
severity, so a positive association between them is partly expected by
construction. A reviewer is right to say the raw ecological correlation cannot
by itself establish the ascertainment interpretation. This script separates
what is expected by construction from what is not.

**The two competing accounts.** Write the detected cases in a territory as
severe cases S plus detected mild cases m·M, where m is the fraction of mild
illness surveillance reaches. Deaths come overwhelmingly from the severe end.

*Account A — ascertainment.* Territories differ in m. Then
    H = S / (S + mM)  and  CFR = f_S · S / (S + mM) = f_S · H,
where f_S is the case fatality among severe/hospitalised cases. This account
makes a sharp, falsifiable prediction: **CFR is proportional to H, with the
constant of proportionality equal to the case fatality within the hospitalised
stratum, and that within-stratum case fatality does not itself vary with H.**

*Account B — real severity.* Territories differ in f_S, or in the true severity
mix, because of serovar, comorbidity, delay, or quality of care. Then case
fatality *within* the hospitalised stratum rises with H as well.

The two accounts are not mutually exclusive, and the data can apportion them,
because the hospitalised stratum is not diluted by the mild cases whose
detection varies. Under A alone the within-hospitalised gradient is flat; under
B alone it is as steep as the overall one.

Outputs to ``data/results/coupling/``.
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

OUT = PATHS.results / "coupling"
PANEL = PATHS.results / "analysis_panel"

#: Regions need enough outcome-known hospitalised cases for a within-stratum
#: case fatality to mean anything. Stated, not buried.
MIN_HOSP_KNOWN = 30


def _decile_by_H(df: pl.DataFrame, n: int = 10) -> pl.DataFrame:
    d = df.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    )
    return d.with_columns(
        (pl.col("_cw") * n).ceil().clip(1, n).cast(pl.Int32).alias("bin")
    ).drop("_cw")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pl.read_parquet(PANEL / "region_totals.parquet")
    reg = reg.filter(pl.col("hosp_known") > 0, pl.col("outcome_known") > 0)

    # ---------------------------------------------------------------- test 1 --
    # Overall vs within-hospitalised case fatality across the exposure.
    elig = reg.filter(pl.col("hk_hosp") >= MIN_HOSP_KNOWN)
    rows = []
    for b, g in _decile_by_H(elig, 5).group_by("bin", maintain_order=True):
        b = b[0]
        o, o_lo, o_hi = binom_ci(int(g["deaths"].sum()), int(g["outcome_known"].sum()))
        h, h_lo, h_hi = binom_ci(int(g["d_hosp"].sum()), int(g["hk_hosp"].sum()))
        nh, nh_lo, nh_hi = binom_ci(int(g["d_nonhosp"].sum()), int(g["hk_nonhosp"].sum()))
        rows.append({
            "bin": int(b), "regions": g.height, "cases": int(g["cases"].sum()),
            "H_median": float(g["H"].median()),
            "H_min": float(g["H"].min()), "H_max": float(g["H"].max()),
            "cfr_overall": float(o), "cfr_overall_lo": float(o_lo), "cfr_overall_hi": float(o_hi),
            "cfr_hosp": float(h), "cfr_hosp_lo": float(h_lo), "cfr_hosp_hi": float(h_hi),
            "cfr_nonhosp": float(nh), "cfr_nonhosp_lo": float(nh_lo), "cfr_nonhosp_hi": float(nh_hi),
        })
    strata = pl.DataFrame(rows)
    strata.write_csv(OUT / "cfr_overall_vs_within_hospitalised.csv")

    grad_overall = rows[-1]["cfr_overall"] / rows[0]["cfr_overall"]
    grad_hosp = rows[-1]["cfr_hosp"] / rows[0]["cfr_hosp"]

    # Rank correlations on the region scale, which does not depend on binning.
    rho_overall = stats.spearmanr(elig["H"], elig["cfr"])
    rho_hosp = stats.spearmanr(elig["H"], elig["cfr_hosp"])

    # ---------------------------------------------------------------- test 2 --
    # Account A predicts CFR = f_S * H exactly. Fit the proportionality and
    # compare the implied f_S against the directly measured hospitalised case
    # fatality; agreement is the quantitative form of the prediction.
    w = elig["outcome_known"].to_numpy().astype(float)
    x = elig["H"].to_numpy()
    y = elig["cfr"].to_numpy()
    f_implied = float(np.sum(w * x * y) / np.sum(w * x * x))   # weighted least squares through 0
    f_measured = float(elig["d_hosp"].sum() / elig["hk_hosp"].sum())

    # How much of the observed overall gradient does pure re-weighting explain?
    # Predict each region's CFR from its own within-stratum fatalities held at
    # the NATIONAL values, letting only the hospitalised/non-hospitalised mix
    # vary. The spread of that prediction is the mechanically expected part.
    f_h = f_measured
    f_n = float(elig["d_nonhosp"].sum() / elig["hk_nonhosp"].sum())
    pred = f_h * x + f_n * (1 - x)
    pb = _decile_by_H(elig.with_columns(pl.Series("pred", pred)), 5)
    pred_by_bin = pb.group_by("bin").agg(
        (pl.col("pred") * pl.col("outcome_known")).sum() / pl.col("outcome_known").sum()
    ).sort("bin")
    pred_grad = float(pred_by_bin[-1, 1] / pred_by_bin[0, 1])

    report = {
        "min_hosp_known": MIN_HOSP_KNOWN,
        "regions_eligible": elig.height,
        "regions_total": reg.height,
        "cases_eligible": int(elig["cases"].sum()),
        "case_share_eligible": float(elig["cases"].sum() / reg["cases"].sum()),
        "gradient_cfr_overall_Q5_over_Q1": grad_overall,
        "gradient_cfr_within_hospitalised_Q5_over_Q1": grad_hosp,
        "gradient_predicted_by_mix_alone": pred_grad,
        "spearman_H_vs_cfr_overall": float(rho_overall.statistic),
        "spearman_H_vs_cfr_overall_p": float(rho_overall.pvalue),
        "spearman_H_vs_cfr_within_hospitalised": float(rho_hosp.statistic),
        "spearman_H_vs_cfr_within_hospitalised_p": float(rho_hosp.pvalue),
        "f_hospitalised_implied_by_proportionality": f_implied,
        "f_hospitalised_measured": f_measured,
        "f_nonhospitalised_measured": f_n,
    }
    (OUT / "coupling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(strata.select("bin", "regions", "cases", "H_median", "cfr_overall",
                        "cfr_hosp", "cfr_nonhosp").to_pandas().to_string(index=False))
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
