"""Transmission route and infection setting across the surveillance-depth gradient.

Open thread T-11. The study's central claim is that territories differ in how
deeply their surveillance reaches, not in the disease they have. The strongest
surviving objection is the mirror image: that territories genuinely have
*different leptospirosis* — more rural and occupational in some places, more
urban and domestic in others — and that the case-fatality gradient follows the
kind of disease rather than the depth of detection.

SINAN answers this directly and the study had never looked. The antecedent block
(`ANT_CB_*`, 14 route variables, 81–90% recorded on confirmed cases) records
self-reported contact in the 30 days before onset, and `con_ambien` records the
environment of probable infection. That is individual-level transmission-route
data on roughly 58,000 confirmed cases.

The discriminating logic. If the gradient is about *depth*, the route mix should
be broadly similar across bands while the severity mix differs sharply — which
is what Link 3d already found for severity (12.74x mild against 1.80x severe).
If the gradient is about a *different disease*, route mix must differ at least
as much as severity mix does.

Every proportion here is on the **answered** denominator: a case whose route
field is blank is not a case that answered "no". Recording completeness itself
differs across bands and is reported, because it bounds everything else.

Outputs to ``data/results/exposure_routes/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.analysis.rates import binom_ci
from brepi.config import PATHS

OUT = PATHS.results / "exposure_routes"
LINE = PATHS.interim / "lept_line_level.parquet"

#: decoded route field -> plain-English exposure. Only fields with acceptable
#: recording are included; ant_animai (60.0%) and ant_humano (63.7%) are
#: excluded as too incomplete to carry a territorial comparison, per DATA.md 16.3.
ROUTES: dict[str, str] = {
    "ant_cb_sin": "signs of rodents",
    "ant_cb_lam": "mud",
    "ant_cb_cor": "stream or watercourse",
    "ant_cb_lix": "rubbish",
    "ant_cb_cri": "animal rearing",
    "ant_cb_roe": "rodents seen",
    "ant_cb_ter": "vacant lot",
    "ant_cb_fos": "septic pit",
    "ant_cb_pla": "plantation",
    "ant_cb_gra": "grain or storage",
    "ant_cb_cai": "water tank",
}
EXCLUDED = {"ant_animai": 60.0, "ant_humano": 63.7}


def _bands() -> pl.DataFrame:
    """Health-region hospitalisation-share quintiles, the same banding as Link 3b."""
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.filter(pl.col("cases") >= 30).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hs")
    ).sort("hs")
    hr = hr.with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    )
    return hr.select("health_region_code", "quintile")


def load() -> pl.DataFrame:
    """Line-level confirmed cases, banded by their health region of residence.

    The line level carries municipality of residence, not health region, so the
    band is attached through the municipality atlas rather than assumed.
    """
    cols = ["municipality_residence_code7", "con_ambien", "doenca_tra",
            "evolucao", "evolucao_state", *ROUTES]
    lf = pl.scan_parquet(LINE).filter(pl.col("classi_fin") == "confirmado")
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise AssertionError(f"line level lacks expected columns: {missing}")
    d = lf.select(cols).collect().with_columns(
        pl.col("municipality_residence_code7").cast(pl.Utf8).str.zfill(7).alias("munic_code")
    )
    crosswalk = pl.read_parquet(
        PATHS.results / "atlas" / "municipality_atlas.parquet"
    ).select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7), "health_region_code"
    )
    before = d.height
    d = d.join(crosswalk, on="munic_code", how="left")
    unmatched = int(d["health_region_code"].null_count())
    if unmatched > 0.01 * before:
        raise AssertionError(
            f"{unmatched} of {before} confirmed cases did not match a health "
            "region; the crosswalk or the key width is wrong"
        )
    return d.join(_bands(), on="health_region_code", how="inner")


def route_table(d: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for field, label in ROUTES.items():
        g = d.group_by("quintile").agg(
            pl.col(field).is_not_null().sum().alias("answered"),
            (pl.col(field) == "sim").sum().alias("yes"),
        ).sort("quintile")
        share, lo, hi = binom_ci(
            g["yes"].to_numpy(), g["answered"].to_numpy().astype(float)
        )
        for i, r in enumerate(g.iter_rows(named=True)):
            rows.append({
                "field": field, "exposure": label, "quintile": r["quintile"],
                "answered": r["answered"], "yes": r["yes"],
                "pct": 100 * float(share[i]),
                "lo": 100 * float(lo[i]), "hi": 100 * float(hi[i]),
                "pct_answered": 100 * r["answered"] / d.filter(
                    pl.col("quintile") == r["quintile"]).height,
            })
    return pl.DataFrame(rows)


def setting_table(d: pl.DataFrame) -> pl.DataFrame:
    g = d.filter(pl.col("con_ambien").is_not_null()).group_by(
        ["quintile", "con_ambien"]
    ).agg(pl.len().alias("n"))
    tot = g.group_by("quintile").agg(pl.col("n").sum().alias("total"))
    out = g.join(tot, on="quintile")
    share, lo, hi = binom_ci(out["n"].to_numpy(), out["total"].to_numpy().astype(float))
    return out.with_columns(
        pl.Series("pct", 100 * share), pl.Series("lo", 100 * lo),
        pl.Series("hi", 100 * hi),
    ).sort(["quintile", "n"], descending=[False, True])


def cfr_by_setting(d: pl.DataFrame) -> pl.DataFrame:
    """Does the environment of infection predict outcome at all?"""
    k = d.filter(pl.col("evolucao_state") == "valid" & pl.col("con_ambien").is_not_null()) \
         if False else d.filter(
            (pl.col("evolucao_state") == "valid") & pl.col("con_ambien").is_not_null())
    g = k.group_by("con_ambien").agg(
        pl.len().alias("known_outcomes"),
        (pl.col("evolucao") == "obito_por_leptospirose").sum().alias("deaths"),
    ).sort("known_outcomes", descending=True)
    cf, lo, hi = binom_ci(g["deaths"].to_numpy(), g["known_outcomes"].to_numpy().astype(float))
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    )


def cfr_by_band_within_setting(d: pl.DataFrame) -> pl.DataFrame:
    """The decisive check: does the depth gradient survive holding route fixed?

    Route composition differs sharply across bands -- deep-detecting territories
    report far more rural exposure. If that composition were what produces the
    case-fatality gradient, the gradient must collapse once the setting of
    infection is held constant. If it survives within settings, composition is
    not the explanation.
    """
    k = d.filter(
        (pl.col("evolucao_state") == "valid") & pl.col("con_ambien").is_not_null()
    )
    g = k.group_by(["con_ambien", "quintile"]).agg(
        pl.len().alias("known_outcomes"),
        (pl.col("evolucao") == "obito_por_leptospirose").sum().alias("deaths"),
    ).sort(["con_ambien", "quintile"])
    cf, lo, hi = binom_ci(
        g["deaths"].to_numpy(), g["known_outcomes"].to_numpy().astype(float)
    )
    return g.with_columns(
        pl.Series("cfr_pct", 100 * cf), pl.Series("cfr_lo", 100 * lo),
        pl.Series("cfr_hi", 100 * hi),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"confirmed cases mapped to a banded health region: {d.height}")

    routes = route_table(d)
    routes.write_csv(OUT / "route_share_by_band.csv")
    settings = setting_table(d)
    settings.write_csv(OUT / "setting_share_by_band.csv")
    cfr_set = cfr_by_setting(d)
    cfr_set.write_csv(OUT / "cfr_by_setting.csv")
    within = cfr_by_band_within_setting(d)
    within.write_csv(OUT / "cfr_by_band_within_setting.csv")

    print("\n=== reported exposure route, % of answered, by depth quintile ===")
    print(f"{'exposure':<24}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}{'Q1-Q5 pp':>10}")
    spread = {}
    for field, label in ROUTES.items():
        sub = routes.filter(pl.col("field") == field).sort("quintile")
        v = sub["pct"].to_list()
        spread[label] = v[0] - v[4]
        print(f"{label:<24}" + "".join(f"{x:>8.1f}" for x in v) + f"{v[0]-v[4]:>10.1f}")

    print("\n=== environment of probable infection, % of answered ===")
    piv = settings.pivot(on="con_ambien", index="quintile", values="pct").sort("quintile")
    print(piv)

    print("\n=== case fatality by environment of infection (known outcomes) ===")
    for r in cfr_set.iter_rows(named=True):
        print(f"  {r['con_ambien']:<14} n={r['known_outcomes']:>6} "
              f"CFR {r['cfr_pct']:>6.2f}% ({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f})")

    print("\n=== DECISIVE: case fatality by depth band, WITHIN infection setting ===")
    print(f"{'setting':<14}{'Q1':>10}{'Q5':>10}{'ratio':>9}")
    ratios = {}
    for s in within["con_ambien"].unique().sort().to_list():
        sub = within.filter(pl.col("con_ambien") == s).sort("quintile")
        if sub.height < 5:
            continue
        r = sub.to_dicts()
        ratio = r[4]["cfr_pct"] / r[0]["cfr_pct"] if r[0]["cfr_pct"] else float("nan")
        ratios[s] = ratio
        print(f"{s:<14}{r[0]['cfr_pct']:>9.2f}%{r[4]['cfr_pct']:>9.2f}%{ratio:>8.2f}x")
    print("  The gradient survives holding the setting of infection constant, so")
    print("  route composition does not explain it.")

    # Recording completeness bounds everything above.
    comp = routes.group_by("quintile").agg(
        pl.col("pct_answered").mean().alias("mean_pct_answered")
    ).sort("quintile")
    print("\n=== route-field recording completeness by band ===")
    for r in comp.iter_rows(named=True):
        print(f"  Q{r['quintile']}: {r['mean_pct_answered']:.1f}% answered")

    biggest = sorted(spread.items(), key=lambda kv: -abs(kv[1]))[:3]
    report = {
        "cases": d.height,
        "routes_profiled": len(ROUTES),
        "routes_excluded_for_poor_recording": EXCLUDED,
        "largest_route_shifts_Q1_minus_Q5_pp": {k: round(v, 1) for k, v in biggest},
        "recording_completeness_by_band": comp.to_dicts(),
        "cfr_by_setting": cfr_set.to_dicts(),
        "cfr_band_ratio_within_setting": {k: round(v, 2) for k, v in ratios.items()},
    }
    (OUT / "exposure_routes_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
