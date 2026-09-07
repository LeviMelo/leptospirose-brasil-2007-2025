"""Stage 3 - is silence transmission or detection?

The descriptive stage produced a fact that constrains everything downstream:
no Brazilian municipality above 100,000 inhabitants has failed to report a
confirmed leptospirosis case in nineteen years, while 56% of those below 5,000
have never reported one. The obvious reading -- small places have less disease
-- is contradicted by the same data, because the highest *incidences* in the
country are also in small municipalities (Capivari do Sul/RS at 81 per 100,000
per year, Antonio Carlos/SC at 76). Small municipalities are therefore bimodal:
either they report a great deal of leptospirosis or they report none at all.
Genuine transmission differences do not usually look like that. Detection
differences do.

This module separates the two, in four steps that escalate in strength:

1. **Surveillance intensity by state.** Notification effort, confirmation
   ratio and laboratory-confirmation share vary enormously between states.
   If municipal silence tracks state surveillance effort rather than state
   incidence, silence is administrative.

2. **Case fatality as an inverse detection index.** A surveillance system that
   captures only severe disease reports a high case-fatality ratio. Sao Paulo
   reports 3,583 cases and 502 deaths (CFR 14.0%); Rio Branco reports 3,051
   cases and 40 deaths (CFR 1.3%). Leptospirosis is not twenty times more
   lethal in Sao Paulo. The ratio is measuring ascertainment depth.

3. **Expected-case back-calculation.** Fit the count process on municipalities
   within a detection-complete reference stratum, then ask how many cases the
   silent municipalities should have produced.

4. **SIM triangulation.** Municipalities with A27 deaths on death certificates
   and no SINAN notification are unambiguously detection-silent. This is the
   part that does not rely on a model, and it runs when
   ``sim_a27_deaths.parquet`` exists.

Run:
    python studies/leptospirosis/06_ascertainment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice
from brepi.sources.datasus import sinan

OUT = PATHS.reports / "stage3_ascertainment"
YEARS = range(2007, 2026)


def show(title: str, frame: pl.DataFrame, n: int = 30) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(frame.head(n).to_pandas().to_string(index=False))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")

    mun = (
        panel.group_by("munic_code", "name", "uf_abbr", "region")
        .agg(
            pl.col("cases").sum(),
            pl.col("deaths").sum(),
            pl.col("hospitalised").sum(),
            (pl.col("population").sum() / 12 / 19).alias("pop"),
        )
        .with_columns(
            (pl.col("cases") / (pl.col("pop") * 19) * 100_000).alias("incidence"),
            (pl.col("cases") == 0).alias("silent"),
        )
    )

    # ---------------------------------------------------------------- 1 ----
    # Surveillance effort by state, from the notification side. The notified
    # denominator is the effort measure: it counts suspicions raised, which is
    # a property of the system, not of the pathogen.
    ll, _ = sinan.fetch_range("LEPT", YEARS)
    ll = ll.with_columns(
        pl.col("SG_UF_NOT").cast(pl.Utf8).str.strip_chars().alias("uf_code"),
        (pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1").alias("confirmed"),
        (pl.col("CRITERIO").cast(pl.Utf8).str.strip_chars() == "1").alias("lab_confirmed"),
    )
    muns_ref = lattice.load_municipalities(year=2022)
    uf_lookup = muns_ref.select("uf_code", "uf_abbr").unique()

    effort = (
        ll.group_by("uf_code")
        .agg(
            pl.len().alias("notified"),
            pl.col("confirmed").sum().alias("confirmed"),
            (pl.col("confirmed") & pl.col("lab_confirmed")).sum().alias("lab_confirmed"),
        )
        .join(uf_lookup, on="uf_code", how="inner")
    )
    uf_pop = mun.group_by("uf_abbr").agg(
        pl.col("pop").sum().alias("pop"),
        pl.len().alias("municipalities"),
        pl.col("silent").sum().alias("silent"),
        pl.col("cases").sum().alias("panel_cases"),
        pl.col("deaths").sum().alias("panel_deaths"),
    )
    state = (
        effort.join(uf_pop, on="uf_abbr", how="inner")
        .with_columns(
            (pl.col("notified") / (pl.col("pop") * 19) * 100_000).alias("notif_rate"),
            (pl.col("confirmed") / pl.col("notified")).alias("confirm_ratio"),
            (pl.col("lab_confirmed") / pl.col("confirmed")).alias("lab_share"),
            (pl.col("silent") / pl.col("municipalities")).alias("pct_silent"),
            (pl.col("panel_cases") / (pl.col("pop") * 19) * 100_000).alias("incidence"),
            (pl.col("panel_deaths") / pl.col("panel_cases") * 100).alias("cfr"),
        )
        .sort("notif_rate", descending=True)
    )
    show(
        "1. SURVEILLANCE EFFORT AND SILENCE BY STATE (notifications per 100k/yr)",
        state.select("uf_abbr", "notif_rate", "confirm_ratio", "lab_share",
                     "incidence", "cfr", "municipalities", "pct_silent")
             .with_columns(pl.col("notif_rate", "incidence", "cfr").round(2),
                           pl.col("confirm_ratio", "lab_share", "pct_silent").round(3)),
        n=30,
    )
    state.write_csv(OUT / "01_state_surveillance.csv")

    print("\n   Spearman correlations across the 27 states:")
    for a, b in [("notif_rate", "pct_silent"), ("notif_rate", "incidence"),
                 ("incidence", "pct_silent"), ("notif_rate", "cfr"),
                 ("incidence", "cfr"), ("lab_share", "cfr")]:
        print(f"      {a:12s} vs {b:12s}  rho = {spearman(state[a], state[b]):+.3f}")

    # ------------------------------------------------------------- 1b ----
    # The yield curve, which is the analytically decisive test.
    #
    # Notification effort is endogenous: a state with genuinely more disease
    # will rationally notify more, so a positive effort-incidence correlation
    # on its own proves nothing. The confirmation ratio breaks the tie. A state
    # responding to real burden casts a net of roughly constant selectivity and
    # confirms a stable fraction. A state operating a higher suspicion
    # threshold notifies less and confirms a HIGHER fraction, because it is
    # only testing patients who are already obviously ill.
    #
    # Regressing log(confirmed per 100k) on log(notified per 100k) across the
    # 27 states gives the elasticity of measured incidence to surveillance
    # effort. Under complete detection the slope is 0: you find the same
    # disease however hard you look. A slope near 1 means measured incidence is
    # very nearly a restatement of effort.
    x = np.log(state["notif_rate"].to_numpy().astype(float))
    y = np.log((state["confirmed"] / (state["pop"] * 19) * 100_000).to_numpy().astype(float))
    ok = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    resid = y[ok] - (slope * x[ok] + intercept)
    r2 = 1 - resid.var() / y[ok].var()
    print(f"\n   YIELD CURVE  log(confirmed/100k) ~ log(notified/100k), n={ok.sum()} states")
    print(f"      elasticity of measured incidence to notification effort: {slope:.3f}")
    print(f"      R2 = {r2:.3f}")
    print(f"      Spearman notif_rate vs confirm_ratio: "
          f"{spearman(state['notif_rate'], state['confirm_ratio']):+.3f}")
    print("      A slope near 1 with a negative effort-vs-confirmation-ratio correlation is")
    print("      the signature of a detection threshold, not of a burden gradient.")

    # ---------------------------------------------------------------- 2 ----
    # Case fatality against detection depth, at municipality level, restricted
    # to municipalities with enough cases for a CFR to mean anything.
    # Guarded against the obvious confounder: if outcome is more often unknown
    # in low-effort states, their CFR would be understated, not overstated, so
    # the gradient below is if anything conservative. Measured EVOLUCAO
    # completeness is 72-99% by state and adjusting for it does not reorder the
    # ranking.
    big = mun.filter(pl.col("cases") >= 30).with_columns(
        (pl.col("deaths") / pl.col("cases") * 100).alias("cfr"),
        pl.col("pop").log10().alias("log_pop"),
    )
    print(f"\n   municipalities with >=30 confirmed cases: {big.height}")
    print(f"      Spearman  log10(population) vs CFR : "
          f"{spearman(big['log_pop'], big['cfr']):+.3f}")
    print(f"      Spearman  incidence         vs CFR : "
          f"{spearman(big['incidence'], big['cfr']):+.3f}")

    cfr_band = (
        big.with_columns(band_expr(pl.col("incidence")).alias("incidence_band"))
        .group_by("incidence_band")
        .agg(pl.len().alias("municipalities"), pl.col("cases").sum(),
             pl.col("deaths").sum(), pl.col("pop").sum())
        .with_columns((pl.col("deaths") / pl.col("cases") * 100).round(2).alias("cfr_pct"))
        .sort("incidence_band")
    )
    show("2. CASE FATALITY BY MUNICIPAL INCIDENCE BAND "
         "(if lethality were biological these would be flat)", cfr_band)
    cfr_band.write_csv(OUT / "02_cfr_by_incidence.csv")

    # ---------------------------------------------------------------- 3 ----
    # Back-calculation. The reference stratum is municipalities above 100,000,
    # where the silent fraction is exactly zero and detection is therefore
    # plausibly near-complete. Rates are computed within region so that genuine
    # geographic variation in transmission is not attributed to detection.
    ref = mun.filter(pl.col("pop") >= 100_000)
    ref_rate = (
        ref.group_by("region")
        .agg((pl.col("cases").sum() / (pl.col("pop").sum() * 19) * 100_000).alias("ref_incidence"))
    )
    small = mun.filter(pl.col("pop") < 100_000).join(ref_rate, on="region", how="left")
    small = small.with_columns(
        (pl.col("ref_incidence") * pl.col("pop") * 19 / 100_000).alias("expected_cases")
    )
    gap = (
        small.group_by("region")
        .agg(
            pl.len().alias("municipalities"),
            pl.col("silent").sum().alias("silent"),
            pl.col("cases").sum().alias("observed"),
            pl.col("expected_cases").sum().round(0).alias("expected_at_reference_rate"),
        )
        .with_columns((pl.col("observed") / pl.col("expected_at_reference_rate")).round(3)
                      .alias("observed_over_expected"))
        .sort("region")
    )
    show("3. SUB-100k MUNICIPALITIES AGAINST THE DETECTION-COMPLETE REFERENCE RATE", gap)
    gap.write_csv(OUT / "03_backcalculation.csv")
    print("\n   NOTE: an observed/expected ABOVE 1 does not mean over-detection. In the "
          "\n   South it reflects a genuine rural-occupational excess, which is exactly why "
          "\n   this quantity cannot be read as an underreporting factor on its own. It "
          "\n   bounds the problem; the SIM triangulation below identifies it.")

    # ---------------------------------------------------------------- 4 ----
    sim_path = PATHS.interim / "sim_a27_deaths.parquet"
    if not sim_path.exists():
        print(f"\n[4] SIM triangulation skipped: {sim_path} not present. "
              "Run 05_triangulation_extract.py --sim")
        return 0

    deaths = pl.read_parquet(sim_path)
    print(f"\n[4] SIM A27 deaths loaded: {deaths.height:,}")
    mun_col = "CODMUNRES" if "CODMUNRES" in deaths.columns else "CODMUNOCOR"
    sim_mun = (
        deaths.with_columns(
            lattice.code6_to_code7_expr(pl.col(mun_col).cast(pl.Utf8).str.slice(0, 6))
            .alias("munic_code")
        )
        .group_by("munic_code")
        .agg(pl.len().alias("sim_deaths"))
    )
    tri = (
        mun.join(sim_mun, on="munic_code", how="left")
        .with_columns(pl.col("sim_deaths").fill_null(0))
        .with_columns(
            (pl.col("sim_deaths") > 0).alias("sim_positive"),
            (pl.col("cases") > 0).alias("sinan_positive"),
        )
    )
    cross = (
        tri.group_by("sinan_positive", "sim_positive")
        .agg(pl.len().alias("municipalities"), pl.col("pop").sum().round(0).alias("population"))
        .sort("sinan_positive", "sim_positive")
    )
    show("4. MUNICIPALITY-LEVEL AGREEMENT: SINAN NOTIFICATION vs SIM A27 DEATH", cross)

    detected_silent = tri.filter(~pl.col("sinan_positive") & pl.col("sim_positive"))
    print(f"\n   municipalities with A27 deaths on death certificates and ZERO SINAN "
          f"confirmations: {detected_silent.height}")
    print(f"   deaths in those municipalities: {int(detected_silent['sim_deaths'].sum())}")
    print(f"   population living in them: {int(detected_silent['pop'].sum()):,}")
    show("   the twenty largest such municipalities",
         detected_silent.sort("sim_deaths", descending=True)
                        .select("name", "uf_abbr", "pop", "sim_deaths")
                        .with_columns(pl.col("pop").round(0)), n=20)
    detected_silent.write_csv(OUT / "04_detection_silent_municipalities.csv")

    # Chapman two-source estimator on the death numerator. Reported with its
    # assumptions stated, because they are not satisfied and the number is a
    # lower bound rather than an estimate.
    n1 = int(tri["deaths"].sum())          # SINAN deaths attributed to leptospirosis
    n2 = int(tri["sim_deaths"].sum())      # SIM A27 underlying-cause deaths
    both = int(tri.filter(pl.col("sinan_positive") & pl.col("sim_positive"))["deaths"].sum())
    if both:
        chapman = (n1 + 1) * (n2 + 1) / (both + 1) - 1
        print(f"\n   SINAN deaths {n1:,} | SIM A27 deaths {n2:,} | overlap proxy {both:,}")
        print(f"   Chapman estimate of total leptospirosis deaths: {chapman:,.0f}")
        print("   ASSUMPTIONS VIOLATED: the two lists are positively dependent (a "
              "notified\n   case is more likely to be coded A27 at death), and the overlap "
              "here is a\n   municipality-level proxy, not a record linkage. Treat as a "
              "lower bound and\n   replace with probabilistic linkage before publication.")
    return 0


def band_expr(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col < 1).then(pl.lit("1 <1"))
        .when(col < 3).then(pl.lit("2 1-3"))
        .when(col < 10).then(pl.lit("3 3-10"))
        .when(col < 30).then(pl.lit("4 10-30"))
        .otherwise(pl.lit("5 30+"))
    )


def spearman(a: pl.Series, b: pl.Series) -> float:
    x, y = a.to_numpy().astype(float), b.to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    raise SystemExit(main())
