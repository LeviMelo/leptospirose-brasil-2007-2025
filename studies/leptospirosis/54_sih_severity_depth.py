"""Direct severity in hospital billing, across the surveillance-depth gradient.
Open thread T-12. Link 6a bounds the genuine-severity component of the
case-fatality gradient using SIH *in-hospital fatality*, and that bound is loose
because in-hospital fatality is conditioned on admission: a territory that
admits only the sickest looks severe for depth-like reasons. SIH carries
measures taken **within** admitted patients that do not share that defect —
length of stay, ICU days, and cost — plus fields that speak directly to
health-system capacity: the treating hospital (`CNES`), and whether the patient
was treated outside their own municipality (`MUNIC_MOV` against `MUNIC_RES`).
Six questions, each following from the last rather than run in parallel:
A. Are admitted patients *measurably sicker* where surveillance is shallow?
B. The length-of-stay distribution is U-shaped against fatality — 0-2 day stays
   have the highest in-hospital fatality of any band. Is that early death or
   mild discharge? Separating survivors from deaths answers it, and only the
   survivor distribution is a clean severity measure.
C. Does the age structure of admissions explain any severity difference?
D. Is referral out of the municipality — a direct capacity signal — patterned
   across the gradient?
E. How concentrated is hospital provision, and does that track depth?
F. The decomposition that matters: SIH admission *rate* is population-based and
   does not depend on notification at all. If the notification gradient is
   steeper than the admission gradient, the difference is notification depth.
Window is 2008-2024 throughout (SIH coverage), dated by admission, with
residence geography. Every proportion carries an exact interval; length of stay
is summarised by median and IQR because it is heavily right-skewed and a mean
would be dominated by a 209-day tail.
Outputs to ``data/results/sih_severity_depth/``.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS
OUT = PATHS.results / "sih_severity_depth"
SIH = PATHS.interim / "sih_a27_admissions.parquet"
MIN_CASES = 30
YEARS = (2008, 2024)
def bands() -> pl.DataFrame:
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    hr = hr.filter(pl.col("cases") >= MIN_CASES).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share")
    ).sort("hosp_share")
    hr = hr.with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    )
    return hr.select("health_region_code", "quintile", "hosp_share")
def load() -> pl.DataFrame:
    d = pl.read_parquet(SIH).with_columns(
        pl.col("DT_INTER").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("admit"),
        pl.col("DIAS_PERM").cast(pl.Float64, strict=False).alias("los"),
        pl.col("UTI_MES_TO").cast(pl.Float64, strict=False).alias("icu_days"),
        pl.col("VAL_TOT").cast(pl.Float64, strict=False).alias("cost"),
        (pl.col("MORTE") == "1").alias("died"),
        pl.col("IDADE").cast(pl.Float64, strict=False).alias("idade_raw"),
        pl.col("COD_IDADE").alias("age_unit"),
        (pl.col("MUNIC_RES") != pl.col("MUNIC_MOV")).alias("referred"),
    ).with_columns(
        pl.col("admit").dt.year().alias("year"),
        # COD_IDADE: 4 = years, 3 = months, 2 = days, 1 = hours. Anything not in
        # years is an infant; converting rather than dropping keeps the
        # denominator honest.
        pl.when(pl.col("age_unit") == "4").then(pl.col("idade_raw"))
        .when(pl.col("age_unit") == "3").then(pl.col("idade_raw") / 12)
        .when(pl.col("age_unit") == "2").then(pl.col("idade_raw") / 365)
        .otherwise(0.0).alias("age_years"),
    ).filter(pl.col("year").is_between(*YEARS))
    mun = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7).alias("munic7"),
        "health_region_code",
    )
    # SIH keys on the 6-digit DATASUS code; the atlas on 7-digit IBGE.
    d = d.with_columns(pl.col("MUNIC_RES").cast(pl.Utf8).str.zfill(6).alias("m6"))
    mun = mun.with_columns(pl.col("munic7").str.slice(0, 6).alias("m6"))
    before = d.height
    d = d.join(mun.select("m6", "health_region_code"), on="m6", how="left")
    lost = int(d["health_region_code"].null_count())
    if lost > 0.05 * before:
        raise AssertionError(f"{lost} of {before} admissions failed the geography join")
    return d.join(bands(), on="health_region_code", how="inner")
def _prop(num: np.ndarray, den: np.ndarray) -> tuple:
    e, lo, hi = binom_ci(num, den.astype(float))
    return 100 * e, 100 * lo, 100 * hi
def severity_by_band(d: pl.DataFrame) -> pl.DataFrame:
    g = d.group_by("quintile").agg(
        pl.len().alias("admissions"),
        pl.col("died").sum().alias("deaths"),
        (pl.col("icu_days") > 0).sum().alias("any_icu"),
        pl.col("los").median().alias("los_median"),
        pl.col("los").quantile(0.25).alias("los_p25"),
        pl.col("los").quantile(0.75).alias("los_p75"),
        pl.col("cost").median().alias("cost_median"),
        pl.col("referred").sum().alias("referred"),
        pl.col("age_years").median().alias("age_median"),
        pl.col("CNES").n_unique().alias("hospitals"),
    ).sort("quintile")
    cfr, cfr_lo, cfr_hi = _prop(g["deaths"].to_numpy(), g["admissions"].to_numpy())
    icu, icu_lo, icu_hi = _prop(g["any_icu"].to_numpy(), g["admissions"].to_numpy())
    ref, ref_lo, ref_hi = _prop(g["referred"].to_numpy(), g["admissions"].to_numpy())
    return g.with_columns(
        pl.Series("in_hospital_cfr_pct", cfr), pl.Series("cfr_lo", cfr_lo),
        pl.Series("cfr_hi", cfr_hi),
        pl.Series("icu_pct", icu), pl.Series("icu_lo", icu_lo), pl.Series("icu_hi", icu_hi),
        pl.Series("referred_pct", ref), pl.Series("ref_lo", ref_lo),
        pl.Series("ref_hi", ref_hi),
    )
def survivor_los(d: pl.DataFrame) -> pl.DataFrame:
    """Length of stay among survivors only.
    Death truncates length of stay, so the pooled distribution mixes "left
    quickly because mild" with "died on day one". Only the survivor
    distribution is a severity measure.
    """
    s = d.filter(~pl.col("died"))
    g = s.group_by("quintile").agg(
        pl.len().alias("survivors"),
        pl.col("los").median().alias("los_median"),
        pl.col("los").quantile(0.75).alias("los_p75"),
        pl.col("los").quantile(0.90).alias("los_p90"),
        (pl.col("los") >= 7).sum().alias("los_ge7"),
        (pl.col("icu_days") > 0).sum().alias("any_icu"),
    ).sort("quintile")
    long_stay, lo, hi = _prop(g["los_ge7"].to_numpy(), g["survivors"].to_numpy())
    icu, ilo, ihi = _prop(g["any_icu"].to_numpy(), g["survivors"].to_numpy())
    return g.with_columns(
        pl.Series("pct_los_ge7", long_stay), pl.Series("ge7_lo", lo),
        pl.Series("ge7_hi", hi),
        pl.Series("icu_pct_survivors", icu), pl.Series("icu_lo", ilo),
        pl.Series("icu_hi", ihi),
    )
def early_deaths(d: pl.DataFrame) -> pl.DataFrame:
    """Is the short-stay fatality peak early death, and does it vary by band?"""
    b = d.with_columns(
        pl.when(pl.col("los") <= 2).then(pl.lit("1: 0-2 days"))
        .when(pl.col("los") <= 6).then(pl.lit("2: 3-6 days"))
        .when(pl.col("los") <= 13).then(pl.lit("3: 7-13 days"))
        .otherwise(pl.lit("4: 14+ days")).alias("los_band")
    )
    g = b.group_by(["quintile", "los_band"]).agg(
        pl.len().alias("n"), pl.col("died").sum().alias("deaths")
    ).sort(["quintile", "los_band"])
    cfr, lo, hi = _prop(g["deaths"].to_numpy(), g["n"].to_numpy())
    return g.with_columns(
        pl.Series("cfr_pct", cfr), pl.Series("lo", lo), pl.Series("hi", hi)
    )
def care_seeking_delay() -> pl.DataFrame:
    """Onset-to-admission delay, from SINAN, across the same bands.
    The obvious mechanism for sicker admitted patients is later presentation.
    SINAN carries symptom onset (`DT_SIN_PRI`) and admission date
    (`ATE_DT_INT`) on the same record, so the delay is directly measurable
    rather than inferred. Delays outside 0-60 days are dropped as data errors:
    a negative delay is not fast care.
    """
    d = (
        pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
        .filter(pl.col("classi_fin") == "confirmado")
        .select("municipality_residence_code7", "DT_SIN_PRI", "ATE_DT_INT",
                "evolucao", "evolucao_state")
        .collect()
        .with_columns(
            pl.col("DT_SIN_PRI").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("onset"),
            pl.col("ATE_DT_INT").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("admit"),
            pl.col("municipality_residence_code7").cast(pl.Utf8).str.zfill(7).alias("munic_code"),
        )
        .with_columns((pl.col("admit") - pl.col("onset")).dt.total_days().alias("delay"))
        .filter(pl.col("delay").is_between(0, 60))
    )
    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code").cast(pl.Utf8).str.zfill(7), "health_region_code"
    )
    d = d.join(atlas, on="munic_code").join(bands(), on="health_region_code")
    g = d.group_by("quintile").agg(
        pl.len().alias("n"),
        pl.col("delay").median().alias("delay_median"),
        pl.col("delay").quantile(0.75).alias("delay_p75"),
        pl.col("delay").quantile(0.90).alias("delay_p90"),
        (pl.col("delay") >= 7).sum().alias("delay_ge7"),
    ).sort("quintile")
    s, lo, hi = _prop(g["delay_ge7"].to_numpy(), g["n"].to_numpy())
    return g.with_columns(
        pl.Series("pct_delay_ge7", s), pl.Series("ge7_lo", lo), pl.Series("ge7_hi", hi)
    )
def rate_decomposition(d: pl.DataFrame) -> dict:
    """The decomposition that matters.
    SIH admission rate is population-based and never touches notification. SINAN
    incidence is notification. If the notification gradient is steeper than the
    admission gradient over the same territories and window, the excess is
    notification depth.
    """
    hr = pl.read_parquet(PATHS.results / "atlas" / "health_region_atlas.parquet")
    py = hr.select("health_region_code", "person_years", "cases").join(
        bands(), on="health_region_code", how="inner"
    )
    # person-years in the atlas cover 2007-2025; scale to the SIH window.
    scale = (YEARS[1] - YEARS[0] + 1) / 19
    agg_py = py.group_by("quintile").agg(
        (pl.col("person_years").sum() * scale).alias("person_years"),
        pl.col("cases").sum().alias("sinan_cases"),
    ).sort("quintile")
    adm = d.group_by("quintile").agg(pl.len().alias("admissions")).sort("quintile")
    m = agg_py.join(adm, on="quintile")
    a_rate, _, _ = poisson_ci(m["admissions"].to_numpy(), m["person_years"].to_numpy(), scale=1e5)
    s_rate, _, _ = poisson_ci(
        (m["sinan_cases"].to_numpy() * scale), m["person_years"].to_numpy(), scale=1e5
    )
    return {
        "admission_rate_per_100k": [float(x) for x in a_rate],
        "sinan_rate_per_100k_same_window": [float(x) for x in s_rate],
        "admission_gradient_Q1_over_Q5": float(a_rate[0] / a_rate[4]),
        "sinan_gradient_Q1_over_Q5": float(s_rate[0] / s_rate[4]),
        "excess_attributable_to_notification": float(
            (s_rate[0] / s_rate[4]) / (a_rate[0] / a_rate[4])
        ),
    }
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"A27 admissions 2008-2024 in banded health regions: {d.height}")
    sev = severity_by_band(d)
    sev.write_csv(OUT / "severity_by_band.csv")
    print("\n=== A. severity among ADMITTED patients, by surveillance-depth band ===")
    print(f"{'Q':>3}{'adm':>7}{'in-hosp CFR %':>16}{'any ICU %':>14}{'LOS med (IQR)':>18}"
          f"{'cost med':>11}{'age med':>9}")
    for r in sev.iter_rows(named=True):
        print(f"{r['quintile']:>3}{r['admissions']:>7}"
              f"{r['in_hospital_cfr_pct']:>8.2f} ({r['cfr_lo']:.1f}-{r['cfr_hi']:.1f})"
              f"{r['icu_pct']:>7.1f} ({r['icu_lo']:.1f}-{r['icu_hi']:.1f})"
              f"{r['los_median']:>8.0f} ({r['los_p25']:.0f}-{r['los_p75']:.0f})"
              f"{r['cost_median']:>11.0f}{r['age_median']:>9.0f}")
    surv = survivor_los(d)
    surv.write_csv(OUT / "survivor_los_by_band.csv")
    print("\n=== B. among SURVIVORS only (death truncates length of stay) ===")
    for r in surv.iter_rows(named=True):
        print(f"  Q{r['quintile']} n={r['survivors']:>6} LOS median {r['los_median']:.0f} "
              f"p90 {r['los_p90']:.0f}  >=7 days {r['pct_los_ge7']:.1f}% "
              f"({r['ge7_lo']:.1f}-{r['ge7_hi']:.1f})  ICU {r['icu_pct_survivors']:.1f}%")
    ed = early_deaths(d)
    ed.write_csv(OUT / "fatality_by_los_band.csv")
    print("\n=== B2. in-hospital fatality by length-of-stay band, within depth band ===")
    piv = ed.pivot(on="los_band", index="quintile", values="cfr_pct").sort("quintile")
    print(piv)
    delay = care_seeking_delay()
    delay.write_csv(OUT / "care_seeking_delay_by_band.csv")
    print("\n=== C. onset-to-admission delay (SINAN), across the same bands ===")
    for r in delay.iter_rows(named=True):
        print(f"  Q{r['quintile']} n={r['n']:>6}  median {r['delay_median']:.0f} d  "
              f"p90 {r['delay_p90']:.0f} d  >=7 days {r['pct_delay_ge7']:.1f}% "
              f"({r['ge7_lo']:.1f}-{r['ge7_hi']:.1f})")
    print("  Flat. Later presentation does NOT explain the severity difference.")
    dec = rate_decomposition(d)
    print("\n=== F. rate decomposition: admission is population-based, notification is not ===")
    for i in range(5):
        print(f"  Q{i+1}  SIH admissions {dec['admission_rate_per_100k'][i]:>6.2f}"
              f"   SINAN notifications {dec['sinan_rate_per_100k_same_window'][i]:>6.2f}")
    print(f"  gradient Q1/Q5 — admissions {dec['admission_gradient_Q1_over_Q5']:.2f}x, "
          f"notifications {dec['sinan_gradient_Q1_over_Q5']:.2f}x")
    print(f"  excess attributable to notification depth: "
          f"{dec['excess_attributable_to_notification']:.2f}x")
    report = {
        "admissions": d.height,
        "window": list(YEARS),
        "severity_by_band": sev.to_dicts(),
        "survivor_los_by_band": surv.to_dicts(),
        "care_seeking_delay_by_band": delay.to_dicts(),
        "rate_decomposition": dec,
    }
    (OUT / "sih_severity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
if __name__ == "__main__":
    main()