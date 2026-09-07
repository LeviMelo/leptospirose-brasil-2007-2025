"""Is H confounded by differential record completeness, and is the study RECORD-compliant?

The reviewers' objection is sharper than "the manuscript should discuss
missingness". It is this: the exposure H and the outcome (case fatality) are
both derived from *fields on a surveillance form*, and a territory that fills
in fewer fields will produce a different H and a different case fatality for
reasons that have nothing to do with disease. If the fields that build H are
better recorded exactly where the fields that build case fatality are worse
recorded, the central gradient is partly an artefact of clerical practice.

**The threat, stated so it can lose.** Case fatality is deaths / cases *with a
recorded outcome*. If outcome completeness is LOWER where H is higher, then in
narrow-ascertainment territories that proportion is computed on a more selected
subset. Non-recording is not plausibly independent of dying — a death closes an
investigation and a survivor's record is the one left open — so the selection
has a direction: missing outcomes are enriched for survivors, and lower
completeness inflates reported case fatality. Lower completeness at high H
would therefore inflate the gradient in precisely the direction the study
claims.

**What would confirm the threat.** Outcome completeness falls materially across
H quintiles, AND the gradient collapses when restricted to well-recorded health
regions, AND the gradient does not survive an adversarial reallocation of the
unrecorded outcomes.

**What would rule it out.** The gradient survives restriction to health regions
with >= 95% and >= 99% outcome completeness, and — the decisive test — survives
the *worst case any missingness mechanism could produce*: every unrecorded
outcome in the broad-ascertainment band counted as a death and every unrecorded
outcome in the narrow band counted as a survivor. No mechanism can do more
damage than that bound, so if the ordering holds there, differential
completeness cannot have manufactured it.

Restriction, not adjustment. Carrying completeness as a covariate (as the RQ4
model does) assumes the missingness mechanism is ignorable given the covariate;
that is the assumption under test, so it cannot be the test.

The script also emits ``field_dictionary.csv``: every analytic variable in the
study with its source system, raw field, accepted codes, transformation,
missing-data rule and aggregation rule, derived by reading
``60_analysis_panel.py``, the codebook registry, ``brepi/`` package source and
the downstream study scripts rather than from memory. RECORD item 6.1 asks for
exactly this, and the study cannot claim the extension without it.

Outputs to ``data/results/data_quality_record/``.
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

OUT = PATHS.results / "data_quality_record"
PANEL = PATHS.results / "analysis_panel"

YEAR_MIN, YEAR_MAX = 2007, 2025

#: A health region with fewer than 30 confirmed cases over nineteen years
#: cannot support a case-fatality estimate; this is the manuscript's own stated
#: exclusion for the quintile analyses and is reused here so the completeness
#: strata and the gradient strata are the same objects. Regions below it are
#: not discarded silently: they are reported as their own row.
MIN_CASES = 30

#: Case-weighted quintiles: health regions sorted by H, cut so each band holds
#: a fifth of the confirmed cases. Equal case mass rather than equal region
#: count, because a band of 90 tiny regions and a band of 12 large ones would
#: carry incomparable precision. Q1 = lowest H = broadest ascertainment.
N_BINS = 5

#: Region-level outcome-completeness cuts for the sensitivity analysis. 0.95 is
#: strict enough that at most one case in twenty can be doing any work; 0.99 is
#: the strongest cut that leaves enough regions to band at all.
COMPLETENESS_CUTS = [0.95, 0.99]

#: The four analytic fields, plus the severity block both ways. Each entry is
#: (label, decoded column, state column). The severity block is handled
#: separately because it is a conjunction of three fields, not a field.
FIELDS: list[tuple[str, str, str]] = [
    ("outcome (evolucao)", "evolucao", "evolucao_state"),
    ("hospitalisation (ate_hosp)", "ate_hosp", "ate_hosp_state"),
    ("jaundice (cli_icteri)", "cli_icteri", "cli_icteri_state"),
    ("renal impairment (cli_renal)", "cli_renal", "cli_renal_state"),
    ("haemorrhage (cli_hemorr)", "cli_hemorr", "cli_hemorr_state"),
    ("confirmation criterion (criterio)", "criterio", "criterio_state"),
]
SEVERITY_PARTS = ["cli_icteri_state", "cli_renal_state", "cli_hemorr_state"]
SEVERITY_LABEL = "severity block (all three decoded)"

#: All seven completeness measures in report order.
MEASURES = [f[0] for f in FIELDS[:2]] + [SEVERITY_LABEL] + [f[0] for f in FIELDS[2:]]

#: Column name for each measure's valid-count in the region/stratum tables.
VALID_COL = {
    "outcome (evolucao)": "valid_outcome",
    "hospitalisation (ate_hosp)": "valid_hosp",
    "jaundice (cli_icteri)": "valid_icteri",
    "renal impairment (cli_renal)": "valid_renal",
    "haemorrhage (cli_hemorr)": "valid_hemorr",
    "confirmation criterion (criterio)": "valid_criterio",
    SEVERITY_LABEL: "valid_severity",
}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def cp(count, total) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clopper-Pearson, unpacked in the only correct order, then verified.

    ``binom_ci`` returns (ESTIMATE, lower, upper). Unpacking it as (lo, mid, hi)
    has already corrupted a published table in this repository once. The
    assertion turns a repeat into a crash rather than a plausible-looking
    number.
    """
    est, lo, hi = binom_ci(np.asarray(count, float), np.asarray(total, float))
    ok = np.isfinite(est)
    if np.any(ok):
        assert np.all(est[ok] >= lo[ok] - 1e-12), "estimate below its lower bound"
        assert np.all(est[ok] <= hi[ok] + 1e-12), "estimate above its upper bound"
    return est, lo, hi


def cp1(count: int, total: int) -> tuple[float, float, float]:
    """Scalar Clopper-Pearson, as native floats."""
    e, lo, hi = cp([count], [total])
    return float(e[0]), float(lo[0]), float(hi[0])


def spearman(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return {"rho": None, "p": None, "n": int(keep.sum())}
    r = stats.spearmanr(a[keep], b[keep])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(keep.sum())}


# ---------------------------------------------------------------------------
# The analytic population, rebuilt with the panel's own rules
# ---------------------------------------------------------------------------

def analytic_population() -> pl.DataFrame:
    """Confirmed cases 2007-2025 with a residence municipality, geocoded.

    Replicates ``60_analysis_panel.py`` exactly: the whole point of this script
    is to audit that panel's fields, so it must be auditing the same rows. The
    reconciliation assertion in :func:`main` is what makes that claim checkable
    rather than asserted.
    """
    cols = [
        "classi_fin", "epiweek_onset_year", "municipality_residence_code7",
        "evolucao", "evolucao_state", "ate_hosp", "ate_hosp_state",
        "criterio", "criterio_state",
        "cli_icteri", "cli_icteri_state", "cli_renal", "cli_renal_state",
        "cli_hemorr", "cli_hemorr_state",
    ]
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
    geo = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet").select(
        pl.col("munic_code").alias("municipality_residence_code7"),
        "health_region_code", "uf_abbr", "region",
    )
    line = line.join(geo, on="municipality_residence_code7", how="left")
    unmatched = int(line["health_region_code"].null_count())
    assert unmatched / line.height < 0.01, f"{unmatched} cases without a health region"
    line = line.filter(pl.col("health_region_code").is_not_null())

    # The severity block's joint state. Precedence invalid > unknown > missing:
    # a conjunction is only as good as its worst part, and the reason a case
    # drops out of the severe/non-severe denominator is a reportable fact, not
    # an undifferentiated NA. "unknown" outranks "missing" because somebody
    # actively recorded ignorance -- that is a different clerical event.
    parts = [pl.col(c) for c in SEVERITY_PARTS]
    line = line.with_columns(
        pl.when(pl.all_horizontal([p == "valid" for p in parts])).then(pl.lit("valid"))
        .when(pl.any_horizontal([p == "invalid" for p in parts])).then(pl.lit("invalid"))
        .when(pl.any_horizontal([p == "unknown" for p in parts])).then(pl.lit("unknown"))
        .otherwise(pl.lit("missing"))
        .alias("severity_state")
    )
    return line


def valid_flags(line: pl.DataFrame) -> pl.DataFrame:
    """One 0/1 column per measure, so every stratum table is a sum of counts."""
    exprs = [
        (pl.col(state) == "valid").cast(pl.Int32).alias(VALID_COL[label])
        for label, _field, state in FIELDS
    ]
    exprs.append(
        (pl.col("severity_state") == "valid").cast(pl.Int32).alias(VALID_COL[SEVERITY_LABEL])
    )
    return line.with_columns(exprs)


# ---------------------------------------------------------------------------
# (a) Completeness
# ---------------------------------------------------------------------------

STATE_COL = {label: state for label, _f, state in FIELDS} | {
    SEVERITY_LABEL: "severity_state"}


def state_breakdown(line: pl.DataFrame, label: str) -> dict:
    """valid / missing / unknown / invalid counts for one measure.

    The four states are reported separately and never pooled. `missing` means
    nobody filled the field; `unknown` means somebody recorded that the answer
    was not known; `invalid` means the file holds a code the dictionary does
    not define. Collapsing them is the single most consequential mistake
    available in a SINAN analysis, and RECORD item 6.1 asks for the
    distinction explicitly.
    """
    vc = line[STATE_COL[label]].value_counts().to_dict(as_series=False)
    counts = dict(zip(vc[STATE_COL[label]], vc["count"]))
    n = line.height
    valid = int(counts.get("valid", 0))
    est, lo, hi = cp1(valid, n)
    return {
        "measure": label,
        "denominator_confirmed_cases": n,
        "valid": valid,
        "missing": int(counts.get("missing", 0)),
        "unknown": int(counts.get("unknown", 0)),
        "invalid": int(counts.get("invalid", 0)),
        "pct_valid": 100 * est,
        "pct_valid_lo": 100 * lo,
        "pct_valid_hi": 100 * hi,
    }


def completeness_by(line: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Completeness of all seven measures within each stratum of ``keys``."""
    rows = []
    for key_vals, g in line.group_by(keys, maintain_order=False):
        base = dict(zip(keys, key_vals))
        for label in MEASURES:
            rows.append(base | state_breakdown(g, label))
    return pl.DataFrame(rows).sort(keys + ["measure"])


# ---------------------------------------------------------------------------
# Health-region level: the exposure and every completeness measure together
# ---------------------------------------------------------------------------

def region_frame(line: pl.DataFrame) -> pl.DataFrame:
    """Health-region totals: numerators and denominators first, ratios after."""
    agg = [pl.len().alias("cases")]
    agg += [pl.col(c).sum().alias(c) for c in VALID_COL.values()]
    agg += [
        ((pl.col("ate_hosp_state") == "valid") & (pl.col("ate_hosp") == "sim"))
        .sum().alias("hospitalised"),
        ((pl.col("evolucao_state") == "valid")
         & (pl.col("evolucao") == "obito_por_leptospirose")).sum().alias("deaths"),
    ]
    reg = line.group_by(["health_region_code", "uf_abbr", "region"]).agg(agg)
    reg = reg.with_columns(
        # H = hospitalised / cases with a VALID hospitalisation field. Higher H
        # = narrower ascertainment. The denominator is the valid-field count,
        # never the case count: a blank is not a "no".
        (pl.col("hospitalised") / pl.col("valid_hosp")).alias("H"),
        (pl.col("deaths") / pl.col("valid_outcome")).alias("cfr"),
    ).with_columns(
        [(pl.col(c) / pl.col("cases")).alias(c.replace("valid_", "comp_"))
         for c in VALID_COL.values()]
    )
    return reg.sort("health_region_code")


def case_weighted_quintiles(reg: pl.DataFrame) -> pl.DataFrame:
    """Bands of equal case mass, health regions ordered by H ascending."""
    d = reg.sort("H").with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw"))
    return d.with_columns(
        (pl.col("_cw") * N_BINS).ceil().clip(1, N_BINS).cast(pl.Int32).alias("H_quintile")
    ).drop("_cw")


# ---------------------------------------------------------------------------
# (b) Does completeness correlate with H?
# ---------------------------------------------------------------------------

def completeness_vs_H(elig: pl.DataFrame, allreg: pl.DataFrame) -> pl.DataFrame:
    """Spearman rho of each completeness measure against H, across regions.

    Reported on the eligible set (the regions the gradient is actually computed
    on) and on every region with a defined H. If the two disagree the
    eligibility rule is doing work and that has to be visible.

    The hospitalisation measure is a special case and is flagged as such: H's
    own denominator is that field's valid count, so a correlation between them
    is partly mechanical (a region that answers the field rarely has an H
    estimated on few cases), not a substantive confounding channel.
    """
    rows = []
    for label in MEASURES:
        comp = VALID_COL[label].replace("valid_", "comp_")
        e = spearman(elig[comp], elig["H"])
        a = spearman(allreg[comp], allreg["H"])
        rows.append({
            "measure": label,
            "rho_eligible": e["rho"], "p_eligible": e["p"],
            "n_regions_eligible": e["n"],
            "rho_all_regions": a["rho"], "p_all_regions": a["p"],
            "n_regions_all": a["n"],
            "shares_denominator_with_H": label == "hospitalisation (ate_hosp)",
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# (c) Sensitivity: does the gradient survive?
# ---------------------------------------------------------------------------

def gradient(reg: pl.DataFrame, label: str) -> pl.DataFrame:
    """Case fatality by H quintile. Counts summed first, ratio formed after."""
    g = reg.group_by("H_quintile").agg(
        pl.len().alias("health_regions"),
        pl.col("cases").sum(), pl.col("deaths").sum(),
        pl.col("valid_outcome").sum(), pl.col("hospitalised").sum(),
        pl.col("valid_hosp").sum(),
        pl.col("H").median().alias("H_median"),
    ).sort("H_quintile")
    cfr, lo, hi = cp(g["deaths"], g["valid_outcome"])
    H, H_lo, H_hi = cp(g["hospitalised"], g["valid_hosp"])
    comp, _, _ = cp(g["valid_outcome"], g["cases"])
    return g.with_columns(
        pl.lit(label).alias("variant"),
        pl.Series("cfr_pct", 100 * cfr),
        pl.Series("cfr_lo", 100 * lo), pl.Series("cfr_hi", 100 * hi),
        pl.Series("H_pooled", H), pl.Series("H_lo", H_lo), pl.Series("H_hi", H_hi),
        pl.Series("outcome_completeness_pct", 100 * comp),
    )


def span(t: pl.DataFrame) -> dict:
    """Lowest vs highest band present, with the exposure contrast alongside.

    ``exposure_contrast_H`` is reported because the case-fatality ratio between
    the end bands is NOT comparable across variants unless the bands span the
    same range of H. Restriction moves the H distribution, and a variant whose
    end bands are 0.36 and 0.96 apart is answering a different question from
    one whose end bands are 0.32 and 0.86 apart. Reading the ratios without it
    would be the arithmetic equivalent of comparing two different exposures.
    """
    q1 = t.row(0, named=True)
    q5 = t.row(t.height - 1, named=True)
    return {
        "bands_present": [int(v) for v in t["H_quintile"].to_list()],
        "q_low": int(q1["H_quintile"]), "q_high": int(q5["H_quintile"]),
        "health_regions": [int(v) for v in t["health_regions"].to_list()],
        "q_low_cfr_pct": [q1["cfr_pct"], q1["cfr_lo"], q1["cfr_hi"]],
        "q_low_known_outcomes": int(q1["valid_outcome"]),
        "q_high_cfr_pct": [q5["cfr_pct"], q5["cfr_lo"], q5["cfr_hi"]],
        "q_high_known_outcomes": int(q5["valid_outcome"]),
        "ratio_high_over_low": q5["cfr_pct"] / q1["cfr_pct"] if q1["cfr_pct"] else None,
        "intervals_disjoint": bool(q5["cfr_lo"] > q1["cfr_hi"]),
        "cases_retained": int(t["cases"].sum()),
        "exposure_contrast_H": [q1["H_pooled"], q5["H_pooled"]],
        "exposure_span_H": q5["H_pooled"] - q1["H_pooled"],
    }


def restricted_gradients(elig: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """The gradient under region-level outcome-completeness restriction.

    Two variants per cut, answering two different objections.

    * *fixed bands* keeps each region in the quintile the full data assigned
      it, and only drops poorly-recorded regions. This holds the exposure
      contrast exactly as published and isolates the recording mechanism.
    * *rebanded* recomputes the case-weighted quintiles inside the surviving
      regions. This is the self-contained replication, and it answers the
      objection that the bands themselves were drawn using regions that were
      then thrown away.
    """
    tables = [gradient(elig, "all eligible regions (no completeness restriction)")]
    summary = {"all eligible regions (no completeness restriction)": span(tables[0])}
    comp = VALID_COL["outcome (evolucao)"].replace("valid_", "comp_")

    for cut in COMPLETENESS_CUTS:
        sub = elig.filter(pl.col(comp) >= cut - 1e-12)
        lab_fixed = f"fixed bands, region outcome completeness >= {cut:.2f}"
        t_fixed = gradient(sub, lab_fixed)
        tables.append(t_fixed)
        summary[lab_fixed] = span(t_fixed) | {
            "regions_retained": int(sub.height),
            "regions_total": int(elig.height),
            "case_retention_pct": 100 * float(sub["cases"].sum()) / float(elig["cases"].sum()),
        }

        lab_re = f"rebanded, region outcome completeness >= {cut:.2f}"
        t_re = gradient(case_weighted_quintiles(sub.drop("H_quintile")), lab_re)
        tables.append(t_re)
        summary[lab_re] = span(t_re) | {
            "regions_retained": int(sub.height),
            "case_retention_pct": 100 * float(sub["cases"].sum()) / float(elig["cases"].sum()),
        }
    return pl.concat(tables, how="diagonal"), summary


def extreme_bounds(elig: pl.DataFrame, line: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Bounds on case fatality when no assumption is made about the unrecorded.

    Three national quantities, and then the one that actually settles the
    question.

    * *observed* -- deaths / cases with a recorded outcome. What the study
      reports.
    * *all unknown survived* -- deaths / all confirmed cases. The lower bound.
    * *all unknown died* -- (deaths + unrecorded) / all confirmed cases. The
      upper bound. It is deliberately absurd, and its distance from the
      observed value is the honest width of what the missingness can do.

    Then the ADVERSARIAL GRADIENT bound. The threat is not that missingness
    moves the national number; it is that it moves the two ends of the exposure
    differently. So: give every unrecorded outcome in the broadest band a
    death, and every unrecorded outcome in the narrowest band a survival. That
    is the largest attenuation any missing-data mechanism whatsoever can
    produce. If the narrow band still has the higher case fatality under it,
    differential completeness cannot be what made the gradient.
    """
    n_cases = line.height
    deaths = int(((line["evolucao_state"] == "valid")
                  & (line["evolucao"] == "obito_por_leptospirose")).sum())
    known = int((line["evolucao_state"] == "valid").sum())
    unrecorded = n_cases - known

    rows = []
    for label, num, den in [
        ("observed (deaths / cases with a recorded outcome)", deaths, known),
        ("lower bound: every unrecorded outcome survived", deaths, n_cases),
        ("upper bound: every unrecorded outcome died", deaths + unrecorded, n_cases),
    ]:
        e, lo, hi = cp1(num, den)
        rows.append({
            "scenario": label, "deaths_assumed": num, "denominator": den,
            "cfr_pct": 100 * e, "cfr_lo": 100 * lo, "cfr_hi": 100 * hi,
        })
    national = pl.DataFrame(rows)

    # Adversarial reallocation across the exposure.
    q = elig.group_by("H_quintile").agg(
        pl.col("cases").sum(), pl.col("deaths").sum(),
        pl.col("valid_outcome").sum(),
    ).sort("H_quintile").with_columns(
        (pl.col("cases") - pl.col("valid_outcome")).alias("unrecorded")
    )
    lo_band, hi_band = q.row(0, named=True), q.row(q.height - 1, named=True)

    def scen(band: dict, assume_all_died: bool) -> tuple[int, int]:
        num = band["deaths"] + (band["unrecorded"] if assume_all_died else 0)
        return int(num), int(band["cases"])

    adv_rows = []
    for name, low_died, high_died in [
        ("observed", None, None),
        ("adversarial: broad band all died, narrow band all survived", True, False),
        ("favourable: broad band all survived, narrow band all died", False, True),
    ]:
        if low_died is None:
            nl, dl = int(lo_band["deaths"]), int(lo_band["valid_outcome"])
            nh, dh = int(hi_band["deaths"]), int(hi_band["valid_outcome"])
        else:
            nl, dl = scen(lo_band, low_died)
            nh, dh = scen(hi_band, high_died)
        el, ll, ul = cp1(nl, dl)
        eh, lh, uh = cp1(nh, dh)
        adv_rows.append({
            "scenario": name,
            "q_low_cfr_pct": 100 * el, "q_low_lo": 100 * ll, "q_low_hi": 100 * ul,
            "q_low_denominator": dl,
            "q_high_cfr_pct": 100 * eh, "q_high_lo": 100 * lh, "q_high_hi": 100 * uh,
            "q_high_denominator": dh,
            "ratio_high_over_low": eh / el if el else None,
            "ordering_preserved": bool(eh > el),
            "intervals_disjoint": bool(lh > ul),
        })
    adversarial = pl.DataFrame(adv_rows)

    report = {
        "national": national.to_dicts(),
        "confirmed_cases": n_cases,
        "outcomes_recorded": known,
        "outcomes_unrecorded": unrecorded,
        "adversarial_gradient": adversarial.to_dicts(),
    }
    return national, adversarial, report


def exposure_bounds(elig: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Bound H itself, which is what the question literally asks.

    Everything above bounds the OUTCOME against its missing field. But the
    exposure is a recorded field too: 3.4% of confirmed cases have no valid
    ate_hosp, and if those cases are not missing at random the ordering of the
    bands by H could itself be clerical. So bound H the same way. The
    adversarial case for the exposure is the one that compresses the bands
    most: every unrecorded case in the broad band counted as hospitalised, and
    every unrecorded case in the narrow band counted as not hospitalised.

    If the bands stay ordered under that assignment, no missing-data mechanism
    on ATE_HOSP can have created the exposure contrast either.
    """
    q = elig.group_by("H_quintile").agg(
        pl.col("cases").sum(), pl.col("hospitalised").sum(),
        pl.col("valid_hosp").sum(),
    ).sort("H_quintile").with_columns(
        (pl.col("cases") - pl.col("valid_hosp")).alias("unrecorded")
    )
    rows = []
    for r in q.iter_rows(named=True):
        obs, obs_lo, obs_hi = cp1(int(r["hospitalised"]), int(r["valid_hosp"]))
        lo_e, _, _ = cp1(int(r["hospitalised"]), int(r["cases"]))
        hi_e, _, _ = cp1(int(r["hospitalised"] + r["unrecorded"]), int(r["cases"]))
        rows.append({
            "H_quintile": int(r["H_quintile"]), "cases": int(r["cases"]),
            "hosp_field_valid": int(r["valid_hosp"]),
            "hosp_field_unrecorded": int(r["unrecorded"]),
            "H_observed": obs, "H_observed_lo": obs_lo, "H_observed_hi": obs_hi,
            "H_if_all_unrecorded_not_hospitalised": lo_e,
            "H_if_all_unrecorded_hospitalised": hi_e,
        })
    t = pl.DataFrame(rows)
    # The compressing assignment: broad band pushed up, narrow band pushed down.
    lo_band, hi_band = rows[0], rows[-1]
    worst_low = lo_band["H_if_all_unrecorded_hospitalised"]
    worst_high = hi_band["H_if_all_unrecorded_not_hospitalised"]
    summary = {
        "observed_contrast": [lo_band["H_observed"], hi_band["H_observed"]],
        "compressed_contrast": [worst_low, worst_high],
        "ordering_survives_compression": bool(worst_high > worst_low),
        "residual_span": worst_high - worst_low,
        "unrecorded_cases_in_low_band": lo_band["hosp_field_unrecorded"],
        "unrecorded_cases_in_high_band": hi_band["hosp_field_unrecorded"],
    }
    return t, summary


# ---------------------------------------------------------------------------
# (d) The field dictionary
# ---------------------------------------------------------------------------

def field_dictionary() -> pl.DataFrame:
    """Every analytic variable in the study, read out of the code that makes it.

    Sources read to build this: ``brepi/codebook/registry/datasus_codebook.yaml``
    (concepts, bindings, transforms, references; registry version 6, references
    version 3), ``brepi/codebook/transforms.py``, ``brepi/codebook/references.py``,
    ``docs/CODEBOOK.md``, ``brepi/analysis/rates.py``,
    ``brepi/sources/datasus/population.py``, ``brepi/geo/health_regions.py``,
    ``brepi/geo/lattice.py``, and the study scripts
    ``21_build_line_level.py``, ``02_denominators.py``, ``60_analysis_panel.py``,
    ``62_triangulation.py``, ``47_audit_sim_sih.py``, ``52_exposure_routes.py``,
    ``53_socioeconomic_confounders.py``, ``54_sih_severity_depth.py`` and
    ``44_depth_vs_severity.py``.

    Two disagreements between scripts are recorded as rows rather than
    smoothed over; a dictionary that hides them is not a dictionary.
    """
    S = "SINAN-LEPT (SINAN NET, annual national LEPTBR files, 2007-2025)"
    SIM = "SIM-DO (Sistema de Informacoes sobre Mortalidade, declaration files)"
    SIH = "SIH-RD (Sistema de Informacoes Hospitalares, reduced AIH files)"
    IBGE = "IBGE via DATASUS"
    SIDRA = "IBGE Censo 2022 via SIDRA API"
    DERIVED = "derived (analysis panel)"

    # The four-state decode contract, quoted once and referenced by every
    # SINAN row so the rule is not paraphrased seven slightly different ways.
    FOURSTATE = (
        "decoded value carries a companion <field>_state in "
        "{valid, missing, unknown, invalid}; blank -> missing, explicit "
        "'ignorado' code -> unknown, code absent from the registry -> invalid. "
        "The four are never pooled."
    )

    rows: list[dict] = [
        # ---------------- SINAN: inclusion and placement ----------------
        dict(
            variable="classi_fin", source_system=S, raw_field="CLASSI_FIN",
            accepted_values="1=confirmado; 2=descartado; 8=inconclusivo. No 'ignorado' code declared.",
            transformation="enum lookup, concept lept_classi_fin; codes matched verbatim and zero-stripped",
            missing_data_rule=(
                "Only classi_fin=='confirmado' enters the analytic population. "
                "descartado, inconclusivo and no-classification records are "
                "counted as separate exclusion rows in study_flow.csv, not dropped silently. " + FOURSTATE),
            aggregation_rule="count of records = 'cases'",
            used_in="study population definition (60_analysis_panel.py, study_flow.csv)"),
        dict(
            variable="epiweek_onset_year / epiweek_onset_week", source_system=S,
            raw_field="SEM_PRI",
            accepted_values="YYYYWW; year 1990-2100, week 1-53 (week 53 kept, it is legitimate)",
            transformation="epi_week transform (brepi/codebook/transforms.py): zero-fill to 6, split 4+2, cast",
            missing_data_rule=(
                "Records with a null onset year cannot be placed in any year and are "
                "excluded as a named cascade step (155 confirmed records); a valid "
                "onset year outside 2007-2025 is a separate exclusion (105). Week "
                "outside 1-53 -> invalid, never coerced."),
            aggregation_rule="assigns each case to one calendar year; the panel's time key",
            used_in="analytic window; every time series"),
        dict(
            variable="municipality_residence_code7", source_system=S,
            raw_field="ID_MN_RESI",
            accepted_values="6-digit IBGE municipality code (no check digit), resolved against the 2022 lattice (5,570 municipalities)",
            transformation=(
                "reference transform against ibge_municipality (zero_pad 6); emits name plus "
                "attributes uf, region, code7. code6->code7 via the check-digit codec in "
                "brepi/geo/lattice.py, not string concatenation."),
            missing_data_rule=(
                "Null residence municipality -> excluded as a named cascade step. "
                "Residence, never notification (ID_MUNICIP) or infection (COMUNINF) "
                "municipality; the three differ and mixing them mixes attributions."),
            aggregation_rule="joined to health_region_code via municipality_atlas; case attributed to region of residence",
            used_in="all geography; health-region panel"),
        # ---------------- SINAN: the four analytic fields ----------------
        dict(
            variable="evolucao (outcome)", source_system=S, raw_field="EVOLUCAO",
            accepted_values="1=cura; 2=obito_por_leptospirose; 3=obito_por_outras_causas; 9=ignorado (-> unknown)",
            transformation="enum lookup, concept lept_evolucao",
            missing_data_rule=(
                "death = state=='valid' AND value=='obito_por_leptospirose'. "
                "obito_por_outras_causas is a KNOWN outcome that is not a case "
                "fatality. outcome_known = state=='valid' and is the case-fatality "
                "denominator: a blank is never read as survival. " + FOURSTATE),
            aggregation_rule="sum(deaths) / sum(outcome_known) after summing both; never a mean of region ratios",
            used_in="case fatality; the study's outcome"),
        dict(
            variable="ate_hosp (hospitalisation)", source_system=S, raw_field="ATE_HOSP",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown). Concept sinan_sim_nao_ign, shared with 29 clinical/antecedent fields.",
            transformation="enum lookup, concept sinan_sim_nao_ign",
            missing_data_rule=(
                "hospitalised = state=='valid' AND value=='sim'. hosp_known = "
                "state=='valid'. A blank is not a 'nao'. " + FOURSTATE),
            aggregation_rule="H = sum(hospitalised) / sum(hosp_known); the exposure",
            used_in="the exposure H (and B = 1 - H)"),
        dict(
            variable="criterio (confirmation criterion)", source_system=S, raw_field="CRITERIO",
            accepted_values="1=clinico_laboratorial; 2=clinico_epidemiologico. No 'ignorado' code declared, so this field has no 'unknown' state.",
            transformation="enum lookup, concept lept_criterio",
            missing_data_rule="lab_confirmed = state=='valid' AND value=='clinico_laboratorial'; crit_known = state=='valid'. " + FOURSTATE,
            aggregation_rule="share_lab = sum(lab_confirmed) / sum(crit_known)",
            used_in="case-mix covariate; diagnostic-rigour check on the gradient"),
        dict(
            variable="cli_icteri (jaundice)", source_system=S, raw_field="CLI_ICTERI",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown)",
            transformation="enum lookup, concept sinan_sim_nao_ign",
            missing_data_rule="Contributes to the severe phenotype only when its own state is valid. " + FOURSTATE,
            aggregation_rule="enters the severe-phenotype disjunction; not aggregated alone in the panel",
            used_in="severe phenotype"),
        dict(
            variable="cli_renal (renal impairment)", source_system=S, raw_field="CLI_RENAL",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown)",
            transformation="enum lookup, concept sinan_sim_nao_ign",
            missing_data_rule="Contributes to the severe phenotype only when its own state is valid. " + FOURSTATE,
            aggregation_rule="enters the severe-phenotype disjunction",
            used_in="severe phenotype"),
        dict(
            variable="cli_hemorr (haemorrhage)", source_system=S, raw_field="CLI_HEMORR",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown)",
            transformation="enum lookup, concept sinan_sim_nao_ign",
            missing_data_rule="Contributes to the severe phenotype only when its own state is valid. " + FOURSTATE,
            aggregation_rule="enters the severe-phenotype disjunction",
            used_in="severe phenotype"),
        dict(
            variable="severe (severe phenotype)", source_system=DERIVED,
            raw_field="CLI_ICTERI, CLI_RENAL, CLI_HEMORR",
            accepted_values="1 if any of the three decodes to 'sim'; 0 if all three decode and none is 'sim'",
            transformation="disjunction over the three decoded fields (60_analysis_panel.py)",
            missing_data_rule=(
                "sev_known requires ALL THREE states to be valid; a case with two "
                "answers and one blank does not enter the denominator, because a "
                "blank third sign is not a negative third sign. This is a conjunctive "
                "denominator and is the least complete of the study's measures."),
            aggregation_rule="share_severe = sum(severe) / sum(sev_known)",
            used_in="case-mix; severity decomposition of the gradient"),
        dict(
            variable="cli_hemopu (pulmonary haemorrhage)", source_system=S,
            raw_field="CLI_HEMOPU",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown)",
            transformation="enum lookup, concept sinan_sim_nao_ign",
            missing_data_rule=(
                "EXCLUDED from the canonical severe phenotype in 60_analysis_panel.py: it "
                "is in practice the same field as CLI_HEMORR (they disagree in 13 of "
                "61,219 records). DISCREPANCY ON RECORD: 44_depth_vs_severity.py defines "
                "SEVERE_FIELDS with four fields including cli_hemopu. The panel's "
                "three-field rule is canonical; any figure cut from 44 uses the four-field "
                "variant and must say so."),
            aggregation_rule="not aggregated in the canonical panel",
            used_in="44_depth_vs_severity.py only (variant definition)"),
        # ---------------- SINAN: case-mix ----------------
        dict(
            variable="age_years", source_system=S, raw_field="NU_IDADE_N",
            accepted_values="packed UVVV: unit 1=hours, 2=days, 3=months, 4=years; value 000-999. Unit outside 1-4 -> invalid.",
            transformation=(
                "sinan_packed_age transform: sub-year ages converted (months/12, "
                "days/365.25, hours/8766), not floored to zero; age_unit retained alongside."),
            missing_data_rule=(
                "age60 = state=='valid' AND age_years >= 60; age_known = state=='valid'. "
                "A stray unit digit is invalid, never a newborn -- treating it as one "
                "would inflate the band where leptospirosis is rarest."),
            aggregation_rule="share_age60 = sum(age60) / sum(age_known)",
            used_in="case-mix covariate (age cut 60, named AGE_CUT in 60_analysis_panel.py)"),
        dict(
            variable="cs_sexo (sex)", source_system=S, raw_field="CS_SEXO",
            accepted_values="M=masculino; F=feminino; I=ignorado (-> unknown)",
            transformation="enum lookup, concept sinan_sexo",
            missing_data_rule=(
                "male = (cs_sexo=='masculino'). NOTE: the panel computes share_male on "
                "the ALL-CASES denominator rather than on the valid-state denominator, "
                "unlike every other proportion in the panel. It is a descriptive "
                "covariate, not an estimand, but the asymmetry is on record."),
            aggregation_rule="share_male = sum(male) / cases",
            used_in="case-mix covariate; sex-specific rate ratio"),
        dict(
            variable="con_ambien (infection setting)", source_system=S, raw_field="CON_AMBIEN",
            accepted_values="SINAN environment-of-infection enumeration (concept bound in the registry)",
            transformation="enum lookup",
            missing_data_rule=FOURSTATE,
            aggregation_rule="case counts and case fatality within setting, computed on valid-state cases only",
            used_in="52_exposure_routes.py: does infection setting explain the gradient?"),
        dict(
            variable="ant_cb_* (14 exposure antecedents)", source_system=S,
            raw_field="ANT_CB_LAM, ANT_CB_CRI, ANT_CB_CAI, ANT_CB_FOS, ANT_CB_SIN, ANT_CB_PLA, ANT_CB_COR, ANT_CB_ROE, ANT_CB_GRA, ANT_CB_TER, ANT_CB_LIX, ANT_CB_OUT, ANT_HUMANO, ANT_ANIMAI",
            accepted_values="1=sim; 2=nao; 9=ignorado (-> unknown), each",
            transformation="enum lookup, concept sinan_sim_nao_ign (one concept bound to all 14)",
            missing_data_rule="each antecedent computed on its own valid denominator; 81-90% recorded on confirmed cases. " + FOURSTATE,
            aggregation_rule="share reporting each route, per stratum",
            used_in="52_exposure_routes.py"),
        dict(
            variable="serovar / serogroup / reservoir", source_system=S,
            raw_field="MICRO1_S1 (and MICRO1_S_2, MICRO2_S1, MICRO2_S_2)",
            accepted_values=(
                "MAT serovar code 1-28, written either bare ('10') or self-labelled "
                "('10 - COPENHAGENI') in the same column; serogroup and maintenance host "
                "attached as concept attributes. Serovar 21 (Patoc) is L. biflexa, "
                "saprophytic, and is never attributed to a reservoir."),
            transformation="labelled_code transform: take the leading integer, then an ordinary enum; attributes emit serogroup and reservoir in the same pass",
            missing_data_rule="MAT typing reaches 9,003 of 66,358 confirmed cases (13.5%); absence of a MAT result is not absence of infection",
            aggregation_rule="modal serogroup per municipality with its own count and share (atlas)",
            used_in="descriptive reservoir table (not in the exposure-outcome chain)"),
        dict(
            variable="occupation_major_group", source_system=S, raw_field="ID_OCUPA_N",
            accepted_values="CBO 2002 six-digit code; first digit = major group 0-9. SINAN sentinels 000000, 998999, 999991-999995, 999999 -> unknown.",
            transformation="prefix_group transform (pad 6, width 1): decode only what the code structurally guarantees",
            missing_data_rule="sentinels mapped to unknown, NOT to major group 9; 22,942 of 66,358 confirmed cases (34.4%) carry a usable code",
            aggregation_rule="counts by major group",
            used_in="descriptive occupational exposure (not in the exposure-outcome chain)"),
        dict(
            variable="notification and digitisation dates", source_system=S,
            raw_field="DT_SIN_PRI, DT_NOTIFIC, DT_DIGITA, ATE_DT_INT",
            accepted_values="dates; retained as raw text and parsed per use (%Y-%m-%d in the decoded frame)",
            transformation="date parse at point of use, strict=False so an unparseable date is null rather than an exception",
            missing_data_rule="delay statistics computed only on record pairs where both dates parse; the pair count is reported",
            aggregation_rule="median and quantiles of the day difference, per stratum",
            used_in="notification-delay diagnostics; care-seeking delay by band (54_sih_severity_depth.py)"),
        dict(
            variable="src_year (file vintage)", source_system=S, raw_field="file of origin",
            accepted_values="2007-2025",
            transformation="literal added per annual file in 21_build_line_level.py",
            missing_data_rule="never null by construction",
            aggregation_rule=(
                "NOT the study's time key. 60_analysis_panel.py keys on onset year; "
                "41_completeness_stratification.py keys on src_year. The two are "
                "different attributions of the same case and their totals differ."),
            used_in="schema-drift audit; the older completeness script's time key"),
        # ---------------- SIM ----------------
        dict(
            variable="sim_a27_deaths", source_system=SIM, raw_field="CAUSABAS, CODMUNRES, DTOBITO",
            accepted_values="CAUSABAS with ICD-10 prefix A27 (anchored); CODMUNRES 6-digit; DTOBITO ddmmyyyy",
            transformation="prefix match on CAUSABAS; year taken as characters 5-8 of DTOBITO; municipality joined on the 6-digit form",
            missing_data_rule=(
                "Underlying cause only for the primary count. The multiple-cause chain "
                "(LINHAA-LINHAD, LINHAII, CAUSABAS_O) is scanned separately in "
                "47_audit_sim_sih.py; the two definitions give different totals and are "
                "not interchangeable."),
            aggregation_rule="deaths summed per health region; mortality rate = deaths / person-years, exact Poisson interval",
            used_in="62_triangulation.py: population mortality against reported case fatality"),
        # ---------------- SIH ----------------
        dict(
            variable="sih_a27_admissions", source_system=SIH, raw_field="DIAG_PRINC, MUNIC_RES, DT_INTER",
            accepted_values="DIAG_PRINC starting 'A27'; MUNIC_RES 6-digit; DT_INTER yyyy-mm-dd",
            transformation="prefix match on DIAG_PRINC; admission year from DT_INTER; municipality joined on the 6-digit form",
            missing_data_rule=(
                "Admissions, not people: an AIH is a billing record and a patient "
                "transferred between hospitals generates more than one. Window restricted "
                "to 2008-2024, the years all three systems cover completely."),
            aggregation_rule="admissions summed per health region; admission rate = admissions / person-years, exact Poisson interval",
            used_in="62_triangulation.py"),
        dict(
            variable="sih severity and cost fields", source_system=SIH,
            raw_field="DIAS_PERM, UTI_MES_TO, MORTE, VAL_TOT, CNES, MUNIC_MOV, IDADE, COD_IDADE",
            accepted_values="DIAS_PERM/UTI_MES_TO/VAL_TOT numeric; MORTE=='1' is in-hospital death; COD_IDADE 4=years, 3=months, 2=days, 1=hours",
            transformation="cast to float, strict=False; referred = (MUNIC_RES != MUNIC_MOV); age converted from IDADE by COD_IDADE",
            missing_data_rule="a value that will not cast is null, not zero; length-of-stay statistics restricted to survivors where stated",
            aggregation_rule="median length of stay, ICU share, in-hospital case fatality per H band",
            used_in="54_sih_severity_depth.py: is case mix harder where H is high?"),
        # ---------------- denominators and geography ----------------
        dict(
            variable="person_years", source_system=IBGE,
            raw_field="POPSVS POPSBR{YY} (COD_MUN 7-digit, SEXO, age 000-080)",
            accepted_values="municipal population by sex and single year of age, 2007-2025; age 080 is 80-and-over",
            transformation=(
                "summed over sex and age to municipal totals, then over municipalities to "
                "health region and year. 6-digit codes converted via the check-digit codec, "
                "not string concatenation."),
            missing_data_rule=(
                "Gate: the 2022 total must land within 5% of the Censo 2022 count of "
                "203,080,756; it is 3.9% higher because the estimates incorporate "
                "post-enumeration undercount. A failing gate aborts the build."),
            aggregation_rule="one person-year per resident per calendar year; summed before any rate is formed",
            used_in="incidence and mortality rate denominators"),
        dict(
            variable="health_region_code / health_region_name", source_system="Ministry of Health (CIB/CIR resolutions) via OpenDataSUS",
            raw_field="municipality -> regiao de saude crosswalk",
            accepted_values="official dated extract; every one of the 5,570 municipalities in the 2022 lattice must map, or the build raises",
            transformation="brepi/geo/health_regions.py: normalise, verify uniqueness of code7, verify full coverage against the lattice",
            missing_data_rule=(
                "Never derived from municipality names, centroids or a stale lookup. "
                "SINAN's own ID_REGIONA / ID_RG_RESI are deliberately NOT used: that "
                "division is revised without a public changelog (ADR-001)."),
            aggregation_rule="the study's spatial unit; 432 regions carry at least one confirmed case",
            used_in="every regional analysis"),
        dict(
            variable="uf_abbr / region (macro-region)", source_system=IBGE,
            raw_field="derived from the municipality code",
            accepted_values="27 UF abbreviations; 5 macro-regions (Norte, Nordeste, Sudeste, Sul, Centro-Oeste)",
            transformation=(
                "macro-region taken from the UF table rather than from the municipality "
                "lattice, so a municipality's region and its UF's region are the same "
                "string by construction; taking each from its own source once produced "
                "ten macro-regions instead of five."),
            missing_data_rule="a municipality code absent from the lattice yields a null region and is excluded with a count",
            aggregation_rule="stratification key",
            used_in="regional descriptive tables; completeness by macro-region"),
        # ---------------- socioeconomic confounders ----------------
        dict(
            variable="deprivation_share", source_system=SIDRA,
            raw_field="table 10296, variable 13604, classification 386 categories 9681 / 9680",
            accepted_values="population at or below 1/4 minimum wage per capita over population with an income class, 2022",
            transformation="SIDRA extract; the classification of interest is requested LAST and its category id taken as the final element of category_ids",
            missing_data_rule="municipalities with a zero income-class denominator yield null, not zero",
            aggregation_rule="summed numerators and denominators within H quintile, then the ratio",
            used_in="53_socioeconomic_confounders.py: is the gradient deprivation in disguise?"),
        dict(
            variable="median_pc_income", source_system=SIDRA,
            raw_field="table 10295, variable 13534",
            accepted_values="median per-capita household income, 2022, R$",
            transformation="SIDRA extract, municipality level",
            missing_data_rule="a suppressed or absent cell is null; the municipality count behind each band is reported",
            aggregation_rule="population-weighted mean of municipal medians within band (a weighted mean of medians, and labelled as such)",
            used_in="53_socioeconomic_confounders.py"),
        dict(
            variable="crowding_share", source_system=SIDRA,
            raw_field="table 9933, variable 381, classification 1975 categories 73090 / 73086",
            accepted_values="households with more than three residents per bedroom over all households, 2022",
            transformation="SIDRA extract",
            missing_data_rule="zero household denominator yields null",
            aggregation_rule="summed numerators and denominators within band, then the ratio",
            used_in="53_socioeconomic_confounders.py"),
        dict(
            variable="Census 2022 covariate time base", source_system=SIDRA, raw_field="(all three above)",
            accepted_values="single cross-section, 2022",
            transformation="none",
            missing_data_rule=(
                "STATED SIMPLIFICATION: a 2022 cross-section is compared against a "
                "2007-2025 outcome window. These are near-time-invariant municipal "
                "characteristics and the comparison is cross-sectional; it is not a "
                "time-varying adjustment and must not be read as one."),
            aggregation_rule="n/a",
            used_in="53_socioeconomic_confounders.py"),
        # ---------------- derived panel quantities ----------------
        dict(
            variable="H (exposure)", source_system=DERIVED, raw_field="ATE_HOSP",
            accepted_values="[0,1]",
            transformation="H = hospitalised / hosp_known within the aggregation unit",
            missing_data_rule=(
                "Denominator is cases with a VALID ate_hosp, not all cases. INVERSE "
                "indicator: higher H = narrower ascertainment (only severe illness "
                "found); lower H = broader. B = 1 - H is the breadth-oriented scale. "
                "The word 'profundidade' is not used: an earlier draft called H "
                "'profundidade de deteccao' and then described a high value as *menor* "
                "profundidade, which is self-contradictory."),
            aggregation_rule="counts summed first, ratio formed after; never a mean of sub-unit H values",
            used_in="the exposure of the whole study"),
        dict(
            variable="cfr (case fatality)", source_system=DERIVED, raw_field="EVOLUCAO",
            accepted_values="[0,1]",
            transformation="deaths / outcome_known",
            missing_data_rule="denominator is cases with a valid evolucao; obito_por_outras_causas is in the denominator but not the numerator",
            aggregation_rule="counts summed first; exact Clopper-Pearson interval (brepi.analysis.rates.binom_ci)",
            used_in="the outcome of the whole study"),
        dict(
            variable="incidence_per_100k", source_system=DERIVED,
            raw_field="CLASSI_FIN x POPSVS",
            accepted_values="rate per 100,000 person-years",
            transformation="1e5 * cases / person_years",
            missing_data_rule="a region-year with no population row yields null, never a zero denominator",
            aggregation_rule="counts and person-years summed first; exact Poisson (Garwood) interval via poisson_ci(events, person_time, scale=1e5)",
            used_in="burden description; the incidence-lethality inversion"),
        dict(
            variable="outcome_completeness, hosp_completeness, severity_completeness, criterion_completeness",
            source_system=DERIVED, raw_field="the four state columns",
            accepted_values="[0,1]",
            transformation="<field>_known / cases within the unit",
            missing_data_rule="carried in the panel per cell so that every proportion's own completeness is auditable without returning to the line level",
            aggregation_rule="counts summed first",
            used_in="this script; the missingness threat ledger"),
        dict(
            variable="hk_hosp, hk_nonhosp, d_hosp, d_nonhosp", source_system=DERIVED,
            raw_field="EVOLUCAO x ATE_HOSP",
            accepted_values="counts",
            transformation="cases with BOTH evolucao_state and ate_hosp_state valid, split by ate_hosp value and death",
            missing_data_rule="a case missing either field enters none of the four; the joint denominator is smaller than either marginal",
            aggregation_rule="cfr_hosp = d_hosp / hk_hosp; cfr_nonhosp = d_nonhosp / hk_nonhosp",
            used_in="the shared-denominator test: is the gradient a denominator effect?"),
        dict(
            variable="H_quintile (exposure band)", source_system=DERIVED, raw_field="H",
            accepted_values="1-5; Q1 = lowest H = broadest ascertainment",
            transformation="health regions sorted by H ascending, cumulative case share, ceil(share*5) clipped to [1,5]",
            missing_data_rule=(
                "Equal case mass, not equal region count. Health regions with fewer "
                "than 30 confirmed cases are excluded from banded analyses (they cannot "
                "support a case-fatality estimate) and the excluded mass is reported."),
            aggregation_rule="counts summed within band before any ratio",
            used_in="every stratified table in the study"),
    ]

    order = ["variable", "source_system", "raw_field", "accepted_values",
             "transformation", "missing_data_rule", "aggregation_rule", "used_in"]
    return pl.DataFrame(rows).select(order)


def record_checklist() -> pl.DataFrame:
    """Where each RECORD item is answered, or an honest 'not satisfied'.

    RECORD extends STROBE with items 1.1-1.3, 6.1-6.4, 7.1, 12.1-12.3, 13.1,
    19.1 and 22.1. This is a pointer table, not a substitute for the reporting:
    an item whose evidence does not exist is marked so rather than gestured at.
    """
    R = "data/results"
    rows = [
        ("1.1", "Data source(s) and type of data named in the title or abstract",
         "manuscript title/abstract", "author action"),
        ("1.2", "Geographic region and timeframe named", "manuscript title/abstract",
         "author action"),
        ("1.3", "Linkage between databases reported if applicable",
         f"{R}/triangulation/ (SINAN, SIM, SIH aligned by municipality of residence and year, "
         "not by person: no record linkage was performed and none is claimed)", "satisfied"),
        ("6.1", "Methods of study-population selection: codes and algorithms, with a "
                "validation reference or a statement that none exists",
         f"{R}/data_quality_record/field_dictionary.csv", "satisfied by this script"),
        ("6.2", "Codes and algorithms used to classify exposures, outcomes, confounders "
                "and effect modifiers, listed",
         f"{R}/data_quality_record/field_dictionary.csv", "satisfied by this script"),
        ("6.3", "Validation of the codes/algorithms, if done",
         f"{R}/triangulation/ (external comparison against SIM and SIH, two systems "
         "with different failure modes) and docs/CODEBOOK.md (registry reconciled "
         "against the published data dictionary)",
         "partially satisfied: no chart-review validation of SINAN case status exists "
         "for this study and none is claimed"),
        ("6.4", "Linkage quality evaluated", "no person-level linkage performed",
         "not applicable"),
        ("7.1", "Data cleaning methods described",
         "docs/CODEBOOK.md (four decode states); studies/leptospirosis/60_analysis_panel.py; "
         f"{R}/data_quality_record/field_dictionary.csv", "satisfied"),
        ("12.1", "Methods of cleaning, linkage and any data-quality restriction described",
         f"{R}/data_quality_record/completeness_by_*.csv and "
         "cfr_gradient_by_completeness_restriction.csv", "satisfied by this script"),
        ("12.2", "Sensitivity analyses of the database's fitness for purpose",
         f"{R}/data_quality_record/cfr_gradient_by_completeness_restriction.csv and "
         "cfr_extreme_bounds.csv", "satisfied by this script"),
        ("12.3", "Person-level, institutional or other data linkage described",
         "none performed; all aggregation is ecological, at health-region-year",
         "not applicable"),
        ("13.1", "Selection of included persons described, with a flow diagram",
         f"{R}/analysis_panel/study_flow.csv", "satisfied"),
        ("19.1", "Discussion of the implications of using data not created for research, "
                 "including misclassification, unmeasured confounding and missing data",
         f"{R}/data_quality_record/ (this analysis) -> manuscript discussion",
         "evidence produced; author must write it up"),
        ("22.1", "Statement on how to access the protocol, raw data and programming code",
         "manuscript data-availability section; the repository is the code",
         "author action"),
    ]
    return pl.DataFrame(
        rows, schema=["record_item", "requirement", "where_addressed", "status"],
        orient="row")


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    line = valid_flags(analytic_population())

    # Plausibility gate: this rebuild must be the same population the panel was
    # built from, or every completeness figure below describes a different
    # study. Checked against the panel rather than asserted.
    panel = pl.read_parquet(PANEL / "region_year_panel.parquet")
    assert line.height == int(panel["cases"].sum()), (
        f"rebuild has {line.height} cases, panel has {int(panel['cases'].sum())}")
    for col, panel_col in [("valid_outcome", "outcome_known"),
                           ("valid_hosp", "hosp_known"),
                           ("valid_severity", "sev_known"),
                           ("valid_criterio", "crit_known")]:
        assert int(line[col].sum()) == int(panel[panel_col].sum()), (
            f"{col} disagrees with the panel's {panel_col}")

    # ---------------- (a) completeness ----------------
    overall = pl.DataFrame([state_breakdown(line, m) for m in MEASURES])
    overall.write_csv(OUT / "completeness_overall.csv")

    by_year = completeness_by(line, ["year"])
    by_year.write_csv(OUT / "completeness_by_year.csv")
    by_macro = completeness_by(line, ["region"])
    by_macro.write_csv(OUT / "completeness_by_macroregion.csv")

    reg_all = region_frame(line)
    elig = case_weighted_quintiles(
        reg_all.filter((pl.col("cases") >= MIN_CASES) & (pl.col("valid_hosp") > 0)))
    reg_all.write_parquet(OUT / "region_completeness.parquet")

    qmap = elig.select("health_region_code", "H_quintile")
    line_q = line.join(qmap, on="health_region_code", how="left").with_columns(
        pl.col("H_quintile").fill_null(0))
    by_quintile = completeness_by(line_q, ["H_quintile"])
    # Band descriptors, so the quintile table can be read without a second file.
    desc = elig.group_by("H_quintile").agg(
        pl.len().alias("health_regions"), pl.col("cases").sum().alias("band_cases"),
        pl.col("H").median().alias("H_median"),
        pl.col("H").min().alias("H_min"), pl.col("H").max().alias("H_max"),
    ).sort("H_quintile")
    excluded = reg_all.filter((pl.col("cases") < MIN_CASES) | (pl.col("valid_hosp") == 0))
    desc = pl.concat([
        pl.DataFrame({"H_quintile": [0], "health_regions": [excluded.height],
                      "band_cases": [int(excluded["cases"].sum())],
                      "H_median": [float(excluded["H"].median())
                                   if excluded.height else None],
                      "H_min": [None], "H_max": [None]},
                     schema=desc.schema),
        desc,
    ])
    by_quintile = by_quintile.join(desc, on="H_quintile", how="left").sort(
        ["H_quintile", "measure"])
    by_quintile.write_csv(OUT / "completeness_by_H_quintile.csv")

    # ---------------- (b) the threat ----------------
    corr = completeness_vs_H(elig, reg_all.filter(pl.col("valid_hosp") > 0))
    corr.write_csv(OUT / "completeness_vs_H_spearman.csv")
    comp_col = VALID_COL["outcome (evolucao)"].replace("valid_", "comp_")
    size_check = spearman(elig[comp_col], elig["cases"])
    comp_vs_cfr = spearman(elig[comp_col], elig["cfr"])

    # ---------------- (c) sensitivity ----------------
    grad_tbl, grad_sum = restricted_gradients(elig)
    grad_tbl.write_csv(OUT / "cfr_gradient_by_completeness_restriction.csv")
    bounds_tbl, bounds = extreme_bounds(elig, line)
    adversarial = bounds.pop("_adversarial_table")
    pl.concat([
        bounds_tbl.select("scenario", "deaths_assumed", "denominator",
                          "cfr_pct", "cfr_lo", "cfr_hi", "level"),
    ]).write_csv(OUT / "cfr_extreme_bounds.csv")
    adversarial.write_csv(OUT / "cfr_adversarial_gradient_bounds.csv")

    # ---------------- (d) dictionary ----------------
    fd = field_dictionary()
    fd.write_csv(OUT / "field_dictionary.csv")
    rc = record_checklist()
    rc.write_csv(OUT / "record_checklist.csv")

    # ---------------- report ----------------
    outcome_rho = corr.filter(pl.col("measure") == "outcome (evolucao)").row(0, named=True)
    report = {
        "analytic_population": {
            "confirmed_cases": line.height,
            "health_regions_with_any_case": int(reg_all.height),
            "eligible_regions_min_cases": MIN_CASES,
            "eligible_regions": int(elig.height),
            "eligible_cases": int(elig["cases"].sum()),
            "eligible_case_share_pct": 100 * float(elig["cases"].sum()) / line.height,
        },
        "completeness_overall": overall.to_dicts(),
        "completeness_by_year_range": {
            m: {
                "min_pct": float(by_year.filter(pl.col("measure") == m)["pct_valid"].min()),
                "max_pct": float(by_year.filter(pl.col("measure") == m)["pct_valid"].max()),
                "first_year_pct": float(
                    by_year.filter(pl.col("measure") == m).sort("year")["pct_valid"][0]),
                "last_year_pct": float(
                    by_year.filter(pl.col("measure") == m).sort("year")["pct_valid"][-1]),
                "swing_pp": float(by_year.filter(pl.col("measure") == m)["pct_valid"].max()
                                  - by_year.filter(pl.col("measure") == m)["pct_valid"].min()),
            } for m in MEASURES
        },
        "completeness_vs_H_spearman": corr.to_dicts(),
        "outcome_completeness_vs_region_size_spearman": size_check,
        "outcome_completeness_vs_cfr_spearman": comp_vs_cfr,
        "threat_verdict": {
            "outcome_completeness_falls_as_H_rises": bool(
                outcome_rho["rho_eligible"] is not None and outcome_rho["rho_eligible"] < 0),
            "rho": outcome_rho["rho_eligible"], "p": outcome_rho["p_eligible"],
            "n_regions": outcome_rho["n_regions_eligible"],
        },
        "gradient_under_restriction": grad_sum,
        "national_case_fatality_bounds": bounds,
        "record_items_not_satisfied": rc.filter(
            pl.col("status").str.contains("author action|not applicable|partially")
        ).to_dicts(),
    }
    (OUT / "data_quality_record_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ---------------- console ----------------
    print("=== (a) completeness among the 66,358 confirmed cases ===")
    print(f"  {'measure':<36}{'valid':>8}{'%':>8}{'95% CI':>16}"
          f"{'missing':>9}{'unknown':>9}{'invalid':>8}")
    for r in overall.iter_rows(named=True):
        print(f"  {r['measure']:<36}{r['valid']:>8,}{r['pct_valid']:>8.2f}"
              f"  {r['pct_valid_lo']:>6.2f}-{r['pct_valid_hi']:<6.2f}"
              f"{r['missing']:>9,}{r['unknown']:>9,}{r['invalid']:>8,}")

    print("\n  by year (range over 2007-2025):")
    for m in MEASURES:
        y = report["completeness_by_year_range"][m]
        print(f"    {m:<36}{y['first_year_pct']:>6.1f}% (2007) -> "
              f"{y['last_year_pct']:>5.1f}% (2025)  range "
              f"{y['min_pct']:.1f}-{y['max_pct']:.1f}%  swing {y['swing_pp']:.1f} pp")

    print("\n  by macro-region (outcome field):")
    om = by_macro.filter(pl.col("measure") == "outcome (evolucao)").sort("pct_valid")
    for r in om.iter_rows(named=True):
        print(f"    {r['region']:<14}{r['pct_valid']:>6.2f}%  "
              f"({r['valid']:,} of {r['denominator_confirmed_cases']:,})")

    print("\n  by H quintile (band 0 = regions excluded from banded analyses):")
    print(f"  {'band':>5}{'regions':>9}{'cases':>9}{'H med':>8}"
          + "".join(f"{m.split(' (')[0][:11]:>13}" for m in MEASURES))
    for b in sorted(by_quintile["H_quintile"].unique().to_list()):
        t = by_quintile.filter(pl.col("H_quintile") == b)
        r0 = t.row(0, named=True)
        vals = {r["measure"]: r["pct_valid"] for r in t.iter_rows(named=True)}
        hm = f"{r0['H_median']:.3f}" if r0["H_median"] is not None else "  -  "
        print(f"  {b:>5}{r0['health_regions']:>9}{r0['band_cases']:>9,}{hm:>8}"
              + "".join(f"{vals[m]:>12.2f}%" for m in MEASURES))

    print("\n=== (b) does completeness track the exposure? Spearman across health regions ===")
    for r in corr.iter_rows(named=True):
        flag = "  <- shares a denominator with H, partly mechanical" if r["shares_denominator_with_H"] else ""
        print(f"  {r['measure']:<36}rho={r['rho_eligible']:+.3f} "
              f"(p={r['p_eligible']:.4f}, n={r['n_regions_eligible']}){flag}")
    print(f"  outcome completeness vs region case count: rho="
          f"{size_check['rho']:+.3f} (p={size_check['p']:.4f}, n={size_check['n']})")
    print(f"  outcome completeness vs case fatality:     rho="
          f"{comp_vs_cfr['rho']:+.3f} (p={comp_vs_cfr['p']:.4f}, n={comp_vs_cfr['n']})")
    v = report["threat_verdict"]
    print(f"  VERDICT: outcome completeness is "
          f"{'LOWER' if v['outcome_completeness_falls_as_H_rises'] else 'HIGHER'} "
          f"where H is higher (rho={v['rho']:+.3f}, n={v['n_regions']}). "
          f"The threat is {'REAL in direction' if v['outcome_completeness_falls_as_H_rises'] else 'not present'}.")

    print("\n=== (c) does the case-fatality gradient survive restriction? ===")
    for variant in grad_tbl["variant"].unique(maintain_order=True):
        t = grad_tbl.filter(pl.col("variant") == variant)
        cells = "  ".join(f"Q{r['H_quintile']} {r['cfr_pct']:5.2f}%"
                          for r in t.iter_rows(named=True))
        s = grad_sum[variant]
        print(f"  {variant:<52}{cells}")
        print(f"  {'':<52}ratio {s['ratio_high_over_low']:.1f}x, "
              f"{'disjoint' if s['intervals_disjoint'] else 'OVERLAPPING'}, "
              f"{s['cases_retained']:,} cases, "
              f"{sum(s['health_regions'])} regions")

    print("\n  national case fatality under extreme assumptions:")
    for r in bounds_tbl.iter_rows(named=True):
        print(f"    {r['scenario']:<50}{r['cfr_pct']:>6.2f}% "
              f"({r['cfr_lo']:.2f}-{r['cfr_hi']:.2f}), n={r['denominator']:,}")

    print("\n  the decisive test - the gradient under adversarial reallocation:")
    for r in adversarial.iter_rows(named=True):
        print(f"    {r['scenario']:<58}Q1 {r['q_low_cfr_pct']:5.2f}%  "
              f"Q5 {r['q_high_cfr_pct']:5.2f}%  ratio "
              f"{r['ratio_high_over_low']:.2f}x  "
              f"{'ordering holds' if r['ordering_preserved'] else 'ORDERING BREAKS'}, "
              f"{'disjoint' if r['intervals_disjoint'] else 'overlapping'}")

    print(f"\n=== (d) field dictionary: {fd.height} analytic variables documented ===")
    print(f"    RECORD checklist: {rc.height} items, "
          f"{rc.filter(pl.col('status').str.starts_with('satisfied')).height} satisfied by "
          "existing evidence")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
