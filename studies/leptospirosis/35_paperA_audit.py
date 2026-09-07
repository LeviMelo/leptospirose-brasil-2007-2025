"""Paper A: residence-aligned cross-system surveillance audit.

This is the paper-facing derivation for the ascertainment manuscript.  It does
not estimate an unobserved national case count.  Public SINAN, SIM and SIH
files cannot be linked person to person, so the identified estimand is
*municipal cross-system discordance*: persistent zero confirmed notifications
in SINAN while SIM or SIH contains an A27 record for residents of the same
municipality during the common 2008--2024 window.

The script deliberately escalates the evidence definition.  A single billing
code and repeated, principal-diagnosis, mortality and dual-system evidence do
not carry the same evidentiary weight. Reporting progressively restrictive
audit-priority tiers makes that heterogeneity visible without claiming that
diagnostic validity was adjudicated against medical records.

Writes rich, presentation-neutral products to ``data/results/paperA_audit``.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice

START, END = 2008, 2024
OUT = PATHS.results / "paperA_audit"
PROXY_2013 = {"1504752", "4212650", "4220000", "4314548", "5006275"}
CREATION_YEAR = {code: 2013 for code in PROXY_2013}


def _classify(sinan: pl.Expr, other: pl.Expr) -> pl.Expr:
    return (
        pl.when((sinan > 0) & (other > 0))
        .then(pl.lit("both source groups observed"))
        .when((sinan > 0) & (other == 0)).then(pl.lit("SINAN only"))
        .when((sinan == 0) & (other > 0)).then(pl.lit("cross-system discordant"))
        .otherwise(pl.lit("unobserved in all three"))
    )


def municipality_table(tri: pl.DataFrame) -> pl.DataFrame:
    d = tri.filter(pl.col("year").is_between(START, END)).with_columns(
        pl.when(pl.col("munic_code").is_in(PROXY_2013) & (pl.col("year") < 2013))
        .then(None)
        .otherwise(pl.col("population"))
        .alias("active_population")
    )
    return (
        d.group_by("munic_code", "name", "uf_abbr", "region")
        .agg(
            pl.col("active_population").mean().alias("mean_population"),
            pl.col("active_population").is_not_null().sum().alias("observation_years"),
            pl.col("sinan_confirmed").sum().alias("sinan_confirmed"),
            pl.col("sinan_deaths").sum().alias("sinan_deaths"),
            pl.col("sim_a27_deaths").sum().alias("sim_a27_deaths"),
            pl.col("sih_a27_admissions").sum().alias("sih_a27_admissions"),
            pl.col("sih_a27_principal").sum().alias("sih_a27_principal"),
            pl.col("sih_a27_deaths_in_hospital").sum().alias("sih_deaths"),
            (pl.col("sim_a27_deaths") > 0).sum().alias("sim_positive_years"),
            (pl.col("sih_a27_admissions") > 0).sum().alias("sih_positive_years"),
            (
                (pl.col("sim_a27_deaths") > 0)
                | (pl.col("sih_a27_admissions") > 0)
            ).sum().alias("comparison_positive_years"),
        )
        .with_columns(
            (pl.col("sim_a27_deaths") + pl.col("sih_a27_admissions"))
            .alias("comparison_records"),
            (pl.col("sim_a27_deaths") + pl.col("sih_a27_principal"))
            .alias("strict_comparison_records"),
            pl.col("munic_code").is_in(PROXY_2013).alias("created_2013_proxy"),
            pl.when(pl.col("munic_code").is_in(PROXY_2013))
            .then(pl.lit(2013)).otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("creation_year"),
        )
        .with_columns(
            _classify(pl.col("sinan_confirmed"), pl.col("comparison_records"))
            .alias("surveillance_class")
        )
        .sort("munic_code")
    )


def evidence_tiers(m: pl.DataFrame) -> pl.DataFrame:
    silent = m.filter(pl.col("sinan_confirmed") == 0)
    tiers = [
        ("any A27 record in SIM or SIH", pl.col("comparison_records") >= 1),
        ("SIM underlying cause or SIH principal diagnosis",
         pl.col("strict_comparison_records") >= 1),
        ("at least two comparison-system records",
         pl.col("comparison_records") >= 2),
        ("comparison-system evidence in at least two calendar years",
         pl.col("comparison_positive_years") >= 2),
        ("SIM underlying-cause death", pl.col("sim_a27_deaths") >= 1),
        ("both SIM and SIH positive",
         (pl.col("sim_a27_deaths") >= 1) & (pl.col("sih_a27_admissions") >= 1)),
        ("SIH in-hospital death", pl.col("sih_deaths") >= 1),
        ("primary definition excluding 2013 successor proxies",
         (pl.col("comparison_records") >= 1) & ~pl.col("created_2013_proxy")),
    ]
    rows = []
    for order, (label, expr) in enumerate(tiers, 1):
        x = silent.filter(expr)
        rows.append({
            "order": order,
            "definition": label,
            "municipalities": x.height,
            "population": float(x["mean_population"].sum()),
            "sim_deaths": int(x["sim_a27_deaths"].sum()),
            "sih_admissions": int(x["sih_a27_admissions"].sum()),
            "comparison_records": int(x["comparison_records"].sum()),
        })
    return pl.DataFrame(rows)


def by_horizon(tri: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, int]] = []
    for end in range(START, END + 1):
        d = tri.filter(pl.col("year").is_between(START, end)).group_by(
            "munic_code"
        ).agg(
            pl.col("sinan_confirmed").sum().alias("sinan"),
            (pl.col("sim_a27_deaths").sum()
             + pl.col("sih_a27_admissions").sum()).alias("other"),
        ).with_columns(_classify(pl.col("sinan"), pl.col("other")).alias("class"))
        counts = {r["class"]: r["len"] for r in d.group_by("class").len().to_dicts()}
        rows.append({"end_year": end, "years_observed": end - START + 1,
                     **{k: counts.get(k, 0) for k in (
                         "both source groups observed", "SINAN only", "cross-system discordant",
                         "unobserved in all three")}})
    return pl.DataFrame(rows)


def transition_matrix(
    tri: pl.DataFrame, *, include_future_units: bool = False,
) -> pl.DataFrame:
    """Exact destination of each historically valid 2008 class.

    The primary matrix excludes municipalities created after baseline.
    ``include_future_units`` exists only to reproduce the fixed-2022-lattice
    sensitivity in which non-existence had formerly been labelled a zero.
    """
    def at_end(end: int, name: str) -> pl.DataFrame:
        d = tri.filter(pl.col("year").is_between(START, end))
        if not include_future_units:
            d = d.filter(~pl.col("munic_code").is_in(PROXY_2013))
        return (
            d
            .group_by("munic_code")
            .agg(
                pl.col("sinan_confirmed").sum().alias("sinan"),
                (pl.col("sim_a27_deaths").sum()
                 + pl.col("sih_a27_admissions").sum()).alias("other"),
            )
            .with_columns(_classify(pl.col("sinan"), pl.col("other")).alias(name))
            .select("munic_code", name)
        )
    return (
        at_end(START, "class_2008")
        .join(at_end(END, "class_2024"), on="munic_code", how="inner")
        .group_by("class_2008", "class_2024").len().rename({"len": "municipalities"})
        .with_columns(
            (pl.col("municipalities")
             / pl.col("municipalities").sum().over("class_2008"))
            .alias("origin_share")
        )
        .sort("class_2008", "class_2024")
    )


def zero_cohort_history(
    tri: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Follow municipalities with no signal in any system in baseline 2008.

    The first signal is a municipality-level competing event: SINAN first,
    SIM/SIH first, or both in the same calendar year. This retains the temporal
    ordering erased by an endpoint transition diagram and separates places that
    remain unobserved from those whose first evidence came from another system.
    """
    d = (
        tri.filter(pl.col("year").is_between(START, END))
        .with_columns(
            (pl.col("sim_a27_deaths") + pl.col("sih_a27_admissions"))
            .alias("other_records")
        )
    )
    baseline = (
        d.filter(pl.col("year") == START)
        .filter(~pl.col("munic_code").is_in(PROXY_2013))
        .filter(
            (pl.col("sinan_confirmed") == 0)
            & (pl.col("other_records") == 0)
        )
        .select(
            "munic_code", "name", "uf_abbr", "region",
            pl.col("population").alias("baseline_population"),
        )
    )
    first = (
        d.filter(pl.col("year") > START)
        .group_by("munic_code")
        .agg(
            pl.col("year").filter(pl.col("sinan_confirmed") > 0)
            .min().alias("first_sinan_year"),
            pl.col("year").filter(pl.col("other_records") > 0)
            .min().alias("first_comparison_year"),
        )
    )
    endpoint = (
        d.group_by("munic_code")
        .agg(
            pl.col("sinan_confirmed").sum().alias("sinan_total"),
            pl.col("other_records").sum().alias("comparison_total"),
        )
        .with_columns(
            _classify(pl.col("sinan_total"), pl.col("comparison_total"))
            .alias("endpoint_class")
        )
    )
    cohort = (
        baseline.join(first, on="munic_code", how="left")
        .join(endpoint, on="munic_code", how="left")
        .with_columns(
            pl.min_horizontal("first_sinan_year", "first_comparison_year")
            .alias("first_signal_year"),
            pl.when(
                pl.col("first_sinan_year").is_null()
                & pl.col("first_comparison_year").is_null()
            ).then(pl.lit("remained zero"))
            .when(
                pl.col("first_comparison_year").is_null()
                | (pl.col("first_sinan_year") < pl.col("first_comparison_year"))
            ).then(pl.lit("SINAN first"))
            .when(
                pl.col("first_sinan_year").is_null()
                | (pl.col("first_comparison_year") < pl.col("first_sinan_year"))
            ).then(pl.lit("SIM/SIH first"))
            .otherwise(pl.lit("same year"))
            .alias("first_signal_type"),
        )
        .sort("munic_code")
    )

    annual_rows = []
    for year in range(START, END + 1):
        annual_rows.append({
            "year": year,
            "cohort_municipalities": cohort.height,
            "still_zero": cohort.filter(
                pl.col("first_signal_year").is_null()
                | (pl.col("first_signal_year") > year)
            ).height,
            "cumulative_sinan_first": cohort.filter(
                (pl.col("first_signal_type") == "SINAN first")
                & (pl.col("first_signal_year") <= year)
            ).height,
            "cumulative_comparison_first": cohort.filter(
                (pl.col("first_signal_type") == "SIM/SIH first")
                & (pl.col("first_signal_year") <= year)
            ).height,
            "cumulative_same_year": cohort.filter(
                (pl.col("first_signal_type") == "same year")
                & (pl.col("first_signal_year") <= year)
            ).height,
            "new_first_signals": cohort.filter(
                pl.col("first_signal_year") == year
            ).height,
        })
    annual = pl.DataFrame(annual_rows).with_columns(
        (pl.col("still_zero") / pl.col("cohort_municipalities"))
        .alias("still_zero_share")
    )
    states = (
        cohort.group_by("uf_abbr", "region")
        .agg(
            pl.len().alias("cohort_municipalities"),
            (pl.col("first_signal_type") == "remained zero")
            .sum().alias("remained_zero"),
            (pl.col("first_signal_type") == "SINAN first")
            .sum().alias("sinan_first"),
            (pl.col("first_signal_type") == "SIM/SIH first")
            .sum().alias("comparison_first"),
            (pl.col("first_signal_type") == "same year")
            .sum().alias("same_year"),
            pl.col("first_signal_year").quantile(0.25)
            .alias("first_signal_year_q25"),
            pl.col("first_signal_year").median().alias("median_first_signal_year"),
            pl.col("first_signal_year").quantile(0.75)
            .alias("first_signal_year_q75"),
        )
        .with_columns(
            (pl.col("remained_zero") / pl.col("cohort_municipalities"))
            .alias("remained_zero_share")
        )
        .sort("remained_zero_share", descending=True)
    )
    pathways = (
        cohort.group_by("first_signal_type", "endpoint_class")
        .len().rename({"len": "municipalities"})
        .sort("first_signal_type", "endpoint_class")
    )
    return cohort, annual, states, pathways


def source_derivation(
    tri: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Document record flow, coding position, identifiers and annual totals.

    The public extracts are record censuses rather than linked people.  This
    audit therefore reports every filter and avoids pretending that source-row
    deduplication is person deduplication.
    """
    line = pl.read_parquet(PATHS.interim / "lept_line_level.parquet")
    confirmed = pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1"
    onset = pl.col("DT_SIN_PRI").cast(pl.Utf8).str.to_date(strict=False)
    sinan_confirmed = line.filter(confirmed).with_columns(onset.alias("_event_date"))
    sinan_dated = sinan_confirmed.filter(pl.col("_event_date").is_not_null())
    sinan_window = sinan_dated.filter(
        pl.col("_event_date").is_between(date(START, 1, 1), date(END, 12, 31))
    )
    sinan_valid = sinan_window.filter(
        (pl.col("municipality_residence_state") == "valid")
        & pl.col("municipality_residence_code7").is_not_null()
    )

    sim_raw = pl.read_parquet(PATHS.interim / "sim_a27_deaths.parquet")
    sim_year = (
        pl.col("DTOBITO").cast(pl.Utf8).str.strip_chars().str.slice(4, 4)
        .cast(pl.Int32, strict=False)
    )
    # Death date is the event-time contract. A source-file-year substitution
    # could silently turn a malformed date into an apparently valid event.
    sim_window = sim_raw.with_columns(
        sim_year.alias("_event_year")
    ).filter(pl.col("_event_year").is_between(START, END))
    sim_resolved = lattice.resolve_municipality_code(
        sim_window, [("CODMUNRES", "CODMUNRES")], lattice_year=2022
    )
    sim_valid = sim_resolved.filter(pl.col("munic_code_status") == "resolved")

    sih_raw = pl.read_parquet(PATHS.interim / "sih_a27_admissions.parquet")
    sih_window = sih_raw.with_columns(
        pl.col("DT_INTER").cast(pl.Utf8).str.to_date(
            "%Y%m%d", strict=False
        ).alias("_admission_date")
    ).with_columns(
        pl.col("_admission_date").dt.year().alias("_event_year")
    ).filter(pl.col("_event_year").is_between(START, END))
    sih_resolved = lattice.resolve_municipality_code(
        sih_window, [("MUNIC_RES", "MUNIC_RES")], lattice_year=2022
    )
    sih_valid = sih_resolved.filter(pl.col("munic_code_status") == "resolved")
    principal = (
        pl.col("DIAG_PRINC").cast(pl.Utf8).str.strip_chars().str.starts_with("A27")
        .fill_null(False)
    )
    secondary = (
        pl.col("DIAG_SECUN").cast(pl.Utf8).str.strip_chars().str.starts_with("A27")
        .fill_null(False)
    )
    sih_position = sih_valid.select(
        principal.sum().alias("principal_a27"),
        ((~principal) & secondary).sum().alias("secondary_only_a27"),
        (~pl.col("DIAG_SECUN").cast(pl.Utf8).str.strip_chars()
         .is_in([None, "", "0000"])).sum().alias("secondary_field_informative"),
        pl.len().alias("analytic_admissions"),
    )

    flow_rows: list[dict[str, object]] = []
    def add(source: str, order: int, stage: str, records: int, detail: str) -> None:
        prior = next(
            (int(x["records"]) for x in reversed(flow_rows) if x["source"] == source),
            records,
        )
        flow_rows.append({
            "source": source, "stage_order": order, "stage": stage,
            "records": records, "excluded_from_previous": prior - records,
            "detail": detail,
        })

    add("SINAN", 1, "Downloaded investigation records, 2007-2025", line.height,
        "LEPT annual final files (2025 preliminary), one public source row per investigation")
    add("SINAN", 2, "Records classified as confirmed", sinan_confirmed.height,
        "CLASSI_FIN=1")
    add("SINAN", 3, "Confirmed records with a valid onset date", sinan_dated.height,
        "DT_SIN_PRI parsed as a calendar date; no date substitution")
    add("SINAN", 4, "Confirmed records with onset in 2008-2024", sinan_window.height,
        "Records outside the common study window excluded")
    add("SINAN", 5, "Confirmed records with valid residence", sinan_valid.height,
        "ID_MN_RESI resolved to the 2022 5,570-code reference; no geographic fallback")
    add("SIM", 1, "Downloaded A27 underlying-cause deaths, 2007-2024", sim_raw.height,
        "A27 prefix in CAUSABAS; final DORES files")
    add("SIM", 2, "Deaths dated 2008-2024", sim_window.height,
        "Year from DTOBITO; no source-file-year substitution")
    add("SIM", 3, "Deaths with valid residence", sim_valid.height,
        "CODMUNRES only; CODMUNOCOR is never a fallback")
    add("SIH", 1, "Downloaded A27 admissions, 2008-2024", sih_raw.height,
        "A27 prefix in DIAG_PRINC or DIAG_SECUN in RD files")
    add("SIH", 2, "Admissions beginning in 2008-2024", sih_window.height,
        "Year from DT_INTER; 153 admissions beginning in 2007 were excluded")
    add("SIH", 3, "Admissions with valid residence", sih_valid.height,
        "MUNIC_RES only; MUNIC_MOV is never a fallback")
    flow = pl.DataFrame(flow_rows)

    src = tri.filter(pl.col("year").is_between(START, END))
    annual = (
        src.group_by("year").agg(
            pl.col("sinan_confirmed").sum().alias("sinan_confirmed"),
            pl.col("sim_a27_deaths").sum().alias("sim_a27_underlying_deaths"),
            pl.col("sih_a27_admissions").sum().alias("sih_a27_any_position"),
            pl.col("sih_a27_principal").sum().alias("sih_a27_principal"),
            pl.col("sih_a27_deaths_in_hospital").sum().alias("sih_in_hospital_deaths"),
        ).sort("year")
    )
    summary = pl.DataFrame([
        {
            "source": "SINAN", "analytic_records": sinan_valid.height,
            "municipalities_represented": src.filter(pl.col("sinan_confirmed") > 0)["munic_code"].n_unique(),
            "missing_or_invalid_residence": sinan_window.height - sinan_valid.height,
            "record_unit": "investigation record classified as confirmed",
        },
        {
            "source": "SIM", "analytic_records": sim_valid.height,
            "municipalities_represented": src.filter(pl.col("sim_a27_deaths") > 0)["munic_code"].n_unique(),
            "missing_or_invalid_residence": sim_window.height - sim_valid.height,
            "record_unit": "death certificate with A27 as underlying cause",
        },
        {
            "source": "SIH", "analytic_records": sih_valid.height,
            "municipalities_represented": src.filter(pl.col("sih_a27_admissions") > 0)["munic_code"].n_unique(),
            "missing_or_invalid_residence": sih_window.height - sih_valid.height,
            "record_unit": "AIH admission with A27 in principal or secondary diagnosis",
        },
    ])
    ids = sih_valid.with_columns(
        pl.col("N_AIH").cast(pl.Utf8).str.strip_chars().alias("_aih")
    ).filter(pl.col("_aih").is_not_null() & (pl.col("_aih") != ""))
    position = sih_position.with_columns(
        pl.lit(ids.height).alias("aih_identifier_nonmissing"),
        pl.lit(ids["_aih"].n_unique()).alias("unique_aih_identifiers"),
        pl.lit(ids.height - ids["_aih"].n_unique()).alias("repeated_aih_identifier_rows"),
    )
    return flow, summary, annual, position


def source_contracts() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Paper-facing source contract and immutable local retrieval manifest."""
    contracts = pl.DataFrame([
        {
            "source": "SINAN", "file_family": "LEPTBRYY.DBC annual files",
            "retrieved": "2026-07-30", "study_files": "2008-2024 final",
            "record_definition": "CLASSI_FIN=1 in the LEPT disease-specific extract",
            "event_time": "DT_SIN_PRI (symptom-onset date); missing/invalid dates excluded",
            "residence_field": "ID_MN_RESI only",
            "duplicate_handling": (
                "No person or episode deduplication: the public analytical extract has no "
                "validated cross-year linkage key; each retained row is an investigation record"
            ),
        },
        {
            "source": "SIM", "file_family": "DORES state-year final files",
            "retrieved": "2026-07-30 to 2026-07-31", "study_files": "2008-2024 final",
            "record_definition": "A27 prefix in CAUSABAS (underlying cause)",
            "event_time": "DTOBITO; invalid death dates excluded without substitution",
            "residence_field": "CODMUNRES only",
            "duplicate_handling": (
                "No person deduplication; the projected public extract lacks a stable death-"
                "certificate identifier and each retained row is a death-certificate record"
            ),
        },
        {
            "source": "SIH", "file_family": "RD state-month reduced-AIH files",
            "retrieved": "2026-07-31", "study_files": "2008-01 to 2024-12",
            "record_definition": (
                "A27 prefix in DIAG_PRINC, plus legacy DIAG_SECUN where informative; "
                "the primary restriction uses DIAG_PRINC"
            ),
            "event_time": "DT_INTER (admission date); no competence-year fallback",
            "residence_field": "MUNIC_RES only",
            "duplicate_handling": (
                "No person or episode deduplication; N_AIH was unique among retained rows, but "
                "different AIHs can represent transfers, readmissions, continuations, or one person"
            ),
        },
    ])
    rows: list[dict[str, object]] = []
    roots = [PATHS.cache / "datasus" / "sim", PATHS.cache / "datasus" / "sih"]
    for root in roots:
        for path in root.rglob("*.provenance.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            key = str(item.get("key", ""))
            if "/sim/do/" in key:
                source = "SIM"
            elif "/sih/rd/" in key:
                source = "SIH"
            else:
                continue
            rows.append({
                "source": source, "key": key, "uri": item.get("uri"),
                "fetched_at": item.get("fetched_at"),
                "remote_modified": item.get("remote_modified"),
                "bytes": item.get("bytes"), "sha256": item.get("sha256"),
            })
    sinan_manifest = json.loads(
        (PATHS.cache / "_manifest_sinan_LEPT_20260730.json").read_text(encoding="utf-8")
    )
    for item in sinan_manifest["artefacts"]:
        rows.append({
            "source": "SINAN", "key": item.get("key"), "uri": item.get("uri"),
            "fetched_at": item.get("fetched_at"),
            "remote_modified": item.get("remote_modified"),
            "bytes": item.get("bytes"), "sha256": item.get("sha256"),
        })
    return contracts, pl.DataFrame(rows).sort("source", "key")


def surveillance_cascade(
    tri: pl.DataFrame, municipalities: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Locate confirmed-record zeros within the full SINAN process.

    Every row in the disease-specific SINAN extract is already an investigation.
    The cascade therefore distinguishes absence before investigation from loss
    at final classification. It does not interpret a discarded or inconclusive
    investigation as a true case.
    """
    line = pl.read_parquet(PATHS.interim / "lept_line_level.parquet")
    inv = (
        line.with_columns(
            pl.col("DT_SIN_PRI").cast(pl.Utf8).str.to_date(strict=False)
            .alias("_onset"),
            pl.col("DT_ENCERRA").cast(pl.Utf8).str.to_date(strict=False)
            .alias("_closure"),
            pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars().alias("_class"),
            pl.col("CRITERIO").cast(pl.Utf8).str.strip_chars().alias("_criterion"),
        )
        .filter(
            pl.col("_onset").is_between(date(START, 1, 1), date(END, 12, 31))
            & (pl.col("municipality_residence_state") == "valid")
            & pl.col("municipality_residence_code7").is_not_null()
        )
        .with_columns(
            pl.col("municipality_residence_code7").alias("munic_code"),
            pl.col("_onset").dt.year().alias("year"),
        )
    )
    annual = inv.group_by("munic_code", "year").agg(
        pl.len().alias("investigations"),
        (pl.col("_class") == "1").sum().alias("confirmed_investigations"),
        (pl.col("_class") == "2").sum().alias("discarded_investigations"),
        (pl.col("_class") == "8").sum().alias("inconclusive_investigations"),
        pl.col("_class").is_null().sum().alias("unclassified_investigations"),
        pl.col("_closure").is_not_null().sum().alias("closure_date_recorded"),
        (pl.col("_criterion") == "1").sum().alias("laboratory_criterion_records"),
        (pl.col("_criterion") == "2").sum().alias("clinical_epidemiological_records"),
        (pl.col("evolucao_state") == "valid").sum().alias("outcome_recorded"),
    )
    total = annual.group_by("munic_code").agg(
        pl.col("investigations").sum(),
        pl.col("confirmed_investigations").sum(),
        pl.col("discarded_investigations").sum(),
        pl.col("inconclusive_investigations").sum(),
        pl.col("unclassified_investigations").sum(),
        pl.col("closure_date_recorded").sum(),
        pl.col("laboratory_criterion_records").sum(),
        pl.col("clinical_epidemiological_records").sum(),
        pl.col("outcome_recorded").sum(),
        pl.col("year").min().alias("first_investigation_year"),
        pl.col("year").max().alias("latest_investigation_year"),
        (pl.col("investigations") > 0).sum().alias("investigation_years"),
    )
    comp_years = (
        tri.filter(pl.col("year").is_between(START, END))
        .with_columns(
            (pl.col("sim_a27_deaths") + pl.col("sih_a27_admissions"))
            .alias("comparison_records_year")
        )
        .filter(pl.col("comparison_records_year") > 0)
        .select("munic_code", "year", "comparison_records_year")
    )
    same_year = (
        annual.join(comp_years, on=["munic_code", "year"], how="inner")
        .group_by("munic_code")
        .agg(
            pl.col("investigations").sum().alias("investigations_in_comparison_years"),
            pl.len().alias("years_with_investigation_and_comparison_record"),
        )
    )
    count_cols = [
        "investigations", "confirmed_investigations", "discarded_investigations",
        "inconclusive_investigations", "unclassified_investigations",
        "closure_date_recorded", "laboratory_criterion_records",
        "clinical_epidemiological_records", "outcome_recorded",
        "investigation_years", "investigations_in_comparison_years",
        "years_with_investigation_and_comparison_record",
    ]
    cascade = municipalities.join(total, on="munic_code", how="left").join(
        same_year, on="munic_code", how="left"
    ).with_columns([pl.col(c).fill_null(0) for c in count_cols]).with_columns(
        pl.when(pl.col("investigations") == 0)
        .then(pl.lit("no SINAN investigation"))
        .when(pl.col("discarded_investigations") == pl.col("investigations"))
        .then(pl.lit("investigated; all classified discarded"))
        .otherwise(pl.lit("investigated; inconclusive or unclassified present"))
        .alias("confirmed_zero_process_state"),
        (pl.col("years_with_investigation_and_comparison_record") > 0)
        .alias("investigation_in_comparison_year"),
    )
    discordant = cascade.filter(
        pl.col("surveillance_class") == "cross-system discordant"
    )
    summary = (
        discordant.group_by("confirmed_zero_process_state")
        .agg(
            pl.len().alias("municipalities"),
            pl.col("investigations").sum().alias("all_investigation_records"),
            pl.col("investigation_in_comparison_year").sum()
            .alias("municipalities_with_same_year_investigation"),
            pl.col("comparison_records").sum().alias("comparison_records"),
        )
        .with_columns(
            (pl.col("municipalities") / discordant.height)
            .alias("share_of_discordant_municipalities")
        )
        .sort("municipalities", descending=True)
    )
    return cascade, summary


def comparison_record_phenotypes(
    municipalities: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Summarise clinical-administrative features without claiming validity."""
    discordant_codes = municipalities.filter(
        pl.col("surveillance_class") == "cross-system discordant"
    ).select("munic_code")

    sih_raw = pl.read_parquet(PATHS.interim / "sih_a27_admissions.parquet")
    sih = lattice.resolve_municipality_code(
        sih_raw, [("MUNIC_RES", "MUNIC_RES")], lattice_year=2022
    ).with_columns(
        lattice.code6_to_code7_expr(pl.col("munic_code6")).alias("munic_code"),
        pl.col("DT_INTER").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        .alias("_admit"),
        pl.col("DT_SAIDA").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        .alias("_discharge"),
        pl.col("DIAG_PRINC").cast(pl.Utf8).str.strip_chars().alias("_principal_code"),
        pl.col("DIAG_SECUN").cast(pl.Utf8).str.strip_chars().alias("_secondary_code"),
        pl.col("DIAS_PERM").cast(pl.Int32, strict=False).alias("_los"),
        pl.col("UTI_MES_TO").cast(pl.Float64, strict=False).alias("_icu_days"),
    ).filter(
        (pl.col("munic_code_status") == "resolved")
        & pl.col("_admit").dt.year().is_between(START, END)
    ).with_columns(
        pl.col("_admit").dt.year().alias("_year"),
        pl.col("_principal_code").str.starts_with("A27").fill_null(False)
        .alias("_principal_a27"),
        ((~pl.col("_principal_code").str.starts_with("A27").fill_null(False))
         & pl.col("_secondary_code").str.starts_with("A27").fill_null(False))
        .alias("_secondary_only_a27"),
        (pl.col("MORTE").cast(pl.Utf8).str.strip_chars() == "1")
        .fill_null(False).alias("_died"),
    ).join(discordant_codes, on="munic_code", how="inner")

    sih_m = sih.group_by("munic_code").agg(
        pl.len().alias("sih_admissions"),
        pl.col("_principal_a27").sum().alias("sih_principal_a27"),
        pl.col("_secondary_only_a27").sum().alias("sih_secondary_only_a27"),
        pl.col("_died").sum().alias("sih_in_hospital_deaths"),
        (pl.col("_icu_days") > 0).sum().alias("sih_with_icu"),
        (pl.col("_los") >= 7).sum().alias("sih_stay_7plus_days"),
        pl.col("_los").median().alias("sih_median_length_of_stay"),
        pl.col("_year").max().alias("latest_sih_year"),
        (pl.col("_year") >= 2022).sum().alias("sih_admissions_2022_2024"),
        (pl.col("_principal_code") == "A270").sum().alias("sih_a270"),
        (pl.col("_principal_code") == "A278").sum().alias("sih_a278"),
        (pl.col("_principal_code") == "A279").sum().alias("sih_a279"),
    )

    sim_raw = pl.read_parquet(PATHS.interim / "sim_a27_deaths.parquet")
    sim = lattice.resolve_municipality_code(
        sim_raw, [("CODMUNRES", "CODMUNRES")], lattice_year=2022
    ).with_columns(
        lattice.code6_to_code7_expr(pl.col("munic_code6")).alias("munic_code"),
        pl.col("DTOBITO").cast(pl.Utf8).str.to_date("%d%m%Y", strict=False)
        .alias("_death_date"),
        pl.col("CAUSABAS").cast(pl.Utf8).str.strip_chars().alias("_cause"),
    ).filter(
        (pl.col("munic_code_status") == "resolved")
        & pl.col("_death_date").dt.year().is_between(START, END)
    ).with_columns(
        pl.col("_death_date").dt.year().alias("_year")
    ).join(discordant_codes, on="munic_code", how="inner")
    sim_m = sim.group_by("munic_code").agg(
        pl.len().alias("sim_deaths"),
        (pl.col("LOCOCOR").cast(pl.Utf8).str.strip_chars().is_in(["1", "2"]))
        .sum().alias("sim_deaths_in_health_facility"),
        (pl.col("ASSISTMED").cast(pl.Utf8).str.strip_chars() == "1")
        .sum().alias("sim_deaths_with_medical_assistance"),
        pl.col("_year").max().alias("latest_sim_year"),
        (pl.col("_year") >= 2022).sum().alias("sim_deaths_2022_2024"),
        (pl.col("_cause") == "A270").sum().alias("sim_a270"),
        (pl.col("_cause") == "A278").sum().alias("sim_a278"),
        (pl.col("_cause") == "A279").sum().alias("sim_a279"),
    )
    phenotype = (
        municipalities.filter(
            pl.col("surveillance_class") == "cross-system discordant"
        )
        .join(sih_m, on="munic_code", how="left")
        .join(sim_m, on="munic_code", how="left")
        .with_columns(
            pl.max_horizontal("latest_sih_year", "latest_sim_year")
            .alias("latest_comparison_year")
        )
        .sort("munic_code")
    )

    # The legacy single secondary-diagnosis field becomes the sentinel 0000
    # in every retained row from 2015. This is a schema boundary, not evidence
    # that secondary diagnoses ceased to exist clinically.
    schema = sih_raw.with_columns(
        pl.col("DT_INTER").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        .alias("_admit"),
        pl.col("DIAG_PRINC").cast(pl.Utf8).str.strip_chars().alias("_p"),
        pl.col("DIAG_SECUN").cast(pl.Utf8).str.strip_chars().alias("_s"),
    ).group_by("_src_year").agg(
        pl.len().alias("extracted_a27_admissions"),
        (~pl.col("_s").is_in([None, "", "0000"])).sum()
        .alias("informative_legacy_secondary_field"),
        (pl.col("_s") == "0000").sum().alias("secondary_sentinel_0000"),
        ((~pl.col("_p").str.starts_with("A27").fill_null(False))
         & pl.col("_s").str.starts_with("A27").fill_null(False)).sum()
        .alias("secondary_only_a27"),
        (pl.col("_admit").dt.year()
         != pl.col("ANO_CMPT").cast(pl.Int32, strict=False)).sum()
        .alias("admission_year_differs_from_competence_year"),
    ).with_columns(
        pl.when(pl.col("informative_legacy_secondary_field") > 0)
        .then(pl.lit("legacy secondary field informative"))
        .otherwise(pl.lit("legacy field unavailable (0000 sentinel)"))
        .alias("secondary_field_status")
    ).sort("_src_year").rename({"_src_year": "year"})

    # Candidate, not definitive, linkage among the mortality-plus-hospital
    # municipalities. No quasi-identifiers or record IDs leave this function.
    sim_link = sim.with_row_index("_sim_id").with_columns(
        pl.col("DTNASC").cast(pl.Utf8).str.to_date("%d%m%Y", strict=False)
        .alias("_birth"),
        pl.col("SEXO").cast(pl.Utf8).str.strip_chars().alias("_sex"),
    ).select("_sim_id", "munic_code", "_birth", "_sex", "_death_date")
    sih_link = sih.with_columns(
        pl.col("NASC").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        .alias("_birth"),
        pl.col("SEXO").cast(pl.Utf8).str.strip_chars().alias("_sex"),
        pl.col("N_AIH").cast(pl.Utf8).str.strip_chars().alias("_aih"),
    ).select("_aih", "munic_code", "_birth", "_sex", "_admit", "_discharge")
    pairs = sim_link.join(
        sih_link, on=["munic_code", "_birth", "_sex"], how="inner"
    ).filter(
        pl.col("_birth").is_not_null()
        & pl.col("_death_date").is_between(
            pl.col("_admit") - pl.duration(days=1),
            pl.col("_discharge") + pl.duration(days=1),
        )
    ).with_columns(
        pl.len().over("_sim_id").alias("_sim_candidates"),
        pl.len().over("_aih").alias("_aih_candidates"),
    )
    if pairs.is_empty():
        candidate = pl.DataFrame(schema={
            "munic_code": pl.Utf8, "candidate_pairs": pl.UInt32,
            "sim_deaths_with_candidate": pl.UInt32,
            "aihs_with_candidate": pl.UInt32,
            "unique_compatible_pairs": pl.UInt32,
        })
    else:
        candidate = pairs.group_by("munic_code").agg(
            pl.len().alias("candidate_pairs"),
            pl.col("_sim_id").n_unique().alias("sim_deaths_with_candidate"),
            pl.col("_aih").n_unique().alias("aihs_with_candidate"),
            ((pl.col("_sim_candidates") == 1) & (pl.col("_aih_candidates") == 1))
            .sum().alias("unique_compatible_pairs"),
        ).sort("munic_code")
    return phenotype, schema, candidate


def operational_queue(
    priority: pl.DataFrame, cascade: pl.DataFrame,
) -> pl.DataFrame:
    """Separate retrospective evidence class from a recency-aware work queue."""
    d = priority.join(
        cascade.select(
            "munic_code", "confirmed_zero_process_state", "investigations",
            "investigation_in_comparison_year",
        ), on="munic_code", how="left"
    ).with_columns(
        pl.when(pl.col("audit_priority_category") ==
                "A27 mortality plus hospital evidence").then(pl.lit(1))
        .when(pl.col("audit_priority_category") ==
              "fatal record in SIM or SIH").then(pl.lit(2))
        .when(pl.col("audit_priority_category") ==
              "repeated non-fatal comparison records").then(pl.lit(3))
        .otherwise(pl.lit(4)).alias("retrospective_category_rank"),
    )
    # Latest comparison year is recovered from the source-year counts already
    # represented by municipality; this ranking is deterministic and contains
    # no pseudo-probabilistic weights.
    tri = pl.read_parquet(PATHS.panel / "triangulation_municipality_year.parquet")
    recent = tri.with_columns(
        (pl.col("sim_a27_deaths") + pl.col("sih_a27_admissions"))
        .alias("_comparison")
    ).filter(pl.col("_comparison") > 0).group_by("munic_code").agg(
        pl.col("year").max().alias("latest_comparison_year")
    )
    return (
        d.join(recent, on="munic_code", how="left")
        .with_columns(
            (END - pl.col("latest_comparison_year")).alias("years_since_latest_record"),
            pl.when(pl.col("latest_comparison_year") >= 2022)
            .then(pl.lit("2022-2024"))
            .when(pl.col("latest_comparison_year") >= 2019)
            .then(pl.lit("2019-2021"))
            .otherwise(pl.lit("2008-2018")).alias("recency_band"),
        )
        .sort(
            "retrospective_category_rank", "latest_comparison_year",
            "comparison_positive_years", "comparison_records", "munic_code",
            descending=[False, True, True, True, False],
        )
        .with_row_index("operational_rank", offset=1)
    )


def audit_priority_partition(m: pl.DataFrame) -> pl.DataFrame:
    """Mutually exclusive audit categories under the versioned review rule."""
    d = m.filter(pl.col("surveillance_class") == "cross-system discordant")
    return d.with_columns(
        pl.when((pl.col("sim_a27_deaths") > 0) & (pl.col("sih_a27_admissions") > 0))
        .then(pl.lit("A27 mortality plus hospital evidence"))
        .when((pl.col("sim_a27_deaths") > 0) | (pl.col("sih_deaths") > 0))
        .then(pl.lit("fatal record in SIM or SIH"))
        .when(pl.col("comparison_records") >= 2)
        .then(pl.lit("repeated non-fatal comparison records"))
        .otherwise(pl.lit("single non-fatal SIH admission"))
        .alias("audit_priority_category")
    )


def opportunity_strata(m: pl.DataFrame) -> pl.DataFrame:
    """Describe classification by population, urbanisation and macro-region."""
    atlas = pl.read_parquet(PATHS.results / "atlas" / "municipality_atlas.parquet")
    d = m.join(atlas.select("munic_code", "urban_share_mean"), on="munic_code", how="left")
    qs = [float(d["mean_population"].quantile(q, interpolation="nearest"))
          for q in (0.2, 0.4, 0.6, 0.8)]
    d = d.with_columns(
        pl.when(pl.col("mean_population") <= qs[0]).then(pl.lit("Q1 smallest"))
        .when(pl.col("mean_population") <= qs[1]).then(pl.lit("Q2"))
        .when(pl.col("mean_population") <= qs[2]).then(pl.lit("Q3"))
        .when(pl.col("mean_population") <= qs[3]).then(pl.lit("Q4"))
        .otherwise(pl.lit("Q5 largest")).alias("population_quintile"),
        pl.when(pl.col("urban_share_mean").is_null()).then(pl.lit("Not available"))
        .when(pl.col("urban_share_mean") < 0.50).then(pl.lit("<50% urban"))
        .when(pl.col("urban_share_mean") < 0.80).then(pl.lit("50-79.9% urban"))
        .otherwise(pl.lit(">=80% urban")).alias("urbanisation_group"),
    )
    frames = []
    for kind, col in (("Population quintile", "population_quintile"),
                      ("Urban resident share", "urbanisation_group"),
                      ("Macro-region", "region")):
        frames.append(
            d.group_by(pl.col(col).alias("stratum"), "surveillance_class")
            .agg(
                pl.len().alias("municipalities"),
                pl.col("mean_population").sum().alias("aggregate_mean_annual_population"),
            )
            .with_columns(
                pl.lit(kind).alias("stratum_type"),
                (pl.col("municipalities") / pl.col("municipalities").sum().over("stratum"))
                .alias("within_stratum_share"),
            )
        )
    return pl.concat(frames, how="diagonal").select(
        "stratum_type", "stratum", "surveillance_class", "municipalities",
        "within_stratum_share", "aggregate_mean_annual_population",
    )


def state_audit_table(m: pl.DataFrame, cohort: pl.DataFrame) -> pl.DataFrame:
    """Complete state table underlying the geographic and temporal claims."""
    base = m.group_by("uf_abbr", "region").agg(
        pl.len().alias("total_municipalities"),
        (pl.col("surveillance_class") == "cross-system discordant").sum()
        .alias("discordant_municipalities"),
        pl.col("mean_population").filter(
            pl.col("surveillance_class") == "cross-system discordant"
        ).sum().alias("discordant_mean_annual_population"),
        pl.col("sim_a27_deaths").filter(
            pl.col("surveillance_class") == "cross-system discordant"
        ).sum().alias("discordant_sim_deaths"),
        pl.col("sih_a27_admissions").filter(
            pl.col("surveillance_class") == "cross-system discordant"
        ).sum().alias("discordant_sih_admissions"),
        ((pl.col("comparison_positive_years") >= 2)
         & (pl.col("surveillance_class") == "cross-system discordant"))
        .sum().alias("discordant_multiyear"),
        ((pl.col("sim_a27_deaths") > 0) & (pl.col("sih_a27_admissions") > 0)
         & (pl.col("surveillance_class") == "cross-system discordant"))
        .sum().alias("discordant_dual_system"),
    ).with_columns(
        (pl.col("discordant_municipalities") / pl.col("total_municipalities"))
        .alias("discordant_share")
    )
    temporal = cohort.group_by("uf_abbr").agg(
        pl.len().alias("baseline_zero_cohort"),
        (pl.col("first_signal_type") == "remained zero").sum().alias("remained_zero_2024"),
        (pl.col("first_signal_type") == "SINAN first").sum().alias("sinan_first"),
        (pl.col("first_signal_type") == "SIM/SIH first").sum().alias("comparison_first"),
        (pl.col("first_signal_type") == "same year").sum().alias("same_year_first"),
        pl.col("first_signal_year").median().alias("median_first_signal_year"),
    ).with_columns(
        (pl.col("remained_zero_2024") / pl.col("baseline_zero_cohort"))
        .alias("remained_zero_share")
    )
    return base.join(temporal, on="uf_abbr", how="left").sort("uf_abbr")


def outcome_completeness() -> pl.DataFrame:
    line = pl.scan_parquet(PATHS.interim / "lept_line_level.parquet")
    confirmed = pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1"
    onset = pl.col("DT_SIN_PRI").cast(pl.Utf8).str.to_date(strict=False)
    d = (
        line.filter(confirmed)
        .with_columns(onset.alias("_onset"))
        .filter(pl.col("_onset").is_between(date(2007, 1, 1), date(2025, 12, 31)))
        .filter(pl.col("uf_residence_state") == "valid")
        .with_columns(
            (pl.col("evolucao_state") == "valid").alias("known_outcome"),
            (pl.col("evolucao") == "obito_por_leptospirose").alias("died"),
        )
        .group_by(pl.col("uf_residence_abbr").alias("uf_abbr"))
        .agg(
            pl.len().alias("confirmed_cases"),
            pl.col("known_outcome").sum().alias("known_outcomes"),
            pl.col("died").sum().alias("deaths"),
        )
        .with_columns(
            (pl.col("known_outcomes") / pl.col("confirmed_cases"))
            .alias("outcome_completeness"),
            (pl.col("deaths") / pl.col("confirmed_cases")).alias("cfr_all_cases"),
            (pl.col("deaths") / pl.col("known_outcomes")).alias("cfr_known_outcome"),
        )
        .collect()
        .sort("outcome_completeness")
    )
    return d


def geography_alignment(m: pl.DataFrame) -> pl.DataFrame:
    """Compare the corrected residence arm with the former exposure arm."""
    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
    former = (
        panel.filter(pl.col("year").is_between(START, END))
        .group_by("munic_code")
        .agg(pl.col("cases").sum().alias("sinan_exposure_geography"))
    )
    return (
        m.join(former, on="munic_code", how="left")
        .with_columns(
            (
                (pl.col("sinan_exposure_geography") == 0)
                & (pl.col("comparison_records") > 0)
            ).alias("discordant_former_mixed_geography"),
            (
                (pl.col("sinan_confirmed") == 0)
                & (pl.col("comparison_records") > 0)
            ).alias("discordant_residence_aligned"),
        )
        .select(
            "munic_code", "name", "uf_abbr", "sinan_exposure_geography",
            "sinan_confirmed", "comparison_records",
            "discordant_former_mixed_geography", "discordant_residence_aligned",
        )
    )


def pnsb_discordance(tri: pl.DataFrame) -> pl.DataFrame:
    p = pl.read_parquet(
        PATHS.results / "structural" / "pnsb_leptospirosis_2008.parquet"
    )
    windows = [("2008", 2008, 2008), ("2007-2009", 2007, 2009),
               ("2007-2025", 2007, 2025)]
    rows = []
    for label, lo, hi in windows:
        cases = (tri.filter(pl.col("year").is_between(lo, hi))
                 .group_by("munic_code")
                 .agg(pl.col("sinan_confirmed").sum().alias("sinan_cases")))
        d = p.join(cases, on="munic_code", how="left").with_columns(
            pl.col("sinan_cases").fill_null(0)
        )
        declared = d.filter(pl.col("pnsb_leptospirosis_2008"))
        denied = d.filter(~pl.col("pnsb_leptospirosis_2008"))
        rows.append({
            "window": label,
            "pnsb_declared": declared.height,
            "declared_sinan_zero": declared.filter(pl.col("sinan_cases") == 0).height,
            "pnsb_denied": denied.height,
            "denied_sinan_positive": denied.filter(pl.col("sinan_cases") > 0).height,
        })
    return pl.DataFrame(rows).with_columns(
        (pl.col("declared_sinan_zero") / pl.col("pnsb_declared"))
        .alias("declared_sinan_zero_share"),
        (pl.col("denied_sinan_positive") / pl.col("pnsb_denied"))
        .alias("denied_sinan_positive_share"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tri = pl.read_parquet(PATHS.panel / "triangulation_municipality_year.parquet")
    m = municipality_table(tri)
    tiers = evidence_tiers(m)
    priority = audit_priority_partition(m)
    horizon = by_horizon(tri)
    transitions = transition_matrix(tri)
    transitions_fixed_lattice = transition_matrix(tri, include_future_units=True)
    zero_cohort, zero_history, zero_states, zero_pathways = zero_cohort_history(tri)
    states = state_audit_table(m, zero_cohort)
    strata = opportunity_strata(m)
    source_flow, source_summary, annual_sources, sih_position = source_derivation(tri)
    contracts, retrieval_manifest = source_contracts()
    alignment = geography_alignment(m)
    cascade, cascade_summary = surveillance_cascade(tri, m)
    phenotypes, sih_schema, candidate_pairs = comparison_record_phenotypes(m)
    current_queue = operational_queue(priority, cascade)

    m.write_parquet(OUT / "municipality_surveillance_audit.parquet")
    m.filter(pl.col("surveillance_class") == "cross-system discordant").write_csv(
        OUT / "cross_system_discordant_municipalities.csv"
    )
    priority.write_csv(OUT / "audit_priority_municipalities.csv")
    (priority.group_by("audit_priority_category").agg(
        pl.len().alias("municipalities"),
        pl.col("mean_population").sum().alias("aggregate_mean_annual_population"),
        pl.col("sim_a27_deaths").sum().alias("all_sim_deaths_in_municipalities"),
        pl.col("sih_a27_admissions").sum().alias("all_sih_admissions_in_municipalities"),
    ).sort("municipalities", descending=True)
     .write_csv(OUT / "audit_priority_partition.csv"))
    tiers.write_csv(OUT / "evidence_tiers.csv")
    horizon.write_csv(OUT / "classification_by_horizon.csv")
    transitions.write_csv(OUT / "transition_matrix_2008_2024.csv")
    transitions_fixed_lattice.write_csv(
        OUT / "transition_matrix_2008_2024_fixed_lattice_sensitivity.csv"
    )
    zero_cohort.write_parquet(OUT / "zero_2008_cohort_first_signal.parquet")
    zero_history.write_csv(OUT / "zero_2008_cohort_history.csv")
    zero_states.write_csv(OUT / "zero_2008_cohort_by_state.csv")
    zero_pathways.write_csv(OUT / "zero_2008_first_signal_to_endpoint.csv")
    states.write_csv(OUT / "state_audit_table.csv")
    strata.write_csv(OUT / "classification_by_opportunity_strata.csv")
    source_flow.write_csv(OUT / "source_record_flow.csv")
    source_summary.write_csv(OUT / "source_analytic_summary.csv")
    annual_sources.write_csv(OUT / "annual_source_records.csv")
    sih_position.write_csv(OUT / "sih_diagnostic_position_audit.csv")
    contracts.write_csv(OUT / "source_contracts.csv")
    retrieval_manifest.write_csv(OUT / "source_retrieval_manifest.csv")
    alignment.write_csv(OUT / "geography_alignment_audit.csv")
    cascade.write_parquet(OUT / "municipality_surveillance_cascade.parquet")
    cascade.filter(
        pl.col("surveillance_class") == "cross-system discordant"
    ).write_csv(OUT / "discordant_surveillance_cascade.csv")
    cascade_summary.write_csv(OUT / "surveillance_cascade_summary.csv")
    phenotypes.write_csv(OUT / "comparison_record_phenotypes.csv")
    sih_schema.write_csv(OUT / "sih_schema_audit_by_year.csv")
    candidate_pairs.write_csv(OUT / "dual_system_candidate_pair_summary.csv")
    current_queue.write_csv(OUT / "current_operational_audit_queue.csv")

    primary = tiers.row(0, named=True)
    strict = tiers.row(1, named=True)
    report = {
        "schema": "brepi.study.paperA-audit/1",
        "window": [START, END],
        "geography": "municipality of residence in SINAN, SIM and SIH",
        "primary": primary,
        "strict_diagnosis": strict,
        "class_counts": {
            r["surveillance_class"]: r["len"]
            for r in m.group_by("surveillance_class").len().to_dicts()
        },
        "baseline_zero_cohort": {
            "municipalities": zero_cohort.height,
            "remained_zero": zero_cohort.filter(
                pl.col("first_signal_type") == "remained zero"
            ).height,
            "sinan_first": zero_cohort.filter(
                pl.col("first_signal_type") == "SINAN first"
            ).height,
            "comparison_first": zero_cohort.filter(
                pl.col("first_signal_type") == "SIM/SIH first"
            ).height,
            "same_year": zero_cohort.filter(
                pl.col("first_signal_type") == "same year"
            ).height,
        },
        "source_records": {
            row["source"]: {
                "analytic_records": row["analytic_records"],
                "municipalities_represented": row["municipalities_represented"],
                "missing_or_invalid_residence": row["missing_or_invalid_residence"],
            }
            for row in source_summary.to_dicts()
        },
        "surveillance_cascade": {
            row["confirmed_zero_process_state"]: row["municipalities"]
            for row in cascade_summary.to_dicts()
        },
        "geography_correction": {
            "former_mixed_geography": int(
                alignment["discordant_former_mixed_geography"].sum()
            ),
            "residence_aligned": int(alignment["discordant_residence_aligned"].sum()),
            "removed": alignment.filter(
                pl.col("discordant_former_mixed_geography")
                & ~pl.col("discordant_residence_aligned")
            ).height,
            "added": alignment.filter(
                ~pl.col("discordant_former_mixed_geography")
                & pl.col("discordant_residence_aligned")
            ).height,
        },
        "interpretation": (
            "The primary quantity is observed cross-system discordance, not a "
            "proof that every comparison-system code is a true leptospirosis "
            "case and not an estimate of total under-ascertainment."
        ),
    }
    (OUT / "paperA_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
