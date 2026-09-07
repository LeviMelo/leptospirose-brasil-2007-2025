"""WP1 - extract, snapshot, profile and reconcile the SINAN leptospirosis numerator.

Run:
    python studies/leptospirosis/01_extract_sinan.py

Reproduces, as a pipeline step, every numerator finding in
``PROTOCOL_v2_AMENDMENTS.md``. Nothing here is exploratory: each stage either
passes a stated gate or fails loudly.

Gates
-----
1. Schema stability across 2007-2025 (single decoder for the NET era).
2. Reconciliation against the published MoH national series, mean absolute
   relative discrepancy < 3%.
3. Sparsity report emitted so the analytic-unit decision (Tier A/B/C) rests on
   measured numbers rather than assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The Windows console defaults to cp1252 and cannot render polars' box-drawing
# frames; force UTF-8 rather than degrading every table to ASCII.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.geo import lattice
from brepi.qa.sinan_quality import profile, reconcile
from brepi.sources.datasus import sinan

AGRAVO = "LEPT"
YEARS = range(2007, 2026)

#: MoH, "Casos e obitos confirmados 2000-2026", snapshot 2026-01-22, by UF of
#: residence. The reconciliation target. Source:
#: gov.br/saude/pt-br/assuntos/saude-de-a-a-z/l/leptospirose/situacao-epidemiologica
MOH_CONFIRMED = {
    2007: 3332, 2008: 3681, 2009: 3986, 2010: 3818, 2011: 4967, 2012: 3269,
    2013: 4150, 2014: 4676, 2015: 4340, 2016: 3065, 2017: 3000, 2018: 3065,
    2019: 3707, 2020: 1890, 2021: 1804, 2022: 3144, 2023: 3428, 2024: 4142,
    2025: 2761,
}

#: Officially documented category codes (Dicionario de Dados SINAN NET v5.0).
#: Anything observed outside these is reported, never silently recoded. On this
#: agravo that surfaces the undocumented CLASSI_FIN='8'.
DICTIONARY = {
    "CLASSI_FIN": ["1", "2"],
    "CRITERIO": ["1", "2"],
    "EVOLUCAO": ["1", "2", "3", "9"],
    "TPAUTOCTO": ["1", "2", "3"],
    "CON_AMBIEN": ["1", "2", "3", "4", "9"],
    "DOENCA_TRA": ["1", "2", "9"],
}

RECONCILIATION_TOLERANCE = 0.03


def main(backend: str, refresh: bool) -> int:
    PATHS.ensure()
    out = PATHS.reports / "wp1_sinan"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] snapshotting SINAN {AGRAVO} {YEARS.start}-{YEARS.stop - 1} via {backend}")
    manifest = sinan.snapshot(AGRAVO, YEARS, backend=backend, refresh=refresh)
    print(f"      manifest: {manifest}")

    print("[2/5] decoding")
    frame, drift = sinan.fetch_range(AGRAVO, YEARS, backend=backend)
    print(f"      {frame.height:,} notification records, {frame.width} columns")
    drift.write_csv(out / "schema_drift.csv")

    # Gate 1 - schema stability.
    unstable = drift.filter(~pl.col("stable"))
    if unstable.height:
        print(f"      WARNING: {unstable.height} columns are not present in every year")
        print(unstable.select("column", "n_years", "missing_in").head(10))
    else:
        print("      GATE 1 PASS: schema identical across all years")

    print("[3/5] quality profile")
    qp = profile(frame, dictionary=DICTIONARY)
    qp.write(out)
    print(qp.by_year.select(
        "year", "notified", "confirmed", "confirmation_ratio",
        "criterion_epi_share", "COMUNINF_missing_share",
        "exposure_block_informative_share",
    ))
    if qp.undocumented.height:
        print("      undocumented codes observed:")
        print(qp.undocumented)

    print("[4/5] reconciliation against the published MoH series")
    rec = reconcile(qp.by_year, MOH_CONFIRMED)
    rec.write_csv(out / "reconciliation.csv")
    mae = float(rec["relative"].abs().mean())
    print(rec)
    print(f"      mean absolute relative discrepancy: {mae:.4f}")
    if mae > RECONCILIATION_TOLERANCE:
        print(f"      GATE 2 FAIL: exceeds tolerance {RECONCILIATION_TOLERANCE}")
        return 1
    print("      GATE 2 PASS")

    print("[5/5] municipality-month aggregation and sparsity")
    resolved = resolve_confirmed_geography(frame)
    counts = aggregate_confirmed_by_municipality_month(resolved)
    counts.write_parquet(PATHS.interim / "lept_confirmed_mun_month.parquet")
    unresolved = (
        resolved.filter(pl.col("munic_code_status") == "unresolved")
        .group_by("_inf", "_res", "period")
        .agg(
            pl.len().alias("cases"),
            (pl.col("EVOLUCAO") == "2").sum().alias("deaths"),
        )
        .sort("period", "_inf", "_res")
    )
    unresolved.write_parquet(
        PATHS.interim / "lept_confirmed_unresolved_geography.parquet"
    )
    geocoded_cases = int(counts["cases"].sum())
    unresolved_cases = int(unresolved["cases"].sum())
    if geocoded_cases + unresolved_cases != resolved.height:
        raise RuntimeError("geography resolution failed to conserve confirmed cases")
    stats = {
        "confirmed_in_window": resolved.height,
        "geocoded_cases": geocoded_cases,
        "unresolved_geography_cases": unresolved_cases,
        "infection_geography_cases": resolved.filter(
            pl.col("geo_key_source") == "infection"
        ).height,
        "residence_fallback_cases": resolved.filter(
            pl.col("geo_key_source") == "residence_fallback"
        ).height,
        "nonzero_cells": counts.height,
        "municipalities_ever_reporting": counts["munic_code6"].n_unique(),
        "median_cases_per_nonzero_cell": float(counts["cases"].median()),
        "cells_ge_5": int(counts.filter(pl.col("cases") >= 5).height),
    }
    print(json.dumps(stats, indent=1))
    (out / "sparsity.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return 0


def confirmed_by_municipality_month(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate confirmed cases to municipality x month of first symptoms.

    Geography key is ``COMUNINF`` (probable municipality of infection), which
    is the only key aligned with environmental exposure. It is missing for
    10-22% of confirmed cases, rising after 2020, so residence is used as a
    documented fallback and the source of the key is retained per row for the
    sensitivity analysis. Silent dropping would confound the COVID-era
    reporting degradation with exposure.
    """
    return aggregate_confirmed_by_municipality_month(
        resolve_confirmed_geography(frame)
    )


def resolve_confirmed_geography(frame: pl.DataFrame) -> pl.DataFrame:
    """Resolve infection then residence against the actual 2022 lattice."""
    c = (
        frame.filter(
            pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1"
        )
        .with_columns(
            pl.col("DT_SIN_PRI").str.to_date(strict=False).alias("_d"),
            pl.col("COMUNINF").cast(pl.Utf8).str.strip_chars().alias("_inf"),
            pl.col("ID_MN_RESI").cast(pl.Utf8).str.strip_chars().alias("_res"),
            pl.col("EVOLUCAO").cast(pl.Utf8).str.strip_chars(),
            pl.col("ATE_HOSP").cast(pl.Utf8).str.strip_chars(),
            pl.col("CRITERIO").cast(pl.Utf8).str.strip_chars(),
            pl.col("ANT_CB_LAM").cast(pl.Utf8).str.strip_chars(),
        )
        .filter(pl.col("_d").is_between(date(2007, 1, 1), date(2025, 12, 31)))
        .with_columns(pl.col("_d").dt.truncate("1mo").alias("period"))
    )
    return lattice.resolve_municipality_code(
        c,
        [("infection", "_inf"), ("residence_fallback", "_res")],
        output_column="munic_code6",
        source_column="geo_key_source",
        status_column="munic_code_status",
    )


def aggregate_confirmed_by_municipality_month(
    resolved: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate only resolved records; unresolved mass is audited separately."""
    return (
        resolved.filter(pl.col("munic_code_status") == "resolved")
        # geo_key_source is carried as a COUNT, never as a grouping key: it must
        # not split a municipality-month into two rows, or the panel join sees
        # duplicate keys and PanelBuild.add rejects the source.
        .group_by("munic_code6", "period")
        .agg(
            pl.len().alias("cases"),
            (pl.col("geo_key_source") == "residence_fallback").sum().alias("cases_residence_fallback"),
            (pl.col("EVOLUCAO") == "2").sum().alias("deaths"),
            (pl.col("ATE_HOSP") == "1").sum().alias("hospitalised"),
            (pl.col("CRITERIO") == "2").sum().alias("criterion_epi"),
            (pl.col("ANT_CB_LAM") == "1").sum().alias("flood_contact"),
        )
        .sort("munic_code6", "period")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ftp", choices=["ftp", "mirror"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(args.backend, args.refresh))
