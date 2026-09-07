#!/usr/bin/env python
"""Assemble the decoded, code-translated SINAN-LEPT line-level extract.

Every aggregate in this study descends from these records, but until now only
one year had ever been materialised, so three things the paper needs could not
be computed at all:

* **Field completeness over time.** A coded field that is 40% unknown cannot
  support a stratified analysis, and one whose completeness *drifts* will
  manufacture a trend. Both are results, not housekeeping, and both need the
  line level.
* **Age- and sex-specific rates.** The panel is aggregated over people; the
  denominator tensor is age-structured. Standardisation needs a numerator with
  the same structure.
* **Notification delay.** Onset to notification to digitisation bounds how much
  of the most recent period is missing rather than absent, which is what makes
  2025 quotable or not.

The download layer already holds every ``.dbc``, so this is pure local decode
work. It is parallelised across processes rather than threads because the
decode is compression plus dataframe construction and holds the GIL throughout
(BREPI-018); per-year results are content-addressed by
``materialise()``, so a rerun costs a parquet read.

Usage:
    python studies/leptospirosis/21_build_line_level.py [--refresh]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brepi.sources.datasus import sinan  # noqa: E402

AGRAVO = "LEPT"
YEARS = range(2007, 2026)
OUT = ROOT / "data" / "interim" / "lept_line_level.parquet"

# Columns carried forward. The current record has 293 columns after decoding
# and geography enrichment (71 retained raw-code fields plus 222 decoded or
# derived fields). Its width is machine-asserted so future dictionary expansion
# is deliberate rather than silent.
IDENT = [
    "NU_ANO", "DT_NOTIFIC", "DT_SIN_PRI", "DT_DIGITA", "DT_INVEST",
    "DT_ENCERRA", "DT_OBITO", "ATE_DT_INT", "SEM_PRI",
    "SG_UF_NOT", "ID_MUNICIP", "SG_UF", "ID_MN_RESI", "COMUNINF", "COUFINF",
    "ANO_NASC", "NU_IDADE_N", "ID_OCUPA_N",
]
# Raw codes retained alongside their decoded labels: completeness must be
# assessed on what the notifier actually entered, and a decoded label cannot
# distinguish "field left blank" from "code absent from the dictionary".
CODED = [
    "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "CS_GESTANT",
    "CLASSI_FIN", "CRITERIO", "EVOLUCAO", "ATE_HOSP", "TPAUTOCTO",
    "DOENCA_TRA", "CON_AMBIEN",
    "LAB_ELIS_1", "LAB_MICR_1", "RES_ISOL", "RES_IMUNO", "RES_PCR",
]
# Clinical syndrome and exposure-antecedent blocks: the descriptive backbone of
# a leptospirosis paper and never yet tabulated in this study.
CLINICAL = [
    "CLI_FEBRE", "CLI_MIALGI", "CLI_CEFALE", "CLI_PROST", "CLI_CONGES",
    "CLI_PANTUR", "CLI_VOMITO", "CLI_DIARRE", "CLI_ICTERI", "CLI_RENAL",
    "CLI_RESPIR", "CLI_CARDIA", "CLI_HEMOPU", "CLI_HEMORR", "CLI_MENING",
]
ANTECEDENT = [
    "ANT_CB_LAM", "ANT_CB_CRI", "ANT_CB_CAI", "ANT_CB_FOS", "ANT_CB_SIN",
    "ANT_CB_PLA", "ANT_CB_COR", "ANT_CB_ROE", "ANT_CB_GRA", "ANT_CB_TER",
    "ANT_CB_LIX", "ANT_CB_OUT", "ANT_HUMANO", "ANT_ANIMAI",
]
# Microscopic agglutination test: serovar and reciprocal titre, first and
# second reacting serovar, first and second sample. The serovar identifies the
# maintenance host -- Copenhageni and Icterohaemorrhagiae mean Rattus, Canicola
# means dogs, the Sejroe group means cattle -- so this is the only field in the
# record that speaks to the transmission source. It was untranslated until the
# codebook learned to read the self-labelling form, and so had never been used.
SEROLOGY = [
    "MICRO1_S1", "MICRO1_T_1", "MICRO1_S_2", "MICRO1_T_2",
    "MICRO2_S1", "MICRO2_T_1", "MICRO2_S_2", "MICRO2_T_2",
]
RAW_KEEP = IDENT + CODED + CLINICAL + ANTECEDENT + SEROLOGY
# Every decoded column the codebook emits is kept, rather than a hand-listed
# subset: the decoded columns ARE the analytic surface, and a keep-list that
# has to be edited whenever a binding is added is a keep-list that will fall
# behind the registry.
KEEP = None  # resolved per frame, see _keep_columns()


def _keep_columns(columns: list[str]) -> list[str]:
    """Raw fields we declared, plus every column the codebook produced.

    A decoded column is any column that is not upper-case: the raw DATASUS
    schema is upper-case throughout and the codebook writes lower-case names.
    """
    raw = [c for c in RAW_KEEP if c in columns]
    decoded = [c for c in columns if not c.isupper()]
    return raw + decoded


def _one(year: int, refresh: bool) -> tuple[int, str | None, int]:
    """Materialise one year. Returns (year, error, rows)."""
    try:
        # fetch_year_normalised() returns (frame, materialisation-metadata).
        frame, _meta = sinan.fetch_year_normalised(AGRAVO, year, refresh=refresh)
        return year, None, frame.height
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return year, f"{type(exc).__name__}: {exc}", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    errors: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, y, args.refresh): y for y in YEARS}
        for i, fut in enumerate(as_completed(futs), 1):
            year, err, rows = fut.result()
            if err:
                errors[year] = err
                print(f"  [{i:2d}/{len(futs)}] {year}  FAILED  {err}", flush=True)
            else:
                print(f"  [{i:2d}/{len(futs)}] {year}  {rows:>7,} rows", flush=True)

    if errors:
        # A missing year silently shortens the study period, so this is fatal
        # rather than a warning. If a year is genuinely unavailable upstream,
        # narrow YEARS deliberately and say so in the manuscript.
        print(f"\n{len(errors)} year(s) failed: {sorted(errors)}", file=sys.stderr)
        return 1

    frames = []
    for year in YEARS:
        f, _ = sinan.fetch_year_normalised(AGRAVO, year)
        present = _keep_columns(f.columns)
        f = f.select(present).with_columns(pl.lit(year).cast(pl.Int32).alias("src_year"))
        # RAW fields drift across the period (added, widened, retyped), so they
        # are normalised to Utf8 before the vertical concat; coercing here
        # rather than letting polars pick a supertype keeps the drift visible in
        # the audit below.
        #
        # DECODED fields are NOT coerced. Their schema is determined by the
        # codebook, not by the file vintage, so it cannot drift -- and casting
        # them back to text would throw away exactly the typing the codebook was
        # run to establish, leaving `age_years` a string for the analysis to
        # re-parse. That is the defect this whole layer exists to remove.
        raw_present = [c for c in present if c.isupper()]
        frames.append(f.with_columns(
            [pl.col(c).cast(pl.Utf8) for c in raw_present]))

    allcols = sorted({c for f in frames for c in f.columns})
    for i, f in enumerate(frames):
        missing = [c for c in allcols if c not in f.columns]
        if missing:
            # Fill a column absent from this vintage with nulls of the dtype
            # the other frames use, so the concat does not silently widen a
            # decoded numeric back to text.
            dtypes = {}
            for other in frames:
                for c in missing:
                    if c in other.columns and c not in dtypes:
                        dtypes[c] = other.schema[c]
            frames[i] = f.with_columns(
                [pl.lit(None, dtype=dtypes.get(c, pl.Utf8)).alias(c)
                 for c in missing])
        frames[i] = frames[i].select(allcols)

    out = pl.concat(frames, how="vertical")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    # Column availability by year: which fields exist in which vintages. This
    # is the difference between "nobody filled it" and "the field did not exist
    # yet", and only the second is a dictionary-version fact.
    audit = []
    for year, f in zip(YEARS, frames):
        for c in allcols:
            if c in ("src_year",):
                continue
            nn = f.select(pl.col(c).is_not_null().sum()).item()
            audit.append({"year": year, "column": c, "non_null": nn, "rows": f.height})
    audit_df = pl.DataFrame(audit)
    audit_path = (ROOT / "data" / "results" / "00_data_quality"
                  / "lept_line_level_column_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.write_csv(audit_path)

    never = (audit_df.group_by("column").agg(pl.col("non_null").sum())
             .filter(pl.col("non_null") == 0)["column"].to_list())

    print(f"\nwrote {OUT}  {out.height:,} rows x {out.width} columns "
          f"({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    print(f"wrote {audit_path}")
    if never:
        print(f"columns empty in EVERY year ({len(never)}): {', '.join(sorted(never))}")
    by_year = out.group_by("src_year").len().sort("src_year")
    print("\nrecords by year:")
    for row in by_year.iter_rows():
        print(f"  {row[0]}  {row[1]:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
