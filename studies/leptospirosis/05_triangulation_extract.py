"""WP2 - extract administratively distinct leptospirosis comparison records.

Leptospirosis reaches three national systems by three different routes:

SINAN
    A notifiable-disease notification, filed by a surveillance service after a
    clinician suspects the disease. Requires suspicion.

SIM
    A death certificate coded to CID-10 A27 as underlying cause. Requires a
    death and a coder, not a notification.

SIH
    A public-hospital admission coded to A27 as principal or secondary
    diagnosis. Requires an admission and a billing code, not a notification.

Their workflows and incentives differ but their clinical pathways overlap.
Disagreement is therefore an audit signal, not proof of a missed notification,
statistical independence, or a person-level capture history.

SIH is the expensive extraction (one file per UF-month, ~6,100 files). Run it
with ``--sih`` when you have the time and disk; ``--sim`` alone already gives
the decisive mortality comparison.

Run:
    python studies/leptospirosis/05_triangulation_extract.py --sim
    python studies/leptospirosis/05_triangulation_extract.py --sih --ufs RS,AC,SP,PE,PR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.sources.datasus import sih, sim

ICD = ["A27"]
YEARS = range(2007, 2025)  # SIM final ends 2024; SIH runs later but align them


def run_sim(years, ufs) -> None:
    print(f"SIM: A27 underlying-cause deaths, {min(years)}-{max(years)}")
    deaths = sim.fetch_icd_deaths(ICD, years, ufs=ufs)
    print(f"      {deaths.height:,} deaths")
    out = PATHS.interim / "sim_a27_deaths.parquet"
    deaths.write_parquet(out)
    by_year = deaths.group_by("_src_year").agg(pl.len().alias("deaths")).sort("_src_year")
    print(by_year.to_pandas().to_string(index=False))
    print(f"      wrote {out}")


def run_sih(years, ufs) -> None:
    """Extract SIH A27 admissions, one calendar year per parquet part.

    The current RD series begins 200801; 2007 lives in the legacy tree under a
    different record layout and is deliberately *not* concatenated with it, so
    a window starting in 2007 silently becomes a 2008 window and the actual
    coverage is written into the part files and reported here.
    """
    years = [y for y in years if y >= 2008]
    if not years:
        raise SystemExit("SIH current series starts in 2008; nothing to do")
    print(f"SIH: A27 admissions, {min(years)}-{max(years)}, ufs={ufs or 'all'}")
    parts_dir = PATHS.interim / "sih_a27_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        part = parts_dir / f"sih_a27_{year}.parquet"
        if part.exists():
            print(f"      {year}: already extracted ({pl.read_parquet(part).height} rows)")
            continue
        adm = sih.fetch_icd_admissions(ICD, [year], ufs=ufs)
        adm.write_parquet(part)
        print(f"      {year}: {adm.height:,} admissions -> {part.name}", flush=True)

    have = sorted(parts_dir.glob("sih_a27_*.parquet"))
    if not have:
        raise SystemExit("no SIH parts produced")
    adm = pl.concat([pl.read_parquet(p) for p in have], how="diagonal")
    out = PATHS.interim / "sih_a27_admissions.parquet"
    adm.write_parquet(out)
    by_year = adm.group_by("_src_year").agg(pl.len().alias("admissions")).sort("_src_year")
    print(by_year.to_pandas().to_string(index=False))
    print(f"      {adm.height:,} admissions over {len(have)} year parts -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sih", action="store_true")
    ap.add_argument("--ufs", default=None, help="comma-separated UF abbreviations")
    ap.add_argument("--start", type=int, default=2007)
    ap.add_argument("--end", type=int, default=2024)
    a = ap.parse_args()
    PATHS.ensure()
    years = range(a.start, a.end + 1)
    ufs = a.ufs.split(",") if a.ufs else None
    if a.sim:
        run_sim(years, ufs)
    if a.sih:
        run_sih(years, ufs)
    if not (a.sim or a.sih):
        ap.error("choose --sim and/or --sih")
