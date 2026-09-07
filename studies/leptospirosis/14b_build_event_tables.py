"""RQ2 - emit the event-level declaration tables the R DiD engine consumes.

`14_build_disaster_panel.py` writes `flood_events.parquet` as a by-product of a
full rebuild (which re-acquires the Atlas). This script emits the same table
from the frozen cache, and adds the table RQ2's decisive robustness check
needs: **drought declarations**.

Drought is the negative control. It travels the identical bureaucratic pathway
(municipal civil-defence files, state/federal authority recognises), so it
carries the same administrative-capacity selection as a flood declaration, but
it has the opposite hydrology and no plausible leptospirosis mechanism. A
non-null drought "effect" in the same design would mean the estimator is
reading administrative capacity, not water.

Run:
    python studies/leptospirosis/14b_build_event_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.config import PATHS
from brepi.sources.disasters import atlas

YEARS = range(2007, 2026)

SPECS = {
    "flood_events.parquet": atlas.COBRADE_FLOOD,
    "drought_events.parquet": atlas.COBRADE_DROUGHT,
}


def main() -> int:
    PATHS.ensure()
    out = PATHS.panel
    for fname, prefixes in SPECS.items():
        events = atlas.disaster_events(YEARS, cobrade_prefixes=list(prefixes))
        print(f"[{fname}] {events.height:,} records, "
              f"{events['munic_code'].n_unique():,} municipalities, "
              f"prefixes={list(prefixes)}")
        print(atlas.cobrade_summary(events).to_pandas().to_string(index=False))
        (
            events.select(
                "munic_code",
                pl.col("event_date").alias("date"),
                "cobrade", "cobrade_label", "recognised", "uf_abbr",
                "deaths", "affected_total",
            )
            .sort("munic_code", "date")
            .write_parquet(out / fname)
        )
        print(f"      wrote {out / fname}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
