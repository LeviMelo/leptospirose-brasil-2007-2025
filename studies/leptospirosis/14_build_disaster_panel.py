"""RQ2 - build the flood-disaster treatment panel for the staggered DiD.

Treatment is an officially recognised hydrological disaster in a municipality
in a month, from the Atlas Digital de Desastres (SEDEC/MIDR, built on S2iD).

Three things about this treatment need stating before any estimate is read.

**Recognition is an administrative act, not a hydrological one.** A
municipality enters the register when its civil-defence apparatus files, and a
state or federal authority recognises. Municipal administrative capacity is
therefore part of the treatment assignment mechanism, and it plausibly
correlates with health-surveillance capacity - which is the outcome-reporting
mechanism. That is a live confounding path, not a footnote, and it is why the
event-study pre-trends matter more here than in a typical application.

**Treatment recurs.** Canonical staggered difference-in-differences assumes
absorbing treatment: once treated, always treated. Flood declarations do not
work that way; many municipalities are declared repeatedly. Both definitions
are therefore built - `absorbing` (first declaration onward) and `episodic`
(each declaration as its own event) - and the share of municipalities with
recurrent treatment is reported so the reader can judge which is credible.

**Never-treated is not a random control group.** Municipalities that never
declare are systematically drier, smaller and administratively weaker. The
not-yet-treated comparison is the defensible one; the never-treated set is
reported for completeness and used only in a sensitivity.

Run:
    python studies/leptospirosis/14_build_disaster_panel.py
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
from brepi.sources.disasters import atlas, treatment

YEARS = range(2007, 2026)
START, END = date(2007, 1, 1), date(2025, 12, 1)


def main() -> int:
    PATHS.ensure()
    out = PATHS.panel
    rep = PATHS.reports
    rep.mkdir(parents=True, exist_ok=True)

    print("[1/5] acquiring the Atlas consolidated base")
    manifest = atlas.snapshot()
    print(f"      snapshot manifest: {manifest}")

    # Treatment must include COBRADE 1.3.2.1.4 "tempestade local/convectiva -
    # chuvas intensas", even though it reads as a meteorological rather than a
    # hydrological category.
    #
    # The reason is empirical and decisive. In the Rio Grande do Sul
    # catastrophe of April-June 2024 -- the largest flood disaster in modern
    # Brazilian history -- 465 of the 533 municipal records were filed under
    # 1.3.2.1.4, against 21 enxurradas, 20 inundacoes and 10 alagamentos.
    # Restricting to the "proper" flood codes 1.2.x drops the state from 467
    # municipalities to 47 and removes the defining event of the study period.
    #
    # COBRADE is therefore a filing convention of the municipal civil-defence
    # service, not a hydrological classification of what happened. Any staggered
    # DiD on Brazilian flood declarations that trusts the 1.2.x codes will
    # silently lose major events. We use the broad definition and address the
    # obvious objection -- that 1.3.2.1.4 may capture heavy rain without
    # flooding -- by conditioning on the precipitation cross-basis, which is the
    # test the design calls for anyway: does the declaration carry risk *beyond*
    # rainfall volume? The strict 1.2.x definition is a prespecified sensitivity.
    print("[2/5] hydrological (flood-type) events -- primary: COBRADE 1.2.x + 1.3.2.1.4")
    events, report = atlas.disaster_events(
        YEARS, cobrade_prefixes=atlas.COBRADE_FLOOD, return_report=True
    )
    print(f"      {events.height:,} flood-type event records "
          f"({events['munic_code'].n_unique():,} municipalities)")
    print("      COBRADE mix:")
    print(atlas.cobrade_summary(events).to_pandas().to_string(index=False))
    (rep / "rq2_atlas_acquisition.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )

    print("[3/5] municipality-month treatment panel")
    mm = treatment.to_municipality_month(events, start=START, end=END)
    print(f"      {mm.height:,} rows; treated municipality-months: "
          f"{int((mm['n_events'] > 0).sum()):,}")

    print("[4/5] treatment timing under both definitions")
    g_abs = treatment.first_treatment_period(events, start=START, end=END)
    cohorts = treatment.treatment_cohorts(g_abs, by="year")
    print(cohorts.to_pandas().to_string(index=False))

    # The universe must be the full municipal lattice, not the set of
    # municipalities that appear in the events file. Inferring it from the
    # events makes every unit look treated and reports zero never-treated
    # controls, which is exactly backwards: the never-treated group is the
    # 1,000-odd municipalities that never declare.
    universe = (
        pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet",
                        columns=["munic_code"])["munic_code"].unique().to_list()
    )
    feas = treatment.did_feasibility_report(
        events, municipalities=universe, start=START, end=END
    )
    print("\n[5/5] DiD feasibility")
    print(json.dumps(feas, indent=1, default=str))
    (rep / "rq2_did_feasibility.json").write_text(
        json.dumps(feas, indent=1, default=str), encoding="utf-8"
    )

    # The R DiD engine consumes declarations at event level (one row per
    # municipality-declaration), not the aggregated municipality-month panel,
    # because it derives its own treatment timing and needs the COBRADE code to
    # apply the sensitivity definitions.
    (
        events.select(
            "munic_code",
            pl.col("event_date").alias("date"),
            "cobrade", "cobrade_label", "recognised", "uf_abbr",
            "deaths", "affected_total",
        )
        .sort("munic_code", "date")
        .write_parquet(out / "flood_events.parquet")
    )
    print(f"      wrote {out / 'flood_events.parquet'} ({events.height:,} events)")

    mm.write_parquet(out / "flood_declarations.parquet")
    g_abs.write_parquet(out / "flood_first_treatment.parquet")
    cohorts.write_csv(rep / "rq2_treatment_cohorts.csv")
    print(f"\n      wrote {out / 'flood_declarations.parquet'}")

    # --- targeted validation ------------------------------------------------
    # If the extraction is right, two events that are matters of public record
    # must be present. If they are not, nothing downstream is trustworthy.
    print("\n[validate] Rio Grande do Sul, April-June 2024")
    rs = mm.filter(
        (pl.col("munic_code").str.starts_with("43"))
        & (pl.col("period").is_between(date(2024, 4, 1), date(2024, 6, 1)))
        & (pl.col("n_events") > 0)
    )
    print(f"      RS municipalities with a flood declaration: "
          f"{rs['munic_code'].n_unique()}")

    print("[validate] Rio de Janeiro Regiao Serrana, January 2011")
    serrana = {"3303401": "Nova Friburgo", "3305802": "Teresopolis",
               "3303906": "Petropolis"}
    jan11 = mm.filter(
        pl.col("munic_code").is_in(list(serrana))
        & (pl.col("period") == date(2011, 1, 1))
    )
    for r in jan11.iter_rows(named=True):
        print(f"      {serrana[r['munic_code']]:<14} events={r['n_events']}")
    missing = set(serrana) - set(jan11.filter(pl.col("n_events") > 0)["munic_code"])
    if missing:
        print(f"      WARNING: no January-2011 declaration for "
              f"{[serrana[m] for m in missing]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
