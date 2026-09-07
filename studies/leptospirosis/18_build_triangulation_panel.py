"""RQ3 - assemble the three-system municipality-year triangulation panel.

Leptospirosis reaches three national systems by three different routes, and
they fail independently:

SINAN
    A notifiable-disease notification. Requires that a clinician suspected
    leptospirosis and that somebody filed the form.

SIM
    A death certificate whose *underlying* cause is coded CID-10 A27. Requires
    a death and a coder. No notification is involved.

SIH
    A public-hospital admission (AIH) carrying A27 as principal or secondary
    diagnosis. Requires an admission and a billing code. No notification is
    involved.

The panel produced here is the input to every RQ3 quantity. It is built on the
**complete municipal lattice x year**, so a municipality-year that appears in
no system survives as an explicit zero rather than as an absent row; the whole
research question is about which zeros are structural and which are
administrative, and a right-join would delete exactly the cells of interest.

Geography is municipality of **residence** in all three systems
(``ID_MN_RESI`` in SINAN, ``CODMUNRES`` in SIM, ``MUNIC_RES`` in SIH). This
must not reuse the canonical exposure panel: that panel intentionally assigns
SINAN cases to probable municipality of infection, with residence as fallback.
Mixing that exposure geography with residence-based SIM/SIH records creates
cross-system discordance by construction. Using SIH's establishment geography
(``MUNIC_MOV``) would similarly concentrate admissions in referral cities.

Coverage is unequal by construction and is written into the output:

* SINAN 2007-2025 (the study panel window).
* SIM final series 2007-2024. There is no final 2025.
* SIH current RD series 2008-2024. The 1992-2007 tree has a different record
  layout and is deliberately not concatenated; 2007 therefore has **no** SIH
  column, and 2025 is excluded to match SIM.

Every derived statistic must be computed inside the window where its sources
exist. ``sim_covered`` and ``sih_covered`` flags are carried per row so that a
downstream analysis cannot silently average a zero that means "system not
published that year".

Run:
    python studies/leptospirosis/18_build_triangulation_panel.py
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
from brepi.panel import spine as spine_mod

YEARS = range(2007, 2026)
SIM_YEARS = (2007, 2024)  # final DO series
SIH_YEARS = (2008, 2024)  # current RD series (200801-)

OUT_PANEL = PATHS.panel / "triangulation_municipality_year.parquet"
OUT_REPORT = PATHS.results / "rq3_ascertainment"


def _resolve(frame: pl.DataFrame, candidates, label: str) -> tuple[pl.DataFrame, dict]:
    """Attach a validated 7-digit municipality code, keeping losses explicit."""
    resolved = lattice.resolve_municipality_code(frame, candidates, lattice_year=2022)
    n_unresolved = int((resolved["munic_code_status"] == "unresolved").sum())
    out = resolved.filter(pl.col("munic_code_status") == "resolved").with_columns(
        lattice.code6_to_code7_expr(pl.col("munic_code6")).alias("munic_code")
    )
    audit = {
        "source": label,
        "records": frame.height,
        "resolved": out.height,
        "unresolved": n_unresolved,
        "code_field_used": (
            resolved.group_by("munic_code_source")
            .len()
            .sort("len", descending=True)
            .to_dicts()
        ),
    }
    return out, audit


def main() -> int:
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    audits: dict[str, object] = {}

    # ------------------------------------------------------------------ 1 ---
    # The canonical panel supplies the dated population and geography spine,
    # but NOT the SINAN numerator: its outcome geography is probable infection
    # with residence fallback. Triangulation requires residence in every arm.
    panel = pl.read_parquet(PATHS.panel / "lept_panel_municipality_month.parquet")
    geo = (
        panel.select(
            "munic_code", "name", "uf_code", "uf_abbr", "region",
            "health_region_code", "health_region_name",
        )
        .unique(subset=["munic_code"])
    )
    population = (
        panel.group_by("munic_code", "year")
        .agg(
            pl.col("population").mean().round(0).cast(pl.Int64).alias("population"),
        )
    )

    line_path = PATHS.interim / "lept_line_level.parquet"
    if not line_path.exists():
        raise SystemExit(
            f"{line_path} absent. Run 21_build_line_level.py; residence-aligned "
            "triangulation must not fall back to the exposure-geography panel."
        )
    line = pl.read_parquet(line_path)
    onset = pl.col("DT_SIN_PRI").cast(pl.Utf8).str.to_date(strict=False)
    confirmed = (
        pl.col("CLASSI_FIN").cast(pl.Utf8).str.strip_chars() == "1"
    )
    residence_valid = (
        (pl.col("municipality_residence_state") == "valid")
        & pl.col("municipality_residence_code7").is_not_null()
    )
    in_window = line.filter(confirmed).with_columns(onset.alias("_onset")).filter(
        pl.col("_onset").is_between(date(min(YEARS), 1, 1), date(max(YEARS), 12, 31))
    )
    unresolved_residence = in_window.filter(~residence_valid)
    sinan = (
        in_window.filter(residence_valid)
        .with_columns(
            pl.col("municipality_residence_code7").alias("munic_code"),
            pl.col("_onset").dt.year().alias("year"),
        )
        .group_by("munic_code", "year")
        .agg(
            pl.len().alias("sinan_confirmed"),
            (pl.col("EVOLUCAO").cast(pl.Utf8).str.strip_chars() == "2")
            .sum().alias("sinan_deaths"),
            (pl.col("ATE_HOSP").cast(pl.Utf8).str.strip_chars() == "1")
            .sum().alias("sinan_hospitalised"),
        )
    )
    audits["sinan_residence"] = {
        "source": "sinan",
        "geography": "municipality of residence",
        "confirmed_in_window": in_window.height,
        "resolved": int(sinan["sinan_confirmed"].sum()),
        "unresolved": unresolved_residence.height,
        "unresolved_states": (
            unresolved_residence.group_by("municipality_residence_state")
            .len().sort("len", descending=True).to_dicts()
        ),
    }
    municipalities = tuple(sorted(geo["munic_code"].to_list()))
    print(f"lattice: {len(municipalities)} municipalities x {len(list(YEARS))} years")

    spine = spine_mod.build_spine(
        spine_mod.SpineSpec(
            municipalities=municipalities,
            start=date(min(YEARS), 1, 1),
            end=date(max(YEARS), 1, 1),
            grain="year",
        )
    ).drop("uf_code")

    build = spine_mod.PanelBuild(spine=spine)
    build.add(
        sinan, name="sinan_municipality_year", on=("munic_code", "year"),
        fill={"sinan_confirmed": 0, "sinan_deaths": 0,
              "sinan_hospitalised": 0},
    )
    build.add(
        population, name="population", on=("munic_code", "year"),
        min_coverage=1.0,
    )
    build.add(geo, name="geography", on=("munic_code",), min_coverage=1.0)

    # ------------------------------------------------------------------ 2 ---
    # SIM: A27 underlying-cause deaths, by municipality of residence and by
    # year of death. The year is taken from DTOBITO where present rather than
    # from the file's year: a death registered late is filed in a later DO but
    # belongs to the calendar year it happened in.
    sim_path = PATHS.interim / "sim_a27_deaths.parquet"
    if not sim_path.exists():
        raise SystemExit(
            f"{sim_path} absent. Run 05_triangulation_extract.py --sim first; "
            "RQ3 has no mortality arm without it."
        )
    sim_raw = pl.read_parquet(sim_path)
    # The Paper A estimand is residence-aligned.  Municipality of occurrence
    # is retained in the extract for audit only and must never rescue a missing
    # or invalid residence code: doing so would silently change the estimand.
    sim_res, sim_audit = _resolve(
        sim_raw, [("CODMUNRES", "CODMUNRES")], "sim"
    )
    audits["sim"] = sim_audit
    sim_year = (
        pl.col("DTOBITO").cast(pl.Utf8).str.strip_chars().str.slice(4, 4)
        .cast(pl.Int32, strict=False)
    )
    sim_res = sim_res.with_columns(
        pl.when(sim_year.is_between(1996, 2026))
        .then(sim_year)
        .otherwise(pl.col("_src_year"))
        .alias("year")
    )
    sim_mun = (
        sim_res.filter(pl.col("year").is_between(min(YEARS), max(YEARS)))
        .group_by("munic_code", "year")
        .agg(pl.len().alias("sim_a27_deaths"))
    )
    print(f"SIM: {sim_raw.height:,} A27 deaths -> {int(sim_mun['sim_a27_deaths'].sum()):,} "
          f"placed in {sim_mun.height:,} municipality-years")
    build.add(
        sim_mun, name="sim_a27_deaths", on=("munic_code", "year"),
        fill={"sim_a27_deaths": 0},
    )

    # ------------------------------------------------------------------ 3 ---
    # SIH: A27 admissions, by municipality of residence and admission year.
    # Two counts are carried: the principal-diagnosis count (a stricter,
    # comparable-to-SIM definition) and the any-diagnosis count. Leptospirosis
    # is frequently a secondary code on an admission filed principally as renal
    # failure or as an unspecified febrile illness, so the two differ a lot and
    # a single number would hide which definition produced it.
    sih_path = PATHS.interim / "sih_a27_admissions.parquet"
    sih_years_present: list[int] = []
    if sih_path.exists():
        sih_raw = pl.read_parquet(sih_path)
        # Likewise, hospital municipality (MUNIC_MOV) is not a fallback for
        # residence. Referral-centre geography would manufacture discordance.
        sih_res, sih_audit = _resolve(
            sih_raw, [("MUNIC_RES", "MUNIC_RES")], "sih"
        )
        audits["sih"] = sih_audit
        # Admission date is the clinical event time. Competence year is a
        # billing period and can cross calendar years (153 AIHs in the 2008
        # files began in 2007), so it cannot define temporal source order.
        # DT_INTER is complete in the frozen A27 extract; a parse failure is
        # an explicit exclusion rather than a silent competence fallback.
        sih_res = sih_res.with_columns(
            pl.col("DT_INTER").cast(pl.Utf8).str.to_date(
                "%Y%m%d", strict=False
            ).alias("_admission_date"),
            pl.col("DIAG_PRINC").cast(pl.Utf8).str.starts_with("A27")
            .fill_null(False).alias("principal"),
            (pl.col("MORTE").cast(pl.Utf8).str.strip_chars() == "1")
            .fill_null(False).alias("died"),
        ).with_columns(
            pl.col("_admission_date").dt.year().alias("year"),
        )
        sih_mun = (
            sih_res.filter(pl.col("year").is_between(max(2008, min(YEARS)), 2024))
            .group_by("munic_code", "year")
            .agg(
                pl.len().alias("sih_a27_admissions"),
                pl.col("principal").sum().alias("sih_a27_principal"),
                pl.col("died").sum().alias("sih_a27_deaths_in_hospital"),
            )
        )
        sih_years_present = sorted(
            sih_res.filter(pl.col("year").is_between(max(2008, min(YEARS)), 2024))["year"]
            .unique().to_list()
        )
        print(f"SIH: {sih_raw.height:,} A27 admissions -> "
              f"{int(sih_mun['sih_a27_admissions'].sum()):,} placed, "
              f"years {min(sih_years_present)}-{max(sih_years_present)}")
        build.add(
            sih_mun, name="sih_a27_admissions", on=("munic_code", "year"),
            fill={"sih_a27_admissions": 0, "sih_a27_principal": 0,
                  "sih_a27_deaths_in_hospital": 0},
        )
    else:
        print(f"SIH: {sih_path} absent - the panel will carry null SIH columns "
              "and every SIH-derived statistic must be reported as unavailable.")
        build.spine = build.spine.with_columns(
            pl.lit(None, dtype=pl.UInt32).alias("sih_a27_admissions"),
            pl.lit(None, dtype=pl.UInt32).alias("sih_a27_principal"),
            pl.lit(None, dtype=pl.UInt32).alias("sih_a27_deaths_in_hospital"),
        )

    # ------------------------------------------------------------------ 4 ---
    # Coverage flags. A zero inside a covered year is a real zero; a zero
    # outside it is "system not published". These flags are the only thing
    # standing between an honest denominator and a fabricated one.
    tri = build.spine
    sih_lo = min(sih_years_present) if sih_years_present else None
    sih_hi = max(sih_years_present) if sih_years_present else None
    tri = tri.with_columns(
        pl.col("year").is_between(*SIM_YEARS).alias("sim_covered"),
        (
            pl.col("year").is_between(sih_lo, sih_hi)
            if sih_lo is not None
            else pl.lit(False)
        ).alias("sih_covered"),
        pl.lit(True).alias("sinan_covered"),
    ).with_columns(
        pl.when(pl.col("sim_covered")).then(pl.col("sim_a27_deaths"))
        .otherwise(None).alias("sim_a27_deaths"),
        pl.when(pl.col("sih_covered")).then(pl.col("sih_a27_admissions"))
        .otherwise(None).alias("sih_a27_admissions"),
        pl.when(pl.col("sih_covered")).then(pl.col("sih_a27_principal"))
        .otherwise(None).alias("sih_a27_principal"),
        pl.when(pl.col("sih_covered")).then(pl.col("sih_a27_deaths_in_hospital"))
        .otherwise(None).alias("sih_a27_deaths_in_hospital"),
    )

    expected = len(municipalities) * len(list(YEARS))
    if tri.height != expected:
        raise AssertionError(f"panel has {tri.height} rows, expected {expected}")

    tri = tri.select(
        "munic_code", "name", "uf_code", "uf_abbr", "region",
        "health_region_code", "health_region_name", "year", "population",
        "sinan_confirmed", "sinan_deaths", "sinan_hospitalised",
        "sim_a27_deaths", "sih_a27_admissions", "sih_a27_principal",
        "sih_a27_deaths_in_hospital",
        "sinan_covered", "sim_covered", "sih_covered",
    ).sort("munic_code", "year")

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    tri.write_parquet(OUT_PANEL)
    print(f"\nwrote {OUT_PANEL}  {tri.height:,} rows x {tri.width} columns")

    for rep in build.reports:
        print("   " + rep.summary())

    coverage = {
        "rows": tri.height,
        "municipalities": len(municipalities),
        "years": [min(YEARS), max(YEARS)],
        "sinan_years": [min(YEARS), max(YEARS)],
        "sim_years": list(SIM_YEARS),
        "sih_years": [sih_lo, sih_hi],
        "sih_complete_national": sih_lo == SIH_YEARS[0] and sih_hi == SIH_YEARS[1],
        "totals": {
            "sinan_confirmed": int(tri["sinan_confirmed"].sum()),
            "sinan_deaths": int(tri["sinan_deaths"].sum()),
            "sim_a27_deaths": int(tri["sim_a27_deaths"].sum()),
            "sih_a27_admissions": (
                int(tri["sih_a27_admissions"].sum())
                if sih_lo is not None else None
            ),
        },
        "join_reports": [
            {
                "source": r.source, "coverage": round(r.coverage, 6),
                "unmatched_source_keys": r.unmatched_source_keys,
            }
            for r in build.reports
        ],
        "geocoding": audits,
    }
    (OUT_REPORT / "00_panel_coverage.json").write_text(
        json.dumps(coverage, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in coverage.items()
                      if k not in ("geocoding", "join_reports")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
