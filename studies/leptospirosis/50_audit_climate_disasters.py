"""Audit of the CLIMATE and DISASTER data surfaces: variables, not files.

Written for the same reason as ``46_variable_catalogue.py``. The study's data
document describes *files* -- "BR-DWGD, 2.16 GB", "Atlas, 2 files" -- and a
document that describes files but not variables lets a study leave half its
evidence untouched. Here the untouched half is large and specific:

* BR-DWGD publishes **seven** meteorological variables. One is cached.
  Temperature -- which governs *Leptospira* survival in standing water and is
  the single most-cited environmental modifier after rainfall -- is not.
* The cached BR-DWGD precipitation artefact carries **six zonal statistics**
  per municipality-day. The study reads one (``mean``). ``max`` is the wettest
  0.1-degree cell inside the municipality on that day, ``stdev`` its spatial
  dispersion, ``count`` its effective cell coverage. All three are on disk and
  none is read.
* ERA5-Land daily **tmax and tmin for 2023, 2024 and 2025 are already in the
  cache** (six files, 220 MB) and are read by nothing.
* The Atlas export has **70 columns**. The adapter carries 13 of them and the
  built event tables carry 8. Everything that measures *how big* a disaster
  was -- houses destroyed, people made homeless, health facilities damaged,
  money lost from the water-supply, sewerage, refuse-collection and
  **pest-control** systems -- is dropped before it reaches any analysis.

Every number written here traces to a cached byte or to a Zenodo record
listing fetched at run time (a few kB of JSON, not a re-download). Nothing is
estimated unless the column says so.

Populations are named explicitly and never mixed:

``panel``        all 5,570 x 228 = 1,269,960 municipality-months, 2007-01..2025-12
``panel_cases``  the subset with at least one confirmed case
``atlas_all``    all 76,191 Atlas protocol rows, 1991-2025
``atlas_flood``  flood-COBRADE protocol rows with an event date in 2007-2025
``atlas_drought`` drought-COBRADE protocol rows, same window

Run from ``brepi/``::

    PYTHONPATH=. python studies/leptospirosis/50_audit_climate_disasters.py
    PYTHONPATH=. python studies/leptospirosis/50_audit_climate_disasters.py --offline

``--offline`` skips the Zenodo record listing and uses the copy recorded in
:data:`ZENODO_RECORDED`, which was fetched 2026-08-14 and is byte-verified
against the one artefact that is cached.

Outputs to ``data/results/audit_climate_disasters/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi import paths
from brepi.config import (
    BRDWGD_ZONAL_ZENODO_RECORD,
    ERA5LAND_ZONAL_ZENODO_RECORDS,
    ZENODO_RECORD_BASE,
)
from brepi.sources.climate import brdwgd as brdwgd_mod
from brepi.sources.disasters import atlas as atlas_mod

OUT = paths.RESULTS / "audit_climate_disasters"
CACHE = paths.CACHE

PANEL = paths.PANEL_CLIMATE
ATLAS_CSV = CACHE / "disasters" / "atlas" / "atlas_consolidado.csv"

STUDY_START, STUDY_END = date(2007, 1, 1), date(2025, 12, 31)

#: Zenodo record listings as fetched 2026-08-14 from ``ZENODO_RECORD_BASE``.
#: Kept so ``--offline`` is honest rather than silent. The ``pr_3.2.3.parquet``
#: size below equals the byte count of the cached artefact exactly, which is
#: what makes this table verifiable rather than remembered.
ZENODO_RECORDED: dict[str, dict[str, int]] = {
    "13906834": {
        "ETo_3.2.3.parquet": 2007739347,
        "RH_3.2.3.parquet": 1948973467,
        "Rs_3.2.3.parquet": 1955508718,
        "Tmax_3.2.3.parquet": 2166594906,
        "Tmin_3.2.3.parquet": 2188281232,
        "pr_3.2.3.parquet": 2163690068,
        "u2_3.2.3.parquet": 1879775556,
    },
    "10036212": {
        "10m_u_component_of_wind_mean.parquet": 2933604489,
        "10m_v_component_of_wind_mean.parquet": 2933644435,
        "2m_dewpoint_temperature_mean.parquet": 2933553292,
        "2m_temperature_max.parquet": 2933247722,
        "2m_temperature_mean.parquet": 2933257428,
        "2m_temperature_min.parquet": 2933251745,
        "surface_pressure_mean.parquet": 2932881383,
        "total_precipitation_sum.parquet": 3235006800,
    },
    "10947952": {
        "10m_u_component_of_wind_mean.parquet": 39935852,
        "10m_v_component_of_wind_mean.parquet": 39935854,
        "2m_dewpoint_temperature_mean.parquet": 39934444,
        "2m_temperature_max.parquet": 39931038,
        "2m_temperature_mean.parquet": 39931469,
        "2m_temperature_min.parquet": 39930053,
        "surface_pressure_mean.parquet": 39927732,
        "total_precipitation_sum.parquet": 42785885,
    },
    "15748125": {
        "10m_u_component_of_wind_mean.parquet": 34629779,
        "10m_v_component_of_wind_mean.parquet": 34629567,
        "2m_dewpoint_temperature_mean.parquet": 34628831,
        "2m_temperature_max.parquet": 34623727,
        "2m_temperature_mean.parquet": 34624640,
        "2m_temperature_min.parquet": 34624244,
        "surface_pressure_mean.parquet": 34619539,
        "total_precipitation_sum.parquet": 37447061,
    },
    "18257037": {
        "10m_u_component_of_wind_mean.parquet": 35581690,
        "10m_v_component_of_wind_mean.parquet": 35581742,
        "2m_dewpoint_temperature_mean.parquet": 35581589,
        "2m_temperature_max.parquet": 35577345,
        "2m_temperature_mean.parquet": 35577686,
        "2m_temperature_min.parquet": 35576247,
        "surface_pressure_mean.parquet": 35576328,
        "total_precipitation_sum.parquet": 39459472,
    },
}

#: Records to enumerate: the BR-DWGD record plus every ERA5-Land annual record
#: the config declares. Asserted against the config below so a new annual
#: record cannot be added upstream without this audit noticing.
RECORD_LABEL = {
    "13906834": ("brdwgd", "BR-DWGD v3.2.3 zonal, 1961-01-01..2024-03-20"),
    "10036212": ("era5land", "ERA5-Land zonal, 1950..2022(+1 Jan 2023)"),
    "10947952": ("era5land", "ERA5-Land zonal, 2023"),
    "15748125": ("era5land", "ERA5-Land zonal, 2024"),
    "18257037": ("era5land", "ERA5-Land zonal, 2025"),
}

_declared_records = {BRDWGD_ZONAL_ZENODO_RECORD, *ERA5LAND_ZONAL_ZENODO_RECORDS.values()}
if _declared_records - set(RECORD_LABEL):
    raise SystemExit(
        f"brepi.config declares climate records {sorted(_declared_records - set(RECORD_LABEL))} "
        "that this audit does not enumerate. Add them to RECORD_LABEL."
    )

#: What each published artefact measures, and why leptospirosis cares. Keyed on
#: the publisher's filename because that is the only stable identifier.
VARIABLE_MEANING: dict[str, tuple[str, str]] = {
    "pr_3.2.3.parquet": ("precipitation, mm/day", "the exposure the study uses"),
    "Tmax_3.2.3.parquet": (
        "daily maximum air temperature, degC",
        "Leptospira survival in water and soil falls sharply above ~35 degC and "
        "below ~10 degC; temperature is the standard effect modifier of the "
        "rainfall-leptospirosis association",
    ),
    "Tmin_3.2.3.parquet": (
        "daily minimum air temperature, degC",
        "with Tmax gives the diurnal range and the cold cut-off on survival",
    ),
    "RH_3.2.3.parquet": (
        "relative humidity, %",
        "governs desiccation of the spirochaete on wet soil between rain events",
    ),
    "Rs_3.2.3.parquet": (
        "global solar radiation, MJ/m2/day",
        "UV inactivation of Leptospira in shallow standing water",
    ),
    "u2_3.2.3.parquet": ("wind speed at 2 m, m/s", "an evaporative-demand term, weak direct relevance"),
    "ETo_3.2.3.parquet": (
        "reference evapotranspiration, mm/day",
        "with precipitation gives a water balance: how long standing water "
        "persists after rain, which is the actual exposure window",
    ),
    "total_precipitation_sum.parquet": ("precipitation, m/day (x1000 -> mm)", "the ERA5-Land tail of the exposure"),
    "2m_temperature_max.parquet": ("daily maximum 2 m temperature, K", "as Tmax; CACHED FOR 2023-2025 AND UNUSED"),
    "2m_temperature_min.parquet": ("daily minimum 2 m temperature, K", "as Tmin; CACHED FOR 2023-2025 AND UNUSED"),
    "2m_temperature_mean.parquet": ("daily mean 2 m temperature, K", "the simplest temperature covariate; not cached"),
    "2m_dewpoint_temperature_mean.parquet": (
        "daily mean 2 m dewpoint, K",
        "with 2m_temperature gives relative humidity for the ERA5 era",
    ),
    "10m_u_component_of_wind_mean.parquet": ("zonal wind at 10 m, m/s", "weak direct relevance"),
    "10m_v_component_of_wind_mean.parquet": ("meridional wind at 10 m, m/s", "weak direct relevance"),
    "surface_pressure_mean.parquet": ("surface pressure, Pa", "no direct relevance; an altitude proxy at best"),
}

#: Definitions of the climate columns actually present in the RQ1 panel, and
#: their downstream fate. "used" means it enters a fitted model; "descriptive
#: only" means it is summarised in an atlas or figure but never modelled.
PANEL_CLIMATE_COLUMNS: dict[str, tuple[str, str]] = {
    "precip_mm": (
        "monthly precipitation total, mm, areal mean over the municipality's "
        "0.1-deg cells, calendar-prorated where source days are missing",
        "used: the RQ1/RQ2 crossbasis exposure (R/02_crossbasis.R, "
        "15_rq2_did.R), on a unit-relative (x - median)/IQR scale",
    ),
    "precip_mm_observed": (
        "the same total before proration, i.e. the sum over observed days only",
        "available_unused: exists for the proration sensitivity, never run",
    ),
    "climate_coverage": (
        "observed days / calendar days in the month",
        "used as a flag only; never entered as a weight",
    ),
    "precip_missing_days": (
        "calendar days minus observed days",
        "available_unused",
    ),
    "rx1day": (
        "ETCCDI RX1day: wettest single day in the month, mm (areal mean of that day)",
        "descriptive only: annual mean in the municipality atlas; no model",
    ),
    "r1": ("ETCCDI R1mm: days with >= 1 mm", "unused anywhere downstream"),
    "r10": ("ETCCDI R10mm: days with >= 10 mm", "unused anywhere downstream"),
    "r20": ("ETCCDI R20mm: days with >= 20 mm", "descriptive only (atlas annual mean)"),
    "r50": (
        "days with >= 50 mm (Brazilian operational heavy-rain alert threshold; "
        "ETCCDI leaves Rnn user-defined)",
        "descriptive only (atlas annual mean)",
    ),
    "climate_product": ("which product supplied the month: brdwgd / era5land / both / proxy", "used as a splice flag"),
    "territorial_imputation": ("parent-municipality proxy tag for units absent from the source lattice", "used as a flag"),
    "territorial_source_codes": ("the parent codes the proxy came from", "provenance"),
    "territorial_source_count": ("how many parents contributed", "provenance"),
    "precip_prorated": ("month had missing source days and was prorated", "used as a flag"),
    "territorial_imputed": ("month uses a parent proxy", "used as a flag"),
}

#: All 70 Atlas columns: an English gloss and an audit category. Nobody in this
#: project had enumerated these; the adapter carries 13 and the built event
#: tables carry 8.
ATLAS_COLUMNS: dict[str, tuple[str, str]] = {
    "Protocolo_S2iD": ("S2iD protocol number; the row identity", "identity"),
    "Nome_Municipio": ("municipality name", "geography"),
    "Sigla_UF": ("state abbreviation", "geography"),
    "regiao": ("macroregion", "geography"),
    "Data_Registro": ("date the record entered S2iD", "timing"),
    "Data_Evento": ("date of occurrence declared by the municipality", "timing"),
    "Cod_Cobrade": ("COBRADE disaster code, dots stripped", "classification"),
    "tipologia": ("short typology label", "classification"),
    "descricao_tipologia": ("long typology description", "classification"),
    "grupo_de_desastre": ("Atlas disaster group (Hidrologico, Climatologico, ...)", "classification"),
    "Cod_IBGE_Mun": ("IBGE municipality code", "geography"),
    "Setores Censitários": ("census tracts affected (sub-municipal footprint)", "geography_subunit"),
    "Status": ("Registro (filed) or Reconhecido (federally recognised)", "recognition"),
    "DH_Descricao": ("free-text description of human damage", "text"),
    "DH_MORTOS": ("deaths", "human_damage"),
    "DH_FERIDOS": ("injured", "human_damage"),
    "DH_ENFERMOS": ("ill", "human_damage"),
    "DH_DESABRIGADOS": ("homeless (lost the dwelling; in public shelter)", "human_damage"),
    "DH_DESALOJADOS": ("displaced (left the dwelling; own arrangements)", "human_damage"),
    "DH_DESAPARECIDOS": ("missing", "human_damage"),
    "DH_AFETADOS_SECA_ESTIAGEM": ("affected by drought/dry spell", "human_damage"),
    "DH_total_danos_humanos_diretos": ("total direct human damage", "human_damage"),
    "DH_OUTROS AFETADOS": ("otherwise affected", "human_damage"),
    "DM_Descricao": ("free-text description of material damage", "text"),
    "DM_Uni Habita Danificadas": ("housing units damaged", "material_damage"),
    "DM_Uni Habita Destruidas": ("housing units destroyed", "material_damage"),
    "DM_Uni Habita Valor": ("value of housing damage, BRL", "material_damage_value"),
    "DM_Inst Saúde Danificadas": ("health facilities damaged", "material_damage"),
    "DM_Inst Saúde Destruidas": ("health facilities destroyed", "material_damage"),
    "DM_Inst Saúde Valor": ("value of health-facility damage, BRL", "material_damage_value"),
    "DM_Inst Ensino Danificadas": ("school facilities damaged", "material_damage"),
    "DM_Inst Ensino Destruidas": ("school facilities destroyed", "material_damage"),
    "DM_Inst Ensino Valor": ("value of school damage, BRL", "material_damage_value"),
    "DM_Inst Serviços Danificadas": ("service installations damaged", "material_damage"),
    "DM_Inst Serviços Destruidas": ("service installations destroyed", "material_damage"),
    "DM_Inst Serviços Valor": ("value of service-installation damage, BRL", "material_damage_value"),
    "DM_Inst Comuni Danificadas": ("community installations damaged", "material_damage"),
    "DM_Inst Comuni Destruidas": ("community installations destroyed", "material_damage"),
    "DM_Inst Comuni Valor": ("value of community-installation damage, BRL", "material_damage_value"),
    "DM_Obras de Infra Danificadas": ("infrastructure works damaged", "material_damage"),
    "DM_Obras de Infra Destruidas": ("infrastructure works destroyed", "material_damage"),
    "DM_Obras de Infra Valor": ("value of infrastructure damage, BRL", "material_damage_value"),
    "DM_total_danos_materiais": ("total material damage, BRL", "material_damage_value"),
    "DA_Descricao": ("free-text description of environmental damage", "text"),
    "DA_Polui/cont da água": ("water pollution/contamination severity code", "environmental_damage"),
    "DA_Polui/cont do ar": ("air pollution/contamination severity code", "environmental_damage"),
    "DA_Polui/cont do solo": ("soil pollution/contamination severity code", "environmental_damage"),
    "DA_Dimi/exauri hídrico": ("reduction/exhaustion of water resources severity code", "environmental_damage"),
    "DA_Incêndi parques/APA's/APP's": ("fire in protected areas severity code", "environmental_damage"),
    "PEPL_Descricao": ("free-text description of public economic loss", "text"),
    "PEPL_Assis_méd e emergên(R$)": ("public loss: medical and emergency assistance, BRL", "public_loss"),
    "PEPL_Abast de água pot(R$)": ("public loss: potable water supply, BRL", "public_loss"),
    "PEPL_sist de esgotos sanit(R$)": ("public loss: sanitary sewerage system, BRL", "public_loss"),
    "PEPL_Sis limp e rec lixo (R$)": ("public loss: cleaning and refuse collection, BRL", "public_loss"),
    "PEPL_Sis cont pragas (R$)": ("public loss: PEST CONTROL system, BRL", "public_loss"),
    "PEPL_distrib energia (R$)": ("public loss: electricity distribution, BRL", "public_loss"),
    "PEPL_Telecomunicações (R$)": ("public loss: telecommunications, BRL", "public_loss"),
    "PEPL_Tran loc/reg/l_curso (R$)": ("public loss: local/regional transport, BRL", "public_loss"),
    "PEPL_Distrib combustíveis(R$)": ("public loss: fuel distribution, BRL", "public_loss"),
    "PEPL_Segurança pública (R$)": ("public loss: public security, BRL", "public_loss"),
    "PEPL_Ensino (R$)": ("public loss: education, BRL", "public_loss"),
    "PEPL_total_publico": ("total public economic loss, BRL", "public_loss"),
    "PEPR_Descricao": ("free-text description of private economic loss", "text"),
    "PEPR_Agricultura (R$)": ("private loss: agriculture, BRL", "private_loss"),
    "PEPR_Pecuária (R$)": ("private loss: livestock, BRL", "private_loss"),
    "PEPR_Indústria (R$)": ("private loss: industry, BRL", "private_loss"),
    "PEPR_Comércio (R$)": ("private loss: commerce, BRL", "private_loss"),
    "PEPR_Serviços (R$)": ("private loss: services, BRL", "private_loss"),
    "PEPR_total_privado": ("total private economic loss, BRL", "private_loss"),
    "PE_PLePR": ("total economic loss, public + private, BRL", "total_loss"),
}

#: The five DA_ columns are **ordinal text**, not numbers: their values are
#: population-share bands ("MAIS DE 20% DA POPULACAO AFETADA"). Parsing them as
#: numerics -- which is what a damage-column loop naturally does -- silently
#: scores them as all-zero. They are profiled separately.
ATLAS_ORDINAL = tuple(c for c, (_g, cat) in ATLAS_COLUMNS.items() if cat == "environmental_damage")

#: Where each Atlas column ends up. The three tiers are: decoded by the adapter
#: AND present in a built table; decoded by the adapter and then dropped when
#: the event tables were written; never decoded at all.
ATLAS_FATE: dict[str, str] = {
    "Protocolo_S2iD": "adapter: event_id; dropped from the built event tables",
    "Cod_IBGE_Mun": "event tables, as munic_code",
    "Sigla_UF": "event tables, as uf_abbr",
    "Data_Evento": "event tables, as date",
    "Data_Registro": "adapter: register_date + register_lag_days; dropped at 14b",
    "Cod_Cobrade": "event tables, as cobrade / cobrade_label",
    "Status": "event tables, as recognised",
    "DH_MORTOS": "event tables, as deaths",
    "DH_FERIDOS": "adapter: injured; reaches the event tables only inside affected_total",
    "DH_ENFERMOS": "adapter: ill; reaches the event tables only inside affected_total",
    "DH_DESABRIGADOS": "adapter: homeless; reaches the event tables only inside affected_total",
    "DH_DESALOJADOS": "adapter: displaced; reaches the event tables only inside affected_total",
    "DH_OUTROS AFETADOS": "adapter: affected_other; reaches the event tables only inside affected_total",
    "DH_DESAPARECIDOS": "adapter: missing; dropped at 14b",
    "DH_AFETADOS_SECA_ESTIAGEM": "adapter: affected_drought; dropped at 14b",
    "DH_total_danos_humanos_diretos": "adapter: affected_direct; dropped at 14b",
    "DM_total_danos_materiais": "adapter: damage_material_brl; kept in flood_declarations, absent from flood_events",
    "PEPL_total_publico": "adapter: damage_public_brl; dropped at 14b",
    "PEPR_total_privado": "adapter: damage_private_brl; dropped at 14b",
    "PE_PLePR": "adapter: damage_economic_brl; kept in flood_declarations, absent from flood_events",
    "regiao": "adapter: region; dropped at 14b",
    "Nome_Municipio": "adapter: munic_name; dropped at 14b",
    "tipologia": "adapter: typology; dropped at 14b",
    "grupo_de_desastre": "adapter: disaster_group; dropped at 14b",
}
ADAPTER_CARRIES = set(atlas_mod._IMPACT_COLUMNS)  # noqa: SLF001 - auditing the module's own map
EVENT_TABLE_COLUMNS = ("munic_code", "date", "cobrade", "cobrade_label", "recognised", "uf_abbr", "deaths", "affected_total")

#: Above this a count field is not credible for a single municipality-protocol
#: (Brazil has ~350,000 health establishments in total, and no municipality has
#: 7,000 of them). Used only to *flag* contamination, never to edit values.
IMPLAUSIBLE_FACILITY_COUNT = 500

TEXT_BLANKS = {"", "-", "--", "NA", "N/A", "null", "NULL", "nan"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(frame: pl.DataFrame, stem: str) -> None:
    frame.write_csv(OUT / f"{stem}.csv")
    print(f"  wrote {stem}.csv  ({frame.height} x {frame.width})")


def _connect():
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    return con


# --------------------------------------------------------------------------
# 1. What the publishers offer, and what is on disk
# --------------------------------------------------------------------------


def zenodo_listing(offline: bool) -> tuple[dict[str, dict[str, int]], str]:
    """``{record: {filename: bytes}}`` for every climate record the study uses.

    A record listing is a few kB of JSON, not a re-download of the artefacts.
    ``--offline`` falls back to :data:`ZENODO_RECORDED` and says so.
    """
    if offline:
        return ZENODO_RECORDED, "recorded 2026-08-14 (--offline)"
    import httpx

    out: dict[str, dict[str, int]] = {}
    try:
        for record in RECORD_LABEL:
            response = httpx.get(
                f"{ZENODO_RECORD_BASE}/{record}", timeout=30, follow_redirects=True
            )
            response.raise_for_status()
            out[record] = {f["key"]: int(f["size"]) for f in response.json().get("files", [])}
        return out, f"zenodo API, fetched {_now()}"
    except Exception as exc:  # noqa: BLE001 - network is optional here
        print(f"  ! Zenodo listing unavailable ({type(exc).__name__}); using the recorded copy")
        return ZENODO_RECORDED, "recorded 2026-08-14 (fetch failed)"


def published_inventory(listing: dict[str, dict[str, int]], provenance: str) -> pl.DataFrame:
    """One row per published artefact, with whether it is cached and what it measures."""
    rows: list[dict[str, object]] = []
    for record, files in listing.items():
        product, coverage = RECORD_LABEL[record]
        for filename, size in sorted(files.items()):
            cached_path = CACHE / "climate" / product / record / filename
            cached = cached_path.exists()
            measures, relevance = VARIABLE_MEANING.get(filename, ("", ""))
            rows.append(
                {
                    "product": product,
                    "record": record,
                    "record_coverage": coverage,
                    "file": filename,
                    "published_bytes": size,
                    "published_gb": round(size / 1e9, 3),
                    "cached": cached,
                    "cached_bytes": cached_path.stat().st_size if cached else None,
                    "bytes_match": (cached_path.stat().st_size == size) if cached else None,
                    "measures": measures,
                    "leptospirosis_relevance": relevance,
                    "listing_provenance": provenance,
                }
            )
    return pl.DataFrame(rows).sort(["product", "record", "file"])


def cache_profile() -> pl.DataFrame:
    """Parquet-metadata profile of every cached climate artefact.

    Reads footers only: statistic names, date span and code span come from
    row-group statistics, so this costs milliseconds on a 2.16 GB file.
    """
    import pyarrow.parquet as pq

    rows: list[dict[str, object]] = []
    for path in sorted((CACHE / "climate").rglob("*.parquet")):
        md = pq.ParquetFile(path).metadata
        names: set[str] = set()
        dmin = dmax = None
        cmin = cmax = None
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            for j in range(md.num_columns):
                col = rg.column(j)
                st = col.statistics
                if st is None:
                    continue
                if col.path_in_schema == "name":
                    names.update({st.min, st.max})
                elif col.path_in_schema == "date":
                    dmin = st.min if dmin is None or st.min < dmin else dmin
                    dmax = st.max if dmax is None or st.max > dmax else dmax
                elif col.path_in_schema == "code_muni":
                    cmin = st.min if cmin is None or st.min < cmin else cmin
                    cmax = st.max if cmax is None or st.max > cmax else cmax
        n_stats = len(names)
        span_days = (dmax - dmin).days + 1 if dmin and dmax else None
        # rows = municipalities x days x statistics, so the mesh size follows.
        implied_units = (
            md.num_rows / (span_days * n_stats) if span_days and n_stats else None
        )
        rows.append(
            {
                "path": str(path.relative_to(paths.ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "gb": round(path.stat().st_size / 1e9, 3),
                "rows": md.num_rows,
                "row_groups": md.num_row_groups,
                "statistics": ";".join(sorted(names)),
                "n_statistics": n_stats,
                "date_min": dmin,
                "date_max": dmax,
                "span_days": span_days,
                "code_min": cmin,
                "code_max": cmax,
                "implied_municipalities": round(implied_units, 2) if implied_units else None,
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. The cached BR-DWGD artefact: six statistics, one read
# --------------------------------------------------------------------------

BRDWGD_PR = CACHE / "climate" / "brdwgd" / BRDWGD_ZONAL_ZENODO_RECORD / "pr_3.2.3.parquet"

STAT_MEANING = {
    "pr_3.2.3_mean": "areal MEAN of the daily grid cells in the polygon, mm -- the only statistic the study reads",
    "pr_3.2.3_max": "WETTEST single 0.1-deg cell inside the municipality that day, mm",
    "pr_3.2.3_min": "driest cell inside the municipality that day, mm",
    "pr_3.2.3_stdev": "spatial standard deviation across the municipality's cells, mm",
    "pr_3.2.3_sum": "sum over cells; scales with municipal area and is NOT an exposure",
    "pr_3.2.3_count": "effective (area-weighted) number of grid cells covering the polygon",
    "pr_3.2.3_sd": "NOT A REAL NAME -- what brdwgd.ClimateProduct.stats declares as 'sd'",
}


def brdwgd_statistic_probe(con, probe_year: int = 2011) -> pl.DataFrame:
    """Which statistic names the cached artefact actually holds, and their scale.

    Probed on one calendar year so the cost is a row-group-pruned scan rather
    than a 771-million-row read. Includes the name the adapter *declares*
    (``..._sd``) to demonstrate that asking for it returns silence, not an
    error.
    """
    src = str(BRDWGD_PR).replace("\\", "/")
    frame = con.execute(
        f"""
        SELECT name,
               count(*) AS n_rows,
               count(DISTINCT code_muni) AS n_municipalities,
               count(DISTINCT date) AS n_days,
               sum(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS n_null,
               min(value) AS v_min,
               quantile_cont(value, 0.5) AS v_median,
               max(value) AS v_max
        FROM read_parquet('{src}')
        WHERE date BETWEEN DATE '{probe_year}-01-01' AND DATE '{probe_year}-12-31'
        GROUP BY name ORDER BY name
        """
    ).pl()
    declared = {f"pr_3.2.3_{s}" for s in brdwgd_mod.PRODUCTS["brdwgd"].stats}
    present = set(frame["name"].to_list())
    missing = pl.DataFrame(
        [
            {
                "name": name,
                "n_rows": 0,
                "n_municipalities": 0,
                "n_days": 0,
                "n_null": 0,
                "v_min": None,
                "v_median": None,
                "v_max": None,
            }
            for name in sorted(declared - present)
        ],
        schema=frame.schema,
    )
    out = pl.concat([frame, missing], how="vertical_relaxed") if missing.height else frame
    return out.with_columns(
        pl.col("name").replace_strict(STAT_MEANING, default="").alias("meaning"),
        pl.col("name").is_in(list(declared)).alias("declared_by_adapter"),
        (pl.col("n_rows") > 0).alias("present_in_artefact"),
        pl.lit(probe_year).alias("probe_year"),
        pl.lit("pr_3.2.3_mean").eq(pl.col("name")).alias("read_by_the_study"),
    ).sort("name")


def brdwgd_within_municipality(con, probe_year: int = 2011) -> tuple[pl.DataFrame, dict]:
    """How much the areal mean hides: mean-of-cells versus wettest-cell.

    For every municipality in ``probe_year`` this compares the study's RX1day
    (the wettest daily *areal mean*) with the wettest daily *cell maximum* in
    the same month-year, and reports the ratio. The gap is the intra-municipal
    rainfall heterogeneity the exposure currently averages away -- and it is
    largest exactly where municipalities are large, which is where a
    localised deluge over the urban core is least visible in the mean.
    """
    src = str(BRDWGD_PR).replace("\\", "/")
    daily = con.execute(
        f"""
        SELECT CAST(code_muni AS VARCHAR) AS munic_code,
               max(CASE WHEN name = 'pr_3.2.3_mean' THEN value END) AS mean_max,
               max(CASE WHEN name = 'pr_3.2.3_max' THEN value END) AS cell_max,
               avg(CASE WHEN name = 'pr_3.2.3_stdev' THEN value END) AS stdev_mean,
               avg(CASE WHEN name = 'pr_3.2.3_count' THEN value END) AS cells
        FROM read_parquet('{src}')
        WHERE date BETWEEN DATE '{probe_year}-01-01' AND DATE '{probe_year}-12-31'
          AND name IN ('pr_3.2.3_mean','pr_3.2.3_max','pr_3.2.3_stdev','pr_3.2.3_count')
        GROUP BY code_muni
        """
    ).pl()
    frame = daily.with_columns(
        (pl.col("cell_max") / pl.col("mean_max")).alias("cellmax_over_arealmean")
    ).sort("cellmax_over_arealmean", descending=True)
    valid = frame.filter(pl.col("mean_max") > 0)
    summary = {
        "probe_year": probe_year,
        "municipalities": frame.height,
        "ratio_p50": round(float(valid["cellmax_over_arealmean"].median()), 3),
        "ratio_p90": round(float(valid["cellmax_over_arealmean"].quantile(0.9)), 3),
        "ratio_max": round(float(valid["cellmax_over_arealmean"].max()), 3),
        "n_ratio_above_1_5": int((valid["cellmax_over_arealmean"] > 1.5).sum()),
        "n_ratio_above_2": int((valid["cellmax_over_arealmean"] > 2.0).sum()),
        "n_municipalities_with_1_cell_or_less": int((frame["cells"] <= 1).sum()),
        "n_municipalities_with_zero_cells": int((frame["cells"] <= 0).sum()),
    }
    return frame, summary


def brdwgd_grid_coverage(con) -> tuple[pl.DataFrame, dict]:
    """Municipalities whose polygon catches no BR-DWGD grid cell.

    ``pr_3.2.3_count`` is the area-weighted cell count. Where it is zero the
    published ``mean`` is NULL: the product has no opinion about that
    municipality's rainfall. The panel, however, contains a number -- see
    :func:`panel_climate_profile`.
    """
    src = str(BRDWGD_PR).replace("\\", "/")
    sample_dates = [f"DATE '{y}-01-15'" for y in range(2007, 2024)] + [
        f"DATE '{y}-07-15'" for y in range(2007, 2024)
    ]
    per_date = con.execute(
        f"""
        SELECT date,
               count(*) FILTER (WHERE name = 'pr_3.2.3_count' AND value <= 0) AS zero_cell_units,
               count(*) FILTER (WHERE name = 'pr_3.2.3_count' AND value < 1) AS sub_cell_units,
               count(*) FILTER (WHERE name = 'pr_3.2.3_mean' AND value IS NULL) AS null_mean_units,
               count(*) FILTER (WHERE name = 'pr_3.2.3_mean') AS units
        FROM read_parquet('{src}')
        WHERE date IN ({', '.join(sample_dates)})
        GROUP BY date ORDER BY date
        """
    ).pl()
    offenders = con.execute(
        f"""
        SELECT CAST(code_muni AS VARCHAR) AS munic_code,
               avg(CASE WHEN name = 'pr_3.2.3_count' THEN value END) AS mean_cells,
               count(*) FILTER (WHERE name = 'pr_3.2.3_mean' AND value IS NULL) AS null_mean_days,
               count(*) FILTER (WHERE name = 'pr_3.2.3_mean') AS days
        FROM read_parquet('{src}')
        WHERE date IN ({', '.join(sample_dates)})
          AND name IN ('pr_3.2.3_count','pr_3.2.3_mean')
        GROUP BY code_muni
        HAVING count(*) FILTER (WHERE name = 'pr_3.2.3_mean' AND value IS NULL) > 0
        ORDER BY null_mean_days DESC
        """
    ).pl()
    summary = {
        "sampled_dates": len(sample_dates),
        "sample_definition": "15 January and 15 July of every year 2007-2023",
        "units_per_date": int(per_date["units"].max()),
        "max_null_mean_units_on_a_sampled_date": int(per_date["null_mean_units"].max()),
        "municipalities_with_any_null_mean": offenders.height,
        "municipalities_with_sub_cell_coverage_max": int(per_date["sub_cell_units"].max()),
    }
    return offenders, summary


# --------------------------------------------------------------------------
# 3. ERA5-Land temperature: cached, complete, unread
# --------------------------------------------------------------------------


def era5_temperature(con) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Profile and monthly-aggregate the cached ERA5-Land tmax/tmin, 2023-2025.

    These six files are already on disk. This function is the whole cost of
    putting temperature into the panel for the last three years: one DuckDB
    aggregation over 220 MB, no download. It is included in the audit so the
    claim "temperature is cheap for 2023-2025" is demonstrated rather than
    asserted.
    """
    blocks: list[pl.DataFrame] = []
    profile_rows: list[dict[str, object]] = []
    for record, year in (("10947952", 2023), ("15748125", 2024), ("18257037", 2025)):
        for kind, filename in (
            ("tmax", "2m_temperature_max.parquet"),
            ("tmin", "2m_temperature_min.parquet"),
        ):
            path = CACHE / "climate" / "era5land" / record / filename
            if not path.exists():
                continue
            src = str(path).replace("\\", "/")
            stat = f"{filename[:-8]}_mean"  # areal mean of the daily max (or min)
            daily = con.execute(
                f"""
                SELECT CAST(code_muni AS VARCHAR) AS munic_code, date,
                       value - 273.15 AS celsius
                FROM read_parquet('{src}') WHERE name = '{stat}'
                """
            ).pl()
            days = daily["date"].n_unique()
            units = daily["munic_code"].n_unique()
            calendar_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
            profile_rows.append(
                {
                    "record": record,
                    "year": year,
                    "variable": kind,
                    "zonal_statistic": stat,
                    "rows": daily.height,
                    "municipalities": units,
                    "observed_days": days,
                    "calendar_days": calendar_days,
                    "day_coverage": round(days / calendar_days, 4),
                    "cells_expected": units * days,
                    "cells_present": daily.height,
                    "completeness": round(daily.height / (units * days), 6),
                    "n_null": int(daily["celsius"].is_null().sum()),
                    "celsius_min": round(float(daily["celsius"].min()), 2),
                    "celsius_median": round(float(daily["celsius"].median()), 2),
                    "celsius_max": round(float(daily["celsius"].max()), 2),
                }
            )
            monthly = (
                daily.with_columns(pl.col("date").dt.truncate("1mo").alias("period"))
                .group_by("munic_code", "period")
                .agg(
                    pl.len().alias(f"n_days_{kind}"),
                    pl.col("celsius").mean().alias(f"{kind}_mean"),
                    pl.col("celsius").max().alias(f"{kind}_max"),
                    pl.col("celsius").min().alias(f"{kind}_min"),
                )
            )
            blocks.append(monthly)

    tmax = pl.concat([b for b in blocks if "tmax_mean" in b.columns], how="vertical")
    tmin = pl.concat([b for b in blocks if "tmin_mean" in b.columns], how="vertical")
    monthly = (
        tmax.join(tmin, on=["munic_code", "period"], how="full", coalesce=True)
        .with_columns((pl.col("tmax_mean") - pl.col("tmin_mean")).alias("dtr_mean"))
        .sort(["munic_code", "period"])
    )
    summary = {
        "files_cached": len(profile_rows),
        "bytes_cached": sum(
            (CACHE / "climate" / "era5land" / r / f).stat().st_size
            for r in ("10947952", "15748125", "18257037")
            for f in ("2m_temperature_max.parquet", "2m_temperature_min.parquet")
            if (CACHE / "climate" / "era5land" / r / f).exists()
        ),
        "monthly_rows_built": monthly.height,
        "municipalities": monthly["munic_code"].n_unique(),
        "periods": monthly["period"].n_unique(),
        "tmax_mean_range_c": [
            round(float(monthly["tmax_mean"].min()), 2),
            round(float(monthly["tmax_mean"].max()), 2),
        ],
        "tmin_mean_range_c": [
            round(float(monthly["tmin_mean"].min()), 2),
            round(float(monthly["tmin_mean"].max()), 2),
        ],
        "dtr_mean_range_c": [
            round(float(monthly["dtr_mean"].min()), 2),
            round(float(monthly["dtr_mean"].max()), 2),
        ],
    }
    return pl.DataFrame(profile_rows), monthly, summary


# --------------------------------------------------------------------------
# 4. The built panel's climate columns
# --------------------------------------------------------------------------


def panel_climate_profile() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Every climate column in the RQ1 panel, on two named populations."""
    keep = ["munic_code", "name", "uf_abbr", "period", "cases", "hospitalised", "deaths"]
    columns = keep + [c for c in PANEL_CLIMATE_COLUMNS]
    panel = pl.read_parquet(PANEL, columns=columns)
    with_cases = panel.filter(pl.col("cases") > 0)

    def profile(frame: pl.DataFrame, population: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        n = frame.height
        for column, (definition, use) in PANEL_CLIMATE_COLUMNS.items():
            series = frame[column]
            numeric = series.dtype.is_numeric()
            n_null = int(series.is_null().sum())
            n_zero = int((series == 0).sum()) if numeric else None
            rows.append(
                {
                    "population": population,
                    "rows": n,
                    "column": column,
                    "dtype": str(series.dtype),
                    "n_null": n_null,
                    "pct_complete": round(100 * (n - n_null) / n, 4),
                    "n_zero": n_zero,
                    "pct_zero": round(100 * n_zero / n, 3) if n_zero is not None else None,
                    "min": float(series.min()) if numeric and n_null < n else None,
                    "p50": float(series.median()) if numeric and n_null < n else None,
                    "p99": float(series.quantile(0.99)) if numeric and n_null < n else None,
                    "max": float(series.max()) if numeric and n_null < n else None,
                    "n_distinct": int(series.n_unique()),
                    "definition": definition,
                    "downstream_use": use,
                }
            )
        return rows

    profile_frame = pl.DataFrame(
        profile(panel, "panel") + profile(with_cases, "panel_cases")
    )

    # The fabricated-zero problem: where the source has no grid cell the panel
    # carries 0.0 mm, because the polars re-aggregation in monthly_municipal
    # sums an all-null group to zero, while the max path (rx1day) correctly
    # yields null. The two disagree, and the disagreement localises the defect.
    suspect = (
        panel.filter(pl.col("rx1day").is_null())
        .group_by("munic_code", "name", "uf_abbr")
        .agg(
            pl.len().alias("months_rx1day_null"),
            (pl.col("precip_mm") == 0).sum().alias("months_precip_zero"),
            pl.col("precip_mm").max().alias("max_precip_mm"),
            pl.col("cases").sum().alias("confirmed_cases_in_those_months"),
            pl.col("period").min().alias("first"),
            pl.col("period").max().alias("last"),
        )
        .sort("months_rx1day_null", descending=True)
    )
    all_months = panel["period"].n_unique()
    fabricated = panel.filter(pl.col("rx1day").is_null() & (pl.col("precip_mm") == 0))
    summary = {
        "panel_rows": panel.height,
        "municipalities": panel["munic_code"].n_unique(),
        "periods": all_months,
        "panel_cases_rows": with_cases.height,
        "confirmed_cases_total": int(panel["cases"].sum()),
        "municipality_months_with_null_rx1day": int(panel["rx1day"].is_null().sum()),
        "municipality_months_with_zero_precip": int((panel["precip_mm"] == 0).sum()),
        "municipality_months_with_fabricated_zero_precip": fabricated.height,
        "fabricated_share_of_all_zero_months": round(
            fabricated.height / max(int((panel["precip_mm"] == 0).sum()), 1), 5
        ),
        "wet_day_counts_also_zero_in_fabricated_cells": int(
            fabricated.select((pl.col("r1") == 0).sum()).item()
        ),
        "municipalities_affected": suspect.height,
        "cases_in_fabricated_zero_cells": int(fabricated["cases"].sum()),
        "municipalities_zero_precip_every_month": int(
            panel.group_by("munic_code")
            .agg((pl.col("precip_mm") == 0).sum().alias("z"))
            .filter(pl.col("z") == all_months)
            .height
        ),
    }
    return profile_frame, suspect, summary


# --------------------------------------------------------------------------
# 5. ENSO
# --------------------------------------------------------------------------


def enso_profile() -> tuple[pl.DataFrame, dict]:
    """What the two cached NOAA series contain, and which columns are read."""
    from brepi.sources.climate import enso as enso_mod

    oni = enso_mod.oni()
    nino = enso_mod.nino34()
    used = {"oni", "period", "phase"}
    rows: list[dict[str, object]] = []
    for label, frame in (("oni", oni), ("nino34", nino)):
        for column in frame.columns:
            series = frame[column]
            rows.append(
                {
                    "series": label,
                    "column": column,
                    "dtype": str(series.dtype),
                    "n": frame.height,
                    "n_null": int(series.is_null().sum()),
                    "period_min": frame["period"].min(),
                    "period_max": frame["period"].max(),
                    "granularity": "national x month (3-month running mean)"
                    if label == "oni"
                    else "national x month (unsmoothed)",
                    "status": "used" if (label == "oni" and column in used) else "available_unused",
                }
            )
    summary = {
        "oni_rows": oni.height,
        "oni_span": [str(oni["period"].min()), str(oni["period"].max())],
        "nino34_rows": nino.height,
        "nino34_span": [str(nino["period"].min()), str(nino["period"].max())],
        "nino34_columns": nino.columns,
        "used_in_analysis": "ONI value and phase only (27_rq1_enso.R)",
    }
    return pl.DataFrame(rows), summary


# --------------------------------------------------------------------------
# 6. The Atlas: 70 columns, 8 survive
# --------------------------------------------------------------------------


def load_atlas() -> pl.DataFrame:
    """The cached consolidated export, every column Utf8, plus derived keys.

    Read straight from the cache path rather than through ``atlas.raw()``,
    which would try to re-resolve the download URL over the network.
    """
    frame = pl.read_csv(
        ATLAS_CSV,
        separator=";",
        encoding="latin-1",
        infer_schema_length=0,
        truncate_ragged_lines=True,
        quote_char='"',
    )
    return frame.with_columns(
        atlas_mod.cobrade_expr("Cod_Cobrade").alias("cobrade"),
        pl.col("Data_Evento").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("event_date"),
        pl.col("Data_Registro").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("register_date"),
    ).with_columns(
        pl.col("cobrade").replace_strict(atlas_mod.COBRADE_LABELS, default=None).alias("cobrade_label"),
        (pl.col("register_date") - pl.col("event_date")).dt.total_days().alias("register_lag_days"),
    )


def _prefix_mask(prefixes) -> pl.Expr:
    expr = pl.lit(False)
    for prefix in prefixes:
        expr = expr | pl.col("cobrade").str.starts_with(prefix)
    return expr


def atlas_populations(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    window = pl.col("event_date").is_between(STUDY_START, STUDY_END)
    return {
        "atlas_all": frame,
        "atlas_flood": frame.filter(_prefix_mask(atlas_mod.COBRADE_FLOOD) & window),
        "atlas_drought": frame.filter(_prefix_mask(atlas_mod.COBRADE_DROUGHT) & window),
    }


def _as_number(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8).str.strip_chars()
    has_comma = text.str.contains(",")
    ptbr = text.str.replace_all(r"\.", "").str.replace(",", ".")
    return pl.when(has_comma).then(ptbr).otherwise(text).cast(pl.Float64, strict=False)


def _f(value) -> float | None:
    """``float`` that survives an all-null column rather than raising."""
    return None if value is None else float(value)


def atlas_column_profile(populations: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Completeness of all 70 Atlas columns on each named population.

    ``pct_populated`` is the share of rows where the field is neither null nor
    a blank sentinel. For numeric fields ``pct_nonzero`` is the share strictly
    above zero, which is the number that matters: the FIDE form is filled with
    zeros for damage categories that did not occur, so a column can be 100%
    "present" and 99% uninformative.
    """
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        n = frame.height
        for column, (gloss, category) in ATLAS_COLUMNS.items():
            if column not in frame.columns:
                continue
            text = frame[column].cast(pl.Utf8).str.strip_chars()
            populated = int((text.is_not_null() & ~text.is_in(list(TEXT_BLANKS))).sum())
            numeric = frame.select(_as_number(column).alias("v"))["v"]
            n_parsed = int(numeric.is_not_null().sum())
            is_numeric_field = n_parsed >= 0.5 * max(populated, 1) and category not in {
                "text",
                "identity",
                "geography",
                "classification",
                "recognition",
                "timing",
            }
            nonzero = int((numeric > 0).sum())
            rows.append(
                {
                    "population": population,
                    "rows": n,
                    "column": column,
                    "category": category,
                    "gloss": gloss,
                    "n_populated": populated,
                    "pct_populated": round(100 * populated / n, 3) if n else None,
                    "n_parsed_numeric": n_parsed,
                    "n_nonzero": nonzero if is_numeric_field else None,
                    "pct_nonzero": round(100 * nonzero / n, 3) if is_numeric_field and n else None,
                    "sum": float(numeric.sum()) if is_numeric_field and n_parsed else None,
                    "median_when_nonzero": float(
                        numeric.filter(numeric > 0).median()
                    ) if is_numeric_field and nonzero else None,
                    "max": float(numeric.max()) if is_numeric_field and n_parsed else None,
                    "n_distinct": int(text.n_unique()),
                    "value_type": "ordinal_text" if column in ATLAS_ORDINAL else (
                        "numeric" if is_numeric_field else "text/key"
                    ),
                    "carried_by_adapter": column in ADAPTER_CARRIES,
                    "fate": ATLAS_FATE.get(column, "never decoded by any adapter"),
                }
            )
    return pl.DataFrame(rows)


def atlas_environmental_levels(populations: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The DA_ ordinal bands, which a numeric parse scores as all-zero.

    ``DA_Polui/cont da agua`` is the one that matters: it records the share of
    the municipal population affected by water contamination, banded. That is a
    graded exposure statement about *contaminated water*, which is the
    transmission route, and it exists in no other national source.
    """
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for column in ATLAS_ORDINAL:
            if column not in frame.columns:
                continue
            counts = (
                frame.select(pl.col(column).cast(pl.Utf8).str.strip_chars().alias("level"))
                .filter(pl.col("level").is_not_null() & ~pl.col("level").is_in(list(TEXT_BLANKS)))
                .group_by("level")
                .len()
                .sort("len", descending=True)
            )
            for row in counts.iter_rows(named=True):
                rows.append(
                    {
                        "population": population,
                        "rows": frame.height,
                        "column": column,
                        "gloss": ATLAS_COLUMNS[column][0],
                        "level": row["level"],
                        "n": row["len"],
                        "pct_of_population": round(100 * row["len"] / frame.height, 3),
                    }
                )
    return pl.DataFrame(rows)


def atlas_census_tracts(populations: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict]:
    """The sub-municipal footprint: comma-separated 15-digit census tracts.

    This is the only sub-municipal geography anywhere in the disaster surface.
    It is a *list* per protocol, not a single code, so it also measures how much
    of the municipality was declared affected.
    """
    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for population, frame in populations.items():
        column = "Setores Censitários"
        if column not in frame.columns:
            continue
        parsed = frame.select(
            pl.col(column)
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("raw")
        ).with_columns(
            pl.when(pl.col("raw").is_null() | pl.col("raw").is_in(list(TEXT_BLANKS)))
            .then(None)
            .otherwise(pl.col("raw").str.split(","))
            .alias("tracts")
        ).with_columns(pl.col("tracts").list.len().alias("n_tracts"))
        populated = parsed.filter(pl.col("n_tracts").is_not_null() & (pl.col("n_tracts") > 0))
        exploded = populated.select(pl.col("tracts").explode().str.strip_chars().alias("tract"))
        rows.append(
            {
                "population": population,
                "rows": frame.height,
                "n_protocols_with_tracts": populated.height,
                "pct_with_tracts": round(100 * populated.height / frame.height, 3),
                "tract_mentions": exploded.height,
                "distinct_tracts": int(exploded["tract"].n_unique()),
                "tracts_per_protocol_p50": float(populated["n_tracts"].median()) if populated.height else None,
                "tracts_per_protocol_p90": float(populated["n_tracts"].quantile(0.9)) if populated.height else None,
                "tracts_per_protocol_max": int(populated["n_tracts"].max()) if populated.height else None,
                "single_tract_protocols": int((populated["n_tracts"] == 1).sum()) if populated.height else 0,
            }
        )
        if population == "atlas_flood":
            detail = rows[-1]
    return pl.DataFrame(rows), detail


def atlas_cobrade_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Every COBRADE code present in the export, with counts and severity sums."""
    window = pl.col("event_date").is_between(STUDY_START, STUDY_END)
    return (
        frame.with_columns(
            _as_number("DH_MORTOS").alias("_deaths"),
            _as_number("DH_DESABRIGADOS").alias("_homeless"),
            _as_number("DH_DESALOJADOS").alias("_displaced"),
            _as_number("PE_PLePR").alias("_loss"),
        )
        .group_by("cobrade", "cobrade_label", "grupo_de_desastre", "tipologia")
        .agg(
            pl.len().alias("n_protocols"),
            pl.col("Cod_IBGE_Mun").n_unique().alias("n_municipalities"),
            (pl.col("Status") == "Reconhecido").sum().alias("n_recognised"),
            window.sum().alias("n_protocols_2007_2025"),
            pl.col("event_date").min().alias("first_event"),
            pl.col("event_date").max().alias("last_event"),
            pl.col("_deaths").sum().alias("deaths"),
            pl.col("_homeless").sum().alias("homeless"),
            pl.col("_displaced").sum().alias("displaced"),
            pl.col("_loss").sum().alias("economic_loss_brl"),
        )
        .with_columns(
            pl.col("cobrade").is_not_null().alias("cobrade_parsed"),
            pl.col("cobrade_label").is_not_null().alias("labelled_by_adapter"),
            _prefix_mask(atlas_mod.COBRADE_FLOOD).alias("in_flood_default"),
            _prefix_mask(atlas_mod.COBRADE_FLOOD_STRICT).alias("in_flood_strict"),
            _prefix_mask(atlas_mod.COBRADE_DROUGHT).alias("in_drought"),
            _prefix_mask(atlas_mod.COBRADE_MASS_MOVEMENT).alias("in_mass_movement"),
        )
        .sort("n_protocols", descending=True)
    )


def atlas_timing(frame: pl.DataFrame, populations: dict[str, pl.DataFrame]) -> dict:
    lag = frame["register_lag_days"].drop_nulls()
    out = {
        "rows": frame.height,
        "status_counts": {
            r["Status"]: r["count"] for r in frame["Status"].value_counts().iter_rows(named=True)
        },
        "event_date_unparsed": int(frame["event_date"].is_null().sum()),
        "register_date_unparsed": int(frame["register_date"].is_null().sum()),
        "event_date_span": [str(frame["event_date"].min()), str(frame["event_date"].max())],
        "register_lag_days": {
            "n": lag.len(),
            "negative": int((lag < 0).sum()),
            "zero": int((lag == 0).sum()),
            "p50": float(lag.median()),
            "p90": float(lag.quantile(0.9)),
            "max": float(lag.max()),
        },
        "population_sizes": {k: v.height for k, v in populations.items()},
        "flood_municipalities": populations["atlas_flood"]["Cod_IBGE_Mun"].n_unique(),
        "drought_municipalities": populations["atlas_drought"]["Cod_IBGE_Mun"].n_unique(),
    }
    return out


def atlas_damage_dose(populations: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict]:
    """The dose measures, ranked by how often a flood protocol actually carries one.

    This is the table the study needs and does not have: a flood declaration is
    currently a 0/1 treatment, and these columns say *how big*.
    """
    flood = populations["atlas_flood"]
    dose_categories = {
        "human_damage",
        "material_damage",
        "material_damage_value",
        "public_loss",
        "private_loss",
        "total_loss",
    }
    rows: list[dict[str, object]] = []
    for column, (gloss, category) in ATLAS_COLUMNS.items():
        if category not in dose_categories or column not in flood.columns:
            continue
        values = flood.select(_as_number(column).alias("v"))["v"]
        nonzero = values.filter(values > 0)
        counts_field = category == "material_damage" or (
            category == "human_damage" and "total" not in column.lower()
        )
        rows.append(
            {
                "column": column,
                "category": category,
                "gloss": gloss,
                "n_flood_protocols": flood.height,
                "n_nonzero": nonzero.len(),
                "pct_nonzero": round(100 * nonzero.len() / flood.height, 3),
                "sum": _f(values.sum()) if values.len() else None,
                "p50_when_nonzero": _f(nonzero.median()) if nonzero.len() else None,
                "p90_when_nonzero": _f(nonzero.quantile(0.9)) if nonzero.len() else None,
                "max": _f(values.max()) if values.len() else None,
                # Tail description, not a judgement -- except that a count of
                # institutions above 500 in one municipality-protocol, and any
                # count above 100,000, cannot be a count. Those rows are
                # misfiled monetary values sitting in a count column.
                "n_above_500": int((values > IMPLAUSIBLE_FACILITY_COUNT).sum())
                if counts_field
                else None,
                "n_above_100k": int((values > 100_000).sum()) if counts_field else None,
                "carried_by_adapter": column in ADAPTER_CARRIES,
                "fate": ATLAS_FATE.get(column, "never decoded by any adapter"),
            }
        )
    frame = pl.DataFrame(rows).sort("pct_nonzero", descending=True)
    summary = {
        "flood_protocols_2007_2025": flood.height,
        "dose_columns_profiled": frame.height,
        "dose_columns_carried_by_adapter": int(frame["carried_by_adapter"].sum()),
        "dose_columns_never_decoded": int(
            frame.filter(pl.col("fate") == "never decoded by any adapter").height
        ),
        "dose_columns_with_any_signal": int((frame["n_nonzero"] > 0).sum()),
        "dose_columns_above_10pct_nonzero": int((frame["pct_nonzero"] >= 10).sum()),
        "note_ordinal_columns_excluded": list(ATLAS_ORDINAL),
    }
    return frame, summary


def atlas_health_relevant(populations: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The subset of loss columns with a direct leptospirosis mechanism.

    Sanitation infrastructure, refuse collection, pest control and medical
    assistance are the four public services whose failure is the textbook
    pathway from a flood to a leptospirosis outbreak, and the Atlas prices each
    of them per protocol.
    """
    wanted = [
        "PEPL_Abast de água pot(R$)",
        "PEPL_sist de esgotos sanit(R$)",
        "PEPL_Sis limp e rec lixo (R$)",
        "PEPL_Sis cont pragas (R$)",
        "PEPL_Assis_méd e emergên(R$)",
        "DM_Inst Saúde Danificadas",
        "DM_Inst Saúde Destruidas",
        "DA_Polui/cont da água",
        "DH_DESABRIGADOS",
        "DH_DESALOJADOS",
        "Setores Censitários",
    ]
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for column in wanted:
            if column not in frame.columns:
                continue
            values = frame.select(_as_number(column).alias("v"))["v"]
            nonzero = values.filter(values > 0)
            text = frame[column].cast(pl.Utf8).str.strip_chars()
            rows.append(
                {
                    "population": population,
                    "rows": frame.height,
                    "column": column,
                    "gloss": ATLAS_COLUMNS[column][0],
                    "value_type": "ordinal_text" if column in ATLAS_ORDINAL else (
                        "tract_list" if column == "Setores Censitários" else "numeric"
                    ),
                    "n_populated": int((text.is_not_null() & ~text.is_in(list(TEXT_BLANKS))).sum()),
                    "n_nonzero": nonzero.len(),
                    "pct_nonzero": round(100 * nonzero.len() / frame.height, 3),
                    "sum": _f(values.sum()) if values.len() else None,
                    "max": _f(values.max()) if values.len() else None,
                    "n_municipalities_nonzero": int(
                        frame.with_columns(_as_number(column).alias("v"))
                        .filter(pl.col("v") > 0)["Cod_IBGE_Mun"]
                        .n_unique()
                    ),
                }
            )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# 7. What the built event tables kept
# --------------------------------------------------------------------------


#: Who actually reads each built table, established by grepping the study's R
#: and Python stages. The severity columns that do survive into a table are
#: still not read by anything: no RQ2 script mentions ``deaths`` or
#: ``affected_total``.
EVENT_TABLE_CONSUMERS = {
    "flood_events.parquet": "15_rq2_did.R, 15a/15b/15c/15d; only munic_code, date, cobrade, recognised are read",
    "drought_events.parquet": "15_rq2_did.R placebo arm; same four columns",
    "flood_declarations.parquet": "declared as a _targets.R target; NO analysis script reads it -- "
    "it is the only built table carrying homeless/displaced/damage_brl at municipality-month",
    "flood_first_treatment.parquet": "31_build_atlas.py, for the municipality atlas's first-treatment year",
}


def event_tables() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for path in (
        paths.FLOOD_EVENTS,
        paths.DROUGHT_EVENTS,
        paths.FLOOD_DECLARATIONS,
        paths.FLOOD_FIRST_TREATMENT,
    ):
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        numeric = [c for c in frame.columns if frame[c].dtype.is_numeric()]
        detail = {
            c: {
                "n_nonzero": int((frame[c] > 0).sum()),
                "sum": float(frame[c].sum()),
                "max": float(frame[c].max()),
            }
            for c in numeric
            if c not in ("time_index", "year")
        }
        rows.append(
            {
                "table": path.name,
                "rows": frame.height,
                "columns": frame.width,
                "column_names": ";".join(frame.columns),
                "municipalities": frame["munic_code"].n_unique() if "munic_code" in frame.columns else None,
                "date_min": str(frame["date"].min()) if "date" in frame.columns else (
                    str(frame["period"].min()) if "period" in frame.columns else None
                ),
                "date_max": str(frame["date"].max()) if "date" in frame.columns else (
                    str(frame["period"].max()) if "period" in frame.columns else None
                ),
                "numeric_detail": json.dumps(detail, ensure_ascii=False),
                "consumed_by": EVENT_TABLE_CONSUMERS.get(path.name, "unknown"),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. The dark inventory
# --------------------------------------------------------------------------


def dark_inventory(era5_bytes: int) -> pl.DataFrame:
    gb = lambda b: f"{b / 1e9:.2f} GB"  # noqa: E731 - a formatting alias, deliberately local
    rec = ZENODO_RECORDED
    items = [
        {
            "item": "BR-DWGD Tmax + Tmin (Tmax_3.2.3.parquet, Tmin_3.2.3.parquet)",
            "question_it_answers": (
                "Does the rainfall-leptospirosis association depend on temperature? "
                "Leptospira survival in standing water is temperature-bounded, so the "
                "same 100 mm month should not carry the same risk in a 32 degC "
                "Northeastern municipality and a 16 degC Southern one. Without it the "
                "national DLNM pools opposite survival regimes."
            ),
            "granularity": "municipality x day, 1961-01-01..2024-03-20 (5,567 municipalities)",
            "effort": "expensive",
            "caveat": f"{gb(rec['13906834']['Tmax_3.2.3.parquet'] + rec['13906834']['Tmin_3.2.3.parquet'])} to download; "
            "then the same monthly_municipal() path already used for precipitation.",
        },
        {
            "item": "ERA5-Land 2m_temperature_max / _min, 2023-2025 (already cached)",
            "question_it_answers": (
                "Same question for the tail of the panel, and it is free: the six files "
                "are on disk. Also gives the diurnal range, which separates 'hot and dry' "
                "from 'warm and humid' months that a mean temperature cannot."
            ),
            "granularity": "municipality x day, 2023-01-01..2025-12-31 (5,565-5,567 municipalities)",
            "effort": "already_extracted",
            "caveat": f"{gb(era5_bytes)} cached; reanalysis, not observation, so it is not "
            "comparable to a BR-DWGD temperature series without the same bias adjustment "
            "the precipitation splice needs.",
        },
        {
            "item": "BR-DWGD within-municipality statistics: pr_3.2.3_max, _stdev, _count",
            "question_it_answers": (
                "Was the rain concentrated on part of the municipality? The study's "
                "exposure is the areal mean, which averages a localised deluge over the "
                "whole polygon. The cell maximum is the rainfall that actually fell on "
                "the wettest place, and the ratio of the two is a measure of how badly "
                "the municipal mean mis-measures exposure -- largest in the biggest "
                "municipalities, which is where most Amazonian cases are."
            ),
            "granularity": "municipality x day, 1961-2024, in the artefact already cached",
            "effort": "already_extracted",
            "caveat": "One extra name in the existing DuckDB predicate; zero download. "
            "brepi.sources.climate.brdwgd declares the statistic 'sd', which does not "
            "exist in the artefact (it is 'stdev'), and requesting it returns an empty "
            "frame rather than an error.",
        },
        {
            "item": "BR-DWGD RH, Rs, ETo (relative humidity, solar radiation, reference evapotranspiration)",
            "question_it_answers": (
                "How long did standing water persist after the rain? P - ETo is a water "
                "balance and is a better proxy for the duration of the exposure window "
                "than rainfall alone; RH and Rs govern desiccation and UV inactivation "
                "of the spirochaete between rain events."
            ),
            "granularity": "municipality x day, 1961-01-01..2024-03-20",
            "effort": "expensive",
            "caveat": f"{gb(rec['13906834']['RH_3.2.3.parquet'] + rec['13906834']['Rs_3.2.3.parquet'] + rec['13906834']['ETo_3.2.3.parquet'])} "
            "for the three; ETo alone is the highest-value single file.",
        },
        {
            "item": "Panel indices r1, r10, r20, r50, rx1day (built, in the panel, unmodelled)",
            "question_it_answers": (
                "Is it the total or the intensity that matters? A 200 mm month falling "
                "as one 200 mm day and as twenty 10 mm days are opposite exposures for "
                "leptospirosis, and only the second set of columns distinguishes them. "
                "r50 is the operational heavy-rain alert threshold and is the natural "
                "'extreme event' exposure for a DLNM."
            ),
            "granularity": "municipality x month, 2007-01..2025-12, 100% complete except 455 cells",
            "effort": "already_extracted",
            "caveat": "In the panel now. rx1day is null in 455 municipality-months; "
            "r1..r50 are 0 in those cells, which is a fabricated zero, not an observation.",
        },
        {
            "item": "Atlas human-damage counts: DH_DESABRIGADOS, DH_DESALOJADOS, DH_FERIDOS, DH_ENFERMOS, DH_DESAPARECIDOS",
            "question_it_answers": (
                "How big was the flood? Treatment is currently binary. Homeless and "
                "displaced counts turn a declaration into a dose and let the DiD ask "
                "whether the leptospirosis response scales with displacement -- which is "
                "the actual mechanism (people in shelters, wading through contaminated "
                "water) rather than the bureaucratic act of declaring."
            ),
            "granularity": "S2iD protocol (municipality x disaster), event-dated; aggregable to municipality x month",
            "effort": "cheap_extract",
            "caveat": "Counts are self-reported on the FIDE form and zero-filled where "
            "the category did not occur; a zero is not distinguishable from 'not "
            "assessed'. Use ranks or an indicator of any displacement, not levels.",
        },
        {
            "item": "Atlas public-service loss columns: water supply, sewerage, refuse collection, PEST CONTROL, medical assistance (BRL)",
            "question_it_answers": (
                "Did the flood break the specific services whose failure causes "
                "leptospirosis? These five columns price, per protocol, the damage to "
                "potable water, sanitary sewerage, refuse collection, rodent control and "
                "emergency medical care. That is a mechanism-specific dose no other "
                "national source carries, and 'pest control damage' is as close to a "
                "measured rodent-exposure shock as Brazilian administrative data gets. "
                "Medical-assistance loss additionally indexes a surveillance shock, "
                "which speaks directly to the hospitalisation-share thesis."
            ),
            "granularity": "S2iD protocol, event-dated, municipality-resolved",
            "effort": "cheap_extract",
            "caveat": "Sparse (see atlas_health_relevant.csv for exact non-zero shares) "
            "and monetary values are deflated to Dec-2022 prices only through 2022; "
            "post-2022 rows are nominal, so levels are not comparable across the splice.",
        },
        {
            "item": "Atlas material-damage counts: housing units damaged/destroyed, health facilities damaged/destroyed",
            "question_it_answers": (
                "Housing destroyed is the best available proxy for the size of the "
                "displaced population when DH_ fields are blank; health facilities "
                "damaged is a direct shock to the reporting apparatus and therefore to "
                "the hospitalisation share the study uses as its depth index -- a flood "
                "that damages hospitals mechanically lowers recorded severity."
            ),
            "granularity": "S2iD protocol, event-dated, municipality-resolved",
            "effort": "cheap_extract",
            "caveat": "Same zero-versus-unassessed ambiguity as the human-damage counts.",
        },
        {
            "item": "Atlas 'Setores Censitarios': a comma-separated list of 15-digit census tracts per protocol",
            "question_it_answers": (
                "Which part of the municipality flooded, and how much of it? This is "
                "the only sub-municipal geography anywhere in the disaster surface. It "
                "would let a flood be matched to the tracts' own sanitation, favela and "
                "density indicators instead of to the municipal average -- which is the "
                "single largest source of exposure misclassification in a "
                "municipality-level flood design -- and the tract count per protocol is "
                "itself a footprint size."
            ),
            "granularity": "census tract list per S2iD protocol; tracts are the 2010 census mesh",
            "effort": "cheap_extract",
            "caveat": "See atlas_census_tracts.csv for the exact populated share and the "
            "tracts-per-protocol distribution. Tract codes are on the 2010 mesh and do "
            "not join to 2022 tracts without a crosswalk; protocols filed before the "
            "field existed are blank, so the covered subset is not random.",
        },
        {
            "item": "Atlas DA_ environmental-damage bands, above all 'Polui/cont da agua'",
            "question_it_answers": (
                "How much of the population had contaminated water? The field is an "
                "ordinal band (0-5%, 5-10%, 10-20%, >20% of the population affected), "
                "which is a graded statement about the transmission route itself rather "
                "than about the hazard. No other national source carries it."
            ),
            "granularity": "S2iD protocol, ordinal band, event-dated, municipality-resolved",
            "effort": "cheap_extract",
            "caveat": "Sparse, and it is TEXT: any numeric parse scores it as all-zero, "
            "which is why it looks empty in a damage-column loop. Level counts are in "
            "atlas_environmental_damage_levels.csv. Band wording is not fully "
            "standardised ('MAIS DE 20%' and 'MAIS DE 20% DA POPULACAO AFETADA' both "
            "occur) and needs harmonising before use.",
        },
        {
            "item": "Atlas COBRADE codes outside the flood/drought sets (mass movement 1.1.3.x, storms 1.3.2.1.x, biological 1.5.x)",
            "question_it_answers": (
                "Are the placebo and the treatment right? Mass movement co-occurs with "
                "the same rainfall but produces mud, not standing water: a sharper "
                "negative control than drought. COBRADE 1.5.1.x is an epidemic "
                "declaration -- a municipality declaring a disease emergency is a "
                "surveillance-intensity event in its own right."
            ),
            "granularity": "S2iD protocol, event-dated, municipality-resolved, 1991-2025",
            "effort": "cheap_extract",
            "caveat": "The full code frequency table is in atlas_cobrade_counts.csv; "
            "several codes present in the export are not in the adapter's label "
            "dictionary and would come through unlabelled.",
        },
        {
            "item": "Atlas Status = Registro versus Reconhecido, and the registration lag",
            "question_it_answers": (
                "Is the treatment measuring water or bureaucracy? The recognised subset "
                "selects on municipal administrative capacity, which plausibly "
                "correlates with health-reporting capacity -- the study's own outcome "
                "mechanism. The registration lag is a direct, continuous index of that "
                "capacity and could be used as a covariate rather than only as a "
                "sensitivity split."
            ),
            "granularity": "S2iD protocol; lag in days",
            "effort": "already_extracted",
            "caveat": "Status is undefined in every official Atlas document; the reading "
            "is inferred. The lag is negative for a minority of rows.",
        },
        {
            "item": "Monthly Nino 3.4 (unsmoothed) and the Nino 1+2 / 3 / 4 anomalies",
            "question_it_answers": (
                "At what lag does ENSO act? The ONI the study uses is a 3-month running "
                "mean, which pre-smooths the exposure and biases any estimated lag "
                "toward zero. Nino 1+2 is the eastern-Pacific index that drives "
                "Northeast Brazilian drought specifically, so the four regions' opposite "
                "responses can be separated."
            ),
            "granularity": "national x month, 1950-2026; cached ASCII",
            "effort": "already_extracted",
            "caveat": "National series: identified only through interactions with "
            "municipality or region, never as a main effect alongside a temporal field.",
        },
    ]
    return pl.DataFrame(items)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip the Zenodo record listing")
    parser.add_argument("--probe-year", type=int, default=2011)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"generated_at": _now(), "probe_year": args.probe_year}
    con = _connect()

    print("[1/9] published climate inventory")
    listing, provenance = zenodo_listing(args.offline)
    published = published_inventory(listing, provenance)
    _write(published, "climate_products_published")
    report["published"] = {
        "provenance": provenance,
        "files_published": published.height,
        "files_cached": int(published["cached"].sum()),
        "published_gb": round(float(published["published_bytes"].sum()) / 1e9, 2),
        "cached_gb": round(float(published.filter(pl.col("cached"))["published_bytes"].sum()) / 1e9, 2),
        "uncached_gb": round(float(published.filter(~pl.col("cached"))["published_bytes"].sum()) / 1e9, 2),
        "bytes_match_where_cached": bool(
            published.filter(pl.col("cached"))["bytes_match"].all()
        ),
        "brdwgd_variables_published": int(published.filter(pl.col("product") == "brdwgd").height),
        "brdwgd_variables_cached": int(
            published.filter((pl.col("product") == "brdwgd") & pl.col("cached")).height
        ),
    }

    print("[2/9] cached artefact profile")
    cached = cache_profile()
    _write(cached, "climate_cache_profile")
    report["cache"] = {
        "artefacts": cached.height,
        "total_gb": round(float(cached["bytes"].sum()) / 1e9, 2),
        "total_rows": int(cached["rows"].sum()),
    }

    print("[3/9] BR-DWGD statistics probe")
    stats = brdwgd_statistic_probe(con, args.probe_year)
    _write(stats, "brdwgd_statistics_probe")
    report["brdwgd_statistics"] = {
        "present": stats.filter(pl.col("present_in_artefact"))["name"].to_list(),
        "declared_but_absent": stats.filter(
            pl.col("declared_by_adapter") & ~pl.col("present_in_artefact")
        )["name"].to_list(),
        "present_but_undeclared": stats.filter(
            pl.col("present_in_artefact") & ~pl.col("declared_by_adapter")
        )["name"].to_list(),
        "read_by_the_study": ["pr_3.2.3_mean"],
    }

    print("[4/9] within-municipality rainfall heterogeneity")
    hetero, hetero_summary = brdwgd_within_municipality(con, args.probe_year)
    _write(hetero.head(200), "brdwgd_within_municipality_top200")
    hetero.write_parquet(OUT / "brdwgd_within_municipality.parquet")
    report["within_municipality"] = hetero_summary

    print("[5/9] BR-DWGD grid coverage gaps")
    gaps, gap_summary = brdwgd_grid_coverage(con)
    _write(gaps, "brdwgd_zero_cell_municipalities")
    report["grid_coverage"] = gap_summary

    print("[6/9] ERA5-Land temperature (cached, unused)")
    era5_profile, era5_monthly, era5_summary = era5_temperature(con)
    _write(era5_profile, "era5land_temperature_profile")
    era5_monthly.write_parquet(OUT / "era5land_temperature_monthly_2023_2025.parquet")
    print(f"  wrote era5land_temperature_monthly_2023_2025.parquet ({era5_monthly.height} rows)")
    report["era5land_temperature"] = era5_summary

    print("[7/9] panel climate columns and ENSO")
    panel_cols, suspect, panel_summary = panel_climate_profile()
    _write(panel_cols, "panel_climate_columns")
    _write(suspect, "panel_fabricated_zero_precipitation")
    report["panel"] = panel_summary
    enso_frame, enso_summary = enso_profile()
    _write(enso_frame, "enso_series_columns")
    report["enso"] = enso_summary

    print("[8/9] Atlas / S2iD surface")
    atlas_frame = load_atlas()
    missing_gloss = [c for c in atlas_frame.columns if c not in ATLAS_COLUMNS and not c.startswith(("cobrade", "event_date", "register_"))]
    if missing_gloss:
        print(f"  ! columns without a gloss: {missing_gloss}")
    pops = atlas_populations(atlas_frame)
    columns = atlas_column_profile(pops)
    _write(columns, "atlas_columns")
    cobrade = atlas_cobrade_counts(atlas_frame)
    _write(cobrade, "atlas_cobrade_counts")
    dose, dose_summary = atlas_damage_dose(pops)
    _write(dose, "atlas_damage_dose")
    health = atlas_health_relevant(pops)
    _write(health, "atlas_health_relevant")
    levels = atlas_environmental_levels(pops)
    _write(levels, "atlas_environmental_damage_levels")
    tracts, tract_detail = atlas_census_tracts(pops)
    _write(tracts, "atlas_census_tracts")
    report["atlas"] = atlas_timing(atlas_frame, pops)
    report["atlas"]["columns_total"] = len(ATLAS_COLUMNS)
    report["atlas"]["columns_glossed"] = len([c for c in ATLAS_COLUMNS if c in atlas_frame.columns])
    report["atlas"]["columns_carried_by_adapter"] = len(ADAPTER_CARRIES)
    report["atlas"]["columns_in_event_tables"] = len(EVENT_TABLE_COLUMNS)
    report["atlas"]["cobrade_codes_present"] = int(cobrade.filter(pl.col("cobrade_parsed")).height)
    report["atlas"]["cobrade_codes_unlabelled"] = int(
        cobrade.filter(pl.col("cobrade_parsed") & ~pl.col("labelled_by_adapter")).height
    )
    report["atlas"]["columns_never_decoded"] = int(
        columns.filter(
            (pl.col("population") == "atlas_flood")
            & (pl.col("fate") == "never decoded by any adapter")
        ).height
    )
    report["atlas_dose"] = dose_summary
    report["atlas_census_tracts_flood"] = tract_detail
    report["atlas_environmental_bands"] = {
        "columns": list(ATLAS_ORDINAL),
        "flood_protocols_with_a_water_contamination_band": int(
            levels.filter(
                (pl.col("population") == "atlas_flood")
                & (pl.col("column") == "DA_Polui/cont da água")
            )["n"].sum()
        ),
        "distinct_water_contamination_bands": int(
            levels.filter(
                (pl.col("population") == "atlas_flood")
                & (pl.col("column") == "DA_Polui/cont da água")
            ).height
        ),
    }

    print("[9/9] built event tables and the dark inventory")
    tables = event_tables()
    _write(tables, "event_tables_profile")
    era5_bytes = int(era5_summary["bytes_cached"])
    dark = dark_inventory(era5_bytes)
    _write(dark, "dark_inventory")
    report["dark_items"] = dark.height

    (OUT / "audit_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
