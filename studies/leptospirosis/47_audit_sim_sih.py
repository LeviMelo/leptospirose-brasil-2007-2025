"""Data-surface audit of the DATASUS mortality (SIM) and hospital (SIH) systems.

The triangulation arms of this study run on three fields each: A27 death counts
from SIM, A27 admission counts from SIH, and the in-hospital death flag. The
cached raw files carry **88 fields per death certificate** and **113 fields per
AIH**. This script enumerates all of them, on the population that matters, and
says which are used, which are available and dark, and which are unusable.

Four stages, each independently runnable.

``headers``
    Layout census. The dBase field descriptor array sits uncompressed at the
    front of a ``.dbc``, so every cached file's schema can be read without
    decompressing a single row. Produces, per system, one row per field per
    year: presence, declared width, and the years it exists in. This is the
    only way to see the *schema breaks* -- SIH gained ``DIAGSEC1..9`` and
    ``TPDISEC1..9`` partway through the series, and a projection written
    against the old layout silently stops seeing secondary diagnoses.

``deep``
    Re-scan of the raw cache keeping **every field** of the A27-relevant
    records. The current interim extracts are projections onto 14 (SIM) and 20
    (SIH) columns chosen in 2026-07; the discarded fields cannot be profiled
    from them. Because decompression dominates the cost and is paid either
    way, taking all fields instead of twenty is nearly free -- the expensive
    part is the pass over 16.5 GB, which is done once here.

    The scan is also *wider on rows*, deliberately, and the difference is a
    finding rather than a nuisance:

    * SIM is matched on ``CAUSABAS`` (underlying cause, anchored) **and** on
      the multiple-cause chain ``LINHAA..LINHAD``, ``LINHAII``, ``CAUSABAS_O``
      by **substring**, because those fields concatenate four-character codes
      with no separator and an anchored match only ever sees the first one.
    * SIH is matched on ``DIAG_PRINC``, ``DIAG_SECUN`` **and** the nine
      ``DIAGSEC*`` slots plus ``CID_ASSO``/``CID_MORTE``/``CID_NOTIF``.

    Per-rule hit counts are recorded per file, so the marginal contribution of
    each field is quantified rather than asserted.

``profile``
    Column profiles: dtype, completeness, cardinality, modal values, and --
    where the repository codebook binds a concept to the field -- the four
    decode states (valid / unknown / missing / invalid). Run over both the
    frozen interim extracts and the full-field deep scans, with the population
    named in every output.

``severity``
    The question this audit exists to answer. The study's central index is the
    *share of confirmed cases recorded as hospitalised* -- a binary proxy for
    how far down the severity distribution a territory's surveillance reaches.
    SIH carries length of stay, ICU days, ICU flag, ICU billing, admission
    character, procedure and discharge reason. This stage measures whether
    those are complete enough, and internally consistent enough, to replace
    the proxy with a direct severity measurement.

Outputs to ``data/results/audit_sim_sih/``. Nothing here writes to
``data/interim/``: the frozen extracts that the published analysis depends on
are not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.codebook import binding_for, categorical_exprs
from brepi.config import PATHS
from brepi.io.dbc import dbf_header, field_offsets

OUT = PATHS.results / "audit_sim_sih"
SIM_CACHE = PATHS.cache / "datasus" / "sim" / "do"
SIH_CACHE = PATHS.cache / "datasus" / "sih" / "rd"
SIM_INTERIM = PATHS.interim / "sim_a27_deaths.parquet"
SIH_INTERIM = PATHS.interim / "sih_a27_admissions.parquet"

SIM_FULL = OUT / "sim_a27_all_fields.parquet"
SIH_FULL = OUT / "sih_a27_all_fields.parquet"

#: The projections the frozen interim extracts were built with. Anything in the
#: raw layout and not in here has never left the cache.
SIM_PROJECTION = (
    "NUMERODO", "DTOBITO", "DTNASC", "IDADE", "SEXO", "RACACOR", "ESC",
    "CODMUNRES", "CODMUNOCOR", "LOCOCOR", "CAUSABAS", "CAUSABAS_O",
    "CIRCOBITO", "ASSISTMED", "TIPOBITO",
)
SIH_PROJECTION = (
    "N_AIH", "ANO_CMPT", "MES_CMPT", "UF_ZI", "MUNIC_RES", "MUNIC_MOV",
    "NASC", "SEXO", "IDADE", "COD_IDADE", "DT_INTER", "DT_SAIDA",
    "DIAG_PRINC", "DIAG_SECUN", "MORTE", "DIAS_PERM", "UTI_MES_TO",
    "VAL_TOT", "CAR_INT", "CNES",
)

#: Fields that a published number in this study currently depends on. Read off
#: 18_build_triangulation_panel.py (the panel builder) and R/06_ascertainment.R.
#: Everything else in the interim extracts is carried but never consumed.
SIM_CONSUMED = ("CODMUNRES", "DTOBITO")
SIH_CONSUMED = ("MUNIC_RES", "DT_INTER", "DIAG_PRINC", "MORTE")

# --------------------------------------------------------------------------
# Epidemiological blocks. Assignment is by field, not by regex: these two
# layouts are small, fixed and documented, so an explicit table is honest and
# a pattern would only hide the cases it gets wrong.
# --------------------------------------------------------------------------

SIM_BLOCKS: dict[str, str] = {
    "CONTADOR": "identification", "ORIGEM": "identification",
    "NUMERODO": "identification", "NUMERODV": "identification",
    "TIPOBITO": "event", "DTOBITO": "event", "HORAOBITO": "event",
    "DTATESTADO": "event",
    "NATURAL": "demography", "CODMUNNATU": "demography", "DTNASC": "demography",
    "IDADE": "demography", "SEXO": "demography", "RACACOR": "demography",
    "ESTCIV": "demography", "ESC": "demography", "ESC2010": "demography",
    "SERIESCFAL": "demography", "ESCFALAGR1": "demography",
    "OCUP": "occupation",
    "CODMUNRES": "geography", "CODMUNOCOR": "geography",
    "LOCOCOR": "place_of_death", "CODESTAB": "place_of_death",
    "ESTABDESCR": "place_of_death", "COMUNSVOIM": "place_of_death",
    "IDADEMAE": "maternal", "ESCMAE": "maternal", "ESCMAE2010": "maternal",
    "SERIESCMAE": "maternal", "OCUPMAE": "maternal", "QTDFILVIVO": "maternal",
    "QTDFILMORT": "maternal", "GRAVIDEZ": "maternal", "SEMAGESTAC": "maternal",
    "GESTACAO": "maternal", "PARTO": "maternal", "OBITOPARTO": "maternal",
    "PESO": "maternal", "TPMORTEOCO": "maternal", "OBITOGRAV": "maternal",
    "OBITOPUERP": "maternal", "ESCMAEAGR1": "maternal", "CAUSAMAT": "maternal",
    "MORTEPARTO": "maternal",
    "ASSISTMED": "care_and_verification", "EXAME": "care_and_verification",
    "CIRURGIA": "care_and_verification", "NECROPSIA": "care_and_verification",
    "ATESTANTE": "care_and_verification", "CRM": "care_and_verification",
    "LINHAA": "cause", "LINHAB": "cause", "LINHAC": "cause",
    "LINHAD": "cause", "LINHAII": "cause", "CAUSABAS": "cause",
    "CAUSABAS_O": "cause", "CB_PRE": "cause", "ATESTADO": "cause",
    "ALTCAUSA": "cause",
    "CIRCOBITO": "external_cause", "ACIDTRAB": "external_cause",
    "FONTE": "investigation", "TPPOS": "investigation",
    "DTINVESTIG": "investigation", "FONTEINV": "investigation",
    "STDOEPIDEM": "investigation", "STDONOVA": "investigation",
    "TPOBITOCOR": "investigation", "DTCONINV": "investigation",
    "FONTES": "investigation", "TPRESGINFO": "investigation",
    "TPNIVELINV": "investigation", "DTCADINV": "investigation",
    "NUDIASINF": "investigation", "DTCADINF": "investigation",
    "DTCONCASO": "investigation", "FONTESINF": "investigation",
    "NUMEROLOTE": "processing", "DTCADASTRO": "processing",
    "STCODIFICA": "processing", "CODIFICADO": "processing",
    "VERSAOSIST": "processing", "VERSAOSCB": "processing",
    "DTRECEBIM": "processing", "DTRECORIGA": "processing",
    "DIFDATA": "timeliness", "NUDIASOBCO": "timeliness",
    "NUDIASOBIN": "timeliness",
}

SIH_BLOCKS: dict[str, str] = {
    "N_AIH": "identification", "IDENT": "identification",
    "SEQUENCIA": "identification", "REMESSA": "identification",
    "SEQ_AIH5": "identification", "NUM_PROC": "identification",
    "CPF_AUT": "identification", "HOMONIMO": "identification",
    "ANO_CMPT": "time", "MES_CMPT": "time", "DT_INTER": "time",
    "DT_SAIDA": "time",
    "UF_ZI": "geography", "MUNIC_RES": "geography", "MUNIC_MOV": "geography",
    "CEP": "geography",
    "NASC": "demography", "SEXO": "demography", "IDADE": "demography",
    "COD_IDADE": "demography", "RACA_COR": "demography", "ETNIA": "demography",
    "INSTRU": "demography", "NACIONAL": "demography", "NUM_FILHOS": "demography",
    "CBOR": "occupation", "CNAER": "occupation", "VINCPREV": "occupation",
    "DIAG_PRINC": "diagnosis", "DIAG_SECUN": "diagnosis",
    "CID_NOTIF": "diagnosis", "CID_ASSO": "diagnosis", "CID_MORTE": "diagnosis",
    "DIAGSEC1": "diagnosis", "DIAGSEC2": "diagnosis", "DIAGSEC3": "diagnosis",
    "DIAGSEC4": "diagnosis", "DIAGSEC5": "diagnosis", "DIAGSEC6": "diagnosis",
    "DIAGSEC7": "diagnosis", "DIAGSEC8": "diagnosis", "DIAGSEC9": "diagnosis",
    "TPDISEC1": "diagnosis", "TPDISEC2": "diagnosis", "TPDISEC3": "diagnosis",
    "TPDISEC4": "diagnosis", "TPDISEC5": "diagnosis", "TPDISEC6": "diagnosis",
    "TPDISEC7": "diagnosis", "TPDISEC8": "diagnosis", "TPDISEC9": "diagnosis",
    "DIAS_PERM": "severity", "UTI_MES_IN": "severity", "UTI_MES_AN": "severity",
    "UTI_MES_AL": "severity", "UTI_MES_TO": "severity", "MARCA_UTI": "severity",
    "UTI_INT_IN": "severity", "UTI_INT_AN": "severity", "UTI_INT_AL": "severity",
    "UTI_INT_TO": "severity", "VAL_UTI": "severity", "MARCA_UCI": "severity",
    "VAL_UCI": "severity", "QT_DIARIAS": "severity", "DIAR_ACOM": "severity",
    "CAR_INT": "severity", "COMPLEX": "severity", "ESPEC": "severity",
    "MORTE": "outcome", "COBRANCA": "outcome",
    "PROC_SOLIC": "procedure", "PROC_REA": "procedure", "TOT_PT_SP": "procedure",
    "CNES": "provider", "CGC_HOSP": "provider", "CNPJ_MANT": "provider",
    "NATUREZA": "provider", "NAT_JUR": "provider", "GESTAO": "provider",
    "GESTOR_COD": "provider", "GESTOR_TP": "provider", "GESTOR_CPF": "provider",
    "GESTOR_DT": "provider",
    "VAL_SH": "cost", "VAL_SP": "cost", "VAL_SADT": "cost", "VAL_RN": "cost",
    "VAL_ACOMP": "cost", "VAL_ORTP": "cost", "VAL_SANGUE": "cost",
    "VAL_SADTSR": "cost", "VAL_TRANSP": "cost", "VAL_OBSANG": "cost",
    "VAL_PED1AC": "cost", "VAL_TOT": "cost", "US_TOT": "cost",
    "VAL_SH_FED": "cost", "VAL_SP_FED": "cost", "VAL_SH_GES": "cost",
    "VAL_SP_GES": "cost", "FINANC": "cost", "FAEC_TP": "cost",
    "REGCT": "cost", "RUBRICA": "cost",
    "IND_VDRL": "maternal_and_other", "GESTRISCO": "maternal_and_other",
    "INSC_PN": "maternal_and_other", "CONTRACEP1": "maternal_and_other",
    "CONTRACEP2": "maternal_and_other", "INFEHOSP": "maternal_and_other",
    "AUD_JUST": "administrative", "SIS_JUST": "administrative",
}

# --------------------------------------------------------------------------
# Stage 1: layout census from uncompressed dBase headers
# --------------------------------------------------------------------------


def _sim_key(name: str) -> tuple[str, int]:
    return name[2:4].upper(), int(name[4:8])


def _sih_key(name: str) -> tuple[str, int, int]:
    yy = int(name[4:6])
    return name[2:4].upper(), (1900 + yy if yy >= 92 else 2000 + yy), int(name[6:8])


def stage_headers() -> None:
    """One row per (system, field, year): presence and declared width."""
    rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for system, cache, keyfn in (
        ("SIM", SIM_CACHE, lambda n: _sim_key(n)[1]),
        ("SIH", SIH_CACHE, lambda n: _sih_key(n)[1]),
    ):
        files = sorted(cache.glob("*.DBC"))
        print(f"{system}: reading {len(files)} headers from {cache}")
        t0 = time.time()
        for p in files:
            year = keyfn(p.name)
            try:
                fields = dbf_header(p)
            except Exception as exc:  # noqa: BLE001
                file_rows.append({"system": system, "file": p.name, "year": year,
                                  "n_fields": None, "error": f"{type(exc).__name__}: {exc}"})
                continue
            file_rows.append({"system": system, "file": p.name, "year": year,
                              "n_fields": len(fields), "error": None})
            for f in fields:
                rows.append({
                    "system": system, "year": year, "field": f["name"],
                    "type": f["type"], "length": int(f["length"]),
                    "decimals": int(f["decimals"]),
                })
        print(f"  {len(files)} headers in {time.time() - t0:.1f}s")

    raw = pl.DataFrame(rows)
    raw.write_parquet(OUT / "layout_field_year_raw.parquet")
    pl.DataFrame(file_rows).write_csv(OUT / "layout_files.csv")

    per_year = (
        raw.group_by("system", "year", "field")
        .agg(
            pl.len().alias("n_files"),
            pl.col("type").unique().sort().str.join("/").alias("types"),
            pl.col("length").min().alias("length_min"),
            pl.col("length").max().alias("length_max"),
        )
        .sort("system", "field", "year")
    )
    per_year.write_csv(OUT / "layout_field_by_year.csv")

    files_per_year = (
        raw.select("system", "year", "field").unique()
        .group_by("system", "year").agg(pl.len().alias("fields_in_year"))
    )
    total_files = (
        pl.DataFrame(file_rows).group_by("system", "year")
        .agg(pl.len().alias("files_in_year"))
    )

    summary = (
        per_year.join(total_files, on=["system", "year"], how="left")
        .with_columns(
            (pl.col("n_files") == pl.col("files_in_year")).alias("in_all_files")
        )
        .group_by("system", "field")
        .agg(
            pl.col("year").min().alias("first_year"),
            pl.col("year").max().alias("last_year"),
            pl.col("year").n_unique().alias("n_years"),
            pl.col("n_files").sum().alias("n_files_present"),
            pl.col("in_all_files").all().alias("in_every_file_of_its_years"),
            pl.col("length_min").min().alias("width_min"),
            pl.col("length_max").max().alias("width_max"),
            pl.col("types").unique().sort().str.join("/").alias("types"),
        )
        .with_columns(
            (pl.col("width_min") != pl.col("width_max")).alias("width_changed")
        )
        .sort("system", "field")
    )

    proj = {"SIM": set(SIM_PROJECTION), "SIH": set(SIH_PROJECTION)}
    used = {"SIM": set(SIM_CONSUMED), "SIH": set(SIH_CONSUMED)}
    blocks = {"SIM": SIM_BLOCKS, "SIH": SIH_BLOCKS}
    summary = summary.with_columns(
        pl.struct("system", "field").map_elements(
            lambda s: blocks[s["system"]].get(s["field"], "other"),
            return_dtype=pl.Utf8,
        ).alias("block"),
        pl.struct("system", "field").map_elements(
            lambda s: s["field"] in proj[s["system"]], return_dtype=pl.Boolean
        ).alias("in_interim_extract"),
        pl.struct("system", "field").map_elements(
            lambda s: s["field"] in used[s["system"]], return_dtype=pl.Boolean
        ).alias("consumed_by_a_published_number"),
    ).with_columns(
        pl.when(pl.col("consumed_by_a_published_number")).then(pl.lit("used"))
        .when(pl.col("in_interim_extract")).then(pl.lit("extracted_but_unused"))
        .otherwise(pl.lit("dark_in_raw_cache"))
        .alias("status")
    ).sort("system", "block", "field")
    summary.write_csv(OUT / "layout_field_summary.csv")

    print("\n=== layout census ===")
    for system in ("SIM", "SIH"):
        s = summary.filter(pl.col("system") == system)
        print(f"{system}: {s.height} distinct fields across the cached series")
        for st in ("used", "extracted_but_unused", "dark_in_raw_cache"):
            n = s.filter(pl.col("status") == st).height
            print(f"   {st:<26} {n:>4}")
        drift = s.filter(pl.col("n_years") < s["n_years"].max())
        if drift.height:
            print(f"   fields NOT present in every year: {drift.height}")
            for r in drift.sort("first_year").iter_rows(named=True):
                print(f"      {r['field']:<12} {r['first_year']}-{r['last_year']} "
                      f"({r['n_years']} yrs, {r['block']})")
        w = s.filter(pl.col("width_changed"))
        if w.height:
            print(f"   fields whose declared width changed: "
                  f"{', '.join(w['field'].to_list())}")


# --------------------------------------------------------------------------
# Stage 2: deep re-scan of the raw cache, all fields, A27-relevant rows
# --------------------------------------------------------------------------

SIM_PREFIX_RULES: dict[str, tuple[str, ...]] = {"CAUSABAS": ("A27",)}
#: The multiple-cause chain packs several 4-character codes into one string
#: with no separator, so it is matched by substring. Any count from these is a
#: *mention* count, not a death count.
SIM_SUBSTR_RULES: dict[str, tuple[str, ...]] = {
    "LINHAA": ("A27",), "LINHAB": ("A27",), "LINHAC": ("A27",),
    "LINHAD": ("A27",), "LINHAII": ("A27",), "CAUSABAS_O": ("A27",),
}

SIH_PREFIX_RULES: dict[str, tuple[str, ...]] = {
    c: ("A27",) for c in (
        "DIAG_PRINC", "DIAG_SECUN", "CID_NOTIF", "CID_ASSO", "CID_MORTE",
        "DIAGSEC1", "DIAGSEC2", "DIAGSEC3", "DIAGSEC4", "DIAGSEC5",
        "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9",
    )
}
SIH_SUBSTR_RULES: dict[str, tuple[str, ...]] = {}


def _scan_one(job: tuple) -> tuple[str, bytes | None, dict[str, Any]]:
    """Decompress one cached file, keep A27-relevant rows, return every field.

    Mirrors :func:`brepi.io.dbc.read_dbc_where` but adds substring rules (for
    the SIM multiple-cause chain, where an anchored match sees only the first
    of several concatenated codes) and returns per-rule hit counts so the
    marginal contribution of each matching field is measured rather than
    assumed.
    """
    import tempfile

    from brepi.io.dbc import _decompress_to, _read_geometry, is_plain_dbf

    name, path_s, prefix_rules, substr_rules, tags = job
    path = Path(path_s)
    stats: dict[str, Any] = {"file": name, **tags, "error": None}
    try:
        with tempfile.TemporaryDirectory(prefix="brepi_audit_") as tmp:
            plain = path if is_plain_dbf(path) else _decompress_to(path, Path(tmp))
            _v, n_records, header_len, record_len = _read_geometry(plain)
            descs = field_offsets(plain)
            by_name = {d["name"]: d for d in descs}
            available = max((plain.stat().st_size - header_len) // record_len, 0)
            n = min(n_records, available)
            stats["n_records_declared"] = int(n_records)
            stats["n_records_read"] = int(n)
            if n == 0:
                return name, None, stats
            block = np.fromfile(
                plain, dtype=np.uint8, count=n * record_len, offset=header_len
            ).reshape(n, record_len)
            live = block[:, 0] == 0x20
            stats["n_live"] = int(live.sum())

            hit = np.zeros(n, dtype=bool)
            for rules, mode in ((prefix_rules, "prefix"), (substr_rules, "substr")):
                for col, prefixes in rules.items():
                    d = by_name.get(col)
                    if d is None:
                        stats[f"hit_{col}"] = None  # field absent from this layout
                        continue
                    length = int(d["length"])
                    raw = np.ascontiguousarray(
                        block[:, d["offset"]: d["offset"] + length]
                    ).tobytes()
                    view = np.char.strip(np.frombuffer(raw, dtype=f"S{length}"))
                    col_hit = np.zeros(n, dtype=bool)
                    for pre in prefixes:
                        needle = pre.strip().upper().encode("ascii")
                        if mode == "prefix":
                            col_hit |= np.char.startswith(view, needle)
                        else:
                            col_hit |= np.char.find(view, needle) >= 0
                    col_hit &= live
                    stats[f"hit_{col}"] = int(col_hit.sum())
                    hit |= col_hit

            keep = live & hit
            stats["n_kept"] = int(keep.sum())
            if not keep.any():
                return name, None, stats
            rows = np.ascontiguousarray(block[keep])
            del block

            from brepi.io.dbc import _decode_column

            data = {
                d["name"]: _decode_column(rows, d["offset"], int(d["length"]))
                for d in descs
            }
            frame = pl.DataFrame(
                data, schema={d["name"]: pl.Utf8 for d in descs}, strict=False
            ).with_columns(
                *[pl.lit(v).alias(k) for k, v in tags.items()],
                pl.lit(name, dtype=pl.Utf8).alias("_src_file"),
            )
            import io

            buf = io.BytesIO()
            frame.write_ipc(buf)
            return name, buf.getvalue(), stats
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the pass
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return name, None, stats


def _run_scan(
    system: str,
    files: Sequence[Path],
    tagger,
    prefix_rules: dict,
    substr_rules: dict,
    out_parquet: Path,
    stats_csv: Path,
) -> None:
    import io as _io

    jobs = [
        (p.name, str(p), prefix_rules, substr_rules, tagger(p.name)) for p in files
    ]
    workers = max(1, min(int(os.environ.get("BREPI_DECODE_WORKERS", "12")), len(jobs)))
    print(f"{system}: scanning {len(jobs)} cached files with {workers} workers")
    parts: list[pl.DataFrame] = []
    stat_rows: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, payload, stats in pool.map(_scan_one, jobs, chunksize=2):
            done += 1
            stat_rows.append(stats)
            if payload is not None:
                parts.append(pl.read_ipc(_io.BytesIO(payload)))
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {system} {done}/{len(jobs)} files, {el:.0f}s elapsed, "
                      f"eta {el / done * (len(jobs) - done):.0f}s, "
                      f"{sum(p.height for p in parts):,} rows kept", flush=True)
    pl.DataFrame(stat_rows, infer_schema_length=None).write_csv(stats_csv)
    if not parts:
        print(f"{system}: nothing matched -- refusing to write an empty extract")
        return
    frame = pl.concat(parts, how="diagonal")
    frame.write_parquet(out_parquet)
    print(f"{system}: {frame.height:,} rows x {frame.width} fields -> {out_parquet}")


def stage_deep(which: str) -> None:
    if which in ("sim", "both"):
        files = sorted(SIM_CACHE.glob("*.DBC"))
        _run_scan(
            "SIM", files,
            lambda n: {"_uf": _sim_key(n)[0], "_file_year": _sim_key(n)[1]},
            SIM_PREFIX_RULES, SIM_SUBSTR_RULES,
            SIM_FULL, OUT / "sim_scan_file_stats.csv",
        )
    if which in ("sih", "both"):
        files = sorted(SIH_CACHE.glob("*.DBC"))
        _run_scan(
            "SIH", files,
            lambda n: {"_uf": _sih_key(n)[0], "_file_year": _sih_key(n)[1],
                       "_file_month": _sih_key(n)[2]},
            SIH_PREFIX_RULES, SIH_SUBSTR_RULES,
            SIH_FULL, OUT / "sih_scan_file_stats.csv",
        )


# --------------------------------------------------------------------------
# Stage 3: column profiles
# --------------------------------------------------------------------------

_BLANK_LIKE = ("", "NA", "NULL", "NONE", ".", "-")


def profile(
    df: pl.DataFrame, system: str, population: str, blocks: dict[str, str],
    projection: set[str], consumed: set[str], cb_system: str,
) -> pl.DataFrame:
    """One row per column: completeness, cardinality, modal values, decode state.

    Completeness is reported in three columns, never one, because DATASUS
    encodes "absent" three different ways and collapsing them produces exactly
    the wrong answer in both directions:

    ``pct_nonnull``
        The field has bytes in it. On a fixed-width dBase record almost
        everything is non-null, so on its own this over-states completeness.
    ``pct_filled``
        Non-null and not one of the blank sentinels (empty, ``NA``, ``.``).
    ``pct_nonzero``
        Filled and not a pure run of zeros. This is the honest denominator for
        *coded* fields, where ``0000`` is the documented "not informed" value
        (``DIAG_SECUN`` after the 2015 break is uniformly ``0000``). It is the
        *wrong* denominator for *quantity* fields, where zero is a real
        measurement: ``MORTE = 0`` means the patient lived and
        ``DIAS_PERM = 0`` means a same-day stay. Both numbers are therefore
        printed and the reader picks the one the field's semantics warrant.

    ``num_*`` columns are populated when the column parses as a number for at
    least 90% of its filled values, which is what separates a quantity from a
    code that happens to be digits.
    """
    n = df.height
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        s = df[col].cast(pl.Utf8, strict=False).str.strip_chars()
        nonnull = int(s.is_not_null().sum())
        filled_mask = s.is_not_null() & ~s.is_in(list(_BLANK_LIKE))
        n_filled = int(filled_mask.sum())
        zero_mask = filled_mask & (s.str.replace_all("0", "") == "")
        n_zero = int(zero_mask.sum())
        n_nonzero = n_filled - n_zero
        distinct = int(s.n_unique())
        vc = s.value_counts(sort=True).head(6)
        top = "; ".join(
            f"{r[col] if r[col] is not None else '<null>'}={r['count']}"
            for r in vc.iter_rows(named=True)
        )
        num = s.cast(pl.Float64, strict=False)
        n_numeric = int(num.is_not_null().sum())
        numeric_like = n_filled > 0 and n_numeric >= 0.9 * n_filled
        concept_id = binding_for(cb_system, col)
        st: dict[str, int] = {}
        if concept_id:
            try:
                dec = df.select(categorical_exprs(concept_id, col, suffix="_d"))
                cnt = dec["_d_state"].value_counts()
                st = {r["_d_state"]: r["count"] for r in cnt.iter_rows(named=True)}
            except Exception:  # noqa: BLE001 - a broken binding is reported, not fatal
                st = {}
        rows.append({
            "system": system,
            "population": population,
            "block": blocks.get(col, "other"),
            "variable": col,
            "n_rows": n,
            "pct_nonnull": round(100 * nonnull / n, 2) if n else None,
            "pct_filled": round(100 * n_filled / n, 2) if n else None,
            "pct_nonzero": round(100 * n_nonzero / n, 2) if n else None,
            "n_distinct": distinct,
            "numeric_like": numeric_like,
            "num_min": float(num.min()) if numeric_like and n_numeric else None,
            "num_median": float(num.median()) if numeric_like and n_numeric else None,
            "num_p90": float(num.quantile(0.9)) if numeric_like and n_numeric else None,
            "num_max": float(num.max()) if numeric_like and n_numeric else None,
            "codebook_concept": concept_id,
            "state_valid": st.get("valid"),
            "state_unknown": st.get("unknown"),
            "state_missing": st.get("missing"),
            "state_invalid": st.get("invalid"),
            "pct_valid": round(100 * st["valid"] / n, 2) if st.get("valid") and n else None,
            "in_interim_extract": col in projection,
            "consumed_by_a_published_number": col in consumed,
            "top_values": top,
        })
    # An explicit schema, not inference: a population where no field happens to
    # carry an "invalid" code infers that column as Null and then refuses to
    # stack against a population that does.
    schema = {
        "system": pl.Utf8, "population": pl.Utf8, "block": pl.Utf8,
        "variable": pl.Utf8, "n_rows": pl.Int64, "pct_nonnull": pl.Float64,
        "pct_filled": pl.Float64, "pct_nonzero": pl.Float64,
        "n_distinct": pl.Int64, "numeric_like": pl.Boolean,
        "num_min": pl.Float64, "num_median": pl.Float64,
        "num_p90": pl.Float64, "num_max": pl.Float64,
        "codebook_concept": pl.Utf8, "state_valid": pl.Int64,
        "state_unknown": pl.Int64, "state_missing": pl.Int64,
        "state_invalid": pl.Int64, "pct_valid": pl.Float64,
        "in_interim_extract": pl.Boolean,
        "consumed_by_a_published_number": pl.Boolean, "top_values": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema).sort(["block", "variable"])


def completeness_by_year(
    df: pl.DataFrame, year_expr: pl.Expr, fields: Sequence[str], label: str,
) -> pl.DataFrame:
    """Per-year filled-rate for named fields, on the frame's own population.

    A field that is 80% complete overall but 0% complete before 2014 is not an
    80%-complete field; it is a field with a schema break, and only the yearly
    view shows which.
    """
    present = [c for c in fields if c in df.columns]
    d = df.with_columns(year_expr.alias("_year"))
    agg = [pl.len().alias("n")]
    for c in present:
        s = pl.col(c).cast(pl.Utf8).str.strip_chars()
        filled = s.is_not_null() & ~s.is_in(list(_BLANK_LIKE))
        nonzero = filled & (s.str.replace_all("0", "") != "")
        agg.append(nonzero.mean().alias(f"{c}_pct_nonzero"))
    out = d.group_by("_year").agg(agg).sort("_year").with_columns(
        pl.lit(label).alias("population")
    )
    return out


def stage_profile() -> None:
    frames: list[pl.DataFrame] = []

    sim_i = pl.read_parquet(SIM_INTERIM)
    frames.append(profile(
        sim_i, "SIM",
        f"frozen interim extract: {sim_i.height} deaths with CAUSABAS starting "
        "A27, DO files 2007-2024, 27 UFs",
        SIM_BLOCKS, set(SIM_PROJECTION), set(SIM_CONSUMED), "SIM-DO",
    ))
    sih_i = pl.read_parquet(SIH_INTERIM)
    frames.append(profile(
        sih_i, "SIH",
        f"frozen interim extract: {sih_i.height} AIH with A27 prefix in "
        "DIAG_PRINC or DIAG_SECUN, RD files 2008-2024, 27 UFs",
        SIH_BLOCKS, set(SIH_PROJECTION), set(SIH_CONSUMED), "SIH-RD",
    ))

    if SIM_FULL.exists():
        sim_f = pl.read_parquet(SIM_FULL)
        frames.append(profile(
            sim_f, "SIM",
            f"deep re-scan, all fields: {sim_f.height} death certificates with "
            "A27 in CAUSABAS or anywhere in the multiple-cause chain, "
            "DO 2007-2024",
            SIM_BLOCKS, set(SIM_PROJECTION), set(SIM_CONSUMED), "SIM-DO",
        ))
    if SIH_FULL.exists():
        sih_f = pl.read_parquet(SIH_FULL)
        frames.append(profile(
            sih_f, "SIH",
            f"deep re-scan, all fields: {sih_f.height} AIH with A27 in any "
            "diagnosis field including DIAGSEC1-9, RD 2008-2024",
            SIH_BLOCKS, set(SIH_PROJECTION), set(SIH_CONSUMED), "SIH-RD",
        ))

    cat = pl.concat(frames, how="diagonal")
    cat.write_csv(OUT / "variable_profile.csv")
    print(f"wrote {OUT / 'variable_profile.csv'} ({cat.height} variable-population rows)")

    for pop in cat["population"].unique().sort():
        sub = cat.filter(pl.col("population") == pop)
        print(f"\n=== {pop} ===")
        print(f"{'block':<22}{'vars':>5}{'median %filled':>17}{'median %nonzero':>18}")
        summ = sub.group_by("block").agg(
            pl.len().alias("vars"),
            pl.col("pct_filled").median().alias("med_filled"),
            pl.col("pct_nonzero").median().alias("med_nonzero"),
        ).sort("block")
        for r in summ.iter_rows(named=True):
            print(f"{r['block']:<22}{r['vars']:>5}{r['med_filled']:>17.1f}"
                  f"{r['med_nonzero']:>18.1f}")

    # --- per-year completeness for the fields an analysis would actually want -
    year_frames: list[pl.DataFrame] = []
    sim_year = (
        pl.col("DTOBITO").cast(pl.Utf8).str.slice(4, 4).cast(pl.Int32, strict=False)
    )
    sih_year = (
        pl.col("DT_INTER").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).dt.year()
    )
    year_frames.append(completeness_by_year(
        sim_i, sim_year,
        ["IDADE", "SEXO", "RACACOR", "ESC", "ASSISTMED", "LOCOCOR", "CODMUNOCOR",
         "CAUSABAS_O", "CIRCOBITO"],
        "SIM interim extract",
    ))
    year_frames.append(completeness_by_year(
        sih_i, sih_year,
        ["DIAS_PERM", "UTI_MES_TO", "VAL_TOT", "CAR_INT", "DIAG_SECUN", "CNES",
         "MORTE", "COD_IDADE"],
        "SIH interim extract",
    ))
    if SIM_FULL.exists():
        sim_f = pl.read_parquet(SIM_FULL)
        year_frames.append(completeness_by_year(
            sim_f, sim_year,
            ["ESTCIV", "OCUP", "NECROPSIA", "EXAME", "CIRURGIA", "ATESTANTE",
             "ESC2010", "CODESTAB", "LINHAA", "LINHAII", "DIFDATA", "NUDIASOBCO",
             "STDOEPIDEM", "TPPOS", "DTINVESTIG", "HORAOBITO"],
            "SIM deep scan (all fields)",
        ))
    if SIH_FULL.exists():
        sih_f = pl.read_parquet(SIH_FULL)
        year_frames.append(completeness_by_year(
            sih_f, sih_year,
            ["UTI_INT_TO", "MARCA_UTI", "VAL_UTI", "QT_DIARIAS", "PROC_REA",
             "COBRANCA", "COMPLEX", "ESPEC", "RACA_COR", "INSTRU", "CBOR",
             "DIAGSEC1", "DIAGSEC2", "TPDISEC1", "CID_ASSO", "CID_MORTE",
             "MARCA_UCI", "GESTOR_TP", "NAT_JUR", "FINANC"],
            "SIH deep scan (all fields)",
        ))
    ybook = pl.concat(year_frames, how="diagonal")
    ybook.write_csv(OUT / "completeness_by_year.csv")
    print(f"\nwrote {OUT / 'completeness_by_year.csv'}")


# --------------------------------------------------------------------------
# Stage 4: severity feasibility
# --------------------------------------------------------------------------


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def stage_severity() -> None:
    """Can SIH measure severity directly instead of by the hospitalised-share proxy?

    The population is the **frozen interim extract** -- the A27 admissions the
    study actually holds -- not the wider deep-scan set. Two reasons: the
    numbers below are then directly comparable to every published SIH figure in
    this project, and the wider set's extra rows are an ascertainment question,
    answered in the ``reach`` stage, not a severity one. Fields absent from the
    interim projection are read from the deep scan and joined on ``N_AIH``,
    which is unique within the extract.
    """
    df = pl.read_parquet(SIH_INTERIM)
    src = SIH_INTERIM
    if SIH_FULL.exists():
        extra = [
            c for c in (
                "IDENT", "COBRANCA", "MARCA_UTI", "MARCA_UCI", "COMPLEX",
                "ESPEC", "PROC_REA", "PROC_SOLIC", "UTI_INT_TO", "VAL_UTI",
                "VAL_UCI", "QT_DIARIAS", "DIAR_ACOM", "UTI_MES_IN",
                "UTI_MES_AN", "UTI_MES_AL",
            )
        ]
        full = pl.read_parquet(SIH_FULL)
        cols = [c for c in extra if c in full.columns]
        if cols and "N_AIH" in full.columns:
            side = full.select(["N_AIH", *cols]).unique(subset=["N_AIH"])
            df = df.join(side, on="N_AIH", how="left")
            print(f"severity: joined {len(cols)} dark field(s) from the deep scan "
                  f"on N_AIH")
    print(f"severity: population = {src.name}, {df.height:,} rows, "
          f"{df.width} fields after join")

    have = set(df.columns)
    d = df.with_columns(
        pl.col("DT_INTER").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("admit"),
        pl.col("DT_SAIDA").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("discharge"),
        _num("DIAS_PERM").alias("los"),
        _num("UTI_MES_TO").alias("icu_days_month"),
        _num("VAL_TOT").alias("cost_total"),
        (pl.col("MORTE").cast(pl.Utf8).str.strip_chars() == "1").alias("died"),
        pl.col("DIAG_PRINC").cast(pl.Utf8).str.strip_chars().str.starts_with("A27")
        .fill_null(False).alias("principal_a27"),
    )
    for extra, alias in (
        ("UTI_INT_TO", "icu_int_total"), ("VAL_UTI", "icu_value"),
        ("QT_DIARIAS", "billed_days"), ("DIAR_ACOM", "companion_days"),
        ("UTI_MES_IN", "icu_days_in"), ("UTI_MES_AN", "icu_days_an"),
        ("UTI_MES_AL", "icu_days_al"), ("VAL_UCI", "uci_value"),
    ):
        if extra in have:
            d = d.with_columns(_num(extra).alias(alias))
    if "MARCA_UTI" in have:
        d = d.with_columns(
            pl.col("MARCA_UTI").cast(pl.Utf8).str.strip_chars().alias("icu_type_code")
        )
    d = d.with_columns(
        pl.col("admit").dt.year().alias("year"),
        (pl.col("discharge") - pl.col("admit")).dt.total_days().alias("los_from_dates"),
    )

    window = d.filter(pl.col("year").is_between(2008, 2024))
    print(f"  analytic window 2008-2024: {window.height:,} of {d.height:,} rows")

    # --- 4a. completeness and distribution of the severity markers, by year --
    icu_expr = (
        pl.col("icu_days_month").fill_null(0) > 0 if "icu_days_month" in d.columns
        else pl.lit(False)
    )
    agg = [
        pl.len().alias("n_admissions"),
        pl.col("los").is_not_null().sum().alias("los_nonnull"),
        (pl.col("los") > 0).sum().alias("los_positive"),
        pl.col("los").median().alias("los_median"),
        pl.col("los").mean().alias("los_mean"),
        pl.col("los").quantile(0.9).alias("los_p90"),
        pl.col("los").max().alias("los_max"),
        (pl.col("los") >= 7).mean().alias("frac_los_ge7"),
        pl.col("icu_days_month").is_not_null().sum().alias("icu_nonnull"),
        icu_expr.sum().alias("n_with_icu"),
        icu_expr.mean().alias("frac_with_icu"),
        pl.col("icu_days_month").filter(icu_expr).median().alias("icu_days_median_if_any"),
        pl.col("died").mean().alias("in_hospital_cfr"),
        pl.col("cost_total").median().alias("cost_median"),
        (pl.col("los") == pl.col("los_from_dates")).mean().alias("frac_los_matches_dates"),
    ]
    if "icu_int_total" in d.columns:
        agg.append((pl.col("icu_int_total").fill_null(0) > 0).mean().alias("frac_uti_int_positive"))
    if "icu_value" in d.columns:
        agg.append((pl.col("icu_value").fill_null(0) > 0).mean().alias("frac_val_uti_positive"))
    if "icu_type_code" in d.columns:
        agg.append(
            (~pl.col("icu_type_code").is_in(["", "00", "0", "99"]) &
             pl.col("icu_type_code").is_not_null()).mean().alias("frac_marca_uti_set")
        )
    by_year = window.group_by("year").agg(agg).sort("year")
    by_year.write_csv(OUT / "severity_by_year.csv")

    print("\n=== SIH severity markers, A27 admissions, by admission year ===")
    print(f"{'year':>5}{'n':>7}{'LOS med':>9}{'LOS p90':>9}{'%LOS>0':>8}"
          f"{'%ICU':>7}{'ICU d med':>10}{'%died':>7}{'%LOS=dates':>11}")
    for r in by_year.iter_rows(named=True):
        print(f"{r['year']:>5}{r['n_admissions']:>7}"
              f"{(r['los_median'] if r['los_median'] is not None else -1):>9.1f}"
              f"{(r['los_p90'] if r['los_p90'] is not None else -1):>9.1f}"
              f"{100 * r['los_positive'] / r['n_admissions']:>8.1f}"
              f"{100 * (r['frac_with_icu'] or 0):>7.1f}"
              f"{(r['icu_days_median_if_any'] if r['icu_days_median_if_any'] is not None else -1):>10.1f}"
              f"{100 * (r['in_hospital_cfr'] or 0):>7.2f}"
              f"{100 * (r['frac_los_matches_dates'] or 0):>11.1f}")

    # --- 4b. does severity discriminate outcome? ----------------------------
    grad = window.group_by(
        pl.when(pl.col("los") <= 2).then(pl.lit("1: 0-2 days"))
        .when(pl.col("los") <= 6).then(pl.lit("2: 3-6 days"))
        .when(pl.col("los") <= 13).then(pl.lit("3: 7-13 days"))
        .when(pl.col("los").is_not_null()).then(pl.lit("4: 14+ days"))
        .otherwise(pl.lit("5: LOS missing")).alias("los_band")
    ).agg(
        pl.len().alias("n"),
        pl.col("died").mean().alias("in_hospital_cfr"),
        icu_expr.mean().alias("frac_with_icu"),
        pl.col("cost_total").median().alias("cost_median"),
    ).sort("los_band")
    grad.write_csv(OUT / "severity_los_gradient.csv")
    print("\n=== in-hospital fatality by length-of-stay band ===")
    for r in grad.iter_rows(named=True):
        print(f"  {r['los_band']:<16} n={r['n']:>6}  CFR={100 * r['in_hospital_cfr']:.2f}%"
              f"  ICU={100 * r['frac_with_icu']:.1f}%")

    icu_tab = window.group_by(icu_expr.alias("any_icu")).agg(
        pl.len().alias("n"),
        pl.col("died").mean().alias("in_hospital_cfr"),
        pl.col("los").median().alias("los_median"),
        pl.col("cost_total").median().alias("cost_median"),
    ).sort("any_icu")
    icu_tab.write_csv(OUT / "severity_icu_contrast.csv")
    print("\n=== in-hospital fatality by any ICU day in the competence month ===")
    for r in icu_tab.iter_rows(named=True):
        print(f"  ICU={str(r['any_icu']):<6} n={r['n']:>6}  "
              f"CFR={100 * r['in_hospital_cfr']:.2f}%  LOS median={r['los_median']}")

    # --- 4c. the AIH-splitting trap ----------------------------------------
    # UTI_MES_TO counts ICU days *in the competence month*. A stay that crosses
    # a month boundary is billed as more than one AIH, so both LOS and ICU days
    # are per-AIH, not per-episode. Quantify how often that can happen.
    cross = window.with_columns(
        (pl.col("discharge").dt.month() != pl.col("admit").dt.month()).alias("crosses_month")
    )
    cm = cross.group_by("crosses_month").agg(
        pl.len().alias("n"), pl.col("los").median().alias("los_median")
    ).sort("crosses_month")
    cm.write_csv(OUT / "severity_month_crossing.csv")
    print("\n=== admissions whose stay crosses a competence month ===")
    for r in cm.iter_rows(named=True):
        print(f"  crosses={str(r['crosses_month']):<6} n={r['n']:>6} "
              f"LOS median={r['los_median']}")

    # --- 4d. health-region aggregability -----------------------------------
    # A severity index is only useful to this study if it survives aggregation
    # to health region x year, the modelling grain. Report the cell census.
    cells = window.group_by("MUNIC_RES", "year").agg(pl.len().alias("n"))
    print(f"\n  municipality x year cells with >=1 A27 admission: {cells.height}, "
          f"median admissions per cell {cells['n'].median()}, "
          f"cells with >=5 admissions {cells.filter(pl.col('n') >= 5).height}")

    hr_cells = None
    tri_path = PATHS.panel / "triangulation_municipality_year.parquet"
    if tri_path.exists():
        xwalk = (
            pl.read_parquet(tri_path)
            .select("munic_code", "health_region_code").unique()
            .with_columns(
                pl.col("munic_code").cast(pl.Utf8).str.slice(0, 6).alias("MUNIC_RES")
            )
            .select("MUNIC_RES", "health_region_code").unique(subset=["MUNIC_RES"])
        )
        joined = window.join(xwalk, on="MUNIC_RES", how="left")
        unmatched = int(joined["health_region_code"].is_null().sum())
        hr_cells = (
            joined.drop_nulls("health_region_code")
            .group_by("health_region_code", "year")
            .agg(
                pl.len().alias("n"),
                pl.col("los").median().alias("los_median"),
                icu_expr.mean().alias("frac_with_icu"),
                pl.col("died").mean().alias("cfr"),
            )
        )
        hr_cells.write_csv(OUT / "severity_health_region_year.csv")
        n_regions = xwalk["health_region_code"].n_unique()
        print(f"  health region x year cells with >=1 A27 admission: {hr_cells.height} "
              f"(of {n_regions} regions x 17 years = {n_regions * 17} possible); "
              f"median admissions per cell {hr_cells['n'].median()}; "
              f">=10 admissions in {hr_cells.filter(pl.col('n') >= 10).height} cells; "
              f"{unmatched} admissions had no health-region match")

    # --- 4e. the fields that decide whether an AIH is an episode ------------
    # ``IDENT`` separates a principal AIH from a long-stay continuation, and
    # ``COBRANCA`` gives the reason the AIH was closed, which is where a
    # transfer announces itself. Both are dark. Without them the admission
    # count double-counts every long or transferred stay, and the severity
    # measures are per-billing-record rather than per-episode.
    dist_rows: list[dict[str, Any]] = []
    for col, concept_id in (
        ("IDENT", "sih_rd_ident"), ("COBRANCA", "sih_rd_cobranca"),
        ("MARCA_UTI", "sih_rd_marca_uti"), ("COMPLEX", "sih_rd_complex"),
        ("ESPEC", "sih_rd_espec"), ("CAR_INT", "sih_rd_car_int"),
        ("MARCA_UCI", "sih_rd_marca_uci"),
    ):
        if col not in have:
            continue
        decoded = window.with_columns(
            categorical_exprs(concept_id, col, suffix="_lbl")
        )
        vc = (
            decoded.group_by(col, "_lbl", "_lbl_state")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True).head(12)
        )
        for r in vc.iter_rows(named=True):
            dist_rows.append({
                "variable": col, "code": r[col], "label": r["_lbl"],
                "decode_state": r["_lbl_state"], "n": r["n"],
                "pct": round(100 * r["n"] / window.height, 2),
            })
    if "PROC_REA" in have:
        vc = (
            window.group_by(pl.col("PROC_REA").cast(pl.Utf8).str.strip_chars())
            .agg(pl.len().alias("n")).sort("n", descending=True).head(12)
        )
        for r in vc.iter_rows(named=True):
            dist_rows.append({
                "variable": "PROC_REA", "code": r["PROC_REA"], "label": None,
                "decode_state": None, "n": r["n"],
                "pct": round(100 * r["n"] / window.height, 2),
            })
    if dist_rows:
        pl.DataFrame(dist_rows).write_csv(OUT / "severity_code_distributions.csv")
        print("\n=== dark administrative-clinical fields, top codes ===")
        for var in dict.fromkeys(r["variable"] for r in dist_rows):
            top = [r for r in dist_rows if r["variable"] == var][:5]
            print(f"  {var}: " + "; ".join(
                f"{t['code']}={t['pct']}%"
                + (f" ({t['label']})" if t["label"] else "") for t in top
            ))

    summary = {
        "source": str(src),
        "rows": df.height,
        "rows_in_window_2008_2024": window.height,
        "los": {
            "pct_nonnull": round(100 * float(window["los"].is_not_null().mean()), 2),
            "pct_positive": round(100 * float((window["los"] > 0).mean()), 2),
            "median": float(window["los"].median()),
            "p90": float(window["los"].quantile(0.9)),
            "max": float(window["los"].max()),
            "pct_equal_to_discharge_minus_admission": round(
                100 * float((window["los"] == window["los_from_dates"]).mean()), 2
            ),
        },
        "icu": {
            "field": "UTI_MES_TO (ICU days billed in the competence month)",
            "pct_nonnull": round(100 * float(window["icu_days_month"].is_not_null().mean()), 2),
            "pct_any_icu": round(100 * float((window["icu_days_month"].fill_null(0) > 0).mean()), 2),
        },
        "in_hospital_cfr_overall": round(100 * float(window["died"].mean()), 3),
        "municipality_year_cells": cells.height,
        "municipality_year_cells_ge5": cells.filter(pl.col("n") >= 5).height,
        "health_region_year_cells": None if hr_cells is None else hr_cells.height,
        "health_region_year_cells_ge10": (
            None if hr_cells is None
            else hr_cells.filter(pl.col("n") >= 10).height
        ),
    }
    (OUT / "severity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'severity_summary.json'}")


# --------------------------------------------------------------------------
# Stage 5: what the wider row filter buys -- the ascertainment finding
# --------------------------------------------------------------------------


def stage_reach() -> None:
    """How many A27 records the current extraction rules never see, and when."""
    report: dict[str, Any] = {}

    if SIH_FULL.exists():
        f = pl.read_parquet(SIH_FULL)
        code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731
        sec_slots = [c for c in f.columns if c.startswith("DIAGSEC") and c[7:].isdigit()]
        principal = code("DIAG_PRINC").str.starts_with("A27").fill_null(False)
        legacy_sec = code("DIAG_SECUN").str.starts_with("A27").fill_null(False)
        new_sec = pl.lit(False)
        for c in sec_slots:
            new_sec = new_sec | code(c).str.starts_with("A27").fill_null(False)
        other = pl.lit(False)
        for c in ("CID_ASSO", "CID_MORTE", "CID_NOTIF"):
            if c in f.columns:
                other = other | code(c).str.starts_with("A27").fill_null(False)
        g = f.with_columns(
            principal.alias("a27_principal"),
            legacy_sec.alias("a27_diag_secun"),
            new_sec.alias("a27_diagsec_slots"),
            other.alias("a27_other_cid_field"),
            pl.col("DT_INTER").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
            .dt.year().alias("year"),
        ).with_columns(
            (pl.col("a27_principal") | pl.col("a27_diag_secun")).alias("caught_by_current_rule")
        )
        by_year = g.group_by("year").agg(
            pl.len().alias("n_a27_any_field"),
            pl.col("a27_principal").sum().alias("principal"),
            pl.col("a27_diag_secun").sum().alias("diag_secun"),
            pl.col("a27_diagsec_slots").sum().alias("diagsec_1_9"),
            pl.col("a27_other_cid_field").sum().alias("cid_asso_morte_notif"),
            pl.col("caught_by_current_rule").sum().alias("caught_by_current_rule"),
            (~pl.col("caught_by_current_rule")).sum().alias("missed_by_current_rule"),
        ).sort("year")
        by_year.write_csv(OUT / "sih_reach_by_year.csv")
        print("\n=== SIH: A27 records by which field carries the code ===")
        print(f"{'year':>5}{'any':>7}{'princ':>7}{'DIAG_SECUN':>12}"
              f"{'DIAGSEC1-9':>12}{'other CID':>11}{'MISSED now':>12}")
        for r in by_year.iter_rows(named=True):
            print(f"{r['year'] if r['year'] is not None else -1:>5}{r['n_a27_any_field']:>7}"
                  f"{r['principal']:>7}{r['diag_secun']:>12}{r['diagsec_1_9']:>12}"
                  f"{r['cid_asso_morte_notif']:>11}{r['missed_by_current_rule']:>12}")
        missed = g.filter(~pl.col("caught_by_current_rule"))
        # Are the missed admissions milder or sicker than the caught ones? A
        # missed set that is *sicker* means the current rule truncates the
        # severity distribution from the top, which would bias every
        # severity-adjusted comparison, not merely the counts.
        cmp = g.with_columns(
            _num("DIAS_PERM").alias("los"),
            (_num("UTI_MES_TO").fill_null(0) > 0).alias("any_icu"),
            (code("MORTE") == "1").alias("died"),
        ).group_by("caught_by_current_rule").agg(
            pl.len().alias("n"),
            pl.col("los").median().alias("los_median"),
            pl.col("any_icu").mean().alias("frac_with_icu"),
            pl.col("died").mean().alias("in_hospital_cfr"),
        ).sort("caught_by_current_rule")
        cmp.write_csv(OUT / "sih_caught_vs_missed_severity.csv")
        print("\n  severity of the caught vs missed A27 admissions:")
        for r in cmp.iter_rows(named=True):
            print(f"     caught={str(r['caught_by_current_rule']):<6} n={r['n']:>6} "
                  f"LOS median={r['los_median']}  "
                  f"ICU={100 * (r['frac_with_icu'] or 0):.1f}%  "
                  f"CFR={100 * (r['in_hospital_cfr'] or 0):.2f}%")
        report["sih"] = {
            "a27_records_any_field": g.height,
            "caught_by_current_extraction_rule": int(g["caught_by_current_rule"].sum()),
            "missed_by_current_extraction_rule": missed.height,
            "missed_in_hospital_deaths": int(
                (missed["MORTE"].cast(pl.Utf8).str.strip_chars() == "1").sum()
            ),
            "diagsec_slots_present": sec_slots,
        }
        if missed.height:
            top = (
                missed.group_by(code("DIAG_PRINC").alias("principal_code"))
                .agg(pl.len().alias("n")).sort("n", descending=True).head(15)
            )
            top.write_csv(OUT / "sih_missed_principal_codes.csv")
            print("\n  principal diagnosis of the A27 admissions the current rule misses:")
            for r in top.iter_rows(named=True):
                print(f"     {r['principal_code']:<8} {r['n']:>6}")

    if SIM_FULL.exists():
        f = pl.read_parquet(SIM_FULL)
        code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731
        underlying = code("CAUSABAS").str.starts_with("A27").fill_null(False)
        chain = pl.lit(False)
        for c in ("LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII", "CAUSABAS_O"):
            if c in f.columns:
                chain = chain | code(c).str.contains("A27", literal=True).fill_null(False)
        g = f.with_columns(
            underlying.alias("a27_underlying"),
            chain.alias("a27_mentioned_in_chain"),
            code("DTOBITO").str.slice(4, 4).cast(pl.Int32, strict=False).alias("year"),
        )
        by_year = g.group_by("year").agg(
            pl.len().alias("n_a27_any"),
            pl.col("a27_underlying").sum().alias("underlying_cause"),
            pl.col("a27_mentioned_in_chain").sum().alias("mentioned_in_chain"),
            (~pl.col("a27_underlying")).sum().alias("mention_only"),
        ).sort("year")
        by_year.write_csv(OUT / "sim_reach_by_year.csv")
        print("\n=== SIM: A27 as underlying cause vs mentioned anywhere ===")
        print(f"{'year':>5}{'any':>7}{'underlying':>12}{'in chain':>10}{'MENTION ONLY':>14}")
        for r in by_year.iter_rows(named=True):
            print(f"{r['year'] if r['year'] is not None else -1:>5}{r['n_a27_any']:>7}"
                  f"{r['underlying_cause']:>12}{r['mentioned_in_chain']:>10}"
                  f"{r['mention_only']:>14}")
        # Where the mention sits matters and the two cases mean different
        # things. A27 on a certificate *line* is a physician writing
        # leptospirosis into the causal chain and the ICD selection rules
        # choosing a different underlying cause. A27 in ``CAUSABAS_O`` only is
        # the coder's *original* underlying cause having been moved by the
        # selection algorithm or by investigation. The first is contributing
        # mortality; the second is a recode.
        lines = pl.lit(False)
        for c in ("LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII"):
            if c in f.columns:
                lines = lines | code(c).str.contains("A27", literal=True).fill_null(False)
        g = g.with_columns(
            lines.alias("a27_on_certificate_line"),
            code("CAUSABAS_O").str.contains("A27", literal=True).fill_null(False)
            .alias("a27_as_original_underlying"),
        )
        mention_only = g.filter(~pl.col("a27_underlying"))
        where = mention_only.group_by(
            "a27_on_certificate_line", "a27_as_original_underlying"
        ).agg(pl.len().alias("n")).sort("a27_on_certificate_line",
                                        "a27_as_original_underlying")
        where.write_csv(OUT / "sim_mention_only_position.csv")
        print("\n  where the mention sits, for the deaths not counted at all:")
        for r in where.iter_rows(named=True):
            print(f"     on_certificate_line={str(r['a27_on_certificate_line']):<6}"
                  f" original_underlying={str(r['a27_as_original_underlying']):<6}"
                  f" n={r['n']}")
        report["sim"] = {
            "a27_records_any_position": g.height,
            "a27_underlying_cause": int(g["a27_underlying"].sum()),
            "a27_mention_only": mention_only.height,
            "a27_mention_only_on_certificate_line": int(
                mention_only["a27_on_certificate_line"].sum()
            ),
            "a27_mention_only_original_underlying_recoded_away": int(
                (mention_only["a27_as_original_underlying"]
                 & ~mention_only["a27_on_certificate_line"]).sum()
            ),
        }
        if mention_only.height:
            top = (
                mention_only.group_by(code("CAUSABAS").alias("underlying_code"))
                .agg(pl.len().alias("n")).sort("n", descending=True).head(15)
            )
            top.write_csv(OUT / "sim_mention_only_underlying_codes.csv")
            print("\n  underlying cause of deaths that only *mention* leptospirosis:")
            for r in top.iter_rows(named=True):
                print(f"     {r['underlying_code']:<8} {r['n']:>6}")

    (OUT / "reach_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'reach_summary.json'}")


# --------------------------------------------------------------------------

#: Organ-system complications of severe leptospirosis, as CID-10 prefixes. The
#: grouping is the clinical syndrome the literature names (Weil's disease:
#: jaundice + renal failure; SPHS: pulmonary haemorrhage), not an ICD chapter.
COMPLICATION_GROUPS: dict[str, tuple[str, ...]] = {
    "acute_renal_failure": ("N17", "N19", "N990"),
    "pulmonary_haemorrhage_or_respiratory_failure": ("J80", "J96", "R048", "J948"),
    "sepsis_or_shock": ("A40", "A41", "R57"),
    "jaundice_or_hepatic_failure": ("K72", "R17"),
    "coagulopathy_or_thrombocytopenia": ("D65", "D693", "D696"),
    "cardiac": ("I46", "I50", "I40"),
    "meningitis_or_cns": ("G00", "G03", "G93"),
    "dengue_or_arbovirus_codiagnosis": ("A90", "A91", "A92", "A928"),
    "covid19": ("B342", "U071", "U072"),
    "hiv": ("B20", "B21", "B22", "B23", "B24"),
}

_CHAIN_FIELDS = ("LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII", "CAUSABAS")


def stage_chain() -> None:
    """What the SIM multiple-cause chain says about *how* these people died.

    The chain fields are the single largest block of unused SIM information:
    six fields, 96% to 31% filled, holding the whole certified causal sequence.
    Because they concatenate four-character codes with no separator they are
    matched by substring, and every count below is therefore a count of death
    certificates *mentioning* a complication, which is what a phenotype
    description wants.
    """
    if not SIM_FULL.exists():
        print("chain: run the deep stage for SIM first")
        return
    f = pl.read_parquet(SIM_FULL)
    code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731
    under = code("CAUSABAS").str.starts_with("A27").fill_null(False)
    d = f.with_columns(under.alias("a27_underlying"))
    present = [c for c in _CHAIN_FIELDS if c in f.columns]
    exprs = []
    for group, prefixes in COMPLICATION_GROUPS.items():
        hit = pl.lit(False)
        for c in present:
            for p in prefixes:
                hit = hit | code(c).str.contains(p, literal=True).fill_null(False)
        exprs.append(hit.alias(group))
    d = d.with_columns(exprs).with_columns(
        code("DTOBITO").str.slice(4, 4).cast(pl.Int32, strict=False).alias("year")
    )
    a27 = d.filter(pl.col("a27_underlying"))
    rows = [{
        "complication": g,
        "prefixes": "/".join(COMPLICATION_GROUPS[g]),
        "n_deaths": int(a27[g].sum()),
        "pct_of_a27_underlying_deaths": round(100 * float(a27[g].mean()), 2),
    } for g in COMPLICATION_GROUPS]
    tab = pl.DataFrame(rows).sort("n_deaths", descending=True)
    tab.write_csv(OUT / "sim_complication_profile.csv")
    print(f"\n=== complications mentioned on the {a27.height} A27 "
          "underlying-cause death certificates ===")
    for r in tab.iter_rows(named=True):
        print(f"  {r['complication']:<46} {r['n_deaths']:>5} "
              f"{r['pct_of_a27_underlying_deaths']:>6.1f}%")

    # Does the phenotype move over time? If it does, it is a covariate; if it
    # is flat, it is a constant and cannot explain a temporal trend.
    by_year = a27.group_by("year").agg(
        pl.len().alias("n"),
        *[pl.col(g).mean().alias(g) for g in COMPLICATION_GROUPS],
    ).sort("year")
    by_year.write_csv(OUT / "sim_complication_by_year.csv")
    n_chain_fields_filled = a27.select(
        sum(
            (code(c).is_not_null() & (code(c) != "")).cast(pl.Int32)
            for c in present if c != "CAUSABAS"
        ).alias("k")
    )["k"]
    print(f"  certificate lines filled per death: median "
          f"{n_chain_fields_filled.median()}, max {n_chain_fields_filled.max()}")
    print(f"wrote {OUT / 'sim_complication_profile.csv'}")


STAGES = {
    "headers": lambda a: stage_headers(),
    "deep": lambda a: stage_deep(a.which),
    "profile": lambda a: stage_profile(),
    "severity": lambda a: stage_severity(),
    "reach": lambda a: stage_reach(),
    "chain": lambda a: stage_chain(),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--which", default="both", choices=["sim", "sih", "both"],
                    help="deep stage: which system to re-scan")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = list(STAGES) if "all" in args.stages else args.stages
    for name in wanted:
        print(f"\n{'=' * 74}\n== {name}\n{'=' * 74}")
        STAGES[name](args)


if __name__ == "__main__":
    main()
