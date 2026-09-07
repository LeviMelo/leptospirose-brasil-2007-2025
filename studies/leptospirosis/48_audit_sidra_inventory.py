"""Complete machine inventory of the SIDRA/IBGE data surface.

Written for the same reason as ``46_variable_catalogue.py``: a source document
that describes *files* but not *variables* lets a study leave half its evidence
untouched. ``docs/sources/SIDRA_COMPENDIUM.md`` is the curated authority on
which SIDRA tables this project recognises, but it is prose. Nothing in the
repository has ever turned it into a machine inventory, so nobody can answer
"what is catalogued, what is wired, what is extracted, what is actually used,
and what is dark" without reading 1,061 lines of Markdown by hand.

This script answers those five questions mechanically. It:

1. Parses every ``#### Tab NNNN`` block in the compendium into structured rows
   (id, group, survey, subject, period string, locality count, Tier, High Dim,
   per-variable rows with unit/type/DK, per-classification rows with axis and
   category count, and the curated Note/Cube-role text).
2. Crosses that against ``brepi/sources/sidra/registry.yaml`` (the 24 tables
   wired for extraction) and against the extraction provenance sidecars in
   ``data/interim/sidra/`` (the tables actually pulled), producing a three-way
   status per table.
3. Classifies TEMPORAL GRAIN. This is the point of the exercise. A covariate
   observed only in 2010 and 2022 is a structural constant with two anchors and
   must never be presented as a time-varying control. The classifier uses the
   *survey's* publication rhythm, not the shape of the period string: Tab 206
   reads ``1970-2010`` and is five decennial censuses, while Tab 5938 reads
   ``2002-2023`` and is twenty-two annual observations.
4. Profiles completeness for every table with cached values, on the population
   that matters: the 5,570 municipalities of the study's territorial mesh.
   Completeness is measured with the project's own sentinel taxonomy
   (``api.classify_value``), so an absolute zero is never counted as a missing
   value and a suppressed cell is never counted as a zero.
5. Lists what is already sitting in ``data/cache/sidra/values`` and therefore
   costs nothing to re-extract.
6. Mines the surface the compendium does not describe at all. Two layers:
   tables whose live ``/metadados`` is already cached but which appear in no
   curated document (fully verifiable - level, periods, variables, categories),
   and the complete IBGE aggregate catalogue cached at
   ``data/cache/sidra/catalog`` (9,298 aggregates across 70 surveys), scanned
   for leptospirosis-relevant survey families the compendium omits entirely.
   Catalogue rows carry only id and name, so their N6 reach is a candidate
   claim, not a verified one, and is labelled as such.

No network access. Everything is read from the repository and from the cache.

Outputs to ``data/results/audit_sidra/``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS
from brepi.sources.sidra.api import ValueStatus, classify_value

ROOT = Path(__file__).resolve().parents[2]
COMPENDIUM = ROOT / "docs" / "sources" / "SIDRA_COMPENDIUM.md"
REGISTRY = ROOT / "brepi" / "sources" / "sidra" / "registry.yaml"
CACHE = PATHS.cache / "sidra"
INTERIM = PATHS.interim / "sidra"
OUT = PATHS.results / "audit_sidra"

#: The project's own extraction ceiling (``brepi.sources.sidra.plan``). Used
#: here to grade how expensive an unused table would be to pull, in requests.
CELL_BUDGET = 45_000

#: The study window. A table that publishes nothing inside it cannot be a
#: covariate, whatever else it contains.
STUDY_FIRST_YEAR = 2007
STUDY_LAST_YEAR = 2025

#: Publication rhythm by survey. This, not the period string, decides whether a
#: table is a time-varying series or a stack of snapshots. Keys are matched as
#: case-insensitive substrings of the compendium's ``Res`` field.
SURVEY_RHYTHM: tuple[tuple[str, str], ...] = (
    ("censo demográfico", "decennial"),
    ("estimativas de população", "annual"),
    ("cadastro central de empresas", "annual"),
    ("cempre", "annual"),
    ("pib dos municípios", "annual"),
    ("contas nacionais", "annual"),
    ("produção agrícola municipal", "annual"),
    ("pesquisa da pecuária municipal", "annual"),
    ("pam", "annual"),
    ("ppm", "annual"),
    ("estatísticas do registro civil", "annual"),
    ("pesquisa nacional de saneamento básico", "episodic"),
    ("pnsb", "episodic"),
    ("fasfil", "episodic"),
)

#: One-line statement of what each thematic group lets an epidemiologist
#: measure. Curated here rather than in the compendium because it is a judgment
#: about *this* study, not about IBGE.
GROUP_EPI_VALUE: dict[int, str] = {
    1: (
        "Annual municipal economic capacity: formal establishments, formal payroll and "
        "GDP by sector. Gives a yearly measure of how much formal economy (and, via VAB "
        "of public administration/health, how much public service) each territory carries "
        "- the only annual municipal proxy for health-system substrate in the whole "
        "compendium."
    ),
    2: (
        "Population denominators and their internal structure: age, sex, race, "
        "urban/rural, density, land area. Lets incidence and case fatality be "
        "age-standardised instead of crude, which matters because leptospirosis lethality "
        "rises steeply with age and the age structure of Brazilian municipalities varies "
        "by a factor of three."
    ),
    3: (
        "Education and literacy of the adult population. The standard proxy for "
        "care-seeking delay: low schooling predicts late presentation, and late "
        "presentation is the mechanism by which severe leptospirosis becomes fatal."
    ),
    4: (
        "Water supply, sewage disposal and refuse destination at household level. The "
        "direct exposure axis for leptospirosis - open sewage, rudimentary pits, refuse "
        "thrown on open ground and unpiped water are the rodent-contact pathways - and "
        "the only axis observed at three census anchors (2000, 2010, 2022)."
    ),
    5: (
        "Household crowding, tenure and assets: persons per bedroom, rooms, bathrooms, "
        "tenure status. Crowding is an independent risk factor and tenure separates "
        "consolidated informal settlement from precarious occupation."
    ),
    6: (
        "Household income level and distribution, labour-force participation and "
        "occupational position. Household income is a far better deprivation measure than "
        "GDP per capita (which is dominated by agribusiness and mining rents), and "
        "occupational position identifies the outdoor and sanitation-worker exposure groups."
    ),
    7: (
        "Prevalence of permanent physical and cognitive disability. A comorbidity/frailty "
        "proxy for case fatality and a mobility barrier to care access."
    ),
    8: (
        "Fertility history and motherhood by age, education and race. Peripheral to "
        "leptospirosis; usable mainly as a demographic-transition index distinguishing "
        "young high-fertility peripheries from ageing consolidated municipalities."
    ),
    9: (
        "Indigenous and quilombola populations, both in and outside demarcated "
        "territories. The equity axis: these populations have both the highest exposure "
        "and the thinnest surveillance reach, which is precisely the study's thesis."
    ),
    10: (
        "Formal employment and payroll by CNAE 2.0 sector, annually. Isolates agricultural "
        "employment (Section A), sanitation and waste employment (Division 37/38) and "
        "human-health employment (Division 86) as an annual municipal series - a direct "
        "measure of both occupational exposure and health-service capacity."
    ),
    11: (
        "Municipal-government statements about sanitation service existence, network "
        "extent, and reported occurrence of sanitation-associated disease including "
        "leptospirosis itself. Administrative, independent of both the census and SINAN."
    ),
    12: (
        "Nonprofit and foundation sector by activity, including health and hospital "
        "nonprofits. A weak proxy for non-state health provision; low priority."
    ),
    13: (
        "Civil-registry births and deaths, annually, at municipality level, 2003-2024. "
        "Deaths carry PLACE OF OCCURRENCE (hospital / home / public street) - an annual "
        "municipal index of how far into the health system the population reaches, "
        "constructed independently of SINAN and of the hospitalisation share the study "
        "currently uses."
    ),
    14: (
        "Annual municipal crop area and livestock head counts, 1974-2024. The rural "
        "exposure axis: flooded rice and sugarcane fields are the classic occupational "
        "setting, and cattle/swine herds are the maintenance-host reservoir."
    ),
    15: (
        "Favelas and urban communities from the 2022 Census: settlement counts, resident "
        "population, density, and the sanitation profile inside them, plus a "
        "within-municipality inside/outside contrast on the same sewage classification. "
        "The urban leptospirosis gradient, nationally measurable for the first time."
    ),
}


# ---------------------------------------------------------------------------
# Compendium parsing
# ---------------------------------------------------------------------------

_GROUP_RE = re.compile(r"^###\s+GROUP\s+(\d+):\s*(.+?)\s*$")
_TABLE_RE = re.compile(r"^####\s+Tab\s+(\d+)\s*\|\s*(.+?)\s*$")
_FIELD_RE = re.compile(r"\*\*([A-Za-z][A-Za-z ]*?)\*\*:\s*")
_BULLET_RE = re.compile(r"^\s*\*\s+")
_ID_RE = re.compile(r"`(\d+)`")
_CLSF_RE = re.compile(r"`Clsf\s+(\d+)`\s*\(([^;)]*)")
_AXIS_RE = re.compile(r"Axis:\s*([A-Za-z_|]+)")
_CATS_RE = re.compile(r"Cats:\s*(\d+)")
_UNIT_RE = re.compile(r"U:\s*([^,;)]+)")
_TYPE_RE = re.compile(r"Type:\s*([a-z]+)")
_DK_RE = re.compile(r"DK:\s*([YN])")
_NAME_RE = re.compile(r"`\d+`\s*\(([^;)]+)")


def _split_fields(text: str) -> dict[str, str]:
    """Split a ``**A**: x | **B**: y`` meta line into a dict.

    Splitting on ``|`` is wrong: Tab 136's period field is literally
    ``1991 | 2000 | 2010``. The field boundaries are the ``**Key**:`` markers.
    """
    out: dict[str, str] = {}
    matches = list(_FIELD_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[m.end() : end].strip()
        value = re.sub(r"^\*\s*", "", value).strip().rstrip("|").strip()
        out[m.group(1).strip()] = value
    return out


def parse_compendium(path: Path) -> list[dict[str, Any]]:
    """One dict per catalogued table, in document order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[dict[str, Any]] = []
    group_id: int | None = None
    group_name = ""
    current: dict[str, Any] | None = None
    block: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal block, buffer
        if current is not None and block is not None:
            current["_blocks"][block] = "\n".join(buffer).strip()
        block, buffer = None, []

    for raw in lines:
        gm = _GROUP_RE.match(raw)
        if gm:
            flush()
            group_id, group_name = int(gm.group(1)), gm.group(2)
            continue
        tm = _TABLE_RE.match(raw)
        if tm:
            flush()
            current = {
                "table_id": tm.group(1),
                "title": tm.group(2),
                "group_id": group_id,
                "group_name": group_name,
                "_blocks": {},
                "_meta": {},
            }
            tables.append(current)
            continue
        if current is None:
            continue
        if _BULLET_RE.match(raw):
            fields = _split_fields(raw)
            keys = set(fields)
            # A meta line carries several keys at once; a block header carries
            # one and its content continues on following lines.
            if keys & {"Tier", "Res", "Per", "Locs", "Subj", "High Dim"}:
                flush()
                current["_meta"].update(fields)
                continue
            if len(keys) == 1:
                key = next(iter(keys))
                flush()
                block = key
                buffer = [fields[key]]
                continue
        if block is not None:
            buffer.append(raw)
    flush()

    for t in tables:
        _finalise(t)
    return tables


def _finalise(t: dict[str, Any]) -> None:
    meta = t.pop("_meta")
    blocks = t.pop("_blocks")
    t["tier"] = meta.get("Tier", "")
    t["survey"] = meta.get("Res", "")
    t["subject"] = meta.get("Subj", "")
    t["period_text"] = meta.get("Per", "")
    t["locs_text"] = meta.get("Locs", "")
    t["high_dim"] = meta.get("High Dim", "").lower().startswith("y")
    t["note"] = blocks.get("Note", "") or blocks.get("Note - restricted universe, and it matters.", "")
    t["cube_role"] = blocks.get("Cube role", "")
    t["variables"] = _parse_variables(blocks.get("Vars", ""))
    t["classifications"] = _parse_classifications(blocks.get("Clsfs", ""))
    t["vars_text"] = blocks.get("Vars", "")
    t["clsfs_text"] = blocks.get("Clsfs", "")
    for key, value in blocks.items():
        if key.startswith("Note"):
            t["note"] = value


def _parse_variables(text: str) -> list[dict[str, Any]]:
    """Variable rows from a ``**Vars**`` block.

    Lines take several shapes: a sub-bullet per variable, a comma list of ids
    sharing one parenthetical, and prose ("Same as Tab 1685 (`706`, ...)").
    All three yield ids; only the first yields a per-id name.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        ids = _ID_RE.findall(line)
        if not ids:
            continue
        unit = _UNIT_RE.search(line)
        vtype = _TYPE_RE.search(line)
        dk = _DK_RE.search(line)
        name = _NAME_RE.search(line)
        shared = len(ids) > 1
        for i, vid in enumerate(ids):
            if vid in seen:
                continue
            seen.add(vid)
            out.append(
                {
                    "variable_id": vid,
                    "name": (name.group(1).strip() if (name and i == 0 and not shared) else None),
                    "unit": unit.group(1).strip() if unit else None,
                    "type": vtype.group(1) if vtype else None,
                    "default_keep": dk.group(1) if dk else None,
                    "shared_line": shared,
                    "raw": line.strip().lstrip("* ").strip(),
                }
            )
    return out


def _parse_classifications(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _CLSF_RE.search(line)
        if not m:
            continue
        axis = _AXIS_RE.search(line)
        cats = _CATS_RE.search(line)
        out.append(
            {
                "classification_id": m.group(1),
                "name": m.group(2).strip(),
                "axis": axis.group(1) if axis else None,
                "n_categories": int(cats.group(1)) if cats else None,
                "raw": line.strip().lstrip("* ").strip(),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Temporal grain
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[\u2013\u2014-]\s*(19\d{2}|20\d{2})\b")


def survey_rhythm(survey: str) -> str:
    low = survey.lower()
    for token, rhythm in SURVEY_RHYTHM:
        if token in low:
            return rhythm
    return "unknown"


def parse_periods(period_text: str, rhythm: str) -> tuple[list[int], str]:
    """Years a table publishes, and how they were derived.

    A dash range means "every publication of this survey between A and B", so
    an annual survey fills it and a decennial one does not. ``1970-2010`` under
    the Censo therefore yields 1970/1980/1991/2000/2010, not 41 years.
    """
    ranges = _RANGE_RE.findall(period_text)
    explicit = [int(y) for y in _YEAR_RE.findall(period_text)]
    census_years = [1970, 1980, 1991, 2000, 2010, 2022]
    if ranges:
        years: set[int] = set()
        for lo, hi in ranges:
            lo_i, hi_i = int(lo), int(hi)
            if rhythm == "annual":
                years.update(range(lo_i, hi_i + 1))
            elif rhythm == "decennial":
                years.update(y for y in census_years if lo_i <= y <= hi_i)
            else:
                years.update({lo_i, hi_i})
        # A range plus stray years (e.g. exclusion prose) keeps the range.
        return sorted(years), "range"
    if explicit:
        return sorted(set(explicit)), "enumerated"
    return [], "unparsed"


def classify_grain(
    *,
    rhythm: str,
    years: list[int],
    municipal: bool,
    live_years: list[int] | None,
) -> tuple[str, list[int]]:
    """Return (grain label, years usable inside the study window)."""
    effective = live_years if live_years else years
    in_window = [y for y in effective if STUDY_FIRST_YEAR <= y <= STUDY_LAST_YEAR]
    unit = "municipality" if municipal else "sub-municipal or supra-municipal only"
    if not municipal:
        return f"{unit}; not joinable to the study mesh", in_window
    if rhythm == "annual" and len(in_window) >= 5:
        return f"municipality x year {min(in_window)}-{max(in_window)}", in_window
    if rhythm == "annual" and effective:
        return f"municipality x year, but only {len(in_window)} year(s) inside 2007-2025", in_window
    if rhythm == "episodic":
        waves = ", ".join(str(y) for y in effective)
        return f"municipality x survey wave ({waves or 'none'})", in_window
    if rhythm == "decennial":
        anchors = ", ".join(str(y) for y in effective)
        return f"municipality, census-year snapshot only ({anchors or 'none'})", in_window
    return f"municipality, period grain unresolved ({rhythm})", in_window


# ---------------------------------------------------------------------------
# Live cached metadata
# ---------------------------------------------------------------------------


def load_live_metadata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    meta_dir = CACHE / "metadados"
    per_dir = CACHE / "periodos"
    if not meta_dir.exists():
        return out
    for path in sorted(meta_dir.glob("*.json")):
        if path.name.endswith(".provenance.json"):
            continue
        tid = path.stem
        meta = json.loads(path.read_text(encoding="utf-8"))
        levels = meta.get("nivelTerritorial") or {}
        offered = sorted(
            set(levels.get("Administrativo") or [])
            | set(levels.get("Especial") or [])
            | set(levels.get("IBGE") or [])
        )
        periodicity = meta.get("periodicidade") or {}
        years: list[int] = []
        per_path = per_dir / f"{tid}.json"
        if per_path.exists():
            for entry in json.loads(per_path.read_text(encoding="utf-8")):
                m = _YEAR_RE.search(str(entry.get("id", "")))
                if m:
                    years.append(int(m.group(1)))
        out[tid] = {
            "live_name": meta.get("nome"),
            "live_survey": meta.get("pesquisa"),
            "live_subject": meta.get("assunto"),
            "live_levels": offered,
            "live_has_n6": "N6" in offered,
            "live_frequency": periodicity.get("frequencia"),
            "live_start": periodicity.get("inicio"),
            "live_end": periodicity.get("fim"),
            "live_years": sorted(set(years)),
            "live_n_variables": len(meta.get("variaveis") or []),
            "live_n_classifications": len(meta.get("classificacoes") or []),
            "live_variables": [
                {"id": str(v["id"]), "name": v.get("nome"), "unit": v.get("unidade")}
                for v in (meta.get("variaveis") or [])
            ],
        }
    return out


# ---------------------------------------------------------------------------
# Registry, extraction provenance, cached values
# ---------------------------------------------------------------------------


def load_registry() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return {str(k): v for k, v in (doc.get("tables") or {}).items()}


def load_extractions() -> dict[str, list[dict[str, Any]]]:
    """Tables actually pulled, from the provenance sidecar of each parquet."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not INTERIM.exists():
        return out
    for path in sorted(INTERIM.glob("*.provenance.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        sel = doc.get("selection") or {}
        out[str(sel.get("agregado"))].append(
            {
                "artefact": path.name.replace(".provenance.json", ""),
                "variables": sel.get("variables") or [],
                "periods": sel.get("periods") or [],
                "level": sel.get("level"),
                "n_localities": sel.get("n_localities"),
                "classifications": sel.get("classifications") or {},
                "rows_returned": doc.get("rows_returned"),
                "value_status": doc.get("value_status") or {},
            }
        )
    return out


def load_cached_values() -> dict[str, dict[str, Any]]:
    """What is already on disk under ``data/cache/sidra/values``."""
    out: dict[str, dict[str, Any]] = {}
    root = CACHE / "values"
    if not root.exists():
        return out
    for table_dir in sorted(root.iterdir()):
        if not table_dir.is_dir():
            continue
        payloads = [p for p in table_dir.glob("*.json") if not p.name.endswith(".provenance.json")]
        periods: set[str] = set()
        variables: set[str] = set()
        classifications: dict[str, set[str]] = defaultdict(set)
        levels: set[str] = set()
        total_bytes = 0
        for payload in payloads:
            total_bytes += payload.stat().st_size
            prov = payload.with_suffix(".json.provenance.json")
            if not prov.exists():
                continue
            q = (json.loads(prov.read_text(encoding="utf-8")).get("params") or {}).get(
                "normalised_query"
            ) or {}
            periods.update(str(p) for p in (q.get("periods") or []))
            variables.update(str(v) for v in (q.get("variables") or []))
            levels.add(str(q.get("level")))
            for clf, cats in (q.get("classifications") or {}).items():
                classifications[str(clf)].update(str(c) for c in cats)
        out[table_dir.name] = {
            "n_payloads": len(payloads),
            "bytes": total_bytes,
            "periods": sorted(periods),
            "variables": sorted(variables),
            "levels": sorted(levels),
            "classifications": {k: sorted(v) for k, v in classifications.items()},
            "paths": [str(p.relative_to(ROOT)) for p in payloads],
        }
    return out


# ---------------------------------------------------------------------------
# Completeness, on the study's municipal universe
# ---------------------------------------------------------------------------


def study_municipalities() -> set[str]:
    """The 5,570 municipal codes the study actually models."""
    panel = PATHS.panel / "lept_panel_municipality_month.parquet"
    codes = (
        pl.scan_parquet(panel)
        .select(pl.col("munic_code").cast(pl.Utf8).unique())
        .collect()["munic_code"]
        .to_list()
    )
    return {c for c in codes if c}


def profile_completeness(
    cached: dict[str, dict[str, Any]], universe: set[str]
) -> list[dict[str, Any]]:
    """Per (table, variable), how many of the study's municipalities return a number.

    POPULATION PROFILED: the 5,570 municipalities of the study's 2022 mesh, as
    they appear in ``lept_panel_municipality_month.parquet``. This is not the
    same as the locality count SIDRA offers - tables built on the 5,565 or 5,507
    lattice are structurally short of the modern mesh, and that shortfall shows
    up here as missing municipalities rather than as missing values.
    """
    rows: list[dict[str, Any]] = []
    root = CACHE / "values"
    for table_id, info in cached.items():
        per_var: dict[str, dict[str, Any]] = {}
        for rel in info["paths"]:
            payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            for var_block in payload:
                vid = str(var_block.get("id"))
                acc = per_var.setdefault(
                    vid,
                    {
                        "variable_name": var_block.get("variavel"),
                        "unit": var_block.get("unidade"),
                        "status": Counter(),
                        "municipalities": set(),
                        "municipalities_with_number": set(),
                        "periods": set(),
                        "off_mesh": set(),
                    },
                )
                for result in var_block.get("resultados") or []:
                    for series in result.get("series") or []:
                        loc = series.get("localidade") or {}
                        if (loc.get("nivel") or {}).get("id") != "N6":
                            continue
                        code = str(loc.get("id"))
                        if code in universe:
                            acc["municipalities"].add(code)
                        else:
                            acc["off_mesh"].add(code)
                        for period, raw in (series.get("serie") or {}).items():
                            acc["periods"].add(str(period))
                            _, numeric, status = classify_value(raw)
                            acc["status"][status.value] += 1
                            if numeric is not None and code in universe:
                                acc["municipalities_with_number"].add(code)
        for vid, acc in sorted(per_var.items()):
            total = sum(acc["status"].values())
            rows.append(
                {
                    "table_id": table_id,
                    "variable_id": vid,
                    "variable_name": acc["variable_name"],
                    "unit": acc["unit"],
                    "n_periods_cached": len(acc["periods"]),
                    "periods_cached": "|".join(sorted(acc["periods"])),
                    "n_cells": total,
                    "n_municipalities_on_mesh": len(acc["municipalities"]),
                    "pct_mesh_covered": round(100 * len(acc["municipalities"]) / len(universe), 2),
                    "n_municipalities_with_number": len(acc["municipalities_with_number"]),
                    "pct_mesh_with_number": round(
                        100 * len(acc["municipalities_with_number"]) / len(universe), 2
                    ),
                    "n_off_mesh_localities": len(acc["off_mesh"]),
                    **{
                        f"cells_{s.value.lower()}": acc["status"].get(s.value, 0)
                        for s in ValueStatus
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# The surface the compendium does not describe
# ---------------------------------------------------------------------------

#: Survey families in the live IBGE catalogue that carry municipal-level
#: leptospirosis-relevant content and appear in the compendium not at all, or
#: only as a token handful. ``compendium_tables`` is filled in at runtime.
CATALOGUE_FAMILIES: dict[str, str] = {
    "Pesquisa de Informações Básicas Municipais": (
        "MUNIC. Surveys every municipal government every year or two. Carries municipal "
        "health-service capacity (dialysis, ICU beds, emergency service, Family Health "
        "Programme coverage), whether the municipality operates epidemiological "
        "surveillance and endemic-disease control at all, whether it has favelas/cortiços, "
        "and its flood and landslide occurrence and disaster-risk instruments. Absent from "
        "the compendium in its entirety."
    ),
    "Censo Agropecuário": (
        "Farm-level agriculture and livestock at municipality level, 2006 and 2017: herd "
        "composition, pasture area, irrigated area, and PERSONS OCCUPIED in agricultural "
        "establishments by family/non-family farming. PAM/PPM give area and head counts "
        "but no exposed workforce; this is where the exposed workforce is. Absent from the "
        "compendium in its entirety."
    ),
    "Objetivos de Desenvolvimento Sustentável": (
        "Pre-computed SDG indicators, including SDG 6.2.1 safely managed sanitation and "
        "SDG 11.1.1 share of urban population in inadequate housing. Absent from the "
        "compendium."
    ),
    "Áreas Urbanizadas": (
        "Mapped urbanised area, vacant lots and subcategories per municipality. A built-"
        "environment denominator that separates urban extent from administrative area. "
        "Absent from the compendium."
    ),
    "Contagem da População ": (
        "The 1996 and 2007 population counts. 2007 is the first year of the study window "
        "and the one year SIDRA's estimate series 6579 omits. Absent from the compendium."
    ),
    "Projeção da População": (
        "Official population projections with life expectancy and infant mortality. UF "
        "level; useful as a mortality-context benchmark, not as a municipal covariate."
    ),
    "Pesquisa de Assistência Médico-Sanitária": (
        "Health establishments and inpatient beds. Annual 1976-2005 but NOT offered at N6 "
        "and ends two years before the study window."
    ),
}

#: Keywords that make a catalogue row worth a metadata call, by theme.
CATALOGUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "health_system_capacity": (
        "saude", "leito", "nefrologia", "uti", "emergencia", "hospital", "vacinacao",
        "agente comunitario", "saude da familia",
    ),
    "surveillance_capacity": ("vigilancia", "endemia", "epidemiolog", "busca ativa"),
    "sanitation": ("esgoto", "agua", "residuo", "lixo", "drenagem", "pluvia", "catador"),
    "flood_disaster": (
        "enchente", "inundacao", "enxurrada", "alagamento", "deslizamento",
        "escorregamento", "desastre", "risco",
    ),
    "informal_settlement": ("favela", "mocambo", "palafita", "cortico", "aglomerado subnormal",
                            "assentamento", "loteamento irregular"),
    "agricultural_exposure": ("bovino", "suino", "pessoal ocupado", "trabalhador", "arroz",
                              "cana", "pastagem"),
    "vital_statistics": ("obito", "nascido vivo"),
}


def _fold(text: str) -> str:
    import unicodedata

    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def load_catalogue() -> list[dict[str, Any]]:
    """The complete live IBGE aggregate catalogue, as cached."""
    root = CACHE / "catalog"
    if not root.exists():
        return []
    payloads = [p for p in root.glob("*.json") if not p.name.endswith(".provenance.json")]
    if not payloads:
        return []
    doc = json.loads(payloads[0].read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for survey in doc:
        for agg in survey.get("agregados") or []:
            rows.append(
                {
                    "table_id": str(agg["id"]),
                    "table_name": agg.get("nome", ""),
                    "survey_id": survey.get("id"),
                    "survey_name": survey.get("nome", ""),
                }
            )
    return rows


def tag_catalogue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        folded = _fold(r["table_name"])
        themes = [
            theme
            for theme, kws in CATALOGUE_KEYWORDS.items()
            if any(k in folded for k in kws)
        ]
        out.append({**r, "themes": "|".join(themes), "n_themes": len(themes)})
    return out


def estimate_extraction_cost(
    *, n_variables: int, cat_product: int, n_localities: int, n_periods: int
) -> tuple[int, int, str]:
    """Cells, requests at the project's budget, and an effort grade.

    ``cat_product`` is the product of category counts across classifications -
    the cube width that :func:`brepi.sources.sidra.plan.plan_requests` refuses
    to split. A table whose single-locality single-period slice already exceeds
    the budget cannot be pulled as published at all.
    """
    width = max(1, n_variables) * max(1, cat_product)
    cells = width * max(1, n_localities) * max(1, n_periods)
    requests = -(-cells // CELL_BUDGET)
    if width > CELL_BUDGET:
        grade = "expensive"
    elif cells <= 500_000:
        grade = "cheap_extract"
    elif cells <= 20_000_000:
        grade = "moderate"
    else:
        grade = "expensive"
    return cells, requests, grade


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


#: Judgment calls that no rule derives. Each carries the evidence for it.
UNUSABLE_OVERRIDES: dict[str, str] = {
    "9723": (
        "Rows are official quilombola TERRITORIES, not municipalities (compendium: "
        "'Clsfs: None (Aggregated by official Quilombola Territories names)'). There is no "
        "municipal universe to join to; use 9578/10089 instead."
    ),
    "21": (
        "Closed PIB series 1999-2012 on the 5,565-municipality lattice. Tab 5938 covers "
        "2002-2023 on the 5,570 lattice with the same variables plus the "
        "services-excluding-public-administration split. Keeping both invites a "
        "double-source join with no gain."
    ),
}

#: Caveats that qualify a table without disqualifying it. These do not change
#: ``status``; they are attached to ``status_reason`` so that no one uses the
#: table without meeting the condition.
CAVEAT_OVERRIDES: dict[str, str] = {
    "5457": (
        "USABLE ONLY AS A REPLACEMENT FOR 1612, never alongside it: it is a superset of "
        "1612 on a DIFFERENT category id space (40102 vs 2692 for rice, per the "
        "compendium's own Note), and the two must never be crosswalked by crop name. Its "
        "gain over 1612 is the permanent crops - coffee, banana, orange, rubber."
    ),
    "10344": (
        "RESTRICTED UNIVERSE: only census tracts selected for the Pesquisa Urbanistica do "
        "Entorno dos Domicilios, and only municipalities that have settlements. Its totals "
        "are not municipal totals and must never be presented as such."
    ),
    "9887": (
        "Settlement-grain table aggregated to N6. A municipality with no settlement is "
        "ABSENT, not zero; the 656 rows are not a municipal universe."
    ),
    "6579": (
        "Wired in the registry as the population denominator spine and described as such "
        "in the compendium's Cube role, but never actually pulled: the panel's population "
        "column comes from DATASUS POPSVS via brepi.sources.datasus.population, not from "
        "SIDRA. The SIDRA series remains available as an independent cross-check on the "
        "denominator, which is a different and useful role."
    ),
}


def usability_verdict(
    *,
    table_id: str,
    municipal: bool,
    rhythm: str,
    years: list[int],
    in_window: list[int],
    lattice: str,
    cube_width: int,
) -> tuple[str, str]:
    """(status, reason). ``status`` is one of used/available_unused/unusable."""
    if table_id in UNUSABLE_OVERRIDES:
        return "unusable", UNUSABLE_OVERRIDES[table_id]
    if not municipal:
        return "unusable", f"does not reach N6; offered lattice is '{lattice}'"
    if cube_width > CELL_BUDGET:
        return (
            "unusable",
            f"a single locality-period slice is {cube_width:,} cells, over the "
            f"{CELL_BUDGET:,}-cell request budget; extractable only with the "
            f"classification restricted to a coarse level",
        )
    if years and max(years) < STUDY_FIRST_YEAR:
        if rhythm == "decennial":
            return (
                "available_unused",
                f"pre-window census anchor(s) {years} on the {lattice} lattice; usable only "
                f"after a territorial crosswalk to the 2022 mesh",
            )
        return (
            "unusable",
            f"series ends {max(years)}, before the {STUDY_FIRST_YEAR} start of the study "
            f"window, and the survey is not repeated after it",
        )
    if not in_window and rhythm != "decennial":
        return "unusable", f"publishes nothing inside {STUDY_FIRST_YEAR}-{STUDY_LAST_YEAR}"
    return "available_unused", ""


def municipal_reach(table: dict[str, Any], live: dict[str, Any] | None) -> tuple[bool, str]:
    """Does the table reach N6, and on what lattice?"""
    if live is not None:
        lattice = table["locs_text"] or "unstated"
        return live["live_has_n6"], lattice
    locs = table["locs_text"]
    counts = [int(m) for m in re.findall(r"\b(\d{4})\b", locs)]
    if counts:
        return max(counts) >= 4000, locs
    if "municipalities" in locs.lower():
        return True, locs
    return False, locs or "unstated"


def build_inventory() -> dict[str, Any]:
    tables = parse_compendium(COMPENDIUM)
    live = load_live_metadata()
    registry = load_registry()
    extractions = load_extractions()
    cached = load_cached_values()
    universe = study_municipalities()

    rows: list[dict[str, Any]] = []
    var_rows: list[dict[str, Any]] = []
    clf_rows: list[dict[str, Any]] = []

    for t in tables:
        tid = t["table_id"]
        lv = live.get(tid)
        rhythm = survey_rhythm(t["survey"])
        years, how = parse_periods(t["period_text"], rhythm)
        municipal, lattice = municipal_reach(t, lv)
        live_years = lv["live_years"] if lv and lv.get("live_years") else None
        grain, in_window = classify_grain(
            rhythm=rhythm, years=years, municipal=municipal, live_years=live_years
        )
        extracted = extractions.get(tid, [])
        pulled_vars = sorted({v for e in extracted for v in e["variables"]})
        catalogued_vars = [v["variable_id"] for v in t["variables"]]

        # Cube width: variables the compendium marks Default-Keep, times the
        # product of category counts. This is what the planner cannot split.
        keep_vars = [v for v in t["variables"] if v["default_keep"] != "N"] or t["variables"]
        cat_product = 1
        for c in t["classifications"]:
            cat_product *= max(1, c["n_categories"] or 1)
        n_loc = 5570
        m = re.search(r"\b(\d{4})\b", t["locs_text"])
        if m:
            n_loc = int(m.group(1))
        n_per = max(1, len(in_window) or len(years))
        cells_full, requests_full, _ = estimate_extraction_cost(
            n_variables=len(keep_vars),
            cat_product=cat_product,
            n_localities=n_loc,
            n_periods=n_per,
        )
        # The realistic pull is not the full cross. An analyst takes ONE
        # classification at full detail and the Total member of the others,
        # which is what makes 2683's place-of-death usable at all.
        widest = max([c["n_categories"] or 1 for c in t["classifications"]] or [1])
        cells, requests, effort = estimate_extraction_cost(
            n_variables=len(keep_vars),
            cat_product=widest,
            n_localities=n_loc,
            n_periods=n_per,
        )
        cube_width = max(1, len(keep_vars)) * cat_product

        status, reason = usability_verdict(
            table_id=tid,
            municipal=municipal,
            rhythm=rhythm,
            years=live_years or years,
            in_window=in_window,
            lattice=lattice,
            cube_width=cube_width,
        )
        if extracted:
            status, reason = "used", f"pulled as {', '.join(e['artefact'] for e in extracted)}"
            effort = "already_extracted"
        if tid in CAVEAT_OVERRIDES:
            reason = f"{CAVEAT_OVERRIDES[tid]}{(' | ' + reason) if reason else ''}"

        rows.append(
            {
                "table_id": tid,
                "group_id": t["group_id"],
                "group_name": t["group_name"],
                "title": t["title"],
                "tier": t["tier"],
                "survey": t["survey"],
                "subject": t["subject"],
                "period_text": t["period_text"],
                "survey_rhythm": rhythm,
                "years_parsed": "|".join(str(y) for y in years),
                "years_parsed_how": how,
                "live_metadata_cached": lv is not None,
                "live_frequency": lv["live_frequency"] if lv else None,
                "live_years": "|".join(str(y) for y in (live_years or [])),
                "reaches_n6": municipal,
                "locality_lattice": lattice,
                "temporal_grain": grain,
                "n_years_in_study_window": len(in_window),
                "years_in_study_window": "|".join(str(y) for y in in_window),
                "annual_municipal_series": bool(
                    municipal and rhythm == "annual" and len(in_window) >= 5
                ),
                "high_dim": t["high_dim"],
                "n_variables_catalogued": len(t["variables"]),
                "n_classifications": len(t["classifications"]),
                "max_classification_cats": max(
                    [c["n_categories"] or 0 for c in t["classifications"]] or [0]
                ),
                "wired_in_registry": tid in registry,
                "registry_role": (registry.get(tid) or {}).get("role"),
                "extracted": bool(extracted),
                "extraction_artefacts": "|".join(e["artefact"] for e in extracted),
                "variables_catalogued": "|".join(catalogued_vars),
                "variables_pulled": "|".join(pulled_vars),
                "variables_catalogued_but_never_pulled": "|".join(
                    v for v in catalogued_vars if v not in set(pulled_vars)
                ),
                "values_cached": tid in cached,
                "cached_payloads": (cached.get(tid) or {}).get("n_payloads", 0),
                "cached_bytes": (cached.get(tid) or {}).get("bytes", 0),
                "status": status,
                "status_reason": reason,
                "cube_width_cells": cube_width,
                "estimated_cells_full_cross": cells_full,
                "estimated_requests_full_cross": requests_full,
                "estimated_cells_one_axis_pull": cells,
                "estimated_requests_one_axis_pull": requests,
                "effort": effort,
                "note": t["note"],
                "cube_role": t["cube_role"],
            }
        )
        for v in t["variables"]:
            var_rows.append(
                {
                    "table_id": tid,
                    "group_id": t["group_id"],
                    "temporal_grain": grain,
                    "annual_municipal_series": bool(
                        municipal and rhythm == "annual" and len(in_window) >= 5
                    ),
                    "wired_in_registry": tid in registry,
                    "pulled": v["variable_id"] in set(pulled_vars),
                    **v,
                }
            )
        for c in t["classifications"]:
            clf_rows.append({"table_id": tid, "group_id": t["group_id"], **c})

    # ---- layer 2: tables with live metadata cached but catalogued nowhere ---
    catalogued_ids = {r["table_id"] for r in rows}
    uncatalogued: list[dict[str, Any]] = []
    for tid, lv in sorted(live.items(), key=lambda kv: int(kv[0])):
        if tid in catalogued_ids:
            continue
        lyears = lv["live_years"] or []
        lw = [y for y in lyears if STUDY_FIRST_YEAR <= y <= STUDY_LAST_YEAR]
        uncatalogued.append(
            {
                "table_id": tid,
                "name": lv["live_name"],
                "survey": lv["live_survey"],
                "subject": lv["live_subject"],
                "reaches_n6": lv["live_has_n6"],
                "levels": "|".join(lv["live_levels"]),
                "frequency": lv["live_frequency"],
                "period_start": lv["live_start"],
                "period_end": lv["live_end"],
                "years": "|".join(str(y) for y in lyears),
                "n_years_in_study_window": len(lw),
                "n_variables": lv["live_n_variables"],
                "n_classifications": lv["live_n_classifications"],
                "variables": "|".join(f"{v['id']}:{v['name']}" for v in lv["live_variables"]),
                "in_registry": tid in registry,
                "extracted": tid in extractions,
                "values_cached": tid in cached,
                "evidence": "live /metadados cached at data/cache/sidra/metadados/"
                f"{tid}.json - level, periods, variables and categories are verified",
            }
        )

    # ---- layer 3: the full IBGE catalogue, name-level only ------------------
    catalogue = tag_catalogue(load_catalogue())
    known = catalogued_ids | set(live)
    for row in catalogue:
        row["in_compendium"] = row["table_id"] in catalogued_ids
        row["metadata_cached"] = row["table_id"] in live
        row["known_to_project"] = row["table_id"] in known

    return {
        "tables": rows,
        "variables": var_rows,
        "classifications": clf_rows,
        "registry": registry,
        "extractions": extractions,
        "cached": cached,
        "live": live,
        "universe": universe,
        "parsed": tables,
        "uncatalogued": uncatalogued,
        "catalogue": catalogue,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = build_inventory()
    tables = pl.DataFrame(inv["tables"], strict=False)
    variables = pl.DataFrame(inv["variables"], strict=False)
    classifications = pl.DataFrame(inv["classifications"], strict=False)

    tables.write_csv(OUT / "compendium_tables.csv")
    variables.write_csv(OUT / "compendium_variables.csv")
    classifications.write_csv(OUT / "compendium_classifications.csv")

    # ---- gap list, grouped by the compendium's 15 thematic groups -----------
    gap = tables.filter(~pl.col("wired_in_registry"))
    gap.write_csv(OUT / "gap_catalogued_not_wired.csv")
    unextracted = tables.filter(~pl.col("extracted"))
    unextracted.write_csv(OUT / "gap_catalogued_not_extracted.csv")

    gap_by_group: list[dict[str, Any]] = []
    for gid in sorted({r["group_id"] for r in inv["tables"] if r["group_id"]}):
        sub = tables.filter(pl.col("group_id") == gid)
        not_wired = sub.filter(~pl.col("wired_in_registry"))
        not_extracted = sub.filter(~pl.col("extracted"))
        annual = sub.filter(pl.col("annual_municipal_series"))
        gap_by_group.append(
            {
                "group_id": gid,
                "group_name": sub["group_name"][0],
                "epidemiological_value": GROUP_EPI_VALUE.get(gid, ""),
                "n_tables_catalogued": sub.height,
                "n_wired": sub.height - not_wired.height,
                "n_not_wired": not_wired.height,
                "n_extracted": sub.height - not_extracted.height,
                "n_never_extracted": not_extracted.height,
                "tables_not_wired": "|".join(sorted(not_wired["table_id"].to_list(), key=int)),
                "tables_never_extracted": "|".join(
                    sorted(not_extracted["table_id"].to_list(), key=int)
                ),
                "annual_municipal_tables": "|".join(sorted(annual["table_id"].to_list(), key=int)),
                "n_variables_catalogued": int(sub["n_variables_catalogued"].sum()),
            }
        )
    pl.DataFrame(gap_by_group, strict=False).write_csv(OUT / "gap_by_group.csv")

    # ---- the headline question: annual municipal series --------------------
    annual = tables.filter(pl.col("annual_municipal_series")).sort(
        ["extracted", "group_id"], descending=[True, False]
    )
    annual.select(
        "table_id",
        "group_id",
        "group_name",
        "title",
        "survey",
        "temporal_grain",
        "n_years_in_study_window",
        "years_in_study_window",
        "reaches_n6",
        "locality_lattice",
        "wired_in_registry",
        "extracted",
        "variables_catalogued",
        "variables_pulled",
        "variables_catalogued_but_never_pulled",
        "high_dim",
        "max_classification_cats",
    ).write_csv(OUT / "annual_municipal_series.csv")

    # ---- cached extracts: what is free -------------------------------------
    cached_rows = [
        {
            "table_id": tid,
            "n_payloads": info["n_payloads"],
            "bytes": info["bytes"],
            "levels": "|".join(info["levels"]),
            "periods": "|".join(info["periods"]),
            "variables": "|".join(info["variables"]),
            "classifications": json.dumps(info["classifications"], ensure_ascii=False),
            "in_compendium": tid in {r["table_id"] for r in inv["tables"]},
            "in_registry": tid in inv["registry"],
            "landed_as_parquet": tid in inv["extractions"],
        }
        for tid, info in sorted(inv["cached"].items(), key=lambda kv: int(kv[0]))
    ]
    cached_df = pl.DataFrame(cached_rows, strict=False)
    cached_df.write_csv(OUT / "cached_extracts.csv")
    # Values on disk that were never turned into a parquet: free evidence.
    cached_df.filter(~pl.col("landed_as_parquet")).write_csv(
        OUT / "cached_but_never_landed.csv"
    )

    # ---- completeness, on the study's 5,570-municipality mesh --------------
    completeness = profile_completeness(inv["cached"], inv["universe"])
    pl.DataFrame(completeness, strict=False).sort("table_id").write_csv(
        OUT / "cached_completeness_on_study_mesh.csv"
    )

    # ---- the surface no curated document describes --------------------------
    uncat = pl.DataFrame(inv["uncatalogued"], strict=False)
    uncat.write_csv(OUT / "dark_uncatalogued_metadata_cached.csv")

    catalogue = pl.DataFrame(inv["catalogue"], strict=False)
    catalogue.write_csv(OUT / "ibge_catalogue_full.csv")
    relevant = catalogue.filter(
        (pl.col("n_themes") > 0) & (~pl.col("known_to_project"))
    ).sort(["survey_name", "table_id"])
    relevant.write_csv(OUT / "ibge_catalogue_relevant_unknown.csv")

    by_survey = (
        catalogue.group_by("survey_id", "survey_name")
        .agg(
            pl.len().alias("n_aggregates"),
            pl.col("in_compendium").sum().alias("n_in_compendium"),
            pl.col("metadata_cached").sum().alias("n_metadata_cached"),
            (pl.col("n_themes") > 0).sum().alias("n_theme_relevant"),
        )
        .with_columns(
            pl.col("survey_name").str.strip_chars().alias("_key"),
        )
        .sort("n_theme_relevant", descending=True)
    )
    by_survey.write_csv(OUT / "ibge_catalogue_by_survey.csv")

    families = [
        {
            "survey_name": name.strip(),
            "why_it_matters": why,
            "n_aggregates": int(
                catalogue.filter(pl.col("survey_name").str.strip_chars() == name.strip()).height
            ),
            "n_in_compendium": int(
                catalogue.filter(
                    (pl.col("survey_name").str.strip_chars() == name.strip())
                    & pl.col("in_compendium")
                ).height
            ),
            "n_theme_relevant": int(
                catalogue.filter(
                    (pl.col("survey_name").str.strip_chars() == name.strip())
                    & (pl.col("n_themes") > 0)
                ).height
            ),
        }
        for name, why in CATALOGUE_FAMILIES.items()
    ]
    pl.DataFrame(families, strict=False).write_csv(OUT / "omitted_survey_families.csv")

    # ---- drift: three populations that should agree and do not -------------
    compendium_ids = {r["table_id"] for r in inv["tables"]}
    registry_ids = set(inv["registry"])
    extracted_ids = set(inv["extractions"])
    cached_meta_ids = set(inv["live"])
    drift = {
        "n_tables_catalogued": len(compendium_ids),
        "n_tables_wired_in_registry": len(registry_ids),
        "n_tables_ever_extracted": len(extracted_ids),
        "registry_not_in_compendium": sorted(registry_ids - compendium_ids, key=int),
        "compendium_not_in_registry": sorted(compendium_ids - registry_ids, key=int),
        "extracted_but_not_in_registry": sorted(extracted_ids - registry_ids - {"None"}, key=int),
        "extracted_but_not_in_compendium": sorted(
            extracted_ids - compendium_ids - {"None"}, key=int
        ),
        "live_metadata_cached_but_uncatalogued": sorted(
            cached_meta_ids - compendium_ids, key=int
        ),
        "catalogue_aggregates_total": len(inv["catalogue"]),
        "catalogue_aggregates_known_to_project": sum(
            1 for r in inv["catalogue"] if r["known_to_project"]
        ),
    }
    (OUT / "registry_drift.json").write_text(
        json.dumps(drift, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    from datetime import datetime, timezone

    summary = {
        "generated_by": "studies/leptospirosis/48_audit_sidra_inventory.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_snapshot_caveat": (
            "data/cache/sidra is written by any extraction run. The cached-values "
            "inventory below is a point-in-time reading at generated_at and will grow if "
            "another job pulls while this one reads."
        ),
        "compendium": str(COMPENDIUM.relative_to(ROOT)),
        "registry": str(REGISTRY.relative_to(ROOT)),
        "study_window": [STUDY_FIRST_YEAR, STUDY_LAST_YEAR],
        "study_mesh_municipalities": len(inv["universe"]),
        "counts": {
            "groups": len({r["group_id"] for r in inv["tables"] if r["group_id"]}),
            "tables_catalogued": tables.height,
            "variables_catalogued": variables.height,
            "distinct_variable_ids": variables["variable_id"].n_unique(),
            "classification_rows": classifications.height,
            "distinct_classification_ids": classifications["classification_id"].n_unique(),
            "tables_wired": int(tables["wired_in_registry"].sum()),
            "tables_not_wired": int((~tables["wired_in_registry"]).sum()),
            "tables_extracted": int(tables["extracted"].sum()),
            "tables_never_extracted": int((~tables["extracted"]).sum()),
            "tables_reaching_n6": int(tables["reaches_n6"].sum()),
            "annual_municipal_series": int(tables["annual_municipal_series"].sum()),
            "annual_municipal_series_extracted": int(
                tables.filter(pl.col("annual_municipal_series"))["extracted"].sum()
            ),
            "variables_pulled": int(variables["pulled"].sum()),
            "tables_with_cached_values": len(inv["cached"]),
            "status_used": int((tables["status"] == "used").sum()),
            "status_available_unused": int((tables["status"] == "available_unused").sum()),
            "status_unusable": int((tables["status"] == "unusable").sum()),
            "uncatalogued_with_verified_metadata": uncat.height,
            "uncatalogued_reaching_n6": int(uncat["reaches_n6"].sum()) if uncat.height else 0,
            "catalogue_aggregates": catalogue.height,
            "catalogue_theme_relevant_unknown": relevant.height,
        },
        "omitted_survey_families": families,
        "drift": drift,
        "gap_by_group": gap_by_group,
        "annual_municipal_series": annual.select(
            "table_id", "group_id", "survey", "temporal_grain", "extracted", "wired_in_registry"
        ).to_dicts(),
    }
    (OUT / "audit_sidra.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # ---- console report ----------------------------------------------------
    print(f"study mesh: {len(inv['universe'])} municipalities")
    print(
        f"catalogued: {tables.height} tables / {variables.height} variable rows "
        f"({variables['variable_id'].n_unique()} distinct ids) in "
        f"{len({r['group_id'] for r in inv['tables'] if r['group_id']})} groups"
    )
    print(
        f"wired: {int(tables['wired_in_registry'].sum())}   "
        f"extracted: {int(tables['extracted'].sum())}   "
        f"never extracted: {int((~tables['extracted']).sum())}"
    )
    print(f"variables ever pulled: {int(variables['pulled'].sum())} of {variables.height}")
    print()
    print("ANNUAL MUNICIPAL SERIES (municipality x year inside 2007-2025):")
    for r in annual.iter_rows(named=True):
        mark = "USED  " if r["extracted"] else "DARK  "
        print(
            f"  {mark} Tab {r['table_id']:>5}  G{r['group_id']:<2} "
            f"{r['n_years_in_study_window']:>2}y  {r['survey'][:34]:<34} "
            f"{r['title'][:52]}"
        )
    print()
    print("GAP BY GROUP (catalogued vs wired vs extracted):")
    for g in gap_by_group:
        print(
            f"  G{g['group_id']:<2} {g['group_name'][:44]:<44} "
            f"tables {g['n_tables_catalogued']:>2}  wired {g['n_wired']:>2}  "
            f"extracted {g['n_extracted']:>2}  vars {g['n_variables_catalogued']:>3}"
        )
    print()
    print("STATUS:")
    for row in tables.group_by("status").len().sort("len", descending=True).iter_rows(named=True):
        print(f"  {row['status']:<20} {row['len']:>3}")
    print()
    print("UNUSABLE, with reason:")
    for r in tables.filter(pl.col("status") == "unusable").sort("table_id").iter_rows(named=True):
        print(f"  Tab {r['table_id']:>5}  {r['status_reason'][:110]}")
    print()
    print("DARK: live metadata cached, catalogued in no document:")
    for r in inv["uncatalogued"]:
        print(
            f"  Tab {r['table_id']:>5}  N6={str(r['reaches_n6']):<5} "
            f"{str(r['period_start'])}-{str(r['period_end'])}  "
            f"{(r['survey'] or '')[:38]:<38} {(r['name'] or '')[:58]}"
        )
    print()
    print("DARK: whole survey families in the live IBGE catalogue:")
    for f in families:
        print(
            f"  {f['survey_name'][:46]:<46} aggregates {f['n_aggregates']:>5}  "
            f"in compendium {f['n_in_compendium']:>3}  theme-relevant {f['n_theme_relevant']:>4}"
        )
    print()
    print("CACHED VALUES NEVER LANDED AS PARQUET (free evidence sitting on disk):")
    for r in cached_rows:
        if not r["landed_as_parquet"]:
            print(
                f"  Tab {r['table_id']:>5}  {r['n_payloads']:>2} payloads  "
                f"{r['bytes']/1e6:>6.2f} MB  {r['levels']}  periods {r['periods']}  "
                f"vars {r['variables']}"
            )
    print()
    print("DRIFT:")
    for k, v in drift.items():
        if isinstance(v, list) and v:
            print(f"  {k}: {v}")
        elif not isinstance(v, list):
            print(f"  {k}: {v}")
    print()
    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}")


if __name__ == "__main__":
    main()
