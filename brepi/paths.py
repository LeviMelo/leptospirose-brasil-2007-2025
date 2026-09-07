"""Canonical data layout.

Every path the study reads or writes is named here once. The alternative --
each script composing its own output path from a string literal --
produced twenty-two loose files at one directory level, three naming
conventions for the same kind of artefact, and log files interleaved with
results. That is not a cosmetic problem: an export script cannot reliably
declare a result set it has to guess the location of, and a staleness check
cannot map a result to its producer if the mapping lives in nobody's head.

The layout separates artefacts by **how they are produced and how long they
live**, not by which script happened to make them:

``cache/``
    Raw bytes exactly as retrieved, content-addressed, with a sibling
    ``.provenance.json``. Never edited, never regenerated in place.

``derived/``
    Decoded and code-translated source files, keyed on the input content hash
    plus the transform version. Deleting this directory costs CPU, not data.

``interim/``
    Study-specific intermediates: the line-level extract, the denominator
    tensor, per-system disease extracts.

``panel/``
    Analysis-ready panels on the municipality-month spine, plus the adjacency
    graphs and geometry.

``results/``
    One directory per analysis stage. A stage owns its directory; nothing else
    writes there. This is what makes the export manifest's producer map
    truthful rather than aspirational.

``logs/``
    Run logs. Separated from results because a log is evidence about a run and
    a result is evidence about the world, and mixing them means a directory
    listing answers neither question.

``export/``
    Versioned manuscript bundles with manifests and provenance.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CACHE = DATA / "cache"
DERIVED = DATA / "derived"
INTERIM = DATA / "interim"
PANEL = DATA / "panel"
GRAPHS = PANEL / "graphs"
RESULTS = DATA / "results"
LOGS = DATA / "logs"
EXPORT = DATA / "export"

# One directory per analysis stage. The numeric prefix orders them the way the
# manuscript reads, not the way they were written.
STAGES: dict[str, str] = {
    "data_quality": "00_data_quality",
    "descriptive": "01_descriptive",
    "rq1": "rq1_exposure_response",
    "rq2": "rq2_flood_disasters",
    "rq3": "rq3_ascertainment",
    "rq4": "rq4_lethality",
    "rq5": "rq5_regimes",
    # Unit-level tables that join every stage's answers onto one spine. Not an
    # analysis in its own right; the place a figure or a map reads from.
    "atlas": "atlas",
    "benchmarks": "benchmarks",
    "dictionary": "dictionary",
}


def stage(name: str, *parts: str, create: bool = True) -> Path:
    """The results directory for an analysis stage.

    >>> stage("rq4").name
    'rq4_lethality'

    Raises rather than inventing a directory for an unknown stage: a typo that
    silently creates ``results/rq4_lethaliy`` is exactly the failure this
    module exists to prevent.
    """
    if name not in STAGES:
        raise KeyError(
            f"unknown stage {name!r}. Known stages: {sorted(STAGES)}. "
            "Add it to brepi.paths.STAGES rather than composing a path.")
    p = RESULTS / STAGES[name]
    for part in parts:
        p = p / part
    if create:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def log(name: str, *, create: bool = True) -> Path:
    """Path for a run log."""
    if create:
        LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / (name if name.endswith(".log") else f"{name}.log")


# Named panels, so a typo in a filename is an import error rather than an
# empty result.
PANEL_BASE = PANEL / "lept_panel_municipality_month.parquet"
PANEL_CLIMATE = PANEL / "lept_panel_rq1_municipality_month.parquet"
PANEL_STRUCTURAL = PANEL / "lept_panel_rq1_structural_municipality_month.parquet"
PANEL_ANALYSIS = PANEL / "lept_panel_rq1_socioeconomic_municipality_month.parquet"

LINE_LEVEL = INTERIM / "lept_line_level.parquet"
POPULATION_TENSOR = INTERIM / "population_tensor_long.parquet"
POPULATION_ANNUAL = INTERIM / "population_municipal_year.parquet"
SIM_DEATHS = INTERIM / "sim_a27_deaths.parquet"
SIH_PARTS = INTERIM / "sih_a27_parts"

FLOOD_EVENTS = PANEL / "flood_events.parquet"
FLOOD_DECLARATIONS = PANEL / "flood_declarations.parquet"
FLOOD_FIRST_TREATMENT = PANEL / "flood_first_treatment.parquet"
DROUGHT_EVENTS = PANEL / "drought_events.parquet"

GRAPH_HEALTH_REGION = GRAPHS / "health_region.adj"
GRAPH_MUNICIPALITY = GRAPHS / "municipality_all.adj"
