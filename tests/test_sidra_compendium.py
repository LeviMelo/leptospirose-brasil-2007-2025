"""The compendium is the single source of truth; the registry is a subset.

``docs/sources/SIDRA_COMPENDIUM.md`` carries the curated judgments -- tier,
default-keep, axis, and the traps recorded under **Note** -- that no IBGE
endpoint returns. ``brepi/sources/sidra/registry.yaml`` is generated from live
metadata and covers only the tables currently wired into a pipeline.

The direction of the invariant matters. A table catalogued but not yet wired is
a normal state: it means someone documented it before using it. A table wired
but not catalogued is a defect, because the extraction then rests on metadata
nobody reviewed. That is how eleven tables -- the PAM/PPM families and the whole
Censo 2022 favela block -- ended up in the registry with no entry stating that
Tab 3939 has no ``Total`` category, or that Tab 10344 covers only the census
tracts sampled for the Pesquisa Urbanística do Entorno.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPENDIUM = ROOT / "docs" / "sources" / "SIDRA_COMPENDIUM.md"
REGISTRY = ROOT / "brepi" / "sources" / "sidra" / "registry.yaml"

#: ``#### Tab 1612 | Planted area ...`` -- the heading form used throughout.
HEADING = re.compile(r"^####\s*Tab\s+(\d+)\s*\|", re.MULTILINE)


def compendium_tables() -> set[str]:
    return set(HEADING.findall(COMPENDIUM.read_text(encoding="utf-8")))


def registry_tables() -> set[str]:
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return {str(k) for k in data["tables"]}


def test_compendium_parses() -> None:
    """Guard the guard: a heading-format change must fail loudly, not silently."""
    found = compendium_tables()
    assert len(found) > 50, (
        f"only {len(found)} table headings matched -- the '#### Tab N | ...' "
        "convention probably changed, which would make every other assertion "
        "in this module vacuously true"
    )


def test_every_registry_table_is_catalogued() -> None:
    missing = sorted(registry_tables() - compendium_tables(), key=int)
    assert not missing, (
        "wired in registry.yaml but absent from SIDRA_COMPENDIUM.md: "
        + ", ".join(missing)
        + ". Add an entry with Tier, Vars, Clsfs and any Note before use -- "
        "the registry records what an extraction asks for, not what an "
        "analyst needs to know to not misuse it."
    )


def test_no_duplicate_entries() -> None:
    text = COMPENDIUM.read_text(encoding="utf-8")
    ids = HEADING.findall(text)
    dupes = sorted({t for t in ids if ids.count(t) > 1}, key=int)
    assert not dupes, (
        "table catalogued more than once: "
        + ", ".join(dupes)
        + ". Two entries for one table means two sets of curation judgments, "
        "and nothing decides which one holds."
    )


@pytest.mark.parametrize(
    "table_id",
    ["1612", "5457", "3939", "9883", "9887", "9888", "9892", "9893", "9894",
     "10344", "4709"],
)
def test_repaired_drift_stays_repaired(table_id: str) -> None:
    """The eleven tables found undocumented on 2026-08-01.

    Pinned by id so that deleting an entry fails here with a name rather than
    only in the set-difference assertion above, which would say the same thing
    less legibly.
    """
    assert table_id in compendium_tables()
