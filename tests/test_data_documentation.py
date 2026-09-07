"""DATA.md must describe every source the codebase can actually read.

The failure this prevents is quiet and specific: someone adds a source adapter,
wires it into a study, and the data document silently stops being a complete
account of what the study reads. A reviewer is then reading documentation that
is true about everything except the newest thing.

The direction matters, as with the SIDRA compendium. A source documented but not
yet implemented is fine — that is a stated plan (DATA.md section 10 is entirely
such entries). A source implemented but undocumented is the defect.

This does not check that the prose is *correct* — nothing can. It checks that a
reader looking for a source by name will find it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_MD = ROOT / "docs" / "DATA.md"
SOURCES = ROOT / "brepi" / "sources"
COMPENDIUM_DB = ROOT / "docs" / "sources" / "reference" / "datasus_compendium.sqlite"

#: Source package -> the terms that count as documenting it in DATA.md. A
#: package is documented if ANY of its terms appears. Aliases are explicit
#: rather than fuzzy-matched so that a rename fails loudly here instead of
#: silently passing on a substring coincidence.
DOCUMENTED_AS: dict[str, tuple[str, ...]] = {
    "datasus": ("SINAN", "SIM", "SIH"),
    "sidra": ("SIDRA",),
    "climate": ("BR-DWGD", "ERA5", "precipitation"),
    "disasters": ("S2iD", "COBRADE", "disaster"),
    "sanitation": ("Sewer coverage", "PNSB", "sanitation"),
    "socioeconomic": ("GDP per capita", "Urban share"),
    "agriculture": ("PAM", "PPM"),
    "favelas": ("favela",),
    "landuse": ("MapBiomas", "land use"),
}


def _packages() -> list[str]:
    return sorted(
        p.name
        for p in SOURCES.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")
    )


def _text() -> str:
    return DATA_MD.read_text(encoding="utf-8")


def test_data_md_exists_and_is_substantial() -> None:
    """Guard the guard: an empty or truncated file must fail here, loudly."""
    text = _text()
    assert len(text) > 5000, (
        f"DATA.md is only {len(text)} characters -- it was probably truncated, "
        "which would make every assertion below vacuously true"
    )
    assert re.search(r"^## 1\.", text, re.MULTILINE), "DATA.md lost its section numbering"


def test_every_source_package_has_an_alias_entry() -> None:
    """A new source package must be given documentation terms deliberately."""
    missing = sorted(set(_packages()) - set(DOCUMENTED_AS))
    assert not missing, (
        f"source package(s) with no entry in DOCUMENTED_AS: {missing}. "
        "Add the terms that count as documenting them, then make sure DATA.md "
        "actually says something about them."
    )


@pytest.mark.parametrize("package", _packages())
def test_source_package_is_described_in_data_md(package: str) -> None:
    text = _text()
    terms = DOCUMENTED_AS.get(package, ())
    assert terms, f"no documentation terms declared for {package!r}"
    assert any(t.lower() in text.lower() for t in terms), (
        f"brepi/sources/{package} is implemented but DATA.md mentions none of "
        f"{list(terms)}. Describe the source there: what it is, how it is keyed, "
        "what is wrong with it, and what was done about that."
    )


def test_reference_compendium_is_documented() -> None:
    """The FTP compendium is an asset a maintainer will not find by accident."""
    text = _text()
    assert "datasus_compendium.sqlite" in text, (
        "DATA.md does not mention docs/sources/reference/datasus_compendium.sqlite. "
        "It is a full scan of the DATASUS FTP and answers 'does this series "
        "exist for the years I need, and does its layout change' before an "
        "adapter is written."
    )


@pytest.mark.skipif(not COMPENDIUM_DB.exists(), reason="compendium database not present")
def test_compendium_tables_match_what_data_md_claims() -> None:
    """The table inventory in DATA.md must match the database on disk."""
    import sqlite3

    con = sqlite3.connect(COMPENDIUM_DB)
    try:
        tables = {
            r[0]
            for r in con.execute("select name from sqlite_master where type='table'")
        }
    finally:
        con.close()
    text = _text()
    undocumented = sorted(t for t in tables if f"`{t}`" not in text)
    assert not undocumented, (
        f"compendium tables present in the database but not listed in DATA.md: "
        f"{undocumented}"
    )
