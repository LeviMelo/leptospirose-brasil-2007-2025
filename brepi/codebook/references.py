"""Reference tables too large for a YAML enumeration but no less coded.

A municipality code is as much a coded value as a sex code; it simply has 5,570
levels instead of three. Leaving it raw in an analysis is the same defect,
harder to notice.

These tables are built from the geography lattice the harness already fetches
from IBGE and are cached as parquet beside the registry so the codebook has a
single self-contained dependency directory. They are rebuilt by
``python -m brepi.codebook.references``.

The UF table keys on **both** the two-digit IBGE code and the alphabetic
abbreviation, because SINAN uses both in the same record: ``SG_UF_NOT`` holds
``43`` while ``COUFINF`` holds ``RS``. Keying on one and hoping is how half a
field silently becomes ``invalid``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

REF_DIR = Path(__file__).resolve().parent / "registry" / "references"

# The 27 federative units. Small, stable, and constitutional -- this is
# reference data, not a guess, and hard-coding it removes a network dependency
# from the decode path.
#
# The macro-region label is IBGE's own Portuguese form, because that is what the
# geography lattice and the canonical panel carry and because IBGE is the
# authority on its own divisions. An English label is available as a SEPARATE
# `region_en` attribute.
#
# This is not a style preference. Emitting English here while the lattice
# emitted Portuguese put two vocabularies for one concept into the analysis
# outputs: fourteen result files split into two camps, and a join between a
# regime assignment and an incidence table returned nothing at all. One concept,
# one canonical value, translations in their own column.
UF: tuple[tuple[str, str, str, str], ...] = (
    ("11", "RO", "Rondonia", "Norte"),
    ("12", "AC", "Acre", "Norte"),
    ("13", "AM", "Amazonas", "Norte"),
    ("14", "RR", "Roraima", "Norte"),
    ("15", "PA", "Para", "Norte"),
    ("16", "AP", "Amapa", "Norte"),
    ("17", "TO", "Tocantins", "Norte"),
    ("21", "MA", "Maranhao", "Nordeste"),
    ("22", "PI", "Piaui", "Nordeste"),
    ("23", "CE", "Ceara", "Nordeste"),
    ("24", "RN", "Rio Grande do Norte", "Nordeste"),
    ("25", "PB", "Paraiba", "Nordeste"),
    ("26", "PE", "Pernambuco", "Nordeste"),
    ("27", "AL", "Alagoas", "Nordeste"),
    ("28", "SE", "Sergipe", "Nordeste"),
    ("29", "BA", "Bahia", "Nordeste"),
    ("31", "MG", "Minas Gerais", "Sudeste"),
    ("32", "ES", "Espirito Santo", "Sudeste"),
    ("33", "RJ", "Rio de Janeiro", "Sudeste"),
    ("35", "SP", "Sao Paulo", "Sudeste"),
    ("41", "PR", "Parana", "Sul"),
    ("42", "SC", "Santa Catarina", "Sul"),
    ("43", "RS", "Rio Grande do Sul", "Sul"),
    ("50", "MS", "Mato Grosso do Sul", "Centro-Oeste"),
    ("51", "MT", "Mato Grosso", "Centro-Oeste"),
    ("52", "GO", "Goias", "Centro-Oeste"),
    ("53", "DF", "Distrito Federal", "Centro-Oeste"),
)


# IBGE macro-region -> English, for presentation only. Downstream code joins on
# `region`; a paper may print `region_en`.
REGION_EN = {
    "Norte": "North",
    "Nordeste": "Northeast",
    "Sudeste": "Southeast",
    "Sul": "South",
    "Centro-Oeste": "Central-West",
}


def uf_table() -> dict[str, Any]:
    """UF reference keyed on both the numeric code and the abbreviation."""
    values: dict[str, str] = {}
    abbr: dict[str, str] = {}
    region: dict[str, str] = {}
    region_en: dict[str, str] = {}
    for code, ab, name, reg in UF:
        for key in (code, ab):
            values[key] = name
            abbr[key] = ab
            region[key] = reg
            region_en[key] = REGION_EN[reg]
    return {"values": values,
            "attributes": {"abbr": abbr, "region": region,
                           "region_en": region_en}}


def municipality_table(*, refresh: bool = False) -> dict[str, Any]:
    """Municipality reference keyed on the six-digit SINAN code.

    SINAN writes the six-digit municipality code (no check digit); IBGE and the
    panels use seven. The lattice carries both, so the mapping is exact rather
    than reconstructed.
    """
    path = REF_DIR / "ibge_municipality.parquet"
    if path.exists() and not refresh:
        df = pl.read_parquet(path)
    else:
        from brepi.geo import lattice
        df = lattice.load_municipalities(refresh=refresh).select(
            ["code6", "code7", "name", "uf_abbr", "region"])
        REF_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    code6 = df["code6"].cast(pl.Utf8).to_list()
    abbrs = df["uf_abbr"].to_list()
    # Macro-region is derived from the UF table rather than taken from the
    # lattice, so that a municipality's region and a UF's region are guaranteed
    # to be the same string by construction rather than by both happening to be
    # right. Taking each from its own source once produced ten macro-regions
    # instead of five.
    uf = uf_table()
    uf_region = uf["attributes"]["region"]
    return {
        "zero_pad": 6,
        "values": dict(zip(code6, df["name"].to_list())),
        "attributes": {
            "uf": dict(zip(code6, abbrs)),
            "region": {c: uf_region.get(a) for c, a in zip(code6, abbrs)},
            "region_en": {c: REGION_EN.get(uf_region.get(a), None)
                          for c, a in zip(code6, abbrs)},
            "code7": dict(zip(code6, df["code7"].cast(pl.Utf8).to_list())),
        },
    }


_CACHE: dict[str, dict[str, Any]] = {}


def load_references(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Every reference table, built once per process."""
    if not _CACHE or refresh:
        _CACHE.clear()
        _CACHE["ibge_uf"] = uf_table()
        _CACHE["ibge_municipality"] = municipality_table(refresh=refresh)
    return _CACHE


if __name__ == "__main__":
    refs = load_references(refresh=True)
    for name, tab in refs.items():
        print(f"{name}: {len(tab['values'])} keys, "
              f"attributes {sorted(tab.get('attributes', {}))}")
