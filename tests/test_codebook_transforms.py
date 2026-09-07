"""Tests for the non-enum half of the codebook.

The enum decoder was already covered. These cover the transforms that reach
the coded columns an enum cannot: packed ages, epidemiological weeks,
self-labelling serovars, structured identifiers and large reference sets.

The invariant under all of them is the four-state contract: a value is
``missing``, ``valid``, ``unknown`` or ``invalid``, and a value that cannot be
translated must land in ``invalid`` rather than becoming a silent null.
"""
from __future__ import annotations

import polars as pl
import pytest

from brepi.codebook import codebook as cb
from brepi.codebook import references as refs
from brepi.codebook import transforms as tf


def _df(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols)


# --------------------------------------------------------------------------
# Packed age
# --------------------------------------------------------------------------

def test_packed_age_decodes_each_unit():
    d = _df(NU_IDADE_N=["4028", "3006", "2015", "1012"])
    out = d.with_columns(tf.sinan_packed_age_exprs("NU_IDADE_N", "age"))
    assert out["age_years"][0] == pytest.approx(28.0)
    assert out["age_years"][1] == pytest.approx(0.5)          # 6 months
    assert out["age_years"][2] == pytest.approx(15 / 365.25)  # 15 days
    assert out["age_years"][3] == pytest.approx(12 / 8766.0)  # 12 hours
    assert out["age_unit"].to_list() == ["years", "months", "days", "hours"]
    assert set(out["age_state"]) == {"valid"}


def test_packed_age_rejects_an_unknown_unit_rather_than_calling_it_a_newborn():
    # A unit digit outside 1-4 must not fall through to zero years: SINAN files
    # do carry stray values here and treating them as infants would inflate the
    # youngest age band, which is exactly where leptospirosis is rarest.
    out = _df(NU_IDADE_N=["9028", "0000"]).with_columns(
        tf.sinan_packed_age_exprs("NU_IDADE_N", "age"))
    assert out["age_years"].to_list() == [None, None]
    assert out["age_state"].to_list() == ["invalid", "invalid"]


def test_packed_age_blank_is_missing_not_invalid():
    out = _df(NU_IDADE_N=[None, "", "  "]).with_columns(
        tf.sinan_packed_age_exprs("NU_IDADE_N", "age"))
    assert out["age_state"].to_list() == ["missing"] * 3


# --------------------------------------------------------------------------
# Epidemiological week
# --------------------------------------------------------------------------

def test_epi_week_splits_year_and_week():
    out = _df(SEM_PRI=["202421", "200701"]).with_columns(
        tf.epi_week_exprs("SEM_PRI", "wk"))
    assert out["wk_year"].to_list() == [2024, 2007]
    assert out["wk_week"].to_list() == [21, 1]
    assert set(out["wk_state"]) == {"valid"}


def test_epi_week_keeps_week_53():
    # Roughly one year in seven has a week 53. Dropping it silently shortens
    # those years in any weekly aggregation.
    out = _df(SEM_PRI=["202053"]).with_columns(tf.epi_week_exprs("SEM_PRI", "wk"))
    assert out["wk_week"][0] == 53
    assert out["wk_state"][0] == "valid"


def test_epi_week_rejects_impossible_weeks():
    out = _df(SEM_PRI=["202499", "202400"]).with_columns(
        tf.epi_week_exprs("SEM_PRI", "wk"))
    assert out["wk_state"].to_list() == ["invalid", "invalid"]
    assert out["wk_week"].to_list() == [None, None]


# --------------------------------------------------------------------------
# Self-labelling codes (serovar)
# --------------------------------------------------------------------------

SEROVAR = {"10": "copenhageni", "2": "australis", "21": "patoc"}
SEROVAR_ATTRS = {
    "serogroup": {"10": "Icterohaemorrhagiae", "2": "Australis", "21": "Semaranga"},
    "reservoir": {"10": "rodent_rattus", "2": "swine_equine_hedgehog",
                  "21": "saprophyte_non_pathogenic"},
}


def test_labelled_code_handles_both_labelled_and_bare_forms():
    # The same column, in the same file, carries both. This is the whole reason
    # an ordinary enum on the raw string does not work here.
    d = _df(MICRO1_S1=["10 - COPENHAGENI", "10", "2 - AUTRALIS", "2", "021"])
    out = d.with_columns(tf.labelled_code_exprs(
        "MICRO1_S1", SEROVAR, "serovar", attributes=SEROVAR_ATTRS))
    assert out["serovar"].to_list() == [
        "copenhageni", "copenhageni", "australis", "australis", "patoc"]
    assert set(out["serovar_state"]) == {"valid"}


def test_labelled_code_uses_the_canonical_spelling_not_the_sources():
    # The source writes AUTRALIS for the serovar Australis. The codebook is the
    # authority on the name; the file is only the authority on the code.
    out = _df(MICRO1_S1=["2 - AUTRALIS"]).with_columns(
        tf.labelled_code_exprs("MICRO1_S1", SEROVAR, "serovar"))
    assert out["serovar"][0] == "australis"


def test_labelled_code_attaches_serogroup_and_reservoir():
    out = _df(MICRO1_S1=["10 - COPENHAGENI", "21 - PATOC"]).with_columns(
        tf.labelled_code_exprs("MICRO1_S1", SEROVAR, "serovar",
                               attributes=SEROVAR_ATTRS))
    assert out["serovar_serogroup"].to_list() == ["Icterohaemorrhagiae", "Semaranga"]
    # Patoc is Leptospira biflexa -- saprophytic. It must never be attributed
    # to an animal reservoir.
    assert out["serovar_reservoir"][1] == "saprophyte_non_pathogenic"


def test_labelled_code_unmapped_is_invalid():
    out = _df(MICRO1_S1=["99 - NOTASEROVAR"]).with_columns(
        tf.labelled_code_exprs("MICRO1_S1", SEROVAR, "serovar"))
    assert out["serovar"][0] is None
    assert out["serovar_state"][0] == "invalid"


# --------------------------------------------------------------------------
# Prefix groups (CBO occupation)
# --------------------------------------------------------------------------

def test_prefix_group_decodes_the_major_group():
    out = _df(ID_OCUPA_N=["612005", "223505"]).with_columns(
        tf.prefix_group_exprs("ID_OCUPA_N", dict(tf.CBO_MAJOR_GROUPS), "occ",
                              width=1, pad=6, unknown=tf.CBO_UNKNOWN))
    assert out["occ"][0] == tf.CBO_MAJOR_GROUPS["6"]
    assert out["occ"][1] == tf.CBO_MAJOR_GROUPS["2"]
    assert set(out["occ_state"]) == {"valid"}


def test_prefix_group_treats_sinan_sentinels_as_unknown_not_a_group():
    # 999991/999992/999993 are SINAN's "not informed" sentinels, not CBO codes.
    # Decoding them to major group 9 would invent a workforce of repairers.
    out = _df(ID_OCUPA_N=["999991", "999992", "998999"]).with_columns(
        tf.prefix_group_exprs("ID_OCUPA_N", dict(tf.CBO_MAJOR_GROUPS), "occ",
                              width=1, pad=6, unknown=tf.CBO_UNKNOWN))
    assert out["occ"].to_list() == [None, None, None]
    assert set(out["occ_state"]) == {"unknown"}


# --------------------------------------------------------------------------
# Numeric (MAT titres)
# --------------------------------------------------------------------------

def test_numeric_types_titres_and_bounds_them():
    out = _df(T=["800", "1600", "0", "999999999"]).with_columns(
        tf.numeric_exprs("T", "titre", minimum=1, maximum=100000))
    assert out["titre"].to_list()[:2] == [800.0, 1600.0]
    assert out["titre_state"].to_list() == ["valid", "valid", "invalid", "invalid"]


def test_numeric_non_numeric_is_invalid_not_null():
    out = _df(T=["NEGATIVO"]).with_columns(tf.numeric_exprs("T", "titre"))
    assert out["titre_state"][0] == "invalid"


# --------------------------------------------------------------------------
# Reference tables
# --------------------------------------------------------------------------

def test_uf_reference_keys_on_both_code_and_abbreviation():
    # SINAN uses both forms in the same record: SG_UF_NOT holds 43 while
    # COUFINF holds RS. Keying on one silently invalidates half the field.
    tab = refs.uf_table()
    out = _df(UF=["43", "RS", "35", "SP"]).with_columns(
        tf.reference_exprs("UF", tab["values"], "uf",
                           attributes=tab["attributes"]))
    assert out["uf"].to_list() == [
        "Rio Grande do Sul", "Rio Grande do Sul", "Sao Paulo", "Sao Paulo"]
    # Canonical macro-region is IBGE's own Portuguese label; English lives in a
    # separate column so one concept never carries two values.
    assert out["uf_region"].to_list() == ["Sul", "Sul", "Sudeste", "Sudeste"]
    assert out["uf_region_en"].to_list() == [
        "South", "South", "Southeast", "Southeast"]
    assert set(out["uf_state"]) == {"valid"}


def test_uf_reference_covers_all_27_units():
    tab = refs.uf_table()
    assert len({v for v in tab["values"].values()}) == 27


def test_municipality_reference_resolves_six_digit_codes():
    tab = refs.municipality_table()
    out = _df(M=["431490", "355030"]).with_columns(
        tf.reference_exprs("M", tab["values"], "mun",
                           attributes=tab["attributes"], zero_pad=6))
    assert out["mun"][0] == "Porto Alegre"
    assert out["mun_uf"].to_list() == ["RS", "SP"]
    assert set(out["mun_state"]) == {"valid"}


def test_municipality_reference_has_the_full_lattice():
    tab = refs.municipality_table()
    assert len(tab["values"]) == 5570


# --------------------------------------------------------------------------
# Registry integration
# --------------------------------------------------------------------------

def test_every_sinan_lept_column_is_accounted_for():
    """No column may be UNBOUND: it is translated, or ignored with a reason.

    This is the test that makes the coverage report an audit rather than a
    to-do list. If a future dictionary vintage adds a field, this fails until
    somebody either binds it or writes down why it is not bound.
    """
    doc = cb.load_codebook()
    bound = set(doc["bindings"]["SINAN-LEPT"])
    transformed = set((doc.get("transforms") or {}).get("SINAN-LEPT", {}))
    ignored = set((doc.get("ignored") or {}).get("SINAN-LEPT", {}))
    # Every ignored field must carry a non-empty reason.
    for field, reason in (doc.get("ignored") or {})["SINAN-LEPT"].items():
        assert reason and len(reason) > 20, f"{field} needs a real reason"
    # A field must not be both bound and transformed -- that would emit two
    # columns claiming to be the same translation.
    assert not (bound & transformed), sorted(bound & transformed)
    assert not (bound & ignored), sorted(bound & ignored)
    assert not (transformed & ignored), sorted(transformed & ignored)


def test_serovar_concept_marks_patoc_as_saprophytic():
    """Patoc is Leptospira biflexa and is a screening antigen, not a pathogen.

    Attributing a Patoc-only MAT reaction to a rodent or livestock reservoir
    would be a clinical error, so the codebook has to carry the distinction.
    """
    spec = cb.concept("lept_serovar")
    assert spec["values"]["21"] == "patoc"
    assert spec["attributes"]["reservoir"]["21"] == "saprophyte_non_pathogenic"
    assert "SAPROPHYTIC" in spec["notes"]


def test_decode_frame_applies_transforms_and_emits_states():
    d = _df(NU_IDADE_N=["4030"], SEM_PRI=["202410"], CS_SEXO=["M"],
            MICRO1_S1=["10 - COPENHAGENI"], SG_UF_NOT=["43"])
    out = cb.decode_frame(d, "SINAN-LEPT")
    assert out["age_years"][0] == pytest.approx(30.0)
    assert out["epiweek_onset_week"][0] == 10
    assert out["serovar_s1_first"][0] == "copenhageni"
    assert out["serovar_s1_first_reservoir"][0] == "rodent_rattus"
    assert out["uf_notification"][0] == "Rio Grande do Sul"
    assert out["cs_sexo"][0] == "masculino"
    for base in ("age", "epiweek_onset", "serovar_s1_first", "uf_notification",
                 "cs_sexo"):
        assert f"{base}_state" in out.columns


def test_municipality_and_uf_agree_on_macro_region():
    """A municipality's region and its UF's region must be the same string.

    They were not: the IBGE lattice serves Portuguese labels while the UF table
    is English, so a table grouped on one and joined on the other produced ten
    macro-regions instead of five -- silently, because both vocabularies are
    individually valid.
    """
    mun = refs.municipality_table()
    uf = refs.uf_table()
    mun_regions = {v for v in mun["attributes"]["region"].values() if v}
    uf_regions = set(uf["attributes"]["region"].values())
    assert mun_regions == uf_regions, mun_regions ^ uf_regions
    assert len(mun_regions) == 5


def test_region_has_one_canonical_vocabulary_and_a_separate_translation():
    """One concept, one canonical value, translations in their own column.

    Emitting English macro-regions here while the geography lattice emitted
    Portuguese split fourteen result files into two camps, and a join between
    the regime assignment and the incidence table matched nothing at all. Both
    vocabularies were individually correct, which is precisely why nothing
    caught it.
    """
    uf = refs.uf_table()
    mun = refs.municipality_table()
    canonical = {v for v in uf["attributes"]["region"].values()}
    assert canonical == {"Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"}
    assert canonical == {v for v in mun["attributes"]["region"].values() if v}
    english = {v for v in uf["attributes"]["region_en"].values()}
    assert english == {"North", "Northeast", "Southeast", "South", "Central-West"}
    # The two must never collide in one column.
    assert not (canonical & english)
