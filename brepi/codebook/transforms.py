"""Field transforms: the coded columns an enum lookup cannot reach.

A codebook that only maps ``code -> label`` translates the easy half of a
DATASUS record. The other half is coded too, just not as a small enumeration:

* **Packed composites.** SINAN stores age as ``UVVV`` -- a unit digit and a
  three-digit value -- so ``4028`` is 28 years and ``3006`` is six months.
  Read as a number it is nonsense; read as a category it has 137 "levels".
* **Structured identifiers.** Epidemiological week is ``YYYYWW``; an IBGE
  municipality code carries its UF in the first two digits; a CBO occupation
  code carries its major group in the first.
* **Self-labelling codes.** The leptospirosis MAT serovar fields hold both
  ``"10 - COPENHAGENI"`` and a bare ``"10"``, in the same column, in the same
  year. Neither an enum on the raw string nor a numeric cast handles both.
* **Large reference sets.** 5,570 municipalities do not belong in a YAML
  enumeration, but they are no less coded for that.

Each transform emits its outputs **plus a state column**, on the same
four-state contract as :func:`brepi.codebook.codebook.categorical_exprs`:
``missing``, ``valid``, ``unknown``, ``invalid``. A value that cannot be
transformed is never silently dropped and never silently coerced to null: it
becomes ``invalid`` and shows up in the coverage report.
"""
from __future__ import annotations

from typing import Any

import polars as pl

BLANK = {"", "NA", "NAN", "NONE", "NULL", "."}

# CBO 2002 major groups (grande grupo), the first digit of the six-digit code.
# The 9999xx range is SINAN's own "not applicable / not informed" sentinel and
# is not part of CBO.
CBO_MAJOR_GROUPS: dict[str, str] = {
    "0": "forcas_armadas_policiais_bombeiros",
    "1": "membros_superiores_do_poder_publico_dirigentes_e_gerentes",
    "2": "profissionais_das_ciencias_e_das_artes",
    "3": "tecnicos_de_nivel_medio",
    "4": "trabalhadores_de_servicos_administrativos",
    "5": "trabalhadores_dos_servicos_comercio_e_vendedores",
    "6": "trabalhadores_agropecuarios_florestais_e_da_pesca",
    "7": "trabalhadores_da_producao_de_bens_e_servicos_industriais",
    "8": "trabalhadores_da_producao_de_bens_e_servicos_industriais_continuos",
    "9": "trabalhadores_de_manutencao_e_reparacao",
}
CBO_UNKNOWN = {"999991", "999992", "999993", "999994", "999995", "998999",
               "000000", "999999"}


def _raw(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()


def _is_blank(raw: pl.Expr) -> pl.Expr:
    return raw.is_null() | raw.str.to_uppercase().is_in(list(BLANK))


def _state(is_blank: pl.Expr, ok: pl.Expr, unknown: pl.Expr | None = None) -> pl.Expr:
    unknown = unknown if unknown is not None else pl.lit(False)
    return (
        pl.when(is_blank).then(pl.lit("missing"))
        .when(ok).then(pl.lit("valid"))
        .when(unknown).then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid"))
    )


def sinan_packed_age_exprs(column: str, name: str = "age") -> list[pl.Expr]:
    """``UVVV`` -> age in years.

    Unit digit: 1 hours, 2 days, 3 months, 4 years. Sub-year ages are converted
    rather than floored to zero, so ``<name>_years`` is a genuine continuous
    age; ``<name>_unit`` is retained because "3 months" and "0 years" are the
    same number and different facts.

    A unit outside 1-4 is ``invalid``, not silently zero -- SINAN files do
    carry stray values here and treating them as newborns would inflate the
    youngest age band.
    """
    raw = _raw(column)
    blank = _is_blank(raw)
    padded = raw.str.zfill(4)
    unit = padded.str.slice(0, 1)
    value = padded.str.slice(1, 3).cast(pl.Int32, strict=False)
    ok = unit.is_in(["1", "2", "3", "4"]) & value.is_not_null()

    years = (
        pl.when(unit == "4").then(value.cast(pl.Float64))
        .when(unit == "3").then(value.cast(pl.Float64) / 12.0)
        .when(unit == "2").then(value.cast(pl.Float64) / 365.25)
        .when(unit == "1").then(value.cast(pl.Float64) / 8766.0)
        .otherwise(None)
    )
    unit_label = (
        pl.when(unit == "4").then(pl.lit("years"))
        .when(unit == "3").then(pl.lit("months"))
        .when(unit == "2").then(pl.lit("days"))
        .when(unit == "1").then(pl.lit("hours"))
        .otherwise(None)
    )
    return [
        pl.when(blank).then(None).otherwise(years).alias(f"{name}_years"),
        pl.when(blank).then(None).otherwise(unit_label).alias(f"{name}_unit"),
        _state(blank, ok).alias(f"{name}_state"),
    ]


def epi_week_exprs(column: str, name: str = "epiweek") -> list[pl.Expr]:
    """``YYYYWW`` -> epidemiological year and week.

    Weeks outside 1-53 are ``invalid``. Week 53 is legitimate and is kept:
    roughly one year in seven has one, and dropping it shortens those years by
    a week in any weekly aggregation.
    """
    raw = _raw(column)
    blank = _is_blank(raw)
    padded = raw.str.zfill(6)
    year = padded.str.slice(0, 4).cast(pl.Int32, strict=False)
    week = padded.str.slice(4, 2).cast(pl.Int32, strict=False)
    ok = (year.is_not_null() & week.is_not_null()
          & (year >= 1990) & (year <= 2100) & (week >= 1) & (week <= 53))
    return [
        pl.when(blank | ~ok).then(None).otherwise(year).alias(f"{name}_year"),
        pl.when(blank | ~ok).then(None).otherwise(week).alias(f"{name}_week"),
        _state(blank, ok).alias(f"{name}_state"),
    ]


def labelled_code_exprs(
    column: str,
    values: dict[str, str],
    name: str,
    *,
    attributes: dict[str, dict[str, str]] | None = None,
    unknown: list[str] | None = None,
) -> list[pl.Expr]:
    """Decode a column holding ``"10 - COPENHAGENI"`` and bare ``"10"`` alike.

    DATASUS forms that offer a picklist sometimes persist the label with the
    code and sometimes only the code, varying by year and by state. Taking the
    leading integer normalises both to the code, after which the mapping is an
    ordinary enum -- and the *dictionary's own label* never has to be trusted
    for spelling, which matters here because the source writes ``AUTRALIS``
    for the serovar Australis.

    ``attributes`` attaches derived columns keyed by the same code, which is
    how a serovar acquires its serogroup and its maintenance host without a
    second join.
    """
    raw = _raw(column)
    blank = _is_blank(raw)
    # Leading run of digits, whether or not a label follows.
    code = raw.str.extract(r"^\s*0*(\d+)", 1)
    mapped = code.replace_strict(values, default=None, return_dtype=pl.Utf8)
    is_unknown = code.is_in(unknown) if unknown else pl.lit(False)

    out = [
        pl.when(blank).then(None).otherwise(mapped).alias(name),
        _state(blank, mapped.is_not_null(), is_unknown).alias(f"{name}_state"),
        pl.when(blank).then(None).otherwise(code).alias(f"{name}_code"),
    ]
    for attr, mapping in (attributes or {}).items():
        out.append(
            pl.when(blank).then(None)
            .otherwise(code.replace_strict(mapping, default=None, return_dtype=pl.Utf8))
            .alias(f"{name}_{attr}")
        )
    return out


def prefix_group_exprs(
    column: str,
    groups: dict[str, str],
    name: str,
    *,
    width: int = 1,
    unknown: set[str] | list[str] | None = None,
    pad: int | None = None,
) -> list[pl.Expr]:
    """Decode the *structure* of an identifier whose full table is unavailable.

    A CBO occupation code is six digits and its authoritative title list is a
    separate download; its first digit, however, is the major occupational
    group and is defined by the classification itself. Decoding what the code
    structurally guarantees is honest partial translation -- and far better
    than leaving 698 six-digit numbers in the analysis or pretending the titles
    are known.
    """
    raw = _raw(column)
    if pad:
        raw = raw.str.zfill(pad)
    blank = _is_blank(raw)
    unknown = set(unknown or ())
    is_unknown = raw.is_in(list(unknown)) if unknown else pl.lit(False)
    key = raw.str.slice(0, width)
    mapped = key.replace_strict(groups, default=None, return_dtype=pl.Utf8)
    ok = mapped.is_not_null() & ~is_unknown
    return [
        pl.when(blank | is_unknown).then(None).otherwise(mapped).alias(name),
        _state(blank, ok, is_unknown).alias(f"{name}_state"),
    ]


def reference_exprs(
    column: str,
    mapping: dict[str, str],
    name: str,
    *,
    attributes: dict[str, dict[str, str]] | None = None,
    zero_pad: int | None = None,
) -> list[pl.Expr]:
    """Resolve a code against a large reference table held in memory.

    Used for municipality and UF codes, where the value set is far too large
    for a YAML enumeration but is no less coded for that. Implemented as a
    vectorised replace rather than a join so it composes with every other
    transform in a single ``with_columns``.
    """
    raw = _raw(column)
    if zero_pad:
        raw = raw.str.zfill(zero_pad)
    blank = _is_blank(raw)
    mapped = raw.replace_strict(mapping, default=None, return_dtype=pl.Utf8)
    out = [
        pl.when(blank).then(None).otherwise(mapped).alias(name),
        _state(blank, mapped.is_not_null()).alias(f"{name}_state"),
    ]
    for attr, m in (attributes or {}).items():
        out.append(
            pl.when(blank).then(None)
            .otherwise(raw.replace_strict(m, default=None, return_dtype=pl.Utf8))
            .alias(f"{name}_{attr}")
        )
    return out


def numeric_exprs(column: str, name: str, *, minimum: float | None = None,
                  maximum: float | None = None) -> list[pl.Expr]:
    """Type a column that is a *number* stored as text, with a plausible range.

    A MAT titre is the reciprocal of the highest reacting dilution -- 100, 200,
    400, 800 -- and is a quantity, not a category; leaving it as a string is as
    much a translation failure as leaving a sex code untranslated, and casting
    it blindly hides the entries that are not numbers at all. The range bound
    makes an implausible value ``invalid`` rather than an outlier that survives
    into a model.
    """
    raw = _raw(column)
    blank = _is_blank(raw)
    val = raw.cast(pl.Float64, strict=False)
    ok = val.is_not_null()
    if minimum is not None:
        ok = ok & (val >= minimum)
    if maximum is not None:
        ok = ok & (val <= maximum)
    return [
        pl.when(blank | ~ok).then(None).otherwise(val).alias(name),
        _state(blank, ok).alias(f"{name}_state"),
    ]


TRANSFORM_KINDS = {
    "sinan_packed_age": sinan_packed_age_exprs,
    "epi_week": epi_week_exprs,
}


def build(kind: str, column: str, spec: dict[str, Any],
          references: dict[str, Any] | None = None) -> list[pl.Expr]:
    """Dispatch one registry transform entry to its expression builder."""
    name = spec.get("name") or column.lower()
    if kind in TRANSFORM_KINDS:
        return TRANSFORM_KINDS[kind](column, name)
    if kind == "labelled_code":
        return labelled_code_exprs(
            column, spec["values"], name,
            attributes=spec.get("attributes"), unknown=spec.get("unknown"))
    if kind == "numeric":
        return numeric_exprs(column, name, minimum=spec.get("minimum"),
                             maximum=spec.get("maximum"))
    if kind == "prefix_group":
        return prefix_group_exprs(
            column, spec["groups"], name, width=spec.get("width", 1),
            unknown=spec.get("unknown"), pad=spec.get("pad"))
    if kind == "reference":
        table = (references or {}).get(spec["table"])
        if table is None:
            raise KeyError(
                f"transform for {column!r} needs reference table "
                f"{spec['table']!r}, which is not loaded")
        return reference_exprs(
            column, table["values"], name,
            attributes=table.get("attributes"),
            zero_pad=spec.get("zero_pad") or table.get("zero_pad"))
    raise KeyError(f"unknown transform kind {kind!r} for column {column!r}")
