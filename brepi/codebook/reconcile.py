"""Reconcile an official DATASUS data dictionary against the actual files.

A published dictionary and the disseminated microdata disagree, always. The
dictionary describes the *form* as designed at some version; the files carry
what the system wrote across two decades of schema evolution, and DATASUS does
not republish a dictionary per vintage. So the question is never "what does the
dictionary say" but "where does the dictionary and the data disagree, and in
which direction".

Four disagreement classes, each with a different consequence:

``documented_absent``
    A field the dictionary specifies that is not in the file. Usually stripped
    at dissemination for disclosure control. On leptospirosis v5.0 this covers
    ``CON_AREA`` (urban/rural of the probable infection site), ``ID_BAIRRO``,
    ``CODISINF``, ``CO_BAINFC`` and ``ATE_HOSPIT`` -- and a plan that assumed
    ``CON_AREA`` was available had to be rewritten because of it.

``undocumented_present``
    A field in the file with no dictionary entry. These are usually the
    reporting-chain fields (``DT_DIGITA``, the ``DT_TRANS*`` block) that carry
    real surveillance-process information nobody uses.

``undocumented_code``
    A *value* observed in a documented field that the dictionary does not list.
    The dangerous class, because it looks like valid data. ``CLASSI_FIN = '8'``
    occurs 11,699 times on leptospirosis and is absent from v5.0.

``unused_code``
    A code the dictionary defines that never occurs. Harmless, but it flags
    fields the form collects and the field never fills.

The output is a table, not a judgement: which disagreements matter is a
scientific decision, and the point of writing them down is that the decision is
made once and visibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import polars as pl

from brepi.codebook.codebook import BLANK


@dataclass(frozen=True)
class DictionaryField:
    """One field as the official dictionary declares it."""

    name: str  # DBF/CSV column name
    label: str | None = None
    dtype: str | None = None
    categories: Mapping[str, str] | None = None
    required: bool | None = None
    note: str | None = None


def reconcile(
    frame: pl.DataFrame,
    declared: Sequence[DictionaryField],
    *,
    max_codes: int = 60,
    sample_note: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Compare a real frame against a declared dictionary.

    ``max_codes`` bounds the cardinality at which a field is treated as
    categorical for code-level comparison; above it the field is free text or
    an identifier and its values are not dictionary-checked.
    """
    by_name = {d.name.upper(): d for d in declared}
    cols = {c.upper(): c for c in frame.columns}

    documented_absent = pl.DataFrame(
        [
            {"field": d.name, "label": d.label, "note": d.note}
            for key, d in by_name.items()
            if key not in cols
        ],
        schema={"field": pl.Utf8, "label": pl.Utf8, "note": pl.Utf8},
    )

    undocumented_present = pl.DataFrame(
        [
            {"field": actual, "n_distinct": frame[actual].n_unique()}
            for key, actual in cols.items()
            if key not in by_name
        ],
        schema={"field": pl.Utf8, "n_distinct": pl.Int64},
    ).sort("n_distinct")

    undoc_rows: list[dict] = []
    unused_rows: list[dict] = []
    for key, actual in cols.items():
        spec = by_name.get(key)
        if spec is None or not spec.categories:
            continue
        n_distinct = frame[actual].n_unique()
        if n_distinct > max_codes:
            continue
        declared_codes = {str(c).strip() for c in spec.categories}
        observed = (
            frame.select(
                pl.col(actual).cast(pl.Utf8).str.strip_chars().alias("code")
            )
            .filter(
                pl.col("code").is_not_null()
                & ~pl.col("code").str.to_uppercase().is_in(list(BLANK))
            )
            .group_by("code")
            .agg(pl.len().alias("n"))
        )
        for row in observed.iter_rows(named=True):
            code = row["code"]
            if code in declared_codes or code.lstrip("0") in declared_codes:
                continue
            undoc_rows.append(
                {"field": actual, "code": code, "n": row["n"], "label": spec.label}
            )
        seen = set(observed["code"].to_list())
        seen |= {c.lstrip("0") for c in seen}
        for code, meaning in spec.categories.items():
            if str(code).strip() not in seen:
                unused_rows.append(
                    {"field": actual, "code": str(code), "meaning": str(meaning)}
                )

    schema_undoc = {"field": pl.Utf8, "code": pl.Utf8, "n": pl.Int64, "label": pl.Utf8}
    schema_unused = {"field": pl.Utf8, "code": pl.Utf8, "meaning": pl.Utf8}

    summary = pl.DataFrame(
        [
            {"class": "documented_absent", "n": documented_absent.height},
            {"class": "undocumented_present", "n": undocumented_present.height},
            {"class": "undocumented_code", "n": len(undoc_rows)},
            {"class": "unused_code", "n": len(unused_rows)},
            {"class": "declared_fields", "n": len(declared)},
            {"class": "file_columns", "n": frame.width},
            {"class": "file_rows", "n": frame.height},
        ]
    )
    if sample_note:
        summary = summary.with_columns(pl.lit(sample_note).alias("sample"))

    return {
        "summary": summary,
        "documented_absent": documented_absent,
        "undocumented_present": undocumented_present,
        "undocumented_code": (
            pl.DataFrame(undoc_rows, schema=schema_undoc).sort("n", descending=True)
            if undoc_rows else pl.DataFrame(schema=schema_undoc)
        ),
        "unused_code": (
            pl.DataFrame(unused_rows, schema=schema_unused).sort(["field", "code"])
            if unused_rows else pl.DataFrame(schema=schema_unused)
        ),
    }


def write_reconciliation(result: Mapping[str, pl.DataFrame], directory) -> None:
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.write_csv(d / f"dictionary_{name}.csv")


__all__ = ["DictionaryField", "reconcile", "write_reconciliation"]
