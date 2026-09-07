"""Codebook resolution: record-level and vectorised, from one dictionary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl
import yaml

REGISTRY = Path(__file__).resolve().parent / "registry" / "datasus_codebook.yaml"

DecodeState = Literal["missing", "valid", "unknown", "invalid"]

#: Tokens that mean "the field is blank", as distinct from "the answer is
#: unknown". DATASUS files carry all of these depending on the writer.
BLANK = frozenset({"", "NA", "NAN", "NULL", "NONE", ".", "-"})


class CodebookError(KeyError):
    """A concept, system or binding was requested that the registry lacks."""


@lru_cache(maxsize=4)
def load_codebook(path: str | Path = REGISTRY) -> dict[str, Any]:
    """Load and cache the registry, validating its top-level shape."""
    p = Path(path)
    if not p.exists():
        raise CodebookError(f"codebook registry not found at {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for key in ("concepts", "bindings"):
        if key not in data:
            raise CodebookError(f"codebook at {p} has no '{key}' section")
    data.setdefault("lookups", {})
    # Every binding must point at a concept that exists. A dangling binding is
    # the failure mode that produces silently untranslated columns.
    dangling: list[str] = []
    for system, fields in data["bindings"].items():
        for field, concept_id in (fields or {}).items():
            if concept_id not in data["concepts"]:
                dangling.append(f"{system}.{field} -> {concept_id}")
    if dangling:
        raise CodebookError(
            f"{len(dangling)} binding(s) reference undefined concepts: "
            + ", ".join(dangling[:8])
        )
    return data


def systems(path: str | Path = REGISTRY) -> list[str]:
    return sorted(load_codebook(path)["bindings"])


def concept(concept_id: str, path: str | Path = REGISTRY) -> dict[str, Any]:
    book = load_codebook(path)
    if concept_id not in book["concepts"]:
        raise CodebookError(f"unknown concept '{concept_id}'")
    return book["concepts"][concept_id]


def binding_for(system: str, field: str, path: str | Path = REGISTRY) -> str | None:
    """Concept bound to ``system.field``, or ``None`` if the field is uncoded."""
    book = load_codebook(path)
    return (book["bindings"].get(system) or {}).get(field)


def concepts_for_system(system: str, path: str | Path = REGISTRY) -> dict[str, str]:
    book = load_codebook(path)
    if system not in book["bindings"]:
        raise CodebookError(
            f"unknown system '{system}'; registry has {sorted(book['bindings'])}"
        )
    return dict(book["bindings"][system] or {})


@dataclass(frozen=True)
class Decoded:
    value: str | None
    state: DecodeState
    raw: str | None


def translate(concept_id: str, code: Any, path: str | Path = REGISTRY) -> Decoded:
    """Resolve one code against one concept. The record-level authority."""
    spec = concept(concept_id, path)
    if code is None:
        return Decoded(None, "missing", None)
    raw = str(code).strip()
    if raw.upper() in BLANK:
        return Decoded(None, "missing", raw or None)
    values = {str(k): v for k, v in (spec.get("values") or {}).items()}
    unknown = {str(u) for u in (spec.get("unknown") or [])}
    # Codes are compared both verbatim and zero-stripped: SINAN writes "01"
    # where the dictionary says "1", and SIH writes "1" where CNES says "01".
    for candidate in (raw, raw.lstrip("0") or "0", raw.zfill(2)):
        if candidate in values:
            return Decoded(str(values[candidate]), "valid", raw)
        if candidate in unknown:
            return Decoded(None, "unknown", raw)
    return Decoded(None, "invalid", raw)


def categorical_exprs(
    concept_id: str,
    column: str,
    *,
    suffix: str | None = None,
    path: str | Path = REGISTRY,
) -> list[pl.Expr]:
    """Vectorised twin of :func:`translate`.

    Returns two expressions: ``<name>`` carrying the canonical value and
    ``<name>_state`` carrying the decode state. Both are always produced, so a
    downstream consumer cannot use the value without the state being available
    to it.
    """
    spec = concept(concept_id, path)
    name = suffix or column.lower()
    values = {str(k): str(v) for k, v in (spec.get("values") or {}).items()}
    unknown = [str(u) for u in (spec.get("unknown") or [])]
    attributes = spec.get("attributes") or {}

    raw = pl.col(column).cast(pl.Utf8).str.strip_chars()
    # Match verbatim and zero-stripped, mirroring translate(). Polars uses the
    # Rust regex engine, which has no look-around, so the leading zeros are
    # stripped with a string op and "000" is mapped back to "0" explicitly
    # rather than to the empty string.
    _s = raw.str.strip_chars_start("0")
    stripped = pl.when(_s == "").then(pl.lit("0")).otherwise(_s)
    is_blank = raw.is_null() | raw.str.to_uppercase().is_in(list(BLANK))

    mapped = (
        pl.coalesce(
            raw.replace_strict(values, default=None, return_dtype=pl.Utf8),
            stripped.replace_strict(values, default=None, return_dtype=pl.Utf8),
        )
    )
    is_unknown = raw.is_in(unknown) | stripped.is_in(unknown) if unknown else pl.lit(False)

    value_expr = (
        pl.when(is_blank).then(None)
        .when(mapped.is_not_null()).then(mapped)
        .otherwise(None)
        .alias(name)
    )
    state_expr = (
        pl.when(is_blank).then(pl.lit("missing"))
        .when(mapped.is_not_null()).then(pl.lit("valid"))
        .when(is_unknown).then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid"))
        .alias(f"{name}_state")
    )
    out = [value_expr, state_expr]
    # Derived attribute columns. Many DATASUS categoricals carry a natural
    # grouping that the analysis wants and the code does not state: a
    # leptospirosis serovar implies a serogroup and a maintenance host, a CID
    # code implies a chapter. Emitting them here keeps the grouping in the
    # codebook -- one reviewable place -- instead of in a dozen study scripts
    # that will disagree with each other.
    for attr, mapping in attributes.items():
        m = {str(k): str(v) for k, v in mapping.items()}
        attr_mapped = pl.coalesce(
            raw.replace_strict(m, default=None, return_dtype=pl.Utf8),
            stripped.replace_strict(m, default=None, return_dtype=pl.Utf8),
        )
        out.append(
            pl.when(is_blank).then(None).otherwise(attr_mapped)
            .alias(f"{name}_{attr}")
        )
    return out


def decode_frame(
    frame: pl.DataFrame,
    system: str,
    *,
    fields: Iterable[str] | None = None,
    keep_raw: bool = True,
    path: str | Path = REGISTRY,
) -> pl.DataFrame:
    """Decode every bound categorical column of ``frame`` for ``system``.

    Columns absent from the frame are skipped silently -- a system's field set
    varies by year -- but columns present and *unbound* are left untouched and
    reported by :func:`coverage_report`, which is where the gap becomes visible.
    """
    bindings = concepts_for_system(system, path)
    if fields is not None:
        wanted = set(fields)
        bindings = {k: v for k, v in bindings.items() if k in wanted}
    exprs: list[pl.Expr] = []
    for field, concept_id in bindings.items():
        if field not in frame.columns:
            continue
        exprs.extend(categorical_exprs(concept_id, field, path=path))

    # Transforms: the coded columns an enum cannot reach -- packed ages,
    # epidemiological weeks, self-labelling serovars, municipality codes.
    # See brepi.codebook.transforms for why each is not an enumeration.
    from . import transforms as _tf
    tr = (load_codebook(path).get("transforms") or {}).get(system, {})
    if fields is not None:
        tr = {k: v for k, v in tr.items() if k in set(fields)}
    refs = None
    if any(s.get("kind") == "reference" for s in tr.values()):
        from .references import load_references
        refs = load_references()
    for field, spec in tr.items():
        if field not in frame.columns:
            continue
        exprs.extend(_tf.build(spec["kind"], field, spec, references=refs))

    if not exprs:
        return frame
    out = frame.with_columns(exprs)
    if not keep_raw:
        drop = [f for f in bindings if f in frame.columns]
        drop += [f for f in tr if f in frame.columns]
        out = out.drop(drop)
    return out


def coverage_report(
    frame: pl.DataFrame, system: str, *, path: str | Path = REGISTRY
) -> pl.DataFrame:
    """Which columns of a real file are translated, and which are not.

    This is the honest answer to "do we understand this dataset". A column that
    is categorical in the source and absent from the bindings is an untranslated
    code sitting in the analysis, and it should appear here rather than be
    discovered later.
    """
    doc = load_codebook(path)
    bindings = concepts_for_system(system, path)
    lookups = (doc["lookups"].get(system) or {})
    transforms = (doc.get("transforms") or {}).get(system, {})
    ignored = (doc.get("ignored") or {}).get(system, {})
    rows = []
    for col in frame.columns:
        concept_id = bindings.get(col)
        lookup = lookups.get(col)
        tr = transforms.get(col)
        reason = ignored.get(col)
        n_distinct = frame[col].n_unique()
        # A date column is coded in the trivial sense only; calling it UNBOUND
        # buries the columns that genuinely are.
        is_date = col.startswith("DT") or "_DT_" in col or col.startswith("DTM")
        status = (
            "coded" if concept_id
            else tr["kind"] if tr
            else "lookup" if lookup
            else "ignored" if reason
            else "date" if is_date
            else "UNBOUND"
        )
        rows.append({
            "column": col,
            "status": status,
            "concept": concept_id or (tr or {}).get("name") or lookup or None,
            "n_distinct": n_distinct,
            # An UNBOUND column with no stated reason is the only unaccounted
            # state. Everything else is either translated or deliberately not.
            "reason": reason,
        })
    return pl.DataFrame(rows).sort(
        [pl.col("status") == "UNBOUND", "n_distinct"], descending=[True, True]
    )
