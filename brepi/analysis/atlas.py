"""Assembling one wide, join-ready table over a fixed spine of units.

A study finishes with its answers scattered: incidence in one table, a
surveillance classification in another, a latent field in a third, structural
covariates in the panel. Every one of them keys on the same unit. Nobody can
plot anything without first performing four joins by hand, and every hand-rolled
join is a chance to lose rows silently.

This module performs those joins once, declaratively, and -- more importantly --
**audits** them. Three failures it exists to catch, all of which happened in
this project:

* **Key dtype drift.** The canonical panel stores ``munic_code`` as a
  zero-padded string; a result CSV round-trips it through integer inference and
  loses the leading zero. Both are "the municipality code". Joined, they match
  nothing. ``key_width`` normalises every layer to one representation before
  any join is attempted.
* **Silent row loss.** A left join drops layer rows whose key is absent from the
  spine without a word. If a layer contributes 497 rows and only 480 land, that
  is a finding, not a detail; it is reported as ``layer_only``.
* **Column collision.** Two layers both carrying ``name`` produce ``name`` and
  ``name_right``, and whichever a downstream script reads is luck. A collision
  is an error unless the layer declares a prefix or an explicit rename.

Nothing here knows what a municipality is. It takes a spine, layers, and a key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

__all__ = ["Layer", "LayerReport", "AtlasReport", "assemble_atlas", "normalise_key"]


def _load(src: Any) -> pl.DataFrame:
    if isinstance(src, pl.DataFrame):
        return src
    if isinstance(src, pl.LazyFrame):
        return src.collect()
    p = Path(src)
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pl.read_csv(p, infer_schema_length=None)
    raise ValueError(f"cannot load {p} -- expected .parquet or .csv")


def normalise_key(df: pl.DataFrame, key: str, *, width: int | None) -> pl.DataFrame:
    """Cast ``key`` to a string and, if ``width`` is given, zero-pad it.

    This is the single point at which "the code as an integer" and "the code as
    a string" are reconciled. Doing it per-join instead is how one join in five
    gets forgotten.
    """
    if key not in df.columns:
        raise KeyError(f"key {key!r} absent; columns are {df.columns}")
    e = pl.col(key).cast(pl.Utf8, strict=False)
    if width is not None:
        e = e.str.zfill(width)
    return df.with_columns(e)


@dataclass
class Layer:
    """One contributing table."""

    name: str
    source: Any
    columns: Sequence[str] | None = None
    rename: dict[str, str] | None = None
    prefix: str | None = None
    #: When true, every spine key must be present in this layer.
    complete: bool = False


@dataclass
class LayerReport:
    name: str
    rows: int
    columns_added: list[str]
    matched: int
    spine_only: int
    layer_only: int
    layer_only_examples: list[str]
    duplicate_keys: int

    def to_dict(self) -> dict[str, Any]:
        return dict(
            layer=self.name, rows=self.rows,
            columns_added=self.columns_added,
            matched_keys=self.matched,
            spine_keys_unmatched=self.spine_only,
            layer_keys_not_in_spine=self.layer_only,
            layer_only_examples=self.layer_only_examples,
            duplicate_keys_in_layer=self.duplicate_keys,
        )


@dataclass
class AtlasReport:
    key: str
    spine_rows: int
    layers: list[LayerReport] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "spine_rows": self.spine_rows,
                "layers": [l.to_dict() for l in self.layers],
                "problems": self.problems, "passed": self.passed}

    def format(self) -> str:
        out = [f"atlas on {self.key!r}: {self.spine_rows} spine row(s), "
               f"{len(self.layers)} layer(s)"]
        for l in self.layers:
            out.append(
                f"  {l.name:<28} rows={l.rows:<7} matched={l.matched:<7} "
                f"+{len(l.columns_added)} col(s)")
            if l.layer_only:
                out.append(f"      {l.layer_only} key(s) NOT IN SPINE -- dropped: "
                           f"{l.layer_only_examples}")
            if l.duplicate_keys:
                out.append(f"      {l.duplicate_keys} duplicate key(s) -- "
                           "the join would multiply spine rows")
        for p in self.problems:
            out.append(f"  PROBLEM: {p}")
        if self.passed:
            out.append("  ok")
        return "\n".join(out)


def assemble_atlas(
    spine: Any,
    layers: Iterable[Layer],
    *,
    key: str,
    key_width: int | None = None,
    strict: bool = True,
) -> tuple[pl.DataFrame, AtlasReport]:
    """Left-join every layer onto ``spine`` and report what happened.

    The spine fixes the row set: the atlas has exactly the spine's units, in the
    spine's order, whatever the layers contain. That is what makes the result
    safe to join further -- a table whose row count depends on which analyses
    happened to finish is not a spine.

    With ``strict``, any of the following raises rather than being reported: a
    column collision, a duplicated key inside a layer (which would multiply
    spine rows), or a ``complete=True`` layer that fails to cover the spine.
    """
    atlas = normalise_key(_load(spine), key, width=key_width)
    if atlas[key].n_unique() != atlas.height:
        raise ValueError(
            f"spine key {key!r} is not unique ({atlas.height} rows, "
            f"{atlas[key].n_unique()} distinct) -- a spine must fix the row set")
    report = AtlasReport(key=key, spine_rows=atlas.height)
    spine_keys = set(atlas[key].to_list())

    for layer in layers:
        df = normalise_key(_load(layer.source), key, width=key_width)

        keep = list(layer.columns) if layer.columns is not None else [
            c for c in df.columns if c != key]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"layer {layer.name!r} lacks column(s) {missing}; "
                           f"it has {df.columns}")
        df = df.select([key, *keep])

        if layer.rename:
            df = df.rename({k: v for k, v in layer.rename.items() if k != key})
        if layer.prefix:
            df = df.rename({c: f"{layer.prefix}{c}" for c in df.columns if c != key})

        added = [c for c in df.columns if c != key]
        collisions = [c for c in added if c in atlas.columns]
        if collisions:
            msg = (f"layer {layer.name!r} would overwrite existing column(s) "
                   f"{collisions}; give it a prefix= or rename=")
            if strict:
                raise ValueError(msg)
            report.problems.append(msg)
            df = df.drop(collisions)
            added = [c for c in added if c not in collisions]

        dup = df.height - df[key].n_unique()
        layer_keys = set(df[key].to_list())
        only = sorted(layer_keys - spine_keys)
        rep = LayerReport(
            name=layer.name, rows=df.height, columns_added=added,
            matched=len(layer_keys & spine_keys),
            spine_only=len(spine_keys - layer_keys),
            layer_only=len(only), layer_only_examples=only[:8],
            duplicate_keys=dup,
        )
        report.layers.append(rep)

        if dup:
            msg = (f"layer {layer.name!r} has {dup} duplicated key(s); joining "
                   "it would multiply spine rows. Aggregate it first.")
            if strict:
                raise ValueError(msg)
            report.problems.append(msg)
            df = df.unique(subset=[key], keep="first")
        if layer.complete and rep.spine_only:
            msg = (f"layer {layer.name!r} is declared complete but misses "
                   f"{rep.spine_only} spine key(s)")
            if strict:
                raise ValueError(msg)
            report.problems.append(msg)

        atlas = atlas.join(df, on=key, how="left")

    return atlas, report
