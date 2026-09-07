"""Vocabulary consistency across analysis outputs.

A study writes dozens of result tables from a dozen scripts over several weeks.
Each one names its strata in whatever vocabulary its own upstream source
happened to use. Nothing checks that they agree, because nothing owns the
question -- and the failure is silent in the worst way: a join on the shared
column returns zero rows, an inner join drops every record, and a figure comes
out empty or, far worse, half-populated.

This module owns that question.

The concrete instance that motivated it: the canonical municipality-month panel
carried IBGE's Portuguese macro-region labels (``Norte``, ``Nordeste``,
``Sudeste``, ``Sul``, ``Centro-Oeste``) because they come from the geography
lattice, while the descriptive tables carried English labels (``North``,
``Northeast``, ...) because they come from the codebook's UF reference. Both are
individually correct. Joined on ``region``, they match nothing. The regime
assignment and the incidence table -- two datasets a reader would obviously
cross-tabulate -- could not be combined at all.

Two checks, because they catch different faults:

* **Disjoint vocabularies.** Two files use the same column name and share *no*
  values. Almost always a translation or an encoding difference; always fatal
  for a join.
* **Partial overlap.** Two files share some values but not all. Sometimes
  legitimate (a filtered subset), sometimes a silent rename of one level. It is
  reported separately so a genuine subset does not read as a failure.

Nothing here knows what a region is. It takes files, a column name, and returns
what the values actually are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import polars as pl

# Columns whose values are free text or identifiers rather than a controlled
# vocabulary; comparing them across files is noise, not signal.
DEFAULT_SKIP = {
    "name", "description", "note", "rationale", "error", "term", "item",
    "metric", "path", "source", "file", "label", "scope", "step", "field",
    "run_id", "id", "method", "aggregation", "column", "reason", "purpose",
}


@dataclass
class VocabularyReport:
    """What a controlled-vocabulary column actually contains, per file."""

    column: str
    values_by_file: dict[str, set[str]] = field(default_factory=dict)
    disjoint_pairs: list[tuple[str, str]] = field(default_factory=list)
    partial_pairs: list[tuple[str, str, int, int]] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.values_by_file)

    @property
    def all_values(self) -> set[str]:
        out: set[str] = set()
        for v in self.values_by_file.values():
            out |= v
        return out

    @property
    def passed(self) -> bool:
        """Disjoint vocabularies fail. Partial overlap is reported, not failed."""
        return not self.disjoint_pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "files": self.n_files,
            "distinct_values_overall": sorted(self.all_values),
            "values_by_file": {k: sorted(v) for k, v in self.values_by_file.items()},
            "disjoint_pairs": [list(p) for p in self.disjoint_pairs],
            "partial_overlap_pairs": [
                {"a": a, "b": b, "shared": s, "union": u}
                for a, b, s, u in self.partial_pairs
            ],
            "passed": self.passed,
        }


def _read_columns(path: Path) -> pl.DataFrame | None:
    """Read a result file defensively.

    Result CSVs routinely begin with integer-looking values and later contain
    floats, so schema inference on the default sample length raises rather than
    widening. Reading the whole file for inference costs nothing at these sizes
    and is the difference between a check that runs and one that throws.
    """
    try:
        if path.suffix.lower() == ".csv":
            return pl.read_csv(path, infer_schema_length=None,
                               truncate_ragged_lines=True)
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
    except Exception:
        return None
    return None


def collect_vocabulary(
    files: Iterable[Path],
    column: str,
    *,
    max_distinct: int = 60,
    root: Path | None = None,
) -> VocabularyReport:
    """Gather the distinct values of ``column`` wherever it appears.

    ``max_distinct`` guards against treating an identifier column as a
    vocabulary: a column with hundreds of distinct values in one file is a key,
    not a codelist, and comparing key sets across files is meaningless.
    """
    rep = VocabularyReport(column=column)
    for f in files:
        df = _read_columns(Path(f))
        if df is None or column not in df.columns:
            continue
        vals = df[column].cast(pl.Utf8, strict=False).drop_nulls().unique().to_list()
        if not vals or len(vals) > max_distinct:
            continue
        key = str(Path(f).relative_to(root)) if root else str(f)
        rep.values_by_file[key] = {str(v) for v in vals}

    names = sorted(rep.values_by_file)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = rep.values_by_file[a], rep.values_by_file[b]
            shared = va & vb
            if not shared:
                rep.disjoint_pairs.append((a, b))
            elif shared != va or shared != vb:
                rep.partial_pairs.append((a, b, len(shared), len(va | vb)))
    return rep


def check_vocabularies(
    root: Path,
    columns: Iterable[str],
    *,
    patterns: Iterable[str] = ("**/*.csv", "**/*.parquet"),
    max_distinct: int = 60,
) -> dict[str, VocabularyReport]:
    """Check every named column across every result file under ``root``."""
    files = sorted({p for pat in patterns for p in Path(root).glob(pat)
                    if p.is_file()})
    return {c: collect_vocabulary(files, c, max_distinct=max_distinct, root=root)
            for c in columns}


def format_report(reports: dict[str, VocabularyReport]) -> str:
    """A human-readable summary that names the offending file pairs."""
    lines: list[str] = []
    for col, rep in sorted(reports.items()):
        if rep.n_files == 0:
            continue
        status = "ok" if rep.passed else "FAIL"
        lines.append(f"[{status}] {col!r}: {rep.n_files} file(s), "
                     f"{len(rep.all_values)} distinct value(s) overall")
        if rep.disjoint_pairs:
            lines.append("   DISJOINT -- a join on this column returns nothing:")
            for a, b in rep.disjoint_pairs[:12]:
                lines.append(f"     {a}")
                lines.append(f"       {sorted(rep.values_by_file[a])}")
                lines.append(f"     {b}")
                lines.append(f"       {sorted(rep.values_by_file[b])}")
            if len(rep.disjoint_pairs) > 12:
                lines.append(f"     ... and {len(rep.disjoint_pairs) - 12} more pair(s)")
        if rep.partial_pairs:
            lines.append(f"   partial overlap in {len(rep.partial_pairs)} pair(s) "
                         "(may be a legitimate subset):")
            for a, b, s, u in rep.partial_pairs[:6]:
                lines.append(f"     {s}/{u} shared: {a} vs {b}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/results")
    ap.add_argument("--columns", nargs="+",
                    default=["region", "uf_abbr", "surveillance_class",
                             "reservoir", "serogroup", "sex", "age_group",
                             "regime", "phase", "level", "climate_product"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    reports = check_vocabularies(Path(a.root), a.columns)
    print(format_report(reports))
    failed = [c for c, r in reports.items() if not r.passed]
    if a.json:
        Path(a.json).write_text(
            json.dumps({c: r.to_dict() for c, r in reports.items()}, indent=2),
            encoding="utf-8")
    if failed:
        print(f"\n{len(failed)} column(s) carry disjoint vocabularies: "
              f"{', '.join(sorted(failed))}")
        print("A join on any of these silently returns nothing. Fix the source, "
              "not the join.")
        return 1
    print("\nno disjoint vocabularies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
