"""Machine-checkable assertions linking scientific claims to result tables.

Existence checks catch missing files but not a stale number copied into prose.
This module evaluates a deliberately small assertion language over CSV,
Parquet, and JSON artefacts so study claim ledgers can verify values as well as
paths without embedding study-specific logic in the harness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl


def _frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Evidence tables are compact and may begin with integer-looking rows
        # before later floating values. Full-column inference avoids a false
        # audit failure caused by Polars' short default inference window.
        return pl.read_csv(path, infer_schema_length=None)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"unsupported tabular evidence format: {path}")


def _close(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected),
                                abs_tol=atol, rel_tol=rtol)
        except (TypeError, ValueError):
            return False
    return actual == expected


def evaluate_assertion(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one declarative evidence assertion and return an audit row."""
    path = root / str(spec["path"])
    kind = str(spec["kind"])
    expected = spec["expected"]
    atol = float(spec.get("atol", 0.0))
    rtol = float(spec.get("rtol", 0.0))
    result: dict[str, Any] = {
        "id": spec["id"], "path": str(spec["path"]), "kind": kind,
        "expected": expected,
    }
    if not path.exists():
        return {**result, "passed": False, "actual": None,
                "error": "evidence path does not exist"}
    try:
        if kind == "json_value":
            value: Any = json.loads(path.read_text(encoding="utf-8"))
            for key in spec["keys"]:
                value = value[key]
            actual = value
        else:
            frame = _frame(path)
            where = spec.get("where", {})
            for column, value in where.items():
                frame = frame.filter(pl.col(column) == value)
            if kind == "row_count":
                actual = frame.height
            elif kind == "column_count":
                actual = frame.width
            elif kind == "column_max":
                actual = frame[spec["column"]].max()
            elif kind == "column_sum":
                # The reconciliation assertion: a derived table's total must
                # equal the source it was aggregated from. A join that quietly
                # drops units changes this number and nothing else.
                actual = frame[spec["column"]].sum()
            elif kind == "cell":
                if frame.height != 1:
                    raise ValueError(
                        f"cell assertion selected {frame.height} rows, expected 1")
                actual = frame[spec["column"]][0]
            elif kind == "correlation":
                columns: Sequence[str] = spec["columns"]
                x = pl.col(columns[0]).cast(pl.Float64)
                y = pl.col(columns[1]).cast(pl.Float64)
                if spec.get("transform") == "log":
                    x, y = x.log(), y.log()
                actual = frame.select(pl.corr(x, y)).item()
            else:
                raise ValueError(f"unknown evidence assertion kind: {kind}")
        return {**result, "passed": _close(actual, expected,
                                             atol=atol, rtol=rtol),
                "actual": actual, "error": None}
    except Exception as exc:  # audit records malformed evidence; it does not hide it
        return {**result, "passed": False, "actual": None,
                "error": f"{type(exc).__name__}: {exc}"}


def evaluate_assertions(root: Path, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate a collection and expose a single fail-closed status."""
    rows = [evaluate_assertion(root, spec) for spec in specs]
    return {"passed": all(row["passed"] for row in rows),
            "assertions_checked": len(rows), "results": rows}
