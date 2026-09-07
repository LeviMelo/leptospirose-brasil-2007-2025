"""The panel spine: the complete municipality x month lattice everything joins onto.

The spine is built first and independently of any data, then every source is
LEFT-joined onto it. That ordering is the whole point. If outcomes are joined
to covariates instead, municipality-months with no cases silently vanish, and
in this study those cells are not absence of data — they are the object of
inquiry. Measured on the 2007-2025 leptospirosis series, 97.2% of the spine is
zero and 2,094 of 5,570 municipalities never report a confirmed case at all.
Whether those are transmission-silent or detection-silent is a research
question, and it cannot even be posed on a panel that dropped them.

Every join records how many spine rows it matched, so an unnoticed key
mismatch (the classic 6-digit-versus-7-digit IBGE code error) shows up as a
coverage number rather than as a quietly wrong result four steps later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Sequence

import polars as pl

TimeGrain = Literal["month", "year"]


@dataclass(frozen=True)
class SpineSpec:
    """Definition of the analytic lattice."""

    municipalities: tuple[str, ...]  # 7-digit IBGE codes, the 2022 lattice
    start: date
    end: date
    grain: TimeGrain = "month"

    @property
    def n_periods(self) -> int:
        if self.grain == "year":
            return self.end.year - self.start.year + 1
        return (self.end.year - self.start.year) * 12 + (self.end.month - self.start.month) + 1

    @property
    def n_rows(self) -> int:
        return len(self.municipalities) * self.n_periods


@dataclass
class JoinReport:
    """Coverage evidence for one join onto the spine."""

    source: str
    columns: tuple[str, ...]
    spine_rows: int
    matched_rows: int
    unmatched_source_keys: int
    duplicate_source_keys: int

    @property
    def coverage(self) -> float:
        return self.matched_rows / self.spine_rows if self.spine_rows else 0.0

    def summary(self) -> str:
        return (
            f"{self.source}: {self.matched_rows:,}/{self.spine_rows:,} spine rows "
            f"({self.coverage:.1%}); {self.unmatched_source_keys:,} source keys had no "
            f"spine row; {self.duplicate_source_keys:,} duplicate source keys"
        )


def build_spine(spec: SpineSpec) -> pl.DataFrame:
    """Materialise the complete lattice.

    Returns columns ``munic_code``, ``period`` (first day of the period),
    ``year``, and — at monthly grain — ``month`` and ``time_index``
    (0-based, for random-walk temporal terms in INLA).
    """
    if spec.end < spec.start:
        raise ValueError("spine end precedes start")
    if len(set(spec.municipalities)) != len(spec.municipalities):
        raise ValueError("duplicate municipality codes in spec")
    bad = [m for m in spec.municipalities if not (m.isdigit() and len(m) == 7)]
    if bad:
        raise ValueError(
            f"{len(bad)} municipality codes are not 7-digit IBGE codes, e.g. {bad[:3]}. "
            "DATASUS emits 6-digit codes; convert with brepi.geo.lattice.code6_to_code7 "
            "before building the spine."
        )

    if spec.grain == "year":
        periods = [date(y, 1, 1) for y in range(spec.start.year, spec.end.year + 1)]
    else:
        periods = []
        y, m = spec.start.year, spec.start.month
        while (y, m) <= (spec.end.year, spec.end.month):
            periods.append(date(y, m, 1))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    spine = pl.DataFrame({"munic_code": list(spec.municipalities)}).join(
        pl.DataFrame({"period": periods}), how="cross"
    )
    spine = spine.with_columns(
        pl.col("period").dt.year().alias("year").cast(pl.Int32),
        pl.col("munic_code").str.slice(0, 2).alias("uf_code"),
    )
    if spec.grain == "month":
        spine = spine.with_columns(pl.col("period").dt.month().alias("month").cast(pl.Int8))
    spine = spine.sort(["munic_code", "period"]).with_row_index("row_id")
    period_index = {p: i for i, p in enumerate(periods)}
    spine = spine.with_columns(
        pl.col("period").replace_strict(period_index, return_dtype=pl.Int32).alias("time_index")
    )
    if spine.height != spec.n_rows:
        raise AssertionError(f"spine has {spine.height} rows, expected {spec.n_rows}")
    return spine


def attach(
    spine: pl.DataFrame,
    source: pl.DataFrame,
    *,
    name: str,
    on: Sequence[str] = ("munic_code", "period"),
    columns: Sequence[str] | None = None,
    fill: dict[str, object] | None = None,
) -> tuple[pl.DataFrame, JoinReport]:
    """Left-join ``source`` onto the spine and report coverage.

    ``fill`` gives per-column defaults for unmatched spine rows. Use it only
    where a missing row genuinely means a known value: case counts fill with
    0 because the spine enumerates all municipality-months and an absent row
    means no notification. Covariates must NOT be filled with 0 — leave them
    null so that missingness stays typed and visible to the model.
    """
    missing_keys = [k for k in on if k not in source.columns]
    if missing_keys:
        raise ValueError(f"{name}: source lacks join keys {missing_keys}")

    keep = list(on) + [c for c in (columns or [c for c in source.columns if c not in on])]
    src = source.select(keep)

    dupes = src.height - src.select(on).unique().height
    if dupes:
        raise ValueError(
            f"{name}: {dupes} duplicate rows on {list(on)}. Aggregate the source to the "
            "spine grain before attaching; silently taking the first row is never right."
        )

    before = spine.height
    out = spine.join(src, on=list(on), how="left")
    if out.height != before:
        raise AssertionError(f"{name}: join changed spine height {before} -> {out.height}")

    value_cols = tuple(c for c in keep if c not in on)
    probe = value_cols[0] if value_cols else on[0]
    matched = int(out.select(pl.col(probe).is_not_null().sum()).item()) if value_cols else before

    spine_keys = spine.select(on).unique()
    unmatched = src.join(spine_keys, on=list(on), how="anti").height

    if fill:
        out = out.with_columns(
            [pl.col(c).fill_null(v) for c, v in fill.items() if c in out.columns]
        )

    report = JoinReport(
        source=name,
        columns=value_cols,
        spine_rows=before,
        matched_rows=matched,
        unmatched_source_keys=unmatched,
        duplicate_source_keys=dupes,
    )
    return out, report


@dataclass
class PanelBuild:
    """Accumulates a panel and the audit trail of how it was assembled."""

    spine: pl.DataFrame
    reports: list[JoinReport] = field(default_factory=list)

    def add(
        self,
        source: pl.DataFrame,
        *,
        name: str,
        on: Sequence[str] = ("munic_code", "period"),
        columns: Sequence[str] | None = None,
        fill: dict[str, object] | None = None,
        min_coverage: float | None = None,
    ) -> "PanelBuild":
        self.spine, rep = attach(
            self.spine, source, name=name, on=on, columns=columns, fill=fill
        )
        self.reports.append(rep)
        if min_coverage is not None and rep.coverage < min_coverage:
            raise ValueError(
                f"{name}: coverage {rep.coverage:.1%} below required {min_coverage:.1%}. "
                f"{rep.unmatched_source_keys:,} source keys matched no spine row, which "
                "usually means a municipality-code convention mismatch."
            )
        return self

    def audit(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "source": r.source,
                    "n_columns": len(r.columns),
                    "matched_rows": r.matched_rows,
                    "coverage": r.coverage,
                    "unmatched_source_keys": r.unmatched_source_keys,
                }
                for r in self.reports
            ]
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.spine.write_parquet(path)
        self.audit().write_csv(path.with_suffix(".audit.csv"))
        return path


def sparsity_report(panel: pl.DataFrame, *, count_col: str = "cases") -> dict[str, float | int]:
    """Quantify the zero structure. Belongs in the paper, not in a log file.

    Zero-inflation at this level is not a nuisance parameter to be absorbed;
    it is the quantity that the ascertainment strand of the study estimates.
    """
    if count_col not in panel.columns:
        raise ValueError(f"panel lacks {count_col!r}")
    n = panel.height
    nz = panel.filter(pl.col(count_col) > 0)
    per_mun = panel.group_by("munic_code").agg(pl.col(count_col).sum().alias("total"))
    active = per_mun.filter(pl.col("total") > 0)
    totals = active["total"].sort(descending=True)
    grand = int(totals.sum()) if totals.len() else 0

    def top_share(k: int) -> float:
        return float(totals.head(k).sum() / grand) if grand else 0.0

    return {
        "spine_rows": n,
        "nonzero_cells": nz.height,
        "nonzero_share": nz.height / n if n else 0.0,
        "total_count": grand,
        "municipalities_total": per_mun.height,
        "municipalities_ever_reporting": active.height,
        "municipalities_never_reporting": per_mun.height - active.height,
        "mean_count_per_nonzero_cell": float(nz[count_col].mean()) if nz.height else 0.0,
        "median_count_per_nonzero_cell": float(nz[count_col].median()) if nz.height else 0.0,
        "cells_ge_5": int(panel.filter(pl.col(count_col) >= 5).height),
        "case_share_top_50_municipalities": top_share(50),
        "case_share_top_500_municipalities": top_share(500),
    }


__all__ = [
    "SpineSpec",
    "JoinReport",
    "PanelBuild",
    "build_spine",
    "attach",
    "sparsity_report",
]
