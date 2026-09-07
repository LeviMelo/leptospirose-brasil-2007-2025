"""Coarsening a panel without inventing quantities.

Every panel analysis eventually asks for a coarser grain: municipality-month to
municipality-year, municipality to health region, monthly to quarterly. The
operation looks trivial and is not, because the columns of a panel are not all
the same kind of thing and only one kind may be summed.

The failure this module exists to prevent is not hypothetical. In this study's
RQ2, a monthly outcome ``asinh(cases)`` was coarsened to quarters by summing it.
The sum of ``asinh`` over three months is not ``asinh`` of the quarterly count
-- it is not any quantity at all -- and the resulting event-study estimate had
to be discarded and refitted. The rule that was violated is simple enough to
state in one line and easy enough to break that stating it is not sufficient:

    **Sum the primitives, then transform. Never sum a transform.**

So the contract is enforced rather than documented. A column whose name marks it
as a rate, a share, a ratio, a z-score, a log or an ``asinh`` cannot be placed
in ``sums``; the call raises and names the alternative. Derived quantities are
declared separately in ``derived`` and are evaluated *after* the aggregation, on
the summed primitives, which is the only order that yields the quantity the
column name claims.

``constants`` carries the other half of the contract: columns that are supposed
to be invariant within a group (an annual covariate broadcast to months, a
geography label). Taking ``first`` of a column that turns out to vary is a
silent averaging-by-accident; here it is checked and reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

__all__ = ["NON_ADDITIVE", "is_non_additive", "AggregationReport", "aggregate_panel"]


# Name fragments that mark a column as a quantity whose sum is meaningless. This
# is a heuristic on names, and deliberately so: the alternative is to trust that
# whoever writes the aggregation remembers, and that is what failed.
NON_ADDITIVE: tuple[str, ...] = (
    "_share", "share_", "_rate", "rate_", "_ratio", "ratio_", "_pct", "pct_",
    "_per_100k", "per_100k", "_per_capita", "per_capita", "_asinh", "asinh_",
    "_log", "log_", "_z", "_zscore", "_mean", "mean_", "_median", "median_",
    "_index", "_prop", "prop_", "_frequency", "incidence", "prevalence",
    "_fraction", "fraction_", "_pc",
)


def is_non_additive(name: str) -> bool:
    """Whether a column name marks a quantity that must not be summed."""
    n = name.lower()
    return any(frag in n for frag in NON_ADDITIVE) or bool(
        re.fullmatch(r".*_(z|sd|se|cv)", n))


@dataclass
class AggregationReport:
    """What the aggregation did, and where a declared constant was not one."""

    keys: list[str]
    rows_in: int
    rows_out: int
    summed: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    varying_constants: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.varying_constants

    def to_dict(self) -> dict[str, Any]:
        return {"keys": self.keys, "rows_in": self.rows_in,
                "rows_out": self.rows_out, "summed": self.summed,
                "constants": self.constants, "derived": self.derived,
                "varying_constants": self.varying_constants,
                "passed": self.passed}

    def format(self) -> str:
        out = [f"aggregate {self.rows_in} -> {self.rows_out} row(s) "
               f"by {self.keys}",
               f"  summed:    {len(self.summed)} column(s)",
               f"  constant:  {len(self.constants)} column(s)",
               f"  derived:   {len(self.derived)} column(s) (after aggregation)"]
        for c, n in self.varying_constants.items():
            out.append(f"  PROBLEM: {c!r} declared constant but varies within "
                       f"{n} group(s); 'first' would silently pick one value")
        if self.passed:
            out.append("  ok")
        return "\n".join(out)


def aggregate_panel(
    panel: pl.DataFrame | pl.LazyFrame,
    *,
    keys: Sequence[str],
    sums: Iterable[str] = (),
    constants: Iterable[str] = (),
    maxima: Iterable[str] = (),
    minima: Iterable[str] = (),
    weighted_means: Mapping[str, str] | None = None,
    derived: Mapping[str, pl.Expr] | None = None,
    extra: Mapping[str, pl.Expr] | None = None,
    check_constants: bool = True,
    allow_non_additive_sums: Sequence[str] = (),
) -> tuple[pl.DataFrame, AggregationReport]:
    """Collapse ``panel`` to ``keys``, summing only what may be summed.

    ``sums``
        Additive primitives: counts, person-time, event totals.
    ``constants``
        Invariant within a group. Verified unless ``check_constants=False``.
    ``weighted_means``
        ``{column: weight_column}``. A population-weighted mean of a share is a
        legitimate quantity; an unweighted mean of one usually is not, which is
        why the weight is required rather than optional.
    ``derived``
        ``{name: expr}`` evaluated **after** the aggregation, so a rate is built
        from summed numerator and summed denominator. This is where every
        transform belongs.
    ``extra``
        Escape hatch for aggregations this signature does not name (medians,
        counts of a condition). Expressions are evaluated inside the group-by.

    ``allow_non_additive_sums`` overrides the name check for the rare column
    that looks non-additive and is not -- ``n_rate_limited``, say. Naming it
    explicitly is the point: the override is visible in the call.
    """
    lf = panel.lazy() if isinstance(panel, pl.DataFrame) else panel
    sums, constants = list(sums), list(constants)
    maxima, minima = list(maxima), list(minima)
    weighted_means = dict(weighted_means or {})
    derived = dict(derived or {})
    extra = dict(extra or {})
    allowed = set(allow_non_additive_sums)

    bad = [c for c in sums if is_non_additive(c) and c not in allowed]
    if bad:
        raise ValueError(
            f"refusing to sum non-additive column(s) {bad}. The sum of a rate, "
            "share, z-score, log or asinh is not that quantity at a coarser "
            "grain -- it is not any quantity. Sum the primitives it was built "
            "from and rebuild it in derived=, or, if the name is misleading, "
            f"pass allow_non_additive_sums={bad!r}.")

    schema = lf.collect_schema().names()
    declared = [*keys, *sums, *constants, *maxima, *minima,
                *weighted_means, *weighted_means.values()]
    missing = [c for c in declared if c not in schema]
    if missing:
        raise KeyError(f"column(s) absent from the panel: {sorted(set(missing))}")

    aggs: list[pl.Expr] = [pl.col(c).sum() for c in sums]
    aggs += [pl.col(c).first() for c in constants]
    aggs += [pl.col(c).max().alias(f"{c}_max") for c in maxima]
    aggs += [pl.col(c).min().alias(f"{c}_min") for c in minima]
    for col, w in weighted_means.items():
        aggs.append(((pl.col(col) * pl.col(w)).sum() / pl.col(w).sum()).alias(col))
    aggs += [e.alias(n) for n, e in extra.items()]
    if check_constants:
        aggs += [pl.col(c).n_unique().alias(f"__nu_{c}") for c in constants]

    out = lf.group_by(list(keys)).agg(aggs).sort(list(keys)).collect()

    varying: dict[str, int] = {}
    if check_constants:
        for c in constants:
            n = int((out[f"__nu_{c}"] > 1).sum())
            if n:
                varying[c] = n
        out = out.drop([f"__nu_{c}" for c in constants])

    if derived:
        out = out.with_columns([e.alias(n) for n, e in derived.items()])

    rows_in = (panel.height if isinstance(panel, pl.DataFrame)
               else int(lf.select(pl.len()).collect().item()))
    report = AggregationReport(
        keys=list(keys), rows_in=rows_in, rows_out=out.height,
        summed=sums, constants=constants, derived=list(derived),
        varying_constants=varying)
    return out, report
