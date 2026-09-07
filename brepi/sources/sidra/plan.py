"""Request planning across SIDRA cell-count and URL-size ceilings.

SIDRA has a documented per-request ceiling of roughly 50,000 cells. Exceeding it
does not produce a clean 4xx; it produces an HTTP 500, which is indistinguishable
from a transient server fault and will therefore be retried four times before
failing -- slowly, and for a reason nobody reads. So the ceiling is enforced
*here*, before anything touches the network, and a plan that cannot fit is a
deterministic error rather than a retry.

The budget is deliberately below the documented ceiling. The ceiling is not
contractual and the response size, not just the cell count, appears to matter.

The API encodes every requested locality in the URL. A request can fit the cell
ceiling and still be blocked by IBGE's web-application firewall because a
5,570-code query string is tens of kilobytes long. Localities are therefore
bounded independently. Classifications are never split because doing so risks
losing the totals needed for a margin check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from brepi.sources.sidra.api import (
    SidraError,
    locality_expr,
    normalise_classifications,
    query_hash,
)

#: Documented ceiling, for reference. Never used as a budget.
SIDRA_DOCUMENTED_CEILING = 50_000

#: What we actually plan against.
CELL_BUDGET = 45_000

#: Seven-digit codes plus percent-encoded separators make about 10 URL bytes
#: each. Four hundred stays well below common 8 KiB proxy/WAF limits after the
#: classifications and endpoint are added.
MAX_LOCALITIES_PER_REQUEST = 400


class PlanTooLarge(SidraError):
    """A request cannot be made to fit the cell budget by splitting.

    Deterministic. Do not retry; change the selection.
    """


def estimate_cells(
    n_periods: int,
    n_variables: int,
    n_localities: int,
    classification_sizes: Sequence[int] | Mapping[Any, int] | None = None,
) -> int:
    """Cells a request would return.

    ``n_periods * n_variables * n_localities * prod(categories per
    classification)``. The variable factor is included -- wrapper documentation
    routinely omits it because it discusses the single-variable case, and a
    planner that copies that formula under-counts by exactly the number of
    variables requested.
    """
    for name, value in (
        ("n_periods", n_periods),
        ("n_variables", n_variables),
        ("n_localities", n_localities),
    ):
        if value < 1:
            raise SidraError(f"{name} must be >= 1, got {value}")
    sizes: Iterable[int]
    if classification_sizes is None:
        sizes = ()
    elif isinstance(classification_sizes, Mapping):
        sizes = classification_sizes.values()
    else:
        sizes = classification_sizes
    total = int(n_periods) * int(n_variables) * int(n_localities)
    for size in sizes:
        if size < 1:
            raise SidraError(f"classification with {size} categories cannot be requested")
        total *= int(size)
    return total


@dataclass(frozen=True)
class RequestSpec:
    """What the caller wants, before it is cut into executable requests."""

    agregado: int
    periods: tuple[str, ...]
    variables: tuple[str, ...]
    level: str = "N6"
    #: Explicit locality ids. Resolve ``"all"`` against
    #: :func:`brepi.sources.sidra.api.get_localities` before planning -- the
    #: planner cannot size an unresolved selection.
    localities: tuple[str, ...] = ()
    classifications: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    view: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "periods", tuple(str(p) for p in self.periods))
        object.__setattr__(self, "variables", tuple(str(v) for v in self.variables))
        object.__setattr__(self, "localities", tuple(str(loc) for loc in self.localities))
        object.__setattr__(
            self, "classifications", normalise_classifications(self.classifications)
        )
        if not self.periods:
            raise SidraError("RequestSpec needs at least one period")
        if not self.variables:
            raise SidraError("RequestSpec needs at least one variable")
        if not self.localities:
            raise SidraError(
                "RequestSpec needs explicit localities; resolve 'all' via "
                "api.get_localities so the plan can be sized"
            )

    @property
    def classification_sizes(self) -> dict[str, int]:
        return {clf: len(cats) for clf, cats in self.classifications.items()}

    def total_cells(self) -> int:
        return estimate_cells(
            len(self.periods),
            len(self.variables),
            len(self.localities),
            self.classification_sizes,
        )


@dataclass(frozen=True)
class Request:
    """One executable SIDRA values call, already checked against the budget."""

    agregado: int
    periods: tuple[str, ...]
    variables: tuple[str, ...]
    level: str
    localities: tuple[str, ...]
    classifications: Mapping[str, tuple[str, ...]]
    estimated_cells: int
    view: str | None = None

    @property
    def locality_expression(self) -> str:
        return locality_expr(self.level, self.localities)

    def identity(self) -> str:
        """Stable digest of this request, for logging and plan diffing."""
        return query_hash(
            {
                "agregado": self.agregado,
                "periods": sorted(self.periods),
                "variables": sorted(self.variables),
                "level": self.level,
                "localities": sorted(self.localities),
                "classifications": {k: list(v) for k, v in self.classifications.items()},
                "view": self.view,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agregado": self.agregado,
            "periods": list(self.periods),
            "variables": list(self.variables),
            "level": self.level,
            "n_localities": len(self.localities),
            "classifications": {k: list(v) for k, v in self.classifications.items()},
            "estimated_cells": self.estimated_cells,
            "view": self.view,
            "identity": self.identity(),
        }


def _chunk(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size < 1:
        raise SidraError("chunk size must be >= 1")
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def plan_requests(
    spec: RequestSpec,
    *,
    budget: int = CELL_BUDGET,
    max_localities: int = MAX_LOCALITIES_PER_REQUEST,
) -> list[Request]:
    """Cut ``spec`` into requests that fit cell and locality/URL budgets.

    Strategy:

    Localities are chunked first because the URL limit is independent of cells.
    Within each locality chunk, as many periods as fit the cell budget are
    packed together. A single locality-period whose classification cube is
    over budget is a deterministic specification error.
    """
    if budget < 1:
        raise SidraError("budget must be >= 1")
    if max_localities < 1:
        raise SidraError("max_localities must be >= 1")

    n_var = len(spec.variables)
    sizes = spec.classification_sizes
    cat_product = estimate_cells(1, n_var, 1, sizes)

    if cat_product > budget:
        raise PlanTooLarge(
            f"agregado {spec.agregado}: a single period for a single locality needs "
            f"{cat_product} cells, over the {budget}-cell budget. Reduce the "
            f"variable or category selection ({n_var} variables, sizes {sizes})."
        )
    localities_per_request = min(max_localities, max(1, budget // cat_product))
    requests: list[Request] = []
    for locality_chunk in _chunk(spec.localities, localities_per_request):
        per_period = cat_product * len(locality_chunk)
        periods_per_request = max(1, budget // per_period)
        for period_chunk in _chunk(spec.periods, periods_per_request):
            requests.append(
                _make(spec, period_chunk, locality_chunk, budget)
            )
    return requests


def _make(
    spec: RequestSpec,
    periods: tuple[str, ...],
    localities: tuple[str, ...],
    budget: int,
) -> Request:
    """Build a request and refuse to return one that is over budget.

    This is the gate referred to in SIDRA_DESC: a request is only executed after
    its planned cell count has been checked, and the check lives at construction
    so no code path can skip it.
    """
    cells = estimate_cells(
        len(periods), len(spec.variables), len(localities), spec.classification_sizes
    )
    if cells > budget:
        raise PlanTooLarge(
            f"agregado {spec.agregado}: planned request of {cells} cells exceeds the "
            f"{budget}-cell budget ({len(periods)} periods x {len(spec.variables)} "
            f"variables x {len(localities)} localities x categories "
            f"{spec.classification_sizes})"
        )
    return Request(
        agregado=spec.agregado,
        periods=periods,
        variables=spec.variables,
        level=spec.level,
        localities=localities,
        classifications=spec.classifications,
        estimated_cells=cells,
        view=spec.view,
    )


def plan_summary(requests: Sequence[Request]) -> dict[str, Any]:
    """Aggregate report over a plan, for the run log."""
    cells = [r.estimated_cells for r in requests]
    return {
        "n_requests": len(requests),
        "total_cells": sum(cells),
        "max_cells": max(cells, default=0),
        "min_cells": min(cells, default=0),
        "mean_cells": (sum(cells) / len(cells)) if cells else 0.0,
        "budget": CELL_BUDGET,
        "max_localities": max(
            (len(request.localities) for request in requests), default=0
        ),
    }


def n_chunks_needed(total_cells: int, budget: int = CELL_BUDGET) -> int:
    """Lower bound on the number of requests a workload will take."""
    return max(1, math.ceil(total_cells / budget))


__all__ = [
    "CELL_BUDGET",
    "MAX_LOCALITIES_PER_REQUEST",
    "SIDRA_DOCUMENTED_CEILING",
    "PlanTooLarge",
    "Request",
    "RequestSpec",
    "estimate_cells",
    "plan_requests",
    "plan_summary",
    "n_chunks_needed",
]
