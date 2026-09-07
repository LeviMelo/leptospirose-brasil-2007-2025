"""High-level SIDRA extraction: plan, fetch, normalise, land, reconcile.

The pipeline is deliberately linear and each step refuses to be skipped:

    registry entry + selection
      -> resolve localities from live /localidades
      -> RequestSpec
      -> plan_requests  (cell ceiling enforced here)
      -> fetch_values   (cached, one call per planned request)
      -> concatenate long facts
      -> parquet in PATHS.interim + provenance sidecar

Margin reconciliation (:func:`check_margins`) is a required QA gate, not an
optional diagnostic: if the components of a classification do not sum to the
published Total, either the category selection is wrong, a Total category has
been mistaken for a member, or suppression has removed cells. All three produce
plausible-looking numbers, which is exactly why the check has to be mechanical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from brepi.config import PATHS
from brepi.io.cache import write_manifest
from brepi.sources.sidra import api, plan as planning, registry

TOOL_VERSION = "brepi.sources.sidra.extract/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """What to pull out of one registry table.

    ``localities=None`` means every locality the table offers at ``level``,
    resolved from live ``/localidades`` before planning -- the planner cannot
    size an unresolved selection and will refuse one.
    """

    agregado: int
    periods: Sequence[str]
    variables: Sequence[str]
    level: str = registry.PRIMARY_LEVEL
    localities: Sequence[str] | None = None
    classifications: Mapping[Any, Sequence[Any]] = field(default_factory=dict)
    view: str | None = None
    label: str | None = None

    def name(self) -> str:
        return self.label or f"sidra_{self.agregado}_{self.level}"


def resolve_localities(selection: Selection, *, refresh: bool = False) -> tuple[str, ...]:
    """Locality ids for a selection, from live metadata when not given."""
    if selection.localities is not None:
        return tuple(str(loc) for loc in selection.localities)
    localities = api.get_localities(selection.agregado, selection.level, refresh=refresh)
    return tuple(str(loc["id"]) for loc in localities)


def build_spec(selection: Selection, *, refresh: bool = False) -> planning.RequestSpec:
    """Turn a :class:`Selection` into a sizeable :class:`~plan.RequestSpec`."""
    return planning.RequestSpec(
        agregado=selection.agregado,
        periods=tuple(str(p) for p in selection.periods),
        variables=tuple(str(v) for v in selection.variables),
        level=selection.level,
        localities=resolve_localities(selection, refresh=refresh),
        classifications=api.normalise_classifications(selection.classifications),
        view=selection.view,
    )


def validate_selection(selection: Selection, *, refresh: bool = False) -> list[str]:
    """Check a selection against live metadata. Returns a list of problems.

    Per SIDRA_DESC section 11: every value request is validated before
    execution, because an empty or symbolic response can mean an invalid
    variable-period pair just as easily as a genuinely missing datum, and the
    two are indistinguishable after the fact.
    """
    problems: list[str] = []
    meta = api.get_metadata(selection.agregado, refresh=refresh)

    available_vars = {str(v["id"]) for v in meta.get("variaveis", [])}
    for var in selection.variables:
        if str(var) not in available_vars:
            problems.append(f"variable {var} not in agregado {selection.agregado}")

    levels = meta.get("nivelTerritorial") or {}
    offered = set(levels.get("Administrativo", [])) | set(levels.get("Especial", []))
    if selection.level not in offered:
        problems.append(
            f"level {selection.level} not offered by agregado {selection.agregado}; "
            f"offers {sorted(offered)}"
        )

    available_periods = {str(p["id"]) for p in api.get_periods(selection.agregado, refresh=refresh)}
    for period in selection.periods:
        if str(period) not in available_periods:
            problems.append(f"period {period} not published for agregado {selection.agregado}")

    by_clf = {str(c["id"]): c for c in meta.get("classificacoes", []) or []}
    for clf, cats in api.normalise_classifications(selection.classifications).items():
        if clf not in by_clf:
            problems.append(f"classification {clf} not in agregado {selection.agregado}")
            continue
        known = {str(c["id"]) for c in by_clf[clf].get("categorias", []) or []}
        for cat in cats:
            if cat not in known and cat not in {"all", "allxp"}:
                problems.append(f"category {cat} not in classification {clf}")
    return problems


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Everything a reviewer needs to judge one extraction."""

    facts: pl.DataFrame
    parquet_path: Path | None
    sidecar_path: Path | None
    plan: list[planning.Request]
    cache_keys: list[str]
    report: dict[str, Any]


def extract(
    selection: Selection,
    *,
    budget: int = planning.CELL_BUDGET,
    validate: bool = True,
    write: bool = True,
    refresh: bool = False,
    out_dir: Path | None = None,
) -> ExtractionResult:
    """Plan, fetch and land one selection as long facts.

    Raises before touching the network if ``validate`` is on and the selection
    contradicts live metadata, or if the plan cannot fit the cell budget.
    """
    if validate:
        problems = validate_selection(selection, refresh=refresh)
        if problems:
            raise api.SidraError(
                f"selection for agregado {selection.agregado} is invalid against live "
                f"metadata: " + "; ".join(problems)
            )

    spec = build_spec(selection, refresh=refresh)
    requests = planning.plan_requests(spec, budget=budget)

    frames: list[pl.DataFrame] = []
    cache_keys: list[str] = []
    for request in requests:
        cache_keys.append(
            api.values_cache_key(
                request.agregado,
                request.periods,
                request.variables,
                level=request.level,
                localities=request.localities,
                classifications=request.classifications,
                view=request.view,
            )
        )
        frames.append(
            api.fetch_values(
                request.agregado,
                request.periods,
                request.variables,
                level=request.level,
                localities=request.localities,
                classifications=request.classifications,
                view=request.view,
                refresh=refresh,
            )
        )

    facts = (
        pl.concat(frames, how="vertical")
        if frames
        else api.empty_facts()
    )

    report: dict[str, Any] = {
        "selection": {
            "agregado": selection.agregado,
            "label": selection.name(),
            "periods": list(spec.periods),
            "variables": list(spec.variables),
            "level": spec.level,
            "n_localities": len(spec.localities),
            "classifications": {k: list(v) for k, v in spec.classifications.items()},
        },
        "plan": planning.plan_summary(requests),
        "requests": [r.to_dict() for r in requests],
        "cache_keys": cache_keys,
        "rows_returned": facts.height,
        "rows_planned": sum(r.estimated_cells for r in requests),
        "value_status": value_status_summary(facts),
        "extracted_at": _now(),
        "tool_version": TOOL_VERSION,
    }
    # A shortfall is normal (tables do not publish every cell) but a surplus
    # means the plan mis-modelled the cube and every downstream count is suspect.
    report["row_surplus"] = facts.height - report["rows_planned"]

    parquet_path: Path | None = None
    sidecar_path: Path | None = None
    if write:
        target_dir = out_dir or PATHS.interim / "sidra"
        target_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = target_dir / f"{selection.name()}.parquet"
        facts.write_parquet(parquet_path)
        sidecar_path = parquet_path.with_suffix(".provenance.json")
        sidecar_path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    return ExtractionResult(
        facts=facts,
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        plan=requests,
        cache_keys=cache_keys,
        report=report,
    )


def extract_many(
    selections: Sequence[Selection],
    *,
    manifest: str | None = None,
    **kwargs: Any,
) -> dict[str, ExtractionResult]:
    """Run several selections and optionally freeze them into one manifest."""
    results: dict[str, ExtractionResult] = {}
    keys: list[str] = []
    for selection in selections:
        result = extract(selection, **kwargs)
        results[selection.name()] = result
        keys.extend(result.cache_keys)
    if manifest:
        write_manifest(manifest, keys)
    return results


def value_status_summary(facts: pl.DataFrame) -> dict[str, int]:
    """Counts per :class:`~api.ValueStatus`. Report this with every extraction."""
    if facts.height == 0:
        return {}
    counts = facts.group_by("value_status").len().sort("value_status")
    return {row[0]: int(row[1]) for row in counts.iter_rows()}


# --------------------------------------------------------------------------
# QA: margin reconciliation
# --------------------------------------------------------------------------

#: Fractions this size are floating-point noise; anything larger is a real
#: disagreement between the components and the published total.
DEFAULT_REL_TOLERANCE = 1e-6
DEFAULT_ABS_TOLERANCE = 1.0


def check_margins(
    df: pl.DataFrame,
    total_category: str,
    component_categories: Sequence[str],
    *,
    classification: str | None = None,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    abs_tolerance: float = DEFAULT_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Verify that component categories sum to the published Total.

    ``df`` is a long-fact frame from :func:`extract`. ``total_category`` is the
    id of the aggregate member (SIDRA marks these with ``nivel: 0``);
    ``component_categories`` are the members that should partition it.

    The comparison is made per (agregado, variable, locality, period) *and* per
    the remaining category tuple, so a table classified by both sex and colour
    is reconciled within each sex rather than across it.

    Only ``OK`` cells contribute to the component sum. Cells that were
    suppressed or not available are counted and reported separately: a group
    containing them is flagged ``incomplete`` rather than ``mismatch``, because
    a shortfall caused by confidentiality suppression is not a data error and
    must not be silently written off as one.

    Returns a report with ``passed``, per-group counts, and the worst offenders.
    """
    if df.height == 0:
        return {
            "passed": True,
            "reason": "empty frame",
            "n_groups": 0,
            "n_mismatched": 0,
            "n_incomplete": 0,
            "mismatches": [],
        }

    total_category = str(total_category)
    components = [str(c) for c in component_categories]
    if total_category in components:
        raise ValueError(
            f"total category {total_category} also listed as a component; that "
            "double-counts and would make the check pass trivially"
        )

    frame = df.with_row_index("_row")

    # Locate the classification axis to reconcile over. When the frame carries
    # several classifications, the caller may name one; otherwise we infer it as
    # the axis on which the total category actually appears.
    exploded = frame.select(
        "_row",
        pl.col("classification_ids"),
        pl.col("category_ids"),
    ).explode(["classification_ids", "category_ids"])

    if classification is None:
        hit = exploded.filter(pl.col("category_ids") == total_category)
        if hit.height == 0:
            return {
                "passed": False,
                "reason": f"total category {total_category} not present in frame",
                "n_groups": 0,
                "n_mismatched": 0,
                "n_incomplete": 0,
                "mismatches": [],
            }
        classification = str(hit["classification_ids"][0])
    classification = str(classification)

    axis = exploded.filter(pl.col("classification_ids") == classification).select(
        "_row", pl.col("category_ids").alias("_axis_category")
    )
    # Everything except the reconciled axis forms the grouping key, so that
    # margins are checked within each combination of the other dimensions.
    others = (
        exploded.filter(pl.col("classification_ids") != classification)
        .group_by("_row")
        .agg(
            pl.concat_str(
                [pl.col("classification_ids"), pl.col("category_ids")], separator=":"
            )
            .sort()
            .str.join("|")
            .alias("_other_key")
        )
    )

    frame = (
        frame.join(axis, on="_row", how="inner")
        .join(others, on="_row", how="left")
        .with_columns(pl.col("_other_key").fill_null(""))
    )

    group_keys = [
        "agregado",
        "variable_id",
        "locality_id",
        "locality_level",
        "period",
        "_other_key",
    ]

    totals = (
        frame.filter(pl.col("_axis_category") == total_category)
        .group_by(group_keys)
        .agg(
            pl.col("value_numeric").sum().alias("total_value"),
            pl.len().alias("n_total_rows"),
        )
    )
    parts = (
        frame.filter(pl.col("_axis_category").is_in(components))
        .group_by(group_keys)
        .agg(
            pl.col("value_numeric").sum().alias("component_sum"),
            (pl.col("value_status") == api.ValueStatus.OK.value).sum().alias("n_ok"),
            (pl.col("value_status") == api.ValueStatus.SUPPRESSED.value)
            .sum()
            .alias("n_suppressed"),
            (
                pl.col("value_status").is_in(
                    [
                        api.ValueStatus.NOT_AVAILABLE.value,
                        api.ValueStatus.UNKNOWN_SENTINEL.value,
                    ]
                )
            )
            .sum()
            .alias("n_unavailable"),
            pl.len().alias("n_component_rows"),
        )
    )

    joined = (
        totals.join(parts, on=group_keys, how="inner")
        .with_columns(
            (pl.col("component_sum") - pl.col("total_value")).alias("difference"),
            (pl.col("n_suppressed") + pl.col("n_unavailable") > 0).alias("incomplete"),
        )
        .with_columns(
            pl.when(pl.col("total_value").abs() > 0)
            .then((pl.col("difference").abs() / pl.col("total_value").abs()))
            .otherwise(pl.col("difference").abs())
            .alias("relative_difference")
        )
        .with_columns(
            (
                (pl.col("difference").abs() <= abs_tolerance)
                | (pl.col("relative_difference") <= rel_tolerance)
            ).alias("within_tolerance")
        )
    )

    mismatched = joined.filter(~pl.col("within_tolerance") & ~pl.col("incomplete"))
    incomplete = joined.filter(~pl.col("within_tolerance") & pl.col("incomplete"))

    worst = (
        mismatched.sort("relative_difference", descending=True)
        .head(20)
        .select(
            "agregado",
            "variable_id",
            "locality_id",
            "period",
            "_other_key",
            "total_value",
            "component_sum",
            "difference",
            "relative_difference",
        )
        .to_dicts()
    )

    return {
        "passed": mismatched.height == 0,
        "classification": classification,
        "total_category": total_category,
        "component_categories": components,
        "n_groups": joined.height,
        "n_within_tolerance": int(joined["within_tolerance"].sum()),
        "n_mismatched": mismatched.height,
        "n_incomplete": incomplete.height,
        "rel_tolerance": rel_tolerance,
        "abs_tolerance": abs_tolerance,
        "max_relative_difference": (
            float(joined["relative_difference"].max()) if joined.height else 0.0
        ),
        "mismatches": worst,
        "incomplete_examples": incomplete.head(10)
        .select(
            "agregado",
            "variable_id",
            "locality_id",
            "period",
            "total_value",
            "component_sum",
            "n_suppressed",
            "n_unavailable",
        )
        .to_dicts(),
        "checked_at": _now(),
    }


def coverage_report(facts: pl.DataFrame, expected_localities: Sequence[str]) -> dict[str, Any]:
    """Which of the expected localities actually came back with a number."""
    expected = {str(loc) for loc in expected_localities}
    if facts.height == 0:
        return {
            "n_expected": len(expected),
            "n_present": 0,
            "n_missing": len(expected),
            "missing": sorted(expected)[:50],
        }
    present = set(facts["locality_id"].unique().to_list())
    with_value = set(
        facts.filter(pl.col("value_status") == api.ValueStatus.OK.value)["locality_id"]
        .unique()
        .to_list()
    )
    missing = sorted(expected - present)
    return {
        "n_expected": len(expected),
        "n_present": len(present & expected),
        "n_with_numeric_value": len(with_value & expected),
        "n_missing": len(missing),
        "missing": missing[:50],
        "unexpected": sorted(present - expected)[:50],
    }


__all__ = [
    "Selection",
    "ExtractionResult",
    "build_spec",
    "resolve_localities",
    "validate_selection",
    "extract",
    "extract_many",
    "check_margins",
    "coverage_report",
    "value_status_summary",
]
