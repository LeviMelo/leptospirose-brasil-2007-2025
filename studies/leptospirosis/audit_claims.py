"""Audit the study claim ledger against material evidence.

This script cannot certify scientific validity. It enforces the narrower,
testable rule that a claim marked as validated has the expected national
artifact and contract, while absent model inputs remain visibly blocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brepi.qa.evidence import evaluate_assertions  # noqa: E402 - local checkout entrypoint

LEDGER = Path(__file__).with_name("CLAIM_LEDGER.yaml")
REPORT = (ROOT / "data" / "results" / "00_data_quality"
          / "claim_ledger_audit.json")
RESULT_ASSERTIONS = Path(__file__).with_name("RESULT_ASSERTIONS.yaml")
ANALYTIC_DATASETS = Path(__file__).with_name("ANALYTIC_DATASETS.yaml")


def _read_json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))



def _declared_evidence_paths(claim: dict) -> list[str]:
    """Every string in a claim's `evidence` block that names a data artefact."""
    found: list[str] = []

    def walk(v):
        if isinstance(v, str):
            if v.startswith("data/") and ("." in Path(v).name or v.endswith("/")):
                found.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(claim.get("evidence"))
    return found


def check_evidence_exists(ledger: dict) -> dict:
    """Does every declared evidence path actually exist?

    A claim ledger is only worth having if its evidence paths resolve. They
    drift for ordinary reasons -- a stage is renamed, a directory is
    restructured -- and the drift is invisible until somebody follows a path
    from the manuscript and finds nothing. Checking it here makes the ledger
    self-verifying rather than aspirational.
    """
    missing: dict[str, list[str]] = {}
    checked = 0
    for claim in ledger["claims"]:
        gone = []
        for rel in _declared_evidence_paths(claim):
            checked += 1
            if not (ROOT / rel).exists():
                gone.append(rel)
        if gone:
            missing[claim["id"]] = sorted(set(gone))
    return {
        "passed": not missing,
        "paths_checked": checked,
        "claims_with_missing_evidence": missing,
    }



# Controlled-vocabulary columns that appear in more than one result file and are
# therefore join keys in practice. A disjoint vocabulary between two of them is
# a silent join failure, not a style difference.
VOCABULARY_COLUMNS = [
    "region", "uf_abbr", "sex", "age_group", "regime", "reservoir",
    "serogroup", "surveillance_class", "level", "phase", "climate_product",
]


def check_vocabularies_consistent() -> dict:
    """No two result files may name the same concept in different vocabularies.

    This caught a live defect: the canonical panel carried IBGE's Portuguese
    macro-regions while the descriptive tables carried English ones, so a join
    between the regime assignment and the incidence table matched nothing at
    all. Both were individually correct, which is exactly why nothing else
    noticed.
    """
    from brepi.qa import vocabulary as _voc

    root = ROOT / "data" / "results"
    if not root.exists():
        return {"passed": True, "skipped": "no results directory"}
    reports = _voc.check_vocabularies(root, VOCABULARY_COLUMNS)
    bad = {c: r.to_dict()["disjoint_pairs"]
           for c, r in reports.items() if not r.passed}
    return {
        "passed": not bad,
        "columns_checked": len(VOCABULARY_COLUMNS),
        "columns_present": sum(1 for r in reports.values() if r.n_files),
        "disjoint_columns": bad,
    }


def main() -> int:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    panel_path = (
        ROOT
        / "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet"
    )
    panel = pl.scan_parquet(panel_path)
    schema = set(panel.collect_schema().names())
    panel_stats = panel.select(
        pl.len().alias("rows"),
        pl.col("munic_code").n_unique().alias("municipalities"),
        pl.struct("munic_code", "period").n_unique().alias("unique_keys"),
        pl.col("cases").sum().alias("geocoded_cases"),
    ).collect().row(0, named=True)

    outcome = _read_json("data/results/00_data_quality/sparsity.json")
    sanitation = _read_json("data/results/00_data_quality/lepto_sanitation_panel_quality.json")
    climate = _read_json("data/results/00_data_quality/lepto_climate_panel_quality.json")
    graph_path = ROOT / "data/panel/graphs/health_region.adj"
    graph_nodes = int(graph_path.read_text(encoding="utf-8").splitlines()[0])

    required_model_columns = next(
        item["required_columns"]
        for item in ledger["claims"]
        if item["id"] == "confirmatory_primary_model"
    )

    def _audit_confirmatory(ledger, schema, missing_columns, graph_nodes):
        """Audit the primary model claim against the artefact it points at.

        A fit that executes is not the same thing as a manuscript estimate. The
        model now runs, converges and diagnoses cleanly, so `executed` is a
        genuine state distinct from the earlier `null`; but `passed` stays
        False while any gate in `remaining_gates` is open. A benchmark that
        proves the numerical route works cannot promote a blocked claim.
        """
        claim = next(
            item for item in ledger["claims"]
            if item["id"] == "confirmatory_primary_model"
        )
        meta_path = ROOT / (
            claim.get("evidence", {}).get("outputs", "")
            + "fit_metadata.json"
        )
        meta = _read_json(str(meta_path.relative_to(ROOT))) if meta_path.exists() else {}
        executed = bool(meta) and meta.get("quotable") is True
        clean = (
            executed
            and meta.get("n_cpo_failure") == 0
            and meta.get("vcov_route") == "correlation.matrix"
        )
        gates = claim.get("remaining_gates", [])
        return {
            # False while gates remain open, by design.
            "passed": bool(clean and not gates),
            "ready": not missing_columns and graph_nodes == 439,
            "missing_columns": missing_columns,
            "executed": executed,
            "diagnostics_clean": clean,
            "scientific_result": (
                {
                    "cumulative_rr_p95_vs_median":
                        claim.get("result", {}).get("cumulative_rr_p95_vs_median"),
                    "exposure_scale": claim.get("result", {}).get("exposure_scale"),
                    "elapsed_seconds": meta.get("elapsed_seconds"),
                    "dic": meta.get("dic"),
                    "waic": meta.get("waic"),
                }
                if executed else None
            ),
            "remaining_gates": gates,
            "quotable_as_manuscript_estimate":
                claim.get("quotable_as_manuscript_estimate", False),
        }
    missing_model_columns = sorted(set(required_model_columns) - schema)
    outcome_conserved = (
        outcome["geocoded_cases"] + outcome["unresolved_geography_cases"]
        == outcome["confirmed_in_window"]
    )
    checks = {
        "outcome_geography": {
            "passed": (
                outcome_conserved
                and panel_stats["geocoded_cases"] == outcome["geocoded_cases"]
            ),
            **outcome,
            "panel_geocoded_cases": panel_stats["geocoded_cases"],
        },
        "canonical_panel": {
            "passed": {
                key: panel_stats[key]
                for key in ("rows", "municipalities", "unique_keys")
            }
            == {
                "rows": 1_269_960,
                "municipalities": 5_570,
                "unique_keys": 1_269_960,
            },
            **panel_stats,
        },
        "sanitation_backbone": {
            "passed": (
                sanitation["missing_panel_cells"] == 0
                and sanitation["territorial_proxy_anchors"] == 5
                and all(
                    report["passed"]
                    and report["n_mismatched"] == 0
                    and report["n_incomplete"] == 0
                    for report in sanitation["margin_checks"].values()
                )
            ),
            "margin_checks": sanitation["margin_checks"],
            "territorial_proxy_codes": sanitation["territorial_proxy_codes"],
        },
        "official_geometry_graph": {
            "passed": graph_path.exists() and graph_nodes == 439,
            "graph_nodes": graph_nodes,
        },
        "climate_block": {
            "passed": "precip_mm" in schema
            and climate.get("missing_climate_cells_after_bridge", 0) == 0,
        },
        "confirmatory_primary_model": _audit_confirmatory(
            ledger, schema, missing_model_columns, graph_nodes
        ),
    }
    checks["_evidence_paths_exist"] = check_evidence_exists(ledger)
    checks["_vocabularies_consistent"] = check_vocabularies_consistent()
    assertion_doc = yaml.safe_load(RESULT_ASSERTIONS.read_text(encoding="utf-8"))
    checks["_result_values_match_claims"] = evaluate_assertions(
        ROOT, assertion_doc.get("assertions", []))
    dataset_doc = yaml.safe_load(ANALYTIC_DATASETS.read_text(encoding="utf-8"))
    dataset_assertions = []
    for dataset in dataset_doc.get("datasets", []):
        for dimension, kind in (("rows", "row_count"),
                                ("columns", "column_count")):
            # Check what the catalogue declares. A dataset that states its row
            # count but not its width is under-specified, not malformed; raising
            # here turned the whole audit into a crash over a missing key.
            if dimension not in dataset:
                continue
            dataset_assertions.append({
                "id": f"dataset_{dataset['id']}_{dimension}",
                "path": dataset["path"], "kind": kind,
                "expected": dataset[dimension],
            })
    checks["_analytic_dataset_contracts"] = evaluate_assertions(
        ROOT, dataset_assertions)
    validated_ids = {
        item["id"]
        for item in ledger["claims"]
        if item["status"] == "validated_real_data"
    }
    failed_validated = sorted(
        claim_id
        for claim_id in validated_ids
        if claim_id in checks and not checks[claim_id]["passed"]
    )
    # A broken evidence path fails the audit regardless of which claim owns it:
    # an unverifiable claim is not a lesser claim, it is not a claim.
    if not checks["_evidence_paths_exist"]["passed"]:
        failed_validated = sorted(set(failed_validated) | {"_evidence_paths_exist"})
    # A join key that matches nothing makes every table built on it wrong, so
    # this fails the audit rather than warning.
    if not checks["_vocabularies_consistent"]["passed"]:
        failed_validated = sorted(
            set(failed_validated) | {"_vocabularies_consistent"})
    if not checks["_result_values_match_claims"]["passed"]:
        failed_validated = sorted(
            set(failed_validated) | {"_result_values_match_claims"})
    if not checks["_analytic_dataset_contracts"]["passed"]:
        failed_validated = sorted(
            set(failed_validated) | {"_analytic_dataset_contracts"})
    report = {
        "schema": "brepi.study.claim-ledger-audit/1",
        "ledger": str(LEDGER.relative_to(ROOT)),
        "checks": checks,
        "failed_validated_claims": failed_validated,
        "passed": not failed_validated,
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return int(bool(failed_validated))


if __name__ == "__main__":
    sys.exit(main())
