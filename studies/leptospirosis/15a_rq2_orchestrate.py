#!/usr/bin/env python
"""Sequential, resumable, resource-bounded orchestration for RQ2 fits.

Each subprocess invokes ``15b_rq2_one.R`` for exactly one estimand. Existing
successful checkpoints are skipped after their status is parsed. Failures,
timeouts, and memory ceilings stop the run by default and are never routed to a
different estimator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brepi.execution import run_bounded  # noqa: E402 - local checkout entrypoint

OUT = ROOT / "data" / "results" / "rq2_flood_disasters"
LOGS = ROOT / "data" / "logs" / "rq2"
RSCRIPT = os.environ.get(
    "BREPI_RSCRIPT", r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
RUNNER = "studies/leptospirosis/15b_rq2_one.R"


def prefix(job: str, outcome: str) -> str:
    return f"corrected_{job}" + ("" if outcome == "rate" else f"_{outcome}")


def checkpoint_ok(job: str, outcome: str, grain: int, bootstrap: bool) -> bool:
    path = OUT / f"{prefix(job, outcome)}_status.csv"
    if not path.exists():
        return False
    try:
        row = pd.read_csv(path).iloc[0]
        return (bool(row["ok"]) and row["outcome"] == outcome and
                int(row["grain"]) == grain and bool(row["bootstrap"]) == bootstrap)
    except (OSError, ValueError, IndexError, KeyError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", default=[
        "primary", "recurrence_clean", "nevertreated", "drought_placebo"])
    parser.add_argument("--outcome", choices=["rate", "asinh", "any"],
                        default="rate")
    parser.add_argument("--grain", type=int, default=2)
    parser.add_argument("--bootstrap", choices=["auto", "true", "false"],
                        default="auto", help="auto uses bootstrap only for the primary rate fit")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--memory-mb", type=float, default=12_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for job in args.jobs:
        bootstrap = ((job == "primary" and args.outcome == "rate")
                     if args.bootstrap == "auto" else args.bootstrap == "true")
        if not args.force and checkpoint_ok(job, args.outcome, args.grain, bootstrap):
            records.append({"job": job, "outcome": args.outcome,
                            "status": "checkpoint_ok"})
            print(f"[{job}] valid checkpoint; skip", flush=True)
            continue
        cmd = [RSCRIPT, "--vanilla", RUNNER, job, args.outcome,
               str(args.grain), str(bootstrap).lower()]
        print(f"[{job}] starting bounded subprocess", flush=True)
        result = run_bounded(cmd, cwd=ROOT, timeout_seconds=args.timeout,
                             memory_limit_mb=args.memory_mb, sample_seconds=2)
        log = LOGS / f"{prefix(job, args.outcome)}.log"
        log.write_text(result.stdout, encoding="utf-8")
        row = {"job": job, "outcome": args.outcome, "grain": args.grain,
               "bootstrap": bootstrap,
               "status": "ok" if result.ok and checkpoint_ok(
                   job, args.outcome, args.grain, bootstrap)
               else "failed", **result.as_dict(), "log": str(log.relative_to(ROOT))}
        records.append(row)
        print(f"[{job}] {row['status']}; {result.seconds:.1f}s, "
              f"peak {result.peak_rss_mb:.0f} MB", flush=True)
        if row["status"] != "ok" and not args.continue_on_error:
            break

    report = OUT / "corrected_orchestration_status.json"
    report.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return 0 if all(r["status"] in {"ok", "checkpoint_ok"} for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
