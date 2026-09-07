"""BREPI-001 / BREPI-002 - benchmark the RQ1 INLA model, decomposed.

Runs each specification as its own R process so that a hang is bounded by a
timeout rather than by the operator's patience, and so peak resident memory is
measured per fit instead of being confounded with everything R has ever
allocated in the session.

The ladder answers, in order:

structure
    M0 intercept -> M1 covariates + cross-basis -> M2 BYM2 -> M3 RW1 +
    seasonality -> M4 residual space-time interaction (annual, then monthly).
    A runtime jump between adjacent rungs is attributable to exactly one added
    component.

diagnostics
    On the two candidate structures, add one expensive INLA option at a time:
    CCD hyperparameter integration, the adaptive latent strategy, DIC/WAIC,
    CPO/PIT, and posterior configuration storage. The monolithic confirmatory
    profile requests all five simultaneously, which is why the two stopped runs
    could not be attributed.

Nothing produced here is confirmatory. Outputs are cost measurements and
coefficient-stability checks used to choose a specification, which is then
fitted separately under a declared confirmatory profile.

Usage:
    python studies/leptospirosis/10_benchmark_inla.py --stage structure
    python studies/leptospirosis/10_benchmark_inla.py --stage diagnostics
    python studies/leptospirosis/10_benchmark_inla.py --spec M4y__ccd --timeout 1800
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
RSCRIPT = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
FIT_ONE = "studies/leptospirosis/bench/fit_one.R"
REPORTS = ROOT / "data" / "reports" / "bench"

STRUCTURE_LADDER = [
    "M0__dev", "M1__dev", "M2__dev", "M3__dev", "M4y__dev", "M4m__dev",
]
DIAGNOSTIC_LADDER = [
    "M3__slaplace", "M3__adaptive", "M3__ccd", "M3__dicwaic", "M3__cpo",
    "M3__config", "M3__full",
    "M4y__slaplace", "M4y__ccd", "M4y__dicwaic", "M4y__cpo", "M4y__config",
]


def run_spec(spec: str, scale: str, timeout: int) -> dict:
    """Run one spec, sampling peak RSS of the whole process tree."""
    out_json = REPORTS / f"{spec}.json"
    if out_json.exists():
        out_json.unlink()

    cmd = [RSCRIPT, FIT_ONE, spec, scale]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    peak = {"rss_mb": 0.0, "cpu_s": 0.0}
    stop = threading.Event()

    def sample() -> None:
        try:
            p = psutil.Process(proc.pid)
        except psutil.Error:
            return
        while not stop.is_set():
            try:
                tree = [p] + p.children(recursive=True)
                rss = sum(c.memory_info().rss for c in tree if c.is_running())
                cpu = sum(c.cpu_times().user + c.cpu_times().system
                          for c in tree if c.is_running())
                peak["rss_mb"] = max(peak["rss_mb"], rss / 1024**2)
                peak["cpu_s"] = max(peak["cpu_s"], cpu)
            except psutil.Error:
                pass
            stop.wait(2.0)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
        proc.kill()
        stdout, _ = proc.communicate()
    finally:
        stop.set()
        t.join(timeout=5)

    wall = time.time() - t0
    row: dict = {
        "spec": spec,
        "wall_seconds": round(wall, 1),
        "peak_rss_mb": round(peak["rss_mb"], 1),
        "cpu_seconds": round(peak["cpu_s"], 1),
        "timed_out": timed_out,
        "returncode": proc.returncode,
    }
    if out_json.exists():
        row.update(json.loads(out_json.read_text(encoding="utf-8")))
    else:
        row["ok"] = False
        tail = "\n".join((stdout or "").strip().splitlines()[-6:])
        row["error"] = "TIMEOUT" if timed_out else tail
    return row


def fmt(rows: list[dict]) -> str:
    hdr = f"{'spec':<18}{'ok':<5}{'wall_s':>9}{'cpu_s':>9}{'RSS_MB':>9}{'latent':>9}{'obj_MB':>8}  notes"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        cpu = r.get("cpu", {}) or {}
        note = ""
        if r.get("timed_out"):
            note = "TIMED OUT"
        elif not r.get("ok"):
            note = str(r.get("error", ""))[:60].replace("\n", " ")
        elif cpu:
            note = " ".join(f"{k}={v}" for k, v in cpu.items() if k in
                            ("pre", "running", "post", "total"))
        lines.append(
            f"{r['spec']:<18}{str(r.get('ok', False)):<5}"
            f"{r.get('wall_seconds', 0):>9.1f}{r.get('cpu_seconds', 0):>9.1f}"
            f"{r.get('peak_rss_mb', 0):>9.1f}{r.get('latent_size', 0):>9}"
            f"{r.get('object_mb', 0) or 0:>8.1f}  {note}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["structure", "diagnostics", "all"],
                    default="structure")
    ap.add_argument("--spec", default=None, help="run a single spec")
    ap.add_argument("--scale", default="health_region")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds per fit; a fit that exceeds it is recorded "
                         "as TIMED OUT rather than being waited on")
    a = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if a.spec:
        specs = [a.spec]
    elif a.stage == "structure":
        specs = STRUCTURE_LADDER
    elif a.stage == "diagnostics":
        specs = DIAGNOSTIC_LADDER
    else:
        specs = STRUCTURE_LADDER + DIAGNOSTIC_LADDER

    rows = []
    for spec in specs:
        print(f"--- {spec} (timeout {a.timeout}s) ---", flush=True)
        row = run_spec(spec, a.scale, a.timeout)
        rows.append(row)
        print(f"    ok={row.get('ok')} wall={row.get('wall_seconds')}s "
              f"cpu={row.get('cpu_seconds')}s rss={row.get('peak_rss_mb')}MB",
              flush=True)

    print()
    print(fmt(rows))
    summary = REPORTS / f"_ladder_{a.stage}.json"
    summary.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
