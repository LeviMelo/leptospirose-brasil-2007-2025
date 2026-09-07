"""Run external statistical jobs with observable resource and failure bounds.

R, INLA, Stan, geospatial CLIs, and other compiled engines cannot reliably be
interrupted by an in-language time limit.  This module puts the bound at the
operating-system process-tree level and reports peak tree memory/CPU.  A timed
out or over-memory job is a failed job; callers must not replace it with a
fallback estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import threading
import time
from typing import Sequence

import psutil


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    returncode: int
    seconds: float
    peak_rss_mb: float
    cpu_seconds: float
    timed_out: bool
    memory_exceeded: bool
    stdout: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.memory_exceeded

    def as_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_bounded(command: Sequence[str], *, cwd: Path | str,
                timeout_seconds: float, memory_limit_mb: float | None = None,
                sample_seconds: float = 1.0) -> ProcessResult:
    """Execute one command, killing its whole tree at either declared bound."""
    cmd = [str(part) for part in command]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    peak_rss = 0.0
    peak_cpu = 0.0
    memory_exceeded = False
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak_rss, peak_cpu, memory_exceeded
        try:
            parent = psutil.Process(proc.pid)
        except psutil.Error:
            return
        while not stop.is_set() and proc.poll() is None:
            try:
                tree = [parent] + parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree if p.is_running()) / 1024**2
                cpu = sum(p.cpu_times().user + p.cpu_times().system
                          for p in tree if p.is_running())
                peak_rss, peak_cpu = max(peak_rss, rss), max(peak_cpu, cpu)
                if memory_limit_mb is not None and rss > memory_limit_mb:
                    memory_exceeded = True
                    _kill_tree(proc)
                    return
            except psutil.Error:
                pass
            stop.wait(sample_seconds)

    watcher = threading.Thread(target=sample, daemon=True)
    started = time.monotonic()
    watcher.start()
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        stdout, _ = proc.communicate()
    finally:
        stop.set()
        watcher.join(timeout=max(2.0, sample_seconds * 2))
    return ProcessResult(cmd, proc.returncode, time.monotonic() - started,
                         peak_rss, peak_cpu, timed_out, memory_exceeded,
                         stdout or "")
