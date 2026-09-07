import sys
from pathlib import Path

from brepi.execution import run_bounded


def test_bounded_process_reports_success(tmp_path: Path) -> None:
    result = run_bounded([sys.executable, "-c", "print('real output')"],
                         cwd=tmp_path, timeout_seconds=5, sample_seconds=0.01)
    assert result.ok
    assert "real output" in result.stdout
    assert result.seconds >= 0


def test_bounded_process_timeout_is_failed_not_fallback(tmp_path: Path) -> None:
    result = run_bounded(
        [sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path,
        timeout_seconds=0.1, sample_seconds=0.01)
    assert not result.ok
    assert result.timed_out
