from __future__ import annotations

from pathlib import Path

import pytest

from brepi import config
from brepi.io import cache


def test_fetch_stream_is_atomic_hashed_and_cached(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cache, "PATHS", config.Paths(tmp_path))
    calls = 0

    def loader(destination: Path):
        nonlocal calls
        calls += 1
        destination.write_bytes(b"large-ish-payload")
        return {"remote_modified": "yesterday", "params": {"attempt": 1}}

    path, provenance = cache.fetch_stream(
        "source/blob.bin", loader, source="test", uri="https://example.test/blob"
    )
    assert path.read_bytes() == b"large-ish-payload"
    assert provenance.bytes == 17
    assert provenance.sha256 == cache.sha256_bytes(b"large-ish-payload")
    assert not list(path.parent.glob("*.part"))

    cached, cached_provenance = cache.fetch_stream(
        "source/blob.bin", loader, source="test", uri="https://example.test/blob"
    )
    assert cached == path
    assert cached_provenance == provenance
    assert calls == 1


def test_fetch_stream_does_not_publish_failed_partial(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cache, "PATHS", config.Paths(tmp_path))

    def loader(destination: Path):
        destination.write_bytes(b"partial")
        raise RuntimeError("network broke")

    with pytest.raises(RuntimeError, match="network broke"):
        cache.fetch_stream(
            "source/blob.bin", loader, source="test", uri="https://example.test/blob"
        )
    assert not (tmp_path / "cache/source/blob.bin").exists()
    assert not list((tmp_path / "cache/source").glob("*.part"))
