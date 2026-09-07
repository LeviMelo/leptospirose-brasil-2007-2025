from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from brepi.geo.meshes import MeshError, _safe_extract, municipality_mesh_url


def test_municipality_mesh_url_is_version_pinned():
    assert municipality_mesh_url(2022).endswith(
        "/municipio_2022/Brasil/BR/BR_Municipios_2022.zip"
    )


def test_safe_extract_refuses_zip_slip(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "no")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(MeshError, match="unsafe path"):
        _safe_extract(archive, destination)
    assert not (tmp_path / "escape.txt").exists()
