import json
from datetime import date

import polars as pl
import pytest

from brepi.geo import health_regions
from brepi.geo.health_regions import (
    HealthRegionError,
    fetch_open_datasus,
    load_official_crosswalk,
)


def test_crosswalk_normalises_six_digit_codes_and_rejects_incomplete(tmp_path, monkeypatch):
    lattice = pl.DataFrame({"code7": ["1200401", "3550308"]})
    monkeypatch.setattr(
        "brepi.geo.health_regions.lattice.load_municipalities", lambda year: lattice
    )
    path = tmp_path / "health.csv"
    path.write_text("COD_MUN;REGIAO_SAUDE;NOME_REGIAO_SAUDE\n120040;01;A\n355030;02;B\n")
    out, report = load_official_crosswalk(path, vintage="official 2026-07")
    assert out.to_dicts() == [
        {"code7": "1200401", "health_region": "01", "health_region_name": "A"},
        {"code7": "3550308", "health_region": "02", "health_region_name": "B"},
    ]
    assert report.n_municipalities == 2


def test_crosswalk_requires_complete_mapping(tmp_path, monkeypatch):
    lattice = pl.DataFrame({"code7": ["1200401", "3550308"]})
    monkeypatch.setattr(
        "brepi.geo.health_regions.lattice.load_municipalities", lambda year: lattice
    )
    path = tmp_path / "health.csv"
    path.write_text("COD_MUN,REGIAO_SAUDE\n120040,01\n")
    with pytest.raises(HealthRegionError, match="covers 1/2"):
        load_official_crosswalk(path, vintage="official 2026-07")


def test_open_datasus_json_prefers_region_code_and_keeps_region_name(
    tmp_path, monkeypatch
):
    target = pl.DataFrame({"code7": ["1200401", "3550308"]})
    monkeypatch.setattr(
        "brepi.geo.health_regions.lattice.load_municipalities", lambda year: target
    )
    document = {
        "macrorregiao_regiao_saude_municipios": [
            {
                "codigo_municipio": "120040",
                "codigo_regiao_saude": "12002",
                "regiao_saude": "BAIXO ACRE E PURUS",
            },
            {
                "codigo_municipio": "355030",
                "codigo_regiao_saude": "35016",
                "regiao_saude": "SAO PAULO",
            },
        ]
    }
    path = tmp_path / "health.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    out, report = load_official_crosswalk(
        path, vintage="OpenDataSUS retrieved 2026-07-30"
    )
    assert out.to_dicts() == [
        {
            "code7": "1200401",
            "health_region": "12002",
            "health_region_name": "BAIXO ACRE E PURUS",
        },
        {
            "code7": "3550308",
            "health_region": "35016",
            "health_region_name": "SAO PAULO",
        },
    ]
    assert report.n_rows_input == 2


def test_open_datasus_pagination_uses_row_offsets(monkeypatch):
    rows = [{"codigo_municipio": f"{i:06d}"} for i in range(5)]
    seen_offsets = []

    class Response:
        headers = {}

        def __init__(self, batch):
            self.batch = batch

        def raise_for_status(self):
            return None

        def json(self):
            return {"macrorregiao_regiao_saude_municipios": self.batch}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, params):
            seen_offsets.append(params["offset"])
            offset, limit = params["offset"], params["limit"]
            return Response(rows[offset : offset + limit])

    captured = {}

    def fake_fetch(key, loader, **kwargs):
        payload, metadata = loader()
        captured["document"] = json.loads(payload)
        captured["metadata"] = metadata
        return "path", "provenance"

    monkeypatch.setattr(health_regions.httpx, "Client", Client)
    monkeypatch.setattr(health_regions.cache, "fetch", fake_fetch)
    result = fetch_open_datasus(snapshot_date=date(2026, 7, 30), page_size=2)
    assert result == ("path", "provenance")
    assert seen_offsets == [0, 2, 4]
    assert len(captured["document"]["macrorregiao_regiao_saude_municipios"]) == 5
    assert captured["metadata"]["params"]["offset_semantics"].startswith("row offset")
