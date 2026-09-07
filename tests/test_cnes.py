from __future__ import annotations

import polars as pl
import pytest

from brepi.sources.datasus import cnes
from brepi.sources.datasus.cnes import (
    CnesFile,
    _laboratories_by_municipality,
    _laboratory_service_code,
    _normalise_cnes_municipality,
)


def test_laboratory_capacity_counts_distinct_service_145_establishments():
    sr = pl.DataFrame(
        {
            "CNES": ["a", "a", "b", "c"],
            "CODUFMUN": ["1", "1", "1", "2"],
            "SERV_ESP": ["145", "145", "120", "145"],
        }
    )
    out = _laboratories_by_municipality(sr).sort("CODUFMUN")
    assert out.to_dicts() == [
        {"CODUFMUN": "1", "n_labs": 1},
        {"CODUFMUN": "2", "n_labs": 1},
    ]


def test_laboratory_taxonomy_switches_after_2007():
    assert _laboratory_service_code(2005) == "013"
    assert _laboratory_service_code(2007) == "013"
    assert _laboratory_service_code(2008) == "145"
    assert _laboratory_service_code(2024) == "145"


def test_df_administrative_regions_collapse_to_brasilia():
    frame = pl.DataFrame(
        {"CODUFMUN": ["530010", "530060", "530180", "355030"]}
    )
    out = _normalise_cnes_municipality(frame)
    assert out["CODUFMUN"].to_list() == [
        "530010",
        "530010",
        "530010",
        "355030",
    ]


def _file(group: str, uf: str, year: int) -> CnesFile:
    name = f"{group}{uf}{year % 100:02d}12.dbc"
    return CnesFile(
        path=f"/CNES/200508_/Dados/{group}/{name}",
        name=name,
        bytes=1,
        modified=None,
        group=group,
        uf=uf,
        year=year,
        month=12,
    )


def test_capacity_series_preflights_complete_inventory_before_download(
    monkeypatch: pytest.MonkeyPatch,
):
    listings = {
        "ST": [_file("ST", "AC", 2024)],
        "LT": [_file("LT", "AC", 2024)],
        "SR": [],
    }
    calls: list[str] = []

    monkeypatch.setattr(
        cnes,
        "list_available",
        lambda group, **_kwargs: listings[group],
    )
    monkeypatch.setattr(
        cnes,
        "capacity_by_municipality",
        lambda *_args, **_kwargs: calls.append("download"),
    )

    with pytest.raises(FileNotFoundError, match="SR-AC-202412"):
        cnes.capacity_series([2024], ufs=["AC"])
    assert calls == []


def test_capacity_series_lists_each_group_once_and_reuses_inventory(
    monkeypatch: pytest.MonkeyPatch,
):
    listed: list[str] = []
    received: list[dict[str, list[CnesFile]]] = []

    def fake_list(group: str, **_kwargs):
        listed.append(group)
        return [_file(group, "AC", year) for year in (2023, 2024)]

    def fake_capacity(year: int, _month: int, **kwargs):
        received.append(kwargs["listings"])
        return pl.DataFrame(
            {
                "munic_code": ["120001"],
                "year": [year],
                "month": [12],
                "n_estab": [1],
            }
        )

    monkeypatch.setattr(cnes, "list_available", fake_list)
    monkeypatch.setattr(cnes, "capacity_by_municipality", fake_capacity)

    result = cnes.capacity_series([2023, 2024], ufs=["AC"])
    assert listed == ["ST", "LT", "SR"]
    assert len(received) == 2
    assert received[0] is received[1]
    assert result["year"].to_list() == [2023, 2024]
