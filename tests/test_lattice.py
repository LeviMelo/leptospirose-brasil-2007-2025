from datetime import date

import polars as pl
import pytest

from brepi.geo.lattice import (
    LatticeError,
    impute_created_unit_covariates,
    resolve_municipality_code,
)


def test_resolve_municipality_code_rejects_six_digit_placeholders():
    frame = pl.DataFrame(
        {
            "infection": ["355030", "350000", None],
            "residence": ["330455", "310620", "530010"],
        }
    )
    out = resolve_municipality_code(
        frame,
        [("infection", "infection"), ("residence_fallback", "residence")],
    )
    assert out["munic_code6"].to_list() == ["355030", "310620", "530010"]
    assert out["munic_code_source"].to_list() == [
        "infection",
        "residence_fallback",
        "residence_fallback",
    ]
    assert out["munic_code_status"].to_list() == ["resolved"] * 3


def test_resolve_municipality_code_preserves_unresolved_row():
    frame = pl.DataFrame({"infection": ["350000"], "residence": [None]})
    out = resolve_municipality_code(
        frame,
        [("infection", "infection"), ("residence_fallback", "residence")],
    )
    assert out["munic_code6"].item() is None
    assert out["munic_code_source"].item() is None
    assert out["munic_code_status"].item() == "unresolved"


def test_created_unit_covariate_copies_single_parent():
    frame = pl.DataFrame(
        {
            "munic_code": ["1506807", "1506807"],
            "period": [date(2012, 1, 1), date(2012, 2, 1)],
            "rain": [100.0, 200.0],
        }
    )
    out = impute_created_unit_covariates(
        frame,
        ["1506807", "1504752"],
        ["rain"],
        source_lattice_year=2010,
    )
    assert out["munic_code"].unique().to_list() == ["1504752"]
    assert out["rain"].to_list() == [100.0, 200.0]
    assert out["territorial_imputation"].unique().to_list() == ["single_parent_copy"]
    assert out["territorial_source_codes"].unique().to_list() == ["1506807"]


def test_created_unit_covariate_averages_all_recorded_parents():
    parents = ["5003256", "5002951", "5000203"]
    frame = pl.DataFrame(
        {
            "munic_code": parents,
            "period": [date(2012, 1, 1)] * 3,
            "rain": [90.0, 120.0, 150.0],
        }
    )
    out = impute_created_unit_covariates(
        frame,
        [*parents, "5006275"],
        ["rain"],
        source_lattice_year=2010,
    )
    assert out["rain"].item() == pytest.approx(120.0)
    assert out["territorial_imputation"].item() == "parent_mean_unweighted"
    assert out["territorial_source_count"].item() == 3


def test_created_unit_covariate_support_weights_parent_mean():
    parents = ["5003256", "5002951", "5000203"]
    frame = pl.DataFrame(
        {
            "munic_code": parents,
            "period": [date(2012, 1, 1)] * 3,
            "share": [0.1, 0.4, 0.9],
            "households": [10.0, 20.0, 70.0],
        }
    )
    out = impute_created_unit_covariates(
        frame,
        [*parents, "5006275"],
        ["share"],
        weight_column="households",
        source_lattice_year=2010,
    )
    assert out["share"].item() == pytest.approx(0.72)
    assert out["territorial_imputation"].item() == (
        "parent_support_weighted_mean"
    )


def test_created_unit_covariate_can_mark_same_year_source_lag():
    frame = pl.DataFrame(
        {
            "munic_code": ["2211001"],
            "year": [2008],
            "gdp_pc": [100.0],
        }
    )
    out = impute_created_unit_covariates(
        frame,
        ["2211001", "2206720"],
        ["gdp_pc"],
        key_columns=("year",),
        source_lattice_year=2008,
        include_same_year=True,
    )
    assert out["munic_code"].item() == "2206720"
    assert out["territorial_source_codes"].item() == "2211001"


def test_created_unit_covariate_refuses_counts_by_type():
    frame = pl.DataFrame(
        {
            "munic_code": ["1506807"],
            "period": [date(2012, 1, 1)],
            "category": ["wet"],
        }
    )
    with pytest.raises(LatticeError, match="numeric covariates only"):
        impute_created_unit_covariates(
            frame,
            ["1506807", "1504752"],
            ["category"],
            source_lattice_year=2010,
        )
