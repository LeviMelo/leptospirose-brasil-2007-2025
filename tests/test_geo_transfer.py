from datetime import date

import polars as pl
import pytest

from brepi.geo.transfer import (
    TransferError,
    VariableKind,
    correspondence_from_support,
    transfer_extensive,
    transfer_intensive,
    transfer_rate,
    validate_correspondence,
)


@pytest.fixture
def correspondence() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_code": ["A", "A", "B", "B"],
            "target_code": ["X", "Y", "X", "Y"],
            "source_fraction": [0.75, 0.25, 0.25, 0.75],
            "target_fraction": [0.75, 0.25, 0.25, 0.75],
            "method": ["atomic_overlay"] * 4,
            "exact": [True] * 4,
        }
    )


def test_atomic_support_compiles_both_weight_systems():
    atoms = pl.DataFrame(
        {
            "old": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "new": ["X", "X", "X", "Y", "X", "Y", "Y", "Y"],
            "people": [25.0] * 8,
        }
    )
    out = correspondence_from_support(
        atoms,
        source_column="old",
        target_column="new",
        mass_column="people",
        source_version="municipality:2010",
        target_version="municipality:2022",
        method="census_block_population",
        exact=True,
        expected_source_codes=["A", "B"],
        expected_target_codes=["X", "Y"],
    )
    ax = out.filter(
        (pl.col("source_code") == "A") & (pl.col("target_code") == "X")
    ).row(0, named=True)
    assert ax["source_fraction"] == pytest.approx(0.75)
    assert ax["target_fraction"] == pytest.approx(0.75)
    assert out["source_version"].unique().item() == "municipality:2010"
    assert out["support_mass"].sum() == pytest.approx(200)


def test_support_builder_refuses_an_expected_uncovered_unit():
    atoms = pl.DataFrame({"old": ["A"], "new": ["X"], "area": [1.0]})
    with pytest.raises(TransferError, match="expected target"):
        correspondence_from_support(
            atoms,
            source_column="old",
            target_column="new",
            mass_column="area",
            source_version="g:1",
            target_version="g:2",
            method="area_overlay",
            exact=False,
            expected_target_codes=["X", "Y"],
        )


def test_extensive_transfer_conserves_mass(correspondence):
    source = pl.DataFrame({"code": ["A", "B"], "cases": [100, 200]})
    out, diagnostics = transfer_extensive(
        source,
        correspondence,
        ["cases"],
        source_version="municipality:2010",
        target_version="municipality:2022",
        source_column="code",
        target_column="code",
    )
    assert out["cases"].sum() == pytest.approx(300)
    assert dict(zip(out["code"], out["cases"])) == pytest.approx(
        {"X": 125, "Y": 175}
    )
    assert diagnostics.kind is VariableKind.EXTENSIVE
    assert diagnostics.max_conservation_error == pytest.approx(0)


def test_intensive_transfer_uses_target_support(correspondence):
    source = pl.DataFrame({"code": ["A", "B"], "rain": [10.0, 20.0]})
    out, diagnostics = transfer_intensive(
        source,
        correspondence,
        ["rain"],
        source_version="climate-zones:v1",
        target_version="municipality:2022",
        source_column="code",
        target_column="code",
    )
    assert dict(zip(out["code"], out["rain"])) == pytest.approx(
        {"X": 12.5, "Y": 17.5}
    )
    assert diagnostics.kind is VariableKind.INTENSIVE


def test_rate_is_rebuilt_from_transferred_components(correspondence):
    source = pl.DataFrame(
        {"code": ["A", "B"], "cases": [10, 40], "population": [100, 200]}
    )
    out, diagnostics = transfer_rate(
        source,
        correspondence,
        numerator="cases",
        denominator="population",
        output="risk",
        source_version="municipality:2010",
        target_version="municipality:2022",
        source_column="code",
        target_column="code",
    )
    x = out.filter(pl.col("code") == "X").row(0, named=True)
    assert x["cases"] == pytest.approx(17.5)
    assert x["population"] == pytest.approx(125)
    assert x["risk"] == pytest.approx(0.14)
    assert diagnostics.kind is VariableKind.RATE


def test_transfer_operates_independently_by_time(correspondence):
    source = pl.DataFrame(
        {
            "code": ["A", "B", "A", "B"],
            "period": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
            ],
            "cases": [100, 200, 10, 20],
        }
    )
    out, _ = transfer_extensive(
        source,
        correspondence,
        ["cases"],
        source_version="g:1",
        target_version="g:2",
        source_column="code",
        target_column="code",
        group_columns=["period"],
    )
    totals = out.group_by("period").agg(pl.col("cases").sum()).sort("period")
    assert totals["cases"].to_list() == pytest.approx([300, 30])


def test_nonconservative_operator_is_rejected(correspondence):
    bad = correspondence.with_columns(
        pl.when((pl.col("source_code") == "A") & (pl.col("target_code") == "X"))
        .then(0.5)
        .otherwise(pl.col("source_fraction"))
        .alias("source_fraction")
    )
    with pytest.raises(TransferError, match="not closed"):
        validate_correspondence(bad, kind="extensive")
