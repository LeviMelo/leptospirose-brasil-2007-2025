from __future__ import annotations

import polars as pl
import pytest

from brepi.temporal import AnchorExpansionError, expand_anchors


def test_expand_anchors_preserves_policy_and_endpoint_provenance():
    anchors = pl.DataFrame(
        {
            "area": ["a", "a"],
            "anchor_year": [2010, 2020],
            "share": [0.2, 0.8],
            "source": ["old", "new"],
        }
    )
    out = expand_anchors(
        anchors,
        entities="area",
        time="anchor_year",
        years=[2009, 2010, 2015, 2020, 2021],
        value_columns=["share"],
        prefix="urban",
        provenance_columns=["source"],
    )
    assert out["share"].to_list() == pytest.approx([0.2, 0.2, 0.5, 0.8, 0.8])
    assert out["urban_method"].to_list() == [
        "nearest_anchor_backcast",
        "observed_anchor",
        "linear_between_anchors",
        "observed_anchor",
        "nearest_anchor_forecast",
    ]
    assert out.filter(pl.col("year") == 2015).row(
        0, named=True
    )["source_hi"] == "new"


def test_expand_anchors_refuses_single_anchor():
    with pytest.raises(AnchorExpansionError, match="fewer than two"):
        expand_anchors(
            pl.DataFrame({"area": ["a"], "year0": [2010], "x": [1.0]}),
            entities="area",
            time="year0",
            years=[2010],
            value_columns=["x"],
            prefix="x",
        )
