from __future__ import annotations

import polars as pl
import pytest

from brepi.sources.sanitation.census import (
    SanitationError,
    expand_census_anchors,
    sewage_selection_2010,
    sewage_selection_2022,
)


def test_sewage_selections_pin_harmonised_exact_network_categories():
    old = sewage_selection_2010()
    new = sewage_selection_2022()
    assert old.agregado == 1394
    assert tuple(old.periods) == ("2010",)
    assert old.classifications["11558"][:2] == ("0", "92855")
    assert new.agregado == 6805
    assert tuple(new.periods) == ("2022",)
    assert new.classifications["11558"][:2] == ("46292", "72110")


def test_expand_census_anchors_marks_every_policy():
    anchors = pl.DataFrame(
        {
            "munic_code": ["1", "1"],
            "year": [2010, 2022],
            "sanitation_sewer_share": [0.2, 0.8],
        }
    )
    out = expand_census_anchors(anchors, years=[2009, 2010, 2016, 2022, 2025])
    assert out["sanitation_sewer_share"].to_list() == pytest.approx(
        [0.2, 0.2, 0.5, 0.8, 0.8]
    )
    assert out["sanitation_method"].to_list() == [
        "nearest_anchor_backcast",
        "observed_anchor",
        "linear_between_censuses",
        "observed_anchor",
        "nearest_anchor_forecast",
    ]


def test_expand_census_anchors_refuses_single_anchor():
    anchors = pl.DataFrame(
        {
            "munic_code": ["1"],
            "year": [2022],
            "sanitation_sewer_share": [0.8],
        }
    )
    with pytest.raises(SanitationError, match="fewer than two"):
        expand_census_anchors(anchors)
