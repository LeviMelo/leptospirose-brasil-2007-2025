from __future__ import annotations

from brepi.sources.socioeconomic import (
    urban_selection_2010,
    urban_selection_2022,
)


def test_urban_selections_pin_census_population_universes():
    old = urban_selection_2010()
    new = urban_selection_2022()
    assert old.agregado == 202
    assert old.periods == ("2010",)
    assert old.classifications == {
        "2": ("0",),
        "1": ("0", "1", "2"),
    }
    assert new.agregado == 9923
    assert new.periods == ("2022",)
    assert new.classifications["1"] == ("6795", "1", "2")
