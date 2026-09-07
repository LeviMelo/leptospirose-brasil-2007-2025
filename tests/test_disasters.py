"""Tests for the disaster treatment layer.

Everything here runs on **synthetic** events whose correct answer is known by
construction. That is the point of keeping ``treatment.py`` pure: the
identification strategy is the part of the pipeline where a silent off-by-one in
the ``G`` variable would shift every event-study coefficient by a month and
nothing downstream would complain.

``atlas.py``'s network paths are not exercised here. Its *pure* pieces —
COBRADE normalisation and prefix matching — are, because those are what decide
which municipalities are treated at all.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from brepi.sources.disasters import atlas
from brepi.sources.disasters import treatment as T


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

#: Five synthetic municipalities with hand-designed treatment histories:
#:
#: A  one flood, 2008-03                      -> absorbing, single episode
#: B  floods 2008-03 and 2015-07              -> recurrent, two episodes
#: C  three filings inside 2010-05..2010-06   -> ONE episode (washout collapses)
#: D  a drought only (COBRADE 1.4.1.1.0)      -> never treated under a flood filter
#: E  nothing at all                          -> never treated
_EVENTS = [
    ("A", date(2008, 3, 11), "1.2.2.0.0", True, 2, 100),
    ("B", date(2008, 3, 4), "1.2.1.0.0", True, 0, 50),
    ("B", date(2015, 7, 20), "1.2.1.0.0", False, 1, 10),
    ("C", date(2010, 5, 2), "1.2.2.0.0", True, 0, 5),
    ("C", date(2010, 5, 28), "1.3.2.1.4", True, 0, 7),
    ("C", date(2010, 6, 15), "1.2.3.0.0", False, 0, 3),
    ("D", date(2012, 9, 1), "1.4.1.1.0", True, 0, 900),
]

MUNICIPALITIES = ["A", "B", "C", "D", "E"]


@pytest.fixture
def events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "munic_code": [e[0] for e in _EVENTS],
            "event_date": [e[1] for e in _EVENTS],
            "period": [e[1].replace(day=1) for e in _EVENTS],
            "cobrade": [e[2] for e in _EVENTS],
            "recognised": [e[3] for e in _EVENTS],
            "deaths": [e[4] for e in _EVENTS],
            "affected_total": [e[5] for e in _EVENTS],
        }
    )


@pytest.fixture
def flood(events: pl.DataFrame) -> pl.DataFrame:
    """Events restricted to the flood definition, as ``disaster_events`` would."""
    mask = pl.lit(False)
    for prefix in atlas.COBRADE_FLOOD:
        mask = mask | pl.col("cobrade").str.starts_with(prefix)
    return events.filter(mask)


# --------------------------------------------------------------------------
# COBRADE
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12200", "1.2.2.0.0"),
        ("1.2.2.0.0", "1.2.2.0.0"),
        (12200, "1.2.2.0.0"),
        ("11321", "1.1.3.2.1"),
        ("13214", "1.3.2.1.4"),
        ("", None),
        ("123", None),
        (None, None),
    ],
)
def test_normalise_cobrade(raw, expected):
    assert atlas.normalise_cobrade(raw) == expected


def test_cobrade_expr_matches_scalar():
    codes = ["12100", "12200", "12300", "13214", "11321", "14110", "bad"]
    df = pl.DataFrame({"Cod_Cobrade": codes}).with_columns(
        atlas.cobrade_expr().alias("dotted")
    )
    assert df["dotted"].to_list() == [atlas.normalise_cobrade(c) for c in codes]


def test_flood_definition_includes_chuvas_intensas():
    """The single decision that governs whether RS 2024 is 47 or 467 municipalities."""
    assert "1.3.2.1.4" in atlas.COBRADE_FLOOD
    assert "1.3.2.1.4" not in atlas.COBRADE_FLOOD_STRICT
    assert set(atlas.COBRADE_FLOOD_STRICT).issubset(set(atlas.COBRADE_FLOOD))


def test_every_declared_prefix_has_a_label():
    """A prefix that matches no labelled code is almost certainly a typo."""
    for prefix in (*atlas.COBRADE_FLOOD, *atlas.COBRADE_MASS_MOVEMENT, *atlas.COBRADE_DROUGHT):
        assert any(code.startswith(prefix) for code in atlas.COBRADE_LABELS), prefix


# --------------------------------------------------------------------------
# Period arithmetic
# --------------------------------------------------------------------------


def test_month_range_is_inclusive_and_first_of_month():
    months = T.month_range(date(2007, 1, 1), date(2007, 12, 31))
    assert len(months) == 12
    assert months[0] == date(2007, 1, 1) and months[-1] == date(2007, 12, 1)
    assert all(m.day == 1 for m in months)


def test_month_range_spans_2007_2025():
    assert len(T.month_range()) == 19 * 12 == 228


def test_month_range_rejects_reversed_window():
    with pytest.raises(T.TreatmentError):
        T.month_range(date(2010, 1, 1), date(2009, 1, 1))


# --------------------------------------------------------------------------
# The G variable
# --------------------------------------------------------------------------


def test_first_treatment_period_g_values(flood: pl.DataFrame):
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES).sort("munic_code")
    got = {r["munic_code"]: r["g_period"] for r in g.to_dicts()}
    assert got == {
        "A": date(2008, 3, 1),
        "B": date(2008, 3, 1),
        "C": date(2010, 5, 1),
        "D": None,  # drought only — not a flood
        "E": None,  # no events at all
    }


def test_never_treated_g_index_is_zero_not_null(flood: pl.DataFrame):
    """CS's ``gname`` convention: never-treated must be 0, and 0 must be unreachable."""
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES)
    never = g.filter(~pl.col("ever_treated"))
    assert never["g_index"].to_list() == [0, 0]
    assert never["g_period"].null_count() == 2
    assert g.filter(pl.col("ever_treated"))["g_index"].min() > 0


def test_g_index_is_one_based_relative_to_spine_time_index(flood: pl.DataFrame):
    """``g_index == time_index + 1`` exactly, for every treated unit."""
    start = date(2007, 1, 1)
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES, start=start)
    for row in g.filter(pl.col("ever_treated")).to_dicts():
        assert row["g_index"] == T.months_between(start, row["g_period"]) + 1


def test_first_treatment_uses_the_window_start_as_origin(flood: pl.DataFrame):
    """Shifting the window shifts g_index by exactly the shift, not by anything else."""
    a = T.first_treatment_period(flood, municipalities=MUNICIPALITIES, start=date(2007, 1, 1))
    b = T.first_treatment_period(flood, municipalities=MUNICIPALITIES, start=date(2008, 1, 1))
    a_t = a.filter(pl.col("ever_treated")).sort("munic_code")
    b_t = b.filter(pl.col("ever_treated")).sort("munic_code")
    assert (a_t["g_index"] - b_t["g_index"]).unique().to_list() == [12]


def test_never_treated_requires_the_lattice(flood: pl.DataFrame):
    """Omitting ``municipalities`` makes every unit treated — the trap being guarded."""
    without = T.first_treatment_period(flood)
    assert without["ever_treated"].all()
    assert without.height == 3
    with_lattice = T.first_treatment_period(flood, municipalities=MUNICIPALITIES)
    assert with_lattice.height == 5
    assert int((~with_lattice["ever_treated"]).sum()) == 2


def test_cobrade_prefixes_refilter_at_the_treatment_layer(events: pl.DataFrame):
    """One download, two treatment definitions."""
    broad = T.first_treatment_period(
        events, atlas.COBRADE_FLOOD, municipalities=MUNICIPALITIES
    )
    strict = T.first_treatment_period(
        events, atlas.COBRADE_FLOOD_STRICT, municipalities=MUNICIPALITIES
    )
    # C's May filing under 1.3.2.1.4 is also matched by a 1.2.x filing that month,
    # so C's G is unchanged; what changes is the event count behind it.
    assert broad.filter(pl.col("munic_code") == "C")["n_events"].item() == 3
    assert strict.filter(pl.col("munic_code") == "C")["n_events"].item() == 2


def test_recognised_only_shrinks_the_treated_set(events: pl.DataFrame):
    """B's 2015 flood is unrecognised, so B loses an episode under recognition."""
    both = T.first_treatment_period(
        events, atlas.COBRADE_FLOOD, municipalities=MUNICIPALITIES
    )
    rec = T.first_treatment_period(
        events, atlas.COBRADE_FLOOD, municipalities=MUNICIPALITIES, recognised_only=True
    )
    assert both.filter(pl.col("munic_code") == "B")["n_episodes"].item() == 2
    assert rec.filter(pl.col("munic_code") == "B")["n_episodes"].item() == 1


def test_severity_filter_can_empty_the_treated_set(flood: pl.DataFrame):
    g = T.first_treatment_period(
        flood, municipalities=MUNICIPALITIES, min_deaths=1000
    )
    assert int(g["ever_treated"].sum()) == 0
    assert g["g_index"].to_list() == [0, 0, 0, 0, 0]


# --------------------------------------------------------------------------
# Episodes: the recurrent-vs-absorbing distinction
# --------------------------------------------------------------------------


def test_washout_collapses_a_burst_into_one_episode(flood: pl.DataFrame):
    """C files three protocols across two adjacent months: one flood, not three."""
    eps = T.episodes(flood, gap_months=3)
    c = eps.filter(pl.col("munic_code") == "C")
    assert c.height == 1
    assert c["n_events"].item() == 3
    assert c["episode_start"].item() == date(2010, 5, 1)
    assert c["episode_end"].item() == date(2010, 6, 1)


def test_zero_washout_separates_adjacent_months(flood: pl.DataFrame):
    """gap_months is a real knob: at 0, C's May and June filings are two episodes."""
    eps = T.episodes(flood, gap_months=0)
    assert eps.filter(pl.col("munic_code") == "C").height == 2


def test_distant_floods_are_separate_episodes(flood: pl.DataFrame):
    eps = T.episodes(flood, gap_months=3)
    assert eps.filter(pl.col("munic_code") == "B").height == 2
    assert eps.filter(pl.col("munic_code") == "A").height == 1


def test_episode_count_reaches_the_g_table(flood: pl.DataFrame):
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES)
    counts = {r["munic_code"]: r["n_episodes"] for r in g.to_dicts()}
    assert counts == {"A": 1, "B": 2, "C": 1, "D": 0, "E": 0}


def test_episodes_on_empty_events_returns_empty_frame():
    empty = pl.DataFrame(schema={"munic_code": pl.Utf8, "period": pl.Date})
    assert T.episodes(empty).height == 0


# --------------------------------------------------------------------------
# Panel and treatment indicators
# --------------------------------------------------------------------------


def test_panel_is_complete_and_zero_filled(flood: pl.DataFrame):
    start, end = date(2008, 1, 1), date(2008, 12, 31)
    panel = T.to_municipality_month(
        flood, municipalities=MUNICIPALITIES, start=start, end=end
    )
    assert panel.height == 5 * 12
    assert panel["n_events"].null_count() == 0
    assert int(panel["event"].sum()) == 2  # A and B, both 2008-03
    assert panel.filter(pl.col("munic_code") == "E")["n_events"].sum() == 0


def test_panel_time_index_matches_spine_convention(flood: pl.DataFrame):
    panel = T.to_municipality_month(
        flood, municipalities=MUNICIPALITIES, start=date(2007, 1, 1), end=date(2007, 3, 31)
    )
    assert sorted(panel["time_index"].unique().to_list()) == [0, 1, 2]


def test_panel_aggregates_impacts(flood: pl.DataFrame):
    panel = T.to_municipality_month(
        flood, municipalities=MUNICIPALITIES, start=date(2010, 5, 1), end=date(2010, 5, 31)
    )
    c = panel.filter(pl.col("munic_code") == "C")
    assert c["n_events"].item() == 2  # two filings in May
    assert c["affected_total"].item() == 12


def test_absorbing_treatment_never_switches_off(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    out = T.treatment_indicator(panel, definition="absorbing")
    b = out.filter(pl.col("munic_code") == "B").sort("period")
    treated = b["treated"].to_list()
    # Monotone: once true, always true.
    assert treated == sorted(treated)
    assert not b.filter(pl.col("period") == date(2008, 2, 1))["treated"].item()
    assert b.filter(pl.col("period") == date(2008, 3, 1))["treated"].item()
    assert b.filter(pl.col("period") == date(2012, 1, 1))["treated"].item()  # between floods


def test_recurrent_treatment_switches_off_again(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    out = T.treatment_indicator(panel, definition="recurrent", duration_months=3)
    b = out.filter(pl.col("munic_code") == "B").sort("period")
    assert b.filter(pl.col("period") == date(2008, 3, 1))["treated"].item()
    assert b.filter(pl.col("period") == date(2008, 5, 1))["treated"].item()  # still in window
    assert not b.filter(pl.col("period") == date(2008, 6, 1))["treated"].item()  # window closed
    assert not b.filter(pl.col("period") == date(2012, 1, 1))["treated"].item()
    assert b.filter(pl.col("period") == date(2015, 7, 1))["treated"].item()  # second flood


def test_absorbing_and_recurrent_disagree_and_that_is_the_point(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    ab = T.treatment_indicator(panel, definition="absorbing")["treated"].sum()
    rc = T.treatment_indicator(panel, definition="recurrent")["treated"].sum()
    assert ab > rc


def test_never_treated_are_untreated_under_both_definitions(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    for definition in ("absorbing", "recurrent"):
        out = T.treatment_indicator(panel, definition=definition)
        for code in ("D", "E"):
            unit = out.filter(pl.col("munic_code") == code)
            assert not unit["treated"].any()
            assert unit["event_time"].null_count() == unit.height


def test_event_time_is_months_since_first_treatment(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    a = T.treatment_indicator(panel).filter(pl.col("munic_code") == "A")
    assert a.filter(pl.col("period") == date(2008, 3, 1))["event_time"].item() == 0
    assert a.filter(pl.col("period") == date(2008, 4, 1))["event_time"].item() == 1
    assert a.filter(pl.col("period") == date(2007, 3, 1))["event_time"].item() == -12


def test_unknown_definition_raises(flood: pl.DataFrame):
    panel = T.to_municipality_month(flood, municipalities=MUNICIPALITIES)
    with pytest.raises(T.TreatmentError):
        T.treatment_indicator(panel, definition="staggered")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Cohorts and feasibility
# --------------------------------------------------------------------------


def test_cohort_table_keeps_the_never_treated_row(flood: pl.DataFrame):
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES)
    cohorts = T.treatment_cohorts(g)
    never = cohorts.filter(pl.col("never_treated"))
    assert never.height == 1
    assert never["n_municipalities"].item() == 2
    assert cohorts["n_municipalities"].sum() == 5


def test_cohort_shares_and_cumulative(flood: pl.DataFrame):
    g = T.first_treatment_period(flood, municipalities=MUNICIPALITIES)
    cohorts = T.treatment_cohorts(g).filter(~pl.col("never_treated")).sort("cohort")
    assert cohorts["cohort"].to_list() == [2008, 2010]
    assert cohorts["n_municipalities"].to_list() == [2, 1]
    assert cohorts["cumulative_treated"].to_list() == [2, 3]
    assert cohorts["share"].sum() == pytest.approx(3 / 5)


def test_feasibility_flags_too_few_treated_as_blocking(flood: pl.DataFrame):
    report = T.did_feasibility_report(flood, municipalities=MUNICIPALITIES)
    assert report.n_units == 5
    assert report.n_treated == 3
    assert report.n_never_treated == 2
    assert not report.feasible
    assert any(w.startswith("BLOCKING") for w in report.warnings)


def test_feasibility_measures_recurrence(flood: pl.DataFrame):
    report = T.did_feasibility_report(flood, municipalities=MUNICIPALITIES)
    assert report.n_recurrent_units == 1  # only B
    assert report.share_recurrent_of_treated == pytest.approx(1 / 3)
    assert report.mean_episodes_per_treated == pytest.approx(4 / 3)


def test_feasibility_blocks_when_no_never_treated():
    """A design with no clean controls must say so rather than quietly proceed."""
    ev = pl.DataFrame(
        {
            "munic_code": [f"M{i:03d}" for i in range(60)],
            "period": [date(2008 + i % 10, 1 + i % 12, 1) for i in range(60)],
            "cobrade": ["1.2.1.0.0"] * 60,
        }
    )
    report = T.did_feasibility_report(ev, municipalities=ev["munic_code"].to_list())
    assert report.n_never_treated == 0
    assert not report.feasible
    assert any("no never-treated units" in w for w in report.warnings)


def test_feasibility_blocks_when_timing_does_not_vary():
    """Every unit treated in the same month is a two-group DiD, not a staggered one."""
    ev = pl.DataFrame(
        {
            "munic_code": [f"M{i:03d}" for i in range(60)],
            "period": [date(2010, 5, 1)] * 60,
            "cobrade": ["1.2.1.0.0"] * 60,
        }
    )
    codes = ev["munic_code"].to_list() + [f"C{i:03d}" for i in range(200)]
    report = T.did_feasibility_report(ev, municipalities=codes)
    assert report.n_cohorts == 1
    assert report.largest_cohort_share == pytest.approx(1.0)
    assert not report.feasible


def test_feasibility_passes_a_well_behaved_design():
    """60 units entering across 12 distinct months, 200 clean controls, no recurrence."""
    ev = pl.DataFrame(
        {
            "munic_code": [f"M{i:03d}" for i in range(60)],
            "period": [date(2010, 1 + i % 12, 1) for i in range(60)],
            "cobrade": ["1.2.1.0.0"] * 60,
        }
    )
    codes = ev["munic_code"].to_list() + [f"C{i:03d}" for i in range(200)]
    report = T.did_feasibility_report(ev, municipalities=codes)
    assert report.feasible
    assert report.n_treated == 60
    assert report.n_never_treated == 200
    assert report.n_cohorts == 12
    assert report.n_recurrent_units == 0
    assert "units" in report.summary()


def test_feasibility_dict_round_trips(flood: pl.DataFrame):
    d = T.did_feasibility_report(flood, municipalities=MUNICIPALITIES).to_dict()
    assert d["n_units"] == 5 and "feasible" in d


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


def test_missing_columns_raise_rather_than_produce_a_wrong_panel():
    bad = pl.DataFrame({"munic_code": ["A"], "when": [date(2010, 1, 1)]})
    with pytest.raises(T.TreatmentError):
        T.to_municipality_month(bad, municipalities=["A"])
    with pytest.raises(T.TreatmentError):
        T.first_treatment_period(bad, municipalities=["A"])


def test_cobrade_filter_without_a_cobrade_column_raises():
    bad = pl.DataFrame({"munic_code": ["A"], "period": [date(2010, 1, 1)]})
    with pytest.raises(T.TreatmentError):
        T.first_treatment_period(bad, atlas.COBRADE_FLOOD, municipalities=["A"])


def test_events_outside_the_window_do_not_create_treatment(flood: pl.DataFrame):
    g = T.first_treatment_period(
        flood, municipalities=MUNICIPALITIES, start=date(2020, 1, 1), end=date(2021, 12, 31)
    )
    assert int(g["ever_treated"].sum()) == 0
