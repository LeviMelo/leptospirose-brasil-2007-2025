"""Turning disaster events into a staggered-adoption treatment design.

Everything here is a **pure transformation** of an events frame: no network, no
cache, no config. That is deliberate — the identification strategy is the part
that most needs to be unit-tested against synthetic data whose right answer is
known by construction, and a function that downloads 86 MB before it can be
tested does not get tested.

The design this serves
----------------------
Callaway & Sant'Anna (2021) estimate group-time average treatment effects
ATT(g,t), where ``g`` is the period in which a unit is **first** treated and
never-treated (or not-yet-treated) units supply the counterfactual. Three
things must hold, and each is measured rather than assumed:

1. **Enough never-treated units.** With ~5,570 municipalities and a broad flood
   definition, the never-treated set can collapse. If it does, the estimator
   falls back to not-yet-treated controls, which is valid but changes the
   estimand's support and weakens late-period identification.
2. **Variation in timing.** If every unit enters in one or two cohorts the design
   degenerates to a two-group DiD and the "staggered" machinery buys nothing.
3. **Absorbing treatment.** This is the assumption that Brazilian flooding
   *violates*. Municipalities flood repeatedly. Canonical CS defines ``G`` as
   the first treatment and thereafter ignores what happens — so ATT(g,t) for
   large ``t-g`` is not "the effect of one flood at lag t-g" but "the effect of
   having flooded at g, in a unit that has probably flooded again since".

   :func:`did_feasibility_report` measures how bad this is.
   :func:`treatment_indicator` implements **both** readings explicitly:
   ``"absorbing"`` (once treated, always treated — the CS-canonical variable)
   and ``"recurrent"`` (an on/off indicator that returns to zero, appropriate
   for an event-study or a distributed-lag specification but *not* for CS).

Never write a design that silently picks one. The two answers differ and the
difference is a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal, Sequence

import polars as pl

TreatmentDefinition = Literal["absorbing", "recurrent"]

#: Default analysis window. 2007 is where the SINAN NET series begins; 2025 is
#: the last complete year in the Atlas release.
DEFAULT_START = date(2007, 1, 1)
DEFAULT_END = date(2025, 12, 31)

#: Default washout used to collapse a burst of filings into one episode. A big
#: flood generates several S2iD protocols in the same municipality across
#: adjacent weeks (different COBRADE codes, a registration then a recognition
#: follow-up); counting those as distinct treatments would manufacture
#: recurrence that is bureaucratic rather than hydrological.
DEFAULT_EPISODE_GAP_MONTHS = 3


class TreatmentError(ValueError):
    """The treatment specification cannot be satisfied by the events given."""


# --------------------------------------------------------------------------
# Period arithmetic
# --------------------------------------------------------------------------


def month_range(start: date = DEFAULT_START, end: date = DEFAULT_END) -> list[date]:
    """Every first-of-month from ``start`` to ``end`` inclusive."""
    if end < start:
        raise TreatmentError("end precedes start")
    out: list[date] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def months_between(a: date, b: date) -> int:
    """Whole months from ``a`` to ``b``, both truncated to first-of-month."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def _period_index_map(periods: Sequence[date]) -> dict[date, int]:
    """``{period: 0-based index}``, matching ``spine.build_spine``'s time_index."""
    return {p: i for i, p in enumerate(periods)}


# --------------------------------------------------------------------------
# Municipality x month panel
# --------------------------------------------------------------------------


def to_municipality_month(
    events: pl.DataFrame,
    *,
    municipalities: Iterable[str] | None = None,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    recognised_only: bool = False,
    min_deaths: int | None = None,
    min_affected: int | None = None,
) -> pl.DataFrame:
    """Collapse events to a complete municipality x month panel.

    The panel is **complete** — every municipality x month in the window gets a
    row, with zeros where nothing happened. That is the spine discipline of
    :mod:`brepi.panel.spine` applied here: a municipality-month with no flood is
    a control observation, not a missing one, and a panel assembled by grouping
    the events alone would contain only treated cells and silently make the
    never-treated set empty.

    ``municipalities`` should be the full lattice (all 5,570). If omitted, only
    municipalities appearing in ``events`` are enumerated, which is correct for
    a within-treated analysis and *wrong* for the never-treated count — so the
    caller is nudged to pass the lattice explicitly.

    Severity filters (``recognised_only``, ``min_deaths``, ``min_affected``)
    apply to individual events before aggregation. They are the honest lever for
    "how big must a flood be to count as treatment"; the answer changes the
    treated set substantially and should be varied in a sensitivity table.
    """
    _require_columns(events, ("munic_code", "period"))

    ev = events
    if recognised_only and "recognised" in ev.columns:
        ev = ev.filter(pl.col("recognised"))
    if min_deaths is not None:
        ev = ev.filter(pl.col("deaths") >= min_deaths)
    if min_affected is not None:
        ev = ev.filter(pl.col("affected_total") >= min_affected)

    periods = month_range(start, end)
    ev = ev.filter(pl.col("period").is_between(periods[0], periods[-1]))

    if municipalities is None:
        codes = sorted(set(ev["munic_code"].to_list()))
    else:
        codes = sorted(set(municipalities))
    if not codes:
        raise TreatmentError("no municipalities to build a panel over")

    agg_exprs = [
        pl.len().alias("n_events"),
        pl.col("cobrade").n_unique().alias("n_cobrade_types"),
    ]
    for col in ("deaths", "injured", "homeless", "displaced", "missing", "affected_total"):
        if col in ev.columns:
            agg_exprs.append(pl.col(col).sum().alias(col))
    for col in ("damage_material_brl", "damage_economic_brl"):
        if col in ev.columns:
            agg_exprs.append(pl.col(col).sum().alias(col))
    if "recognised" in ev.columns:
        agg_exprs.append(pl.col("recognised").sum().cast(pl.Int64).alias("n_recognised"))

    grouped = ev.group_by("munic_code", "period").agg(agg_exprs)

    panel = (
        pl.DataFrame({"munic_code": codes})
        .join(pl.DataFrame({"period": periods}), how="cross")
        .join(grouped, on=["munic_code", "period"], how="left")
    )

    fill_int = [c for c in panel.columns if c not in ("munic_code", "period") and not c.endswith("_brl")]
    panel = panel.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int64) for c in fill_int]
        + [pl.col(c).fill_null(0.0) for c in panel.columns if c.endswith("_brl")]
    ).with_columns(
        (pl.col("n_events") > 0).alias("event"),
        pl.col("period").dt.year().cast(pl.Int32).alias("year"),
        pl.col("period")
        .replace_strict(_period_index_map(periods), return_dtype=pl.Int32)
        .alias("time_index"),
    )
    return panel.sort(["munic_code", "period"])


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


def episodes(
    events: pl.DataFrame,
    *,
    gap_months: int = DEFAULT_EPISODE_GAP_MONTHS,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
) -> pl.DataFrame:
    """Collapse a municipality's events into distinct treatment episodes.

    Two events in the same municipality belong to the same episode when they are
    at most ``gap_months`` apart. With the default of three months, the ~470 RS
    filings of May 2024 become one episode per municipality rather than several,
    and a municipality that floods every rainy season still shows one episode a
    year.

    Returns one row per ``(munic_code, episode)`` with ``episode_start``,
    ``episode_end``, ``n_events`` and summed impacts. ``episode`` is 1-based.

    The choice of ``gap_months`` is not innocent: it is what separates "this
    municipality was treated once" from "twice", and therefore what determines
    the recurrence share that :func:`did_feasibility_report` reports. Vary it.
    """
    _require_columns(events, ("munic_code", "period"))
    if gap_months < 0:
        raise TreatmentError("gap_months must be non-negative")

    lo, hi = month_range(start, end)[0], month_range(start, end)[-1]
    ev = events.filter(pl.col("period").is_between(lo, hi))
    if ev.height == 0:
        return pl.DataFrame(
            schema={
                "munic_code": pl.Utf8,
                "episode": pl.Int32,
                "episode_start": pl.Date,
                "episode_end": pl.Date,
                "n_events": pl.Int64,
            }
        )

    origin = lo
    ev = ev.with_columns(
        (
            (pl.col("period").dt.year() - origin.year) * 12
            + (pl.col("period").dt.month() - origin.month)
        )
        .cast(pl.Int32)
        .alias("_m")
    ).sort(["munic_code", "_m"])

    ev = ev.with_columns(
        (
            (pl.col("_m") - pl.col("_m").shift(1).over("munic_code")) > gap_months
        )
        .fill_null(True)
        .alias("_new_episode")
    ).with_columns(
        pl.col("_new_episode").cum_sum().over("munic_code").cast(pl.Int32).alias("episode")
    )

    agg = [
        pl.col("period").min().alias("episode_start"),
        pl.col("period").max().alias("episode_end"),
        pl.len().alias("n_events"),
    ]
    for col in ("deaths", "affected_total", "damage_economic_brl"):
        if col in ev.columns:
            agg.append(pl.col(col).sum().alias(col))
    if "recognised" in ev.columns:
        agg.append(pl.col("recognised").any().alias("any_recognised"))

    return ev.group_by("munic_code", "episode").agg(agg).sort(["munic_code", "episode"])


# --------------------------------------------------------------------------
# The G variable
# --------------------------------------------------------------------------


def first_treatment_period(
    events: pl.DataFrame,
    cobrade_prefixes: Sequence[str] | None = None,
    *,
    municipalities: Iterable[str] | None = None,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    recognised_only: bool = False,
    min_deaths: int | None = None,
    min_affected: int | None = None,
) -> pl.DataFrame:
    """The Callaway-Sant'Anna ``G``: the period of each unit's first treatment.

    Returns one row per municipality with

    ``g_period``     first-of-month date of first treatment; ``null`` if never.
    ``g_index``      **1-based** period index, and ``0`` for never-treated. This
                     is the CS convention (``gname`` in the ``did`` R package):
                     zero must be a value no treated unit can take, so the index
                     cannot be the 0-based ``time_index`` the spine uses.
                     ``g_index = time_index + 1`` for treated units, exactly.
    ``ever_treated`` boolean.
    ``n_events`` / ``n_episodes`` / ``last_period``  for the recurrence audit.

    ``cobrade_prefixes`` re-filters the events, so a single ``disaster_events``
    pull can generate the strict-1.2.x and the flood-plus-chuvas-intensas ``G``
    variables without a second download.

    Never-treated units must be enumerated from the **lattice**, not from the
    events: pass ``municipalities``. Omitting it produces a table in which every
    unit is treated, which is not a bug you will notice downstream.
    """
    _require_columns(events, ("munic_code", "period"))

    ev = events
    if cobrade_prefixes:
        if "cobrade" not in ev.columns:
            raise TreatmentError("events lack a 'cobrade' column to filter on")
        mask = pl.lit(False)
        for prefix in cobrade_prefixes:
            mask = mask | pl.col("cobrade").str.starts_with(prefix)
        ev = ev.filter(mask)
    if recognised_only and "recognised" in ev.columns:
        ev = ev.filter(pl.col("recognised"))
    if min_deaths is not None and "deaths" in ev.columns:
        ev = ev.filter(pl.col("deaths") >= min_deaths)
    if min_affected is not None and "affected_total" in ev.columns:
        ev = ev.filter(pl.col("affected_total") >= min_affected)

    periods = month_range(start, end)
    ev = ev.filter(pl.col("period").is_between(periods[0], periods[-1]))

    codes = sorted(set(municipalities)) if municipalities is not None else sorted(
        set(ev["munic_code"].to_list())
    )
    index = _period_index_map(periods)

    first = ev.group_by("munic_code").agg(
        pl.col("period").min().alias("g_period"),
        pl.col("period").max().alias("last_period"),
        pl.len().alias("n_events"),
    )
    eps = episodes(ev, start=start, end=end)
    n_eps = (
        eps.group_by("munic_code").agg(pl.col("episode").max().alias("n_episodes"))
        if eps.height
        else pl.DataFrame(schema={"munic_code": pl.Utf8, "n_episodes": pl.Int32})
    )

    out = (
        pl.DataFrame({"munic_code": codes})
        .join(first, on="munic_code", how="left")
        .join(n_eps, on="munic_code", how="left")
        .with_columns(
            pl.col("n_events").fill_null(0).cast(pl.Int64),
            pl.col("n_episodes").fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            pl.col("g_period").is_not_null().alias("ever_treated"),
            pl.col("g_period")
            .replace_strict(index, default=None, return_dtype=pl.Int32)
            .alias("_g0"),
        )
        .with_columns(
            (pl.col("_g0") + 1).fill_null(0).cast(pl.Int32).alias("g_index"),
            pl.col("g_period").dt.year().cast(pl.Int32).alias("g_year"),
        )
        .drop("_g0")
    )
    return out.sort("munic_code")


def treatment_indicator(
    panel: pl.DataFrame,
    *,
    definition: TreatmentDefinition = "absorbing",
    duration_months: int = 1,
) -> pl.DataFrame:
    """Attach a ``treated`` column under one of the two treatment readings.

    ``"absorbing"``
        ``treated`` turns on at the unit's first event and never turns off. This
        is what Callaway-Sant'Anna's ``G`` implies and the only definition
        consistent with the canonical estimator. Interpret ATT(g,t) at long
        horizons accordingly: it is the effect of *having become* a flood
        municipality, cumulative over any subsequent floods.

    ``"recurrent"``
        ``treated`` is on for ``duration_months`` from each event and off
        otherwise. Appropriate for an event-study / distributed-lag / two-way
        fixed-effects-with-lags specification. **Feeding this to CS is a
        specification error**, because CS reads a unit that switches off as a
        control for its own future self.

    Also emits ``event_time`` (months since first treatment; ``null`` for
    never-treated) which is what an event-study plot is drawn against.
    """
    _require_columns(panel, ("munic_code", "period", "event"))
    if duration_months < 1:
        raise TreatmentError("duration_months must be at least 1")

    p = panel.sort(["munic_code", "period"])
    first = (
        p.filter(pl.col("event"))
        .group_by("munic_code")
        .agg(pl.col("period").min().alias("g_period"))
    )
    p = p.join(first, on="munic_code", how="left")

    if definition == "absorbing":
        treated = (pl.col("g_period").is_not_null()) & (pl.col("period") >= pl.col("g_period"))
    elif definition == "recurrent":
        # On for `duration_months` starting at each event month: a rolling max of
        # the event flag over the trailing window.
        treated = (
            pl.col("event")
            .cast(pl.Int8)
            .rolling_max(window_size=duration_months, min_samples=1)
            .over("munic_code")
            .cast(pl.Boolean)
        )
    else:
        raise TreatmentError(
            f"definition must be 'absorbing' or 'recurrent', got {definition!r}"
        )

    return p.with_columns(
        treated.alias("treated"),
        pl.lit(definition).alias("treatment_definition"),
    ).with_columns(
        pl.when(pl.col("g_period").is_null())
        .then(None)
        .otherwise(
            (pl.col("period").dt.year() - pl.col("g_period").dt.year()) * 12
            + (pl.col("period").dt.month() - pl.col("g_period").dt.month())
        )
        .cast(pl.Int32)
        .alias("event_time")
    )


def treatment_cohorts(g: pl.DataFrame, *, by: Literal["year", "month"] = "year") -> pl.DataFrame:
    """How many municipalities enter first treatment in each period.

    Takes the output of :func:`first_treatment_period`. Never-treated units are
    reported as a row with ``cohort = null``, deliberately kept in the table:
    the size of that row relative to the rest *is* the feasibility of the design,
    and a cohort table that quietly omits it flatters every study that prints one.
    """
    _require_columns(g, ("munic_code", "g_period", "ever_treated"))
    key = (
        pl.col("g_period").dt.year().cast(pl.Int32)
        if by == "year"
        else pl.col("g_period").dt.truncate("1mo")
    )
    total = g.height
    out = (
        g.with_columns(key.alias("cohort"))
        .group_by("cohort")
        .agg(
            pl.len().alias("n_municipalities"),
            pl.col("n_events").sum().alias("n_events"),
            pl.col("n_episodes").mean().alias("mean_episodes"),
        )
        .sort("cohort", nulls_last=True)
    )
    return out.with_columns(
        (pl.col("n_municipalities") / total).alias("share"),
        pl.col("cohort").is_null().alias("never_treated"),
        pl.when(pl.col("cohort").is_null())
        .then(None)
        .otherwise(pl.col("n_municipalities").cum_sum())
        .alias("cumulative_treated"),
    )


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


@dataclass
class DiDFeasibility:
    """Whether a staggered-adoption design is actually identified on these data."""

    n_units: int
    n_treated: int
    n_never_treated: int
    share_never_treated: float
    n_cohorts: int
    largest_cohort_share: float
    first_cohort: date | None
    last_cohort: date | None
    n_periods: int
    n_units_with_pre_periods: int
    n_recurrent_units: int
    share_recurrent_of_treated: float
    mean_episodes_per_treated: float
    median_gap_months: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """No blocking warning. Advisory warnings do not clear ``feasible``."""
        return not any(w.startswith("BLOCKING") for w in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["feasible"] = self.feasible
        return d

    def summary(self) -> str:
        lines = [
            f"units                {self.n_units:,}",
            f"treated              {self.n_treated:,} ({1 - self.share_never_treated:.1%})",
            f"never treated        {self.n_never_treated:,} ({self.share_never_treated:.1%})",
            f"cohorts              {self.n_cohorts} "
            f"({self.first_cohort} .. {self.last_cohort}); "
            f"largest holds {self.largest_cohort_share:.1%} of treated",
            f"periods              {self.n_periods}",
            f"treated with pre-period  {self.n_units_with_pre_periods:,}",
            f"recurrently treated  {self.n_recurrent_units:,} "
            f"({self.share_recurrent_of_treated:.1%} of treated), "
            f"mean {self.mean_episodes_per_treated:.2f} episodes, "
            f"median gap {self.median_gap_months} months",
        ]
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def did_feasibility_report(
    events: pl.DataFrame,
    *,
    municipalities: Iterable[str] | None = None,
    cobrade_prefixes: Sequence[str] | None = None,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    gap_months: int = DEFAULT_EPISODE_GAP_MONTHS,
    recognised_only: bool = False,
    min_never_treated: int = 100,
    min_cohorts: int = 5,
    max_largest_cohort_share: float = 0.5,
    max_recurrence_share: float = 0.5,
) -> DiDFeasibility:
    """Answer, with numbers, whether the staggered DiD is identified here.

    The four questions, and the thresholds each is judged against:

    * **Enough treated units?** Below ~30 the group-time ATTs are estimated on
      handfuls and the multiplier bootstrap is unreliable.
    * **Enough never-treated controls?** ``min_never_treated``. Falling below it
      is not fatal — CS can use not-yet-treated units — but it must be a stated
      choice, because the two control groups answer slightly different questions
      and the not-yet-treated group empties out in late periods.
    * **Variation in timing?** ``min_cohorts`` distinct entry periods, and no
      single cohort holding more than ``max_largest_cohort_share`` of the
      treated. A design where 80% of units enter in one month is a two-group
      DiD wearing a costume.
    * **Recurrence.** The share of treated units with more than one episode. This
      is the assumption-violation that matters most for Brazilian flooding, and
      it is reported whether or not it trips a threshold, because the honest
      response is usually to report both the absorbing and the recurrent-event
      specification rather than to pick the flattering one.

    Warnings prefixed ``BLOCKING`` clear :attr:`DiDFeasibility.feasible`;
    warnings prefixed ``ADVISORY`` do not.
    """
    g = first_treatment_period(
        events,
        cobrade_prefixes,
        municipalities=municipalities,
        start=start,
        end=end,
        recognised_only=recognised_only,
    )
    periods = month_range(start, end)
    n_units = g.height
    treated = g.filter(pl.col("ever_treated"))
    n_treated = treated.height
    n_never = n_units - n_treated

    cohort_sizes = treated.group_by(pl.col("g_period")).len()
    n_cohorts = cohort_sizes.height
    largest = (
        float(cohort_sizes["len"].max() / n_treated) if n_treated and n_cohorts else 0.0
    )

    # A unit whose first treatment is in period 1 contributes no pre-period and
    # is dropped from every ATT(g,t) that needs one.
    n_with_pre = int(treated.filter(pl.col("g_period") > periods[0]).height)

    recurrent = treated.filter(pl.col("n_episodes") > 1)
    n_recurrent = recurrent.height
    mean_eps = float(treated["n_episodes"].mean()) if n_treated else 0.0

    eps = episodes(events, gap_months=gap_months, start=start, end=end)
    median_gap: float | None = None
    if eps.height:
        gaps = (
            eps.sort(["munic_code", "episode"])
            .with_columns(
                (
                    (pl.col("episode_start").dt.year() - pl.col("episode_end").shift(1).over("munic_code").dt.year()) * 12
                    + (pl.col("episode_start").dt.month() - pl.col("episode_end").shift(1).over("munic_code").dt.month())
                ).alias("gap")
            )["gap"]
            .drop_nulls()
        )
        if gaps.len():
            median_gap = float(gaps.median())

    warnings: list[str] = []
    if n_treated < 30:
        warnings.append(
            f"BLOCKING: only {n_treated} treated units; group-time ATTs and the "
            "multiplier bootstrap are not credible at this size."
        )
    if n_never == 0:
        warnings.append(
            "BLOCKING: no never-treated units. CS must fall back on not-yet-treated "
            "controls, which vanish in the final periods; the late-horizon ATT(g,t) "
            "will be identified off almost nothing."
        )
    elif n_never < min_never_treated:
        warnings.append(
            f"ADVISORY: only {n_never} never-treated units (<{min_never_treated}). "
            "Prefer control_group='notyettreated' and say so in the paper."
        )
    if n_cohorts < min_cohorts:
        warnings.append(
            f"BLOCKING: {n_cohorts} distinct treatment cohorts (<{min_cohorts}); "
            "there is no meaningful variation in timing to exploit."
        )
    if largest > max_largest_cohort_share:
        warnings.append(
            f"ADVISORY: the largest cohort holds {largest:.1%} of treated units; "
            "the design is close to a single-event two-group DiD."
        )
    if n_treated and n_recurrent / n_treated > max_recurrence_share:
        warnings.append(
            f"ADVISORY: {n_recurrent / n_treated:.1%} of treated municipalities are "
            "treated more than once. The absorbing-treatment assumption behind "
            "Callaway-Sant'Anna is violated: ATT(g,t) at horizon t-g mixes the "
            "first flood with later ones. Report the recurrent-event "
            "specification alongside, and consider restricting the horizon to "
            f"the median inter-episode gap ({median_gap} months)."
        )
    if n_with_pre < n_treated:
        warnings.append(
            f"ADVISORY: {n_treated - n_with_pre} treated units are already treated in "
            "the first period and contribute no pre-treatment observation; they are "
            "dropped by any event-study normalisation."
        )

    return DiDFeasibility(
        n_units=n_units,
        n_treated=n_treated,
        n_never_treated=n_never,
        share_never_treated=n_never / n_units if n_units else 0.0,
        n_cohorts=n_cohorts,
        largest_cohort_share=largest,
        first_cohort=treated["g_period"].min() if n_treated else None,
        last_cohort=treated["g_period"].max() if n_treated else None,
        n_periods=len(periods),
        n_units_with_pre_periods=n_with_pre,
        n_recurrent_units=n_recurrent,
        share_recurrent_of_treated=n_recurrent / n_treated if n_treated else 0.0,
        mean_episodes_per_treated=mean_eps,
        median_gap_months=median_gap,
        warnings=warnings,
    )


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TreatmentError(f"frame lacks required columns {missing}")


__all__ = [
    "DEFAULT_START",
    "DEFAULT_END",
    "DEFAULT_EPISODE_GAP_MONTHS",
    "TreatmentError",
    "DiDFeasibility",
    "month_range",
    "months_between",
    "to_municipality_month",
    "episodes",
    "first_treatment_period",
    "treatment_indicator",
    "treatment_cohorts",
    "did_feasibility_report",
]
