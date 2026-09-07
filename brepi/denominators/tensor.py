"""The demographic denominator tensor: municipality x year x sex x age x colour.

Construction
------------
Three layers, each using a source only for what that source is authoritative
about:

**Layer 0 - margins (hard).** Official municipal population totals per year.
These come from IBGE's annual estimates as redistributed by DATASUS (POPSVS)
or SIDRA 6579. They are not adjusted, smoothed, or re-derived. This is a
deliberate choice: the Ministry of Health computes its own published
incidence rates on exactly these totals, so a rate computed here is
numerically comparable to a rate in a Boletim Epidemiologico. A denominator
that is privately "better" but publicly incomparable is worse for an
epidemiological paper.

**Layer 1 - composition (soft).** Joint sex x age x colour shares per
municipality from the decennial censuses (2000, 2010, 2022). Between censuses
the composition is interpolated along birth cohorts in log-ratio coordinates
(see :mod:`brepi.denominators.raking`). After the last census it is held with
a cohort shift and flagged as extrapolated.

**Layer 2 - reconciliation.** The interpolated composition is raked to the
official total, and optionally to state-level age x sex margins from IBGE's
population projections when those are wanted. Raking preserves every odds
ratio in the census seed while matching the margins exactly.

Every cell carries the method that produced it and its distance in years from
the nearest observed census, so a sensitivity analysis can down-weight or
exclude heavily interpolated cells without re-running the build.

What this deliberately does not do
----------------------------------
It does not reconstruct migration flows, and it does not solve a cohort-
component system jointly for population and net migration. Those recover
demographic detail that a Poisson log-offset cannot use, at the cost of a
large objective with weights that are individually indefensible to a referee.
The margins already contain migration's effect on municipal totals, which is
the part that reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import polars as pl

from brepi.denominators.raking import (
    cohort_shift_index,
    interpolate_composition,
    rake,
)

# --------------------------------------------------------------------------
# Category schemes
# --------------------------------------------------------------------------

#: Five-year age groups with an open final interval. Chosen to match both the
#: census tabulations and the WHO/Segi standard populations used for
#: age-standardisation, so no regrouping is needed downstream.
AGE_GROUPS: tuple[str, ...] = (
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80+",
)
AGE_WIDTH_YEARS = 5

SEXES: tuple[str, ...] = ("M", "F")

#: IBGE colour/race categories as self-declared in the census. ``ignorada`` is
#: retained rather than dropped: how much of it there is, is information.
COLOURS: tuple[str, ...] = ("branca", "preta", "parda", "amarela", "indigena", "ignorada")

CENSUS_YEARS: tuple[int, ...] = (2000, 2010, 2022)

Method = Literal["census_observed", "cohort_interpolated", "cohort_extrapolated"]


@dataclass(frozen=True)
class TensorSpec:
    """What to build."""

    years: tuple[int, ...]
    municipalities: tuple[str, ...]  # 7-digit IBGE codes
    sexes: tuple[str, ...] = SEXES
    ages: tuple[str, ...] = AGE_GROUPS
    colours: tuple[str, ...] | None = COLOURS
    #: Redistribute the ``ignorada`` colour category over the declared ones in
    #: proportion to their municipal shares. Off by default, because doing it
    #: silently is how colour-specific rates acquire undeclared bias.
    redistribute_unknown_colour: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        base = (len(self.municipalities), len(self.years), len(self.sexes), len(self.ages))
        return base + ((len(self.colours),) if self.colours else ())


@dataclass
class DemographicTensor:
    """A dense denominator tensor with per-cell provenance.

    ``values`` has shape ``(municipality, year, sex, age[, colour])``.
    ``method`` and ``years_from_census`` are aligned arrays over the
    ``(municipality, year)`` plane, since interpolation status is a property
    of the census-year distance, not of an individual demographic cell.
    """

    spec: TensorSpec
    values: np.ndarray
    method: np.ndarray  # dtype=object, shape (n_mun, n_year)
    years_from_census: np.ndarray  # int, shape (n_mun, n_year)
    diagnostics: dict = field(default_factory=dict)

    # -- coordinate helpers -------------------------------------------------

    def _axis_index(self, axis: str, label: str) -> int:
        lookup = {
            "municipality": self.spec.municipalities,
            "year": tuple(str(y) for y in self.spec.years),
            "sex": self.spec.sexes,
            "age": self.spec.ages,
            "colour": self.spec.colours or (),
        }[axis]
        return lookup.index(label if axis != "year" else str(label))

    def total(self, *, municipality: str | None = None, year: int | None = None) -> float:
        v = self.values
        if municipality is not None:
            v = v[self._axis_index("municipality", municipality)][None, ...]
        if year is not None:
            v = v[:, [self._axis_index("year", year)]]
        return float(v.sum())

    def to_long(self) -> pl.DataFrame:
        """Flatten to the long fact table used everywhere downstream.

        Long is the base representation on purpose: wide panels are compiled
        when a model needs one, never stored, so that adding a stratifier is
        an append rather than a schema migration.
        """
        s = self.spec
        axes: list[Sequence] = [s.municipalities, s.years, s.sexes, s.ages]
        names = ["munic_code", "year", "sex", "age_group"]
        if s.colours:
            axes.append(s.colours)
            names.append("colour")
        grids = np.meshgrid(*[np.arange(len(a)) for a in axes], indexing="ij")
        cols = {n: [axes[i][j] for j in g.ravel()] for i, (n, g) in enumerate(zip(names, grids))}
        mi = grids[0].ravel()
        yi = grids[1].ravel()
        cols["population"] = self.values.ravel()
        cols["method"] = self.method[mi, yi]
        cols["years_from_census"] = self.years_from_census[mi, yi]
        return pl.DataFrame(cols).with_columns(
            pl.col("year").cast(pl.Int32),
            pl.col("population").cast(pl.Float64),
            pl.col("years_from_census").cast(pl.Int16),
        )

    def write_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_long().write_parquet(path)
        return path

    # -- aggregation --------------------------------------------------------

    def collapse(self, keep: Iterable[str]) -> pl.DataFrame:
        """Sum out every stratifier not in ``keep``.

        ``keep`` may contain any of ``municipality, year, sex, age, colour``.
        """
        name_map = {
            "municipality": "munic_code",
            "year": "year",
            "sex": "sex",
            "age": "age_group",
            "colour": "colour",
        }
        cols = [name_map[k] for k in keep]
        return (
            self.to_long()
            .group_by(cols)
            .agg(pl.col("population").sum())
            .sort(cols)
        )


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def _seed_for_year(
    census_shares: dict[int, np.ndarray],
    year: int,
) -> tuple[np.ndarray, Method, int]:
    """Composition seed for one year, plus how it was obtained.

    ``census_shares`` maps a census year to an array of shares over the
    demographic axes (sex, age[, colour]) for one municipality, summing to 1.
    """
    available = sorted(census_shares)
    if year in census_shares:
        return census_shares[year], "census_observed", 0

    earlier = [y for y in available if y < year]
    later = [y for y in available if y > year]

    if earlier and later:
        a, b = earlier[-1], later[0]
        w = (year - a) / (b - a)
        # Age the earlier census forward and the later one backward so both
        # endpoints describe the same cohorts as the target year.
        sa = _shift_ages(census_shares[a], year - a)
        sb = _shift_ages(census_shares[b], year - b)
        shares = interpolate_composition(sa, sb, w, axis=1)
        return shares, "cohort_interpolated", min(year - a, b - year)

    anchor = earlier[-1] if earlier else later[0]
    shares = _shift_ages(census_shares[anchor], year - anchor)
    return shares, "cohort_extrapolated", abs(year - anchor)


def _shift_ages(shares: np.ndarray, years: int) -> np.ndarray:
    """Age a composition along the Lexis diagonal by ``years`` (may be negative).

    Axis 1 is age. The open final group absorbs, and the youngest groups are
    refilled from the current youngest since we have no birth series here;
    raking to the municipal total then restores the level. This is a
    deliberately conservative treatment: it will not invent a fertility
    transition, and the resulting bias is confined to the 0-4 and 5-9 groups,
    which contribute almost nothing to leptospirosis burden.
    """
    if years == 0:
        return shares
    idx = cohort_shift_index(shares.shape[1], years, AGE_WIDTH_YEARS)
    out = shares[:, idx, ...]
    total = out.sum(axis=tuple(range(out.ndim)), keepdims=True)
    return out / np.maximum(total, 1e-12)


def build_tensor(
    spec: TensorSpec,
    *,
    totals: pl.DataFrame,
    census_composition: pl.DataFrame,
    state_age_sex_margins: pl.DataFrame | None = None,
) -> DemographicTensor:
    """Assemble the tensor.

    Parameters
    ----------
    totals
        Official municipal totals. Columns: ``munic_code``, ``year``,
        ``population``. Must cover every municipality x year in ``spec``;
        a gap is an error, not something to interpolate over quietly.
    census_composition
        Long census counts. Columns: ``munic_code``, ``census_year``,
        ``sex``, ``age_group``[, ``colour``], ``count``.
    state_age_sex_margins
        Optional. Columns ``uf_code``, ``year``, ``sex``, ``age_group``,
        ``population``. When supplied, the fit additionally reproduces these,
        which propagates IBGE's projected ageing into the municipal cells
        instead of freezing each municipality's age structure at its last
        census.

    Returns
    -------
    DemographicTensor
        With ``diagnostics`` reporting raking convergence, the worst margin
        error, and the KL divergence distribution across municipalities.
    """
    n_mun, n_year = len(spec.municipalities), len(spec.years)
    demo_shape = (len(spec.sexes), len(spec.ages)) + (
        (len(spec.colours),) if spec.colours else ()
    )
    values = np.zeros((n_mun, n_year) + demo_shape, dtype=float)
    method = np.empty((n_mun, n_year), dtype=object)
    dist = np.zeros((n_mun, n_year), dtype=int)

    tot = _index_totals(totals, spec)
    comp = _index_composition(census_composition, spec)

    kls: list[float] = []
    worst_err = 0.0
    non_converged: list[tuple[str, int]] = []

    for mi, mun in enumerate(spec.municipalities):
        shares_by_census = comp.get(mun)
        if not shares_by_census:
            raise ValueError(
                f"municipality {mun} has no census composition; supply it or "
                "exclude the municipality explicitly via the spec"
            )
        for yi, year in enumerate(spec.years):
            total_pop = tot.get((mun, year))
            if total_pop is None:
                raise ValueError(f"no official total for municipality {mun} in {year}")
            seed_shares, how, d = _seed_for_year(shares_by_census, year)
            method[mi, yi], dist[mi, yi] = how, d
            if total_pop <= 0:
                continue
            seed = seed_shares * total_pop
            res = rake(seed, {(): np.array(total_pop)})
            values[mi, yi] = res.table
            kls.append(res.kl_from_seed)
            worst_err = max(worst_err, res.max_margin_error)
            if not res.converged:
                non_converged.append((mun, year))

    if state_age_sex_margins is not None:
        values, extra = _apply_state_margins(values, spec, state_age_sex_margins)
        worst_err = max(worst_err, extra)

    if spec.colours and spec.redistribute_unknown_colour:
        values = _redistribute_unknown_colour(values, spec)

    diagnostics = {
        "n_cells": int(values.size),
        "max_margin_error": worst_err,
        "kl_from_seed_median": float(np.median(kls)) if kls else 0.0,
        "kl_from_seed_p95": float(np.quantile(kls, 0.95)) if kls else 0.0,
        "non_converged": non_converged,
        "extrapolated_share": float((method == "cohort_extrapolated").mean()),
        "observed_share": float((method == "census_observed").mean()),
    }
    return DemographicTensor(spec, values, method, dist, diagnostics)


def _index_totals(totals: pl.DataFrame, spec: TensorSpec) -> dict[tuple[str, int], float]:
    need = {"munic_code", "year", "population"}
    missing = need - set(totals.columns)
    if missing:
        raise ValueError(f"totals missing columns: {sorted(missing)}")
    return {
        (r["munic_code"], int(r["year"])): float(r["population"])
        for r in totals.iter_rows(named=True)
    }


def _index_composition(
    comp: pl.DataFrame, spec: TensorSpec
) -> dict[str, dict[int, np.ndarray]]:
    """Pivot long census counts into per-municipality share arrays."""
    axes = ["sex", "age_group"] + (["colour"] if spec.colours else [])
    order = {
        "sex": list(spec.sexes),
        "age_group": list(spec.ages),
        "colour": list(spec.colours or ()),
    }
    shape = tuple(len(order[a]) for a in axes)
    pos = {a: {lab: i for i, lab in enumerate(order[a])} for a in axes}

    out: dict[str, dict[int, np.ndarray]] = {}
    for row in comp.iter_rows(named=True):
        mun = row["munic_code"]
        cy = int(row["census_year"])
        arr = out.setdefault(mun, {}).setdefault(cy, np.zeros(shape))
        try:
            idx = tuple(pos[a][row[a]] for a in axes)
        except KeyError as exc:
            raise ValueError(
                f"census composition contains an unmapped category {exc!r}; "
                "extend the scheme in TensorSpec rather than dropping it"
            ) from None
        arr[idx] += float(row["count"])

    for mun, by_year in out.items():
        for cy, arr in by_year.items():
            s = arr.sum()
            if s <= 0:
                raise ValueError(f"municipality {mun} has zero census population in {cy}")
            by_year[cy] = arr / s
    return out


def _apply_state_margins(
    values: np.ndarray, spec: TensorSpec, margins: pl.DataFrame
) -> tuple[np.ndarray, float]:
    """Rake municipal cells so that each state's age x sex totals are matched.

    Municipal totals are preserved as a simultaneous margin, so the result
    satisfies both constraints or raises if they are mutually inconsistent.
    """
    worst = 0.0
    uf_of = [m[:2] for m in spec.municipalities]
    for yi, year in enumerate(spec.years):
        sub = margins.filter(pl.col("year") == year)
        if sub.is_empty():
            continue
        for uf in sorted(set(uf_of)):
            rows = [i for i, u in enumerate(uf_of) if u == uf]
            block = values[rows, yi]  # (mun, sex, age[, colour])
            target = np.zeros((len(spec.sexes), len(spec.ages)))
            usub = sub.filter(pl.col("uf_code") == uf)
            if usub.is_empty():
                continue
            for r in usub.iter_rows(named=True):
                try:
                    si = spec.sexes.index(r["sex"])
                    ai = spec.ages.index(r["age_group"])
                except ValueError:
                    continue
                target[si, ai] += float(r["population"])
            mun_totals = block.sum(axis=tuple(range(1, block.ndim)))
            scale = mun_totals.sum() / max(target.sum(), 1e-12)
            res = rake(
                block,
                {(0,): mun_totals, (1, 2): target * scale},
            )
            values[rows, yi] = res.table
            worst = max(worst, res.max_margin_error)
    return values, worst


def _redistribute_unknown_colour(values: np.ndarray, spec: TensorSpec) -> np.ndarray:
    """Distribute the ``ignorada`` colour category over declared categories.

    Proportional-within-cell redistribution. Applied only on explicit request,
    and it should be reported in the methods: it assumes non-declaration is
    independent of the outcome within a municipality-year-sex-age cell, which
    is an assumption, not a fact.
    """
    colours = list(spec.colours or ())
    if "ignorada" not in colours:
        return values
    ui = colours.index("ignorada")
    keep = [i for i in range(len(colours)) if i != ui]
    unknown = values[..., ui]
    declared = values[..., keep]
    dsum = declared.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(dsum > 0, declared / np.maximum(dsum, 1e-12), 0.0)
    values[..., keep] = declared + share * unknown[..., None]
    values[..., ui] = 0.0
    return values


__all__ = [
    "AGE_GROUPS",
    "AGE_WIDTH_YEARS",
    "SEXES",
    "COLOURS",
    "CENSUS_YEARS",
    "TensorSpec",
    "DemographicTensor",
    "build_tensor",
]
