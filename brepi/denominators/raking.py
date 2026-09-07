"""Iterative proportional fitting (raking) and compositional interpolation.

The mathematics under the demographic tensor, kept separate from the data
plumbing so it can be tested on its own.

Why raking rather than a constrained-optimisation reconstruction
----------------------------------------------------------------
Given a seed contingency table and a set of known margins, IPF returns the
unique table that matches every margin while minimising Kullback-Leibler
divergence from the seed (Deming & Stephan 1940; Ireland & Kullback 1968;
Fienberg 1970). Two properties earn it its place here:

1. It **preserves the seed's interaction structure exactly**. All odds ratios
   of the seed survive the fit. Demographically this is the right invariant:
   the census tells us how age, sex and colour co-vary within a municipality,
   and the annual estimate tells us only how many people there are. Raking
   uses each source for precisely what it is authoritative about, and cannot
   invent structure that no source supplied.

2. It is a fixed-point iteration with a closed-form step, no free
   hyperparameters, no solver state, and a convergence criterion a reader can
   check. A referee can reimplement it in twenty lines. That is worth a great
   deal more than a marginally better fit from an objective with a dozen
   weights nobody can justify individually.

The alternative — jointly estimating population and migration flows under a
cohort-component constraint system — buys accuracy the epidemiology cannot
use. A denominator enters the model as ``log(P)`` in an offset. Errors of a
few percent in a municipal age-sex cell move a log-offset by ~0.01 and are
swamped by the ascertainment variance we are explicitly modelling elsewhere.
Spending modelling risk there is a poor trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class RakingResult:
    """Fitted table plus the evidence needed to defend it."""

    table: np.ndarray
    iterations: int
    converged: bool
    #: Max absolute relative margin discrepancy at the final iteration.
    max_margin_error: float
    #: KL divergence from the seed; large values mean the margins disagreed
    #: strongly with the seed's structure and deserve inspection.
    kl_from_seed: float


def rake(
    seed: np.ndarray,
    margins: dict[tuple[int, ...], np.ndarray],
    *,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> RakingResult:
    """Fit ``seed`` to ``margins`` by iterative proportional fitting.

    Parameters
    ----------
    seed
        Non-negative array of any dimension. Zeros are structural: a cell that
        is zero in the seed stays zero, which is how impossible combinations
        (say, a colour category not collected in a given census) are enforced.
    margins
        Maps a tuple of axes to the required marginal totals over those axes.
        ``{(0,): totals_by_axis0}`` constrains the sum over all other axes.
        An empty tuple ``()`` constrains the grand total.

    Raises
    ------
    ValueError
        If margins are inconsistent (differing grand totals), negative, or
        shaped wrongly. These are specification errors and must not be
        silently absorbed by the iteration.
    """
    if np.any(seed < 0):
        raise ValueError("seed contains negative entries")
    if not np.isfinite(seed).all():
        raise ValueError("seed contains non-finite entries")

    ndim = seed.ndim
    grand: float | None = None
    for axes, target in margins.items():
        target = np.asarray(target, dtype=float)
        if np.any(target < 0) or not np.isfinite(target).all():
            raise ValueError(f"margin {axes} has negative or non-finite entries")
        expected_shape = tuple(seed.shape[a] for a in axes)
        if target.shape != expected_shape:
            raise ValueError(
                f"margin {axes} has shape {target.shape}, expected {expected_shape}"
            )
        total = float(target.sum())
        if grand is None:
            grand = total
        elif abs(total - grand) > 1e-6 * max(1.0, abs(grand)):
            raise ValueError(
                f"inconsistent margins: grand total {total} vs {grand}. "
                "Reconcile the sources before raking; IPF cannot resolve a "
                "genuine disagreement about how many people there are."
            )

    fitted = seed.astype(float).copy()
    if fitted.sum() <= 0:
        raise ValueError("seed sums to zero; nothing to distribute")

    converged = False
    max_err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        max_err = 0.0
        for axes, target in margins.items():
            target = np.asarray(target, dtype=float)
            other = tuple(a for a in range(ndim) if a not in axes)
            current = fitted.sum(axis=other) if other else fitted
            # Where the target is zero, zero the whole slice; where the current
            # sum is zero but the target is not, the seed has a structural zero
            # that cannot absorb the margin, which is a specification error.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current > _EPS, target / np.maximum(current, _EPS), 0.0)
            stranded = (current <= _EPS) & (target > _EPS)
            if np.any(stranded):
                raise ValueError(
                    f"margin {axes} requires mass in cells the seed makes structurally "
                    f"zero ({int(stranded.sum())} slice(s)); the seed and margin disagree "
                    "about which combinations exist"
                )
            rel = np.abs(current - target) / np.maximum(target, _EPS)
            max_err = max(max_err, float(rel.max()) if rel.size else 0.0)
            shape = [1] * ndim
            for pos, a in enumerate(axes):
                shape[a] = seed.shape[a]
            fitted = fitted * ratio.reshape(shape)
        if max_err < tol:
            converged = True
            break

    mask = seed > _EPS
    kl = float(
        np.sum(
            fitted[mask]
            * np.log(np.maximum(fitted[mask], _EPS) / np.maximum(seed[mask], _EPS))
        )
    )
    return RakingResult(
        table=fitted,
        iterations=it,
        converged=converged,
        max_margin_error=float(max_err),
        kl_from_seed=kl,
    )


# --------------------------------------------------------------------------
# Compositional interpolation
# --------------------------------------------------------------------------


def alr(shares: np.ndarray, axis: int = -1) -> np.ndarray:
    """Additive log-ratio transform against the last category."""
    s = np.maximum(np.asarray(shares, dtype=float), _EPS)
    ref = np.take(s, [-1], axis=axis)
    return np.log(np.delete(s, -1, axis=axis) / ref)


def alr_inv(coords: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse ALR; returns shares summing to one along ``axis``."""
    e = np.exp(np.asarray(coords, dtype=float))
    shape = list(e.shape)
    shape[axis] = 1
    full = np.concatenate([e, np.ones(shape)], axis=axis)
    return full / full.sum(axis=axis, keepdims=True)


def interpolate_composition(
    shares_a: np.ndarray,
    shares_b: np.ndarray,
    weight: float,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Geodesic interpolation between two compositions on the simplex.

    Linear interpolation of shares is the usual shortcut and it is wrong in a
    specific, consequential way: it is not scale-invariant, it can drift off
    the simplex once several categories move, and it treats a change from
    0.001 to 0.002 as negligible when demographically it is a doubling. Working
    in log-ratio coordinates and mapping back fixes all three, and reduces to
    the familiar answer when the compositions are close.

    ``weight`` is 0 at ``shares_a`` and 1 at ``shares_b``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}")
    # Categories structurally absent from both endpoints stay absent.
    both_zero = (shares_a <= _EPS) & (shares_b <= _EPS)
    ca, cb = alr(shares_a, axis=axis), alr(shares_b, axis=axis)
    out = alr_inv((1.0 - weight) * ca + weight * cb, axis=axis)
    out = np.where(both_zero, 0.0, out)
    total = out.sum(axis=axis, keepdims=True)
    return np.divide(out, np.maximum(total, _EPS))


def cohort_shift_index(n_ages: int, years: int, width: int = 5) -> np.ndarray:
    """Index map that ages a population by ``years`` along the Lexis diagonal.

    Returns, for each destination age group, the source age group whose cohort
    arrives there. The final group is open-ended and absorbs everything above.

    This is the correction that makes between-census interpolation
    demographically coherent. Interpolating "the 20-24 share" linearly from
    2010 to 2022 compares the 1986-1990 birth cohort with the 1998-2002 birth
    cohort, which is not a trajectory of anything. Interpolating along the
    diagonal compares each cohort with itself.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    shift = int(round(years / width))
    idx = np.arange(n_ages) - shift
    idx = np.clip(idx, 0, n_ages - 1)
    return idx


__all__ = [
    "RakingResult",
    "rake",
    "alr",
    "alr_inv",
    "interpolate_composition",
    "cohort_shift_index",
]
