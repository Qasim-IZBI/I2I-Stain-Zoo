"""The biological floor (kidney_ood_data_plan.md §6).

With non-adjacent levels you never observe the same-level target `y`, only `y'`
from a different level. Expanding the identity,

    ‖μ − φ(y')‖²  =  ‖μ − φ(y)‖²  +  2⟨μ − φ(y), φ(y) − φ(y')⟩  +  ‖φ(y) − φ(y')‖²

so if the level offset is zero-mean in φ-space the cross term vanishes in
expectation and what remains is `bias² + floor²`. **Non-adjacency does not break
the estimator; it inflates the floor.**

Real-vs-real cross-level discrepancy cannot be measured directly — there is only
one stain per level — so it is bracketed from both sides and reported as a
sensitivity band, never as a single number:

* **upper bound** — stain-invariant descriptors on real H&E at level A against
  real PSR at level B. Spans levels *and* absorbs stain/protocol differences, so
  it over-estimates the floor, which under-states bias: conservative, the right
  direction.
* **lower bound** — split-half within one slide. Two disjoint region sets from
  the same section give the spatial sampling variability at that region size,
  but span no levels at all, so it under-estimates.

Everything here returns a **covariance**, not a scalar: `whiten.py` needs the
full Σ, and it must come from here rather than from observed virtual-vs-real
differences (§5.5.2 #3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from uncertainty_phi.descriptors import PHI_DIM, PHI_NAMES, PHI_REFERENCE
from uncertainty_phi.whiten import ledoit_wolf


@dataclass
class FloorEstimate:
    """A floor covariance plus the provenance needed to interpret it."""
    sigma: np.ndarray
    kind: str                 # "split_half" | "cross_stain"
    n_samples: int
    shrinkage: float
    components: Tuple[str, ...]

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.sigma))

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "n_samples": self.n_samples,
            "shrinkage": self.shrinkage,
            "components": list(self.components),
            "floor_sd": {n: float(s) for n, s in zip(self.components, self.sd)},
        }


def split_half_floor(phi_a: np.ndarray, phi_b: np.ndarray) -> FloorEstimate:
    """Lower bound — spatial sampling variability within a single real slide.

    `phi_a`, `phi_b` are [n_pairs, d]: descriptors of two disjoint region sets
    drawn from the same section. Their difference carries no level offset, so
    this bounds the floor from below.

    The difference of two independent draws has twice the variance of one, so
    the covariance is halved to express a per-observation floor.
    """
    a = np.atleast_2d(np.asarray(phi_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(phi_b, dtype=np.float64))
    if a.shape != b.shape:
        raise ValueError(f"halves must match in shape, got {a.shape} vs {b.shape}")

    delta = a - b
    delta = delta[np.isfinite(delta).all(axis=1)]
    if delta.shape[0] < 2:
        raise ValueError("need >= 2 finite pairs to estimate a floor covariance")

    sigma, shrink = ledoit_wolf(delta, assume_centered=True)
    return FloorEstimate(
        sigma=sigma / 2.0,
        kind="split_half",
        n_samples=int(delta.shape[0]),
        shrinkage=float(shrink),
        components=PHI_NAMES,
    )


def cross_stain_floor(phi_he: np.ndarray, phi_psr: np.ndarray) -> FloorEstimate:
    """Upper bound — stain-invariant descriptors across the two levels.

    `phi_he` from real H&E at level A, `phi_psr` from real PSR at level B, both
    [n_regions, d]. Only the components whose reference class is "he" are
    stain-invariant and therefore comparable across the two images; the collagen
    terms are not computable from H&E and come back as NaN rows in Σ.

    Over-estimates the floor because it absorbs stain and protocol differences
    on top of the level offset — which under-states bias, the safe direction.
    """
    he = np.atleast_2d(np.asarray(phi_he, dtype=np.float64))
    psr = np.atleast_2d(np.asarray(phi_psr, dtype=np.float64))
    if he.shape != psr.shape:
        raise ValueError(f"shape mismatch: {he.shape} vs {psr.shape}")

    invariant = [i for i, ref in enumerate(PHI_REFERENCE) if ref == "he"]
    if not invariant:
        raise RuntimeError("no stain-invariant components declared in PHI_REFERENCE")

    delta = (he - psr)[:, invariant]
    delta = delta[np.isfinite(delta).all(axis=1)]
    if delta.shape[0] < 2:
        raise ValueError("need >= 2 finite regions to estimate a floor covariance")

    sub, shrink = ledoit_wolf(delta, assume_centered=True)

    sigma = np.full((PHI_DIM, PHI_DIM), np.nan, dtype=np.float64)
    for i, gi in enumerate(invariant):
        for j, gj in enumerate(invariant):
            sigma[gi, gj] = sub[i, j]

    return FloorEstimate(
        sigma=sigma,
        kind="cross_stain",
        n_samples=int(delta.shape[0]),
        shrinkage=float(shrink),
        components=tuple(PHI_NAMES[i] for i in invariant),
    )


def split_regions(n_regions: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Disjoint halves of a region index set, for `split_half_floor`.

    Randomised rather than spatially blocked: a left/right split would confound
    the sampling floor with any real spatial gradient across the section.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_regions)
    half = n_regions // 2
    return idx[:half], idx[half : 2 * half]


def sensitivity_band(lower: FloorEstimate, upper: FloorEstimate) -> dict:
    """Report the floor as an interval, per §6.1 — never a point estimate.

    Only the components present in both estimates can be bracketed; the collagen
    terms have no cross-stain upper bound because they are not computable from
    H&E, and that is stated rather than papered over.
    """
    lo_sd, hi_sd = lower.sd, upper.sd
    band = {}
    for i, name in enumerate(PHI_NAMES):
        lo, hi = float(lo_sd[i]), float(hi_sd[i])
        band[name] = {
            "lower_sd": lo if np.isfinite(lo) else None,
            "upper_sd": hi if np.isfinite(hi) else None,
            "bracketed": bool(np.isfinite(lo) and np.isfinite(hi)),
        }
    return {
        "band": band,
        "note": (
            "collagen terms have no cross-stain upper bound: they are not "
            "computable from H&E, so only the split-half lower bound applies"
        ),
    }
