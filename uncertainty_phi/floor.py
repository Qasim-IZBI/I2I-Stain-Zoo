"""The biological floor (kidney_ood_data_plan.md §6).

With non-adjacent levels you never observe the same-level target `y`, only `y'`
from a different level. Expanding the identity,

    ‖μ − φ(y')‖²  =  ‖μ − φ(y)‖²  +  2⟨μ − φ(y), φ(y) − φ(y')⟩  +  ‖φ(y) − φ(y')‖²

so if the level offset is zero-mean in φ-space the cross term vanishes in
expectation and what remains is `bias² + floor²`. **Non-adjacency does not break
the estimator; it inflates the floor.**

Where a **second real PSR level** exists per case, `cross_level_floor` measures
this directly and nothing needs bracketing. Absent that, real-vs-real cross-level
discrepancy cannot be observed — there is only one stain per level — so it is
bracketed from both sides and reported as a sensitivity band, never as a single
number:

* **upper bound** — stain-invariant descriptors on real H&E at level A against
  real PSR at level B. Spans levels *and* absorbs stain/protocol differences, so
  it over-estimates the floor, which under-states bias: conservative, the right
  direction.
* **lower bound** — split-half within one slide. Two disjoint region sets from
  the same section give the spatial sampling variability at that region size,
  but span no levels at all, so it under-estimates.

Every estimator returns Σ = Cov(δ) for δ the **difference** between two
observations that ought to agree — never a per-observation variance. That is the
quantity the identity calls the floor, and it is what makes `bias² = observed² −
d` exact. `whiten.py` needs the full Σ, and it must come from here rather than
from observed virtual-vs-real differences (§5.5.2 #3).

`per_descriptor_report` is the §7 go/no-go readout: one row per descriptor, since
a pooled floor hides whether any *individual* component — β₀/β₁ especially — is
stable enough between levels to carry a bias signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from uncertainty_phi.descriptors import PHI_DIM, PHI_NAMES, PHI_REFERENCE
from uncertainty_phi.whiten import ledoit_wolf


@dataclass
class FloorEstimate:
    """A floor covariance plus the provenance needed to interpret it."""
    sigma: np.ndarray
    kind: str                 # "split_half" | "cross_stain" | "cross_level"
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


def _difference_covariance(a: np.ndarray, b: np.ndarray, kind: str) -> FloorEstimate:
    """Σ = Cov(a − b), the quantity the §2.1 identity calls the floor.

    Every estimator in this module returns the covariance of the **difference**
    between two observations that ought to agree — not the per-observation
    variance. That is what makes the subtraction exact: with Σ = Cov(δ),
    E‖δ‖²_{Σ⁻¹} = d, so bias² = observed² − d.

    Halving to a per-observation variance (as an earlier version did) doubles
    every whitened observation and manufactures a constant bias² of exactly d on
    pure floor data.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match in shape, got {a.shape} vs {b.shape}")

    delta = a - b
    delta = delta[np.isfinite(delta).all(axis=1)]
    if delta.shape[0] < 2:
        raise ValueError("need >= 2 finite pairs to estimate a floor covariance")

    sigma, shrink = ledoit_wolf(delta, assume_centered=True)
    return FloorEstimate(
        sigma=sigma,
        kind=kind,
        n_samples=int(delta.shape[0]),
        shrinkage=float(shrink),
        components=PHI_NAMES,
    )


def split_half_floor(phi_a: np.ndarray, phi_b: np.ndarray) -> FloorEstimate:
    """Lower bound — spatial sampling variability within a single real slide.

    `phi_a`, `phi_b` are [n_pairs, d]: descriptors of two disjoint region sets
    drawn from the same section, paired arbitrarily. Their difference carries no
    level offset at all, so this under-estimates the true floor — it captures
    only how much a descriptor moves between two comparable patches of the same
    section.
    """
    return _difference_covariance(phi_a, phi_b, "split_half")


def cross_level_floor(phi_level_a: np.ndarray, phi_level_b: np.ndarray) -> FloorEstimate:
    """Direct measurement — two real levels of the same block.

    This is the floor as the identity defines it, with nothing to bracket: both
    inputs are real, so the difference is pure level offset. Requires a second
    real PSR level per case (open question 5 in kidney_ood_data_plan.md §9);
    where it is available it replaces the §6.1 upper/lower bracket entirely and
    is the single biggest de-risking available for the §7 go/no-go.
    """
    return _difference_covariance(phi_level_a, phi_level_b, "cross_level")


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

    # difference covariance, same convention as _difference_covariance
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


def per_descriptor_report(
    phi_observed: np.ndarray,
    *,
    lower: Optional[FloorEstimate] = None,
    upper: Optional[FloorEstimate] = None,
    direct: Optional[FloorEstimate] = None,
    marginal: float = 0.5,
    limited: float = 0.9,
) -> List[dict]:
    """Per-descriptor floor readout — the §7 go/no-go, one row per component.

    A single pooled floor number hides the question that actually decides the
    design: *is this particular descriptor stable enough between levels to carry
    a bias signal?* CPA averages over millions of pixels and concentrates fast.
    β₀/β₁ count discrete events, so they behave Poisson-like — a region holding
    ~50 loops has a relative SD near 1/√50 ≈ 14% between levels before threshold
    sensitivity is added. β may be a valid direction but a noisy one, and that is
    an empirical question, not something to assume either way.

    The decisive column is `floor_to_signal` = floor SD / between-region SD. A
    descriptor whose floor approaches its biological spread cannot discriminate
    regions no matter how large the true bias is.

    Whitening already downweights high-floor directions automatically (§5.5), so
    a noisy β is handled gracefully rather than corrupting the result. What this
    readout catches is the harder case: β so floor-limited that it contributes
    nothing, leaving CPA alone — which reopens the §5.3 lumen-filler blind spot.

    `phi_observed` is [n_regions, d] from real data, supplying the between-region
    spread. Pass whichever floor estimates exist; `direct` supersedes the bracket.
    """
    phi = np.atleast_2d(np.asarray(phi_observed, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        signal_sd = np.nanstd(phi, axis=0, ddof=1)

    def _sd(est: Optional[FloorEstimate], i: int) -> Optional[float]:
        if est is None:
            return None
        v = float(est.sd[i])
        return v if np.isfinite(v) else None

    rows: List[dict] = []
    for i, name in enumerate(PHI_NAMES):
        lo, hi, dr = _sd(lower, i), _sd(upper, i), _sd(direct, i)
        # prefer the direct measurement; else the conservative (upper) bound
        best = dr if dr is not None else (hi if hi is not None else lo)
        sig = float(signal_sd[i]) if np.isfinite(signal_sd[i]) else None

        ratio = None
        if best is not None and sig not in (None, 0.0):
            ratio = best / sig

        if ratio is None:
            verdict = "unknown (no floor estimate for this component)"
        elif ratio < marginal:
            verdict = "usable"
        elif ratio < limited:
            verdict = "marginal"
        else:
            verdict = "floor-limited"

        rows.append({
            "descriptor": name,
            "reference_class": PHI_REFERENCE[i],
            "floor_sd_direct": dr,
            "floor_sd_lower": lo,
            "floor_sd_upper": hi,
            "floor_sd_used": best,
            "between_region_sd": sig,
            "floor_to_signal": ratio,
            "verdict": verdict,
        })
    return rows


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
