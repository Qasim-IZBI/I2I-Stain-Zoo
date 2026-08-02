"""Normalising φ_struct (kidney_ood_data_plan.md §5.5).

The components are incommensurable — CPA and the fractions live in [0,1], β₀/β₁
are densities in the hundreds — so an unnormalised ‖·‖² is dominated by whichever
has the largest raw units. In simulation β₀ alone takes 68% of the squared norm
and the four bounded descriptors contribute nothing measurable.

Z-scoring is not enough: it fixes scale but not correlation. β₀ and β₁ covary
strongly (more collagen → more components *and* more loops), so diagonal scaling
double-counts that shared direction. Whitening by the **full** floor covariance
handles both, and buys two further properties:

* signal-to-noise weighting — directions where levels naturally disagree are
  downweighted, which is what you want when the floor is the adversary;
* an exact subtraction — with Σ = Cov(δ) for the level-offset noise,
  E‖δ‖²_{Σ⁻¹} = d, so  bias² = observed²_{Σ⁻¹} − d.

The trap (§5.5.2 #3): Σ must be estimated from the **real-vs-real floor**, never
from the observed virtual-vs-real discrepancies — whitening by the covariance of
what you are measuring normalises the bias away. This module therefore never
computes Σ from observations; it is always an explicit argument, supplied by
`floor.py`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ledoit_wolf(x: np.ndarray, assume_centered: bool = False) -> Tuple[np.ndarray, float]:
    """Ledoit–Wolf shrinkage covariance, applied to the **correlation** matrix.

    d = 6 means 21 free parameters in Σ, while the effective n is closer to the
    case count (20 for liver) than to the region count, because regions within a
    slide are correlated. The raw empirical covariance is too noisy — and may be
    singular — at that ratio, hence shrinkage.

    Why correlation and not covariance
    ----------------------------------
    Textbook Ledoit–Wolf shrinks toward a scaled identity `μI` with
    `μ = trace(Σ)/p`. That target presumes comparable per-feature variances, which
    φ_struct violently violates: β₀ has variance ~1600 while CPA has ~1e-4, so μ is
    ~291 and even a 0.07% shrinkage adds ~0.2 to *every* diagonal — swamping the
    small-variance directions entirely. Measured on a synthetic case that pushed
    the recovered bias² to 4.96 against a true 9.51.

    Standardising first, shrinking the correlation matrix toward I, then rescaling
    leaves the per-feature variances exact and shrinks only the correlation
    structure — which is the part that is actually poorly determined at n≈20.

    Returns (Sigma, shrinkage), shrinkage in [0, 1]: 0 = empirical, 1 = diagonal.
    """
    X = np.asarray(x, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected [n_samples, n_features], got {X.shape}")
    n, p = X.shape
    if n < 2:
        raise ValueError(f"need >= 2 samples to estimate a covariance, got {n}")

    Xc = X if assume_centered else X - X.mean(axis=0, keepdims=True)

    sd = np.sqrt((Xc ** 2).mean(axis=0))
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = Xc / sd_safe                       # unit-variance columns

    emp = (Z.T @ Z) / n                    # empirical correlation
    target = np.eye(p)                     # the right target once standardised

    delta = ((emp - target) ** 2).sum()

    z2 = Z ** 2
    phi = ((z2.T @ z2) / n - emp ** 2).sum()

    shrink = 0.0 if delta <= 0 else float(np.clip(phi / (n * delta), 0.0, 1.0))
    corr = shrink * target + (1.0 - shrink) * emp
    sigma = corr * np.outer(sd_safe, sd_safe)
    return sigma, shrink


def mahalanobis_sq(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """‖delta‖²_{Σ⁻¹}, row-wise.

    `delta` is [n, d] (or [d], treated as one row). Solved rather than inverted:
    a shrunk Σ is well conditioned but an explicit inverse still throws away
    precision for no gain.
    """
    D = np.atleast_2d(np.asarray(delta, dtype=np.float64))
    S = np.asarray(sigma, dtype=np.float64)
    if S.shape[0] != S.shape[1] or S.shape[0] != D.shape[1]:
        raise ValueError(f"sigma {S.shape} incompatible with delta {D.shape}")
    sol = np.linalg.solve(S, D.T)          # [d, n]
    return np.einsum("ij,ji->i", D, sol)


def whitening_matrix(sigma: np.ndarray) -> np.ndarray:
    """W with W Σ Wᵀ = I, from the Cholesky factor.

    Useful for attributing a whitened norm back to individual directions — the
    per-dimension shares in §5.5 are computed as (delta @ W.T)**2.
    """
    L = np.linalg.cholesky(np.asarray(sigma, dtype=np.float64))
    return np.linalg.inv(L)


def bias_sq(observed_sq: np.ndarray, d: int, clip: bool = False) -> np.ndarray:
    """bias² = observed²_{Σ⁻¹} − d.

    `clip=False` by design (§7): negative point estimates must be **reported, not
    clipped**. Clipping at zero biases the whole error budget upward, and a
    negative value is exactly the signal that the observed discrepancy has sunk
    into the floor — which is the go/no-go outcome the pilot exists to detect.
    """
    out = np.asarray(observed_sq, dtype=np.float64) - float(d)
    return np.maximum(out, 0.0) if clip else out


def dimension_shares(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Fraction of the whitened squared norm contributed by each direction.

    Reproduces the §5.5 diagnostic table. Under pure floor noise every direction
    contributes ~1/d; a bias shows up as one direction dominating.
    """
    W = whitening_matrix(sigma)
    D = np.atleast_2d(np.asarray(delta, dtype=np.float64))
    per_dim = ((D @ W.T) ** 2).mean(axis=0)
    total = per_dim.sum()
    return per_dim / total if total > 0 else per_dim
