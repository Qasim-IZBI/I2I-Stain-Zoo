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

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_phi.descriptors import PHI_DIM, PHI_NAMES, PHI_REFERENCE
from uncertainty_phi.whiten import ledoit_wolf


@dataclass
class FloorEstimate:
    """A floor covariance plus the provenance needed to interpret it."""
    sigma: np.ndarray
    kind: str                 # split_half | cross_stain | cross_level | variogram_sill
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


def _computable_columns(delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a difference array into the descriptors it can speak to, and rows.

    A descriptor that is not computable in this run — lumen_fraction and
    tissue_fraction whenever no H&E is supplied — is NaN in every row. Filtering
    rows on `isfinite(...).all(axis=1)` then discards the entire array and the
    four collagen descriptors go down with the two that were never there.

    So: drop columns that are NaN throughout, then drop rows still carrying a NaN
    among the survivors. Returns (column indices, the surviving rows). The caller
    places the covariance into a full PHI_DIM x PHI_DIM matrix left NaN
    elsewhere, which is how `cross_stain_floor` has always reported the
    components it cannot reach.
    """
    delta = np.atleast_2d(np.asarray(delta, dtype=np.float64))
    cols = np.flatnonzero(np.isfinite(delta).any(axis=0))
    if cols.size == 0:
        return cols, delta[:0]
    sub = delta[:, cols]
    return cols, sub[np.isfinite(sub).all(axis=1)]


def _embed(sub: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Place a covariance over `cols` into a full PHI_DIM matrix, NaN elsewhere."""
    sigma = np.full((PHI_DIM, PHI_DIM), np.nan, dtype=np.float64)
    for i, gi in enumerate(cols):
        for j, gj in enumerate(cols):
            sigma[gi, gj] = sub[i, j]
    return sigma


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

    cols, delta = _computable_columns(a - b)
    if cols.size == 0:
        raise ValueError("no descriptor is computable: every column is NaN")
    if delta.shape[0] < 2:
        raise ValueError(
            f"need >= 2 finite pairs to estimate a floor covariance, got "
            f"{delta.shape[0]} over descriptors "
            f"{[PHI_NAMES[i] for i in cols]}"
        )

    sub, shrink = ledoit_wolf(delta, assume_centered=True)
    return FloorEstimate(
        sigma=_embed(sub, cols),
        kind=kind,
        n_samples=int(delta.shape[0]),
        shrinkage=float(shrink),
        components=tuple(PHI_NAMES[i] for i in cols),
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


def _within_group_pairs(
    groups: Sequence[str],
    max_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Index pairs drawn only from *within* the same slide.

    Pairing across slides would fold case-to-case biology into the estimate,
    which is not a floor — it is exactly the between-specimen variation the study
    is trying to measure against.
    """
    ii, jj = [], []
    g = np.asarray(groups)
    for key in np.unique(g):
        idx = np.flatnonzero(g == key)
        if idx.size < 2:
            continue
        a, b = np.triu_indices(idx.size, k=1)
        ii.append(idx[a])
        jj.append(idx[b])
    if not ii:
        raise ValueError("no slide contains two or more regions to pair")
    ii, jj = np.concatenate(ii), np.concatenate(jj)
    if ii.size > max_pairs:
        sel = rng.choice(ii.size, size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    return ii, jj


def variogram(
    phi: np.ndarray,
    coords: np.ndarray,
    groups: Sequence[str],
    *,
    n_bins: int = 12,
    max_lag_fraction: float = 0.5,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """Per-descriptor semivariance against in-plane separation.

    Returns lag-binned `Cov(φ(x) − φ(x'))` — the same difference-covariance
    convention as every other estimator here, so no factor-of-two conversion is
    needed downstream.

    `coords` is [n_regions, 2] in millimetres (`regions.region_centres_mm`).

    Lags beyond `max_lag_fraction` of the largest separation are discarded, per
    standard geostatistical practice: at extreme lags only a handful of
    region pairs survive, they all sit at opposite corners of the slide, and edge
    effects produce a spurious upturn that would corrupt the sill.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    if coords.shape[0] != phi.shape[0]:
        raise ValueError(f"{phi.shape[0]} descriptors but {coords.shape[0]} coordinates")
    if len(groups) != phi.shape[0]:
        raise ValueError(f"{phi.shape[0]} descriptors but {len(groups)} group labels")

    rng = np.random.default_rng(seed)
    ii, jj = _within_group_pairs(groups, max_pairs, rng)

    lag = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    delta = phi[ii] - phi[jj]

    keep = lag <= max_lag_fraction * lag.max()
    if keep.sum() < 2:
        raise ValueError("max_lag_fraction discarded almost every pair")
    lag, delta = lag[keep], delta[keep]

    edges = np.quantile(lag, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("all pairs share one separation; cannot build a variogram")

    # one set of columns for every bin, so the stacked covariances align
    cols = np.flatnonzero(np.isfinite(delta).any(axis=0))
    if cols.size == 0:
        raise ValueError("no descriptor is computable: every column is NaN")

    centres, per_bin, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lag >= lo) & (lag <= hi)
        d = delta[sel][:, cols]
        d = d[np.isfinite(d).all(axis=1)]
        if d.shape[0] < 2:
            continue
        centres.append(float((lo + hi) / 2))
        per_bin.append(_embed(np.cov(d, rowvar=False, bias=True), cols))
        counts.append(int(d.shape[0]))

    if not per_bin:
        raise ValueError("no lag bin held enough finite pairs")
    return {"lag_mm": np.asarray(centres),
            "cov": np.stack(per_bin, axis=0),
            "n_pairs": np.asarray(counts)}


def variogram_floor(
    phi: np.ndarray,
    coords: np.ndarray,
    groups: Sequence[str],
    *,
    n_bins: int = 12,
    sill_quantile: float = 0.5,
    max_lag_fraction: float = 0.5,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> Tuple[FloorEstimate, dict]:
    """Upper bound on the floor from **in-plane** spatial variation.

    Why this exists: without a second real PSR level, the collagen descriptors
    (CPA, β₀, β₁, dispersion) have no cross-level upper bound at all — they are
    not computable from H&E, so `cross_stain_floor` cannot reach them. Only the
    split-half *lower* bound applies, and a floor that is too small makes
    `bias² = observed² − floor²` too large: the unsafe direction.

    The substitute is spatial. Semivariance rises with separation and flattens at
    a **sill**, the fully-decorrelated limit. Two properties make it usable:

    * When structures no longer align between levels, the effective through-plane
      separation is already large, so the relevant lag sits at or near the sill —
      the exact level spacing need not be known.
    * `γ(∞) ≥ γ(h)` for any finite lag, so the sill **over-estimates** the floor,
      which under-states bias. Conservative, which is what §6.1 asks for.

    The assumption is rough isotropy at region scale: that moving 200 µm sideways
    perturbs a descriptor about as much as moving 200 µm deeper. Defensible for
    liver; shakier for kidney's cortex/medulla layering, so restrict to a cortex
    mask there (§8).

    `sill_quantile` sets which lags count as "large" — 0.5 averages the covariance
    over the upper half of the lag range.

    Returns (estimate, variogram_curve) so the flattening can be inspected rather
    than assumed. `curve["sill_reached"]` is reported **per descriptor**: where it
    is False the curve is still climbing, the sill has not been reached, and that
    component's bound is an under-estimate — the slide is too small relative to
    its correlation length.
    """
    curve = variogram(phi, coords, groups, n_bins=n_bins,
                      max_lag_fraction=max_lag_fraction,
                      max_pairs=max_pairs, seed=seed)
    lags, covs, counts = curve["lag_mm"], curve["cov"], curve["n_pairs"]

    cut = np.quantile(lags, sill_quantile)
    tail = lags >= cut
    if not tail.any():
        tail = np.zeros_like(lags, dtype=bool)
        tail[-1] = True

    w = counts[tail].astype(np.float64)
    w = w / w.sum()
    sigma = np.tensordot(w, covs[tail], axes=(0, 0))

    # Is the curve actually flat? Compare the first and last tail bins.
    tail_idx = np.flatnonzero(tail)
    d_first = np.sqrt(np.diag(covs[tail_idx[0]]))
    d_last = np.sqrt(np.diag(covs[tail_idx[-1]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = np.abs(d_last - d_first) / np.maximum(d_first, 1e-12)
    # Per descriptor, not one pooled bool: a single noisy component should not
    # veto the five that have plateaued, and the report is per-descriptor anyway.
    curve["tail_drift"] = {n: float(v) for n, v in zip(PHI_NAMES, drift)}
    curve["sill_reached"] = {n: bool(v < 0.15) for n, v in zip(PHI_NAMES, drift)}
    curve["sill_reached_all"] = bool(np.nanmax(drift) < 0.15)

    # Only advertise the descriptors the sill actually covers: with no H&E the
    # two level-A terms are NaN throughout and claiming them would make the
    # report look bounded where it is not.
    covered = tuple(n for n, v in zip(PHI_NAMES, np.diag(sigma)) if np.isfinite(v))
    return FloorEstimate(
        sigma=sigma,
        kind="variogram_sill",
        n_samples=int(counts[tail].sum()),
        shrinkage=0.0,
        components=covered,
    ), curve


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

    if not np.any(delta):
        raise ValueError(
            "cross-stain delta is identically zero on "
            f"{[PHI_NAMES[i] for i in invariant]} — the two sides were computed "
            "from the same image. phi_he must come from the real H&E and phi_psr "
            "from the real PSR RGB; a zero floor makes bias^2 = observed^2 - 0, "
            "which reads maximal bias, the unsafe direction."
        )

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
    variogram: Optional[FloorEstimate] = None,
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
    spread. Pass whichever floor estimates exist; precedence per component is

        direct  >  variogram  >  cross-stain upper  >  split-half lower

    `direct` needs a second real PSR level and settles the matter. Failing that,
    `variogram` is the only upper bound that reaches the collagen descriptors —
    cross-stain cannot, since collagen is not measurable in H&E. The split-half
    lower bound is the last resort and is **anti-conservative**: too small a floor
    inflates bias, so a component resting on it can only support an upper-bound
    claim about bias, never a point estimate.
    """
    phi = np.atleast_2d(np.asarray(phi_observed, dtype=np.float64))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # an all-NaN column is a descriptor that was never computable here; NaN
        # is the intended answer, not a condition to warn about
        warnings.simplefilter("ignore", RuntimeWarning)
        signal_sd = np.nanstd(phi, axis=0, ddof=1)

    def _sd(est: Optional[FloorEstimate], i: int) -> Optional[float]:
        if est is None:
            return None
        v = float(est.sd[i])
        return v if np.isfinite(v) else None

    rows: List[dict] = []
    for i, name in enumerate(PHI_NAMES):
        lo, hi, dr = _sd(lower, i), _sd(upper, i), _sd(direct, i)
        vg = _sd(variogram, i)
        best = next((v for v in (dr, vg, hi, lo) if v is not None), None)
        source = next((n for n, v in (("direct", dr), ("variogram", vg),
                                      ("cross_stain", hi), ("split_half", lo))
                       if v is not None), None)
        sig = float(signal_sd[i]) if np.isfinite(signal_sd[i]) else None

        ratio = None
        if best is not None and sig not in (None, 0.0):
            ratio = best / sig

        if ratio is None:
            verdict = ("unknown (no floor estimate for this component)"
                       if best is None else
                       "degenerate (no between-region variation to compare against)")
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
            "floor_sd_variogram": vg,
            "floor_sd_upper": hi,
            "floor_sd_lower": lo,
            "floor_sd_used": best,
            "floor_source": source,
            # a lower bound inflates bias; such a component supports only an
            # upper-bound claim about bias, never a point estimate
            "bound_direction": (
                None if source is None else
                ("measured" if source == "direct" else
                 ("conservative" if source in ("variogram", "cross_stain")
                  else "anti-conservative"))
            ),
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
