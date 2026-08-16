"""Law of total variance over the (fold × seed) grid (uncertainty_strategy.md E2).

    Var_total  =  Var_folds( E_seed[·] )  +  E_folds[ Var_seed(·) ]
                  \_____________________/    \___________________/
                     data-exposure                procedural

Members within a fold share a training set and differ only by seed, so their
spread is **procedural** — algorithmic randomness. Fold means differ because the
folds saw different slides, so their spread is **data-exposure**.

Two cautions carried over from the strategy work:

* The construction here is a crossed grid, which is *not* what
  arXiv:2605.18329 does — that paper builds a 5-fold CV ensemble and a 5-seed
  deep ensemble separately and compares them. The decomposition below is our
  extension; do not cite them as having performed it.
* With a single fold (the vanilla 10-seed ensemble) the data component is
  **undefined**, not zero. Reporting zero would assert that data exposure
  contributes nothing, which the design cannot support.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class VarianceComponents:
    """Per-region decomposition, all in descriptor space.

    Arrays are [n_regions] for the scalar totals and [n_regions, d] for the
    per-descriptor breakdowns. `data` is None when only one fold is present.
    """
    total: np.ndarray
    procedural: np.ndarray
    data: Optional[np.ndarray]
    total_per_dim: np.ndarray
    procedural_per_dim: np.ndarray
    data_per_dim: Optional[np.ndarray]
    grand_mean: np.ndarray
    n_folds: int
    n_seeds_per_fold: List[int]

    def summary(self) -> Dict[str, object]:
        out = {
            "n_folds": self.n_folds,
            "n_seeds_per_fold": self.n_seeds_per_fold,
            "total_mean": _nanmean_or_none(self.total),
            "procedural_mean": _nanmean_or_none(self.procedural),
            "data_mean": None if self.data is None else _nanmean_or_none(self.data),
            "data_component": "undefined (single fold)" if self.data is None else "estimated",
        }
        return out


def _nanmean_or_none(a: Optional[np.ndarray]):
    if a is None:
        return None
    with np.errstate(invalid="ignore"):
        v = float(np.nanmean(a))
    return None if not np.isfinite(v) else v


def decompose(phi: Sequence[np.ndarray]) -> VarianceComponents:
    """Decompose descriptor-space variance over folds and seeds.

    Parameters
    ----------
    phi : sequence of length n_folds; element f is [n_seeds_f, n_regions, d].
        Seed counts may differ per fold. Regions must be aligned across folds —
        index r is the same physical region everywhere.

    Notes
    -----
    Estimated as one-way ANOVA components of variance, **not** as the raw
    plug-in split. Two corrections matter and both bias the naive version low:

    * within-fold variance uses ddof=1; the population form underestimates by
      (S−1)/S, i.e. 10% at S=10.
    * the spread of fold means is *contaminated by procedural noise*, since each
      fold mean is itself an average of S noisy members:
      `Var(fold means) = σ²_data + σ²_proc / S`. Subtracting `σ²_proc / n₀`
      recovers the data component; without it, procedural leaks into data.

    Measured on a synthetic grid with σ²_proc = 0.09 and σ²_data = 0.25, the
    uncorrected estimator returned 0.081 and 0.206.

    `n₀` is the ANOVA effective group size, which equals S for a balanced design
    and degrades gracefully when seed counts differ.

    The data component may come out **negative** when the true data variance is
    near zero. That is reported, not clipped, for the same reason as
    `whiten.bias_sq` — clipping biases the budget and hides the "no signal here"
    outcome.

    NaNs propagate via nan-aware reductions rather than silently dropping
    regions, so a descriptor that failed on one member stays visible.
    """
    if len(phi) == 0:
        raise ValueError("no folds supplied")

    folds = [np.asarray(f, dtype=np.float64) for f in phi]
    for i, f in enumerate(folds):
        if f.ndim != 3:
            raise ValueError(f"fold {i}: expected [n_seeds, n_regions, d], got {f.shape}")
    n_regions, d = folds[0].shape[1], folds[0].shape[2]
    for i, f in enumerate(folds):
        if f.shape[1:] != (n_regions, d):
            raise ValueError(
                f"fold {i} has shape {f.shape[1:]}, expected ({n_regions}, {d}) — "
                "regions must be aligned across folds"
            )

    n_folds = len(folds)
    counts = np.array([f.shape[0] for f in folds], dtype=np.float64)

    # A region where every member is NaN is "not computable here", which is the
    # intended outcome for background — not a condition to warn about, and at
    # twenty slides it would bury the log. errstate does not cover these: they
    # are Python warnings from the nan-reductions, not numpy FP errors.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # per-fold seed mean and *unbiased* seed variance: [n_folds, n_regions, d]
        fold_means = np.stack([np.nanmean(f, axis=0) for f in folds], axis=0)
        fold_vars = np.stack(
            [np.nanvar(f, axis=0, ddof=1) if f.shape[0] > 1
             else np.full(f.shape[1:], np.nan) for f in folds],
            axis=0,
        )

        # procedural = E_folds[ Var_seed ] — unbiased within-fold variance
        procedural_per_dim = np.nanmean(fold_vars, axis=0)                 # [R, d]

        grand_mean = np.nanmean(fold_means, axis=0)                        # [R, d]

        if n_folds > 1:
            # ANOVA effective group size: equals S for a balanced design
            total_n = counts.sum()
            n0 = (total_n - (counts ** 2).sum() / total_n) / (n_folds - 1)
            between = np.nanvar(fold_means, axis=0, ddof=1)                # [R, d]
            # strip the procedural noise that leaks into the fold means
            data_per_dim = between - procedural_per_dim / n0
        else:
            n0 = float("nan")
            data_per_dim = None

        procedural = np.nansum(procedural_per_dim, axis=1)
        if data_per_dim is None:
            data = None
            total_per_dim = procedural_per_dim
        else:
            data = np.nansum(data_per_dim, axis=1)
            total_per_dim = procedural_per_dim + data_per_dim
        total = np.nansum(total_per_dim, axis=1)

    return VarianceComponents(
        total=total,
        procedural=procedural,
        data=data,
        total_per_dim=total_per_dim,
        procedural_per_dim=procedural_per_dim,
        data_per_dim=data_per_dim,
        grand_mean=grand_mean,
        n_folds=len(folds),
        n_seeds_per_fold=[int(f.shape[0]) for f in folds],
    )


def decompose_whitened(
    phi: Sequence[np.ndarray],
    sigma: np.ndarray,
) -> Dict[str, Optional[np.ndarray]]:
    """Same decomposition, measured under the floor metric ‖·‖²_{Σ⁻¹}.

    The per-dimension variances are combined with the inverse floor covariance
    rather than summed raw, so the components are on the same footing as the
    whitened bias² they will be compared against. Σ must come from `floor.py`
    (see the warning in `whiten.py`).
    """
    from uncertainty_phi.whiten import whitening_matrix

    comps = decompose(phi)
    W = whitening_matrix(sigma)
    scale = (W ** 2).sum(axis=0)          # per-direction weight under Σ⁻¹

    out: Dict[str, Optional[np.ndarray]] = {
        "procedural": np.nansum(comps.procedural_per_dim * scale, axis=1),
        "data": None,
        "total": None,
    }
    if comps.data_per_dim is not None:
        out["data"] = np.nansum(comps.data_per_dim * scale, axis=1)
        out["total"] = out["procedural"] + out["data"]
    else:
        out["total"] = out["procedural"]
    return out
