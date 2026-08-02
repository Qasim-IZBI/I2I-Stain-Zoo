"""Variance decomposition tests — uncertainty_strategy.md E2.

The estimator must recover injected procedural and data components without the
two biases that the naive plug-in split carries (ddof, and procedural leakage
into the fold means).
"""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_phi.decompose import decompose, decompose_whitened

F, S, R, D = 5, 10, 400, 6


def _grid(proc_sd, data_sd, seed=0, n_folds=F, n_seeds=S):
    rng = np.random.default_rng(seed)
    fold_offset = rng.standard_normal((n_folds, 1, R, D)) * data_sd
    noise = rng.standard_normal((n_folds, n_seeds, R, D)) * proc_sd
    return [fold_offset[f] + noise[f] for f in range(n_folds)]


class TestRecovery:
    def test_recovers_both_components(self):
        proc_sd, data_sd = 0.30, 0.50
        c = decompose(_grid(proc_sd, data_sd))
        # totals are summed over D descriptors
        assert np.nanmean(c.procedural) / D == pytest.approx(proc_sd ** 2, rel=0.05)
        assert np.nanmean(c.data) / D == pytest.approx(data_sd ** 2, rel=0.05)

    def test_uncorrected_estimator_would_be_biased(self):
        """Guards the two ANOVA corrections. The naive version returns
        (S-1)/S * proc, and data inflated by proc/S."""
        proc_sd, data_sd = 0.30, 0.50
        folds = _grid(proc_sd, data_sd)
        c = decompose(folds)

        naive_proc = np.mean([np.var(f, axis=0, ddof=0) for f in folds])
        naive_data = np.var(np.stack([f.mean(0) for f in folds]), axis=0, ddof=0).mean()

        assert naive_proc < 0.95 * (proc_sd ** 2)              # biased low
        assert naive_data < 0.95 * (data_sd ** 2)              # biased low
        assert np.nanmean(c.procedural) / D > naive_proc       # correction helps
        assert np.nanmean(c.data) / D > naive_data

    def test_pure_procedural_gives_near_zero_data(self):
        c = decompose(_grid(0.4, 0.0, seed=3))
        assert abs(np.nanmean(c.data)) / D < 0.01

    def test_data_component_may_go_negative(self):
        """With no true data variance the estimate straddles zero; that must be
        reported, not clipped (same rule as whiten.bias_sq)."""
        neg = 0
        for s in range(12):
            c = decompose(_grid(0.4, 0.0, seed=100 + s, n_folds=3, n_seeds=4))
            if np.nanmean(c.data) < 0:
                neg += 1
        assert neg > 0, "estimator appears to be clipping at zero"


class TestSingleFold:
    def test_data_is_undefined_not_zero(self):
        c = decompose([_grid(0.3, 0.5)[0]])
        assert c.data is None
        assert c.data_per_dim is None
        assert c.summary()["data_component"] == "undefined (single fold)"
        assert c.summary()["data_mean"] is None

    def test_procedural_still_estimated(self):
        c = decompose([_grid(0.3, 0.5)[0]])
        assert np.nanmean(c.procedural) / D == pytest.approx(0.09, rel=0.1)


class TestShapesAndValidation:
    def test_shapes(self):
        c = decompose(_grid(0.3, 0.5))
        assert c.total.shape == (R,)
        assert c.total_per_dim.shape == (R, D)
        assert c.grand_mean.shape == (R, D)
        assert c.n_folds == F
        assert c.n_seeds_per_fold == [S] * F

    def test_total_is_sum_of_parts(self):
        c = decompose(_grid(0.3, 0.5))
        np.testing.assert_allclose(c.total, c.procedural + c.data, rtol=1e-10)

    def test_unbalanced_seed_counts_allowed(self):
        folds = _grid(0.3, 0.5)
        folds[0] = folds[0][:4]          # one fold with fewer seeds
        c = decompose(folds)
        assert c.n_seeds_per_fold == [4, S, S, S, S]
        assert np.nanmean(c.data) / D == pytest.approx(0.25, rel=0.15)

    def test_misaligned_regions_rejected(self):
        folds = _grid(0.3, 0.5)
        folds[1] = folds[1][:, : R - 1]
        with pytest.raises(ValueError, match="aligned"):
            decompose(folds)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            decompose([])

    def test_nan_member_does_not_silently_vanish(self):
        folds = _grid(0.3, 0.5)
        folds[0] = folds[0].copy()
        folds[0][0, 0, 0] = np.nan
        c = decompose(folds)
        assert np.isfinite(c.procedural).all()   # nan-aware, region survives


class TestWhitenedDecomposition:
    def test_runs_under_a_floor_metric(self):
        folds = _grid(0.3, 0.5)
        sigma = np.eye(D) * 0.25
        out = decompose_whitened(folds, sigma)
        assert set(out) == {"procedural", "data", "total"}
        assert out["procedural"].shape == (R,)
        # identity/0.25 scaling multiplies every direction by 4
        plain = decompose(folds)
        np.testing.assert_allclose(out["procedural"], plain.procedural * 4, rtol=1e-8)

    def test_single_fold_leaves_data_none(self):
        out = decompose_whitened([_grid(0.3, 0.5)[0]], np.eye(D))
        assert out["data"] is None
