"""Whitening and floor tests — kidney_ood_data_plan.md §5.5 and §6."""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_phi.descriptors import PHI_DIM, PHI_NAMES
from uncertainty_phi.floor import (
    cross_stain_floor,
    sensitivity_band,
    split_half_floor,
    split_regions,
)
from uncertainty_phi.whiten import (
    bias_sq,
    dimension_shares,
    ledoit_wolf,
    mahalanobis_sq,
    whitening_matrix,
)

# A realistic floor: wildly heterogeneous scales, with beta_0 <-> beta_1
# correlated because more collagen means more components AND more loops.
FLOOR_SD = np.array([0.010, 40.0, 12.0, 0.030, 0.008, 0.006])
CORR = np.eye(6)
CORR[1, 2] = CORR[2, 1] = 0.75
CORR[0, 1] = CORR[1, 0] = 0.45
CORR[0, 2] = CORR[2, 0] = 0.40
SIGMA_TRUE = np.outer(FLOOR_SD, FLOOR_SD) * CORR
BIAS = np.array([0.004, 5.0, 25.0, 0.02, 0.0, 0.0])   # concentrated in beta_1


def _sample(n, seed=0, mean=None):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIGMA_TRUE)
    x = rng.standard_normal((n, 6)) @ L.T
    return x if mean is None else x + mean


class TestLedoitWolf:
    def test_recovers_heterogeneous_scales(self):
        """The bug this guards: shrinking toward mu*I with mu = trace/p destroys
        directions whose variance is orders of magnitude below the mean."""
        sigma, shrink = ledoit_wolf(_sample(4000, seed=1))
        assert 0.0 <= shrink <= 1.0
        rel = np.abs(np.diag(sigma) - np.diag(SIGMA_TRUE)) / np.diag(SIGMA_TRUE)
        assert rel.max() < 0.10, "per-feature variances must survive shrinkage"

    def test_preserves_correlation_sign_and_rough_size(self):
        sigma, _ = ledoit_wolf(_sample(4000, seed=2))
        corr = sigma[1, 2] / np.sqrt(sigma[1, 1] * sigma[2, 2])
        assert corr == pytest.approx(0.75, abs=0.08)

    def test_positive_definite(self):
        sigma, _ = ledoit_wolf(_sample(30, seed=3))     # n barely above d
        np.linalg.cholesky(sigma)                        # raises if not PD

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            ledoit_wolf(np.zeros((1, 6)))
        with pytest.raises(ValueError):
            ledoit_wolf(np.zeros(6))


class TestMahalanobis:
    def test_matches_explicit_solve(self):
        x = _sample(50, seed=4)
        expected = np.array([xi @ np.linalg.solve(SIGMA_TRUE, xi) for xi in x])
        np.testing.assert_allclose(mahalanobis_sq(x, SIGMA_TRUE), expected, rtol=1e-10)

    def test_pure_noise_averages_to_d(self):
        m2 = mahalanobis_sq(_sample(20000, seed=5), SIGMA_TRUE).mean()
        assert m2 == pytest.approx(PHI_DIM, rel=0.05)

    def test_whitening_matrix_is_a_whitener(self):
        W = whitening_matrix(SIGMA_TRUE)
        np.testing.assert_allclose(W @ SIGMA_TRUE @ W.T, np.eye(6), atol=1e-9)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            mahalanobis_sq(np.zeros((3, 5)), SIGMA_TRUE)


class TestBiasSq:
    def test_recovers_injected_bias(self):
        sigma_hat, _ = ledoit_wolf(_sample(4000, seed=6))          # from the FLOOR
        observed = mahalanobis_sq(_sample(20000, seed=7, mean=BIAS), sigma_hat).mean()
        truth = float(BIAS @ np.linalg.solve(SIGMA_TRUE, BIAS))
        assert bias_sq(observed, PHI_DIM) == pytest.approx(truth, rel=0.10)

    def test_pure_floor_gives_about_zero(self):
        sigma_hat, _ = ledoit_wolf(_sample(4000, seed=8))
        observed = mahalanobis_sq(_sample(20000, seed=9), sigma_hat).mean()
        assert abs(bias_sq(observed, PHI_DIM)) < 0.5

    def test_negatives_reported_by_default(self):
        """Clipping biases the error budget upward and hides the go/no-go
        outcome the section 7 pilot exists to detect."""
        assert bias_sq(np.array([4.0]), 6)[0] == pytest.approx(-2.0)
        assert bias_sq(np.array([4.0]), 6, clip=True)[0] == pytest.approx(0.0)


class TestDimensionShares:
    def test_concentrates_on_the_biased_direction(self):
        sigma_hat, _ = ledoit_wolf(_sample(4000, seed=10))
        shares = dimension_shares(_sample(20000, seed=11, mean=BIAS), sigma_hat)
        assert shares.sum() == pytest.approx(1.0)
        assert PHI_NAMES[int(np.argmax(shares))] == "beta1_per_mm2"
        assert shares.max() > 0.5

    def test_pure_noise_is_roughly_uniform(self):
        sigma_hat, _ = ledoit_wolf(_sample(4000, seed=12))
        shares = dimension_shares(_sample(20000, seed=13), sigma_hat)
        assert shares.max() < 0.30          # vs 1/6 = 0.167 under the null

    def test_raw_norm_would_be_dominated_by_counts(self):
        """Motivates the whole module: unnormalised, beta_0 eats the norm."""
        obs = _sample(20000, seed=14, mean=BIAS)
        raw = (obs ** 2).mean(0)
        raw = raw / raw.sum()
        assert PHI_NAMES[int(np.argmax(raw))] == "beta0_per_mm2"
        assert raw[0] < 0.001               # CPA numerically invisible


class TestFloor:
    def test_split_half_recovers_the_scale(self):
        a = _sample(2000, seed=15)
        b = _sample(2000, seed=16)
        est = split_half_floor(a, b)
        assert est.kind == "split_half"
        rel = np.abs(est.sd - FLOOR_SD) / FLOOR_SD
        assert rel.max() < 0.15, "halving the difference covariance should recover sd"

    def test_split_half_rejects_mismatched_halves(self):
        with pytest.raises(ValueError):
            split_half_floor(_sample(10), _sample(9))

    def test_cross_stain_covers_only_invariant_terms(self):
        he = _sample(500, seed=17)
        psr = _sample(500, seed=18)
        est = cross_stain_floor(he, psr)
        assert est.kind == "cross_stain"
        assert est.components == ("lumen_fraction", "tissue_fraction")
        # collagen terms are not computable from H&E -> NaN, not a fabricated 0
        assert np.isnan(est.sigma[0, 0])
        assert np.isfinite(est.sigma[4, 4])

    def test_split_regions_are_disjoint(self):
        a, b = split_regions(101, seed=0)
        assert len(a) == len(b) == 50
        assert not (set(a.tolist()) & set(b.tolist()))

    def test_sensitivity_band_marks_unbracketed_terms(self):
        lo = split_half_floor(_sample(500, seed=19), _sample(500, seed=20))
        hi = cross_stain_floor(_sample(500, seed=21), _sample(500, seed=22))
        band = sensitivity_band(lo, hi)["band"]
        assert band["lumen_fraction"]["bracketed"] is True
        assert band["task_specific_value"]["bracketed"] is False
        assert band["task_specific_value"]["upper_sd"] is None

    def test_summary_is_json_friendly(self):
        import json
        est = split_half_floor(_sample(200, seed=23), _sample(200, seed=24))
        json.dumps(est.summary())
