"""Per-pixel split of ensemble spread into procedural and data-exposure.

`uncertainty.py` sees the members of one training subset, so it measures
procedural uncertainty and cannot see the other half. These tests pin the ANOVA
estimator that separates them, and the two conventions that make its output
usable downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from decompose_pixel_uncertainty import decompose_stack, to_sd


def _grid(sig_p, sig_d, K=5, S=10, H=24, W=24, seed=0):
    """K subsets x S seeds with known variance components."""
    rng = np.random.default_rng(seed)
    return [100.0 + rng.normal(0, sig_d, (1, H, W, 3))
            + rng.normal(0, sig_p, (S, H, W, 3)) for _ in range(K)]


class TestEstimator:
    def test_recovers_known_components(self):
        """Averaged over repeats, because with K=5 the between-subset estimate is
        itself noisy — one grid is not enough to test an unbiased estimator."""
        sp, sd = 3.0, 5.0
        got_p = got_d = 0.0
        R = 8
        for r in range(R):
            res = decompose_stack(_grid(sp, sd, seed=r))
            got_p += float(np.mean(res["procedural"]))
            got_d += float(np.mean(res["data_exposure"]))
        assert got_p / R == pytest.approx(3 * sp ** 2, rel=0.05)
        assert got_d / R == pytest.approx(3 * sd ** 2, rel=0.05)

    def test_the_correction_removes_the_procedural_leak(self):
        """Each subset mean is itself an average of S noisy members, so the raw
        spread of subset means is inflated by sigma_proc/n0. Without subtracting
        it, procedural leaks into data."""
        sp, sd = 3.0, 5.0
        folds = _grid(sp, sd, seed=11)
        res = decompose_stack(folds)
        raw = np.stack([f.mean(0) for f in folds]).var(0, ddof=1).sum(2).mean()
        leak = float(np.mean(res["procedural"])) / 10.0
        assert raw - float(np.mean(res["data_exposure"])) == pytest.approx(leak, rel=1e-6)
        assert raw > 3 * sd ** 2          # the uncorrected value is biased HIGH

    def test_n0_is_the_seed_count_when_balanced(self):
        assert decompose_stack(_grid(2.0, 2.0))["n0"] == pytest.approx(10.0)

    def test_additivity_holds_in_variance(self):
        """total = procedural + data as VARIANCES. In SD they do not add, which
        is why the maps are documented as sqrt(sum of channel variances)."""
        res = decompose_stack(_grid(3.0, 5.0, seed=2))
        np.testing.assert_allclose(
            res["total"], res["procedural"] + res["data_exposure"], rtol=1e-10)

    def test_no_data_exposure_straddles_zero(self):
        """The estimator is unbiased, so a truly zero component comes out
        negative about half the time. That is the correct behaviour and the
        reason negatives are reported rather than clipped."""
        rng = np.random.default_rng(5)
        folds = [100.0 + rng.normal(0, 3.0, (10, 24, 24, 3)) for _ in range(5)]
        res = decompose_stack(folds)
        assert abs(float(np.mean(res["data_exposure"]))) < 0.5
        assert 0.3 < float((res["data_exposure"] < 0).mean()) < 0.7

    def test_one_subset_is_refused(self):
        """With a single subset there is no between-subset term at all, which is
        what uncertainty.py already computes."""
        with pytest.raises(ValueError, match="at least two"):
            decompose_stack(_grid(3.0, 5.0, K=1))


class TestSdConversion:
    def test_negative_variance_becomes_nan_not_zero(self):
        """NaN says 'not defined here'; zero would say 'no uncertainty here'.
        Those are opposite claims, and the second one is a lie about a component
        the ANOVA could not resolve."""
        got = to_sd(np.array([[4.0, -1.0], [9.0, 0.0]]))
        assert got[0, 0] == pytest.approx(2.0)
        assert got[1, 0] == pytest.approx(3.0)
        assert got[1, 1] == pytest.approx(0.0)
        assert np.isnan(got[0, 1])

    def test_units_match_uncertainty_py(self):
        """sqrt of the channel-summed variance, so the maps are interchangeable
        with raw_npy/ from uncertainty.py."""
        res = decompose_stack(_grid(3.0, 0.001, seed=7))
        assert float(np.nanmean(to_sd(res["procedural"]))) == pytest.approx(
            np.sqrt(3) * 3.0, rel=0.05)


class TestFromUncertainty:
    """The same decomposition from compute_ensemble_uncertainty.sh output.

    That stage already wrote, per subset, the two things the ANOVA needs, so
    reading inference/ again is fifty RGB arrays where this is ten.
    """

    @staticmethod
    def _from_folds(folds):
        """What uncertainty.py would have written for each subset."""
        sigmas = [np.sqrt(f.var(axis=0, ddof=1).sum(axis=2)) for f in folds]
        means = [np.clip(f.mean(axis=0), 0, 255).astype(np.uint8) for f in folds]
        return sigmas, means

    def test_procedural_is_exact(self):
        """raw_npy is sqrt(sum of the ddof=1 within-subset variance), so squaring
        recovers that variance with no loss at all."""
        from decompose_pixel_uncertainty import (decompose_from_uncertainty,
                                                 decompose_stack)
        folds = _grid(3.0, 5.0, seed=1)
        direct = decompose_stack(folds)
        sigmas, means = self._from_folds(folds)
        via = decompose_from_uncertainty(sigmas, means, n0=10.0)
        np.testing.assert_allclose(via["procedural"], direct["procedural"],
                                   rtol=1e-10)

    def test_data_term_agrees_within_the_quantisation_bound(self):
        """mean_rgb is uint8, so each subset mean carries a rounding error of
        variance 1/12 per channel — under 0.25 summed over three."""
        from decompose_pixel_uncertainty import (decompose_from_uncertainty,
                                                 decompose_stack)
        folds = _grid(3.0, 5.0, seed=2)
        direct = decompose_stack(folds)
        sigmas, means = self._from_folds(folds)
        via = decompose_from_uncertainty(sigmas, means, n0=10.0)
        gap = float(np.mean(via["data_exposure"] - direct["data_exposure"]))
        assert abs(gap) < 0.4
        # relative to a real signal it is negligible
        assert abs(gap) / float(np.mean(direct["data_exposure"])) < 0.02

    def test_quantisation_understates_rather_than_manufactures(self):
        """The direction that matters. Where subsets differ by less than half an
        intensity unit every mean rounds to the SAME integer, so the between term
        collapses to zero and the data component goes negative — it cannot invent
        data exposure that is not there."""
        from decompose_pixel_uncertainty import decompose_from_uncertainty
        rng = np.random.default_rng(4)
        H = W = 32
        # five subsets whose means are within a fraction of one intensity unit
        means = [np.full((H, W, 3), 128, np.uint8) for _ in range(5)]
        sigmas = [np.abs(rng.normal(5.0, 0.1, (H, W))) for _ in range(5)]
        res = decompose_from_uncertainty(means=means, sigmas=sigmas, n0=10.0)
        assert float(np.mean(res["data_exposure"])) < 0
        assert float(np.mean(res["procedural"])) > 0

    def test_additivity_still_holds(self):
        from decompose_pixel_uncertainty import decompose_from_uncertainty
        folds = _grid(3.0, 5.0, seed=3)
        sigmas, means = self._from_folds(folds)
        res = decompose_from_uncertainty(sigmas, means, n0=10.0)
        np.testing.assert_allclose(
            res["total"], res["procedural"] + res["data_exposure"], rtol=1e-10)
