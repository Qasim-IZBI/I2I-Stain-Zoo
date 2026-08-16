"""Whitening and floor tests — kidney_ood_data_plan.md §5.5 and §6."""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_phi.descriptors import PHI_DIM, PHI_NAMES, PHI_REFERENCE
from uncertainty_phi.floor import (
    cross_level_floor,
    variogram_floor,
    cross_stain_floor,
    per_descriptor_report,
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
# One entry per PHI_NAMES slot: CPA, beta0/beta1 collagen, dispersion,
# lumen_fraction, beta0/beta1 lumen. Wildly heterogeneous scales on purpose.
FLOOR_SD = np.array([0.010, 40.0, 12.0, 0.030, 0.008, 30.0, 20.0])
assert FLOOR_SD.size == PHI_DIM, "FLOOR_SD must track PHI_NAMES"
CORR = np.eye(PHI_DIM)
CORR[1, 2] = CORR[2, 1] = 0.75
CORR[0, 1] = CORR[1, 0] = 0.45
CORR[0, 2] = CORR[2, 0] = 0.40
SIGMA_TRUE = np.outer(FLOOR_SD, FLOOR_SD) * CORR
BIAS = np.zeros(PHI_DIM)
BIAS[:4] = [0.004, 5.0, 25.0, 0.02]        # concentrated in beta_1


def _sample(n, seed=0, mean=None):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIGMA_TRUE)
    x = rng.standard_normal((n, PHI_DIM)) @ L.T
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
            ledoit_wolf(np.zeros((1, PHI_DIM)))
        with pytest.raises(ValueError):
            ledoit_wolf(np.zeros(PHI_DIM))


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
        np.testing.assert_allclose(W @ SIGMA_TRUE @ W.T, np.eye(PHI_DIM), atol=1e-9)

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
        assert shares.max() < 0.30          # vs 1/PHI_DIM under the null

    def test_raw_norm_would_be_dominated_by_counts(self):
        """Motivates the whole module: unnormalised, beta_0 eats the norm."""
        obs = _sample(20000, seed=14, mean=BIAS)
        raw = (obs ** 2).mean(0)
        raw = raw / raw.sum()
        assert PHI_NAMES[int(np.argmax(raw))] == "beta0_per_mm2"
        assert raw[0] < 0.001               # CPA numerically invisible


class TestFloor:
    def test_returns_difference_covariance_not_per_observation(self):
        """Regression: an earlier version halved Sigma to a per-observation
        variance. That doubles every whitened observation and manufactures a
        constant bias^2 of exactly d on pure floor data."""
        a, b = _sample(4000, seed=15), _sample(4000, seed=16)
        est = split_half_floor(a, b)
        assert est.kind == "split_half"
        # Cov(a - b) = 2 * Cov(one observation)
        rel = np.abs(est.sd - np.sqrt(2) * FLOOR_SD) / (np.sqrt(2) * FLOOR_SD)
        assert rel.max() < 0.15

    def test_floor_satisfies_the_identity(self):
        """The property the whole subtraction rests on: E||delta||^2_Sigma^-1 = d."""
        a, b = _sample(8000, seed=25), _sample(8000, seed=26)
        est = split_half_floor(a, b)
        assert mahalanobis_sq(a - b, est.sigma).mean() == pytest.approx(PHI_DIM, rel=0.05)

    def test_cross_level_is_the_direct_measurement(self):
        a, b = _sample(4000, seed=27), _sample(4000, seed=28)
        est = cross_level_floor(a, b)
        assert est.kind == "cross_level"
        assert mahalanobis_sq(a - b, est.sigma).mean() == pytest.approx(PHI_DIM, rel=0.05)

    def test_split_half_rejects_mismatched_halves(self):
        with pytest.raises(ValueError):
            split_half_floor(_sample(10), _sample(9))

    def test_cross_stain_covers_only_invariant_terms(self):
        he = _sample(500, seed=17)
        psr = _sample(500, seed=18)
        est = cross_stain_floor(he, psr)
        assert est.kind == "cross_stain"
        # exactly the H&E-referenced block, whatever PHI_NAMES currently holds
        expected = tuple(n for n, ref in zip(PHI_NAMES, PHI_REFERENCE) if ref == "he")
        assert est.components == expected
        # collagen terms are not computable from H&E -> NaN, not a fabricated 0
        assert np.isnan(est.sigma[0, 0])
        assert np.isfinite(est.sigma[4, 4])

    def test_cross_stain_rejects_one_image_used_twice(self):
        """estimate_floor once built both sides from --real_he, making delta
        identically zero. A zero floor reads as maximal bias, so this has to
        fail loudly rather than return a bound that measures nothing."""
        phi = _sample(500, seed=17)
        with pytest.raises(ValueError, match="identically zero"):
            cross_stain_floor(phi, phi)

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


class TestPerDescriptorReport:
    """The section 7 go/no-go readout. A pooled floor hides whether any single
    descriptor - beta_0/beta_1 especially - is stable enough between levels."""

    def _phi(self, n=400, seed=30, scale=None):
        rng = np.random.default_rng(seed)
        scale = FLOOR_SD * 5 if scale is None else scale
        return rng.standard_normal((n, PHI_DIM)) * scale

    def test_one_row_per_descriptor_with_verdicts(self):
        est = split_half_floor(_sample(2000, seed=31), _sample(2000, seed=32))
        rows = per_descriptor_report(self._phi(), direct=est)
        assert [r["descriptor"] for r in rows] == list(PHI_NAMES)
        assert all(r["verdict"] in
                   {"usable", "marginal", "floor-limited",
                    "unknown (no floor estimate for this component)"} for r in rows)

    def test_flags_a_floor_limited_descriptor(self):
        """Give beta_0 a between-region spread no larger than its own floor."""
        est = split_half_floor(_sample(4000, seed=33), _sample(4000, seed=34))
        scale = FLOOR_SD * 5
        scale[1] = FLOOR_SD[1] * 0.8              # beta_0 signal below its floor
        rows = per_descriptor_report(self._phi(scale=scale), direct=est)
        by = {r["descriptor"]: r for r in rows}
        assert by["beta0_per_mm2"]["verdict"] == "floor-limited"
        assert by["beta0_per_mm2"]["floor_to_signal"] > 0.9
        assert by["task_specific_value"]["verdict"] == "usable"

    def test_direct_supersedes_the_bracket(self):
        lo = split_half_floor(_sample(500, seed=35), _sample(500, seed=36))
        direct = cross_level_floor(_sample(500, seed=37), _sample(500, seed=38))
        rows = per_descriptor_report(self._phi(), lower=lo, direct=direct)
        r = rows[0]
        assert r["floor_sd_used"] == pytest.approx(r["floor_sd_direct"])
        assert r["floor_sd_lower"] is not None

    def test_missing_floor_is_unknown_not_usable(self):
        """Collagen terms have no cross-stain upper bound; that must not read as
        a pass."""
        hi = cross_stain_floor(_sample(500, seed=39), _sample(500, seed=40))
        rows = per_descriptor_report(self._phi(), upper=hi)
        by = {r["descriptor"]: r for r in rows}
        assert by["task_specific_value"]["verdict"].startswith("unknown")
        assert by["lumen_fraction"]["verdict"] in {"usable", "marginal", "floor-limited"}

    def test_report_is_json_friendly(self):
        import json
        est = split_half_floor(_sample(300, seed=41), _sample(300, seed=42))
        json.dumps(per_descriptor_report(self._phi(), direct=est))


class TestVariogramFloor:
    """In-plane spatial variation as an upper bound on the level-offset floor.

    Without a second real PSR level the collagen descriptors have no cross-level
    upper bound at all - cross_stain cannot reach them, since collagen is not
    measurable in H&E. Only the split-half LOWER bound applies, and too small a
    floor inflates bias. This supplies the missing conservative bound.
    """

    def _field(self, n=900, corr_mm=2.0, extent=30.0, seed=0, n_groups=3):
        """Descriptors that decorrelate over a known length scale."""
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, extent, size=(n, 2))
        groups = np.repeat([f"wsi{i}" for i in range(n_groups)], n // n_groups)
        centres = rng.uniform(0, extent, size=(120, 2))
        W = rng.standard_normal((120, PHI_DIM))
        K = np.exp(-((coords[:, None, :] - centres[None]) ** 2).sum(-1) / (2 * corr_mm ** 2))
        phi = K @ W
        return phi / phi.std(0) * FLOOR_SD, coords, groups[: len(coords)]

    def test_sill_matches_the_decorrelated_limit(self):
        """At full decorrelation Var(a-b) = 2 Var(a), so the sill sd should be
        sqrt(2) times the field sd. That is the known answer here."""
        from uncertainty_phi.floor import variogram_floor
        phi, coords, groups = self._field()
        est, _ = variogram_floor(phi, coords, groups, n_bins=10)
        theory = np.sqrt(2) * phi.std(0, ddof=1)
        np.testing.assert_allclose(est.sd, theory, rtol=0.12)

    def test_semivariance_increases_with_lag(self):
        from uncertainty_phi.floor import variogram
        phi, coords, groups = self._field()
        c = variogram(phi, coords, groups, n_bins=8)
        sds = np.sqrt(c["cov"][:, 0, 0])
        assert sds[0] < sds[-1], "variogram must rise before it flattens"
        assert (c["lag_mm"][1:] > c["lag_mm"][:-1]).all()

    def test_extreme_lags_are_truncated(self):
        """Beyond ~half the domain only corner-to-corner pairs remain and edge
        effects produce a spurious upturn."""
        from uncertainty_phi.floor import variogram
        phi, coords, groups = self._field()
        wide = variogram(phi, coords, groups, n_bins=8, max_lag_fraction=1.0)
        narrow = variogram(phi, coords, groups, n_bins=8, max_lag_fraction=0.5)
        assert narrow["lag_mm"].max() < wide["lag_mm"].max()

    def test_pairs_never_cross_slides(self):
        """Pairing across slides would fold case-to-case biology into a floor."""
        from uncertainty_phi.floor import _within_group_pairs
        groups = np.array(["a"] * 5 + ["b"] * 5)
        ii, jj = _within_group_pairs(groups, 10_000, np.random.default_rng(0))
        assert (groups[ii] == groups[jj]).all()
        assert len(ii) == 2 * (5 * 4 // 2)          # within-group pairs only

    def test_sill_flag_is_per_descriptor(self):
        from uncertainty_phi.floor import variogram_floor
        phi, coords, groups = self._field()
        _, curve = variogram_floor(phi, coords, groups, n_bins=10)
        assert set(curve["sill_reached"]) == set(PHI_NAMES)
        assert isinstance(curve["sill_reached_all"], bool)

    def test_singleton_groups_rejected(self):
        from uncertainty_phi.floor import variogram_floor
        phi, coords, _ = self._field(n=6)
        with pytest.raises(ValueError, match="two or more"):
            variogram_floor(phi, coords, [f"w{i}" for i in range(6)])

    def test_report_marks_variogram_as_conservative(self):
        from uncertainty_phi.floor import variogram_floor
        phi, coords, groups = self._field()
        est, _ = variogram_floor(phi, coords, groups, n_bins=10)
        rows = per_descriptor_report(phi, variogram=est)
        for r in rows:
            assert r["floor_source"] == "variogram"
            assert r["bound_direction"] == "conservative"

    def test_split_half_alone_is_flagged_anti_conservative(self):
        """A component resting on the lower bound can only support an upper-bound
        claim about bias; the report must say so."""
        lo = split_half_floor(_sample(500, seed=60), _sample(500, seed=61))
        rows = per_descriptor_report(_sample(300, seed=62), lower=lo)
        assert all(r["floor_source"] == "split_half" for r in rows)
        assert all(r["bound_direction"] == "anti-conservative" for r in rows)


class TestPartiallyComputableDescriptors:
    """Running the floor without --real_he leaves lumen/tissue NaN throughout.

    Filtering rows on isfinite(...).all(axis=1) then threw away every row and
    took the four collagen descriptors down with the two that were never there —
    the whole go/no-go died on `need >= 2 finite pairs`.
    """

    @staticmethod
    def _phi(n=200, seed=0):
        rng = np.random.default_rng(seed)
        phi = _sample(n, seed=seed)
        phi[:, 4:] = np.nan            # no H&E supplied
        return phi

    def test_split_half_covers_the_computable_four(self):
        phi = self._phi()
        est = split_half_floor(phi[:100], phi[100:])
        assert est.components == PHI_NAMES[:4]
        assert np.isfinite(est.sigma[:4, :4]).all()
        assert np.isnan(est.sigma[4:, 4:]).all()

    def test_variogram_advertises_only_what_it_covers(self):
        phi = self._phi(280)
        rng = np.random.default_rng(1)
        est, _ = variogram_floor(phi, rng.random((280, 2)) * 20,
                                 [f"w{i // 14}" for i in range(280)], seed=0)
        assert est.components == PHI_NAMES[:4]
        assert np.isnan(est.sd[4:]).all()

    def test_report_marks_the_missing_unknown_not_usable(self):
        phi = self._phi()
        rows = per_descriptor_report(phi, lower=split_half_floor(phi[:100], phi[100:]))
        by_name = {r["descriptor"]: r for r in rows}
        assert by_name["beta0_per_mm2"]["floor_source"] == "split_half"
        for name in (n for n, ref in zip(PHI_NAMES, PHI_REFERENCE) if ref == "he"):
            assert by_name[name]["floor_source"] is None
            assert by_name[name]["floor_to_signal"] is None
            assert "unknown" in by_name[name]["verdict"]

    def test_all_nan_input_is_refused(self):
        phi = np.full((10, PHI_DIM), np.nan)
        with pytest.raises(ValueError, match="no descriptor is computable"):
            split_half_floor(phi[:5], phi[5:])


class TestSplitRegionsRespectsSlides:
    """Pairing across cases turns the split-half floor into between-case biology.

    On the liver cohort that pushed the "lower" bound above the variogram upper
    bound for every descriptor — an incoherent bracket.
    """

    def test_pairs_stay_within_a_slide(self):
        groups = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 11)
        ia, ib = split_regions(groups.size, seed=0, groups=groups)
        assert not (set(ia.tolist()) & set(ib.tolist()))
        assert (groups[ia] == groups[ib]).all()

    def test_odd_slides_drop_their_leftover_region(self):
        groups = np.array(["a"] * 11)
        ia, ib = split_regions(11, seed=0, groups=groups)
        assert len(ia) == len(ib) == 5

    def test_single_region_slides_contribute_nothing(self):
        groups = np.array(["a", "b", "c"])
        ia, ib = split_regions(3, seed=0, groups=groups)
        assert len(ia) == len(ib) == 0

    def test_cross_slide_pairing_inflates_the_floor(self):
        """The bug, quantified: with a per-case offset the pooled split reads a
        floor that is mostly case-to-case difference."""
        rng = np.random.default_rng(0)
        n_per, n_slides = 30, 8
        groups = np.repeat([f"s{i}" for i in range(n_slides)], n_per)
        phi = _sample(n_per * n_slides, seed=3)

        # a per-case offset larger than beta_0's own spread (FLOOR_SD[1] = 40),
        # which is the regime the liver cohort is in
        offsets = rng.normal(0, 100.0, n_slides).repeat(n_per)
        shifted = phi.copy()
        shifted[:, 1] += offsets

        ja, jb = split_regions(phi.shape[0], seed=0, groups=groups)
        within = split_half_floor(shifted[ja], shifted[jb]).sd[1]
        baseline = split_half_floor(phi[ja], phi[jb]).sd[1]
        # within-slide pairing cancels the offset entirely
        assert within == pytest.approx(baseline, rel=0.02)

        ia, ib = split_regions(phi.shape[0], seed=0)               # pooled: wrong
        pooled = split_half_floor(shifted[ia], shifted[ib]).sd[1]
        # ~1.8x here, not more: with 8 slides one pooled pair in eight happens to
        # stay within a slide anyway, and Ledoit-Wolf shrinks what is left
        assert pooled > 1.5 * within
