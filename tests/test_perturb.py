"""Stain-perturbation tests — kidney_ood_data_plan.md §6.2."""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_phi.perturb import (
    StainStats,
    appearance_gap,
    interpolate_stats,
    lab_to_rgb,
    perturbation_series,
    pool_stats,
    reinhard_transfer,
    rgb_to_lab,
    stain_stats,
)


def _img(seed=0, tint=(0.0, 0.0, 0.0), shape=(64, 64)):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.75, size=(*shape, 3)).astype(np.float32)
    return np.clip(base + np.asarray(tint, np.float32), 0, 1)


class TestLabRoundTrip:
    def test_round_trip_is_lossless(self):
        rgb = _img(1)
        np.testing.assert_allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1e-4)

    def test_known_anchors(self):
        white = rgb_to_lab(np.ones((1, 1, 3), np.float32))[0, 0]
        black = rgb_to_lab(np.zeros((1, 1, 3), np.float32))[0, 0]
        assert white[0] == pytest.approx(100.0, abs=0.1)
        assert abs(white[1]) < 0.1 and abs(white[2]) < 0.1
        assert black[0] == pytest.approx(0.0, abs=0.1)

    def test_uint8_accepted(self):
        a = stain_stats((_img(2) * 255).astype(np.uint8))
        b = stain_stats(_img(2))
        np.testing.assert_allclose(a.mean, b.mean, atol=0.6)


class TestStainStats:
    def test_mask_restricts_to_tissue(self):
        img = _img(3)
        img[:32] = 1.0                       # bright "background" half
        m = np.zeros((64, 64), bool)
        m[32:] = True
        masked = stain_stats(img, m)
        assert masked.n_pixels == 32 * 64
        assert masked.mean[0] < stain_stats(img).mean[0]   # excluding white lowers L

    def test_empty_mask_rejected(self):
        with pytest.raises(ValueError):
            stain_stats(_img(4), np.zeros((64, 64), bool))

    def test_pool_weights_by_pixel_count(self):
        a = StainStats(np.array([10.0, 0, 0]), np.array([1.0, 1, 1]), 100)
        b = StainStats(np.array([20.0, 0, 0]), np.array([1.0, 1, 1]), 300)
        pooled = pool_stats([a, b])
        assert pooled.mean[0] == pytest.approx(17.5)       # 0.25*10 + 0.75*20
        assert pooled.n_pixels == 400
        # pooled sd must exceed the within-slide sd, since the means differ
        assert pooled.std[0] > 1.0


class TestTransfer:
    def test_transfer_to_own_stats_is_identity(self):
        img = _img(5)
        s = stain_stats(img)
        np.testing.assert_allclose(reinhard_transfer(img, s, s), img, atol=1e-3)

    def test_transfer_matches_target_stats(self):
        src_img, dst_img = _img(6), _img(7, tint=(0.15, -0.05, 0.10))
        s, d = stain_stats(src_img), stain_stats(dst_img)
        out = reinhard_transfer(src_img, s, d)
        got = stain_stats(out)
        np.testing.assert_allclose(got.mean, d.mean, atol=1.5)
        np.testing.assert_allclose(got.std, d.std, rtol=0.15)

    def test_background_left_untouched_when_masked(self):
        img = _img(8)
        m = np.zeros((64, 64), bool)
        m[32:] = True
        s = stain_stats(img, m)
        d = stain_stats(_img(9, tint=(0.2, 0, 0)), m)
        out = reinhard_transfer(img, s, d, tissue_mask=m)
        np.testing.assert_allclose(out[:32], img[:32], atol=1e-5)   # background
        assert not np.allclose(out[32:], img[32:], atol=1e-3)       # tissue moved


class TestInterpolation:
    def test_t0_is_source_t1_is_target(self):
        s = StainStats(np.array([50.0, 1, 2]), np.array([10.0, 3, 4]), 10)
        d = StainStats(np.array([60.0, 5, 6]), np.array([20.0, 7, 8]), 10)
        np.testing.assert_allclose(interpolate_stats(s, d, 0.0).mean, s.mean)
        np.testing.assert_allclose(interpolate_stats(s, d, 1.0).mean, d.mean)
        np.testing.assert_allclose(interpolate_stats(s, d, 0.5).mean, (s.mean + d.mean) / 2)


class TestPerturbationSeries:
    def test_t0_returns_the_original(self):
        """The anchor of the whole test: at t=0 nothing has changed, so any
        descriptor movement across the series is attributable to appearance."""
        img = _img(10)
        s = stain_stats(img)
        d = stain_stats(_img(11, tint=(0.2, -0.1, 0.05)))
        series = perturbation_series(img, s, d)
        np.testing.assert_allclose(series[0.0], img, atol=1e-3)

    def test_monotone_progression_toward_the_target(self):
        img = _img(12)
        s = stain_stats(img)
        d = stain_stats(_img(13, tint=(0.25, 0, 0)))
        series = perturbation_series(img, s, d)
        dists = [abs(stain_stats(series[t]).mean[0] - d.mean[0])
                 for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert dists == sorted(dists, reverse=True), "should approach the target"

    def test_transform_is_pointwise_so_anatomy_cannot_move(self):
        """The invariance that matters: the map depends only on a pixel's colour,
        never on its position. Two pixels sharing an input colour must share an
        output colour, so no spatial structure can be created or destroyed.

        (Correlating output-R against input-R is the WRONG test — LAB transfer
        mixes channels, so per-channel monotonicity is not expected.)"""
        rng = np.random.default_rng(14)
        base = rng.uniform(0.25, 0.75, size=(8, 8, 3)).astype(np.float32)
        img = np.tile(base, (4, 4, 1))                 # same 8x8 patch, 16 places
        s = stain_stats(img)
        d = stain_stats(_img(15, tint=(0.2, 0.1, -0.1)))
        for t, out in perturbation_series(img, s, d).items():
            tiles = [out[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] for r in range(4) for c in range(4)]
            for tile in tiles[1:]:
                np.testing.assert_allclose(tile, tiles[0], atol=1e-6,
                                           err_msg=f"map is position-dependent at t={t}")

    def test_clipping_is_reported(self):
        """Clipping is non-invertible and would break the fixed-anatomy premise,
        so the series must expose how much of it is happening."""
        from uncertainty_phi.perturb import clipped_fraction
        img = _img(30)
        s = stain_stats(img)
        mild = stain_stats(_img(31, tint=(0.02, 0.0, 0.0)))
        assert clipped_fraction(img, s, mild) < 0.05
        extreme = StainStats(s.mean + np.array([80.0, 60.0, 60.0]), s.std * 4, s.n_pixels)
        assert clipped_fraction(img, s, extreme) > 0.5

    def test_custom_fractions(self):
        img = _img(16)
        s = stain_stats(img)
        series = perturbation_series(img, s, s, fractions=(0.0, 0.5))
        assert sorted(series) == [0.0, 0.5]


class TestAppearanceGap:
    def test_reports_scale_free_shift(self):
        s = StainStats(np.array([50.0, 0, 0]), np.array([10.0, 10, 10]), 10)
        d = StainStats(np.array([60.0, 0, 0]), np.array([20.0, 10, 10]), 10)
        gap = appearance_gap(s, d)
        assert gap["delta_mean_L"] == pytest.approx(10.0)
        assert gap["delta_mean_over_sd"][0] == pytest.approx(1.0)
        assert gap["std_ratio"][0] == pytest.approx(2.0)

    def test_json_friendly(self):
        import json
        s = stain_stats(_img(17))
        json.dumps(appearance_gap(s, s))
        json.dumps(s.as_dict())
