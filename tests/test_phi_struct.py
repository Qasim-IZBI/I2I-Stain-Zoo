"""φ_struct descriptor tests — known-answer cases from kidney_ood_data_plan.md §5."""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_phi.descriptors import (
    PHI_DIM,
    PHI_NAMES,
    PHI_REFERENCE,
    betti,
    clean_mask,
    collagen_fraction,
    lumen_tissue_fraction,
    phi_struct,
    regional_dispersion,
)

H = W = 512
YY, XX = np.mgrid[0:H, 0:W]


def _rings_and_blobs():
    """Rings around lumens vs solid blobs of EQUAL collagen area — the §5.3
    lumen-filler: same CPA, same beta_0, beta_1 collapses."""
    R, r = 34, 24
    Rf = np.sqrt(R ** 2 - r ** 2)          # disk with the annulus's area
    rings = np.zeros((H, W), bool)
    blobs = np.zeros((H, W), bool)
    for cy in range(64, H, 96):
        for cx in range(64, W, 96):
            d2 = (YY - cy) ** 2 + (XX - cx) ** 2
            rings |= (d2 <= R * R) & (d2 >= r * r)
            blobs |= d2 <= Rf * Rf
    return rings, blobs


def _fibres(theta, thickness=6, spacing=24):
    d = XX * np.cos(theta) + YY * np.sin(theta)
    return np.abs((d % spacing) - spacing / 2) < thickness / 2


def _labels(collagen):
    lab = np.ones(collagen.shape, np.uint8)   # all tissue
    lab[collagen] = 2                         # collagen-positive
    return lab


class TestBetti:
    def test_lumen_filler_is_caught(self):
        rings, blobs = _rings_and_blobs()
        assert abs(rings.mean() - blobs.mean()) < 0.005      # CPA indistinguishable
        b0_r, b1_r = betti(rings)
        b0_b, b1_b = betti(blobs)
        assert b0_r == b0_b                                   # beta_0 blind too
        assert b1_r >= 20 and b1_b == 0                       # only beta_1 sees it

    def test_empty_mask(self):
        assert betti(np.zeros((32, 32), bool)) == (0, 0)

    def test_solid_mask_has_no_loops(self):
        assert betti(np.ones((32, 32), bool)) == (1, 0)

    def test_disjoint_components_counted(self):
        m = np.zeros((64, 64), bool)
        m[5:15, 5:15] = True
        m[40:50, 40:50] = True
        assert betti(m) == (2, 0)


class TestRegionalDispersion:
    def test_aligned_vs_scrambled(self):
        aligned = _fibres(np.deg2rad(30))
        rng = np.random.default_rng(0)
        scrambled = np.zeros((H, W), bool)
        B = 128
        for by in range(0, H, B):
            for bx in range(0, W, B):
                f = _fibres(rng.uniform(0, np.pi))
                scrambled[by : by + B, bx : bx + B] = f[by : by + B, bx : bx + B]

        # same amount of collagen, same connectivity scale
        assert abs(aligned.mean() - scrambled.mean()) < 0.02
        d_aligned = regional_dispersion(aligned)
        d_scrambled = regional_dispersion(scrambled)
        assert d_aligned < 0.05
        assert d_scrambled > 0.3
        assert d_scrambled > 5 * d_aligned

    def test_empty_is_nan(self):
        assert np.isnan(regional_dispersion(np.zeros((32, 32), bool)))


class TestCleanMask:
    def test_removes_speckle_but_keeps_structure(self):
        m = np.zeros((128, 128), bool)
        m[20:60, 20:60] = True                 # a real object
        rng = np.random.default_rng(1)
        for _ in range(200):                   # single-pixel noise
            m[rng.integers(0, 128), rng.integers(0, 128)] = True
        assert betti(m)[0] > 50
        cleaned = clean_mask(m, min_object_px=16)
        assert betti(cleaned)[0] == 1


class TestCollagenFraction:
    def test_matches_manual_count(self):
        lab = np.ones((10, 10), np.uint8)
        lab[:3] = 2                            # 30 of 100 positive
        assert collagen_fraction(lab) == pytest.approx(0.30)

    def test_background_excluded_from_denominator(self):
        lab = np.zeros((10, 10), np.uint8)     # half background
        lab[:5, :] = 1
        lab[:1, :] = 2                         # 10 positive, 40 tissue
        assert collagen_fraction(lab) == pytest.approx(10 / 50)

    def test_no_tissue_is_nan(self):
        assert np.isnan(collagen_fraction(np.zeros((8, 8), np.uint8)))


class TestDensityInvariance:
    def test_counts_are_per_mm2_not_raw(self):
        """Tiling the same structure 2x2 must not change the density."""
        rings, _ = _rings_and_blobs()
        small = phi_struct(_labels(rings))
        big = phi_struct(_labels(np.tile(rings, (2, 2))))
        assert big[1] == pytest.approx(small[1], rel=1e-6)   # beta_0 / mm^2
        assert big[2] == pytest.approx(small[2], rel=1e-6)   # beta_1 / mm^2

    def test_mpp_scales_density(self):
        rings, _ = _rings_and_blobs()
        a = phi_struct(_labels(rings), mpp=0.442)
        b = phi_struct(_labels(rings), mpp=0.221)
        # 4x the pixels per mm^2 at half the mpp -> 4x the density
        assert b[1] == pytest.approx(4 * a[1], rel=1e-6)


class TestLumenTissueFraction:
    def test_recovers_known_lumen(self):
        he = np.full((200, 200, 3), 0.5, np.float32)     # tissue
        he[80:120, 80:120] = 0.98                        # a 40x40 lumen
        lumen, tissue = lumen_tissue_fraction(he)
        assert tissue == pytest.approx(1.0, abs=1e-6)
        assert lumen == pytest.approx(1600 / 40000, rel=0.05)

    def test_uint8_input_accepted(self):
        he = np.full((64, 64, 3), 128, np.uint8)
        lumen, tissue = lumen_tissue_fraction(he)
        assert 0.0 <= lumen <= 1.0 and tissue == pytest.approx(1.0)

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError):
            lumen_tissue_fraction(np.zeros((10, 10)))

    def test_border_touching_lumen_is_not_lost(self):
        """Regression: binary_fill_holes only fills *enclosed* background, so a
        lumen cut by the crop edge used to vanish and be counted as non-tissue.
        At 1.5mm regions with 100-500um vessels that is a lot of lumens."""
        he = np.full((200, 200, 3), 0.5, np.float32)
        he[150:, 150:] = 0.98                     # lumen running off the corner
        lumen, tissue = lumen_tissue_fraction(he)
        assert lumen > 0.0, "border-touching lumen was dropped"
        assert lumen == pytest.approx(2500 / 40000, rel=0.05)
        assert tissue == pytest.approx(1.0, abs=1e-6)

    def test_wsi_footprint_encloses_interior_lumen(self):
        from uncertainty_phi.descriptors import he_tissue_footprint
        he = np.full((200, 200, 3), 0.5, np.float32)
        he[80:120, 80:120] = 0.98
        fp = he_tissue_footprint(he)
        assert fp.all(), "an interior lumen must be inside the tissue footprint"

    def test_supplied_footprint_overrides_heuristic(self):
        """The pipeline passes a WSI-level footprint crop; it must be honoured."""
        he = np.full((100, 100, 3), 0.5, np.float32)
        he[:50] = 0.98                            # top half bright
        forced = np.zeros((100, 100), bool)
        forced[:50] = True                        # declare that half as tissue
        lumen, tissue = lumen_tissue_fraction(he, tissue_mask=forced)
        assert tissue == pytest.approx(0.5)
        assert lumen == pytest.approx(1.0)        # all of it is bright


class TestPhiStruct:
    def test_shape_and_names_agree(self):
        assert len(PHI_NAMES) == PHI_DIM == len(PHI_REFERENCE)
        v = phi_struct(_labels(_rings_and_blobs()[0]))
        assert v.shape == (PHI_DIM,)

    def test_he_absent_gives_nan_not_zero(self):
        """A zero would be silently absorbed by downstream means; NaN is visible."""
        v = phi_struct(_labels(_rings_and_blobs()[0]))
        assert np.isnan(v[4]) and np.isnan(v[5])
        assert np.isfinite(v[:4]).all()

    def test_he_present_fills_geometric_terms(self):
        rings, _ = _rings_and_blobs()
        he = np.full((H, W, 3), 0.6, np.float32)
        he[rings] = 0.3
        v = phi_struct(_labels(rings), he)
        assert np.isfinite(v).all()

    def test_reference_classes_are_declared(self):
        """The two-reference split of section 6.0 must be machine-readable."""
        assert PHI_REFERENCE[:4] == ("psr", "psr", "psr", "psr")
        assert PHI_REFERENCE[4:] == ("he", "he")

    def test_rejects_3d_labels(self):
        with pytest.raises(ValueError):
            phi_struct(np.zeros((8, 8, 3), np.uint8))
