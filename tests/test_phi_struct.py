"""φ_struct descriptor tests — known-answer cases from kidney_ood_data_plan.md §5."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
        assert PHI_REFERENCE[4:] == ("he", "he", "he")

    def test_rejects_3d_labels(self):
        with pytest.raises(ValueError):
            phi_struct(np.zeros((8, 8, 3), np.uint8))


class TestRoiFilter:
    """Anatomical restriction, e.g. cortex on the kidney arm.

    The threshold is on coverage rather than on the centre point, because a
    region straddling the cortex/medulla boundary is not a cortex measurement.
    """

    @staticmethod
    def _grid():
        from uncertainty_phi.regions import Region
        # four 10x10 regions in a row, x = 0,10,20,30
        return [Region(wsi="w", index=i, y0=0, y1=10, x0=10 * i, x1=10 * i + 10)
                for i in range(4)]

    def test_keeps_only_regions_inside_the_roi(self):
        from uncertainty_phi.regions import filter_by_roi
        roi = np.zeros((10, 40), bool)
        roi[:, :20] = True                      # first two regions fully inside
        kept = filter_by_roi(self._grid(), roi, min_roi_fraction=0.5)
        assert [r.index for r in kept] == [0, 1]

    def test_half_covered_region_is_excluded_at_a_strict_threshold(self):
        """Centre-point selection would admit it; coverage must not."""
        from uncertainty_phi.regions import filter_by_roi
        roi = np.zeros((10, 40), bool)
        roi[:, :25] = True                      # region 2 is 50% covered
        assert [r.index for r in filter_by_roi(self._grid(), roi,
                                               min_roi_fraction=0.9)] == [0, 1]
        assert [r.index for r in filter_by_roi(self._grid(), roi,
                                               min_roi_fraction=0.5)] == [0, 1, 2]

    def test_empty_roi_keeps_nothing(self):
        from uncertainty_phi.regions import filter_by_roi
        assert filter_by_roi(self._grid(), np.zeros((10, 40), bool)) == []

    def test_mask_is_resized_to_the_reconstruction(self):
        """Cortex is annotated on a thumbnail, so a size mismatch is normal."""
        import tifffile
        from uncertainty_phi.ensemble import load_roi_mask
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.tif"
            small = np.zeros((5, 20), np.uint8)
            small[:, :10] = 255                 # left half
            tifffile.imwrite(str(p), small)
            roi = load_roi_mask(p, (10, 40))
            assert roi.shape == (10, 40)
            assert roi.dtype == bool
            assert roi[:, :20].all() and not roi[:, 20:].any()


class TestWhiteThresh:
    """The whitespace cut is scanner-dependent and must be tunable end to end.

    Lumens at grey ~185 sit below the 0.85 default, so they read as tissue and
    lumen_fraction collapses to ~0 — the failure seen on the UC liver cohort.

    Every case goes through the WSI-level footprint, as the pipeline does.
    Without one, `lumen_tissue_fraction` pads the border as tissue and the slide
    background itself is counted as lumen.
    """

    @staticmethod
    def _slide(lumen_value: int) -> np.ndarray:
        """Tissue square with an enclosed lumen, on white slide background."""
        he = np.full((60, 60, 3), 250, np.uint8)      # background
        he[10:50, 10:50] = 150                        # tissue
        he[25:35, 25:35] = lumen_value                # lumen inside the tissue
        return he

    @staticmethod
    def _measure(he, thresh):
        from uncertainty_phi.descriptors import (
            he_tissue_footprint, lumen_tissue_fraction,
        )
        fp = he_tissue_footprint(he, white_thresh=thresh)
        return lumen_tissue_fraction(he, tissue_mask=fp, white_thresh=thresh)

    def test_default_misses_a_dim_lumen(self):
        he = self._slide(185)                    # 185/255 = 0.725
        lumen, tissue = self._measure(he, 0.85)
        assert lumen == 0.0
        assert tissue == pytest.approx(1600 / 3600, rel=1e-6)

    def test_lowered_threshold_recovers_it(self):
        lumen, tissue = self._measure(self._slide(185), 0.70)
        assert lumen == pytest.approx(100 / 1600, rel=1e-6)   # 10x10 in 40x40
        assert tissue == pytest.approx(1600 / 3600, rel=1e-6) # footprint unchanged

    def test_a_bright_lumen_is_found_either_way(self):
        for thresh in (0.70, 0.85):
            lumen, _ = self._measure(self._slide(250), thresh)
            assert lumen == pytest.approx(100 / 1600, rel=1e-6)

    def test_threshold_reaches_phi_struct(self):
        from uncertainty_phi.descriptors import he_tissue_footprint, phi_struct
        he = self._slide(185)
        labels = np.ones(he.shape[:2], np.uint8)

        default = phi_struct(labels, he,
                             tissue_mask=he_tissue_footprint(he))
        lowered = phi_struct(labels, he, white_thresh=0.70,
                             tissue_mask=he_tissue_footprint(he, white_thresh=0.70))
        assert default[4] == 0.0
        assert lowered[4] == pytest.approx(100 / 1600, rel=1e-6)

    def test_phi_for_wsi_passes_it_to_the_footprint_too(self, tmp_path):
        """A footprint built at a different cut would intersect two different
        definitions of whitespace, so phi_for_wsi must forward the threshold."""
        import tifffile

        from uncertainty_phi.ensemble import phi_for_wsi
        from uncertainty_phi.regions import Region

        he = self._slide(185)
        member = tmp_path / "model_01"
        member.mkdir()
        tifffile.imwrite(str(member / "s.tif"), np.ones(he.shape[:2], np.uint8))
        he_path = tmp_path / "s_he.tif"
        tifffile.imwrite(str(he_path), he)

        regions = [Region(wsi="s", index=0, y0=0, y1=60, x0=0, x1=60)]
        out, tissue_frac = phi_for_wsi([member], "s", regions, he_path=he_path,
                                       mpp=0.221, white_thresh=0.70)
        assert out[0, 0, 4] == pytest.approx(100 / 1600, rel=1e-6)
        # rides alongside phi rather than inside it
        assert tissue_frac.shape == (len(regions),)
        assert 0.0 <= tissue_frac[0] <= 1.0


class TestGridWithoutTiling:
    """The real SR is evaluated whole-slide and has no tiles_metadata.csv.

    Sizing the grid from the image instead must give the same construction, so
    the arm that was never tiled is not a second code path with its own bugs.
    """

    def test_matches_the_metadata_route_on_a_tile_aligned_extent(self, tmp_path):
        import pandas as pd

        from uncertainty_phi.regions import region_grid, region_grid_from_extent

        h = w = 2048
        rows = [{"source_file": "s.tif", "x": x, "y": y, "tile_size": 512}
                for y in range(0, h, 512) for x in range(0, w, 512)]
        csv = tmp_path / "tiles_metadata.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)

        a = region_grid(csv, region_mm=0.2, mpp=0.221)
        b = region_grid_from_extent("s.tif", h, w, region_mm=0.2, mpp=0.221)
        assert [(r.y0, r.y1, r.x0, r.x1) for r in a] == \
               [(r.y0, r.y1, r.x0, r.x1) for r in b]

    def test_untiled_extent_can_hold_one_more_region_row(self):
        """A tiled extent is truncated to whole tiles, so the two routes are not
        interchangeable — they differ by at most an edge row/column."""
        from uncertainty_phi.regions import region_grid_from_extent

        side = int(round(0.2 * 1000.0 / 0.221))
        tiled = region_grid_from_extent("s", 3 * side, 3 * side,
                                        region_mm=0.2, mpp=0.221)
        untiled = region_grid_from_extent("s", 3 * side + 10, 3 * side + 10,
                                          region_mm=0.2, mpp=0.221)
        assert len(tiled) == len(untiled) == 9      # drop_partial eats the sliver


class TestLumenQC:
    """One QC image per WSI, so a thresholded lumen_fraction can be eyeballed.

    The number alone cannot distinguish "found the lumens" from "found pale
    tissue", which is the whole reason the cohort's 1e-5 went unnoticed.
    """

    @staticmethod
    def _slide():
        he = np.full((120, 200, 3), 250, np.uint8)
        he[20:100, 20:180] = 150                   # tissue
        he[50:70, 40:60] = 250                     # a bright lumen inside it
        return he

    def _regions(self):
        from uncertainty_phi.regions import Region
        # r0 straddles the tissue edge, r1 is fully interior
        return [Region(wsi="s", index=0, y0=0, y1=60, x0=0, x1=100),
                Region(wsi="s", index=1, y0=30, y1=90, x0=30, x1=130)]

    def test_writes_a_label_tif_and_its_he_crop(self, tmp_path):
        import json

        import tifffile

        from uncertainty_phi.descriptors import he_tissue_footprint
        from uncertainty_phi.ensemble import save_lumen_qc

        he = self._slide()
        fp = he_tissue_footprint(he, white_thresh=0.70)
        out = save_lumen_qc(he, fp, self._regions(), "slide_a", tmp_path,
                            white_thresh=0.70)

        # the interior region wins on footprint coverage, not the edge one
        assert out.name == "slide_a_r0001_y30_x30_lumen.tif"
        assert (tmp_path / "slide_a_r0001_y30_x30_he.tif").is_file()

        label = tifffile.imread(str(out))
        assert label.dtype == np.uint8
        assert set(np.unique(label)) <= {0, 1, 2}
        assert (label == 2).any()          # the lumen is inside this region
        assert label.shape == (60, 100)     # full resolution, not downsampled

        with tifffile.TiffFile(out) as tf:
            meta = json.loads(tf.pages[0].description)
        assert meta["white_thresh"] == 0.70
        assert meta["labels"]["2"] == "lumen"
        assert meta["lumen_fraction"] == pytest.approx(
            (label == 2).sum() / (label >= 1).sum(), rel=1e-9)

    def test_max_px_downsamples_and_records_it(self, tmp_path):
        import json

        import tifffile

        from uncertainty_phi.descriptors import he_tissue_footprint
        from uncertainty_phi.ensemble import save_lumen_qc

        he = self._slide()
        out = save_lumen_qc(he, he_tissue_footprint(he, white_thresh=0.70),
                            self._regions(), "s", tmp_path,
                            white_thresh=0.70, max_px=50)
        assert max(tifffile.imread(str(out)).shape) <= 50
        with tifffile.TiffFile(out) as tf:
            assert json.loads(tf.pages[0].description)["downsample"] == 2

    def test_no_regions_writes_nothing(self, tmp_path):
        from uncertainty_phi.descriptors import he_tissue_footprint
        from uncertainty_phi.ensemble import save_lumen_qc

        he = self._slide()
        assert save_lumen_qc(he, he_tissue_footprint(he), [], "s", tmp_path,
                             white_thresh=0.70) is None
        assert not list(tmp_path.glob("*.tif"))

    def test_off_by_default(self, tmp_path):
        """qc_dir=None must not create anything — it is opt-in."""
        import tifffile

        from uncertainty_phi.ensemble import phi_for_wsi

        he_path = tmp_path / "s.tif"
        tifffile.imwrite(str(he_path), self._slide())
        member = tmp_path / "model_01"
        member.mkdir()
        tifffile.imwrite(str(member / "s.tif"),
                         np.ones(self._slide().shape[:2], np.uint8))

        phi_for_wsi([member], "s", self._regions(), he_path=he_path, mpp=0.221,
                    white_thresh=0.70)
        assert not list(tmp_path.glob("*_lumen.tif"))


class TestLumenTopology:
    """β₀/β₁ of the lumen space — the direct test of the §5.3 lumen-filler failure.

    A model that paints collagen over vessels keeps the lumen *area* roughly and
    loses the loops, so area alone cannot see it.
    """

    @staticmethod
    def _slide(lumens):
        """White background, tissue block, `lumens` as (y, x, size) bright squares."""
        he = np.full((200, 300, 3), 250, np.uint8)
        he[20:180, 20:280] = 150
        for y, x, s in lumens:
            he[y:y + s, x:x + s] = 250
        return he

    def _phi(self, he, **kw):
        from uncertainty_phi.descriptors import he_tissue_footprint, phi_struct
        fp = he_tissue_footprint(he, white_thresh=0.70)
        labels = np.ones(he.shape[:2], np.uint8)
        return phi_struct(labels, he, mpp=1.0, tissue_mask=fp, white_thresh=0.70,
                          min_object_px=1, **kw)

    def test_counts_separate_lumens_as_components(self):
        one = self._phi(self._slide([(60, 60, 20)]))
        three = self._phi(self._slide([(60, 60, 20), (60, 120, 20), (120, 60, 20)]))
        assert three[5] == pytest.approx(3 * one[5], rel=1e-9)   # beta0_lumen density
        assert three[4] == pytest.approx(3 * one[4], rel=1e-9)   # lumen_fraction

    def test_area_and_topology_disagree_when_a_ring_is_filled(self):
        """Same bright area, different loop count: an annulus of whitespace has
        β₁ = 1, and filling its centre with tissue leaves β₁ = 0."""
        from uncertainty_phi.descriptors import (
            betti, he_tissue_footprint, lumen_mask,
        )

        # the WSI-level footprint, as the pipeline supplies it — passing None
        # instead uses the standalone fallback, which pads the border as tissue
        # and would count the slide background as lumen
        he = self._slide([(60, 60, 40)])
        he[70:90, 70:90] = 150            # tissue plug inside the lumen -> a ring
        ring, _ = lumen_mask(he, he_tissue_footprint(he, white_thresh=0.70), 0.70)
        assert betti(ring)[1] == 1

        flat = self._slide([(60, 60, 40)])
        solid, _ = lumen_mask(flat, he_tissue_footprint(flat, white_thresh=0.70), 0.70)
        assert betti(solid)[1] == 0
        assert np.count_nonzero(solid) > np.count_nonzero(ring)   # area differs too

    def test_densities_use_the_footprint_denominator(self):
        """Not the label mask's tissue: the reference side is the real H&E and
        has no labels, so only the footprint is available to both sides."""
        he = self._slide([(60, 60, 20)])
        v = self._phi(he)
        footprint_mm2 = (160 * 260) * (1.0 ** 2) / 1e6
        assert v[5] == pytest.approx(1 / footprint_mm2, rel=1e-9)

    def test_no_rgb_leaves_all_three_lumen_terms_nan(self):
        from uncertainty_phi.descriptors import phi_struct
        v = phi_struct(np.ones((40, 40), np.uint8), None, mpp=1.0)
        assert np.isnan(v[4:]).all()
        assert np.isfinite(v[0])


class TestPrecomputedLumen:
    """phi_struct takes a lumen mask from make_lumen_masks.py, not just RGB.

    The mask is member-specific, so the RGB path would load several GB fifty
    times per slide. It is also cleaned once per slide, which is what keeps
    lumen_fraction and the Betti numbers measuring the same object.
    """

    @staticmethod
    def _case():
        lum = np.zeros((100, 100), bool)
        lum[20:40, 20:40] = True          # one 20x20 lumen
        lum[60, 60] = True                # one speck
        return lum, np.ones((100, 100), bool)

    def test_precomputed_matches_thresholding_the_same_image(self):
        from uncertainty_phi.descriptors import (
            he_tissue_footprint, lumen_mask, phi_struct,
        )

        he = np.full((100, 140, 3), 150, np.uint8)
        he[30:50, 30:50] = 250
        fp = he_tissue_footprint(he, white_thresh=0.70)
        labels = np.ones(he.shape[:2], np.uint8)

        from_rgb = phi_struct(labels, he, mpp=1.0, tissue_mask=fp, white_thresh=0.70)
        lum, _ = lumen_mask(he, fp, 0.70)
        from_mask = phi_struct(labels, None, mpp=1.0, tissue_mask=fp, lumen=lum)
        np.testing.assert_allclose(from_rgb[4:], from_mask[4:], rtol=1e-12)

    def test_area_and_topology_see_the_same_cleaned_mask(self):
        """The bug this closes: cleaning inside phi_struct left lumen_fraction on
        the raw mask while betti ran on the cleaned one, so a speck counted for
        area but not for beta0."""
        from uncertainty_phi.descriptors import clean_mask, phi_struct

        lum, fp = self._case()
        cleaned = clean_mask(lum, 64, 0)          # drops the single-pixel speck
        v = phi_struct(np.ones((100, 100), np.uint8), None, mpp=1.0,
                       tissue_mask=fp, lumen=cleaned)
        assert v[4] == pytest.approx(400 / 10000)   # area of the 20x20 only
        assert v[5] == pytest.approx(1 / (10000 * 1e-6))  # exactly one component

    def test_lumen_overrides_rgb_when_both_are_given(self):
        from uncertainty_phi.descriptors import phi_struct
        lum, fp = self._case()
        he = np.full((100, 100, 3), 250, np.uint8)   # would be all-bright
        v = phi_struct(np.ones((100, 100), np.uint8), he, mpp=1.0,
                       tissue_mask=fp, lumen=lum)
        assert v[4] < 0.05          # the mask won, not the all-white RGB


class TestFootprintFromTissueMask:
    """The footprint comes from the H&E tissue mask the CPA pipeline uses.

    Deriving it by thresholding instead puts white_thresh under every lumen
    density and under the enclosure test — and the footprint is the thing that
    breaks across the threshold sweep, not the numerator.
    """

    def test_fills_lumens_a_tissue_segmentation_left_out(self):
        """A real tissue mask has lumens as holes. Unfilled, every lumen would
        sit outside the tissue it belongs to and count as slide background."""
        from uncertainty_phi.descriptors import tissue_footprint_from_mask

        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:80] = 1
        mask[40:60, 40:60] = 0              # a lumen, excluded by the segmenter
        fp = tissue_footprint_from_mask(mask)
        assert fp[50, 50]                    # restored
        assert not fp[5, 5]                  # true background stays out
        assert fp.sum() == 60 * 60

    def test_is_a_no_op_when_lumens_are_already_included(self):
        from uncertainty_phi.descriptors import tissue_footprint_from_mask
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:80] = 1
        np.testing.assert_array_equal(tissue_footprint_from_mask(mask), mask > 0)

    def test_agrees_with_the_brightness_footprint_on_the_same_slide(self):
        """The two routes must not disagree, or the study carries two
        definitions of tissue."""
        from uncertainty_phi.descriptors import (
            he_tissue_footprint, tissue_footprint_from_mask,
        )
        he = np.full((120, 160, 3), 250, np.uint8)
        he[20:100, 20:140] = 150
        he[50:70, 50:70] = 250                       # a lumen

        by_brightness = he_tissue_footprint(he, white_thresh=0.90)
        by_mask = tissue_footprint_from_mask(he.min(axis=2) < 230)
        np.testing.assert_array_equal(by_brightness, by_mask)


class TestPlanarTiffReading:
    """Planar (C,H,W) exports must read the same as interleaved (H,W,C).

    Both readers index the LAST axis — arr[..., :3], arr[..., 0] — so on a
    planar file they slice along width and return silent garbage rather than
    failing. One SR slide in the UC cohort is stored this way.
    """

    @staticmethod
    def _pair(tmp_path):
        import tifffile
        img = np.zeros((40, 60, 3), np.uint8)
        img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30
        tifffile.imwrite(tmp_path / "hwc.tif", img)
        tifffile.imwrite(tmp_path / "chw.tif", np.moveaxis(img, -1, 0),
                         planarconfig="separate")
        return tmp_path / "hwc.tif", tmp_path / "chw.tif"

    def test_load_rgb_agrees_across_layouts(self, tmp_path):
        from uncertainty_phi.ensemble import load_rgb
        hwc, chw = self._pair(tmp_path)
        a, b = load_rgb(hwc), load_rgb(chw)
        assert a.shape == b.shape == (40, 60, 3)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a.reshape(-1, 3).mean(0), [10, 20, 30])

    def test_load_label_mask_agrees_across_layouts(self, tmp_path):
        from uncertainty_phi.ensemble import load_label_mask
        hwc, chw = self._pair(tmp_path)
        a, b = load_label_mask(hwc), load_label_mask(chw)
        assert a.shape == b.shape == (40, 60)
        np.testing.assert_array_equal(a, b)

    def test_a_genuinely_small_three_row_image_is_left_alone(self):
        """The shape heuristic only applies without axis metadata, and a channel
        axis is the shortest by orders of magnitude on a slide."""
        from uncertainty_phi.ensemble import _to_channels_last
        arr = np.zeros((3, 4, 5), np.uint8)
        # axes say channels-last already -> untouched
        np.testing.assert_array_equal(_to_channels_last(arr, "YXS").shape, (3, 4, 5))
        # axes say channels-first -> moved
        assert _to_channels_last(arr, "SYX").shape == (4, 5, 3)


class TestFoldIndependentGrid:
    """Every fold of a crossed grid must be scored on the SAME regions.

    The bug this closes: `phi_over_ensemble` filtered the grid by tissue coverage
    read off *the first member of whichever ensemble it was given*. A member's
    collagen mask is a model output, so a region sitting near
    `min_tissue_fraction` was kept by one fold and dropped by another, and
    `compute_phi_uncertainty.py` rejected the run for a fold-count mismatch —
    naming the wrong cause, since --tiles_metadata and --region_mm were shared.

    The H&E footprint is a property of the slide, so filtering on it is identical
    on every fold. It is also already the denominator for every density, so this
    is one definition of tissue rather than a second.
    """

    H = W = 512
    REGION = 256

    @staticmethod
    def _metadata(root: Path) -> Path:
        import pandas as pd
        rows = [{"source_file": "case1.tif", "x": x, "y": y, "tile_size": 256}
                for y in (0, 256) for x in (0, 256)]
        d = root / "tiles" / "001"
        d.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / "tiles_metadata.csv", index=False)
        return root / "tiles"

    @classmethod
    def _fold(cls, root: Path, name: str, tissue_rows: int) -> Path:
        """A two-member fold whose top-right region holds `tissue_rows` of tissue."""
        from utils import write_label_mask
        for m in (1, 2):
            d = root / name / f"model_{m:02d}"
            d.mkdir(parents=True)
            lab = np.zeros((cls.H, cls.W), np.uint8)
            lab[:, :256] = 1
            lab[:tissue_rows, 256:] = 1
            lab[::7, ::7] = np.where(lab[::7, ::7] > 0, 2, 0)
            write_label_mask(d / "case1.tif", lab)
        return root / name

    @classmethod
    def _he_masks(cls, root: Path) -> Path:
        from utils import write_label_mask
        he = np.zeros((cls.H, cls.W), np.uint8)
        he[:, :256] = 1
        he[:80, 256:] = 1               # top-right region: 31% -> kept at 0.25
        d = root / "he"
        d.mkdir()
        write_label_mask(d / "case1.tif", he)
        return d

    def _counts(self, tmp_path, he_masks):
        from uncertainty_phi.ensemble import phi_over_ensemble
        meta = self._metadata(tmp_path)
        # 31% vs 23% of the top-right region: the two straddle 0.25
        folds = [self._fold(tmp_path, "foldA", 80),
                 self._fold(tmp_path, "foldB", 60)]
        out = []
        for f in folds:
            _, regions, _, _, _ = phi_over_ensemble(
                f, meta, he_masks_dir=he_masks, region_px=self.REGION,
                mpp=0.221, min_tissue_fraction=0.25,
            )
            out.append(len(regions))
        return out

    def test_folds_agree_when_filtered_on_the_he_footprint(self, tmp_path):
        a, b = self._counts(tmp_path, self._he_masks(tmp_path))
        assert a == b == 3

    def test_folds_disagree_when_filtered_on_a_member_mask(self, tmp_path):
        """Pins the failure mode itself, so the fix cannot be silently undone by
        restoring the member-mask reference."""
        a, b = self._counts(tmp_path, None)
        assert a != b
