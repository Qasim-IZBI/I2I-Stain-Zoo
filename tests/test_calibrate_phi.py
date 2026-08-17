"""Calibration of phi_struct uncertainty against real tissue.

The claim under test is that ensemble spread predicts structural error. These
pin the two things that would silently invalidate it: a scorer that cannot tell
a calibrated ensemble from an uninformative one, and a reference cropped from a
different coordinate frame.
"""

import numpy as np
import pandas as pd
import pytest

from calibrate_phi import HALF_NORMAL, pair, score


def _synthetic(n=4000, seed=0):
    """Three descriptors with known calibration injected."""
    rng = np.random.default_rng(seed)
    frames = []
    for label, factor in (("calibrated", 1.0), ("overconfident", 2.5),
                          ("uninformative", None)):
        sd = rng.uniform(0.01, 0.10, n)
        err = (np.abs(rng.normal(0, 0.05, n)) if factor is None
               else np.abs(rng.normal(0, sd * factor)))
        frames.append(pd.DataFrame({"descriptor": label, "sd": sd, "error": err,
                                    "z": err / sd, "wsi": "w", "region_index": 0}))
    return pd.concat(frames, ignore_index=True)


class TestScore:
    def test_recovers_an_injected_calibration(self):
        by = {r["descriptor"]: r for r in score(_synthetic(), 10)}
        assert by["calibrated"]["calibration_ratio"] == pytest.approx(1.0, abs=0.08)
        assert by["overconfident"]["calibration_ratio"] == pytest.approx(2.5, abs=0.15)

    def test_rho_separates_informative_from_useless(self):
        by = {r["descriptor"]: r for r in score(_synthetic(), 10)}
        assert by["calibrated"]["spearman_rho"] > 0.4
        assert abs(by["uninformative"]["spearman_rho"]) < 0.05

    def test_rho_is_scale_free_but_the_ratio_is_not(self):
        """They answer different questions: ranking versus scale. The same data
        with every error multiplied by 2.5 ranks identically and reads 2.5x
        more over-confident."""
        base = _synthetic()
        base = base[base["descriptor"] == "calibrated"].copy()
        scaled = base.copy()
        scaled["error"] *= 2.5
        scaled["z"] *= 2.5

        a = score(base, 10)[0]
        b = score(scaled, 10)[0]
        assert a["spearman_rho"] == pytest.approx(b["spearman_rho"], abs=1e-12)
        assert b["calibration_ratio"] == pytest.approx(
            2.5 * a["calibration_ratio"], rel=1e-9)

    def test_calibrated_means_mean_abs_z_is_the_half_normal_constant(self):
        by = {r["descriptor"]: r for r in score(_synthetic(), 10)}
        assert by["calibrated"]["mean_abs_z"] == pytest.approx(HALF_NORMAL, abs=0.07)

    def test_degenerate_descriptor_is_reported_not_scored(self):
        t = pd.DataFrame({"descriptor": ["d"] * 5, "sd": [0.0] * 5,
                          "error": [0.1] * 5, "z": [np.nan] * 5,
                          "wsi": "w", "region_index": range(5)})
        r = score(t, 4)[0]
        assert "spearman_rho" not in r
        assert "too few" in r["note"]


class TestPair:
    @staticmethod
    def _virtual():
        return pd.DataFrame({
            "wsi": ["a"] * 3, "region_index": [0, 1, 2],
            "mu_task_specific_value": [0.10, 0.20, 0.30],
            "sd_total_task_specific_value": [0.01, 0.02, 0.03],
            "fold1_mu_task_specific_value": [0.11, 0.21, 0.31],
            "fold1_sd_task_specific_value": [0.005, 0.010, 0.015],
        })

    @staticmethod
    def _ref():
        return pd.DataFrame({"wsi": ["a"] * 3, "region_index": [0, 1, 2],
                             "real_task_specific_value": [0.12, 0.19, 0.34]})

    def test_grand_pairs_the_mean_with_the_total_spread(self):
        t = pair(self._virtual(), self._ref(), "grand", 5)
        assert set(t["prediction"]) == {"grand"}
        np.testing.assert_allclose(t["error"], [0.02, 0.01, 0.04], atol=1e-12)
        np.testing.assert_allclose(t["z"], [2.0, 0.5, 4 / 3], atol=1e-9)

    def test_fold_mode_uses_the_subset_prediction(self):
        t = pair(self._virtual(), self._ref(), "fold", 1)
        assert set(t["prediction"]) == {"fold1"}
        np.testing.assert_allclose(t["error"], [0.01, 0.02, 0.03], atol=1e-12)

    def test_descriptors_without_a_reference_are_dropped_not_faked(self):
        t = pair(self._virtual(), self._ref(), "grand", 5)
        assert set(t["descriptor"]) == {"task_specific_value"}

    def test_no_overlap_is_an_error(self):
        ref = self._ref().assign(region_index=[7, 8, 9])
        with pytest.raises(SystemExit, match="no regions matched"):
            pair(self._virtual(), ref, "grand", 5)


class TestHeatmapRaster:
    """Painting per-region values back onto slide geometry."""

    @staticmethod
    def _group():
        import pandas as pd
        # two regions of a 2x2 grid; the other two were dropped by the filter
        return pd.DataFrame({
            "y0": [0, 100], "y1": [100, 200],
            "x0": [0, 100], "x1": [100, 200],
            "sd_total_task_specific_value": [0.4, 0.8],
        })

    def test_each_region_paints_its_own_block(self):
        from plot_uncertainty_heatmap import raster_for
        r = raster_for(self._group(), "sd_total_task_specific_value", 10, (20, 20))
        assert r[0, 0] == pytest.approx(0.4)
        assert r[15, 15] == pytest.approx(0.8)

    def test_regions_the_filter_dropped_stay_blank(self):
        """An absent measurement is not a low one — it must not read as dark."""
        from plot_uncertainty_heatmap import raster_for
        r = raster_for(self._group(), "sd_total_task_specific_value", 10, (20, 20))
        assert np.isnan(r[0, 15])      # top-right region was never measured
        assert np.isnan(r[15, 0])

    def test_a_missing_column_gives_an_empty_raster_not_a_crash(self):
        from plot_uncertainty_heatmap import raster_for
        r = raster_for(self._group(), "sd_total_absent", 10, (20, 20))
        assert np.isnan(r).all()


class TestStemMatching:
    """The real PSR masks are named after the SR slides; φ is gridded on the H&E.

    So `SR_d31_BDL+A_M2` has to reach `HE_d31_BDL+A_M2` — the rule
    `apply_he_mask.py` and `compare_psr.py` already carry, and the reason a
    calibration run otherwise skips all twenty slides and exits with "no
    reference regions produced".
    """

    NAMES = ("d31_BDL+A_M2", "w10_BDL+A_M3")

    @staticmethod
    def _dir(tmp_path, sub, prefix, names):
        from utils import write_label_mask
        d = tmp_path / sub
        d.mkdir()
        for n in names:
            write_label_mask(d / f"{prefix}_{n}.tif", np.ones((8, 8), np.uint8))
        return d

    def test_strip_prefix_bridges_sr_to_he(self, tmp_path):
        from calibrate_phi import _indexed
        d = self._dir(tmp_path, "psr", "SR", self.NAMES)
        assert set(_indexed(d, False, "x")) == {f"SR_{n}" for n in self.NAMES}
        assert set(_indexed(d, True, "x")) == set(self.NAMES)

    def test_he_still_matches_itself_when_stripping(self, tmp_path):
        """Both sides are keyed the same way, so turning it on must not break the
        arm that already matched."""
        from calibrate_phi import _indexed
        from apply_he_mask import normalize_stem
        d = self._dir(tmp_path, "he", "HE", self.NAMES)
        keys = set(_indexed(d, True, "x"))
        assert keys == {normalize_stem(f"HE_{n}", True) for n in self.NAMES}
        assert keys == set(self.NAMES)

    def test_a_collision_is_fatal(self, tmp_path):
        """SR_x and HE_x in one directory collapse to 'x'. Picking either scores
        one slide's regions against another slide's tissue, which is invisible in
        the output — so it is refused."""
        from calibrate_phi import _indexed
        d = tmp_path / "mixed"
        d.mkdir()
        from utils import write_label_mask
        for p in ("SR", "HE"):
            write_label_mask(d / f"{p}_slide.tif", np.ones((8, 8), np.uint8))
        with pytest.raises(SystemExit, match="collapses two files"):
            _indexed(d, True, "--real_psr")


class TestUndefinedRho:
    """rho is undefined when either side has no spread, and which side it was
    means the opposite thing — a constant sigma is an ensemble that agrees
    everywhere, a constant error is a reference that cannot tell regions apart.
    """

    @staticmethod
    def _table(sd, err):
        return pd.DataFrame({
            "descriptor": ["task_specific_value"] * len(sd),
            "wsi": ["a.tif"] * len(sd),
            "region_index": range(len(sd)),
            "sd": sd, "error": err, "z": np.asarray(err) / np.asarray(sd),
        })

    def test_constant_sigma_is_named(self):
        from calibrate_phi import score
        r = score(self._table([2.0] * 12, np.linspace(1, 5, 12)), 4)[0]
        assert not np.isfinite(r["spearman_rho"])
        assert r["undefined_because"] == "σ constant"

    def test_constant_error_is_named(self):
        from calibrate_phi import score
        r = score(self._table(np.linspace(1, 5, 12), [2.0] * 12), 4)[0]
        assert not np.isfinite(r["spearman_rho"])
        assert r["undefined_because"] == "error constant"

    def test_a_real_rho_carries_no_note(self):
        from calibrate_phi import score
        sd = np.linspace(1, 5, 40)
        r = score(self._table(sd, sd * 0.8), 4)[0]
        assert r["spearman_rho"] > 0.9
        assert r["undefined_because"] is None


class TestFrameGuard:
    """A reference on a different frame scores different tissue under the same
    region id, so it must be refused. But one difference is benign and common:
    `utils.reconstruct_wsi` truncates to a whole number of tiles, so the phi
    frame is a PREFIX of the untruncated original at the same origin and scale,
    and the boxes index identical pixels.

    The bound separates the two. Truncation cannot lose a whole tile, so an
    excess below one tile means the reference truncates to exactly this frame.
    The UC M3 case is over by 2273x4741 px and aligns with nothing.
    """

    T, R = 512, 1024

    def _run(self, tmp_path, ref_shape, phi_shape, extra=()):
        import subprocess
        import sys
        from utils import write_label_mask

        psr = tmp_path / "psr"; psr.mkdir()
        he = tmp_path / "he"; he.mkdir()
        rng = np.random.default_rng(1)
        lab = np.ones(ref_shape, np.uint8)
        lab[rng.random(ref_shape) < 0.05] = 2
        write_label_mask(psr / "SR_slide.tif", lab)
        write_label_mask(he / "HE_slide.tif", np.ones(ref_shape, np.uint8))

        R = self.R
        rows = []
        for i, (y, x) in enumerate([(0, 0), (0, R), (R, 0), (R, R)]):
            rows.append(dict(
                wsi="HE_slide.tif", region_index=i, y0=y, y1=y + R, x0=x, x1=x + R,
                area_mm2=0.2, wsi_h=phi_shape[0], wsi_w=phi_shape[1],
                tissue_fraction=1.0,
                mu_task_specific_value=0.05 + 0.01 * i,
                sd_total_task_specific_value=0.004 + 0.003 * i,
                sd_procedural_task_specific_value=0.008,
                mu_beta0_per_mm2=500.0 + i, sd_total_beta0_per_mm2=50.0 + i,
                sd_procedural_beta0_per_mm2=40.0,
                mu_beta1_per_mm2=90.0 + i, sd_total_beta1_per_mm2=12.0 + i,
                sd_procedural_beta1_per_mm2=9.0,
                mu_regional_dispersion=0.4,
                sd_total_regional_dispersion=0.05 + 0.01 * i,
                sd_procedural_regional_dispersion=0.04))
        csv = tmp_path / "pr.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return subprocess.run(
            [sys.executable, "calibrate_phi.py", "--phi_csv", str(csv),
             "--real_psr", str(psr), "--he_masks", str(he), "--strip_prefix",
             "--outdir", str(tmp_path / "out")] + list(extra),
            capture_output=True, text=True,
        )

    def test_truncated_reconstruction_frame_is_accepted(self, tmp_path):
        """The real UC arithmetic: 24967 -> 48 whole 512px tiles -> 24576, an
        excess of 391 px; 34757 -> 67 -> 34304, an excess of 453."""
        r = self._run(tmp_path, (2048 + 391, 3072 + 453), (2048, 3072))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "[note]" in r.stdout
        assert "untruncated original" in r.stdout

    def test_a_frame_off_by_a_whole_tile_is_refused(self, tmp_path):
        r = self._run(tmp_path, (2048 + 2273, 3072 + 4741), (2048, 3072))
        assert r.returncode != 0
        assert "Different frames" in r.stdout + r.stderr

    def test_exact_match_needs_no_note(self, tmp_path):
        r = self._run(tmp_path, (2048, 3072), (2048, 3072))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "[note]" not in r.stdout

    def test_tile_size_must_match_the_tiling(self, tmp_path):
        """The slack is only benign up to one tile, so an understated --tile_size
        turns a fine run into a refusal rather than the reverse."""
        r = self._run(tmp_path, (2048 + 391, 3072 + 453), (2048, 3072),
                      extra=("--tile_size", "256"))
        assert r.returncode != 0
        assert "--tile_size must match the tiling" in r.stdout + r.stderr


class TestReliabilityBins:
    """The reliability diagram's own data is written out, not only plotted.

    A figure whose numbers exist only inside a PNG cannot be restyled for the
    manuscript, quoted in text, or checked by a reviewer.
    """

    @staticmethod
    def _table(n_wsi=8, per=30, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for w in range(n_wsi):
            sd = np.abs(rng.normal(2.0 + rng.normal(0, 0.4), 0.5, per))
            err = sd * np.abs(rng.normal(0, 1, per))
            out.append(pd.DataFrame({
                "descriptor": "task_specific_value", "wsi": f"w{w}",
                "region_index": range(per), "sd": sd, "error": err,
                "z": err / sd}))
        return pd.concat(out, ignore_index=True)

    def test_bins_carry_the_plotted_quantities(self):
        from calibrate_phi import score
        b = score(self._table(), 5)[0]["bins"]
        assert len(b) == 5
        for d in b:
            assert {"bin", "sd_lo", "sd_hi", "mean_sd", "mean_error",
                    "expected_error", "ratio_obs_over_expected",
                    "se_error_by_case", "n", "n_wsi"} <= set(d)

    def test_expected_error_is_the_calibrated_line(self):
        """0.80σ, not σ. A diagonal would call a calibrated ensemble 20%
        over-confident."""
        from calibrate_phi import HALF_NORMAL, score
        for d in score(self._table(), 5)[0]["bins"]:
            assert d["expected_error"] == pytest.approx(HALF_NORMAL * d["mean_sd"])
            assert d["ratio_obs_over_expected"] == pytest.approx(
                d["mean_error"] / d["expected_error"])

    def test_error_bars_are_clustered_on_the_case(self):
        """A plain SEM over regions would be ~sqrt(regions/cases) times tighter,
        which at these counts is the difference between a visible error bar and
        an invisible one."""
        from calibrate_phi import score
        b = score(self._table(n_wsi=8, per=30), 5)[0]["bins"]
        for d in b:
            assert d["n_wsi"] <= 8 and d["n_wsi"] > 1
            assert d["n"] > d["n_wsi"]          # many regions per case
            assert np.isfinite(d["se_error_by_case"])

    def test_a_bin_from_one_case_reports_no_error_bar(self):
        """One case gives no between-case spread to estimate. NaN is the honest
        answer; 0 would draw a point with false precision."""
        from calibrate_phi import score
        t = self._table(n_wsi=1, per=40)
        b = score(t, 4)[0]["bins"]
        assert all(np.isnan(d["se_error_by_case"]) for d in b)
        assert all(d["n_wsi"] == 1 for d in b)


class TestReferenceCache:
    """Reference φ is the expensive half and does not depend on the ensemble at
    all — only on the real masks and the region boxes. Caching it turns a re-plot
    from hours into seconds. The whole risk is staleness, so reuse is verified
    rather than trusted.
    """

    @staticmethod
    def _args(tmp_path, **over):
        import argparse
        d = dict(mpp=0.221, min_object_px=16, closing_px=0, white_thresh=0.85,
                 real_psr=tmp_path / "psr", real_lumen=None,
                 he_masks=tmp_path / "he", he_dir=None, strip_prefix=True)
        d.update(over)
        return argparse.Namespace(**d)

    @staticmethod
    def _grid(region=1024, n_wsi=2, per_side=2):
        rows = []
        for w in range(n_wsi):
            for i, (y, x) in enumerate([(a * region, b * region)
                                        for a in range(per_side)
                                        for b in range(per_side)]):
                rows.append({"wsi": f"HE_s{w}.tif", "region_index": i,
                             "y0": y, "y1": y + region,
                             "x0": x, "x1": x + region})
        return pd.DataFrame(rows)

    def _ref(self, df):
        out = df.copy()
        out["real_task_specific_value"] = np.linspace(0.02, 0.08, len(df))
        return out

    def test_round_trip(self, tmp_path):
        from calibrate_phi import load_reference, save_reference
        df = self._grid()
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(df), path, self._args(tmp_path))
        back = load_reference(path, df, self._args(tmp_path))
        assert len(back) == len(df)
        assert "real_task_specific_value" in back.columns

    def test_a_changed_parameter_is_refused(self, tmp_path):
        """Same boxes, different measurement — the cache is a different quantity
        wearing the same region ids."""
        from calibrate_phi import load_reference, save_reference
        df = self._grid()
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(df), path, self._args(tmp_path))
        with pytest.raises(SystemExit, match="different parameters"):
            load_reference(path, df, self._args(tmp_path, white_thresh=0.5))

    def test_a_regrid_is_refused_even_though_ids_match(self, tmp_path):
        """The hole a parameter check alone leaves: --region_px 2048 against a
        cache built at 1024 keeps every parameter identical, and region 7 of
        slide 3 exists in both — on different tissue."""
        from calibrate_phi import load_reference, save_reference
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(self._grid(region=1024)), path,
                       self._args(tmp_path))
        with pytest.raises(SystemExit, match="DIFFERENT boxes"):
            load_reference(path, self._grid(region=2048), self._args(tmp_path))

    def test_a_short_cache_is_refused(self, tmp_path):
        from calibrate_phi import load_reference, save_reference
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(self._grid(n_wsi=1)), path, self._args(tmp_path))
        with pytest.raises(SystemExit, match="covers 4 of the 8 regions"):
            load_reference(path, self._grid(n_wsi=2), self._args(tmp_path))

    def test_a_cache_without_boxes_is_refused(self, tmp_path):
        """Predates --save_reference, so nothing about it can be checked."""
        from calibrate_phi import load_reference
        df = self._grid()
        path = tmp_path / "old.csv"
        self._ref(df).drop(columns=["y0", "y1", "x0", "x1"]).to_csv(path, index=False)
        with pytest.raises(SystemExit, match="no y0/y1/x0/x1"):
            load_reference(path, df, self._args(tmp_path))
