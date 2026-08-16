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
