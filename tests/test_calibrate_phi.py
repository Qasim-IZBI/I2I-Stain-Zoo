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


class TestVarianceComponents:
    """Total, procedural and data-exposure sigma are all scored, against the SAME
    error.

    In `grand` mode the prediction is the mean of all fifty whichever spread is
    paired with it, so the error is identical across the three and only sigma
    moves. That is what makes them comparable on one axis, and it is the
    comparison the crossed 5x10 grid exists to support: a flat seed-only ensemble
    has no data-exposure term to compare against.
    """

    @staticmethod
    def _frames(n_wsi=6, per=9, seed=0):
        rng = np.random.default_rng(seed)
        phi, ref = [], []
        for w in range(n_wsi):
            for i in range(per):
                sp, sd = abs(rng.normal(6, 2)), abs(rng.normal(8, 3))
                phi.append({"wsi": f"w{w}", "region_index": i,
                            "mu_task_specific_value": 100.0,
                            "sd_total_task_specific_value": float(np.hypot(sp, sd)),
                            "sd_procedural_task_specific_value": sp,
                            "sd_data_task_specific_value": sd})
                ref.append({"wsi": f"w{w}", "region_index": i,
                            "real_task_specific_value": 100.0 + rng.normal(0, sd)})
        return pd.DataFrame(phi), pd.DataFrame(ref)

    def test_all_three_components_are_scored(self):
        from calibrate_phi import pair, score
        phi, ref = self._frames()
        rows = score(pair(phi, ref, "grand", 5), 4)
        assert {r["component"] for r in rows} == {"total", "procedural",
                                                  "data_exposure"}

    def test_the_error_is_identical_across_components(self):
        """The whole basis of the comparison. If the prediction changed with the
        spread, a difference in rho would confound the two."""
        from calibrate_phi import pair
        phi, ref = self._frames()
        t = pair(phi, ref, "grand", 5)
        by = {c: g.sort_values("region_index")["error"].to_numpy()
              for c, g in t.groupby("component")}
        np.testing.assert_allclose(by["total"], by["procedural"])
        np.testing.assert_allclose(by["total"], by["data_exposure"])

    def test_sigma_differs_across_components(self):
        from calibrate_phi import pair
        phi, ref = self._frames()
        t = pair(phi, ref, "grand", 5)
        by = {c: g.sort_values("region_index")["sd"].to_numpy()
              for c, g in t.groupby("component")}
        assert not np.allclose(by["total"], by["procedural"])
        # total is the quadrature sum of the two, so it exceeds each
        assert (by["total"] >= by["procedural"] - 1e-9).all()
        assert (by["total"] >= by["data_exposure"] - 1e-9).all()

    def test_a_missing_component_is_counted_not_hidden(self):
        """A negative ANOVA variance component is a real outcome near zero and
        has no SD, so its column is empty. Those regions drop out of that
        component only, and how many did is reported — a component estimated as
        zero on half the regions is a finding, not a missing measurement."""
        from calibrate_phi import pair, score
        phi, ref = self._frames()
        phi.loc[:9, "sd_data_task_specific_value"] = np.nan
        rows = score(pair(phi, ref, "grand", 5), 4)
        data = next(r for r in rows if r["component"] == "data_exposure")
        total = next(r for r in rows if r["component"] == "total")
        assert data["n_dropped"] == 10
        assert data["n"] == total["n"] - 10

    def test_rows_are_descriptor_major(self):
        """pair() emits component-major; the table has to read descriptor-major
        or the one comparison this exists for is scattered down the page."""
        from calibrate_phi import pair, score
        phi, ref = self._frames()
        phi["mu_beta0_per_mm2"] = 500.0
        phi["sd_total_beta0_per_mm2"] = 30.0
        phi["sd_procedural_beta0_per_mm2"] = 24.0
        phi["sd_data_beta0_per_mm2"] = 18.0
        ref["real_beta0_per_mm2"] = 505.0
        rows = score(pair(phi, ref, "grand", 5), 4)
        names = [r["descriptor"] for r in rows]
        assert names == sorted(names, key=names.index)      # contiguous blocks
        assert [r["component"] for r in rows[:3]] == ["total", "procedural",
                                                      "data_exposure"]


class TestPerFoldScoring:
    """Each subset is scored on its own, never pooled into one rho.

    Pooling enters every region five times against ONE shared target, so it adds
    no evidence — and worse, it can manufacture signal. Subsets sit at different
    sigma AND different error levels, so pooling induces a between-subset trend
    that exists inside no subset. On the real liver run the pooled beta0 rho was
    +0.312 while the five subsets gave +0.015, -0.017, +0.109, +0.123, +0.091 —
    larger than any of them.
    """

    @staticmethod
    def _two_folds_no_within_correlation(n_wsi=8, per=25, seed=0):
        """Zero correlation inside each fold, different levels between them."""
        rng = np.random.default_rng(seed)
        rows = []
        for fi, (sd_level, err_level) in enumerate([(1.0, 1.0), (3.0, 3.0)], start=1):
            for w in range(n_wsi):
                sd = np.abs(rng.normal(sd_level, 0.15, per))
                err = np.abs(rng.normal(err_level, 0.15, per))   # independent
                rows.append(pd.DataFrame({
                    "descriptor": "task_specific_value",
                    "component": "procedural_within_fold",
                    "prediction": f"fold{fi}", "wsi": f"w{w}",
                    "region_index": range(per), "sd": sd, "error": err,
                    "z": err / sd}))
        return pd.concat(rows, ignore_index=True)

    def test_each_fold_is_scored_separately(self):
        from calibrate_phi import score
        rows = score(self._two_folds_no_within_correlation(), 5)
        assert {r["prediction"] for r in rows} == {"fold1", "fold2"}
        for r in rows:
            assert r["n"] == 8 * 25          # not 2 x that

    def test_pooling_would_manufacture_a_correlation(self):
        """The reason per-fold is the default. Within each fold sigma and error
        are independent; pooled they correlate strongly, purely from the level
        difference between folds."""
        from calibrate_phi import score
        t = self._two_folds_no_within_correlation()
        per_fold = [r["spearman_rho"] for r in score(t, 5)]
        pooled = score(t.drop(columns=["prediction"]), 5)[0]["spearman_rho"]
        assert all(abs(r) < 0.15 for r in per_fold)   # nothing inside a fold
        assert pooled > 0.6                            # everything between them
        # the shape of the artefact, not a threshold: pooling exceeds every
        # subset it pools, which is what happened to beta0 on the real run
        assert pooled > 4 * max(abs(r) for r in per_fold)

    def test_agreement_flags_a_sign_flip(self):
        from calibrate_phi import fold_agreement, score
        rows = score(self._two_folds_no_within_correlation(seed=3), 5)
        a = fold_agreement(rows)
        assert len(a) == 1 and a[0]["n_folds"] == 2
        assert a[0]["rho_min"] <= a[0]["rho_median"] <= a[0]["rho_max"]
        signs = {np.sign(r["spearman_rho"]) for r in rows}
        assert a[0]["consistent_sign"] == (len(signs) <= 1)

    def test_grand_mode_is_unaffected(self):
        """`prediction` is constant there, so adding it to the keys changes
        nothing — the grand result must not move."""
        from calibrate_phi import score
        rng = np.random.default_rng(1)
        sd = np.abs(rng.normal(2, 0.5, 200))
        t = pd.DataFrame({"descriptor": "task_specific_value", "component": "total",
                          "prediction": "grand", "wsi": np.repeat(np.arange(10), 20),
                          "region_index": np.tile(np.arange(20), 10),
                          "sd": sd, "error": sd * np.abs(rng.normal(0, 1, 200))})
        t["z"] = t["error"] / t["sd"]
        with_pred = score(t, 5)
        without = score(t.drop(columns=["prediction"]), 5)
        assert len(with_pred) == len(without) == 1
        assert with_pred[0]["spearman_rho"] == pytest.approx(without[0]["spearman_rho"])


class TestRiskCoverage:
    """Selective prediction: keep the most certain regions, measure what remains.

    The question rho invites and does not answer. rho = 0.22 is real but a reader
    is entitled to ask what it buys; this answers in the units of the task.
    """

    @staticmethod
    def _table(sd, err, n_wsi=6):
        n = len(sd)
        return pd.DataFrame({
            "descriptor": "task_specific_value", "component": "total",
            "prediction": "grand",
            "wsi": [f"w{i % n_wsi}" for i in range(n)],
            "region_index": range(n), "sd": sd, "error": err,
            "z": np.asarray(err) / np.asarray(sd)})

    def test_full_coverage_is_a_no_op(self):
        from calibrate_phi import risk_coverage
        rng = np.random.default_rng(0)
        sd = np.abs(rng.normal(2, 0.5, 300))
        r = risk_coverage(self._table(sd, sd * np.abs(rng.normal(0, 1, 300))),
                          [1.0], 0, 0)[0]
        assert r["rel_change"] == pytest.approx(0.0, abs=1e-9)
        assert r["mae"] == pytest.approx(r["mae_random"])
        # 0/0 renders as a confident 100% in floating point — the one number
        # here a reader must not misread
        assert np.isnan(r["capture_of_oracle"])

    def test_a_perfect_uncertainty_matches_the_oracle(self):
        from calibrate_phi import risk_coverage
        err = np.linspace(0.1, 5.0, 300)
        r = risk_coverage(self._table(err.copy(), err), [0.8], 0, 0)[0]
        assert r["rel_change"] == pytest.approx(r["rel_change_oracle"])
        assert r["capture_of_oracle"] == pytest.approx(1.0)

    def test_a_useless_uncertainty_buys_nothing(self):
        from calibrate_phi import risk_coverage
        rng = np.random.default_rng(1)
        err = np.abs(rng.normal(1, 0.3, 2000))
        sd = rng.permutation(err)              # independent of the error
        r = risk_coverage(self._table(sd, err), [0.8], 0, 0)[0]
        assert abs(r["rel_change"]) < 0.05
        assert r["rel_change_oracle"] < -0.1   # the ceiling is still there

    def test_the_oracle_is_never_beaten(self):
        """Ranking by true error is the ceiling by construction, so no sigma can
        do better. A violation would mean the selection is not what it claims."""
        from calibrate_phi import risk_coverage
        rng = np.random.default_rng(2)
        sd = np.abs(rng.normal(2, 0.6, 500))
        err = sd * np.abs(rng.normal(0, 1, 500))
        for r in risk_coverage(self._table(sd, err), [0.9, 0.8, 0.5], 0, 0):
            assert r["rel_change"] >= r["rel_change_oracle"] - 1e-12

    def test_random_selection_is_the_overall_mean(self):
        """Not simulated: dropping a random subset changes nothing in
        expectation, so the curve's departure from it IS the effect."""
        from calibrate_phi import risk_coverage
        rng = np.random.default_rng(3)
        sd = np.abs(rng.normal(2, 0.5, 400))
        err = sd * np.abs(rng.normal(0, 1, 400))
        for r in risk_coverage(self._table(sd, err), [0.9, 0.5], 0, 0):
            assert r["mae_random"] == pytest.approx(err.mean())

    def test_bootstrap_resamples_slides(self):
        from calibrate_phi import risk_coverage
        rng = np.random.default_rng(4)
        sd = np.abs(rng.normal(2, 0.6, 600))
        err = sd * np.abs(rng.normal(0, 1, 600))
        r = risk_coverage(self._table(sd, err, n_wsi=8), [0.8], 500, 0)[0]
        assert r["rel_ci_lo"] <= r["rel_change"] <= r["rel_ci_hi"]


class TestWithinSlide:
    """Per-slide rho, and rho with the point prediction partialled out.

    Two problems a pooled correlation has. The unit of replication is the slide,
    not the region — 2850 regions come from twenty cases. And sigma tracks how
    much structure a region holds while absolute error does too, so a raw rho is
    largely the two sharing that dependence. On the UC liver cohort
    rho(sigma, mu_CPA) = +0.76, and partialling drops rho from +0.28 to +0.15.
    """

    @staticmethod
    def _table(sd, err, mu, wsi):
        return pd.DataFrame({
            "descriptor": "task_specific_value", "component": "total",
            "prediction": "grand", "wsi": wsi,
            "region_index": np.arange(len(sd)), "sd": sd, "error": err, "mu": mu,
            "z": np.asarray(err) / np.asarray(sd)})

    def test_the_slide_is_the_unit(self):
        from calibrate_phi import within_slide
        rng = np.random.default_rng(0)
        n_wsi, per = 9, 40
        wsi = np.repeat([f"w{i}" for i in range(n_wsi)], per)
        sd = np.abs(rng.normal(2, 0.5, n_wsi * per))
        err = sd * np.abs(rng.normal(0, 1, n_wsi * per))
        r = within_slide(self._table(sd, err, sd.copy(), wsi), 15, 500, 0)[0]
        assert r["n_slides"] == n_wsi
        assert len(r["per_slide_raw"]) == n_wsi
        assert 0 <= r["n_positive_raw"] <= n_wsi

    def test_short_slides_are_dropped(self):
        """A rho over a handful of regions is noise, and the per-slide mean would
        weight it equally with a well-estimated one."""
        from calibrate_phi import within_slide
        rng = np.random.default_rng(1)
        # three long slides, so the >=3 slides the bootstrap needs survive, plus
        # one short slide that must not
        wsi = np.array(["big"] * 60 + ["small"] * 5 + ["ok"] * 30 + ["also"] * 25)
        sd = np.abs(rng.normal(2, 0.5, len(wsi)))
        err = sd * np.abs(rng.normal(0, 1, len(wsi)))
        r = within_slide(self._table(sd, err, sd.copy(), wsi), 15, 500, 0)[0]
        assert r["n_slides"] == 3
        assert r["regions_per_slide_min"] == 25

    def test_too_few_slides_yields_nothing(self):
        """Under three slides there is no bootstrap over cases to speak of, so
        the analysis is withheld rather than reported from two numbers."""
        from calibrate_phi import within_slide
        rng = np.random.default_rng(9)
        wsi = np.repeat(["a", "b"], 40)
        sd = np.abs(rng.normal(2, 0.5, 80))
        err = sd * np.abs(rng.normal(0, 1, 80))
        assert within_slide(self._table(sd, err, sd.copy(), wsi), 15, 200, 0) == []

    def test_a_pure_level_confound_is_removed(self):
        """The failure the partial exists to catch: sigma and error both driven
        by the region's level and by nothing else. The raw rho is large and means
        nothing; the partial must collapse."""
        from calibrate_phi import within_slide
        rng = np.random.default_rng(2)
        n_wsi, per = 8, 60
        wsi = np.repeat([f"w{i}" for i in range(n_wsi)], per)
        level = np.abs(rng.normal(1.0, 0.35, n_wsi * per))
        sd = level * 0.5 * np.exp(rng.normal(0, 0.05, n_wsi * per))
        err = level * 0.4 * np.exp(rng.normal(0, 0.05, n_wsi * per))
        r = within_slide(self._table(sd, err, level, wsi), 15, 1000, 0)[0]
        assert r["rho_raw_mean"] > 0.7
        assert abs(r["rho_partial_mu_mean"]) < 0.2

    def test_genuine_signal_survives_the_partial(self):
        """The complement: sigma predicts error for reasons unrelated to the
        level, so partialling must leave it standing."""
        from calibrate_phi import within_slide
        rng = np.random.default_rng(3)
        n_wsi, per = 8, 60
        wsi = np.repeat([f"w{i}" for i in range(n_wsi)], per)
        level = np.abs(rng.normal(1.0, 0.3, n_wsi * per))
        sd = np.abs(rng.normal(2, 0.7, n_wsi * per))     # independent of level
        err = sd * np.abs(rng.normal(0, 1, n_wsi * per))
        r = within_slide(self._table(sd, err, level, wsi), 15, 1000, 0)[0]
        assert r["rho_partial_mu_mean"] > 0.3
        assert r["rho_partial_mu_ci_lo"] > 0

    def test_partial_uses_mu_not_the_reference(self):
        """mu is available at inference; `real` is not. Partialling on the
        reference would remove the signal being tested, so the function must not
        even accept it — it reads `mu` by name."""
        from calibrate_phi import within_slide
        rng = np.random.default_rng(4)
        wsi = np.repeat([f"w{i}" for i in range(6)], 40)
        sd = np.abs(rng.normal(2, 0.5, len(wsi)))
        err = sd * np.abs(rng.normal(0, 1, len(wsi)))
        t = self._table(sd, err, sd.copy(), wsi)
        assert within_slide(t.drop(columns=["mu"]), 15, 200, 0) == []

    def test_partial_spearman_matches_a_direct_computation(self):
        from calibrate_phi import _partial_spearman
        from scipy.stats import rankdata, spearmanr
        rng = np.random.default_rng(5)
        z = rng.normal(size=300)
        a, b = z + rng.normal(size=300), z + rng.normal(size=300)
        got = _partial_spearman(a, b, z)
        ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
        A = np.c_[np.ones_like(rz, dtype=float), rz]
        want = spearmanr(ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0],
                         rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]).statistic
        assert got == pytest.approx(want)
        assert got < spearmanr(a, b).statistic      # z drove part of it


class TestPhiRunProvenance:
    """The calibration output must say how its regions were defined.

    --roi_dir, --region_px and --min_tissue_fraction belong to the phi run, not
    to this one: the grid arrives already built. So without copying them forward
    a result cannot answer "was the kidney arm restricted to cortex?" — which is
    unanswerable from the numbers and decides whether the grid mixed two
    populations.
    """

    @staticmethod
    def _run(tmp_path, params):
        import json
        (tmp_path / "per_region.csv").write_text("wsi\n")
        if params is not None:
            with open(tmp_path / "summary.json", "w") as fh:
                json.dump({"params": params}, fh)
        from calibrate_phi import phi_run_provenance
        return phi_run_provenance(tmp_path / "per_region.csv")

    def test_roi_is_recorded(self, tmp_path):
        got = self._run(tmp_path, {"region_px": 2048, "roi_dir": "/p/cortex",
                                   "min_roi_fraction": 0.5,
                                   "min_tissue_fraction": 0.25})
        assert got["roi_dir"] == "/p/cortex"
        assert got["region_px"] == 2048
        assert got["summary"].endswith("summary.json")

    def test_absence_of_roi_is_also_recorded(self, tmp_path, capsys):
        """'no ROI key' and 'ROI not applied' must not be the same evidence as
        'never looked'."""
        self._run(tmp_path, {"region_px": 2048, "min_tissue_fraction": 0.25})
        assert "ROI=none" in capsys.readouterr().out

    def test_a_missing_summary_says_so(self, tmp_path, capsys):
        assert self._run(tmp_path, None) == {}
        assert "cannot be recorded" in capsys.readouterr().out

    def test_a_corrupt_summary_does_not_kill_the_run(self, tmp_path, capsys):
        (tmp_path / "per_region.csv").write_text("wsi\n")
        (tmp_path / "summary.json").write_text("{not json")
        from calibrate_phi import phi_run_provenance
        assert phi_run_provenance(tmp_path / "per_region.csv") == {}
        assert "could not read" in capsys.readouterr().out


class TestRegenComparison:
    """Cycle error and ensemble spread, scored on the same regions.

    The paper's central contrast is currently a citation beside a measurement.
    This puts both on one footing: same regions, same target, same
    slide-clustered statistics, same partial on the point prediction — so a null
    for either cannot be blamed on the protocol.
    """

    R, T = 2048, 512

    def _cohort(self, tmp_path, n_wsi=3, grid=2):
        """A tiled cohort with a regen error map per tile."""
        import json
        meta = tmp_path / "tiles"
        regen = tmp_path / "regen" / "m01"
        regen.mkdir(parents=True)
        rng = np.random.default_rng(0)
        rows = []
        for w in range(n_wsi):
            folder = f"{w + 1:03d}"
            (meta / folder).mkdir(parents=True)
            ed = regen / f"wsi{folder}" / "error_npy"
            ed.mkdir(parents=True)
            trows, tid = [], 0
            for ry in range(grid):
                for rx in range(grid):
                    for ty in range(4):
                        for tx in range(4):
                            tid += 1
                            name = f"{tid:07d}"      # zero-padded, as tile.py writes
                            trows.append({
                                "tile_name": name, "source_file": f"HE_s{w}.tif",
                                "x": rx * self.R + tx * self.T,
                                "y": ry * self.R + ty * self.T,
                                "tile_size": self.T, "saved_size": 256})
                            np.save(ed / f"{name}.npy",
                                    np.full((8, 8), float(ry * grid + rx),
                                            np.float32))
                    rows.append({"wsi": f"HE_s{w}.tif",
                                 "region_index": ry * grid + rx,
                                 "y0": ry * self.R, "y1": ry * self.R + self.R,
                                 "x0": rx * self.R, "x1": rx * self.R + self.R})
            pd.DataFrame(trows).to_csv(meta / folder / "tiles_metadata.csv",
                                       index=False)
        return meta, regen, pd.DataFrame(rows)

    def test_tiles_nest_exactly_inside_regions(self, tmp_path):
        """A 2048 px region holds sixteen 512 px tiles and no tile straddles a
        boundary — the property that makes the aggregation exact rather than an
        approximation."""
        from compare_uncertainty_sources import (assign_tiles_to_regions,
                                                 tile_table)
        meta, _, regions = self._cohort(tmp_path)
        pairs = assign_tiles_to_regions(tile_table(meta), regions)
        counts = pairs.groupby(["wsi", "region_index"]).size()
        assert set(counts) == {16}
        # every tile used exactly once: no double-counting, none dropped
        assert len(pairs) == len(pairs.drop_duplicates(["folder", "tile_name"]))

    def test_zero_padded_tile_names_survive_the_csv(self, tmp_path):
        """The bug this closes: pandas reads '0000001' as int 1, the lookup asks
        for 1.npy, every tile 'has no error map', and the run dies claiming the
        directory layout is wrong when only the padding was lost."""
        from compare_uncertainty_sources import tile_table
        meta, _, _ = self._cohort(tmp_path)
        names = tile_table(meta)["tile_name"]
        assert names.map(lambda s: isinstance(s, str)).all()
        assert "0000001" in set(names)

    def test_region_error_is_the_mean_of_its_tiles(self, tmp_path):
        """Each region's tiles were written with its own constant value, so the
        aggregate must come back as exactly that value."""
        from compare_uncertainty_sources import regen_per_region, tile_table
        meta, regen, regions = self._cohort(tmp_path, grid=2)
        got = regen_per_region(regions, tile_table(meta), [regen], None,
                               cache=tmp_path / "cache.csv")
        for r in got.itertuples():
            assert r.regen == pytest.approx(float(r.region_index))

    def test_members_are_averaged(self, tmp_path):
        from compare_uncertainty_sources import regen_per_region, tile_table
        meta, r1, regions = self._cohort(tmp_path, grid=2)
        r2 = tmp_path / "regen" / "m02"
        for src in sorted(r1.rglob("*.npy")):
            dst = r2 / src.relative_to(r1)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, np.load(src) + 10.0)
        got = regen_per_region(regions, tile_table(meta), [r1, r2], None,
                               cache=tmp_path / "c2.csv")
        for r in got.itertuples():
            assert r.regen == pytest.approx(float(r.region_index) + 5.0)

    def test_the_tile_cache_round_trips(self, tmp_path):
        """The tile pass is the slow half; a reuse that lost the padding would
        silently recompute or mismatch."""
        from compare_uncertainty_sources import regen_per_region, tile_table
        meta, regen, regions = self._cohort(tmp_path, grid=2)
        tiles, cache = tile_table(meta), tmp_path / "cache.csv"
        first = regen_per_region(regions, tiles, [regen], None, cache=cache)
        assert cache.is_file()
        # point at an empty root: everything must now come from the cache
        empty = tmp_path / "nothing"
        empty.mkdir()
        second = regen_per_region(regions, tiles, [empty], None, cache=cache)
        pd.testing.assert_frame_equal(first, second)
