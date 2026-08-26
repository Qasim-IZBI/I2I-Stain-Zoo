"""W-2: the shape factor behind the 0.80 sigma reference line.

The load-bearing tests are in `TestRecoversKnownShapes`: the estimator is only
worth running if it returns 0.798 on Gaussian residuals, 0.707 on Laplace and
0.866 on uniform. Everything else here guards a way of getting the right answer
for the wrong reason.
"""

import numpy as np
import pandas as pd
import pytest

from estimate_shape_factor import (GAUSSIAN, REFERENCE_SHAPES, bias_share,
                                   check_ordering, combine, per_slide,
                                   shape_centred, shape_uncentred, verdict)


class TestRecoversKnownShapes:
    """E|X| / sd(X) for each reference distribution, on a large sample."""

    @pytest.mark.parametrize("name,draw", [
        ("Gaussian", lambda r, n: r.normal(0, 3.0, n)),
        ("Laplace", lambda r, n: r.laplace(0, 1.7, n)),
        ("Uniform", lambda r, n: r.uniform(-2, 2, n)),
        ("Student-t, 5 df", lambda r, n: r.standard_t(5, n) * 0.8),
    ])
    def test_centred_matches_the_reference_value(self, name, draw):
        want = dict(REFERENCE_SHAPES)[name]
        rng = np.random.default_rng(0)
        # t5 has a heavy tail, so its sample kappa converges slowly; the
        # tolerance is the sampling error, not slack in the estimator.
        tol = 0.02 if name.startswith("Student") else 0.005
        assert shape_centred(draw(rng, 400_000)) == pytest.approx(want, abs=tol)

    def test_gaussian_is_the_value_the_manuscript_assumes(self):
        assert GAUSSIAN == pytest.approx(0.7979, abs=1e-4)


class TestInvariances:
    def test_scale_invariant(self):
        """A constant miscalibration must divide out — the whole point."""
        rng = np.random.default_rng(1)
        u = rng.normal(0, 1, 5000)
        assert shape_centred(u * 17.3) == pytest.approx(shape_centred(u))
        assert shape_uncentred(u * 17.3) == pytest.approx(shape_uncentred(u))

    def test_offset_moves_uncentred_only(self):
        rng = np.random.default_rng(2)
        u = rng.normal(0, 1, 20000)
        assert shape_centred(u + 5.0) == pytest.approx(shape_centred(u), abs=1e-9)
        assert shape_uncentred(u + 5.0) > shape_uncentred(u)

    def test_uncentred_tends_to_one_as_bias_dominates(self):
        rng = np.random.default_rng(3)
        u = rng.normal(0, 1, 20000)
        assert shape_uncentred(u + 200.0) == pytest.approx(1.0, abs=1e-3)

    def test_uncentred_at_least_centred_on_a_biased_sample(self):
        rng = np.random.default_rng(4)
        u = rng.normal(1.5, 1, 5000)
        assert shape_uncentred(u) >= shape_centred(u)


class TestBiasShare:
    def test_pure_offset_is_one(self):
        assert bias_share(np.full(50, -0.3)) == pytest.approx(1.0)

    def test_symmetric_scatter_is_near_zero(self):
        rng = np.random.default_rng(5)
        assert bias_share(rng.normal(0, 1, 200_000)) < 0.01

    def test_half_and_half(self):
        """mean r = -1, mean |r| = 2 -> 0.5, by construction."""
        r = np.array([-3.0, 1.0, -3.0, 1.0])
        assert bias_share(r) == pytest.approx(0.5)


class TestPooledIsBiasedDown:
    def test_mixture_of_scales_looks_leptokurtic(self):
        """The reason kappa is computed within slide and then combined.

        Every slide is Gaussian; only the scale differs between them. Pooling
        makes the sample heavy-tailed and drags kappa below 0.798, which would
        be read as a non-Gaussian error if nobody had checked.
        """
        rng = np.random.default_rng(6)
        parts = [rng.normal(0, s, 4000) for s in (0.2, 1.0, 5.0)]
        within = float(np.mean([shape_centred(p) for p in parts]))
        pooled = shape_centred(np.concatenate(parts))
        assert within == pytest.approx(GAUSSIAN, abs=0.01)
        assert pooled < within - 0.05


class TestPerSlide:
    def _frame(self, n_slides=6, n_per=80, offset=0.0, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for w in range(n_slides):
            sd = np.abs(rng.gamma(2, 0.01, n_per)) + 1e-3
            r = rng.normal(offset, 1.0, n_per) * sd
            for i in range(n_per):
                rows.append(dict(wsi=f"wsi{w:03d}", sd=sd[i], r=r[i]))
        return pd.DataFrame(rows)

    def test_one_row_per_slide(self):
        out = per_slide(self._frame(), min_regions=10)
        assert len(out) == 6
        assert {r["wsi"] for r in out} == {f"wsi{w:03d}" for w in range(6)}

    def test_short_slides_dropped(self):
        f = self._frame(n_slides=4, n_per=40)
        f = pd.concat([f, pd.DataFrame([dict(wsi="wsi999", sd=0.1, r=0.01)] * 3)])
        out = per_slide(f, min_regions=10)
        assert all(r["wsi"] != "wsi999" for r in out)

    def test_nonpositive_sd_dropped_not_propagated_as_inf(self):
        f = self._frame(n_slides=3, n_per=40)
        f.loc[f.index[:5], "sd"] = 0.0
        out = per_slide(f, min_regions=5)
        assert all(np.isfinite(r["kappa_centred"]) for r in out)

    def test_under_predicts_tracks_the_sign_of_r(self):
        """r = mu - real, so under-predicting means r < 0."""
        out = per_slide(self._frame(offset=-3.0, seed=2), min_regions=10)
        assert all(r["under_predicts"] for r in out)
        out = per_slide(self._frame(offset=+3.0, seed=2), min_regions=10)
        assert not any(r["under_predicts"] for r in out)


class TestCombineAndVerdict:
    def test_ci_brackets_the_mean(self):
        c = combine([0.79, 0.80, 0.81, 0.78, 0.82], n_boot=500, seed=0)
        assert c["ci_lo"] <= c["mean"] <= c["ci_hi"]
        assert c["n_slides"] == 5

    def test_too_few_slides_gives_no_interval(self):
        c = combine([0.79, 0.80], n_boot=500, seed=0)
        assert "ci_lo" not in c

    def test_nonfinite_dropped(self):
        c = combine([0.79, float("nan"), 0.81], n_boot=0, seed=0)
        assert c["n_slides"] == 2

    def test_verdict_identifies_gaussian(self):
        v = verdict({"mean": GAUSSIAN, "ci_lo": 0.78, "ci_hi": 0.81})
        assert v["nearest"] == "Gaussian" and v["gaussian_covered"] is True
        assert "Uniform" not in v["covers"]

    def test_verdict_flags_an_excluded_gaussian(self):
        v = verdict({"mean": 0.71, "ci_lo": 0.69, "ci_hi": 0.73})
        assert v["gaussian_covered"] is False
        assert v["nearest"] in ("Laplace", "Student-t, 5 df")
        # 0.80 would over-state the calibrated line by ~12%.
        assert v["relative_error_of_0_80_line"] == pytest.approx(0.124, abs=0.01)


class TestOrderingCheck:
    """The check is on the MEANS. Per-slide crossing is the Gaussian null."""

    def _summ(self, centred, uncentred):
        return [{"component": "total", "kappa_centred_mean": centred,
                 "kappa_uncentred_mean": uncentred}]

    def test_passes_when_the_means_are_ordered(self):
        rows = [{"wsi": "a", "n_regions": 100, "kappa_centred": 0.79,
                 "kappa_uncentred": 0.85, "component": "total"}]
        assert check_ordering(rows, self._summ(0.79, 0.85))["passes"]

    def test_fails_when_the_means_invert(self):
        rows = [{"wsi": "a", "n_regions": 500, "kappa_centred": 0.85,
                 "kappa_uncentred": 0.70, "component": "total"}]
        r = check_ordering(rows, self._summ(0.85, 0.70))
        assert not r["passes"]
        assert r["by_component_mean"][0]["deficit_of_means"] > 0

    def test_gaussian_null_does_not_fail_the_run(self):
        """The bug this replaced: unbiased Gaussian data failed the check.

        About one slide in six crosses under a pure Gaussian null at any n, so a
        tight per-slide tolerance rejects correct data most of the time on a
        twenty-case cohort.
        """
        rng = np.random.default_rng(0)
        rows, cs, us = [], [], []
        for w in range(20):
            u = rng.normal(0, 1, 150)
            c, un = shape_centred(u), shape_uncentred(u)
            cs.append(c)
            us.append(un)
            rows.append({"wsi": f"w{w}", "n_regions": 150, "kappa_centred": c,
                         "kappa_uncentred": un, "component": "total"})
        r = check_ordering(rows, self._summ(float(np.mean(cs)), float(np.mean(us))))
        assert r["passes"]
        assert not r["crossings_beyond_null"]

    def test_flags_a_crossing_rate_far_above_the_null(self):
        rows = [{"wsi": f"w{i}", "n_regions": 200, "kappa_centred": 0.85,
                 "kappa_uncentred": 0.70, "component": "total"} for i in range(10)]
        assert check_ordering(rows, self._summ(0.85, 0.70))["crossings_beyond_null"]


class TestScaleMixtureConfound:
    """Standardising by a mis-tracking sigma manufactures heavy tails."""

    def test_mistracking_sigma_drags_kappa_to_laplace(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0, 1, 200_000)           # errors are Gaussian throughout
        sigma = np.abs(rng.normal(1, 0.6, 200_000)) + 1e-6
        assert shape_centred(r) == pytest.approx(GAUSSIAN, abs=0.005)
        # ...yet the standardised residual reads as Laplace.
        assert shape_centred(r / sigma) < 0.72

    def test_flat_sigma_leaves_kappa_alone(self):
        rng = np.random.default_rng(8)
        r = rng.normal(0, 1, 100_000)
        assert shape_centred(r / 2.5) == pytest.approx(shape_centred(r))

    def test_per_slide_reports_both_readings_and_cv(self):
        rng = np.random.default_rng(9)
        n = 4000
        sd = np.abs(rng.normal(1, 0.5, n)) + 1e-3
        f = pd.DataFrame({"wsi": ["a"] * n, "sd": sd,
                          "r": rng.normal(0, 1, n)})
        row = per_slide(f, min_regions=10)[0]
        assert row["cv_sd"] == pytest.approx(np.std(sd, ddof=1) / sd.mean())
        # raw is the honest Gaussian; standardised is dragged down by the mixture
        assert row["kappa_centred_raw"] == pytest.approx(GAUSSIAN, abs=0.02)
        assert row["kappa_centred"] < row["kappa_centred_raw"] - 0.02
