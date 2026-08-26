"""W-29: stability of the data-exposure component at K = 5.

The load-bearing test is `test_matches_decompose_exactly`: rebuilding the
decomposition from per-fold means and SDs must give bit-comparable numbers to
`uncertainty_phi.decompose` run on the members themselves. If it does not, every
leave-one-subset-out replicate is a different estimator and the spec's "the five
should bracket 0.508" would be testing this script rather than the grid.
"""

import numpy as np
import pandas as pd
import pytest

from stability_data_exposure import (cluster_bootstrap_median,
                                     matched_summaries,
                                     components_from_folds, df_asymmetry,
                                     jackknife, leave_one_subset_out, load_folds,
                                     seed_subsample_exact,
                                     seed_subsample_parametric, share,
                                     summarise_share)
from uncertainty_phi.decompose import decompose


def _grid(K=5, S=10, R=40, seed=0, d=1):
    """[K][S, R, d] members, the shape decompose() consumes."""
    rng = np.random.default_rng(seed)
    fold_level = rng.normal(0, 0.5, (K, 1, R, d))
    return [fold_level[k] + rng.normal(0, 1.0, (S, R, d)) for k in range(K)]


def _summaries(folds):
    mu = np.stack([f[:, :, 0].mean(axis=0) for f in folds])
    var = np.stack([f[:, :, 0].var(axis=0, ddof=1) for f in folds])
    counts = np.array([float(f.shape[0]) for f in folds])
    return mu, var, counts


class TestMatchesTheRealEstimator:
    def test_matches_decompose_exactly(self):
        folds = _grid()
        ref = decompose(folds)
        mu, var, counts = _summaries(folds)
        got = components_from_folds(mu, var, counts)
        assert got["procedural"] == pytest.approx(ref.procedural_per_dim[:, 0])
        assert got["data"] == pytest.approx(ref.data_per_dim[:, 0])
        assert got["total"] == pytest.approx(ref.total_per_dim[:, 0])

    def test_matches_with_unbalanced_seed_counts(self):
        """n0 is the ANOVA effective group size, not S — check it degrades right."""
        folds = _grid(K=4, S=12)
        folds = [folds[0][:12], folds[1][:7], folds[2][:10], folds[3][:5]]
        ref = decompose(folds)
        mu, var, counts = _summaries(folds)
        got = components_from_folds(mu, var, counts)
        assert counts.tolist() != [counts[0]] * len(counts)
        assert got["data"] == pytest.approx(ref.data_per_dim[:, 0])

    def test_n0_is_S_for_a_balanced_grid(self):
        mu, var, counts = _summaries(_grid(K=5, S=10))
        assert components_from_folds(mu, var, counts)["n0"] == pytest.approx(10.0)

    def test_bias_correction_is_actually_applied(self):
        """Without the -procedural/n0 term the data component reads high.

        Verified on a grid with NO true between-fold signal: the uncorrected
        estimate is high by precisely procedural/n0.
        """
        rng = np.random.default_rng(3)
        folds = [rng.normal(0, 1.0, (10, 4000, 1)) for _ in range(5)]
        mu, var, counts = _summaries(folds)
        got = components_from_folds(mu, var, counts)
        between = np.var(mu, axis=0, ddof=1)
        assert np.mean(got["data"]) == pytest.approx(0.0, abs=0.02)
        assert np.mean(between) == pytest.approx(np.mean(got["procedural"]) / 10.0,
                                                 rel=0.15)


class TestShare:
    def test_negative_data_component_is_nan_not_zero(self):
        comp = {"data": np.array([0.5, -0.1, 0.0]),
                "total": np.array([1.0, 0.9, 1.0])}
        s = share(comp)
        assert s[0] == pytest.approx(0.5)
        assert np.isnan(s[1]) and np.isnan(s[2])

    def test_summarise_counts_the_drops(self):
        s = np.array([0.4, 0.6, np.nan, np.nan])
        out = summarise_share(s, n_all=4)
        assert out["n_with_data"] == 2
        assert out["n_dropped_no_data_component"] == 2
        assert out["median_share"] == pytest.approx(0.5)

    def test_additivity_puts_the_share_in_the_unit_interval(self):
        folds = _grid(seed=4)
        mu, var, counts = _summaries(folds)
        s = share(components_from_folds(mu, var, counts))
        v = s[np.isfinite(s)]
        assert len(v) and ((v > 0) & (v <= 1)).all()


class TestLeaveOneSubsetOut:
    def test_one_replicate_per_subset_with_one_fewer_fold(self):
        mu, var, counts = _summaries(_grid())
        out, shares = leave_one_subset_out(
            mu, var, counts,
            components_from_folds(mu, var, counts)["procedural"],
            hold_procedural=True)
        assert len(out) == 5 and len(shares) == 5
        assert [r["dropped_fold"] for r in out] == [1, 2, 3, 4, 5]
        assert all(r["n_folds_used"] == 4 and r["df_between"] == 3 for r in out)

    def test_selection_lifts_the_unmatched_median(self):
        """Why the spec's bracketing check is not a diagnostic.

        Dropping a subset makes the between term noisier, so MORE regions fall
        at or below zero and drop out — and the survivors are the ones with the
        larger data estimates. Every replicate then lands above the full grid,
        which the spec would read as a broken recomputation.
        """
        mu, var, counts = _summaries(_grid(seed=5, R=400))
        full = components_from_folds(mu, var, counts)
        med = float(np.nanmedian(share(full)))
        n_full = int(np.isfinite(share(full)).sum())
        out, _ = leave_one_subset_out(mu, var, counts, full["procedural"], True)
        assert all(r["n_with_data"] < n_full for r in out)
        assert min(r["median_share"] for r in out) > med

    def test_jensen_lowers_the_matched_median(self):
        """The opposite direction, on a fixed region set.

        share = 1/(1 + proc/data) is concave in data, so a noisier data term
        pulls the median down once selection is removed. Both effects are real
        and neither is a bug — which is the finding.
        """
        mu, var, counts = _summaries(_grid(seed=5, R=400))
        full = components_from_folds(mu, var, counts)
        out, shares = leave_one_subset_out(mu, var, counts, full["procedural"], True)
        m_full, m_rows, common = matched_summaries(share(full), shares)
        assert m_full["n_matched"] < int(np.isfinite(share(full)).sum())
        assert all(r["median_share_matched"] < m_full["median_share_matched"]
                   for r in m_rows)

    def test_matched_uses_one_common_region_set(self):
        mu, var, counts = _summaries(_grid(seed=2, R=200))
        full = components_from_folds(mu, var, counts)
        out, shares = leave_one_subset_out(mu, var, counts, full["procedural"], True)
        m_full, m_rows, common = matched_summaries(share(full), shares)
        assert all(r["n_matched"] == m_full["n_matched"] for r in m_rows)
        assert common.sum() == m_full["n_matched"]
        for sh in shares:
            assert np.isfinite(sh[common]).all()

    def test_holding_procedural_fixed_barely_moves_it(self):
        """45 df against 4: the procedural term is not what is unstable."""
        mu, var, counts = _summaries(_grid(seed=6, R=400))
        full = components_from_folds(mu, var, counts)
        held, _ = leave_one_subset_out(mu, var, counts, full["procedural"], True)
        recomputed, _ = leave_one_subset_out(mu, var, counts, full["procedural"], False)
        d = max(abs(a["median_share"] - b["median_share"])
                for a, b in zip(held, recomputed))
        assert d < 0.05


class TestJackknife:
    def test_range_and_spread_reported(self):
        j = jackknife(0.5, [0.48, 0.52, 0.50, 0.49, 0.51])
        assert j["range_lo"] == 0.48 and j["range_hi"] == 0.52
        assert j["spread"] == pytest.approx(0.04)

    def test_se_formula(self):
        v = [0.40, 0.60]
        j = jackknife(0.5, v)
        # sqrt((K-1)/K * sum (x - xbar)^2) with K=2 -> sqrt(0.5 * 0.02)
        assert j["jackknife_se"] == pytest.approx(np.sqrt(0.5 * 0.02))

    def test_bias_is_zero_when_replicates_centre_on_the_estimate(self):
        assert jackknife(0.5, [0.5] * 5)["jackknife_bias"] == pytest.approx(0.0)


class TestSeedContrast:
    def test_exact_route_uses_fewer_seeds(self):
        folds = _grid(K=5, S=10, R=200, seed=7)
        members = [f[:, :, 0] for f in folds]
        v = seed_subsample_exact(members, 5, 30, seed=0)
        assert len(v) == 30 and all(np.isfinite(v))

    def test_parametric_route_tracks_the_exact_one(self):
        """The model is only worth shipping if it lands where the real thing does."""
        folds = _grid(K=5, S=10, R=400, seed=8)
        members = [f[:, :, 0] for f in folds]
        mu, var, counts = _summaries(folds)
        ex = np.array(seed_subsample_exact(members, 5, 120, seed=0))
        pa = np.array(seed_subsample_parametric(mu, var, counts, 5, 120, seed=0))
        assert np.nanmean(pa) == pytest.approx(np.nanmean(ex), abs=0.05)

    def test_the_raw_loso_sd_understates_and_must_be_inflated(self):
        """Why the spec's "seed spread is far tighter" does not survive.

        The K replicates each share K-1 of K folds, so their raw SD is a
        correlated dispersion, not a sampling SE. Putting it beside an
        independent bootstrap SD compares two different objects; the jackknife
        inflation by (K-1)/sqrt(K) is what makes them comparable.
        """
        folds = _grid(K=5, S=10, R=400, seed=9)
        mu, var, counts = _summaries(folds)
        full = components_from_folds(mu, var, counts)
        loso, _ = leave_one_subset_out(mu, var, counts, full["procedural"], True)
        v = [r["median_share"] for r in loso]
        raw_sd = float(np.std(v, ddof=1))
        se = jackknife(float(np.nanmedian(share(full))), v)["jackknife_se"]
        assert se == pytest.approx(raw_sd * 4 / np.sqrt(5), rel=1e-9)
        assert se > raw_sd

    def test_bigger_seed_cut_moves_the_answer_more(self):
        """Matched fractions are the only like-for-like comparison.

        Halving the seeds is a larger perturbation than removing a fifth of the
        subsets, so the spec's S->5 arm is not comparable to the subset arm at
        all; the matched size is S*(1 - 1/K).
        """
        folds = _grid(K=5, S=10, R=400, seed=9)
        mu, var, counts = _summaries(folds)
        sds = {}
        for s_sub in (8, 5):
            d = seed_subsample_parametric(mu, var, counts, s_sub, 150, seed=0)
            sds[s_sub] = float(np.nanstd(d, ddof=1))
        assert sds[5] > sds[8]


class TestDfAsymmetry:
    def test_reads_the_grid_rather_than_assuming(self):
        d = df_asymmetry(5, [10, 10, 10, 10, 10])
        assert d["df_procedural"] == 45 and d["df_data_exposure"] == 4
        assert d["relative_se_data_exposure"] == pytest.approx(np.sqrt(0.5))
        assert d["relative_se_procedural"] == pytest.approx(np.sqrt(2 / 45))

    def test_unbalanced_grid(self):
        d = df_asymmetry(3, [10, 8, 6])
        assert d["df_procedural"] == 21 and d["df_data_exposure"] == 2


class TestLoadFolds:
    def test_reads_fold_columns(self):
        t = pd.DataFrame({f"fold{f}_{k}_x": np.arange(3.0) + f
                          for f in (1, 2, 3) for k in ("mu", "sd")})
        mu, var, names = load_folds(t, "x")
        assert mu.shape == (3, 3) and names == ["fold1", "fold2", "fold3"]
        assert var[0] == pytest.approx((np.arange(3.0) + 1) ** 2)

    def test_single_fold_refused(self):
        t = pd.DataFrame({"fold1_mu_x": [1.0], "fold1_sd_x": [0.1]})
        with pytest.raises(SystemExit, match="CROSSED grid"):
            load_folds(t, "x")

    def test_missing_sd_column_named(self):
        t = pd.DataFrame({"fold1_mu_x": [1.0], "fold1_sd_x": [0.1],
                          "fold2_mu_x": [1.0]})
        with pytest.raises(SystemExit, match="fold2_sd_x"):
            load_folds(t, "x")


class TestBootstrap:
    def test_resamples_slides_not_regions(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 600)
        wsi = np.repeat([f"w{i}" for i in range(20)], 30)
        b = cluster_bootstrap_median(s, wsi, 300, 0)
        assert b["n_slides"] == 20
        assert b["ci_lo"] < np.median(s) < b["ci_hi"]

    def test_too_few_slides_gives_nothing(self):
        s = np.random.default_rng(0).uniform(0, 1, 40)
        assert cluster_bootstrap_median(s, np.repeat(["a", "b"], 20), 300, 0) == {}
