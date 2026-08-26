"""W-30: ranking rules beyond sigma alone.

The load-bearing test is `test_sd_rule_matches_risk_coverage`: the new pooled
sigma curve must be the SAME NUMBER as `calibrate_phi.risk_coverage()` produces,
or the comparison against mu is happening on a different footing than the
published sigma result and nothing in the table is quotable.
"""

import numpy as np
import pandas as pd
import pytest

from compare_ranking_rules import (build_scores, by_slide_pct_rank, curve_pooled,
                                   curve_within, loco_fitted_score, pct_rank,
                                   run_component)


def _table(n_slides=8, n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_slides):
        mu = np.abs(rng.gamma(2.0, 0.02, n_per))
        sd = 0.15 * mu + 0.002 * rng.gamma(2, 1, n_per)
        err = np.abs(rng.normal(0, 1, n_per)) * (0.6 * sd + 0.25 * mu)
        for i in range(n_per):
            rows.append(dict(descriptor="task_specific_value", component="total",
                             prediction="grand", wsi=f"wsi{w:03d}",
                             region_index=i, mu=mu[i], sd=sd[i],
                             real=mu[i] - err[i], error=err[i]))
    return pd.DataFrame(rows)


class TestRanks:
    def test_pct_rank_is_open_unit_interval(self):
        r = pct_rank(np.array([3.0, 1.0, 2.0]))
        assert (r > 0).all() and (r < 1).all()
        assert list(np.argsort(r)) == [1, 2, 0]

    def test_ties_average(self):
        r = pct_rank(np.array([1.0, 1.0, 2.0]))
        assert r[0] == r[1]

    def test_by_slide_rank_is_independent_across_slides(self):
        v = np.array([1.0, 2.0, 100.0, 200.0])
        groups = [np.array([0, 1]), np.array([2, 3])]
        r = by_slide_pct_rank(v, groups)
        # The second slide sits at much larger values but gets the same ranks,
        # which is the point: a rank of mu must not encode which case it is.
        assert r[0] == r[2] and r[1] == r[3]


class TestCurves:
    def test_random_is_exactly_flat(self):
        t = _table()
        err = t["error"].to_numpy()
        c = curve_pooled(err, {"sd": t["sd"].to_numpy()}, [1.0, 0.8, 0.5])
        assert c["random"][0.8] == pytest.approx(err.mean())
        assert c["random"][0.5] == pytest.approx(err.mean())

    def test_oracle_is_the_floor(self):
        t = _table()
        err = t["error"].to_numpy()
        c = curve_pooled(err, {"sd": t["sd"].to_numpy(), "mu": t["mu"].to_numpy()},
                         [0.8, 0.5])
        for cov in (0.8, 0.5):
            assert c["oracle"][cov] <= c["sd"][cov] + 1e-12
            assert c["oracle"][cov] <= c["mu"][cov] + 1e-12

    def test_full_coverage_keeps_everything(self):
        t = _table()
        err = t["error"].to_numpy()
        c = curve_pooled(err, {"sd": t["sd"].to_numpy()}, [1.0])
        assert c["sd"][1.0] == pytest.approx(err.mean())
        assert c["oracle"][1.0] == pytest.approx(err.mean())

    def test_within_weights_slides_equally(self):
        # One huge slide and one small one: the pooled base is dominated by the
        # big slide, the within base is their unweighted mean.
        err = np.concatenate([np.full(100, 1.0), np.full(4, 5.0)])
        groups = [np.arange(100), np.arange(100, 104)]
        c = curve_within(err, {"sd": err.copy()}, groups, [1.0])
        assert c["random"][1.0] == pytest.approx(3.0)
        assert curve_pooled(err, {"sd": err.copy()}, [1.0])["random"][1.0] < 1.2


class TestFitted:
    def test_no_leakage_from_the_held_out_slide(self):
        """A held-out case's own errors must not move its own score."""
        t = _table(seed=3)
        mu, sd, err = (t[c].to_numpy() for c in ("mu", "sd", "error"))
        wsi = t["wsi"].to_numpy()
        ids = list(pd.unique(wsi))
        groups = [np.where(wsi == s)[0] for s in ids]
        s1, _ = loco_fitted_score(by_slide_pct_rank(mu, groups),
                                  by_slide_pct_rank(sd, groups), err, groups, ids)
        # Scramble the FIRST slide's errors only. Its own out-of-fold score is
        # fit on the other slides, so it must be untouched.
        err2 = err.copy()
        rng = np.random.default_rng(0)
        err2[groups[0]] = rng.permutation(err2[groups[0]]) * 7.0
        s2, _ = loco_fitted_score(by_slide_pct_rank(mu, groups),
                                  by_slide_pct_rank(sd, groups), err2, groups, ids)
        assert np.allclose(s1[groups[0]], s2[groups[0]])
        # ...while the other slides, which had that slide in their training set,
        # do move. Otherwise the test would pass on a model that ignores y.
        assert not np.allclose(s1[groups[1]], s2[groups[1]])

    def test_too_few_slides_returns_nan_rather_than_a_two_slide_model(self):
        err = np.arange(8.0)
        groups = [np.arange(4), np.arange(4, 8)]
        s, coefs = loco_fitted_score(err / 8, err / 8, err, groups, ["a", "b"])
        assert np.isnan(s).all() and coefs == []

    def test_ranksum_is_the_pinned_case_of_the_fit(self):
        t = _table(seed=5)
        mu, sd, err = (t[c].to_numpy() for c in ("mu", "sd", "error"))
        wsi = t["wsi"].to_numpy()
        ids = list(pd.unique(wsi))
        groups = [np.where(wsi == s)[0] for s in ids]
        scores, _ = build_scores(mu, sd, err, groups, ids)
        assert np.allclose(scores["ranksum"],
                           scores["mu_rank"] + scores["sd_rank"])


class TestAgainstPublishedPath:
    def test_sd_rule_matches_risk_coverage(self):
        """The pooled sigma curve must equal calibrate_phi's, exactly.

        Same regions, same ranking, same coverage grid — so any difference is a
        bug here, and the mu comparison would be against a sigma number the
        paper does not report.
        """
        from calibrate_phi import risk_coverage
        t = _table(n_slides=10, n_per=80, seed=11)
        covs = [1.0, 0.9, 0.8, 0.7, 0.5]
        ref = {r["coverage"]: r for r in risk_coverage(t, covs, 0, 0)}
        rows, _, _ = run_component(t, covs, n_boot=0, seed=0,
                                   reference_rule="mu", min_regions_per_slide=1)
        mine = {r["coverage"]: r for r in rows
                if r["scope"] == "pooled" and r["rule"] == "sd"}
        assert set(mine) == set(covs)
        for c in covs:
            assert mine[c]["mae"] == pytest.approx(ref[c]["mae"])
            assert mine[c]["rel_change"] == pytest.approx(ref[c]["rel_change"])
            assert mine[c]["mae_oracle"] == pytest.approx(ref[c]["mae_oracle"])


class TestRunComponent:
    def test_every_rule_and_scope_present(self):
        t = _table()
        rows, deltas, coefs = run_component(t, [1.0, 0.8], n_boot=0, seed=0,
                                            reference_rule="mu",
                                            min_regions_per_slide=1)
        df = pd.DataFrame(rows)
        assert set(df["scope"]) == {"pooled", "within"}
        assert {"mu", "sd", "mu_rank", "sd_rank", "ranksum", "fitted",
                "oracle", "random"} <= set(df["rule"])
        assert len(coefs) == t["wsi"].nunique()
        assert all(d["reference_rule"] == "mu" for d in deltas)

    def test_mu_and_mu_rank_agree_within_slide(self):
        """Not a finding — the same ordering under two names."""
        t = _table(seed=7)
        rows, _, _ = run_component(t, [1.0, 0.8], n_boot=0, seed=0,
                                   reference_rule="mu", min_regions_per_slide=1)
        w = {(r["rule"], r["coverage"]): r["mae"] for r in rows
             if r["scope"] == "within"}
        assert w[("mu", 0.8)] == pytest.approx(w[("mu_rank", 0.8)])

    def test_short_slides_are_dropped_from_both_scopes(self):
        t = _table(n_slides=6, n_per=40)
        extra = t[t["wsi"] == "wsi000"].head(3).copy()
        extra["wsi"] = "wsi999"
        rows, _, coefs = run_component(pd.concat([t, extra]), [1.0, 0.8],
                                       n_boot=0, seed=0, reference_rule="mu",
                                       min_regions_per_slide=10)
        assert all(c["held_out_wsi"] != "wsi999" for c in coefs)
        assert rows[0]["n_slides"] == 6

    def test_too_few_slides_scores_nothing(self):
        t = _table(n_slides=2, n_per=50)
        rows, deltas, coefs = run_component(t, [1.0, 0.8], n_boot=0, seed=0,
                                            reference_rule="mu",
                                            min_regions_per_slide=1)
        assert rows == [] and deltas == [] and coefs == []

    def test_bootstrap_pairs_the_rules(self):
        """The delta CI must be narrower than the two marginal CIs summed.

        If the rules were bootstrapped independently the difference would carry
        both curves' cohort variance instead of cancelling it, which is exactly
        the mistake W-30 exists to avoid.
        """
        t = _table(n_slides=12, n_per=70, seed=13)
        rows, deltas, _ = run_component(t, [1.0, 0.8], n_boot=200, seed=0,
                                        reference_rule="mu",
                                        min_regions_per_slide=1)
        w = {(r["rule"], r["scope"]): r for r in rows if r["coverage"] == 0.8}
        d = next(x for x in deltas if x["rule"] == "ranksum"
                 and x["scope"] == "pooled" and x["coverage"] == 0.8)
        marginal = ((w[("ranksum", "pooled")]["rel_ci_hi"]
                     - w[("ranksum", "pooled")]["rel_ci_lo"])
                    + (w[("mu", "pooled")]["rel_ci_hi"]
                       - w[("mu", "pooled")]["rel_ci_lo"]))
        assert (d["delta_ci_hi"] - d["delta_ci_lo"]) < marginal
