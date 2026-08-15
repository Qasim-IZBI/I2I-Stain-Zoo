"""Plateau detection for --white_thresh.

These decide a number that every level-A descriptor inherits, so the failure
that matters is the confident wrong answer: reporting a plateau on a curve that
is uniformly sloped, or reading lumen values from a footprint that has swallowed
the slide background.
"""

import numpy as np
import pytest

from calibrate_white_thresh import find_plateau, footprint_breakdown, suggest_threshold


def _sloped(t: np.ndarray, per_step: float = 0.877) -> np.ndarray:
    """Constant relative decline — the UC liver H&E, which has no flat spot."""
    return 0.073 * per_step ** np.arange(len(t))


class TestFootprintBreakdown:
    def test_flags_the_step_where_background_is_absorbed(self):
        t = np.arange(0.50, 0.7251, 0.025)
        tissue = np.full(len(t), 0.606)
        tissue[-2] *= 1.006          # transition
        tissue[-1] = 0.741           # background swallowed
        assert footprint_breakdown(t, tissue) == pytest.approx(0.700)

    def test_stable_footprint_reports_none(self):
        t = np.arange(0.50, 0.7001, 0.025)
        tissue = 0.606 + np.linspace(0, 0.0004, len(t))   # 0.07% drift
        assert footprint_breakdown(t, tissue) is None


class TestFindPlateau:
    def test_uniform_slope_is_not_a_plateau(self):
        """The defect this replaced: a 2x-the-minimum rule with no absolute
        anchor called the middle of a 12%-per-step slope 'flat'."""
        t = np.arange(0.50, 0.6751, 0.025)
        out = find_plateau(t, _sloped(t))
        assert out["lo"] is None
        assert "no threshold is flat" in out["reason"]
        assert "convention, not a measurement" in out["reason"]

    def test_genuinely_flat_curve_is_found(self):
        t = np.arange(0.50, 0.7001, 0.025)
        lumen = _sloped(t)
        lumen[3:7] = lumen[3]                      # a real plateau
        out = find_plateau(t, lumen)
        # central differences, so only the interior of the flat run reads as flat:
        # the endpoints straddle the slope on one side
        assert out["lo"] == pytest.approx(t[4])
        assert out["hi"] == pytest.approx(t[5])

    def test_broken_footprint_is_excluded_before_anything_else(self):
        """Past the breakdown lumen_fraction goes non-monotonic, which can look
        flat to a gradient. Those thresholds must not be candidates."""
        t = np.arange(0.50, 0.7751, 0.025)
        lumen = np.concatenate([_sloped(t[:8]), [0.012, 0.024, 0.001, 0.0005]])
        tissue = np.concatenate([np.full(8, 0.606), [0.610, 0.741, 0.748, 0.748]])
        out = find_plateau(t, lumen, tissue)
        # 0.610/0.606 is +0.66%, over the 0.5% rule — the transition step itself
        # is caught, not merely the fully-broken one after it
        assert out["breakdown"] == pytest.approx(0.700)
        assert out["valid_hi"] == pytest.approx(0.675)
        if out["lo"] is not None:
            assert out["hi"] <= out["valid_hi"]

    def test_too_few_usable_thresholds_says_so(self):
        t = np.array([0.50, 0.525, 0.55])
        tissue = np.array([0.6, 0.9, 0.9])          # breaks immediately
        out = find_plateau(t, _sloped(t), tissue)
        assert out["lo"] is None
        assert "fewer than three" in out["reason"]


class TestSuggestThreshold:
    def test_two_modes_give_the_valley_between_them(self):
        c = np.linspace(0, 1, 128)
        counts = (1e6 * np.exp(-((c - 0.45) ** 2) / 0.004)
                  + 1e5 * np.exp(-((c - 0.75) ** 2) / 0.002))
        out = suggest_threshold(counts, c)
        assert 0.55 < out["threshold"] < 0.72
        assert out["tissue_mode"] < out["threshold"] < out["whitespace_mode"]

    def test_one_mode_returns_none_rather_than_a_number(self):
        c = np.linspace(0, 1, 128)
        assert suggest_threshold(1e6 * np.exp(-((c - 0.45) ** 2) / 0.004), c) is None
