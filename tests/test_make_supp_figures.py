"""The two supplement figures of FIGURE_REQUESTS.md.

These are rendering jobs, so the tests guard the two things a rendering job can
silently get wrong: the numbers behind F-1 (which the paper already quotes) and
the geometry behind F-2 (whose whole claim is that one grid indexes the same
tissue in both stains).
"""

import numpy as np
import pandas as pd
import pytest

from make_supp_figures import (BAR_UM, EXPECT, TEXT_IN, _stability_checks,
                               check_frame, fill, find_slide, fit_for_print,
                               kept_regions, strip_prefix, thumbnail)


class TestCaptions:
    def test_fill_substitutes_tokens(self):
        assert fill("a <<X>> b <<Y>>", X=1, Y="q") == "a 1 b q"

    def test_unfilled_placeholder_raises(self):
        """LaTeX braces and str.format do not mix; this is the replacement.

        A missing value must fail loudly — a caption shipped with a literal
        <<MED>> in it is worse than a crash.
        """
        with pytest.raises(ValueError, match="MED"):
            fill("value <<MED>> and <<HI>>", HI=1)

    def test_latex_braces_pass_through_untouched(self):
        out = fill("\\caption{x is <<V>>}\\label{fig:y}", V=3)
        assert out == "\\caption{x is 3}\\label{fig:y}"


class TestStabilityChecks:
    def _res(self, med, lo, hi, n=5, spread=0.001):
        return {"full": {"median_share": med},
                "jk": {"range_lo": lo, "range_hi": hi},
                "loso": [{}] * n,
                "seeds": {5: list(np.full(50, 0.5) + spread * np.arange(50)
                                  / 49 * np.sqrt(2))}}

    def test_published_values_pass(self):
        r = self._res(EXPECT["median_share"], EXPECT["loso_lo"], EXPECT["loso_hi"])
        assert all(c["ok"] for c in _stability_checks(r))

    def test_a_moved_median_fails(self):
        r = self._res(0.55, EXPECT["loso_lo"], EXPECT["loso_hi"])
        bad = [c for c in _stability_checks(r) if not c["ok"]]
        assert [c["quantity"] for c in bad] == ["full-grid median share"]

    def test_tolerance_is_the_papers_own_precision(self):
        """Three decimals is what the manuscript prints."""
        r = self._res(EXPECT["median_share"] + 0.0004, EXPECT["loso_lo"],
                      EXPECT["loso_hi"])
        assert _stability_checks(r)[0]["ok"]
        r = self._res(EXPECT["median_share"] + 0.002, EXPECT["loso_lo"],
                      EXPECT["loso_hi"])
        assert not _stability_checks(r)[0]["ok"]

    def test_wrong_replicate_count_fails(self):
        r = self._res(EXPECT["median_share"], EXPECT["loso_lo"],
                      EXPECT["loso_hi"], n=4)
        assert not [c for c in _stability_checks(r)
                    if c["quantity"] == "n LOSO replicates"][0]["ok"]


class TestSlideMatching:
    def test_strip_prefix_pairs_the_two_stains(self):
        assert strip_prefix("SR_d31_BDL+A_M2") == strip_prefix("HE_d31_BDL+A_M2")

    def test_no_underscore_is_left_alone(self):
        assert strip_prefix("case1") == "case1"

    def test_exact_stem_wins(self, tmp_path):
        for n in ("HE_case_M3.tif", "SR_case_M3.tif"):
            (tmp_path / n).write_bytes(b"")
        assert find_slide(tmp_path, "HE_case_M3").name == "HE_case_M3.tif"

    def test_prefix_stripped_match(self, tmp_path):
        (tmp_path / "SR_case_M3.tif").write_bytes(b"")
        assert find_slide(tmp_path, "HE_case_M3").name == "SR_case_M3.tif"

    def test_two_files_on_one_key_is_fatal(self, tmp_path):
        """The wrong slide beside the right one is invisible in the output."""
        for n in ("SR_case_M3.tif", "XX_case_M3.tif"):
            (tmp_path / n).write_bytes(b"")
        with pytest.raises(SystemExit, match="match"):
            find_slide(tmp_path, "HE_case_M3")

    def test_no_match_names_what_is_there(self, tmp_path):
        (tmp_path / "SR_other.tif").write_bytes(b"")
        with pytest.raises(SystemExit, match="SR_other"):
            find_slide(tmp_path, "HE_case_M3")


class TestFrameCheck:
    def test_exact_frame_passes(self):
        check_frame("SR", np.zeros((100, 200, 3), np.uint8), 100, 200, 512)

    def test_truncation_excess_below_one_tile_is_accepted(self):
        """reconstruct_wsi truncates to a whole number of tiles."""
        check_frame("SR", np.zeros((400, 700, 3), np.uint8), 100, 200, 512)

    def test_excess_of_a_whole_tile_exits(self):
        with pytest.raises(SystemExit, match="different frames"):
            check_frame("SR", np.zeros((700, 200, 3), np.uint8), 100, 200, 512)

    def test_shorter_than_the_grid_exits(self):
        with pytest.raises(SystemExit, match="SHORTER"):
            check_frame("SR", np.zeros((90, 200, 3), np.uint8), 100, 200, 512)


class TestGeometry:
    def test_thumbnail_stride_bounds_the_long_side(self):
        a = np.zeros((5000, 9000, 3), np.uint8)
        t, step = thumbnail(a, 1000)
        assert max(t.shape[:2]) <= 1000 and step == 9

    def test_same_shape_gives_the_same_stride(self):
        """Different strides would draw the two grids at different scales."""
        a = np.zeros((5000, 9000, 3), np.uint8)
        b = np.zeros((5000, 9000, 3), np.uint8)
        assert thumbnail(a, 1200)[1] == thumbnail(b, 1200)[1]

    def test_fit_for_print_shrinks_to_the_panel(self):
        a = np.zeros((2048, 2048, 3), np.uint8)
        out = fit_for_print(a, TEXT_IN / 2, 300)
        assert max(out.shape[:2]) == pytest.approx(round(TEXT_IN / 2 * 300), abs=1)

    def test_fit_for_print_never_upsamples(self):
        a = np.zeros((80, 60, 3), np.uint8)
        assert fit_for_print(a, TEXT_IN / 2, 300).shape == a.shape

    def test_scale_bar_lengths_are_round(self):
        assert list(BAR_UM) == sorted(BAR_UM, reverse=True)
        assert all(int(u) == u for u in BAR_UM)


class TestKeptRegions:
    def _t(self):
        return pd.DataFrame({
            "wsi": ["HE_a"] * 3 + ["HE_b"] * 2,
            "region_index": [0, 1, 2, 0, 1],
            "y0": [0, 0, 2048, 0, 0], "y1": [2048] * 3 + [2048] * 2,
            "x0": [0, 2048, 0, 0, 2048], "x1": [2048, 4096, 2048, 2048, 4096],
            "wsi_h": [4096] * 5, "wsi_w": [4096] * 5})

    def test_selects_one_case_and_its_frame(self):
        g, h, w = kept_regions(self._t(), "HE_a")
        assert len(g) == 3 and (h, w) == (4096, 4096)

    def test_unknown_case_lists_what_is_present(self):
        with pytest.raises(SystemExit, match="HE_a"):
            kept_regions(self._t(), "HE_zzz")

    def test_missing_box_columns_named(self):
        t = self._t().drop(columns=["y0", "wsi_h"])
        with pytest.raises(SystemExit, match="wsi_h"):
            kept_regions(t, "HE_a")
