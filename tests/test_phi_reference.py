"""The reference half: phi of the real tissue, and its handoff to calibration.

Split from the calibration itself so measuring tissue happens once. These tests
cover what that split has to guarantee — that a reference is matched to the right
slides, cut from the right frame, and provably belongs to the grid it is reused
on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
        from uncertainty_phi.reference import _indexed
        d = self._dir(tmp_path, "psr", "SR", self.NAMES)
        assert set(_indexed(d, False, "x")) == {f"SR_{n}" for n in self.NAMES}
        assert set(_indexed(d, True, "x")) == set(self.NAMES)

    def test_he_still_matches_itself_when_stripping(self, tmp_path):
        """Both sides are keyed the same way, so turning it on must not break the
        arm that already matched."""
        from uncertainty_phi.reference import _indexed
        from apply_he_mask import normalize_stem
        d = self._dir(tmp_path, "he", "HE", self.NAMES)
        keys = set(_indexed(d, True, "x"))
        assert keys == {normalize_stem(f"HE_{n}", True) for n in self.NAMES}
        assert keys == set(self.NAMES)

    def test_a_collision_is_fatal(self, tmp_path):
        """SR_x and HE_x in one directory collapse to 'x'. Picking either scores
        one slide's regions against another slide's tissue, which is invisible in
        the output — so it is refused."""
        from uncertainty_phi.reference import _indexed
        d = tmp_path / "mixed"
        d.mkdir()
        from utils import write_label_mask
        for p in ("SR", "HE"):
            write_label_mask(d / f"{p}_slide.tif", np.ones((8, 8), np.uint8))
        with pytest.raises(SystemExit, match="collapses two files"):
            _indexed(d, True, "--real_psr")


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
            [sys.executable, "compute_phi_reference.py", "--phi_csv", str(csv),
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
        # specifically the frame-slack note; the stage prints others
        assert "untruncated original" not in r.stdout

    def test_tile_size_must_match_the_tiling(self, tmp_path):
        """The slack is only benign up to one tile, so an understated --tile_size
        turns a fine run into a refusal rather than the reverse."""
        r = self._run(tmp_path, (2048 + 391, 3072 + 453), (2048, 3072),
                      extra=("--tile_size", "256"))
        assert r.returncode != 0
        assert "--tile_size must match the tiling" in r.stdout + r.stderr


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
        from uncertainty_phi.reference import load_reference, save_reference
        df = self._grid()
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(df), path, self._args(tmp_path))
        back, meta = load_reference(path, df)
        assert len(back) == len(df)
        assert "real_task_specific_value" in back.columns
        assert meta["params"]["white_thresh"] == 0.85   # provenance survives

    def test_provenance_travels_with_the_reference(self):
        """The parameter check moved to compute_phi_reference, which compares
        against the phi run. What load_reference owes the caller is the record,
        so a result carries a trace of the thresholds behind its target."""
        from uncertainty_phi.reference import REFERENCE_PARAMS
        assert {"mpp", "min_object_px", "closing_px", "white_thresh",
                "real_psr", "he_masks", "strip_prefix"} <= set(REFERENCE_PARAMS)

    def test_a_regrid_is_refused_even_though_ids_match(self, tmp_path):
        """The hole a parameter check alone leaves: --region_px 2048 against a
        cache built at 1024 keeps every parameter identical, and region 7 of
        slide 3 exists in both — on different tissue."""
        from uncertainty_phi.reference import load_reference, save_reference
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(self._grid(region=1024)), path,
                       self._args(tmp_path))
        with pytest.raises(SystemExit, match="DIFFERENT boxes"):
            load_reference(path, self._grid(region=2048))

    def test_a_short_cache_is_refused(self, tmp_path):
        from uncertainty_phi.reference import load_reference, save_reference
        path = tmp_path / "reference_phi.csv"
        save_reference(self._ref(self._grid(n_wsi=1)), path, self._args(tmp_path))
        with pytest.raises(SystemExit, match="covers 4 of the 8 regions"):
            load_reference(path, self._grid(n_wsi=2))

    def test_a_cache_without_boxes_is_refused(self, tmp_path):
        """Predates --save_reference, so nothing about it can be checked."""
        from uncertainty_phi.reference import load_reference
        df = self._grid()
        path = tmp_path / "old.csv"
        self._ref(df).drop(columns=["y0", "y1", "x0", "x1"]).to_csv(path, index=False)
        with pytest.raises(SystemExit, match="no y0/y1/x0/x1"):
            load_reference(path, df)
