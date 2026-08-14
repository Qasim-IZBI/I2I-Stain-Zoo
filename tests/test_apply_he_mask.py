"""Stem matching and masking in apply_he_mask.

The real-SR arm names its masks after the SR slides and the tissue masks after
the H&E ones, so pairing goes through --strip_prefix. A mismatch there skips
slides rather than crashing, which is why it is worth pinning down.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from apply_he_mask import apply_mask, index_he_masks, normalize_stem

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr.astype(np.uint8))


def test_normalize_stem_drops_first_token_only_when_asked():
    assert normalize_stem("SR_w10_BDL+A_M7", False) == "SR_w10_BDL+A_M7"
    assert normalize_stem("SR_w10_BDL+A_M7", True) == "w10_BDL+A_M7"
    assert normalize_stem("HE_w10_BDL+A_M7", True) == "w10_BDL+A_M7"
    # nothing to strip: left alone rather than emptied
    assert normalize_stem("slide", True) == "slide"


def test_index_matches_across_prefixes(tmp_path):
    he = tmp_path / "he"
    _write(he / "HE_slide_a.tif", np.ones((4, 4)))
    _write(he / "HE_slide_b.tif", np.ones((4, 4)))

    exact = index_he_masks(he)
    assert set(exact) == {"HE_slide_a", "HE_slide_b"}

    stripped = index_he_masks(he, strip_prefix=True)
    assert set(stripped) == {"slide_a", "slide_b"}
    assert stripped[normalize_stem("SR_slide_a", True)].name == "HE_slide_a.tif"


def test_colliding_keys_raise_rather_than_pick_one(tmp_path):
    he = tmp_path / "he"
    _write(he / "HE_slide.tif", np.ones((4, 4)))
    _write(he / "SR_slide.tif", np.ones((4, 4)))

    index_he_masks(he)  # distinct stems, fine
    with pytest.raises(RuntimeError, match="both match key"):
        index_he_masks(he, strip_prefix=True)


def test_apply_mask_zeroes_outside_tissue_and_keeps_labels():
    psr = np.array([[2, 2], [1, 2]], dtype=np.uint8)
    he = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    out = apply_mask(psr, he)
    assert out.tolist() == [[2, 0], [1, 0]]


def test_apply_mask_resizes_a_mismatched_he_mask():
    psr = np.zeros((8, 8), dtype=np.uint8) + 2
    he = np.array([[1, 0], [1, 0]], dtype=np.uint8)  # left half tissue
    out = apply_mask(psr, he)
    assert out[:, :4].all() and not out[:, 4:].any()


def test_cli_pairs_across_prefixes_and_fails_loudly_without(tmp_path):
    psr_dir, he_dir = tmp_path / "psr", tmp_path / "he"
    _write(psr_dir / "SR_slide.tif", np.full((4, 4), 2))
    _write(he_dir / "HE_slide.tif", np.array([[1, 1, 0, 0]] * 4))

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(REPO / "apply_he_mask.py"),
             "--psr_masks", str(psr_dir), "--he_masks", str(he_dir),
             "--outdir", str(tmp_path / "out"), *extra],
            capture_output=True, text=True, cwd=REPO,
        )

    # Without the flag nothing pairs. That must be an error: an empty output
    # directory otherwise reads like a completed stage to the SLURM skip guards.
    bad = run()
    assert bad.returncode != 0
    assert "--strip_prefix" in bad.stderr

    ok = run("--strip_prefix")
    assert ok.returncode == 0, ok.stderr
    out = tifffile.imread(str(tmp_path / "out" / "SR_slide.tif"))
    assert out.tolist() == [[2, 2, 0, 0]] * 4
