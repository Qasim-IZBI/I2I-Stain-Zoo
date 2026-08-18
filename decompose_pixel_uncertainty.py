#!/usr/bin/env python
"""Per-pixel uncertainty split into procedural and data-exposure components.

`uncertainty.py` takes the spread across the members of ONE ensemble, which on
the crossed grid means one training subset — so it measures **procedural**
uncertainty and cannot see the other half. This reads all K subsets at once and
decomposes each pixel by the law of total variance:

    procedural     spread between seeds, within a subset
    data_exposure  spread between subsets, i.e. WHICH slides were seen
    total          their sum

so the three feed `uncertainty_calibration.py` unchanged and can be scored
against the same cycle-reconstruction error.

    python decompose_pixel_uncertainty.py \\
        --fold /path/ensemble/data_001_007/model_small/inference \\
        --fold /path/ensemble/data_008_014/model_small/inference \\
        ... (all five) \\
        --data_range 1,1 --output ./pixel_components/

Estimator
---------
One-way ANOVA components of variance, per channel, matching
`uncertainty_phi/decompose.py` exactly so the pixel and descriptor answers are
the same quantity at two scales. Two corrections, both of which bias the naive
split low:

* within-subset variance uses **ddof=1**; the population form under-estimates by
  (S−1)/S, which is 10% at S=10.
* the spread of subset means is **contaminated by procedural noise**, since each
  subset mean is itself an average of S noisy members:
  Var(subset means) = σ²_data + σ²_proc / S. Subtracting σ²_proc / n₀ recovers
  the data component; without it procedural leaks into data.

`n₀` is the ANOVA effective group size, equal to S for a balanced design.

Conventions that make the output drop-in compatible
---------------------------------------------------
Components are decomposed **per channel and summed**, then square-rooted — the
same √(Σ per-channel variance) in 0–255 intensity units that `uncertainty.py`
writes to `raw_npy/`. The additive identity therefore holds in VARIANCE, not in
SD: σ_total² = σ_proc² + σ_data², so σ_total < σ_proc + σ_data.

The data component may come out **negative** where the true between-subset
variance is near zero. It is reported, not clipped, for the same reason as in
descriptor space — clipping biases the budget and hides the "no signal here"
outcome. Its SD map stores NaN there, and `summary.json` counts how many pixels
that was, because a component estimated as zero over much of the slide is a
finding rather than a defect.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from uncertainty import (
    discover_common_filenames,
    discover_ensemble_dirs,
    read_rgb_tiff,
)

COMPONENTS = ("total", "procedural", "data_exposure")


def decompose_stack(folds: List[np.ndarray]) -> dict:
    """ANOVA components for one tile.

    Parameters
    ----------
    folds : list of length K; element f is (S_f, H, W, 3) member predictions.

    Returns dict of (H, W) VARIANCE maps, summed over channels.
    """
    counts = np.array([f.shape[0] for f in folds], dtype=np.float64)
    if len(folds) < 2:
        raise ValueError(
            "at least two --fold directories are needed: with one subset there "
            "is no between-subset term, which is what uncertainty.py already does"
        )

    fold_means = np.stack([f.mean(axis=0) for f in folds], axis=0)   # (K,H,W,3)
    fold_vars = np.stack(
        [f.var(axis=0, ddof=1) if f.shape[0] > 1
         else np.full(f.shape[1:], np.nan, np.float64) for f in folds], axis=0)

    procedural = np.nanmean(fold_vars, axis=0)                       # (H,W,3)

    total_n = counts.sum()
    n0 = (total_n - (counts ** 2).sum() / total_n) / (len(folds) - 1)
    between = fold_means.var(axis=0, ddof=1)                         # (H,W,3)
    data = between - procedural / n0                                 # may be < 0

    return {
        "procedural": procedural.sum(axis=2),
        "data_exposure": data.sum(axis=2),
        "total": (procedural + data).sum(axis=2),
        "grand_mean": fold_means.mean(axis=0),
        "n0": float(n0),
    }


def to_sd(var_map: np.ndarray) -> np.ndarray:
    """Variance -> SD, leaving NaN where the component came out negative.

    A negative variance has no square root. NaN says "not defined here" where a
    clip to zero would say "no uncertainty here", and those are opposite claims.
    """
    out = np.full(var_map.shape, np.nan, np.float32)
    ok = var_map >= 0
    out[ok] = np.sqrt(var_map[ok])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        "Per-pixel procedural / data-exposure uncertainty over a crossed grid")
    ap.add_argument("--fold", type=Path, action="append", required=True,
                    help="One inference directory per training subset, each "
                         "holding model_NN/ member dirs. Repeat once per subset; "
                         "at least two are needed for a data-exposure term.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data_range", type=str, default=None,
                    help="START,END WSI folders, as uncertainty.py takes it. "
                         "One WSI per job keeps memory flat and the summary "
                         "per-WSI, so parallel jobs cannot race on it.")
    ap.add_argument("--save_mean_rgb", action="store_true",
                    help="Also write the grand mean over all members.")
    args = ap.parse_args()

    rng = None
    if args.data_range:
        a, b = args.data_range.split(",")
        rng = (int(a), int(b))

    fold_dirs = [discover_ensemble_dirs(f) for f in args.fold]
    for f, dirs in zip(args.fold, fold_dirs):
        if not dirs:
            raise SystemExit(f"no model_* directories under {f}")
    sizes = [len(d) for d in fold_dirs]
    print(f"[1/3] {len(fold_dirs)} subsets, members per subset: {sizes}")
    if len(set(sizes)) > 1:
        # Not fatal — n0 handles it — but it changes the effective group size,
        # so it must be visible rather than absorbed silently.
        print(f"[WARN] unbalanced design; n0 will differ from {sizes[0]}")

    # Filenames must be common to EVERY member of EVERY subset, or a pixel would
    # be decomposed over a different member set than its neighbour.
    flat = [d for dirs in fold_dirs for d in dirs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = discover_common_filenames(flat, data_range=rng)
    if not names:
        raise SystemExit(
            "no tile is present in every member of every subset — check "
            "--data_range against what the inference actually covered"
        )
    print(f"[2/3] {len(names)} tile(s) common to all {len(flat)} members")

    out_dirs = {c: args.output / c / "raw_npy" for c in COMPONENTS}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    mean_dir = args.output / "mean_rgb"
    if args.save_mean_rgb:
        mean_dir.mkdir(parents=True, exist_ok=True)

    neg_px = 0
    tot_px = 0
    n0_seen: Optional[float] = None
    for i, rel in enumerate(names, 1):
        folds = [np.stack([read_rgb_tiff(d / rel).astype(np.float64)
                           for d in dirs], axis=0) for dirs in fold_dirs]
        res = decompose_stack(folds)
        n0_seen = res["n0"]

        rel_stem = Path(rel).with_suffix("")
        for c in COMPONENTS:
            path = out_dirs[c] / rel_stem.with_suffix(".npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), to_sd(res[c]))
        neg_px += int((res["data_exposure"] < 0).sum())
        tot_px += int(res["data_exposure"].size)

        if args.save_mean_rgb:
            import tifffile
            path = mean_dir / rel_stem.with_suffix(".tif")
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path),
                             np.clip(res["grand_mean"], 0, 255).astype(np.uint8))
        if i % 200 == 0:
            print(f"   {i}/{len(names)}")

    summary = {
        "n_tiles": len(names),
        "n_subsets": len(fold_dirs),
        "members_per_subset": sizes,
        "n0": n0_seen,
        "data_range": args.data_range,
        # A component estimated as zero over much of the slide is a finding, not
        # a defect, so it is counted rather than hidden by the clip.
        "negative_data_pixels": neg_px,
        "negative_data_fraction": (neg_px / tot_px) if tot_px else None,
        "convention": {
            "value": "sqrt(sum over channels of the variance component), "
                     "0-255 intensity units — same as uncertainty.py raw_npy/",
            "additivity": "holds in VARIANCE: total^2 = procedural^2 + data^2",
            "negative_data": "NaN in the SD map; not clipped to zero",
        },
        "folds": [str(f) for f in args.fold],
    }
    name = (f"summary_wsi{rng[0]:03d}.json"
            if rng and rng[0] == rng[1] else "summary.json")
    with open(args.output / name, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[3/3] wrote {len(names)} tile(s) x {len(COMPONENTS)} components")
    for c in COMPONENTS:
        print(f"   {args.output / c / 'raw_npy'}")
    if neg_px:
        print(f"\n[note] data_exposure was negative at "
              f"{neg_px / tot_px:.1%} of pixels and is NaN there. Near-zero "
              f"between-subset variance estimates negative about half the time, "
              f"so a large fraction means the subsets barely differ — a result, "
              f"not an error.")


if __name__ == "__main__":
    main()
