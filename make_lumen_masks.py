#!/usr/bin/env python
"""Lumen masks from a stained WSI, inside the H&E tissue footprint.

A pipeline stage in the shape of segment → clean → fill: it turns whole-slide
RGB into a compact binary mask once, so `compute_phi_uncertainty.py` reads masks
rather than several GB of RGB per ensemble member.

    # virtual side — one run per member
    python make_lumen_masks.py \\
        --rgb_dir /path/ensemble/.../reconstructed/model_01 \\
        --he_masks /path/HE_tissue \\
        --white_thresh 0.65 --min_object_px 64 \\
        --outdir  /path/ensemble/.../lumen_masks/model_01

    # reference side — the real H&E thresholded against its own footprint
    python make_lumen_masks.py \\
        --rgb_dir /path/real_he_wsis --he_masks /path/HE_tissue \\
        --white_thresh 0.65 --min_object_px 64 --outdir /path/lumen_masks_real

Two decisions this stage encodes
-------------------------------
**The footprint comes from the H&E tissue mask, never from thresholding
`--rgb_dir`.** Two reasons. It is the same tissue boundary `apply_he_mask.py`
applies to the collagen masks, so the study carries one definition of tissue
rather than two. And it removes `white_thresh` from the denominator: the footprint
is exactly what breaks across the threshold sweep, so deriving it by thresholding
would leave an unstable parameter under every lumen density. The threshold then
affects only the lumen numerator.

**Small components are removed here, once, not per region.** Cleaning at region
scale would leave `lumen_fraction` measured on the raw mask while β₀/β₁ were
measured on a cleaned one — area and topology disagreeing about what a lumen is.
The parameter must be identical for the virtual and reference arms, the same rule
§5.4.4 imposes on the collagen mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from uncertainty_phi.descriptors import (
    WHITE_THRESH,
    clean_mask,
    lumen_mask,
    tissue_footprint_from_mask,
)
from uncertainty_phi.ensemble import _stem_index, load_rgb, load_roi_mask
from utils import write_label_mask


def main() -> None:
    ap = argparse.ArgumentParser("Lumen masks inside the H&E tissue footprint")
    ap.add_argument("--rgb_dir", type=Path, required=True,
                    help="WSIs whose whitespace is being read: a member's "
                         "reconstructed SR on the virtual side, the real H&E on "
                         "the reference side.")
    ap.add_argument("--he_masks", type=Path, required=True,
                    help="H&E tissue masks — the same ones apply_he_mask.py "
                         "applies to the collagen masks. Supplies the footprint: "
                         "the enclosure test and the density denominator. Matched "
                         "to --rgb_dir by stem, resized nearest-neighbour if it "
                         "was annotated at a different magnification.")
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--white_thresh", type=float, default=WHITE_THRESH,
                    help="Whitespace cut. EVERY channel must clear it, so set it "
                         "from the per-pixel channel minimum, not from a Fiji "
                         "8-bit grey level. [%(default)s]")
    ap.add_argument("--min_object_px", type=int, default=64,
                    help="Drop lumen components below this many pixels. At "
                         "0.221 um/px, 64 px is ~3.1 um2 — speckle, not lumen. "
                         "Raise it to exclude sinusoids: a 10 um capillary is "
                         "~1600 px. Must match between the two arms. [%(default)s]")
    ap.add_argument("--overwrite", action="store_true",
                    help="Redo slides that already have an output mask.")
    args = ap.parse_args()

    rgb_index = _stem_index(args.rgb_dir)
    he_index = _stem_index(args.he_masks)
    if not rgb_index:
        raise SystemExit(f"no images in {args.rgb_dir}")

    missing = sorted(set(rgb_index) - set(he_index))
    if missing:
        # An H&E-less slide has no footprint, so its lumen would be measured
        # against the whole canvas — a different quantity wearing the same name.
        print(f"[WARN] no H&E tissue mask for {len(missing)} slide(s), skipping: "
              f"{', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stats = []

    for stem in sorted(set(rgb_index) & set(he_index)):
        out_path = args.outdir / f"{stem}.tif"
        if out_path.is_file() and not args.overwrite:
            print(f"[skip] {stem}")
            continue

        rgb = load_rgb(rgb_index[stem])
        # load_roi_mask resizes nearest-neighbour, the same convention
        # apply_he_mask uses when the tissue mask was made at a different
        # magnification from the reconstruction.
        footprint = tissue_footprint_from_mask(
            load_roi_mask(he_index[stem], rgb.shape[:2])
        )

        lum, _ = lumen_mask(rgb, footprint, args.white_thresh)
        del rgb
        raw = int(np.count_nonzero(lum))
        if args.min_object_px > 1:
            lum = clean_mask(lum, args.min_object_px, 0)
        kept = int(np.count_nonzero(lum))

        write_label_mask(out_path, lum.astype(np.uint8))
        n_tissue = int(np.count_nonzero(footprint))
        stats.append({
            "wsi": stem,
            "lumen_fraction": kept / n_tissue if n_tissue else None,
            "removed_fraction_of_lumen": (raw - kept) / raw if raw else 0.0,
            "tissue_fraction": n_tissue / footprint.size,
        })
        print(f"{stem}: lumen {stats[-1]['lumen_fraction']:.5f}  "
              f"speckle removed {stats[-1]['removed_fraction_of_lumen'] * 100:.1f}% "
              f"of raw lumen area")

    with open(args.outdir / "lumen_masks.json", "w") as fh:
        json.dump({"per_wsi": stats,
                   "params": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()}}, fh, indent=2)

    print(f"\nwrote {len(stats)} mask(s) to {args.outdir}")
    if stats:
        removed = np.mean([s["removed_fraction_of_lumen"] for s in stats])
        print(f"--min_object_px {args.min_object_px} removed {removed * 100:.1f}% "
              f"of raw lumen area on average.")
        if removed > 0.5:
            print("[WARN] over half the lumen area was removed as speckle. Either "
                  "the threshold is catching noise, or min_object_px is eating "
                  "real structure — look at a mask before trusting it.")


if __name__ == "__main__":
    main()
