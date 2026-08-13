"""Fill holes in the tissue segmentation of PSR masks.

Enclosed background regions inside the tissue footprint (labels 1 + 2 combined)
are filled and assigned label 1 (tissue). PSR-positive pixels (label 2) and
existing tissue pixels (label 1) are not modified.

Why combine labels 1 and 2 before filling: PSR+ regions are always inside tissue.
Treating only label==1 as foreground would mark every PSR+ pixel as a "hole" and
incorrectly relabel them. Using the union of tissue and PSR+ as the foreground
fills only genuine enclosed background gaps.

Label convention (nnUNet Dataset214_SR):
  0 — background
  1 — tissue
  2 — PSR-positive

Usage
-----
# Directory of WSI mask TIFs
python fill_tissue_holes.py \
    --masks  ./psr_masks_wsi/ \
    --outdir ./psr_masks_filled/

# Single file
python fill_tissue_holes.py \
    --masks  ./psr_masks_wsi/slide_001.tif \
    --outdir ./psr_masks_filled/
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_fill_holes

from utils import write_label_mask


TIFF_EXTS = {".tif", ".tiff"}


def load_mask(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background holes in the tissue+PSR footprint.

    New pixels (were background, now inside filled region) are assigned label 1.
    Existing label 1 and label 2 pixels are unchanged.
    """
    foreground = (mask == 1) | (mask == 2)
    filled = binary_fill_holes(foreground)
    new_tissue = filled & ~foreground
    result = mask.copy()
    result[new_tissue] = 1
    return result


def collect_paths(arg: Path) -> list[Path]:
    if arg.is_file():
        return [arg]
    return sorted(p for p in arg.iterdir()
                  if p.is_file() and p.suffix.lower() in TIFF_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="Fill enclosed background holes inside the tissue footprint "
                    "of PSR segmentation masks. Filled pixels are assigned label 1 "
                    "(tissue); existing labels 0/1/2 are preserved elsewhere."
    )
    parser.add_argument("--masks", type=Path, required=True,
                        help="Directory of PSR mask TIFs, or a single TIF file.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for hole-filled masks.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = collect_paths(args.masks)
    if not paths:
        raise RuntimeError(f"No TIF files found in {args.masks}")

    for path in paths:
        mask   = load_mask(path)
        result = fill_holes(mask)

        n_filled = int(np.sum((result == 1) & (mask != 1)))
        if n_filled:
            print(f"{path.name}: filled {n_filled:,} background pixel(s)")

        write_label_mask(args.outdir / path.name, result.astype(np.uint8))

    print(f"Done — {len(paths)} mask(s) saved to {args.outdir}")


if __name__ == "__main__":
    main()
