"""Apply HE tissue mask to PSR segmentation masks.

Zeros out any PSR mask pixel that falls outside the HE tissue region.
Pixels inside the HE tissue boundary keep their original label (0, 1, or 2).

Label convention (nnUNet Dataset214_SR):
  0 — background
  1 — tissue
  2 — PSR-positive

Usage
-----
# Directories matched by stem name
python apply_he_mask.py \
    --psr_masks  ./psr_masks_wsi/ \
    --he_masks   ./he_tissue_masks/ \
    --outdir     ./psr_masks_cleaned/

# Single-file pair
python apply_he_mask.py \
    --psr_masks  ./psr_masks_wsi/slide_001.tif \
    --he_masks   ./he_tissue_masks/slide_001.tif \
    --outdir     ./psr_masks_cleaned/
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from utils import write_label_mask


TIFF_EXTS = {".tif", ".tiff"}


def load_mask(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def find_he_mask(he_dir: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff"):
        p = he_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def apply_mask(psr: np.ndarray, he: np.ndarray) -> np.ndarray:
    """Zero out PSR mask pixels where HE tissue mask is background (0)."""
    tissue = he > 0
    if tissue.shape != psr.shape:
        tissue_img = Image.fromarray(tissue.astype(np.uint8) * 255).resize(
            (psr.shape[1], psr.shape[0]), Image.NEAREST
        )
        tissue = np.array(tissue_img) > 0
    cleaned = psr.copy()
    cleaned[~tissue] = 0
    return cleaned


def collect_psr_paths(psr_arg: Path) -> list[Path]:
    if psr_arg.is_file():
        return [psr_arg]
    return sorted(p for p in psr_arg.iterdir()
                  if p.is_file() and p.suffix.lower() in TIFF_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="Apply HE tissue mask to PSR segmentation masks to remove "
                    "spurious background predictions. Output keeps labels 0/1/2 "
                    "inside the HE tissue region; outside is forced to 0."
    )
    parser.add_argument("--psr_masks", type=Path, required=True,
                        help="Directory of PSR mask TIFs, or a single TIF file.")
    parser.add_argument("--he_masks", type=Path, required=True,
                        help="Directory of HE tissue mask TIFs (matched by stem), "
                             "or a single TIF file when --psr_masks is also a file.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for cleaned masks.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    psr_paths = collect_psr_paths(args.psr_masks)
    if not psr_paths:
        raise RuntimeError(f"No TIF files found in {args.psr_masks}")

    he_is_file = args.he_masks.is_file()

    n_ok = 0
    n_skip = 0
    for psr_path in psr_paths:
        if he_is_file:
            he_path = args.he_masks
        else:
            he_path = find_he_mask(args.he_masks, psr_path.stem)
            if he_path is None:
                print(f"[WARN] No HE mask found for '{psr_path.name}' — skipping.")
                n_skip += 1
                continue

        psr = load_mask(psr_path)
        he  = load_mask(he_path)

        if he.shape != psr.shape:
            print(f"[INFO] Resizing HE mask {he.shape} → {psr.shape} for '{psr_path.name}'")

        cleaned = apply_mask(psr, he)
        out_path = args.outdir / psr_path.name
        write_label_mask(out_path, cleaned.astype(np.uint8))
        n_ok += 1

    print(f"Done — {n_ok} mask(s) saved to {args.outdir}"
          + (f", {n_skip} skipped (no matching HE mask)." if n_skip else "."))


if __name__ == "__main__":
    main()
