"""Aggregate per-tile uncertainty maps to one CSV per WSI.

Reads per-tile [H, W] float32 .npy files from uncertainty.py (raw_npy/) and
computes the tissue-masked mean uncertainty for each tile. Saves one CSV per
WSI, where each row is one tile.

Output per WSI: {outdir}/{wsi_stem}.csv
Columns: tile_name, mean_uncertainty

Example
-------
python aggregate_uncertainty.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --tiles_metadata  /path/to/tiles/testA \
    --mask_dir        /path/to/tiles/testA \
    --outdir          ./uncertainty_out/cyclegan/per_wsi_csv/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from uncertainty_calibration import build_stem_to_wsi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_mask(stem: str, mask_dir: Path) -> np.ndarray | None:
    """Return boolean tissue mask for a tile, or None if not found.

    Tries (in order):
      1. Flat file: mask_dir/{stem}.tif
      2. Subdir layout: mask_dir/NNN/masks/{stem}.tif
    """
    for ext in (".tif", ".tiff", ".png"):
        flat = mask_dir / f"{stem}{ext}"
        if flat.exists():
            arr = tifffile.imread(str(flat))
            return arr.astype(bool) if arr.ndim == 2 else arr[..., 0].astype(bool)

    for sub in sorted(mask_dir.glob("*/masks")):
        for ext in (".tif", ".tiff", ".png"):
            p = sub / f"{stem}{ext}"
            if p.exists():
                arr = tifffile.imread(str(p))
                return arr.astype(bool) if arr.ndim == 2 else arr[..., 0].astype(bool)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Save mean uncertainty per tile as one CSV per WSI."
    )
    ap.add_argument("--uncertainty_dir", type=Path, required=True,
                    help="Directory of <stem>.npy uncertainty maps (raw_npy/ from uncertainty.py).")
    ap.add_argument("--tiles_metadata", type=Path, required=True,
                    help="Dataset root containing per-WSI tiles_metadata.csv files.")
    ap.add_argument("--mask_dir", type=Path, default=None,
                    help="Directory of tissue masks (<stem>.tif or NNN/masks/<stem>.tif). "
                         "If omitted, all pixels are used.")
    ap.add_argument("--min_tissue_fraction", type=float, default=0.0,
                    help="Minimum fraction [0–1] of tissue pixels required to include a tile "
                         "(default: 0.0 = keep all tiles).")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Output directory; one {wsi_stem}.csv is written per WSI.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- tile stem → WSI source file ---
    stem_to_wsi = build_stem_to_wsi(args.tiles_metadata)
    if not stem_to_wsi:
        raise RuntimeError("No tile→WSI mapping found. Check --tiles_metadata path.")

    npy_files = sorted(args.uncertainty_dir.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found under {args.uncertainty_dir}")

    # --- collect rows grouped by WSI ---
    # wsi_stem -> list of {"tile_name": str, "mean_uncertainty": float}
    wsi_rows: dict[str, list[dict]] = {}

    for npy_path in tqdm(npy_files, desc="Tiles"):
        stem = npy_path.stem
        wsi = stem_to_wsi.get(stem)
        if wsi is None:
            continue

        u_map = np.load(npy_path).astype(np.float64)  # [H, W]

        if args.mask_dir is not None:
            mask = _find_mask(stem, args.mask_dir)
            if mask is not None:
                if mask.shape != u_map.shape:
                    from PIL import Image
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize(
                        (u_map.shape[1], u_map.shape[0]), resample=Image.NEAREST
                    )
                    mask = np.array(mask_img).astype(bool)
                if mask.mean() < args.min_tissue_fraction:
                    continue
                tile_mean = float(u_map[mask].mean())
            else:
                tile_mean = float(u_map.mean())
        else:
            tile_mean = float(u_map.mean())

        wsi_stem = Path(wsi).stem
        wsi_rows.setdefault(wsi_stem, []).append(
            {"tile_name": stem, "mean_uncertainty": tile_mean}
        )

    if not wsi_rows:
        raise RuntimeError("No tiles were processed. Check paths and --tiles_metadata.")

    for wsi_stem, rows in sorted(wsi_rows.items()):
        df = pd.DataFrame(rows).sort_values("tile_name").reset_index(drop=True)
        out_csv = args.outdir / f"{wsi_stem}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  {wsi_stem}: {len(df)} tiles → {out_csv}")

    print(f"\nDone. {len(wsi_rows)} CSV files written to {args.outdir}")


if __name__ == "__main__":
    main()
