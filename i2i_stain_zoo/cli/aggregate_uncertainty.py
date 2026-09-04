"""Aggregate per-tile uncertainty maps to one CSV per WSI.

Reads per-tile [H, W] float32 .npy files from uncertainty.py (raw_npy/) and
computes the tissue-masked mean uncertainty for each tile. Saves one CSV per
WSI, where each row is one tile.

WSI membership is derived from the NNN/ subfolder in the npy path
(e.g. raw_npy/001/images/0000001.npy → WSI 001), then matched to the WSI
source filename via tiles_metadata/001/tiles_metadata.csv. This avoids the
collision problem that arises when tile IDs repeat across WSIs.

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
from tqdm import tqdm

from i2i_stain_zoo.cli.uncertainty import _build_mask_lookup, _find_tile_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wsi_source_map(tiles_metadata_root: Path) -> dict[str, str]:
    """Return {wsi_folder: wsi_stem} by reading source_file from each
    tiles_metadata/NNN/tiles_metadata.csv.  NNN is the zero-padded folder
    name (e.g. '001').
    """
    mapping: dict[str, str] = {}
    for csv_path in sorted(tiles_metadata_root.glob("*/tiles_metadata.csv")):
        wsi_folder = csv_path.parent.name  # e.g. "001"
        df = pd.read_csv(csv_path, usecols=["source_file"], nrows=1)
        if not df.empty:
            mapping[wsi_folder] = Path(df["source_file"].iloc[0]).stem
    return mapping


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

    # --- NNN folder → WSI source stem ---
    folder_to_wsi = _wsi_source_map(args.tiles_metadata)
    if not folder_to_wsi:
        raise RuntimeError(
            f"No tiles_metadata.csv files found under {args.tiles_metadata}"
        )
    print(f"Found {len(folder_to_wsi)} WSI(s): {sorted(folder_to_wsi)}")

    # --- pre-build mask stem lookup (same as uncertainty.py) ---
    mask_by_stem = _build_mask_lookup(args.mask_dir)

    npy_files = sorted(args.uncertainty_dir.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found under {args.uncertainty_dir}")

    # --- collect rows grouped by WSI ---
    wsi_rows: dict[str, list[dict]] = {}
    n_skipped = 0

    for npy_path in tqdm(npy_files, desc="Tiles"):
        # Derive WSI folder from path: raw_npy/NNN/images/stem.npy
        rel = npy_path.relative_to(args.uncertainty_dir)
        wsi_folder = rel.parts[0] if len(rel.parts) >= 2 else None
        wsi_stem = folder_to_wsi.get(wsi_folder) if wsi_folder else None
        if wsi_stem is None:
            continue

        u_map = np.load(npy_path).astype(np.float64)  # [H, W]

        # fname as NNN/images/stem.tif — used by _find_tile_mask for the
        # primary mask_dir/NNN/masks/ lookup (same logic as uncertainty.py)
        fname = str(rel.with_suffix(".tif"))

        if args.mask_dir is not None:
            mask = _find_tile_mask(
                tile_path=str(npy_path),
                mask_by_stem=mask_by_stem,
                mask_dir=args.mask_dir,
                fname=fname,
            )
            if mask is not None:
                if mask.shape != u_map.shape:
                    from PIL import Image
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize(
                        (u_map.shape[1], u_map.shape[0]), resample=Image.NEAREST
                    )
                    mask = np.array(mask_img).astype(bool)
                if mask.mean() < args.min_tissue_fraction:
                    n_skipped += 1
                    continue
                tile_mean = float(u_map[mask].mean())
            else:
                tile_mean = float(u_map.mean())
        else:
            tile_mean = float(u_map.mean())

        wsi_rows.setdefault(wsi_stem, []).append(
            {"tile_name": npy_path.stem, "mean_uncertainty": tile_mean}
        )

    if not wsi_rows:
        raise RuntimeError("No tiles were processed. Check paths and --tiles_metadata.")

    if n_skipped:
        print(f"Skipped {n_skipped} tiles below --min_tissue_fraction {args.min_tissue_fraction}")

    for wsi_stem, rows in sorted(wsi_rows.items()):
        df = pd.DataFrame(rows).sort_values("tile_name").reset_index(drop=True)
        out_csv = args.outdir / f"{wsi_stem}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  {wsi_stem}: {len(df)} tiles → {out_csv}")

    print(f"\nDone. {len(wsi_rows)} CSV files written to {args.outdir}")


if __name__ == "__main__":
    main()
