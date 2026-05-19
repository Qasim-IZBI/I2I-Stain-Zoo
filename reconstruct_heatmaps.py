"""Reconstruct WSI-level heatmaps from per-tile .npy uncertainty or error maps.

Reads tiles_metadata.csv to get tile coordinates, loads per-tile .npy files from
--npy_dir, and stitches them into a full WSI-level float canvas using the same
weighted-average accumulation as reconstruct.py.  The canvas is then normalised
and colourised with a matplotlib colormap and saved as a TIFF.

Tile path layouts supported (tried in this order):
  1. npy_dir/<NNN>/images/<tile_name>.npy   (uncertainty.py nested output)
  2. npy_dir/<NNN>/<tile_name>.npy          (nested, no images/ subdir)
  3. npy_dir/<NNN>_images_<tile_name>.npy   (flat with collision-avoidance suffix)
  4. npy_dir/<tile_name>.npy                (flat — evaluation.py error_npy default)

Usage
-----
# Uncertainty heatmaps (magma colormap)
python reconstruct_heatmaps.py \\
    --metadata path/to/tiles/testA \\
    --npy_dir ./uncertainty_out/cyclegan/raw_npy/ \\
    --output  ./wsi_uncertainty/ \\
    --colormap magma --save_npy

# Regen-error heatmaps (hot colormap)
python reconstruct_heatmaps.py \\
    --metadata path/to/tiles/testA \\
    --npy_dir ./regen_error_out/error_npy/ \\
    --output  ./wsi_regen_error/ \\
    --colormap hot --save_npy

# Global normalization (consistent colour scale across all WSIs)
python reconstruct_heatmaps.py \\
    --metadata path/to/tiles/testA \\
    --npy_dir ./uncertainty_out/cyclegan/raw_npy/ \\
    --output  ./wsi_uncertainty/ \\
    --global_norm --colormap magma
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tissue mask helpers  (mirrors evaluation.py build_tissue_filter pattern)
# ---------------------------------------------------------------------------

def _auto_mask_path(img_path: str) -> Optional[str]:
    """Auto-detect mask by replacing the 'images' dir component with 'masks'."""
    parts = img_path.replace("\\", "/").split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "masks"
        candidate = "/".join(parts)
        return candidate if Path(candidate).is_file() else None
    except ValueError:
        return None


def _load_mask_array(mask_path: str, target_size: Optional[tuple] = None) -> np.ndarray:
    """Return a boolean (H, W) tissue mask; any non-zero pixel is tissue.
    Resizes to target_size (W, H) with nearest-neighbour if needed."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    mask = arr > 0
    if target_size is not None and mask.shape != (target_size[1], target_size[0]):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(target_size, Image.NEAREST)
        ) > 127
    return mask


def _build_mask_lookup(mask_dir: Optional[Path]) -> dict:
    """Walk mask_dir recursively and return stem → Path mapping."""
    if mask_dir is None:
        return {}
    exts = {".tif", ".tiff", ".png"}
    return {p.stem: p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts}


def _find_tile_mask(
    row: pd.Series,
    mask_by_stem: dict,
    tile_size: int,
) -> Optional[np.ndarray]:
    """Find and load the tissue mask for one metadata row.

    Priority (mirrors evaluation.py):
      1. mask_path column from tiles_metadata.csv
      2. Auto-detect images/ → masks/ on image_path
      3. mask_by_stem (from --mask_dir) keyed by tile_name stem
    Tiles with no mask are always included.
    """
    tile_name = f"{int(row['tile_name']):07d}"
    target = (tile_size, tile_size)

    # 1. mask_path from metadata
    mp = row.get("mask_path", None)
    if mp and isinstance(mp, str) and mp.strip() and Path(mp).is_file():
        return _load_mask_array(mp, target)

    # 2. auto-detect from image_path
    ip = row.get("image_path", None)
    if ip and isinstance(ip, str):
        auto = _auto_mask_path(ip)
        if auto is not None:
            return _load_mask_array(auto, target)

    # 3. explicit mask_dir by stem
    if tile_name in mask_by_stem:
        return _load_mask_array(str(mask_by_stem[tile_name]), target)

    return None


# ---------------------------------------------------------------------------
# Tile .npy path resolution
# ---------------------------------------------------------------------------

def _find_npy(npy_dir: Path, img_idx_str: Optional[str], tile_name: str) -> Optional[Path]:
    """Locate a per-tile .npy file, trying multiple directory layouts."""
    candidates = []
    if img_idx_str is not None:
        candidates += [
            npy_dir / img_idx_str / "images" / f"{tile_name}.npy",
            npy_dir / img_idx_str / f"{tile_name}.npy",
            npy_dir / f"{img_idx_str}_images_{tile_name}.npy",
        ]
    candidates.append(npy_dir / f"{tile_name}.npy")
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Canvas normalization
# ---------------------------------------------------------------------------

def _percentile_bounds(
    canvas: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> tuple[float, float]:
    """Compute percentile-based bounds, ignoring zero-weight (empty) pixels."""
    vals = canvas[canvas > 0]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, low_pct)), float(np.percentile(vals, high_pct))


def _normalise(canvas: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((canvas - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Colormap & output helpers
# ---------------------------------------------------------------------------

def _apply_colormap(norm: np.ndarray, cmap_name: str) -> np.ndarray:
    """Colourize a normalized [0, 1] canvas → uint8 RGB (H, W, 3)."""
    rgba = plt.get_cmap(cmap_name)(norm)          # (H, W, 4) float64 [0,1]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _save_colorbar_legend(
    path: Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str = "",
) -> None:
    """Save a small standalone colorbar legend PNG."""
    fig, ax = plt.subplots(figsize=(1.5, 4))
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label, fraction=1.0)
    ax.set_visible(False)
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Colorbar legend → {path}")


# ---------------------------------------------------------------------------
# Per-WSI canvas reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_canvas(
    group: pd.DataFrame,
    npy_dir: Path,
    blend: str,
    mask_by_stem: dict,
    min_tissue_fraction: float,
) -> tuple[np.ndarray, int]:
    """Stitch per-tile .npy files into a float32 WSI canvas.

    Non-tissue pixels are zeroed before accumulation (mirrors evaluation.py
    build_tissue_filter: tiles with no mask are always included).

    Returns (canvas, n_missing) where canvas is (H, W) float32 and n_missing
    is the number of tiles whose .npy file could not be located.
    """
    max_x = int((group["x"] + group["tile_size"]).max())
    max_y = int((group["y"] + group["tile_size"]).max())

    canvas = np.zeros((max_y, max_x), dtype=np.float32)
    weight = np.zeros((max_y, max_x), dtype=np.float32)
    n_missing = 0

    for _, row in group.iterrows():
        x, y = int(row["x"]), int(row["y"])
        tile_size = int(row["tile_size"])
        saved_size = (
            int(row["saved_size"])
            if "saved_size" in row.index and pd.notna(row["saved_size"])
            else tile_size
        )
        tile_name = f"{int(row['tile_name']):07d}"
        img_idx = row.get("img_idx", None)
        img_idx_str = (
            f"{int(img_idx):03d}"
            if img_idx is not None and pd.notna(img_idx)
            else None
        )

        npy_path = _find_npy(npy_dir, img_idx_str, tile_name)
        if npy_path is None:
            n_missing += 1
            continue

        arr = np.load(npy_path).astype(np.float32)

        # If tiling used --resize_to, the .npy was saved at saved_size but the
        # coordinate space uses tile_size — resize back before placement.
        if arr.shape != (tile_size, tile_size):
            arr = np.array(
                Image.fromarray(arr, mode="F").resize(
                    (tile_size, tile_size), Image.BILINEAR
                )
            )

        # Tissue masking: zero non-tissue pixels; skip sub-threshold tiles.
        # Tiles with no mask are always included (evaluation.py convention).
        mask_arr = _find_tile_mask(row, mask_by_stem, tile_size)
        if mask_arr is not None:
            if float(mask_arr.mean()) < min_tissue_fraction:
                continue
            arr[~mask_arr] = 0.0

        canvas[y : y + tile_size, x : x + tile_size] += arr
        weight[y : y + tile_size, x : x + tile_size] += 1

    if blend == "average":
        np.divide(canvas, weight, out=canvas, where=weight > 0)

    return canvas, n_missing


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _load_metadata(metadata_path: Path, data_range: Optional[tuple]) -> pd.DataFrame:
    if metadata_path.is_dir():
        csv_files = sorted(
            metadata_path.glob("*/tiles_metadata.csv"),
            key=lambda p: int(p.parent.name),
        )
        if data_range is not None:
            csv_files = [
                p for p in csv_files
                if data_range[0] <= int(p.parent.name) <= data_range[1]
            ]
        if not csv_files:
            raise FileNotFoundError(
                f"No per-WSI tiles_metadata.csv files found under: {metadata_path}\n"
                f"Expected layout: {metadata_path}/<NNN>/tiles_metadata.csv"
            )
        return pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    return pd.read_csv(metadata_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct WSI-level heatmaps from per-tile .npy maps."
    )
    parser.add_argument("--metadata", required=True,
                        help="Dataset root with per-WSI tiles_metadata.csv files, or a single CSV")
    parser.add_argument("--npy_dir", required=True,
                        help="Directory containing per-tile .npy files "
                             "(e.g. uncertainty_out/cyclegan/raw_npy/ or error_npy/)")
    parser.add_argument("--output", required=True,
                        help="Output directory for reconstructed WSI heatmaps")
    parser.add_argument("--colormap", default="magma",
                        help="Matplotlib colormap name (default: magma). "
                             "Use 'hot' for regen-error maps")
    parser.add_argument("--blend", choices=["average", "overwrite"], default="average",
                        help="Overlap blending mode (default: average)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Fixed lower bound for colormap normalization (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed upper bound for colormap normalization (default: auto)")
    parser.add_argument("--percentile_low", type=float, default=1.0,
                        help="Lower percentile for auto normalization (default: 1)")
    parser.add_argument("--percentile_high", type=float, default=99.0,
                        help="Upper percentile for auto normalization (default: 99)")
    parser.add_argument("--global_norm", action="store_true",
                        help="Compute vmin/vmax globally across all WSIs (default: per-WSI). "
                             "Enables a meaningful shared colorbar legend.")
    parser.add_argument("--save_npy", action="store_true",
                        help="Also save raw float32 WSI canvas as <stem>.npy")
    parser.add_argument("--data_range", type=str, default=None,
                        help="Limit to WSI folders START,END inclusive (e.g. '1,5')")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory of tissue mask TIFs (walked recursively, matched by "
                             "tile_name stem). If omitted, masks are auto-detected from the "
                             "tile structure (images/ → masks/ sibling) or from the "
                             "mask_path column in tiles_metadata.csv. "
                             "Tiles with no matching mask are always included.")
    parser.add_argument("--min_tissue_fraction", type=float, default=0.0,
                        help="Minimum tissue fraction [0–1] to include a tile (default 0 = all "
                             "tiles). Set >0 to skip background tiles (e.g. 0.1).")
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_range = None
    if args.data_range:
        start, end = args.data_range.split(",")
        data_range = (int(start), int(end))

    mask_by_stem = _build_mask_lookup(Path(args.mask_dir) if args.mask_dir else None)
    min_tissue_fraction = args.min_tissue_fraction

    df = _load_metadata(Path(args.metadata), data_range)
    wsis = list(df.groupby("source_file"))
    print(f"Loaded metadata: {len(df)} tiles across {len(wsis)} WSIs")

    # --- Optional first pass: reconstruct all canvases to compute global bounds ---
    if args.global_norm and args.vmin is None and args.vmax is None:
        print("Global normalization: reconstruction pass 1/2 ...")
        canvases: dict[str, np.ndarray] = {}
        for source_file, group in tqdm(wsis, desc="Pass 1"):
            canvas, _ = _reconstruct_canvas(group, npy_dir, args.blend, mask_by_stem, min_tissue_fraction)
            canvases[source_file] = canvas

        all_vals = np.concatenate([c.ravel() for c in canvases.values()])
        nonzero = all_vals[all_vals > 0]
        if nonzero.size > 0:
            global_vmin = float(np.percentile(nonzero, args.percentile_low))
            global_vmax = float(np.percentile(nonzero, args.percentile_high))
        else:
            global_vmin, global_vmax = 0.0, 1.0
        print(f"  Global bounds: vmin={global_vmin:.4g}, vmax={global_vmax:.4g}")
    else:
        canvases = None
        global_vmin = args.vmin
        global_vmax = args.vmax

    # Determine colorbar legend bounds (only saved when a global scale exists)
    if args.vmin is not None and args.vmax is not None:
        legend_vmin, legend_vmax = args.vmin, args.vmax
    elif args.global_norm and canvases is not None:
        legend_vmin, legend_vmax = global_vmin, global_vmax  # type: ignore[assignment]
    else:
        legend_vmin, legend_vmax = None, None  # per-WSI: no single shared scale

    if legend_vmin is not None and legend_vmax is not None:
        label = "uncertainty" if args.colormap == "magma" else "MAE [0–255]"
        _save_colorbar_legend(
            output_dir / "colorbar_legend.png",
            args.colormap, legend_vmin, legend_vmax, label=label,
        )

    # --- Render and save each WSI ---
    pass_label = "Pass 2" if canvases is not None else "Reconstruction"
    n_total_missing = 0

    for source_file, group in tqdm(wsis, desc=pass_label):
        stem = Path(source_file).stem

        if canvases is not None:
            canvas = canvases[source_file]
            n_missing = 0
        else:
            canvas, n_missing = _reconstruct_canvas(group, npy_dir, args.blend, mask_by_stem, min_tissue_fraction)

        n_total_missing += n_missing
        if n_missing:
            tqdm.write(f"  {source_file}: {n_missing}/{len(group)} tiles not found in npy_dir")

        # Determine normalization bounds for this WSI
        if args.vmin is not None and args.vmax is not None:
            _vmin, _vmax = args.vmin, args.vmax
        elif args.global_norm and global_vmin is not None and global_vmax is not None:
            _vmin, _vmax = global_vmin, global_vmax  # type: ignore[assignment]
        else:
            _vmin, _vmax = _percentile_bounds(canvas, args.percentile_low, args.percentile_high)

        norm_canvas = _normalise(canvas, _vmin, _vmax)
        rgb = _apply_colormap(norm_canvas, args.colormap)

        Image.fromarray(rgb).save(output_dir / f"{stem}_heatmap.tif")

        if args.save_npy:
            np.save(output_dir / f"{stem}.npy", canvas)

    if n_total_missing:
        print(f"\nWarning: {n_total_missing} tiles total had no matching .npy file.")
    print(f"\nDone. {len(wsis)} WSI heatmap(s) saved → {output_dir}")


if __name__ == "__main__":
    main()
