"""Compute globally normalised epistemic uncertainty maps from deep ensemble outputs.

Given ensemble predictions from one or more image-translation architectures,
this script:
  1. Stacks RGB predictions per filename across ensemble members.
  2. Computes per-pixel sample variance (ddof=1) per channel.
  3. Reduces to a scalar uncertainty map: U = var_R + var_G + var_B.
  4. Optionally applies log1p compression.
  5. Computes global normalisation bounds (percentile-based) per architecture.
  6. Saves raw / normalised .npy maps, heatmap PNGs, optional overlays, and
     a summary JSON with per-image statistics.

Example usage
-------------
# Basic
python uncertainty.py --model cyclegan --data /path/to/cyclegan/output --output ./uncertainty_out

# With log compression and overlays
python uncertainty.py \\
    --model cyclegan \\
    --data /path/to/cyclegan/output \\
    --log-compress \\
    --overlays \\
    --lower-percentile 1 \\
    --upper-percentile 99 \\
    --output ./uncertainty_out
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Tissue mask helpers  (mirrors evaluation.py build_tissue_filter pattern)
# ---------------------------------------------------------------------------

def _auto_mask_path(tile_path: str) -> Optional[str]:
    """Auto-detect mask by replacing the 'images' dir component with 'masks'."""
    parts = tile_path.replace("\\", "/").split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "masks"
        candidate = "/".join(parts)
        return candidate if Path(candidate).is_file() else None
    except ValueError:
        return None


def _load_mask_array(mask_path: str) -> np.ndarray:
    """Return a boolean (H, W) tissue mask; any non-zero pixel is tissue."""
    arr = tifffile.imread(mask_path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr > 0


def _build_mask_lookup(mask_dir: Optional[Path]) -> dict:
    """Walk mask_dir recursively and return stem → Path mapping."""
    if mask_dir is None:
        return {}
    exts = {".tif", ".tiff", ".png"}
    return {p.stem: p for p in mask_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts}


def _find_tile_mask(
    tile_path: str,
    mask_by_stem: dict,
    mask_dir: Optional[Path] = None,
    fname: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Find and load the tissue mask for a tile.

    Tries (in order):
      1. mask_dir + fname with images/ replaced by masks/  (e.g. mask_dir/001/masks/0000001.tif)
         This handles multi-WSI runs where tile IDs repeat across WSIs — the WSI
         subfolder in fname disambiguates them.
      2. mask_by_stem keyed by tile basename stem (flat dir, single-WSI safe)
      3. Auto-detect images/ → masks/ sibling on the tile path
    Returns a boolean (H, W) array, or None if no mask found.
    Tiles with no mask are always included (backward-compatible with evaluation.py).
    """
    # 1. Path-based via fname structure (mask_dir/NNN/masks/tile.tif)
    if mask_dir is not None and fname is not None:
        parts = fname.replace("\\", "/").split("/")
        try:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "masks"
            candidate = mask_dir / Path("/".join(parts))
            if candidate.is_file():
                return _load_mask_array(str(candidate))
        except ValueError:
            pass
    # 2. Flat stem lookup from mask_dir (single-WSI or flat mask dirs)
    stem = Path(tile_path).stem
    if stem in mask_by_stem:
        return _load_mask_array(str(mask_by_stem[stem]))
    # 3. Auto-detect images/ → masks/ sibling on the tile path itself
    auto = _auto_mask_path(tile_path)
    if auto is not None:
        return _load_mask_array(auto)
    return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def discover_ensemble_dirs(root: Path) -> list[Path]:
    """Return sorted list of ensemble member directories (e.g. model_01 … model_10)."""
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    )
    return dirs


def discover_common_filenames(
    ensemble_dirs: list[Path],
    data_range: tuple[int, int] | None = None,
) -> list[str]:
    """Return sorted relative paths present in *all* ensemble directories.

    Two layouts are supported:
    - Flat:   member_dir/*.tif  (legacy)
    - Subdir: member_dir/{NNN}/images/*.tif  (inference.py default output)

    When *data_range* is given only the specified WSI folders are walked.
    Without it the layout is auto-detected: flat files are preferred; if none
    are found the subdir layout is assumed and all numbered WSI folders are
    walked.

    Relative paths (e.g. "001/images/0000001.tif") are used as keys so that
    tiles from different WSIs with identical basenames do not collide.
    """
    TIFF_EXTS = {".tif", ".tiff"}
    sets: list[set[str]] = []

    for d in ensemble_dirs:
        names: set[str] = set()

        if data_range is not None:
            # Explicit range: walk only the requested WSI subdirs.
            for i in range(data_range[0], data_range[1] + 1):
                img_dir = d / f"{i:03d}" / "images"
                if img_dir.is_dir():
                    for f in img_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in TIFF_EXTS:
                            names.add(str(f.relative_to(d)))
        else:
            # Auto-detect layout.
            flat = [f for f in d.iterdir()
                    if f.is_file() and f.suffix.lower() in TIFF_EXTS]
            if flat:
                # Flat layout — use basename as key (backward-compatible).
                names = {f.name for f in flat}
            else:
                # Subdir layout — walk all {NNN}/images/ directories.
                for img_dir in sorted(d.glob("*/images")):
                    if img_dir.is_dir():
                        for f in img_dir.iterdir():
                            if f.is_file() and f.suffix.lower() in TIFF_EXTS:
                                names.add(str(f.relative_to(d)))

        sets.append(names)

    if not sets:
        return []

    common = sorted(set.intersection(*sets))
    all_names = sorted(set.union(*sets))

    skipped = set(all_names) - set(common)
    if skipped:
        warnings.warn(
            f"Skipping {len(skipped)} path(s) not present in all ensemble "
            f"directories: {sorted(skipped)[:5]}{'…' if len(skipped) > 5 else ''}"
        )

    return common


def read_rgb_tiff(path: Path) -> np.ndarray:
    """Read a TIFF image and return it as float32 array with shape (H, W, 3)."""
    img = tifffile.imread(str(path))
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {img.shape} for {path}"
        )
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------

def compute_uncertainty_map(stack: np.ndarray) -> np.ndarray:
    """Compute per-pixel epistemic uncertainty from an ensemble stack.

    Parameters
    ----------
    stack : np.ndarray
        Shape (N, H, W, 3) — N ensemble RGB predictions.

    Returns
    -------
    np.ndarray
        Shape (H, W) — scalar uncertainty map (sqrt of summed per-channel
        variances), in pixel units matching the regen-error MAE scale.
    """
    # Per-channel sample variance across ensemble axis (ddof=1)
    channel_var = np.var(stack, axis=0, ddof=1)  # (H, W, 3)
    # sqrt(sum of variances) → pixel units, comparable to regen-error MAE
    return np.sqrt(channel_var.sum(axis=2))  # (H, W)


def compute_global_bounds(
    uncertainty_maps: list[np.ndarray],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> tuple[float, float]:
    """Compute global lower/upper normalisation bounds from all pixels."""
    all_pixels = np.concatenate([u.ravel() for u in uncertainty_maps])
    low = float(np.percentile(all_pixels, lower_pct))
    high = float(np.percentile(all_pixels, upper_pct))
    return low, high


def normalise(u: np.ndarray, low: float, high: float, eps: float = 1e-8) -> np.ndarray:
    """Clip-normalise an uncertainty map to [0, 1] using global bounds."""
    return np.clip((u - low) / (high - low + eps), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

CMAP = "magma"


def save_heatmap(u_norm: np.ndarray, path: Path) -> None:
    """Save a normalised uncertainty map as a heatmap PNG with a colourbar."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(u_norm, cmap=CMAP, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overlay(rgb: np.ndarray, u_norm: np.ndarray, path: Path, alpha: float = 0.5) -> None:
    """Overlay a heatmap on the reference RGB image and save as PNG."""
    # Normalise RGB to [0, 1] if needed
    if rgb.max() > 1.0:
        rgb_vis = rgb / 255.0 if rgb.max() > 1.0 else rgb.copy()
    else:
        rgb_vis = rgb.copy()
    rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

    cmap = plt.get_cmap(CMAP)
    heat = cmap(u_norm)[..., :3]  # (H, W, 3)
    blended = (1 - alpha) * rgb_vis + alpha * heat

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(np.clip(blended, 0, 1))
    ax.set_axis_off()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image statistics
# ---------------------------------------------------------------------------

def image_stats(u: np.ndarray) -> dict:
    """Return summary statistics for a single uncertainty map."""
    return {
        "mean": float(np.mean(u)),
        "median": float(np.median(u)),
        "p95": float(np.percentile(u, 95)),
        "p99": float(np.percentile(u, 99)),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_architecture(
    name: str,
    root: Path,
    output_root: Path,
    log_compress: bool,
    lower_pct: float,
    upper_pct: float,
    save_overlays: bool,
    data_range: tuple[int, int] | None = None,
    mask_dir: Optional[Path] = None,
    min_tissue_fraction: float = 0.0,
) -> None:
    """Full uncertainty pipeline for one architecture."""
    print(f"\n{'=' * 60}")
    print(f"Architecture: {name}")
    print(f"Ensemble root: {root}")
    if data_range is not None:
        print(f"WSI range: {data_range[0]:03d} – {data_range[1]:03d}")
    print(f"{'=' * 60}")

    # --- discover ensemble members ---
    ensemble_dirs = discover_ensemble_dirs(root)
    if len(ensemble_dirs) < 2:
        raise RuntimeError(
            f"Need at least 2 ensemble directories in {root}, "
            f"found {len(ensemble_dirs)}: {[d.name for d in ensemble_dirs]}"
        )
    print(f"Found {len(ensemble_dirs)} ensemble members: "
          f"{ensemble_dirs[0].name} … {ensemble_dirs[-1].name}")

    filenames = discover_common_filenames(ensemble_dirs, data_range=data_range)
    if not filenames:
        raise RuntimeError(f"No common TIFF paths across ensemble dirs in {root}")
    print(f"Processing {len(filenames)} common image(s)")

    # --- create output dirs ---
    out = output_root / name
    dirs = {
        "raw_npy":  out / "raw_npy",
        "norm_npy": out / "norm_npy",
        "heatmaps": out / "heatmaps",
        "mean_rgb": out / "mean_rgb",
    }
    if save_overlays:
        dirs["overlays"] = out / "overlays"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- build mask lookup once (empty dict = no masking) ---
    mask_by_stem = _build_mask_lookup(mask_dir)
    apply_masking = mask_dir is not None or min_tissue_fraction > 0.0
    n_skipped = 0

    # --- pass 1: compute raw uncertainty maps and mean prediction ---
    raw_maps:  dict[str, np.ndarray] = {}
    mean_maps: dict[str, np.ndarray] = {}
    ref_images: dict[str, np.ndarray] = {}  # first ensemble member for overlays

    for fname in filenames:
        stack = []
        for edir in ensemble_dirs:
            img = read_rgb_tiff(edir / fname)
            stack.append(img)
        stack = np.stack(stack, axis=0)  # (N, H, W, 3)

        u = compute_uncertainty_map(stack)
        if log_compress:
            u = np.log1p(u)

        # Apply tissue mask: zero non-tissue pixels; skip sub-threshold tiles.
        # Matches evaluation.py build_tissue_filter: tiles with no mask are included.
        if apply_masking:
            tile_path = str(ensemble_dirs[0] / fname)
            mask_arr = _find_tile_mask(tile_path, mask_by_stem, mask_dir, fname)
            if mask_arr is not None:
                frac = float(mask_arr.mean())
                if frac < min_tissue_fraction:
                    n_skipped += 1
                    print(f"  [{name}] skip (tissue {frac:.2f} < {min_tissue_fraction:.2f}): {fname}")
                    continue
                u[~mask_arr] = 0.0

        raw_maps[fname]  = u
        mean_maps[fname] = np.mean(stack, axis=0)   # (H, W, 3) mean prediction
        ref_images[fname] = stack[0]
        print(f"  [{name}] computed uncertainty: {fname}")

    if n_skipped:
        print(f"  [{name}] Skipped {n_skipped} tiles below tissue threshold ({min_tissue_fraction:.2f})")

    # --- pass 2: global normalisation ---
    low, high = compute_global_bounds(
        list(raw_maps.values()), lower_pct=lower_pct, upper_pct=upper_pct
    )
    print(f"  Global bounds (p{lower_pct:.0f}, p{upper_pct:.0f}): "
          f"low={low:.6f}, high={high:.6f}")

    # --- pass 3: normalise, save, and summarise ---
    summary: dict = {
        "architecture": name,
        "ensemble_root": str(root),
        "n_ensemble_members": len(ensemble_dirs),
        "n_images": len(filenames),
        "log_compress": log_compress,
        "lower_percentile": lower_pct,
        "upper_percentile": upper_pct,
        "global_low": low,
        "global_high": high,
        "images": {},
    }

    for fname in filenames:
        if fname not in raw_maps:   # skipped in pass 1 (below tissue threshold)
            continue
        # Preserve the relative path structure (e.g. "001/images/0000001") so
        # that per-WSI parallel jobs writing to the same output root do not
        # overwrite each other's files.
        rel_stem = Path(fname).with_suffix("")  # "001/images/0000001" or "0000001"
        u_raw = raw_maps[fname]
        u_norm = normalise(u_raw, low, high)

        for base_dir in dirs.values():
            (base_dir / rel_stem).parent.mkdir(parents=True, exist_ok=True)

        np.save(str(dirs["raw_npy"] / rel_stem.with_suffix(".npy")), u_raw)
        np.save(str(dirs["norm_npy"] / rel_stem.with_suffix(".npy")), u_norm)
        save_heatmap(u_norm, dirs["heatmaps"] / rel_stem.with_suffix(".png"))
        if save_overlays:
            save_overlay(ref_images[fname], u_norm,
                         dirs["overlays"] / rel_stem.with_suffix(".png"))

        # Mean prediction — clamp to [0, 255] and save as uint8 TIF
        mean_uint8 = np.clip(mean_maps[fname], 0, 255).astype(np.uint8)
        tifffile.imwrite(str(dirs["mean_rgb"] / rel_stem.with_suffix(".tif")),
                         mean_uint8)

        summary["images"][fname] = image_stats(u_raw)

    # --- write summary JSON ---
    # When data_range is a single WSI, tag the filename to avoid races between
    # parallel jobs writing to the same output directory.
    if data_range is not None and data_range[0] == data_range[1]:
        summary_name = f"summary_wsi{data_range[0]:03d}.json"
    elif data_range is not None:
        summary_name = f"summary_wsi{data_range[0]:03d}-{data_range[1]:03d}.json"
    else:
        summary_name = "summary.json"

    summary_path = out / summary_name
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")
    print(f"  Done — outputs in {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute globally normalised epistemic uncertainty maps "
                    "from deep ensemble image-translation outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Architecture name (e.g. cyclegan, munit). Used for labelling outputs.",
    )
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Path to ensemble output directory containing model_01/, model_02/, etc.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("uncertainty_output"),
        help="Root directory for all outputs (default: uncertainty_output).",
    )
    parser.add_argument(
        "--log-compress", action="store_true",
        help="Apply log1p compression to uncertainty maps before normalisation.",
    )
    parser.add_argument(
        "--lower-percentile", type=float, default=1.0,
        help="Lower percentile for global normalisation bounds (default: 1).",
    )
    parser.add_argument(
        "--upper-percentile", type=float, default=99.0,
        help="Upper percentile for global normalisation bounds (default: 99).",
    )
    parser.add_argument(
        "--overlays", action="store_true",
        help="Save overlay PNGs (heatmap blended on reference image).",
    )
    parser.add_argument(
        "--data_range", type=str, default=None,
        metavar="START,END",
        help="Limit processing to WSI folders {START:03d}–{END:03d} under each "
             "member directory (e.g. '1,5'). Enables per-WSI parallel jobs when "
             "combined with a SLURM array. Without this flag all tiles are "
             "discovered automatically.",
    )
    parser.add_argument(
        "--mask_dir", type=Path, default=None,
        help="Directory of tissue mask TIFs (walked recursively, matched by stem). "
             "If omitted, masks are auto-detected from the tile structure by "
             "replacing the 'images' path component with 'masks'. "
             "Tiles with no matching mask are always included.",
    )
    parser.add_argument(
        "--min_tissue_fraction", type=float, default=0.0,
        help="Minimum tissue fraction [0–1] to include a tile (default 0 = all tiles). "
             "Set >0 to skip background-only tiles (e.g. 0.1). "
             "Requires --mask_dir or a tile structure with a 'masks/' sibling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.data.is_dir():
        print(f"Error: {args.data} is not a directory.", file=sys.stderr)
        sys.exit(1)

    data_range: tuple[int, int] | None = None
    if args.data_range is not None:
        try:
            start, end = args.data_range.split(",")
            data_range = (int(start.strip()), int(end.strip()))
        except ValueError:
            print(f"Error: --data_range must be 'START,END', got '{args.data_range}'",
                  file=sys.stderr)
            sys.exit(1)

    process_architecture(
        name=args.model,
        root=args.data,
        output_root=args.output,
        log_compress=args.log_compress,
        lower_pct=args.lower_percentile,
        upper_pct=args.upper_percentile,
        save_overlays=args.overlays,
        data_range=data_range,
        mask_dir=args.mask_dir,
        min_tissue_fraction=args.min_tissue_fraction,
    )

    print(f"\nDone. Outputs written to {args.output / args.model}")


if __name__ == "__main__":
    main()
