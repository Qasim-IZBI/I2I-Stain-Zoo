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

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


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


def discover_common_filenames(ensemble_dirs: list[Path]) -> list[str]:
    """Return sorted filenames present in *all* ensemble directories.

    Filenames that are missing from at least one directory are skipped with a
    warning.
    """
    sets = []
    for d in ensemble_dirs:
        names = {f.name for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".tif", ".tiff"}}
        sets.append(names)

    common = sorted(set.intersection(*sets))
    all_names = sorted(set.union(*sets))

    skipped = set(all_names) - set(common)
    if skipped:
        warnings.warn(
            f"Skipping {len(skipped)} filename(s) not present in all ensemble "
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
        Shape (H, W) — scalar uncertainty map (sum of per-channel variances).
    """
    # Per-channel sample variance across ensemble axis (ddof=1)
    channel_var = np.var(stack, axis=0, ddof=1)  # (H, W, 3)
    # Collapse channels by summation
    return channel_var.sum(axis=2)  # (H, W)


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
) -> None:
    """Full uncertainty pipeline for one architecture."""
    print(f"\n{'=' * 60}")
    print(f"Architecture: {name}")
    print(f"Ensemble root: {root}")
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

    filenames = discover_common_filenames(ensemble_dirs)
    if not filenames:
        raise RuntimeError(f"No common TIFF filenames across ensemble dirs in {root}")
    print(f"Processing {len(filenames)} common image(s)")

    # --- create output dirs ---
    out = output_root / name
    dirs = {
        "raw_npy": out / "raw_npy",
        "norm_npy": out / "norm_npy",
        "heatmaps": out / "heatmaps",
    }
    if save_overlays:
        dirs["overlays"] = out / "overlays"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- pass 1: compute raw uncertainty maps ---
    raw_maps: dict[str, np.ndarray] = {}
    ref_images: dict[str, np.ndarray] = {}  # first ensemble member as reference

    for fname in filenames:
        stack = []
        for edir in ensemble_dirs:
            img = read_rgb_tiff(edir / fname)
            stack.append(img)
        stack = np.stack(stack, axis=0)  # (N, H, W, 3)

        u = compute_uncertainty_map(stack)
        if log_compress:
            u = np.log1p(u)

        raw_maps[fname] = u
        ref_images[fname] = stack[0]  # keep first member for overlays
        print(f"  [{name}] computed uncertainty: {fname}")

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
        stem = Path(fname).stem
        u_raw = raw_maps[fname]
        u_norm = normalise(u_raw, low, high)

        # Save raw and normalised .npy
        np.save(str(dirs["raw_npy"] / f"{stem}.npy"), u_raw)
        np.save(str(dirs["norm_npy"] / f"{stem}.npy"), u_norm)

        # Save heatmap
        save_heatmap(u_norm, dirs["heatmaps"] / f"{stem}.png")

        # Save overlay
        if save_overlays:
            save_overlay(ref_images[fname], u_norm, dirs["overlays"] / f"{stem}.png")

        summary["images"][fname] = image_stats(u_raw)

    # --- write summary JSON ---
    summary_path = out / "summary.json"
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.data.is_dir():
        print(f"Error: {args.data} is not a directory.", file=sys.stderr)
        sys.exit(1)

    process_architecture(
        name=args.model,
        root=args.data,
        output_root=args.output,
        log_compress=args.log_compress,
        lower_pct=args.lower_percentile,
        upper_pct=args.upper_percentile,
        save_overlays=args.overlays,
    )

    print(f"\nDone. Outputs written to {args.output / args.model}")


if __name__ == "__main__":
    main()
