"""Boxplot and violin plot of per-tile mean regen_error for each model family.

Reads per-tile [H, W] error .npy maps from
  {base}/{model}/data_large/{model_size}/regen_error/wsi{NNN}/error_npy/

and computes the tissue-masked mean MAE per tile.

Produces:
  - Pooled boxplot and violin (all WSIs combined)
  - Per-WSI boxplot and violin under {outdir}/per_wsi/
  - Summary CSV with quantile statistics: regen_error_quantiles.csv

Example
-------
python plot_regen_error_boxplot.py \
    --base    /work2/bz66izin-VSproject/ensemble \
    --test_a  /work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA \
    --outdir  ./regen_error_boxplot/

python plot_regen_error_boxplot.py   # uses default paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Config — mirrors plot_uncertainty_boxplot.py conventions
# ---------------------------------------------------------------------------

MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]

MODEL_DISPLAY_NAMES = {
    "cyclegan":       "CycleGAN",
    "unit":           "UNIT",
    "munit":          "MUNIT",
    "dclgan":         "DCLGAN",
    "uvcgan":         "UVCGAN",
    "cyclediffusion": "CycleDiffusion",
}

MODEL_SIZES = {
    "cyclegan":       "model_medium",
    "unit":           "model_medium",
    "munit":          "model_medium",
    "dclgan":         "model_small",
    "uvcgan":         "model_small",
    "cyclediffusion": "model_small",
}

N_WSIS = 5

DEFAULT_BASE  = Path("/work2/bz66izin-VSproject/ensemble")
DEFAULT_TEST_A = Path("/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA")


# ---------------------------------------------------------------------------
# Tissue mask helpers
# ---------------------------------------------------------------------------

def _build_mask_lookup(mask_root: Path) -> dict[str, Path]:
    """Flat stem → mask-path dict for the entire mask_root tree."""
    lookup: dict[str, Path] = {}
    for p in mask_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png"}:
            lookup[p.stem] = p
    return lookup


def _load_mask(stem: str, wsi_folder: str, mask_root: Path,
               mask_lookup: dict[str, Path], target_shape: tuple[int, int]) -> np.ndarray | None:
    """Return boolean tissue mask for *stem*, or None if not found."""
    # Primary: mask_root/{NNN}/masks/{stem}.tif
    for ext in (".tif", ".tiff", ".png"):
        p = mask_root / wsi_folder / "masks" / f"{stem}{ext}"
        if p.exists():
            m = tifffile.imread(str(p))
            break
    else:
        # Fallback: flat lookup by stem
        p = mask_lookup.get(stem)
        if p is None:
            return None
        m = tifffile.imread(str(p))

    if m.ndim > 2:
        m = m[..., 0]
    if m.shape != target_shape:
        from PIL import Image as PILImage
        m = np.array(
            PILImage.fromarray(m.astype(np.uint8)).resize(
                (target_shape[1], target_shape[0]), resample=PILImage.NEAREST
            )
        )
    return m > 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_data(
    model: str,
    base: Path,
    mask_root: Path,
    min_tissue_fraction: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return {wsi_stem: array_of_per_tile_mean_regen_error} for one model.

    wsi_stem is the folder name (e.g. '001', '002', ...).
    """
    model_size   = MODEL_SIZES[model]
    ensemble_root = base / model / "data_large" / model_size
    mask_lookup  = _build_mask_lookup(mask_root)

    wsi_data: dict[str, np.ndarray] = {}

    for wsi_num in range(1, N_WSIS + 1):
        wsi_folder = f"{wsi_num:03d}"
        error_dir  = ensemble_root / "regen_error" / f"wsi{wsi_folder}" / "error_npy"

        if not error_dir.exists():
            print(f"  [WARN] Not found: {error_dir}")
            continue

        npy_files = sorted(error_dir.glob("*.npy"))
        if not npy_files:
            print(f"  [WARN] Empty: {error_dir}")
            continue

        tile_means: list[float] = []
        n_skipped = 0

        for npy_path in npy_files:
            err_map = np.load(npy_path).astype(np.float64)  # [H, W], values in [0, 255]
            target_shape = (err_map.shape[0], err_map.shape[1])

            mask = _load_mask(npy_path.stem, wsi_folder, mask_root, mask_lookup, target_shape)
            if mask is not None:
                tissue_frac = float(mask.mean())
                if tissue_frac < min_tissue_fraction:
                    n_skipped += 1
                    continue
                tile_mean = float(err_map[mask].mean())
            else:
                tile_mean = float(err_map.mean())

            tile_means.append(tile_mean)

        if tile_means:
            wsi_data[wsi_folder] = np.array(tile_means, dtype=np.float64)

        print(f"  {MODEL_DISPLAY_NAMES[model]} WSI {wsi_folder}: "
              f"{len(tile_means)} tiles  (skipped {n_skipped} low-tissue)")

    return wsi_data


def load_all_data(base: Path, mask_root: Path,
                  min_tissue_fraction: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    all_data: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        print(f"\nLoading {MODEL_DISPLAY_NAMES[model]} …")
        all_data[model] = load_model_data(model, base, mask_root, min_tissue_fraction)
    return all_data


def pool_across_wsis(all_data: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        model: (np.concatenate(list(wsi_data.values()))
                if wsi_data else np.array([]))
        for model, wsi_data in all_data.items()
    }


# ---------------------------------------------------------------------------
# Quantile stats
# ---------------------------------------------------------------------------

def compute_stats(data: dict[str, np.ndarray], wsi: str) -> list[dict]:
    rows = []
    for model in MODELS:
        vals = data[model]
        if len(vals) == 0:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        rows.append({
            "wsi":          wsi,
            "model":        MODEL_DISPLAY_NAMES[model],
            "n_tiles":      len(vals),
            "min":          float(vals.min()),
            "whisker_low":  float(max(vals.min(), q1 - 1.5 * iqr)),
            "q1":           float(q1),
            "median":       float(median),
            "mean":         float(vals.mean()),
            "q3":           float(q3),
            "whisker_high": float(min(vals.max(), q3 + 1.5 * iqr)),
            "max":          float(vals.max()),
            "iqr":          float(iqr),
        })
    return rows


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _colors(n: int) -> np.ndarray:
    return plt.cm.tab10(np.linspace(0, 0.6, n))


def _apply_common_style(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Mean regen error per tile (MAE, [0–255])", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

def plot_boxplot(data: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    models_with_data = [m for m in MODELS if len(data[m]) > 0]
    if not models_with_data:
        return
    labels = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    values = [data[m] for m in models_with_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        values,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], _colors(len(models_with_data))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    _apply_common_style(ax, labels)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_violin(data: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    models_with_data = [m for m in MODELS if len(data[m]) > 0]
    if not models_with_data:
        return
    labels = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    values = [data[m] for m in models_with_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    parts = ax.violinplot(values, positions=range(1, len(values) + 1),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], _colors(len(models_with_data))):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_linewidth(1.2)
        parts[key].set_color("black")

    _apply_common_style(ax, labels)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Bar chart: mean ± std across tiles (summary view)
# ---------------------------------------------------------------------------

def plot_mean_bar(data: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    models_with_data = [m for m in MODELS if len(data[m]) > 0]
    if not models_with_data:
        return
    labels  = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    means   = [data[m].mean() for m in models_with_data]
    stds    = [data[m].std()  for m in models_with_data]
    colors  = _colors(len(models_with_data))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=4, linewidth=1.2, capthick=1.2)

    # Annotate values above bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Mean regen error (MAE, [0–255])", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Box and violin plots of per-tile regen_error per model family."
    )
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"Ensemble root directory (default: {DEFAULT_BASE})")
    ap.add_argument("--test_a", type=Path, default=DEFAULT_TEST_A,
                    help=f"testA tiles root for tissue masks (default: {DEFAULT_TEST_A})")
    ap.add_argument("--min_tissue_fraction", type=float, default=0.1,
                    help="Minimum tissue fraction to include a tile (default: 0.1)")
    ap.add_argument("--outdir", type=Path, default=Path("regen_error_boxplot"),
                    help="Output directory (default: ./regen_error_boxplot/)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading regen_error .npy maps …")
    all_data = load_all_data(args.base, args.test_a, args.min_tissue_fraction)

    pooled = pool_across_wsis(all_data)
    if all(len(v) == 0 for v in pooled.values()):
        raise RuntimeError("No data found for any model. Check --base and --test_a paths.")

    all_stats: list[dict] = []

    # --- pooled plots (all WSIs combined) ---
    print("\nPlotting pooled figures …")
    pooled_title = "Reconstruction error (regen_error) distribution across model families"
    plot_boxplot(pooled, pooled_title, args.outdir / "regen_error_boxplot.png")
    plot_violin(pooled,  pooled_title, args.outdir / "regen_error_violin.png")
    plot_mean_bar(pooled, pooled_title, args.outdir / "regen_error_bar.png")
    all_stats.extend(compute_stats(pooled, wsi="all"))

    # --- per-WSI plots ---
    all_wsis = sorted({wsi for wsi_data in all_data.values() for wsi in wsi_data})
    print(f"\nPlotting per-WSI figures for {len(all_wsis)} WSI(s) …")
    per_wsi_dir = args.outdir / "per_wsi"

    for wsi in all_wsis:
        wsi_data = {model: all_data[model].get(wsi, np.array([])) for model in MODELS}
        title = f"Reconstruction error (regen_error) — WSI {wsi}"
        plot_boxplot(wsi_data, title, per_wsi_dir / f"{wsi}_boxplot.png")
        plot_violin(wsi_data,  title, per_wsi_dir / f"{wsi}_violin.png")
        all_stats.extend(compute_stats(wsi_data, wsi=wsi))

    # --- save quantile CSV ---
    stats_df = pd.DataFrame(all_stats)
    stats_path = args.outdir / "regen_error_quantiles.csv"
    stats_df.to_csv(stats_path, index=False, float_format="%.6f")
    print(f"\nSaved quantile stats → {stats_path}")

    # --- print summary table ---
    print("\n--- Pooled mean regen_error per model (MAE in [0–255]) ---")
    for model in MODELS:
        vals = pooled[model]
        if len(vals) > 0:
            print(f"  {MODEL_DISPLAY_NAMES[model]:>15}: "
                  f"mean={vals.mean():.2f}  median={np.median(vals):.2f}  "
                  f"std={vals.std():.2f}  n={len(vals)}")
        else:
            print(f"  {MODEL_DISPLAY_NAMES[model]:>15}: no data")


if __name__ == "__main__":
    main()
