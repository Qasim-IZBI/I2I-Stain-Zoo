"""Boxplot of per-tile mean uncertainty for each model family.

Reads the per-WSI CSVs produced by aggregate_uncertainty.py (one CSV per WSI,
one row per tile with mean_uncertainty), pools all tiles across WSIs for each
model family, and draws one box per model on the x-axis.

Example
-------
python plot_uncertainty_boxplot.py \
    --base   /work2/bz66izin-VSproject/ensemble \
    --outdir ./uncertainty_boxplot/

python plot_uncertainty_boxplot.py   # uses default paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — mirrors plot_combined_metrics.py conventions
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

# Must match aggregate_uncertainty.sh
MODEL_SIZES = {
    "cyclegan":       "model_medium",
    "unit":           "model_medium",
    "munit":          "model_medium",
    "dclgan":         "model_small",
    "uvcgan":         "model_small",
    "cyclediffusion": "model_small",
}

DEFAULT_BASE = Path("/work2/bz66izin-VSproject/ensemble")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_tiles(base: Path, model: str) -> np.ndarray:
    """Return all per-tile mean_uncertainty values for one model family."""
    model_size = MODEL_SIZES[model]
    csv_dir = base / model / "data_large" / model_size / "uncertainty" / model / "per_wsi_csv"
    if not csv_dir.exists():
        print(f"  [WARN] Not found: {csv_dir}")
        return np.array([])

    dfs = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, usecols=["mean_uncertainty"])
        dfs.append(df)

    if not dfs:
        print(f"  [WARN] No CSVs in {csv_dir}")
        return np.array([])

    values = pd.concat(dfs, ignore_index=True)["mean_uncertainty"].dropna().to_numpy()
    print(f"  {MODEL_DISPLAY_NAMES[model]}: {len(values):,} tiles from {len(dfs)} WSI(s)")
    return values


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _apply_common_style(ax: plt.Axes, labels: list[str], n: int) -> None:
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean uncertainty per tile", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_boxplot(data: dict[str, np.ndarray], outdir: Path) -> None:
    models_with_data = [m for m in MODELS if len(data[m]) > 0]
    labels = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    values = [data[m] for m in models_with_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        values,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker=".", markersize=1.5, alpha=0.3, color="gray"),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(models_with_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate median values above each box
    for i, (vals, med_line) in enumerate(zip(values, bp["medians"]), start=1):
        median_val = float(np.median(vals))
        ax.text(i, med_line.get_ydata()[1], f"{median_val:.4f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    _apply_common_style(ax, labels, len(labels))
    ax.set_title("Epistemic uncertainty distribution across model families", fontsize=11)

    fig.tight_layout()
    out_path = outdir / "uncertainty_boxplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_violin(data: dict[str, np.ndarray], outdir: Path) -> None:
    models_with_data = [m for m in MODELS if len(data[m]) > 0]
    labels = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    values = [data[m] for m in models_with_data]

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(models_with_data)))

    fig, ax = plt.subplots(figsize=(10, 5))

    parts = ax.violinplot(values, positions=range(1, len(values) + 1),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_linewidth(1.2)
        parts[key].set_color("black")

    # Annotate median values
    for i, vals in enumerate(values, start=1):
        median_val = float(np.median(vals))
        ax.text(i, median_val, f"{median_val:.4f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    _apply_common_style(ax, labels, len(labels))
    ax.set_title("Epistemic uncertainty distribution across model families", fontsize=11)

    fig.tight_layout()
    out_path = outdir / "uncertainty_violin.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Boxplot of per-tile uncertainty per model family.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"Ensemble root directory (default: {DEFAULT_BASE})")
    ap.add_argument("--outdir", type=Path, default=Path("uncertainty_boxplot"),
                    help="Output directory for the figure (default: ./uncertainty_boxplot/)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading uncertainty CSVs …")
    data = {model: load_model_tiles(args.base, model) for model in MODELS}

    n_missing = sum(1 for v in data.values() if len(v) == 0)
    if n_missing == len(MODELS):
        raise RuntimeError("No data found for any model. Check --base path.")

    plot_boxplot(data, args.outdir)
    plot_violin(data, args.outdir)


if __name__ == "__main__":
    main()
