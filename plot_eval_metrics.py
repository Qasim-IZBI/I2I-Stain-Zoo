"""
Scatter plot of evaluation metrics across all 54 model configurations.

Layout : 1 row × 3 columns  (FID | SSIM | LPIPS)
X-axis : model name  (6 models)
Y-axis : metric score
Color  : model size  (small=blue, medium=orange, large=green)
Marker : data size   (small=circle, medium=square, large=triangle)

Points are jittered on x so all 9 configurations per model are visible.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EVAL_BASE = "/work2/bz66izin-VSproject/evaluation"

MODELS    = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]
SIZES     = ["small", "medium", "large"]      # model sizes  → color
DATASIZES = ["small", "medium", "large"]      # data sizes   → marker

COLORS  = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green  (same as plot_all_losses.py)
MARKERS = ["o", "s", "^"]                     # circle, square, triangle

# X-jitter per (size_idx, datasize_idx) so 9 points per model don't overlap.
# size groups are spread ±0.25; within each group data sizes add ±0.07.
_SIZE_OFFSET = [-0.25, 0.0, 0.25]
_DATA_OFFSET = [-0.07, 0.0, 0.07]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _csv_path(model: str, size: str, datasize: str, metric: str) -> str:
    return os.path.join(
        EVAL_BASE, model, "results",
        f"data_{datasize}", f"model_{size}",
        f"{metric}.csv",
    )


def load_metric(model: str, size: str, datasize: str, metric: str):
    """Return a single scalar for the given config/metric, or None if missing."""
    path = _csv_path(model, size, datasize, metric)
    if not os.path.exists(path):
        warnings.warn(f"Missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None

    if metric == "fid":
        if "value" not in df.columns:
            warnings.warn(f"Expected 'value' column in {path}")
            return None
        return float(df["value"].iloc[0])
    elif metric == "patch_ssim":
        if "patch_ssim" not in df.columns:
            warnings.warn(f"Expected 'patch_ssim' column in {path}")
            return None
        return float(df["patch_ssim"].mean())
    elif metric == "lpips":
        if "lpips" not in df.columns:
            warnings.warn(f"Expected 'lpips' column in {path}")
            return None
        return float(df["lpips"].mean())
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_META = {
    "fid":        {"label": "FID ↓",              "title": "FID (Fréchet Inception Distance)"},
    "patch_ssim": {"label": "Patch-SSIM ↑",       "title": "Patch-SSIM (mean, 64×64 patches)"},
    "lpips":      {"label": "LPIPS ↓",            "title": "LPIPS (mean per image)"},
}


def main(out_path: str):
    metrics = ["fid", "patch_ssim", "lpips"]
    x_positions = np.arange(len(MODELS))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    csv_rows = []

    for ax, metric in zip(axes, metrics):
        meta = METRIC_META[metric]
        any_plotted = False

        for s_idx, size in enumerate(SIZES):
            for d_idx, datasize in enumerate(DATASIZES):
                xs, ys = [], []
                for m_idx, model in enumerate(MODELS):
                    val = load_metric(model, size, datasize, metric)
                    if val is None:
                        continue
                    jitter = _SIZE_OFFSET[s_idx] + _DATA_OFFSET[d_idx]
                    xs.append(m_idx + jitter)
                    ys.append(val)
                    csv_rows.append({
                        "metric":      metric,
                        "model":       model,
                        "model_size":  size,
                        "data_size":   datasize,
                        "value":       val,
                    })

                if xs:
                    ax.scatter(
                        xs, ys,
                        color=COLORS[s_idx],
                        marker=MARKERS[d_idx],
                        s=70,
                        alpha=0.85,
                        linewidths=0.5,
                        edgecolors="white",
                        zorder=3,
                    )
                    any_plotted = True

        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(meta["label"], fontsize=10)
        ax.set_title(meta["title"], fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)

        if not any_plotted:
            ax.text(0.5, 0.5, "No data found", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=11)

    # ------------------------------------------------------------------
    # Shared legend: colors = model size, markers = data size
    # ------------------------------------------------------------------
    size_handles = [
        mlines.Line2D([], [], color=COLORS[i], marker="o", linestyle="None",
                      markersize=8, label=f"Model size: {SIZES[i]}")
        for i in range(len(SIZES))
    ]
    data_handles = [
        mlines.Line2D([], [], color="dimgray", marker=MARKERS[i], linestyle="None",
                      markersize=8, label=f"Data size: {DATASIZES[i]}")
        for i in range(len(DATASIZES))
    ]
    fig.legend(
        handles=size_handles + data_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        "Evaluation Metrics Across All Model Configurations",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for all 54 configs.")
    parser.add_argument(
        "--out",
        default=os.path.join(EVAL_BASE, "eval_metrics.png"),
        help="Output PNG path (default: $EVAL_BASE/eval_metrics.png)",
    )
    args = parser.parse_args()
    main(args.out)
