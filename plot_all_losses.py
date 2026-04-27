"""
Plot generator loss (loss_G) for all 54 training runs.

Layout: 6 subplots (one per model type), each with 9 lines
        (3 model sizes × 3 data sizes).
Color  → model size  (small=blue, medium=orange, large=green)
Style  → data size   (small=solid, medium=dashed, large=dotted)
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
BASE = "/work2/bz66izin-VSproject/Outputs_noamp"

MODELS = ["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"]
SIZES = ["small", "medium", "large"]      # model sizes  → color
DATASIZES = ["small", "medium", "large"]  # data sizes   → linestyle

COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green
STYLES = ["-", "--", ":"]                     # solid, dashed, dotted

# Multi-stage models and their ordered stages
STAGES = {
    "miudiff": [1, 2, 3],
    "uvcgan":  [1, 2],
}

SMOOTH_WINDOW = 20   # rolling-mean window (set to 1 to disable)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _csv_paths(model: str, datasize: str, size: str) -> list[str]:
    """Return ordered list of loss_log.csv paths for this run."""
    base_dir = os.path.join(BASE, model, "results", f"data_{datasize}", f"model_{size}")
    if model in STAGES:
        return [
            os.path.join(base_dir, f"stage{s}", "loss_log.csv")
            for s in STAGES[model]
        ]
    return [os.path.join(base_dir, "loss_log.csv")]


def load_loss(model: str, datasize: str, size: str):
    """
    Load loss_G for a single run.  For multi-stage models, concatenate stages
    with sequential step offsets so the x-axis is monotonically increasing.

    Returns (steps_array, loss_G_array) or None if no files are found.
    """
    paths = _csv_paths(model, datasize, size)
    frames = []
    step_offset = 0

    for path in paths:
        if not os.path.exists(path):
            warnings.warn(f"Missing: {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            warnings.warn(f"Could not read {path}: {exc}")
            continue
        if "global_step" not in df.columns or "loss_G" not in df.columns:
            warnings.warn(f"Expected columns missing in {path}")
            continue
        df = df[["global_step", "loss_G"]].dropna()
        df["global_step"] = df["global_step"] + step_offset
        step_offset = int(df["global_step"].iloc[-1]) + 1
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    return combined["global_step"].to_numpy(), combined["loss_G"].to_numpy()


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().to_numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def main(out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(MODELS):
        ax = axes_flat[ax_idx]

        any_plotted = False
        for s_idx, size in enumerate(SIZES):
            for d_idx, datasize in enumerate(DATASIZES):
                result = load_loss(model, datasize, size)
                if result is None:
                    continue
                steps, loss = result
                color = COLORS[s_idx]
                style = STYLES[d_idx]

                # faint raw curve
                ax.plot(steps, loss, color=color, linestyle=style,
                        alpha=0.15, linewidth=0.8)
                # smoothed foreground
                ax.plot(steps, smooth(loss, SMOOTH_WINDOW),
                        color=color, linestyle=style,
                        linewidth=1.8, alpha=0.95)
                any_plotted = True

        ax.set_title(model.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Steps", fontsize=9)
        ax.set_ylabel("Generator Loss (loss_G)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if not any_plotted:
            ax.text(0.5, 0.5, "No data found", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=11)

    # ------------------------------------------------------------------
    # Shared legend: colors = model size, styles = data size
    # ------------------------------------------------------------------
    size_handles = [
        mlines.Line2D([], [], color=COLORS[i], linewidth=2,
                      label=f"Model: {SIZES[i]}")
        for i in range(len(SIZES))
    ]
    data_handles = [
        mlines.Line2D([], [], color="black", linestyle=STYLES[i], linewidth=2,
                      label=f"Data: {DATASIZES[i]}")
        for i in range(len(DATASIZES))
    ]
    fig.legend(
        handles=size_handles + data_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Generator Loss Across 54 Training Runs", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(BASE, "all_losses.png"),
        help="Output PNG path (default: $BASE/all_losses.png)",
    )
    args = parser.parse_args()
    main(args.out)
