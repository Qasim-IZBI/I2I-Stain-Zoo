"""Boxplot and violin plot of per-tile mean uncertainty for each model family.

Produces:
  - One pooled figure (all WSIs combined) for box and violin
  - One per-WSI figure (box and violin) saved under {outdir}/per_wsi/

Reads the per-WSI CSVs produced by aggregate_uncertainty.py.

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

def load_all_data(base: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load uncertainty values per model and per WSI.

    Returns: {model: {wsi_stem: np.ndarray of per-tile mean_uncertainty}}
    """
    all_data: dict[str, dict[str, np.ndarray]] = {}

    for model in MODELS:
        model_size = MODEL_SIZES[model]
        csv_dir = (base / model / "data_large" / model_size
                   / "uncertainty" / model / "per_wsi_csv")

        wsi_data: dict[str, np.ndarray] = {}
        if not csv_dir.exists():
            print(f"  [WARN] Not found: {csv_dir}")
        else:
            for csv_path in sorted(csv_dir.glob("*.csv")):
                wsi_stem = csv_path.stem
                vals = (pd.read_csv(csv_path, usecols=["mean_uncertainty"])
                        ["mean_uncertainty"].dropna().to_numpy())
                wsi_data[wsi_stem] = vals

            n_tiles = sum(len(v) for v in wsi_data.values())
            print(f"  {MODEL_DISPLAY_NAMES[model]}: {n_tiles:,} tiles "
                  f"from {len(wsi_data)} WSI(s)")

        all_data[model] = wsi_data

    return all_data


def pool_across_wsis(all_data: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate all WSI arrays per model."""
    return {
        model: (np.concatenate(list(wsi_data.values()))
                if wsi_data else np.array([]))
        for model, wsi_data in all_data.items()
    }


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _apply_common_style(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean uncertainty per tile", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _colors(n: int) -> np.ndarray:
    return plt.cm.tab10(np.linspace(0, 0.6, n))


# ---------------------------------------------------------------------------
# Quantile stats
# ---------------------------------------------------------------------------

def compute_stats(data: dict[str, np.ndarray], wsi: str) -> list[dict]:
    """Return one row per model with standard boxplot quantile statistics."""
    rows = []
    for model in MODELS:
        vals = data[model]
        if len(vals) == 0:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        rows.append({
            "wsi":           wsi,
            "model":         MODEL_DISPLAY_NAMES[model],
            "n_tiles":       len(vals),
            "min":           float(vals.min()),
            "whisker_low":   float(max(vals.min(), q1 - 1.5 * iqr)),
            "q1":            float(q1),
            "median":        float(median),
            "mean":          float(vals.mean()),
            "q3":            float(q3),
            "whisker_high":  float(min(vals.max(), q3 + 1.5 * iqr)),
            "max":           float(vals.max()),
            "iqr":           float(iqr),
        })
    return rows


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
        showfliers=True,
        flierprops=dict(marker=".", markersize=1.5, alpha=0.3, color="gray"),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], _colors(len(models_with_data))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (vals, med_line) in enumerate(zip(values, bp["medians"]), start=1):
        ax.text(i, med_line.get_ydata()[1], f"{np.median(vals):.4f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    _apply_common_style(ax, labels)
    ax.set_title(title, fontsize=11)

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

    for i, vals in enumerate(values, start=1):
        ax.text(i, float(np.median(vals)), f"{np.median(vals):.4f}",
                ha="center", va="bottom", fontsize=7.5, color="black")

    _apply_common_style(ax, labels)
    ax.set_title(title, fontsize=11)

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
        description="Box and violin plots of per-tile uncertainty per model family."
    )
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"Ensemble root directory (default: {DEFAULT_BASE})")
    ap.add_argument("--outdir", type=Path, default=Path("uncertainty_boxplot"),
                    help="Output directory (default: ./uncertainty_boxplot/)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading uncertainty CSVs …")
    all_data = load_all_data(args.base)

    pooled = pool_across_wsis(all_data)
    if all(len(v) == 0 for v in pooled.values()):
        raise RuntimeError("No data found for any model. Check --base path.")

    all_stats: list[dict] = []

    # --- pooled plots (all WSIs combined) ---
    print("\nPlotting pooled figures …")
    pooled_title = "Epistemic uncertainty distribution across model families"
    plot_boxplot(pooled, pooled_title, args.outdir / "uncertainty_boxplot.png")
    plot_violin(pooled,  pooled_title, args.outdir / "uncertainty_violin.png")
    all_stats.extend(compute_stats(pooled, wsi="all"))

    # --- per-WSI plots ---
    all_wsis = sorted({wsi for wsi_data in all_data.values() for wsi in wsi_data})
    print(f"\nPlotting per-WSI figures for {len(all_wsis)} WSI(s) …")
    per_wsi_dir = args.outdir / "per_wsi"

    for wsi in all_wsis:
        wsi_data = {model: all_data[model].get(wsi, np.array([])) for model in MODELS}
        title = f"Epistemic uncertainty — {wsi}"
        plot_boxplot(wsi_data, title, per_wsi_dir / f"{wsi}_boxplot.png")
        plot_violin(wsi_data,  title, per_wsi_dir / f"{wsi}_violin.png")
        all_stats.extend(compute_stats(wsi_data, wsi=wsi))

    # --- save quantile CSV ---
    stats_df = pd.DataFrame(all_stats)
    stats_path = args.outdir / "uncertainty_quantiles.csv"
    stats_df.to_csv(stats_path, index=False, float_format="%.6f")
    print(f"\nSaved quantile stats → {stats_path}")


if __name__ == "__main__":
    main()
