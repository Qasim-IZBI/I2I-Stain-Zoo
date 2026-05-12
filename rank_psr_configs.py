"""
rank_psr_configs.py — find the best model size / data size config per model family.

Reads summary.json files produced by compare_psr_all_configs.sh (one per model),
ranks the 9 configs (3 model sizes × 3 data sizes) by Spearman ρ descending,
tiebreak MAE ascending, and reports the winner per model.

Outputs
-------
best_per_model.csv   — one row per model: best config + its paired metrics
all_configs.csv      — all configs × models with paired metrics (for inspection)
best_per_model.png   — 2×3 scatter grid: top row = one panel per data size,
                       bottom row = one panel per model size; Pearson r vs MAE,
                       colour = model family, marker = complementary size axis,
                       star = best config per model family
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]


def load_all(indir: Path, models: list = None) -> pd.DataFrame:
    rows = []
    for model in (models if models is not None else MODELS):
        json_path = indir / model / "summary.json"
        if not json_path.exists():
            print(f"[WARN] Missing: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        for config, m in data.get("pairwise_vs_real", {}).items():
            rows.append({
                "model":            model,
                "config":           config,
                "spearman_rho":     m.get("spearman_rho"),
                "pearson_r":        m.get("pearson_r"),
                "mae_paired":       m.get("mae_paired"),
                "mean_paired_diff": m.get("mean_paired_diff_generated_minus_real"),
                "n_matched":        m.get("n_matched"),
            })
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["pearson_r"])
          .sort_values(["pearson_r", "mae_paired"], ascending=[False, True])
          .groupby("model", sort=False)
          .first()
          .reset_index()
    )


DATASIZES   = ["small", "medium", "large"]
MODEL_SIZES = ["small", "medium", "large"]
SIZE_MARKERS = {"small": "o", "medium": "s", "large": "^"}


def _parse_config(config: str):
    """Return (model_size, data_size) from 'small_model/medium_data'."""
    left, right = config.split("/")
    return left.replace("_model", ""), right.replace("_data", "")


def _scatter_panel(ax, sub: pd.DataFrame, best_subset: pd.DataFrame,
                   model_colors: dict, vary_col: str) -> None:
    """Scatter one panel. vary_col is the column whose value drives marker shape."""
    for size in MODEL_SIZES:
        for _, row in sub[sub[vary_col] == size].dropna(
                subset=["pearson_r", "mae_paired"]).iterrows():
            ax.scatter(row["mae_paired"], row["pearson_r"],
                       color=model_colors[row["model"]],
                       marker=SIZE_MARKERS[size],
                       s=80, edgecolors="black", linewidths=0.7, zorder=2)

    for _, brow in best_subset.iterrows():
        ax.scatter(brow["mae_paired"], brow["pearson_r"],
                   color=model_colors[brow["model"]],
                   marker="*", s=260, edgecolors="black", linewidths=0.8, zorder=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("MAE (paired)", fontsize=9)


def plot_grid(df: pd.DataFrame, best: pd.DataFrame, outpath: Path) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models       = [m for m in MODELS if m in df["model"].values]
    model_colors = {m: plt.cm.tab10.colors[i] for i, m in enumerate(models)}

    df = df.copy()
    df[["model_size", "data_size"]] = df["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )
    best = best.copy()
    best[["model_size", "data_size"]] = best["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

    # ---- top row: fixed data size, marker = model size ----
    for ax, ds in zip(axes[0], DATASIZES):
        _scatter_panel(ax, df[df["data_size"] == ds],
                       best[best["data_size"] == ds],
                       model_colors, vary_col="model_size")
        ax.set_title(f"data size: {ds}", fontsize=10, fontweight="bold")

    # ---- bottom row: fixed model size, marker = data size ----
    for ax, ms in zip(axes[1], MODEL_SIZES):
        _scatter_panel(ax, df[df["model_size"] == ms],
                       best[best["model_size"] == ms],
                       model_colors, vary_col="data_size")
        ax.set_title(f"model size: {ms}", fontsize=10, fontweight="bold")

    axes[0, 0].set_ylabel("Pearson r", fontsize=9)
    axes[1, 0].set_ylabel("Pearson r", fontsize=9)

    # ---- shared legend (bottom-right panel) ----
    color_handles = [Patch(facecolor=model_colors[m], edgecolor="black",
                           linewidth=0.7, label=m)
                     for m in models]
    marker_handles = [Line2D([0], [0], marker=SIZE_MARKERS[s], color="w",
                             markerfacecolor="gray", markeredgecolor="black",
                             markersize=8, label=s)
                      for s in MODEL_SIZES]
    star_handle = [Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="gray", markeredgecolor="black",
                          markersize=12, label="best config")]

    axes[1, 2].legend(handles=color_handles + [Line2D([], [], linestyle="none")] +
                      marker_handles + star_handle,
                      fontsize=7, loc="lower right",
                      title="model  |  size  |  best", title_fontsize=7)

    fig.suptitle(
        "PSR configs — Pearson r (↑) vs MAE (←)   "
        "Top: grouped by data size   Bottom: grouped by model size\n"
        "Colour = model family   Marker = complementary size axis   "
        "Star = best config per model",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank PSR configs per model family using paired WSI metrics "
                    "from compare_psr_all_configs.sh summary.json files."
    )
    parser.add_argument(
        "--indir", type=Path, required=True,
        help="Directory containing one subdirectory per model, each with summary.json "
             "(i.e. the --outdir root passed to compare_psr_all_configs.sh, "
             "e.g. /work2/bz66izin-VSproject/psr_comparison/).",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("psr_best_config"),
        help="Output directory for CSVs and plot [%(default)s]",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        metavar="MODEL",
        help="Restrict to these model(s) (default: all six). "
             "Used by the SLURM array script to process one model per task.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    models_to_run = args.models if args.models else MODELS
    df = load_all(args.indir, models=models_to_run)
    if df.empty:
        raise RuntimeError(f"No summary.json files found under {args.indir}")

    best = pick_best(df)

    df.to_csv(args.outdir / "all_configs.csv", index=False)
    best.to_csv(args.outdir / "best_per_model.csv", index=False)

    cols = ["model", "config", "pearson_r", "mae_paired", "spearman_rho", "n_matched"]
    print("\nBest config per model family (ranked by Pearson r, tiebreak MAE):")
    print(best[cols].to_string(index=False))

    plot_grid(df, best, args.outdir / "best_per_model.png")


if __name__ == "__main__":
    main()
