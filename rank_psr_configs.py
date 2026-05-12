"""
rank_psr_configs.py — find the best model size / data size config per model family.

Reads summary.json files produced by compare_psr_all_configs.sh (one per model),
ranks the 9 configs (3 model sizes × 3 data sizes) by Spearman ρ descending,
tiebreak MAE ascending, and reports the winner per model.

Outputs
-------
best_per_model.csv   — one row per model: best config + its paired metrics
all_configs.csv      — all configs × models with paired metrics (for inspection)
best_per_model.png   — 2×2 line+marker grid: rows = metric (Pearson r / MAE),
                       cols = x-axis (data size / model size); colour = complementary
                       size axis, marker = model family, fixed x-offset per
                       (model, size) combo to prevent overlap, star = best config
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


DATASIZES     = ["small", "medium", "large"]
MODEL_SIZES   = ["small", "medium", "large"]
SIZE_COLORS   = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _parse_config(config: str):
    """Return (model_size, data_size) from 'small_model/medium_data'."""
    left, right = config.split("/")
    return left.replace("_model", ""), right.replace("_data", "")


def _metric_panel(ax, df: pd.DataFrame, best: pd.DataFrame,
                  metric: str, x_col: str, style_col: str,
                  model_markers: dict, xlabel: str, ylabel: str) -> None:
    """Draw one metric panel.
    x_col varies along x; style_col is the complementary size axis (drives colour).
    Each (model, style_val) combination gets a fixed x-offset so lines don't overlap."""
    x_pos    = {s: i for i, s in enumerate(MODEL_SIZES)}
    models   = [m for m in MODELS if m in df["model"].values]
    combos   = [(m, sv) for m in models for sv in MODEL_SIZES]
    n        = len(combos)
    offsets  = {c: (i - (n - 1) / 2) * (0.7 / n) for i, c in enumerate(combos)}

    for model, style_val in combos:
        mdf = df[(df["model"] == model) & (df[style_col] == style_val)].dropna(
            subset=[metric])
        if mdf.empty:
            continue
        mdf = mdf.assign(_x=mdf[x_col].map(x_pos)).sort_values("_x")
        off = offsets[(model, style_val)]
        ax.plot(mdf["_x"] + off, mdf[metric],
                color=SIZE_COLORS[style_val],
                marker=model_markers[model],
                linestyle="-", linewidth=0.9, markersize=6,
                markeredgecolor="black", markeredgewidth=0.5,
                alpha=0.85)

    # star = best config per model
    for _, brow in best.iterrows():
        xi  = x_pos.get(brow[x_col])
        val = brow.get(metric)
        sv  = brow.get(style_col)
        if xi is None or val is None or pd.isna(val):
            continue
        off = offsets.get((brow["model"], sv), 0)
        ax.scatter([xi + off], [val],
                   color=SIZE_COLORS.get(sv, "gray"),
                   marker="*", s=220, edgecolors="black",
                   linewidths=0.8, zorder=5)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(list(x_pos.keys()), fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if metric == "pearson_r":
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    if metric == "mae_paired":
        ax.set_ylim(bottom=0)


def plot_grid(df: pd.DataFrame, best: pd.DataFrame, outpath: Path) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models        = [m for m in MODELS if m in df["model"].values]
    model_markers = {m: MODEL_MARKERS[i] for i, m in enumerate(models)}

    df = df.copy()
    df[["model_size", "data_size"]] = df["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )
    best = best.copy()
    best[["model_size", "data_size"]] = best["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # row 0 — Pearson r
    _metric_panel(axes[0, 0], df, best, "pearson_r",
                  x_col="data_size",  style_col="model_size",
                  model_markers=model_markers,
                  xlabel="Data size", ylabel="Pearson r")
    axes[0, 0].set_title("Pearson r  vs  data size\n(colour = model size)", fontsize=9)

    _metric_panel(axes[0, 1], df, best, "pearson_r",
                  x_col="model_size", style_col="data_size",
                  model_markers=model_markers,
                  xlabel="Model size", ylabel="Pearson r")
    axes[0, 1].set_title("Pearson r  vs  model size\n(colour = data size)", fontsize=9)

    # row 1 — MAE
    _metric_panel(axes[1, 0], df, best, "mae_paired",
                  x_col="data_size",  style_col="model_size",
                  model_markers=model_markers,
                  xlabel="Data size", ylabel="MAE (paired)")
    axes[1, 0].set_title("MAE  vs  data size\n(colour = model size)", fontsize=9)

    _metric_panel(axes[1, 1], df, best, "mae_paired",
                  x_col="model_size", style_col="data_size",
                  model_markers=model_markers,
                  xlabel="Model size", ylabel="MAE (paired)")
    axes[1, 1].set_title("MAE  vs  model size\n(colour = data size)", fontsize=9)

    # ---- legend (outside, right side) ----
    color_handles  = [Patch(facecolor=SIZE_COLORS[s], edgecolor="black",
                            linewidth=0.7, label=s)
                      for s in MODEL_SIZES]
    marker_handles = [Line2D([0], [0], color="gray",
                             marker=model_markers[m], markersize=7,
                             markeredgecolor="black", markeredgewidth=0.5,
                             linestyle="-", label=m)
                      for m in models]
    star_handle    = [Line2D([0], [0], marker="*", color="w",
                             markerfacecolor="gray", markeredgecolor="black",
                             markersize=12, label="best config")]

    fig.legend(handles=color_handles + [Line2D([], [], linestyle="none")] +
               marker_handles + star_handle,
               fontsize=8, loc="center right", bbox_to_anchor=(1.15, 0.5),
               title="Size (colour)  |  Model (marker)  |  Best",
               title_fontsize=8, frameon=True)

    fig.suptitle(
        "PSR paired metrics — colour = complementary size axis   "
        "marker = model family   star = best config per model",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
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
