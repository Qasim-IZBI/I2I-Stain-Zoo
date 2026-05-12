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
                       cols = x-axis (data size / model size); colour = model family,
                       line style + marker = complementary size axis,
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


DATASIZES    = ["small", "medium", "large"]
MODEL_SIZES  = ["small", "medium", "large"]
SIZE_MARKERS = {"small": "o", "medium": "s", "large": "^"}
SIZE_STYLES  = {"small": "-", "medium": "--", "large": ":"}


def _parse_config(config: str):
    """Return (model_size, data_size) from 'small_model/medium_data'."""
    left, right = config.split("/")
    return left.replace("_model", ""), right.replace("_data", "")


def _metric_panel(ax, df: pd.DataFrame, best: pd.DataFrame,
                  metric: str, x_col: str, style_col: str,
                  model_colors: dict, xlabel: str, ylabel: str) -> None:
    """Draw one metric panel. x_col varies along x; style_col drives line+marker style."""
    x_pos = {s: i for i, s in enumerate(MODEL_SIZES)}

    for model in [m for m in MODELS if m in df["model"].values]:
        mdf = df[df["model"] == model].dropna(subset=[metric])
        for style_val in MODEL_SIZES:
            sub = mdf[mdf[style_col] == style_val].copy()
            if sub.empty:
                continue
            sub = sub.assign(_x=sub[x_col].map(x_pos)).sort_values("_x")
            ax.plot(sub["_x"], sub[metric],
                    color=model_colors[model],
                    linestyle=SIZE_STYLES[style_val],
                    marker=SIZE_MARKERS[style_val],
                    linewidth=1.0, markersize=7, alpha=0.85)

    # star = best config per model
    for _, brow in best.iterrows():
        xi = x_pos.get(brow[x_col])
        val = brow.get(metric)
        if xi is not None and val is not None and not pd.isna(val):
            ax.scatter([xi], [val],
                       color=model_colors.get(brow["model"], "gray"),
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

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # row 0 — Pearson r
    _metric_panel(axes[0, 0], df, best, "pearson_r",
                  x_col="data_size",  style_col="model_size",
                  model_colors=model_colors,
                  xlabel="Data size", ylabel="Pearson r")
    axes[0, 0].set_title("Pearson r  vs  data size\n(line style = model size)", fontsize=9)

    _metric_panel(axes[0, 1], df, best, "pearson_r",
                  x_col="model_size", style_col="data_size",
                  model_colors=model_colors,
                  xlabel="Model size", ylabel="Pearson r")
    axes[0, 1].set_title("Pearson r  vs  model size\n(line style = data size)", fontsize=9)

    # row 1 — MAE
    _metric_panel(axes[1, 0], df, best, "mae_paired",
                  x_col="data_size",  style_col="model_size",
                  model_colors=model_colors,
                  xlabel="Data size", ylabel="MAE (paired)")
    axes[1, 0].set_title("MAE  vs  data size\n(line style = model size)", fontsize=9)

    _metric_panel(axes[1, 1], df, best, "mae_paired",
                  x_col="model_size", style_col="data_size",
                  model_colors=model_colors,
                  xlabel="Model size", ylabel="MAE (paired)")
    axes[1, 1].set_title("MAE  vs  model size\n(line style = data size)", fontsize=9)

    # ---- legend (outside, right side) ----
    color_handles = [Patch(facecolor=model_colors[m], edgecolor="black",
                           linewidth=0.7, label=m)
                     for m in models]
    style_handles = [Line2D([0], [0], color="gray",
                            linestyle=SIZE_STYLES[s], marker=SIZE_MARKERS[s],
                            markersize=7, label=s)
                     for s in MODEL_SIZES]
    star_handle   = [Line2D([0], [0], marker="*", color="w",
                            markerfacecolor="gray", markeredgecolor="black",
                            markersize=12, label="best config")]

    fig.legend(handles=color_handles + [Line2D([], [], linestyle="none")] +
               style_handles + star_handle,
               fontsize=8, loc="center right", bbox_to_anchor=(1.13, 0.5),
               title="Model family  |  Size  |  Best", title_fontsize=8,
               frameon=True)

    fig.suptitle(
        "PSR paired metrics — colour = model family   "
        "line style/marker = complementary size axis   star = best config per model",
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
