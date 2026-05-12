"""
rank_psr_configs.py — find the best model size / data size config per model family.

Reads summary.json files produced by compare_psr_all_configs.sh (one per model),
ranks the 3 large-data-size configs (small/medium/large model) by MAE ascending,
tiebreak Pearson r descending, and reports the winner per model.
Only large data size is considered — intended for uncertainty analysis where
the largest available dataset is used.

Outputs
-------
best_per_model.csv   — one row per model: best config + its paired metrics
all_configs.csv      — all configs × models with paired metrics (for inspection)
best_per_model.png   — 1×2 line+marker grid: cols = x-axis (data size / model size);
                       MAE ±1 std error bars, colour = complementary size axis,
                       marker = model family, fixed x-offset per (model, size) combo,
                       star = best config per model (lowest MAE)
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
                "mae_paired_std":   m.get("mae_paired_std"),
                "mean_paired_diff": m.get("mean_paired_diff_generated_minus_real"),
                "n_matched":        m.get("n_matched"),
            })
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["config"].apply(lambda c: pd.Series(_parse_config(c)))
    parsed.columns = ["model_size", "data_size"]
    df = df.copy()
    df[["model_size", "data_size"]] = parsed
    return (
        df[df["data_size"] == "large"]
          .dropna(subset=["mae_paired"])
          .sort_values(["mae_paired", "pearson_r"], ascending=[True, False])
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
                  x_col: str, style_col: str,
                  model_markers: dict, xlabel: str) -> None:
    """Draw one MAE panel with ±1 std error bars.
    x_col varies along x; style_col is the complementary size axis (drives colour).
    Each (model, style_val) combination gets a fixed x-offset so lines don't overlap."""
    x_pos   = {s: i for i, s in enumerate(MODEL_SIZES)}
    models  = [m for m in MODELS if m in df["model"].values]
    combos  = [(m, sv) for m in models for sv in MODEL_SIZES]
    n       = len(combos)
    offsets = {c: (i - (n - 1) / 2) * (0.2 / n) for i, c in enumerate(combos)}

    for model, style_val in combos:
        mdf = df[(df["model"] == model) & (df[style_col] == style_val)].dropna(
            subset=["mae_paired"])
        if mdf.empty:
            continue
        mdf = mdf.assign(_x=mdf[x_col].map(x_pos)).sort_values("_x")
        off = offsets[(model, style_val)]
        xs  = mdf["_x"] + off
        ys  = mdf["mae_paired"]
        std_col  = "mae_paired_std"
        err      = mdf[std_col].fillna(0).values if std_col in mdf.columns else np.zeros(len(ys))
        lower    = np.minimum(err, ys.values)   # clip so bar never goes below 0
        upper    = err
        ax.errorbar(xs, ys, yerr=[lower, upper],
                    color=SIZE_COLORS[style_val],
                    marker=model_markers[model],
                    linestyle="none", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5,
                    ecolor="black", elinewidth=1.5, capsize=5, alpha=0.85)

    # star = best config per model
    for _, brow in best.iterrows():
        xi  = x_pos.get(brow[x_col])
        val = brow.get("mae_paired")
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
    ax.set_ylabel("MAE (paired) ± std", fontsize=9)
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    _metric_panel(axes[0], df, best,
                  x_col="data_size",  style_col="model_size",
                  model_markers=model_markers, xlabel="Data size")
    axes[0].set_title("MAE  vs  data size\n(colour = model size)", fontsize=9)

    _metric_panel(axes[1], df, best,
                  x_col="model_size", style_col="data_size",
                  model_markers=model_markers, xlabel="Model size")
    axes[1].set_title("MAE  vs  model size\n(colour = data size)", fontsize=9)

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
        "PSR MAE (paired, ±1 std) — colour = complementary size axis   "
        "marker = model family   star = best config per model (lowest MAE)",
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

    cols = ["model", "config", "mae_paired", "mae_paired_std", "pearson_r", "n_matched"]
    print("\nBest config per model family — large data size only, ranked by MAE (tiebreak Pearson r):")
    print(best[cols].to_string(index=False))

    plot_grid(df, best, args.outdir / "best_per_model.png")


if __name__ == "__main__":
    main()
