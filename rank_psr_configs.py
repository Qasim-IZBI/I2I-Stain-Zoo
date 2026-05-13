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
best_per_model.png   — single panel: x = model family, y = MAE ±1 std error bars,
                       colour = model size, line style = data size,
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


DATASIZES        = ["small", "medium", "large"]
MODEL_SIZES      = ["small", "medium", "large"]
SIZE_COLORS      = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
DATA_SIZE_STYLES = {"small": "-",      "medium": "--",       "large": ":"}
MODEL_MARKERS    = ["o", "s", "^", "D", "v", "P"]


def _parse_config(config: str):
    """Return (model_size, data_size) from 'small_model/medium_data'."""
    left, right = config.split("/")
    return left.replace("_model", ""), right.replace("_data", "")


def plot_merged(df: pd.DataFrame, best: pd.DataFrame, outpath: Path) -> None:
    """Single-panel MAE plot: x = model family, colour = model size, style = data size."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models = [m for m in MODELS if m in df["model"].values]

    df = df.copy()
    df[["model_size", "data_size"]] = df["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )
    best = best.copy()
    best[["model_size", "data_size"]] = best["config"].apply(
        lambda c: pd.Series(_parse_config(c))
    )

    x_pos = {m: i for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(10, 5))

    # 9 lines: 3 model sizes × 3 data sizes
    for model_size in MODEL_SIZES:
        for data_size in DATASIZES:
            xs, ys, lowers, uppers = [], [], [], []
            for model in models:
                row = df[
                    (df["model"] == model) &
                    (df["model_size"] == model_size) &
                    (df["data_size"] == data_size)
                ].dropna(subset=["mae_paired"])
                if row.empty:
                    continue
                mae = row["mae_paired"].iloc[0]
                std = row["mae_paired_std"].iloc[0] if "mae_paired_std" in row.columns else 0
                if std is None or (isinstance(std, float) and np.isnan(std)):
                    std = 0
                xs.append(x_pos[model])
                ys.append(mae)
                lowers.append(min(float(std), float(mae)))  # clip so bar never goes below 0
                uppers.append(float(std))
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=[lowers, uppers],
                color=SIZE_COLORS[model_size],
                linestyle=DATA_SIZE_STYLES[data_size],
                marker="o", markersize=5,
                markeredgecolor="black", markeredgewidth=0.5,
                ecolor="gray", elinewidth=1, capsize=3, alpha=0.85,
            )

    # star = best config per model
    for _, brow in best.iterrows():
        model = brow["model"]
        if model not in x_pos:
            continue
        mae = brow.get("mae_paired")
        model_size = brow.get("model_size")
        if mae is None or pd.isna(mae) or model_size is None:
            continue
        ax.scatter(
            [x_pos[model]], [mae],
            color=SIZE_COLORS.get(model_size, "gray"),
            marker="*", s=260, edgecolors="black", linewidths=0.8, zorder=5,
        )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(list(x_pos.keys()), fontsize=9)
    ax.set_ylabel("MAE (paired) ± std", fontsize=9)
    ax.set_ylim(bottom=0)

    # ---- legend ----
    color_handles = [
        Patch(facecolor=SIZE_COLORS[s], edgecolor="black", linewidth=0.7,
              label=f"{s} model")
        for s in MODEL_SIZES
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle=DATA_SIZE_STYLES[s],
               linewidth=1.5, label=f"{s} data")
        for s in DATASIZES
    ]
    star_handle = [
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="best config (lowest MAE)")
    ]
    ax.legend(
        handles=color_handles + style_handles + star_handle,
        fontsize=8, loc="upper right",
        title="Colour = model size  |  Style = data size",
        title_fontsize=7.5, frameon=True,
    )

    ax.set_title(
        "PSR MAE (paired, ±1 std) — colour = model size   style = data size   "
        "star = best config per model",
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

    plot_merged(df, best, args.outdir / "best_per_model.png")


if __name__ == "__main__":
    main()
