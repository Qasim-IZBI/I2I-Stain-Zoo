"""
rank_psr_configs.py — find the best model size / data size config per model family.

Reads summary.json files produced by compare_psr_all_configs.sh (one per model),
ranks the 9 configs (3 model sizes × 3 data sizes) by Spearman ρ descending,
tiebreak MAE ascending, and reports the winner per model.

Outputs
-------
best_per_model.csv   — one row per model: best config + its paired metrics
all_configs.csv      — all configs × models with paired metrics (for inspection)
best_per_model.png   — 2×3 scatter grid: Spearman ρ vs MAE, best config in red
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]


def load_all(indir: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
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
        df.dropna(subset=["spearman_rho"])
          .sort_values(["spearman_rho", "mae_paired"], ascending=[False, True])
          .groupby("model", sort=False)
          .first()
          .reset_index()
    )


def plot_grid(df: pd.DataFrame, best: pd.DataFrame, outpath: Path) -> None:
    models  = [m for m in MODELS if m in df["model"].values]
    ncols   = 3
    nrows   = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax, model in zip(axes.flat, models):
        sub      = df[df["model"] == model].copy()
        best_row = best[best["model"] == model]

        valid = sub.dropna(subset=["spearman_rho", "mae_paired"])
        ax.scatter(valid["mae_paired"], valid["spearman_rho"],
                   color="steelblue", s=60, edgecolors="black",
                   linewidths=0.7, zorder=2)

        for _, row in valid.iterrows():
            ax.annotate(row["config"],
                        (row["mae_paired"], row["spearman_rho"]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points", color="#333333")

        if not best_row.empty:
            ax.scatter(best_row["mae_paired"], best_row["spearman_rho"],
                       color="red", s=140, edgecolors="black",
                       linewidths=1.0, zorder=3,
                       label=f"best: {best_row['config'].values[0]}")
            ax.legend(fontsize=7, loc="lower right")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("MAE (paired)", fontsize=8)
        ax.set_ylabel("Spearman ρ", fontsize=8)
        ax.set_title(model, fontsize=10, fontweight="bold")

    for ax in axes.flat[len(models):]:
        ax.set_visible(False)

    fig.suptitle(
        "Best config per model family — Spearman ρ (↑) vs MAE (←)\n"
        "Red = winner (highest ρ, tiebreak: lowest MAE)",
        fontsize=10,
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
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df   = load_all(args.indir)
    if df.empty:
        raise RuntimeError(f"No summary.json files found under {args.indir}")

    best = pick_best(df)

    df.to_csv(args.outdir / "all_configs.csv", index=False)
    best.to_csv(args.outdir / "best_per_model.csv", index=False)

    cols = ["model", "config", "spearman_rho", "mae_paired", "pearson_r", "n_matched"]
    print("\nBest config per model family (ranked by Spearman ρ, tiebreak MAE):")
    print(best[cols].to_string(index=False))

    plot_grid(df, best, args.outdir / "best_per_model.png")


if __name__ == "__main__":
    main()
