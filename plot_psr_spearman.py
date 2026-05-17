"""
plot_psr_spearman.py — Spearman correlation between PSR-MAE and each image
quality metric (Patch-SSIM, LPIPS, FID), computed per model type across
the 9 configs (3 model sizes × 3 data sizes).

Input
-----
  combined_metrics.csv  — produced by plot_combined_metrics.py

Usage
-----
  python plot_psr_spearman.py \
      --csv  ./combined_metrics_plot/combined_metrics.csv \
      --outdir ./combined_metrics_plot/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]

METRICS = {
    "patch_ssim": {"label": "Patch-SSIM", "color": "#1f77b4"},
    "lpips":      {"label": "LPIPS",      "color": "#ff7f0e"},
    "fid":        {"label": "FID",        "color": "#2ca02c"},
}


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Spearman ρ and p-value per (model, metric)."""
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model].dropna(subset=["mae_paired"])
        for metric, mcfg in METRICS.items():
            pair = sub[["mae_paired", metric]].dropna()
            if len(pair) < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(pair["mae_paired"], pair[metric])
            rows.append({
                "model":  model,
                "metric": metric,
                "label":  mcfg["label"],
                "rho":    rho,
                "pval":   pval,
                "n":      len(pair),
            })
    return pd.DataFrame(rows)


def plot_correlations(corr: pd.DataFrame, outpath: Path) -> None:
    models  = [m for m in MODELS if m in corr["model"].values]
    n_mod   = len(models)
    n_met   = len(METRICS)
    x       = np.arange(n_mod)
    width   = 0.22
    offsets = np.linspace(-(n_met - 1) / 2, (n_met - 1) / 2, n_met) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for (metric, mcfg), off in zip(METRICS.items(), offsets):
        sub = corr[corr["metric"] == metric]
        rhos  = [float(sub[sub["model"] == m]["rho"].iloc[0])
                 if not sub[sub["model"] == m].empty else np.nan
                 for m in models]
        bars = ax.bar(
            x + off, rhos, width=width,
            color=mcfg["color"], label=mcfg["label"],
            edgecolor="black", linewidth=0.6, alpha=0.85,
        )
        # significance markers (* p<0.05, ** p<0.01)
        for bar, model in zip(bars, models):
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            pval = float(row["pval"].iloc[0])
            rho  = float(row["rho"].iloc[0])
            if np.isnan(pval):
                continue
            marker = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
            if marker:
                y_pos = rho + (0.03 if rho >= 0 else -0.06)
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos,
                    marker, ha="center", va="bottom", fontsize=8,
                )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Spearman ρ  (vs PSR-MAE)", fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Spearman correlation between PSR-MAE and image quality metrics per model."
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="combined_metrics.csv produced by plot_combined_metrics.py",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("combined_metrics_plot"),
        help="Output directory [%(default)s]",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    corr = compute_correlations(df)
    csv_path = args.outdir / "psr_spearman.csv"
    corr.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    plot_correlations(corr, args.outdir / "psr_spearman.png")


if __name__ == "__main__":
    main()
