"""
plot_ranking_correlation.py — Spearman rank correlation table between
FID, LPIPS, Patch-SSIM, and CPA-MAE rankings across all configurations.

Each of the 54 configs (6 models × 3 model sizes × 3 data sizes) is ranked
independently per metric (rank 1 = best). Spearman ρ is then computed between
every pair of ranking vectors, giving a 4×4 correlation matrix.

Ranking direction:
  FID      — ascending  (lower is better)
  LPIPS    — ascending  (lower is better)
  Patch-SSIM — descending (higher is better)
  CPA-MAE  — ascending  (lower is better)

Input
-----
  combined_metrics.csv  — produced by plot_combined_metrics.py

Usage
-----
  python plot_ranking_correlation.py \
      --csv    ./combined_metrics_plot/combined_metrics.csv \
      --outdir ./combined_metrics_plot/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRIC_COLS = {
    "Patch-SSIM": ("patch_ssim", False),   # (column, ascending)
    "LPIPS":      ("lpips",      True),
    "FID":        ("fid",        True),
    "CPA-MAE":    ("mae_paired", True),
}


def compute_ranking_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    corr_df  : 4×4 Spearman ρ matrix
    pval_df  : 4×4 p-value matrix
    """
    ranked = pd.DataFrame(index=df.index)
    for label, (col, ascending) in METRIC_COLS.items():
        valid = df[col].notna()
        ranked[label] = np.nan
        ranked.loc[valid, label] = df.loc[valid, col].rank(ascending=ascending)

    labels = list(METRIC_COLS.keys())
    n = len(labels)
    rho_mat  = np.full((n, n), np.nan)
    pval_mat = np.full((n, n), np.nan)

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            pair = ranked[[a, b]].dropna()
            if len(pair) < 3:
                continue
            result = spearmanr(pair[a].values, pair[b].values)
            # scipy ≥1.9 may return arrays instead of scalars
            rho_mat[i, j]  = float(np.atleast_1d(result.statistic).flat[0])
            pval_mat[i, j] = float(np.atleast_1d(result.pvalue).flat[0])

    corr_df = pd.DataFrame(rho_mat,  index=labels, columns=labels)
    pval_df = pd.DataFrame(pval_mat, index=labels, columns=labels)
    return corr_df, pval_df


def sig_marker(pval: float) -> str:
    if np.isnan(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def plot_table(corr_df: pd.DataFrame, pval_df: pd.DataFrame, n_configs: int,
               outpath: Path) -> None:
    labels = list(corr_df.columns)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")

    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            rho  = corr_df.iloc[i, j]
            pval = pval_df.iloc[i, j]
            color = cmap(norm(rho)) if not np.isnan(rho) else (0.9, 0.9, 0.9, 1)
            rect = plt.Rectangle(
                [j, n - 1 - i], 1, 1,
                facecolor=color, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)

            if i == j:
                txt = "1.00"
                marker = ""
            else:
                txt    = f"{rho:.2f}" if not np.isnan(rho) else "—"
                marker = sig_marker(pval)

            # value in bold on top line, significance marker below
            ax.text(
                j + 0.5, n - i - 0.42, txt,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="black" if abs(rho) < 0.7 else "white",
            )
            if marker:
                ax.text(
                    j + 0.5, n - i - 0.72, marker,
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(rho) < 0.7 else "white",
                )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([x + 0.5 for x in range(n)])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([y + 0.5 for y in range(n)])
    ax.set_yticklabels(reversed(labels), fontsize=10)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Spearman rank correlation table between metric rankings "
                    "across all 54 configurations."
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

    # drop rows missing all four metrics
    metric_cols = [c for _, (c, _) in METRIC_COLS.items()]
    df = df.dropna(subset=metric_cols, how="all").reset_index(drop=True)
    n_configs = len(df)
    print(f"Configs with at least one metric: {n_configs}")

    corr_df, pval_df = compute_ranking_matrix(df)

    csv_path = args.outdir / "ranking_correlation.csv"
    corr_df.to_csv(csv_path)
    print(f"Saved → {csv_path}")

    pval_path = args.outdir / "ranking_correlation_pvalues.csv"
    pval_df.to_csv(pval_path)
    print(f"Saved → {pval_path}")

    plot_table(corr_df, pval_df, n_configs, args.outdir / "ranking_correlation.png")


if __name__ == "__main__":
    main()
