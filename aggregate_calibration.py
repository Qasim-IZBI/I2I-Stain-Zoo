"""Aggregate per-WSI uncertainty calibration results into per-model summaries.

Reads the per_tile.csv files written by run_calibration_all.sh for each
model (5 WSIs × 6 models) and re-computes all metrics on the combined
tile pool. This gives the correct per-model statistics: the Spearman
distribution, across-tile correlations, and reliability diagram are all
computed from all tiles together rather than averaged over per-WSI summaries.

Outputs per model:
  {outdir}/{model}/summary.json    — aggregated calibration metrics
  {outdir}/{model}/calibration.png — 4-panel figure (same layout as per-WSI)

Outputs overall:
  {outdir}/all_models.csv          — one row per model for quick comparison

Usage
-----
python aggregate_calibration.py \\
    --base /work2/bz66izin-VSproject/ensemble \\
    --outdir ./calibration_combined/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from uncertainty_calibration import reliability_bins, expected_calibration_error, make_plot


MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]
MODEL_SIZES = {
    "cyclegan":       "model_medium",
    "unit":           "model_medium",
    "munit":          "model_medium",
    "dclgan":         "model_small",
    "uvcgan":         "model_small",
    "cyclediffusion": "model_small",
}
N_WSIS = 5


def aggregate_model(
    model: str,
    base: Path,
    outdir: Path,
    n_bins: int = 10,
) -> Optional[dict]:
    model_size = MODEL_SIZES[model]
    ensemble_root = base / model / "data_large" / model_size
    calib_root = ensemble_root / "calibration" / model

    # Collect per_tile.csv from all WSIs
    dfs = []
    for wsi_num in range(1, N_WSIS + 1):
        csv_path = calib_root / f"wsi{wsi_num:03d}" / "per_tile.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"  [{model}] WARNING: missing {csv_path}")

    if not dfs:
        print(f"  [{model}] No per_tile.csv files found — skipping.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"  [{model}] {len(df)} tiles from {len(dfs)}/{N_WSIS} WSI(s)")

    # --- within-tile Spearman distribution ---
    rho_w = df["spearman_rho"].values
    rho_w_finite = rho_w[np.isfinite(rho_w)]

    # --- across-tile correlations (all tiles pooled) ---
    finite = df[df["mean_u"].notna() & df["mean_e"].notna()]
    if len(finite) >= 3 and finite["mean_u"].std() > 0 and finite["mean_e"].std() > 0:
        across_pearson  = float(pearsonr(finite["mean_u"].values, finite["mean_e"].values).statistic)
        across_spearman = float(spearmanr(finite["mean_u"].values, finite["mean_e"].values).statistic)
    else:
        across_pearson  = float("nan")
        across_spearman = float("nan")

    # --- tile-mean reliability + ECE (all tiles pooled) ---
    u_tile = finite["mean_u"].values
    e_tile = finite["mean_e"].values
    u_lo = float(np.percentile(u_tile, 1))
    u_hi = float(np.percentile(u_tile, 99))
    e_lo = float(np.percentile(e_tile, 1))
    e_hi = float(np.percentile(e_tile, 99))
    bin_u, bin_e, bin_counts = reliability_bins(
        u_tile, e_tile, n_bins, u_lo, u_hi, e_lo, e_hi
    )
    ece = expected_calibration_error(bin_u, bin_e, bin_counts)

    summary = {
        "model":      model,
        "model_size": model_size,
        "n_wsi":      len(dfs),
        "n_tiles":    int(len(df)),
        "n_tiles_finite_rho": int(rho_w_finite.size),
        "within_tile": {
            "spearman_mean":   float(rho_w_finite.mean())       if rho_w_finite.size     else float("nan"),
            "spearman_std":    float(rho_w_finite.std(ddof=1))  if rho_w_finite.size > 1 else 0.0,
            "spearman_median": float(np.median(rho_w_finite))   if rho_w_finite.size     else float("nan"),
        },
        "across_tile": {
            "pearson":  across_pearson,
            "spearman": across_spearman,
        },
        "reliability": {
            "ece":                   ece,
            "bin_mean_u_normalised": bin_u.tolist(),
            "bin_mean_e_normalised": bin_e.tolist(),
            "bin_counts":            bin_counts.tolist(),
            "u_tile_mean_p1_p99":    [u_lo, u_hi],
            "e_tile_mean_p1_p99":    [e_lo, e_hi],
        },
        "params": {
            "n_bins":                    n_bins,
            "reliability_granularity":   "tile_means",
            "n_wsi_loaded":              len(dfs),
        },
    }

    # --- save ---
    model_outdir = outdir / model
    model_outdir.mkdir(parents=True, exist_ok=True)

    json_path = model_outdir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{model}] Saved summary  → {json_path}")

    make_plot(
        bin_mean_u=bin_u,
        bin_mean_e=bin_e,
        ece=ece,
        rho_within=rho_w,
        rho_within_mean=summary["within_tile"]["spearman_mean"],
        mean_u_per_tile=df["mean_u"].values,
        mean_e_per_tile=df["mean_e"].values,
        pearson_across=across_pearson,
        spearman_across=across_spearman,
        outpath=model_outdir / "calibration.png",
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate per-WSI calibration CSVs into per-model summaries."
    )
    ap.add_argument(
        "--base", type=Path, required=True,
        help="Ensemble base directory containing cyclegan/, unit/, etc. subdirectories.",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("calibration_combined"),
        help="Output root for combined results (default: calibration_combined/).",
    )
    ap.add_argument(
        "--n_bins", type=int, default=10,
        help="Number of quantile bins for the reliability diagram and ECE (default: 10).",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        result = aggregate_model(model, args.base, args.outdir, n_bins=args.n_bins)
        if result is None:
            continue
        all_rows.append({
            "model":           result["model"],
            "model_size":      result["model_size"],
            "n_wsi":           result["n_wsi"],
            "n_tiles":         result["n_tiles"],
            "spearman_mean":   result["within_tile"]["spearman_mean"],
            "spearman_std":    result["within_tile"]["spearman_std"],
            "spearman_median": result["within_tile"]["spearman_median"],
            "across_pearson":  result["across_tile"]["pearson"],
            "across_spearman": result["across_tile"]["spearman"],
            "ece":             result["reliability"]["ece"],
        })

    if all_rows:
        csv_path = args.outdir / "all_models.csv"
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nSaved all-models table → {csv_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
