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

import matplotlib.pyplot as plt

from uncertainty_calibration import reliability_bins, expected_calibration_error


DISPLAY_NAMES = {
    "cyclegan":       "CycleGAN",
    "unit":           "UNIT",
    "munit":          "MUNIT",
    "dclgan":         "DCLGAN",
    "uvcgan":         "UVCGAN",
    "cyclediffusion": "CycleDiffusion",
}


def _save_reliability(bin_mean_u, bin_mean_e, ece, model_name, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.plot(bin_mean_u, bin_mean_e, "o-", color="C0", label="bin means")
    ax.set_xlabel("Mean uncertainty per tile (bin, normalised)")
    ax.set_ylabel("Mean error per tile (bin, normalised)")
    ax.set_title(f"{DISPLAY_NAMES.get(model_name, model_name)}   ECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def _save_spearman_hist(rho_within, rho_within_mean, model_name, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = rho_within[np.isfinite(rho_within)]
    ax.hist(valid, bins=30, color="C2", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.axvline(rho_within_mean, color="black", linestyle="--",
               label=f"mean = {rho_within_mean:.3f}")
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Within-tile Spearman $\\rho$ (uncertainty vs error)")
    ax.set_ylabel("Tile count")
    ax.set_title(f"{DISPLAY_NAMES.get(model_name, model_name)}   N = {len(valid)}   mean $\\rho$ = {rho_within_mean:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def _save_across_tile(mean_u_per_tile, mean_e_per_tile, pearson_across, spearman_across, model_name, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean_u_per_tile, mean_e_per_tile, s=15, alpha=0.6,
               edgecolors="black", linewidths=0.3, color="C3")
    ax.set_xlabel("Mean uncertainty per tile")
    ax.set_ylabel("Mean error per tile")
    ax.set_title(f"{DISPLAY_NAMES.get(model_name, model_name)}   $\\rho$ = {pearson_across:.3f}   $r_s$ = {spearman_across:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


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
LAYOUT = [
    ["cyclegan", "unit",   "munit"],
    ["dclgan",   "uvcgan", "cyclediffusion"],
]

# Where per-WSI results sit under the ensemble root. run_calibration_all.sh
# writes `calibration/`; some earlier runs went to `calibration_nolog/` while
# the logging behaviour was being decided, so that name is still accepted rather
# than silently reporting "no per_tile.csv found" on an old tree.
CALIB_SUBDIRS = ("calibration", "calibration_nolog")

# label -> directory holding wsi{NNN}/per_tile.csv. Empty means the default
# six-family layout; --group fills it and bypasses the path construction.
GROUP_ROOTS: dict = {}


def grid_layout(names: list[str], ncol: int = 3) -> list[list[str]]:
    """Row-major grid over however many groups there are.

    The hardcoded 2x3 above is the six model families. Any other grouping — the
    UGAC and grid chains compare one family across five training subsets — needs
    a shape derived from the count, or the last group falls off the figure.
    """
    return [names[i:i + ncol] for i in range(0, len(names), ncol)] or [[]]


def parse_groups(specs: list[str]) -> tuple[list[str], dict[str, Path]]:
    """Parse `LABEL=/path/to/calibration/{model}` pairs, order preserved.

    PATH is the directory that holds `wsi{NNN}/per_tile.csv`, so the caller
    supplies the layout and this module needs to know none of it.
    """
    order: list[str] = []
    roots: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--group expects LABEL=PATH, got {spec!r}. PATH is the "
                f"directory holding wsi{{NNN}}/per_tile.csv."
            )
        label, path = spec.split("=", 1)
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise SystemExit(f"--group has an empty label or path: {spec!r}")
        if label in roots:
            raise SystemExit(f"--group label {label!r} given twice")
        order.append(label)
        roots[label] = Path(path)
    return order, roots


def aggregate_model(
    model: str,
    base: Path,
    outdir: Path,
    n_bins: int = 10,
) -> Optional[dict]:
    if GROUP_ROOTS:
        model_size = ""
        calib_root = GROUP_ROOTS[model]
    else:
        model_size = MODEL_SIZES[model]
        ensemble_root = base / model / "data_large" / model_size
        calib_root = next(
            (ensemble_root / sub / model for sub in CALIB_SUBDIRS
             if (ensemble_root / sub / model).is_dir()),
            ensemble_root / CALIB_SUBDIRS[0] / model,
        )

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

    _save_reliability(bin_u, bin_e, ece, model, model_outdir / "reliability.png")
    _save_spearman_hist(rho_w, summary["within_tile"]["spearman_mean"], model,
                        model_outdir / "spearman_hist.png")
    _save_across_tile(df["mean_u"].values, df["mean_e"].values,
                      across_pearson, across_spearman, model,
                      model_outdir / "across_tile.png")

    plot_data = {
        "bin_u":           bin_u,
        "bin_e":           bin_e,
        "ece":             ece,
        "rho_w":           rho_w,
        "rho_within_mean": summary["within_tile"]["spearman_mean"],
        "mean_u":          df["mean_u"].values,
        "mean_e":          df["mean_e"].values,
        "pearson_across":  across_pearson,
        "spearman_across": across_spearman,
    }
    return summary, plot_data


def _apply_grid_visibility(ax, row, col):
    if row == 0:
        ax.tick_params(labelbottom=False, bottom=False)
        ax.set_xlabel("")
    if col > 0:
        ax.tick_params(labelleft=False, left=False)
        ax.set_ylabel("")


def make_combined_reliability(all_plot_data: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, row_models in enumerate(LAYOUT):
        for c, model in enumerate(row_models):
            ax = axes[r, c]
            if model not in all_plot_data:
                ax.set_visible(False)
                continue
            d = all_plot_data[model]
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
            ax.plot(d["bin_u"], d["bin_e"], "o-", color="C0")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{DISPLAY_NAMES.get(model, model)}   ECE = {d['ece']:.4f}")
            ax.set_xlabel("Mean uncertainty per tile (bin, normalised)")
            ax.set_ylabel("Mean error per tile (bin, normalised)")
            ax.grid(alpha=0.3)
            _apply_grid_visibility(ax, r, c)
    fig.tight_layout()
    outpath = outdir / "combined_reliability.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def make_combined_spearman_hist(all_plot_data: dict, outdir: Path) -> None:
    all_rho = np.concatenate([
        all_plot_data[m]["rho_w"][np.isfinite(all_plot_data[m]["rho_w"])]
        for m in MODELS if m in all_plot_data
    ])
    bins = np.linspace(all_rho.min(), all_rho.max(), 31)
    x_lim = (bins[0] - (bins[-1] - bins[0]) * 0.02,
              bins[-1] + (bins[-1] - bins[0]) * 0.02)
    y_max = max(
        np.histogram(all_plot_data[m]["rho_w"][np.isfinite(all_plot_data[m]["rho_w"])], bins=bins)[0].max()
        for m in MODELS if m in all_plot_data
    )
    y_lim = (0, y_max * 1.1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, row_models in enumerate(LAYOUT):
        for c, model in enumerate(row_models):
            ax = axes[r, c]
            if model not in all_plot_data:
                ax.set_visible(False)
                continue
            d = all_plot_data[model]
            valid = d["rho_w"][np.isfinite(d["rho_w"])]
            ax.hist(valid, bins=bins, color="C2", alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.axvline(d["rho_within_mean"], color="black", linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(f"{DISPLAY_NAMES.get(model, model)}   N = {len(valid)}   mean $\\rho$ = {d['rho_within_mean']:.3f}")
            ax.set_xlabel("Within-tile Spearman $\\rho$ (uncertainty vs error)")
            ax.set_ylabel("Tile count")
            ax.grid(alpha=0.3)
            _apply_grid_visibility(ax, r, c)
    fig.tight_layout()
    outpath = outdir / "combined_spearman_hist.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def make_combined_across_tile(all_plot_data: dict, outdir: Path) -> None:
    all_u = np.concatenate([all_plot_data[m]["mean_u"] for m in MODELS if m in all_plot_data])
    all_e = np.concatenate([all_plot_data[m]["mean_e"] for m in MODELS if m in all_plot_data])
    all_u = all_u[np.isfinite(all_u)]
    all_e = all_e[np.isfinite(all_e)]
    u_pad = (all_u.max() - all_u.min()) * 0.05
    e_pad = (all_e.max() - all_e.min()) * 0.05
    u_lim = (all_u.min() - u_pad, all_u.max() + u_pad)
    e_lim = (all_e.min() - e_pad, all_e.max() + e_pad)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, row_models in enumerate(LAYOUT):
        for c, model in enumerate(row_models):
            ax = axes[r, c]
            if model not in all_plot_data:
                ax.set_visible(False)
                continue
            d = all_plot_data[model]
            ax.scatter(d["mean_u"], d["mean_e"], s=15, alpha=0.6,
                       edgecolors="black", linewidths=0.3, color="C3")
            ax.set_xlim(u_lim)
            ax.set_ylim(e_lim)
            ax.set_title(f"{DISPLAY_NAMES.get(model, model)}   $\\rho$ = {d['pearson_across']:.3f}   $r_s$ = {d['spearman_across']:.3f}")
            ax.set_xlabel("Mean uncertainty per tile")
            ax.set_ylabel("Mean error per tile")
            ax.grid(alpha=0.3)
            _apply_grid_visibility(ax, r, c)
    fig.tight_layout()
    outpath = outdir / "combined_across_tile.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def main() -> None:
    # Declared up front: N_WSIS is read below as an argparse default, and Python
    # rejects a `global` that follows any use of the name in the same scope.
    global MODELS, MODEL_SIZES, LAYOUT, N_WSIS, GROUP_ROOTS

    ap = argparse.ArgumentParser(
        description="Aggregate per-WSI calibration CSVs into per-model summaries."
    )
    ap.add_argument(
        "--group", action="append", default=None, metavar="LABEL=PATH",
        help="Aggregate explicitly named groups instead of the six model "
             "families. PATH is the directory holding wsi{NNN}/per_tile.csv. "
             "Repeat once per group; order is preserved. Use this for any "
             "layout without the scaling study's {model}/data_large/{size}/ "
             "tree — the UGAC and grid chains group by training subset.",
    )
    ap.add_argument(
        "--n_wsis", type=int, default=N_WSIS,
        help=f"WSIs expected per group; a missing one is warned about rather "
             f"than passed over silently (default: {N_WSIS}).",
    )
    ap.add_argument(
        "--base", type=Path, required=False, default=None,
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

    N_WSIS = args.n_wsis
    if args.group:
        # The group list IS these constants, so redefining the groups means
        # rebinding them — every helper below reads them at call time.
        MODELS, GROUP_ROOTS = parse_groups(args.group)
        MODEL_SIZES = {m: "" for m in MODELS}
        LAYOUT = grid_layout(MODELS)
    elif args.base is None:
        ap.error("give --base for the six-family layout, or --group LABEL=PATH")

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_plot_data = {}
    for model in MODELS:
        print(f"\n=== {model} ===")
        result = aggregate_model(model, args.base, args.outdir, n_bins=args.n_bins)
        if result is None:
            continue
        summary, plot_data = result
        all_plot_data[model] = plot_data
        all_rows.append({
            "model":           summary["model"],
            "model_size":      summary["model_size"],
            "n_wsi":           summary["n_wsi"],
            "n_tiles":         summary["n_tiles"],
            "spearman_mean":   summary["within_tile"]["spearman_mean"],
            "spearman_std":    summary["within_tile"]["spearman_std"],
            "spearman_median": summary["within_tile"]["spearman_median"],
            "across_pearson":  summary["across_tile"]["pearson"],
            "across_spearman": summary["across_tile"]["spearman"],
            "ece":             summary["reliability"]["ece"],
        })

    if all_rows:
        csv_path = args.outdir / "all_models.csv"
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nSaved all-models table → {csv_path}")

    if all_plot_data:
        print("\n=== Combined plots ===")
        make_combined_reliability(all_plot_data, args.outdir)
        make_combined_spearman_hist(all_plot_data, args.outdir)
        make_combined_across_tile(all_plot_data, args.outdir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
