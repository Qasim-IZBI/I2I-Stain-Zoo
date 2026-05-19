"""Uncertainty calibration analysis.

Pairs per-pixel ensemble-uncertainty maps (from uncertainty.py) with per-pixel
cycle-reconstruction error maps (from evaluation.py --metric regen_error
--save_error_npy) and reduces them to numerical calibration scores.

Reports:
  - Within-tile Spearman/Pearson rho   "are uncertain pixels the wrong pixels?"
  - Across-tile Pearson/Spearman rho   "are uncertain tiles the wrong tiles?"
  - AUSE + sparsification curve        standard depth-uncertainty metric
  - Reliability diagram + ECE          monotonicity of error vs uncertainty bin

Outputs (mirrors compare_psr.py / cross_stain_consistency.py convention):
  per_tile.csv, per_wsi.csv (if --tiles_metadata given),
  summary.json, calibration.png

Example
-------
# 1. Produce error .npy maps with model_01 (single member)
python evaluation.py --metric regen_error --path_A /path/to/testA \\
    --model cyclegan --ckpt model_01.pt --direction A2B \\
    --overlay_dir ./regen_cyclegan_m01/ --save_error_npy

# 2. Already have uncertainty maps from uncertainty.py at uncertainty_out/cyclegan/
python uncertainty_calibration.py \\
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \\
    --error_dirs ./regen_cyclegan_m01/error_npy/ \\
    --mask_dir ./tile_masks_flat/ \\
    --tiles_metadata /path/to/testA \\
    --outdir ./calibration_cyclegan/

# Average across N ensemble members:
python uncertainty_calibration.py \\
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \\
    --error_dirs ./regen_m01/error_npy/ ./regen_m02/error_npy/ ... \\
    --mask_dir ./tile_masks_flat/ --outdir ./calibration_cyclegan/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def list_npy_stems(d: Path) -> set[str]:
    return {p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".npy"}


def list_mask_stems(d: Path) -> set[str]:
    return {p.stem for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png"}}


def find_mask_path(mask_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".tif", ".tiff", ".png"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_error_map(error_dirs: list[Path], stem: str) -> np.ndarray:
    """Load one error map (single dir) or per-pixel mean across N dirs."""
    if len(error_dirs) == 1:
        return np.load(error_dirs[0] / f"{stem}.npy").astype(np.float64)
    stack = np.stack(
        [np.load(d / f"{stem}.npy").astype(np.float64) for d in error_dirs],
        axis=0,
    )
    return stack.mean(axis=0)


def load_tissue_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a tissue mask as a boolean array (H, W). Any non-zero pixel is tissue."""
    m = tifffile.imread(str(path)) if path.suffix.lower() in {".tif", ".tiff"} \
        else np.array(plt.imread(str(path)) * 255, dtype=np.uint8)
    if m.ndim > 2:
        m = m[..., 0]
    if m.shape != target_shape:
        raise ValueError(
            f"Mask shape {m.shape} does not match uncertainty shape {target_shape} "
            f"({path}). Resize masks during tiling or pass a matching mask dir."
        )
    return m > 0


def build_stem_to_wsi(tiles_metadata_root: Path) -> dict[str, str]:
    """Walk per-WSI tiles_metadata.csv files and build stem -> source_file map.

    If tile_name collides across WSIs, the offending stems are dropped and warned.
    """
    csvs = sorted(tiles_metadata_root.glob("*/tiles_metadata.csv"))
    if not csvs:
        # User may have passed a single CSV path
        if tiles_metadata_root.is_file():
            csvs = [tiles_metadata_root]
        else:
            raise FileNotFoundError(
                f"No per-WSI tiles_metadata.csv files under {tiles_metadata_root}"
            )

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    counts = df.groupby("tile_name")["source_file"].nunique()
    colliding = set(counts[counts > 1].index)
    if colliding:
        warnings.warn(
            f"{len(colliding)} tile_name(s) appear under multiple WSIs in "
            f"tiles_metadata; per-WSI aggregation will skip them. "
            f"Examples: {sorted(colliding)[:3]}"
        )
    keep = df[~df["tile_name"].isin(colliding)]
    return dict(zip(keep["tile_name"].astype(str), keep["source_file"].astype(str)))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def sparsification_curve(u: np.ndarray, e: np.ndarray, n_steps: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fractions_removed, S_pred, S_oracle).

    S_pred(k):   mean error on the (1-k) least-uncertain pixels.
    S_oracle(k): mean error on the (1-k) lowest-error pixels (oracle ordering).
    A perfectly calibrated u matches the oracle (AUSE = integral of S_pred - S_oracle = 0).
    """
    n = u.size
    e_by_u = e[np.argsort(u)]      # ascending u → keep prefix as we remove top
    e_by_e = e[np.argsort(e)]      # ascending e
    fractions = np.linspace(0.0, 1.0, n_steps + 1)

    # Cumulative means computed once via cumsum (O(n)) then indexed (O(n_steps))
    csum_u = np.cumsum(e_by_u)
    csum_e = np.cumsum(e_by_e)
    s_pred = np.empty_like(fractions)
    s_oracle = np.empty_like(fractions)
    for i, k in enumerate(fractions):
        keep = max(1, int(round((1.0 - k) * n)))
        s_pred[i] = csum_u[keep - 1] / keep
        s_oracle[i] = csum_e[keep - 1] / keep
    return fractions, s_pred, s_oracle


def ause(fractions: np.ndarray, s_pred: np.ndarray, s_oracle: np.ndarray) -> float:
    return float(np.trapz(s_pred - s_oracle, fractions))


def reliability_bins(
    u: np.ndarray,
    e: np.ndarray,
    n_bins: int,
    u_lo: float, u_hi: float,
    e_lo: float, e_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin pixels by quantile of u, return (bin_mean_u_norm, bin_mean_e_norm, bin_counts).

    Both u and e are min-max normalised to [0,1] using global (u_lo,u_hi)/(e_lo,e_hi)
    so the diagonal y=x has meaning.
    """
    u_n = np.clip((u - u_lo) / (u_hi - u_lo + 1e-12), 0.0, 1.0)
    e_n = np.clip((e - e_lo) / (e_hi - e_lo + 1e-12), 0.0, 1.0)
    edges = np.quantile(u_n, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # absorb extremes
    idx = np.digitize(u_n, edges[1:-1])    # bin index 0..n_bins-1
    bin_mean_u = np.zeros(n_bins)
    bin_mean_e = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            bin_mean_u[b] = u_n[sel].mean()
            bin_mean_e[b] = e_n[sel].mean()
            bin_counts[b] = sel.sum()
    return bin_mean_u, bin_mean_e, bin_counts


def expected_calibration_error(
    bin_mean_u: np.ndarray, bin_mean_e: np.ndarray, bin_counts: np.ndarray
) -> float:
    """ECE = sum_b (n_b / N) * |bin_mean_u_norm - bin_mean_e_norm|."""
    total = bin_counts.sum()
    if total == 0:
        return float("nan")
    return float(np.sum((bin_counts / total) * np.abs(bin_mean_u - bin_mean_e)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(
    bin_mean_u: np.ndarray,
    bin_mean_e: np.ndarray,
    ece: float,
    fractions: np.ndarray,
    s_pred_avg: np.ndarray,
    s_oracle_avg: np.ndarray,
    ause_mean: float,
    rho_within: np.ndarray,
    rho_within_mean: float,
    mean_u_per_tile: np.ndarray,
    mean_e_per_tile: np.ndarray,
    pearson_across: float,
    spearman_across: float,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Reliability diagram
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.plot(bin_mean_u, bin_mean_e, "o-", color="C0", label="bin means")
    ax.set_xlabel("Mean uncertainty per tile (bin, normalised)")
    ax.set_ylabel("Mean error per tile (bin, normalised)")
    ax.set_title(f"Reliability diagram (tile means)   ECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Sparsification curve (averaged across tiles)
    ax = axes[0, 1]
    ax.plot(fractions, s_pred_avg, color="C1", label="Predicted (sort by u)")
    ax.plot(fractions, s_oracle_avg, color="gray", linestyle="--", label="Oracle (sort by e)")
    ax.fill_between(fractions, s_pred_avg, s_oracle_avg, color="C1", alpha=0.2)
    ax.set_xlabel("Fraction of pixels removed")
    ax.set_ylabel("Mean residual error")
    ax.set_title(f"Sparsification (avg over tiles)   AUSE = {ause_mean:.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Histogram of within-tile Spearman rho
    ax = axes[1, 0]
    valid = rho_within[np.isfinite(rho_within)]
    ax.hist(valid, bins=30, color="C2", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.axvline(rho_within_mean, color="black", linestyle="--",
               label=f"mean = {rho_within_mean:.3f}")
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Within-tile Spearman ρ (uncertainty vs error)")
    ax.set_ylabel("Tile count")
    ax.set_title(f"Per-tile spatial calibration   N = {len(valid)}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Across-tile scatter
    ax = axes[1, 1]
    ax.scatter(mean_u_per_tile, mean_e_per_tile, s=15, alpha=0.6,
               edgecolors="black", linewidths=0.3, color="C3")
    ax.set_xlabel("Mean uncertainty per tile")
    ax.set_ylabel("Mean error per tile")
    ax.set_title(f"Across-tile   Pearson = {pearson_across:.3f}   Spearman = {spearman_across:.3f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Uncertainty calibration analysis: pairs ensemble uncertainty "
                    "with cycle-reconstruction error, reduces to scalar metrics, "
                    "and produces a 4-panel figure."
    )
    ap.add_argument("--uncertainty_dir", type=Path, required=True,
                    help="Directory of <stem>.npy uncertainty maps "
                         "(typically uncertainty_out/<arch>/raw_npy/).")
    ap.add_argument("--error_dirs", type=Path, nargs="+", required=True,
                    help="One or more directories of <stem>.npy error maps "
                         "(from evaluation.py --save_error_npy). Pass multiple to "
                         "average per-pixel across ensemble members.")
    ap.add_argument("--mask_dir", type=Path, default=None,
                    help="Directory of tissue masks (<stem>.tif). Required unless --no_mask.")
    ap.add_argument("--no_mask", action="store_true",
                    help="Skip tissue masking and use all pixels (will inflate calibration).")
    ap.add_argument("--tiles_metadata", type=Path, default=None,
                    help="Dataset root containing per-WSI tiles_metadata.csv files. "
                         "Enables per-WSI aggregation in per_wsi.csv.")
    ap.add_argument("--outdir", type=Path, default=Path("calibration_out"),
                    help="Output directory.")
    ap.add_argument("--n_bins", type=int, default=10,
                    help="Number of quantile bins for reliability diagram and ECE.")
    ap.add_argument("--ause_steps", type=int, default=100,
                    help="Number of fraction-removed steps for sparsification curve.")
    ap.add_argument("--min_tissue_pixels", type=int, default=256,
                    help="Skip tiles with fewer tissue pixels than this.")
    args = ap.parse_args()

    if not args.no_mask and args.mask_dir is None:
        ap.error("--mask_dir is required (or pass --no_mask to compute on all pixels).")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- discover paired stems ----
    u_stems = list_npy_stems(args.uncertainty_dir)
    e_stems_list = [list_npy_stems(d) for d in args.error_dirs]
    common = set.intersection(u_stems, *e_stems_list)
    if args.mask_dir is not None and not args.no_mask:
        common &= list_mask_stems(args.mask_dir)
    common = sorted(common)
    if not common:
        raise RuntimeError("No common <stem> across uncertainty / error / mask dirs.")
    print(f"Found {len(common)} paired tile(s) for calibration.")

    # ---- optional stem -> source_wsi map ----
    stem_to_wsi: dict[str, str] = {}
    if args.tiles_metadata is not None:
        stem_to_wsi = build_stem_to_wsi(args.tiles_metadata)
        n_mapped = sum(s in stem_to_wsi for s in common)
        print(f"Mapped {n_mapped}/{len(common)} stems to source WSI via tiles_metadata.")

    # ---- per-tile pass ----
    rows: list[dict] = []
    spar_per_tile_pred: list[np.ndarray] = []
    spar_per_tile_oracle: list[np.ndarray] = []
    fractions_grid: Optional[np.ndarray] = None

    for stem in tqdm(common, desc="Tiles"):
        u_full = np.load(args.uncertainty_dir / f"{stem}.npy").astype(np.float64)
        e_full = load_error_map(args.error_dirs, stem)
        if u_full.shape != e_full.shape:
            warnings.warn(f"Shape mismatch on {stem}: u={u_full.shape} e={e_full.shape}; skipping.")
            continue

        if args.no_mask:
            mask = np.ones(u_full.shape, dtype=bool)
        else:
            mpath = find_mask_path(args.mask_dir, stem)
            if mpath is None:
                warnings.warn(f"No mask for {stem}; skipping.")
                continue
            mask = load_tissue_mask(mpath, u_full.shape)

        n_tissue = int(mask.sum())
        if n_tissue < args.min_tissue_pixels:
            warnings.warn(f"{stem}: only {n_tissue} tissue pixels (< {args.min_tissue_pixels}); skipping.")
            continue

        u = u_full[mask].ravel()
        e = e_full[mask].ravel()

        # within-tile rank correlations
        if u.std() == 0 or e.std() == 0:
            sp_rho, sp_p = float("nan"), float("nan")
            pe_rho, pe_p = float("nan"), float("nan")
        else:
            sp = spearmanr(u, e)
            pe = pearsonr(u, e)
            sp_rho, sp_p = float(sp.statistic), float(sp.pvalue)
            pe_rho, pe_p = float(pe.statistic), float(pe.pvalue)

        # sparsification + AUSE
        fractions, s_pred, s_oracle = sparsification_curve(u, e, n_steps=args.ause_steps)
        a = ause(fractions, s_pred, s_oracle)
        spar_per_tile_pred.append(s_pred)
        spar_per_tile_oracle.append(s_oracle)
        if fractions_grid is None:
            fractions_grid = fractions

        rows.append({
            "tile_stem":     stem,
            "source_wsi":    stem_to_wsi.get(stem, ""),
            "n_tissue_pixels": n_tissue,
            "spearman_rho":  sp_rho,
            "spearman_pvalue": sp_p,
            "pearson_rho_within": pe_rho,
            "pearson_pvalue_within": pe_p,
            "mean_u":        float(u.mean()),
            "mean_e":        float(e.mean()),
            "ause":          a,
        })

    if not rows:
        raise RuntimeError("No tiles produced valid metrics — check inputs and --min_tissue_pixels.")

    # ---- per-tile CSV ----
    df = pd.DataFrame(rows)
    csv_path = args.outdir / "per_tile.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-tile CSV → {csv_path}")

    # ---- across-tile correlations ----
    finite = df[df["mean_u"].notna() & df["mean_e"].notna()]
    if len(finite) >= 3 and finite["mean_u"].std() > 0 and finite["mean_e"].std() > 0:
        across_pearson = float(pearsonr(finite["mean_u"].values, finite["mean_e"].values).statistic)
        across_spearman = float(spearmanr(finite["mean_u"].values, finite["mean_e"].values).statistic)
    else:
        across_pearson = float("nan")
        across_spearman = float("nan")

    # ---- tile-mean reliability + ECE ----
    # Use per-tile mean_u / mean_e (one point per tile) rather than subsampled
    # pixels. The within-tile Spearman rho already captures pixel-level spatial
    # calibration; the reliability diagram answers the tile-level question:
    # "do tiles with higher uncertainty have higher error?"
    u_tile = finite["mean_u"].values
    e_tile = finite["mean_e"].values
    u_lo, u_hi = float(np.percentile(u_tile, 1)), float(np.percentile(u_tile, 99))
    e_lo, e_hi = float(np.percentile(e_tile, 1)), float(np.percentile(e_tile, 99))
    bin_u, bin_e, bin_counts = reliability_bins(
        u_tile, e_tile, args.n_bins, u_lo, u_hi, e_lo, e_hi
    )
    ece = expected_calibration_error(bin_u, bin_e, bin_counts)

    # ---- average sparsification curve ----
    s_pred_avg = np.mean(np.stack(spar_per_tile_pred, axis=0), axis=0)
    s_oracle_avg = np.mean(np.stack(spar_per_tile_oracle, axis=0), axis=0)

    # ---- summary aggregates ----
    rho_w = df["spearman_rho"].values
    rho_w_finite = rho_w[np.isfinite(rho_w)]
    summary = {
        "n_tiles": int(len(df)),
        "n_tiles_finite_rho": int(rho_w_finite.size),
        "n_wsi": int(df["source_wsi"].replace("", np.nan).dropna().nunique()) if "source_wsi" in df.columns else 0,
        "within_tile": {
            "spearman_mean": float(rho_w_finite.mean()) if rho_w_finite.size else float("nan"),
            "spearman_std":  float(rho_w_finite.std(ddof=1)) if rho_w_finite.size > 1 else 0.0,
            "spearman_median": float(np.median(rho_w_finite)) if rho_w_finite.size else float("nan"),
            "ause_mean":     float(df["ause"].mean()),
            "ause_std":      float(df["ause"].std(ddof=1)) if len(df) > 1 else 0.0,
        },
        "across_tile": {
            "pearson":  across_pearson,
            "spearman": across_spearman,
        },
        "reliability": {
            "ece":         ece,
            "bin_mean_u_normalised": bin_u.tolist(),
            "bin_mean_e_normalised": bin_e.tolist(),
            "bin_counts":            bin_counts.tolist(),
            "u_tile_mean_p1_p99":    [u_lo, u_hi],
            "e_tile_mean_p1_p99":    [e_lo, e_hi],
        },
        "sparsification": {
            "fractions":      fractions_grid.tolist(),
            "s_pred_avg":     s_pred_avg.tolist(),
            "s_oracle_avg":   s_oracle_avg.tolist(),
        },
        "params": {
            "uncertainty_dir":   str(args.uncertainty_dir),
            "error_dirs":        [str(d) for d in args.error_dirs],
            "error_aggregation": "first" if len(args.error_dirs) == 1 else f"mean_of_{len(args.error_dirs)}",
            "mask_dir":          str(args.mask_dir) if args.mask_dir else None,
            "no_mask":           args.no_mask,
            "n_bins":            args.n_bins,
            "ause_steps":        args.ause_steps,
            "reliability_granularity": "tile_means",
            "min_tissue_pixels": args.min_tissue_pixels,
        },
    }

    json_path = args.outdir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {json_path}")

    # ---- per-WSI aggregation ----
    if stem_to_wsi:
        df_wsi = df[df["source_wsi"] != ""].copy()
        if not df_wsi.empty:
            agg = df_wsi.groupby("source_wsi").agg(
                n_tiles=("tile_stem", "count"),
                spearman_rho_mean=("spearman_rho", "mean"),
                spearman_rho_std=("spearman_rho", "std"),
                ause_mean=("ause", "mean"),
                mean_u=("mean_u", "mean"),
                mean_e=("mean_e", "mean"),
            ).reset_index()
            wsi_path = args.outdir / "per_wsi.csv"
            agg.to_csv(wsi_path, index=False)
            print(f"Saved per-WSI CSV → {wsi_path}")

    # ---- 2x2 figure ----
    make_plot(
        bin_mean_u=bin_u,
        bin_mean_e=bin_e,
        ece=ece,
        fractions=fractions_grid,
        s_pred_avg=s_pred_avg,
        s_oracle_avg=s_oracle_avg,
        ause_mean=summary["within_tile"]["ause_mean"],
        rho_within=rho_w,
        rho_within_mean=summary["within_tile"]["spearman_mean"],
        mean_u_per_tile=df["mean_u"].values,
        mean_e_per_tile=df["mean_e"].values,
        pearson_across=across_pearson,
        spearman_across=across_spearman,
        outpath=args.outdir / "calibration.png",
    )

    # ---- console summary ----
    print("\n=== Calibration summary ===")
    print(f"N tiles                  : {summary['n_tiles']}")
    print(f"Within-tile Spearman ρ   : {summary['within_tile']['spearman_mean']:.4f} "
          f"± {summary['within_tile']['spearman_std']:.4f}")
    print(f"Across-tile Pearson ρ    : {across_pearson:.4f}")
    print(f"Across-tile Spearman ρ   : {across_spearman:.4f}")
    print(f"AUSE (mean)              : {summary['within_tile']['ause_mean']:.4f}")
    print(f"ECE (10-bin, p1–p99 norm): {ece:.4f}")


if __name__ == "__main__":
    main()
