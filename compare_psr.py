import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr, spearmanr


def compute_psr_fraction(mask_path: Path, label_tissue: int, label_psr: int):
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[..., 0]
    tissue = int(np.sum(mask == label_tissue))
    psr    = int(np.sum(mask == label_psr))
    denom  = tissue + psr
    if denom == 0:
        return None
    return psr / denom


def normalize_stem(stem: str, strip_prefix: bool) -> str:
    if not strip_prefix:
        return stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def load_condition(mask_dir: Path, label_tissue: int, label_psr: int,
                   strip_prefix: bool = False) -> list:
    records = []
    for p in sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff"))):
        frac = compute_psr_fraction(p, label_tissue, label_psr)
        if frac is None:
            print(f"[WARN] No tissue pixels in {p.name} — skipping.")
            continue
        records.append((normalize_stem(p.stem, strip_prefix), frac))
    return records


def bootstrap_wasserstein(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0):
    """Bootstrap CI (2.5th, 50th, 97.5th percentile) for Wasserstein-1 distance.
    Resampling is done at the WSI level to respect the correlation structure."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        stats.append(wasserstein_distance(sa, sb))
    return np.percentile(stats, [2.5, 50, 97.5])


def plot_paired_scatter(real_dict: dict, gen_dicts: dict, outpath: Path) -> None:
    labels = list(gen_dicts.keys())
    ncols  = max(1, len(labels))
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)

    for ax, label in zip(axes[0], labels):
        gen_dict = gen_dicts[label]
        common   = sorted(set(real_dict) & set(gen_dict))
        if len(common) < 2:
            ax.set_title(f"{label}\n(too few matched WSIs)")
            continue
        rx = np.array([real_dict[s] for s in common])
        gy = np.array([gen_dict[s]  for s in common])

        r,   _   = pearsonr(rx, gy)
        rho, _   = spearmanr(rx, gy)

        ax.scatter(rx, gy, s=60, edgecolors="black", linewidths=0.7,
                   color=plt.cm.tab10.colors[list(gen_dicts).index(label)])
        lim = (0.0, max(rx.max(), gy.max()) * 1.05)
        ax.plot(lim, lim, "--", color="gray", linewidth=0.9, label="y = x")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("Real PSR fraction")
        ax.set_ylabel("Generated PSR fraction")
        ax.set_title(f"{label}\nr={r:.2f}  ρ={rho:.2f}  n={len(common)}")
        ax.legend(fontsize=7)

    fig.suptitle("Paired PSR fractions: real vs. generated (matched by WSI stem)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved paired scatter → {outpath}")


def plot_comparison(conditions: dict, real_label: str, outpath: Path) -> None:
    labels = [real_label] + [k for k in conditions if k != real_label]
    fracs  = [conditions[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5))

    bp = ax.boxplot(fracs, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    widths=0.5)

    colors = ["#bbbbbb"] + list(plt.cm.tab10.colors[:len(labels) - 1])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    for i, (frac_arr, color) in enumerate(zip(fracs, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(frac_arr))
        ax.scatter(i + jitter, frac_arr, color=color, zorder=5,
                   s=50, edgecolors="black", linewidths=0.7)

    real_mean = conditions[real_label].mean()
    ax.axhline(real_mean, color="gray", linestyle="--", linewidth=0.9,
               label=f"Real mean ({real_mean:.3f})")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 0.4)
    ax.set_ylabel("PSR-positive area fraction\n(PSR pixels / (Tissue + PSR pixels))")
    ax.set_title("Task-based evaluation: PSR segmentation — real vs. generated SR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PSR-positive area fraction distributions between real and "
                    "generated SR segmentation masks (output of segment_psr.py). "
                    "Computes Wasserstein-1, KS test, and mean difference vs. real SR, "
                    "with bootstrap confidence intervals on Wasserstein distance."
    )
    parser.add_argument("--masks_real", type=Path, required=True,
                        help="Directory of WSI mask TIFs for real SR (from segment_psr.py)")
    parser.add_argument("--masks_generated", type=Path, nargs="+", required=True,
                        help="One or more directories of WSI mask TIFs for generated SR. "
                             "Pass multiple directories to compare several model configurations at once.")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Label for each --masks_generated directory "
                             "(default: directory name). Must match count of --masks_generated.")
    parser.add_argument("--outdir", type=Path, default=Path("psr_comparison"),
                        help="Output directory for CSV, JSON, and plot [%(default)s]")
    parser.add_argument("--label_tissue", type=int, default=1,
                        help="nnUNet label index for tissue class [%(default)s]")
    parser.add_argument("--label_psr", type=int, default=2,
                        help="nnUNet label index for PSR-positive class [%(default)s]")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for Wasserstein-1 CI [%(default)s]")
    parser.add_argument("--strip_prefix", action="store_true",
                        help="Strip the first '_'-delimited token from filenames before "
                             "matching (e.g. SR_slide.tif and HE_slide.tif both become "
                             "'slide' for pairing). Use when real and generated masks have "
                             "different prefixes.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.masks_generated):
        parser.error("--labels must have the same number of entries as --masks_generated")

    labels = args.labels or [p.name for p in args.masks_generated]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- load real SR ----
    real_records = load_condition(args.masks_real, args.label_tissue, args.label_psr,
                                  args.strip_prefix)
    if not real_records:
        raise RuntimeError(f"No valid mask TIFs found in {args.masks_real}")
    real_fracs = np.array([r[1] for r in real_records])
    real_dict  = {s: f for s, f in real_records}
    print(f"Real SR : {len(real_fracs)} WSI(s)  mean={real_fracs.mean():.4f}  "
          f"std={real_fracs.std(ddof=1) if len(real_fracs) > 1 else 0:.4f}")

    # ---- load generated conditions ----
    conditions     = {"real": real_fracs}
    gen_records    = {}
    gen_dicts      = {}
    for label, mask_dir in zip(labels, args.masks_generated):
        records = load_condition(mask_dir, args.label_tissue, args.label_psr,
                                  args.strip_prefix)
        if not records:
            print(f"[WARN] No valid masks in {mask_dir} — skipping '{label}'")
            continue
        fracs = np.array([r[1] for r in records])
        conditions[label] = fracs
        gen_records[label] = records
        gen_dicts[label]   = {s: f for s, f in records}
        print(f"{label:15s}: {len(fracs)} WSI(s)  mean={fracs.mean():.4f}  "
              f"std={fracs.std(ddof=1) if len(fracs) > 1 else 0:.4f}")

    # ---- per-WSI CSV ----
    rows = [{"wsi": s, "condition": "real", "psr_fraction": f} for s, f in real_records]
    for label, records in gen_records.items():
        rows += [{"wsi": s, "condition": label, "psr_fraction": f} for s, f in records]
    csv_path = args.outdir / "per_wsi.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved per-WSI CSV → {csv_path}")

    # ---- summary stats + pairwise metrics ----
    per_condition = {}
    for label, fracs in conditions.items():
        per_condition[label] = {
            "n_wsi":  int(len(fracs)),
            "mean":   float(fracs.mean()),
            "std":    float(fracs.std(ddof=1)) if len(fracs) > 1 else 0.0,
            "median": float(np.median(fracs)),
            "min":    float(fracs.min()),
            "max":    float(fracs.max()),
        }

    pairwise = {}
    for label, fracs in conditions.items():
        if label == "real":
            continue
        # distribution-level metrics (unpaired)
        w1            = float(wasserstein_distance(real_fracs, fracs))
        ci            = bootstrap_wasserstein(real_fracs, fracs, n=args.n_bootstrap)
        ks_stat, ks_p = ks_2samp(real_fracs, fracs)
        mean_diff     = float(fracs.mean() - real_fracs.mean())
        real_std      = real_fracs.std(ddof=1) if len(real_fracs) > 1 else 0
        std_ratio     = float(fracs.std(ddof=1) / real_std) if real_std > 0 and len(fracs) > 1 else None

        # paired metrics (matched by WSI stem)
        gen_dict     = gen_dicts[label]
        common_stems = sorted(set(real_dict) & set(gen_dict))
        n_matched    = len(common_stems)
        if n_matched < len(real_dict) or n_matched < len(fracs):
            print(f"[INFO] {label}: {n_matched}/{len(real_dict)} real / "
                  f"{len(fracs)} generated WSIs matched by stem.")
        if n_matched >= 3:
            real_m = np.array([real_dict[s] for s in common_stems])
            gen_m  = np.array([gen_dict[s]  for s in common_stems])
            r,   r_p   = pearsonr(real_m, gen_m)
            rho, rho_p = spearmanr(real_m, gen_m)
            mae_paired       = float(np.mean(np.abs(gen_m - real_m)))
            mean_paired_diff = float(np.mean(gen_m - real_m))
        else:
            r = r_p = rho = rho_p = mae_paired = mean_paired_diff = None

        pairwise[label] = {
            "wasserstein_1":                          w1,
            "wasserstein_bootstrap_95ci":             {"low": float(ci[0]),
                                                       "median": float(ci[1]),
                                                       "high": float(ci[2])},
            "ks_statistic":                           float(ks_stat),
            "ks_pvalue":                              float(ks_p),
            "mean_diff_generated_minus_real":         mean_diff,
            "std_ratio_generated_over_real":          std_ratio,
            "n_matched":                              n_matched,
            "pearson_r":                              float(r)   if r   is not None else None,
            "pearson_pvalue":                         float(r_p) if r_p is not None else None,
            "spearman_rho":                           float(rho)   if rho   is not None else None,
            "spearman_pvalue":                        float(rho_p) if rho_p is not None else None,
            "mae_paired":                             mae_paired,
            "mean_paired_diff_generated_minus_real":  mean_paired_diff,
        }

    json_path = args.outdir / "summary.json"
    with open(json_path, "w") as f:
        json.dump({"per_condition": per_condition, "pairwise_vs_real": pairwise},
                  f, indent=2)
    print(f"Saved summary → {json_path}")

    # ---- plot ----
    plot_comparison(conditions, real_label="real", outpath=args.outdir / "comparison.png")
    if gen_dicts:
        plot_paired_scatter(real_dict, gen_dicts, outpath=args.outdir / "paired_scatter.png")


if __name__ == "__main__":
    main()
