import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import ks_2samp, wasserstein_distance


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


def load_condition(mask_dir: Path, label_tissue: int, label_psr: int) -> list:
    records = []
    for p in sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff"))):
        frac = compute_psr_fraction(p, label_tissue, label_psr)
        if frac is None:
            print(f"[WARN] No tissue pixels in {p.name} — skipping.")
            continue
        records.append((p.stem, frac))
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
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.masks_generated):
        parser.error("--labels must have the same number of entries as --masks_generated")

    labels = args.labels or [p.name for p in args.masks_generated]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- load real SR ----
    real_records = load_condition(args.masks_real, args.label_tissue, args.label_psr)
    if not real_records:
        raise RuntimeError(f"No valid mask TIFs found in {args.masks_real}")
    real_fracs = np.array([r[1] for r in real_records])
    print(f"Real SR : {len(real_fracs)} WSI(s)  mean={real_fracs.mean():.4f}  "
          f"std={real_fracs.std(ddof=1) if len(real_fracs) > 1 else 0:.4f}")

    # ---- load generated conditions ----
    conditions     = {"real": real_fracs}
    gen_records    = {}
    for label, mask_dir in zip(labels, args.masks_generated):
        records = load_condition(mask_dir, args.label_tissue, args.label_psr)
        if not records:
            print(f"[WARN] No valid masks in {mask_dir} — skipping '{label}'")
            continue
        fracs = np.array([r[1] for r in records])
        conditions[label] = fracs
        gen_records[label] = records
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
        w1       = float(wasserstein_distance(real_fracs, fracs))
        ci       = bootstrap_wasserstein(real_fracs, fracs, n=args.n_bootstrap)
        ks_stat, ks_p = ks_2samp(real_fracs, fracs)
        mean_diff    = float(fracs.mean() - real_fracs.mean())
        real_std = real_fracs.std(ddof=1) if len(real_fracs) > 1 else 0
        std_ratio = float(fracs.std(ddof=1) / real_std) if real_std > 0 and len(fracs) > 1 else None
        pairwise[label] = {
            "wasserstein_1":                      w1,
            "wasserstein_bootstrap_95ci":         {"low": float(ci[0]),
                                                   "median": float(ci[1]),
                                                   "high": float(ci[2])},
            "ks_statistic":                       float(ks_stat),
            "ks_pvalue":                          float(ks_p),
            "mean_diff_generated_minus_real":     mean_diff,
            "std_ratio_generated_over_real":      std_ratio,
        }

    json_path = args.outdir / "summary.json"
    with open(json_path, "w") as f:
        json.dump({"per_condition": per_condition, "pairwise_vs_real": pairwise},
                  f, indent=2)
    print(f"Saved summary → {json_path}")

    # ---- plot ----
    plot_comparison(conditions, real_label="real", outpath=args.outdir / "comparison.png")


if __name__ == "__main__":
    main()
