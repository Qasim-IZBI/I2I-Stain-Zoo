"""
Cross-stain consistency evaluation for virtual SR staining.

For each matched H&E WSI / generated SR mask pair:
  1. Extract acellular eosinophilic regions from the H&E (collagen proxy) via
     colour deconvolution + nuclear exclusion.
  2. Extract PSR-positive regions from the nnUNet segmentation mask.
  3. Compute Dice coefficient and IoU between the two binary masks.

H&E and generated SR are pixel-aligned by construction (same tile positions),
so spatial overlap is meaningful without any registration step.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.color import separate_stains, hed_from_rgb
from skimage.morphology import binary_dilation, disk, remove_small_objects


# ---------------------------------------------------------------------------
# Collagen proxy extraction
# ---------------------------------------------------------------------------

def extract_collagen_mask(he_rgb: np.ndarray,
                           eosin_thresh: float,
                           haem_thresh: float,
                           nuclear_dilation: int,
                           min_area: int) -> np.ndarray:
    """
    Binary mask of acellular eosinophilic regions (collagen proxy) from H&E.

    Steps:
      1. Macenko-style colour deconvolution → haematoxylin (H) and eosin (E) channels
      2. Threshold E channel → eosinophilic mask (collagen + cytoplasm)
      3. Threshold H channel, dilate → nuclear+cytoplasm exclusion mask
      4. Subtract nuclear mask → acellular ECM regions
      5. Remove small debris below min_area

    Returns uint8 binary mask (1 = collagen proxy, 0 = elsewhere).
    """
    img = he_rgb.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0)

    # separate_stains output channels: 0=haematoxylin, 1=eosin, 2=DAB (unused)
    stains = separate_stains(img, hed_from_rgb)
    haem  = stains[..., 0]
    eosin = stains[..., 1]

    eosinophilic = eosin > eosin_thresh
    nuclear_mask = binary_dilation(haem > haem_thresh, disk(nuclear_dilation))

    collagen = eosinophilic & ~nuclear_mask
    if min_area > 0:
        collagen = remove_small_objects(collagen, min_size=min_area)
    return collagen.astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    """
    Returns (dice, iou). Returns (None, None) when both masks are empty
    (agreement that nothing is present, but the metric is undefined).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = int((a & b).sum())
    union        = int((a | b).sum())
    sum_ab       = int(a.sum() + b.sum())
    if sum_ab == 0:
        return None, None
    dice = 2 * intersection / sum_ab
    iou  = intersection / union if union > 0 else 0.0
    return float(dice), float(iou)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_he(path: Path) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def load_psr_binary(path: Path, label_psr: int) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim > 2:
        mask = mask[..., 0]
    return (mask == label_psr).astype(np.uint8)


def find_tifs(directory: Path) -> dict:
    return {p.stem: p for p in sorted(
        list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    )}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(records: list, outdir: Path) -> None:
    df = pd.DataFrame(records).dropna(subset=["dice"])
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Dice per WSI ---
    ax = axes[0]
    ax.scatter(range(len(df)), df["dice"], s=70, zorder=5,
               edgecolors="black", linewidths=0.7, color="#4c72b0")
    ax.axhline(df["dice"].mean(), color="gray", linestyle="--", linewidth=0.9,
               label=f"mean = {df['dice'].mean():.3f}")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["wsi"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice  (H&E collagen proxy ∩ PSR+)")
    ax.set_title("Cross-stain consistency — Dice per WSI")
    ax.legend(fontsize=8)

    # --- Scatter: H&E collagen fraction vs generated PSR fraction ---
    ax = axes[1]
    ax.scatter(df["collagen_fraction"], df["psr_fraction"], s=70, zorder=5,
               edgecolors="black", linewidths=0.7, color="#dd8452")
    for _, row in df.iterrows():
        ax.annotate(row["wsi"], (row["collagen_fraction"], row["psr_fraction"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    lims = [0, max(df["collagen_fraction"].max(), df["psr_fraction"].max()) * 1.1]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="y = x")
    ax.set_xlim(0, lims[1])
    ax.set_ylim(0, lims[1])
    ax.set_xlabel("H&E collagen proxy fraction")
    ax.set_ylabel("Generated SR PSR+ fraction")
    ax.set_title("H&E collagen vs generated PSR+")
    ax.legend(fontsize=8)

    fig.tight_layout()
    outpath = outdir / "consistency.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-stain consistency: Dice/IoU between H&E collagen proxy "
                    "(colour deconvolution) and generated SR PSR+ mask (nnUNet). "
                    "H&E WSIs and PSR masks are matched by filename stem."
    )
    parser.add_argument("--he_wsis", type=Path, required=True,
                        help="Directory of reconstructed H&E WSI TIFs (testA tiles → reconstruct.py)")
    parser.add_argument("--psr_masks", type=Path, required=True,
                        help="Directory of PSR segmentation mask TIFs (segment_psr.py output)")
    parser.add_argument("--outdir", type=Path, default=Path("cross_stain_out"),
                        help="Output directory [%(default)s]")

    parser.add_argument("--label_psr", type=int, default=2,
                        help="nnUNet label index for PSR-positive class [%(default)s]")

    # Colour deconvolution thresholds
    parser.add_argument("--eosin_thresh", type=float, default=0.05,
                        help="Eosin channel threshold for eosinophilic mask. "
                             "Increase to be more conservative (less collagen detected). [%(default)s]")
    parser.add_argument("--haem_thresh", type=float, default=0.05,
                        help="Haematoxylin threshold for nuclear mask. "
                             "Increase to dilate less aggressively around nuclei. [%(default)s]")
    parser.add_argument("--nuclear_dilation", type=int, default=8,
                        help="Dilation radius (px) applied to nuclear mask to cover cytoplasm. [%(default)s]")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Minimum connected component area (px) retained in collagen mask. [%(default)s]")
    parser.add_argument("--save_collagen_masks", action="store_true",
                        help="Save H&E collagen proxy masks to {outdir}/collagen_masks/ "
                             "(useful for tuning thresholds)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.save_collagen_masks:
        (args.outdir / "collagen_masks").mkdir(exist_ok=True)

    # Match files by stem
    he_files  = find_tifs(args.he_wsis)
    psr_files = find_tifs(args.psr_masks)
    common    = sorted(set(he_files) & set(psr_files))

    unmatched_he  = set(he_files)  - set(psr_files)
    unmatched_psr = set(psr_files) - set(he_files)
    if unmatched_he:
        print(f"[WARN] H&E WSIs without matching PSR mask (skipped): {sorted(unmatched_he)}")
    if unmatched_psr:
        print(f"[WARN] PSR masks without matching H&E WSI (skipped): {sorted(unmatched_psr)}")
    if not common:
        raise RuntimeError(
            "No matching filename stems between --he_wsis and --psr_masks.\n"
            f"  H&E stems:  {sorted(he_files)}\n"
            f"  PSR stems:  {sorted(psr_files)}"
        )

    print(f"Found {len(common)} matched WSI pair(s).")

    records = []
    for stem in common:
        print(f"  {stem} ...", end=" ", flush=True)

        he_rgb  = load_he(he_files[stem])
        psr_bin = load_psr_binary(psr_files[stem], args.label_psr)

        collagen = extract_collagen_mask(
            he_rgb,
            eosin_thresh=args.eosin_thresh,
            haem_thresh=args.haem_thresh,
            nuclear_dilation=args.nuclear_dilation,
            min_area=args.min_area,
        )

        if args.save_collagen_masks:
            tifffile.imwrite(
                args.outdir / "collagen_masks" / f"{stem}.tif",
                (collagen * 255).astype(np.uint8),
            )

        dice, iou = dice_iou(collagen, psr_bin)

        total_px      = float(collagen.size)
        collagen_frac = float(collagen.sum()) / total_px
        psr_frac      = float(psr_bin.sum()) / total_px

        if dice is None:
            print("both masks empty — skipped")
        else:
            print(f"Dice={dice:.3f}  IoU={iou:.3f}  "
                  f"collagen={collagen_frac:.3f}  PSR+={psr_frac:.3f}")

        records.append({
            "wsi":               stem,
            "dice":              dice,
            "iou":               iou,
            "collagen_fraction": collagen_frac,
            "psr_fraction":      psr_frac,
        })

    # CSV
    df = pd.DataFrame(records)
    csv_path = args.outdir / "per_wsi.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    # Summary JSON
    valid = df.dropna(subset=["dice"])
    summary = {
        "n_wsi":                  len(valid),
        "dice_mean":              float(valid["dice"].mean())  if not valid.empty else None,
        "dice_std":               float(valid["dice"].std(ddof=1)) if len(valid) > 1 else 0.0,
        "iou_mean":               float(valid["iou"].mean())   if not valid.empty else None,
        "iou_std":                float(valid["iou"].std(ddof=1))  if len(valid) > 1 else 0.0,
        "collagen_fraction_mean": float(valid["collagen_fraction"].mean()) if not valid.empty else None,
        "psr_fraction_mean":      float(valid["psr_fraction"].mean())      if not valid.empty else None,
        "params": {
            "eosin_thresh":     args.eosin_thresh,
            "haem_thresh":      args.haem_thresh,
            "nuclear_dilation": args.nuclear_dilation,
            "min_area":         args.min_area,
            "label_psr":        args.label_psr,
        },
    }
    json_path = args.outdir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {json_path}")

    if not valid.empty:
        plot_results(records, args.outdir)


if __name__ == "__main__":
    main()
