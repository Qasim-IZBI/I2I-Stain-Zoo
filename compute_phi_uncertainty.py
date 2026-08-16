#!/usr/bin/env python
"""Descriptor-space uncertainty decomposition over an ensemble of WSI masks.

Implements the φ_struct pipeline of `kidney_ood_data_plan.md` §5 and the
epistemic split of `uncertainty_strategy.md` E2. Averages in **descriptor
space**, never pixel space — this is the counterpart to `uncertainty.py`, not a
replacement for it.

Single ensemble (procedural only — one training set, N seeds):

    python compute_phi_uncertainty.py \\
        --ensemble  /path/ensemble/cyclegan/data_large/model_medium/wsi_masks_final \\
        --tiles_metadata /path/tiles/testA \\
        --he_dir    /path/reconstructed_he \\
        --outdir    ./phi_uncertainty/

Fold x seed grid (procedural AND data-exposure, one --fold per data block):

    python compute_phi_uncertainty.py \\
        --fold /path/ensemble_ugac/cyclegan/data_001_007/model_small/wsi_masks_final \\
        --fold /path/ensemble_ugac/cyclegan/data_008_014/model_small/wsi_masks_final \\
        --fold ... \\
        --tiles_metadata /path/tiles/testA --outdir ./phi_uncertainty/

Outputs
-------
per_region.csv   one row per region: mu per descriptor, Var, and the components
summary.json     dataset aggregates, floor provenance, full parameter record

Notes
-----
* Reconstructed WSIs sit at the SOURCE resolution (0.221 um/px), not the
  0.442 um/px the model consumed — `utils.reconstruct_wsi` upsamples tiles back
  to `tile_size`. `--mpp` defaults accordingly; override only if your
  reconstructions differ.
* Bias against a real target is deliberately NOT computed here. It needs the
  floor measured first (§7 go/no-go), and on kidney it additionally needs the
  liver-trained segmenter validated (§6.2). `floor.py` provides the estimators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from uncertainty_phi.decompose import decompose
from uncertainty_phi.descriptors import PHI_NAMES, PHI_REFERENCE, WHITE_THRESH
from uncertainty_phi.ensemble import mean_and_variance, phi_over_ensemble
from uncertainty_phi.regions import SOURCE_MPP, region_area_mm2


def _finite(v) -> Optional[float]:
    """None rather than NaN, so the CSV reads as missing instead of as a value."""
    v = float(v)
    return v if np.isfinite(v) else None


def _sd(per_dim, i: int, j: int) -> Optional[float]:
    """SD from a variance component, or None where it is not defined.

    Negative variance is a real outcome of the ANOVA estimator when the true
    component is near zero, and it is reported rather than clipped elsewhere —
    but there is no SD for it, so this returns None instead of a NaN that would
    read as a failed computation.
    """
    if per_dim is None:
        return None
    v = float(per_dim[i, j])
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else None


def _collect(roots: List[Path], args) -> tuple:
    """φ for each ensemble root, on a shared region grid."""
    blocks, regions_ref, members, tissue_ref = [], None, [], None
    for root in roots:
        phi, regions, member_dirs, tissue_frac = phi_over_ensemble(
            root,
            Path(args.tiles_metadata),
            he_dir=Path(args.he_dir) if args.he_dir else None,
            roi_dir=Path(args.roi_dir) if args.roi_dir else None,
            qc_dir=Path(args.qc_dir) if args.qc_dir else None,
            qc_max_px=args.qc_max_px,
            min_roi_fraction=args.min_roi_fraction,
            region_mm=args.region_mm,
            region_px=args.region_px,
            mpp=args.mpp,
            min_tissue_fraction=args.min_tissue_fraction,
            min_object_px=args.min_object_px,
            closing_px=args.closing_px,
            white_thresh=args.white_thresh,
        )
        if regions_ref is None:
            regions_ref, tissue_ref = regions, tissue_frac
        elif len(regions) != len(regions_ref):
            raise SystemExit(
                f"fold {root} produced {len(regions)} regions but the first fold "
                f"produced {len(regions_ref)} — every fold must be scored on the "
                f"same grid, so they must share --tiles_metadata and --region_mm"
            )
        blocks.append(phi)
        members.append([str(m) for m in member_dirs])
    return blocks, regions_ref, members, tissue_ref


def main() -> None:
    ap = argparse.ArgumentParser(
        "Descriptor-space (phi_struct) uncertainty decomposition"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ensemble", type=Path,
                     help="One directory of model_NN/ mask dirs. Procedural only.")
    src.add_argument("--fold", type=Path, action="append",
                     help="Repeat once per data block to enable the procedural "
                          "vs data-exposure split.")

    ap.add_argument("--tiles_metadata", type=Path, required=True,
                    help="Dataset root holding per-WSI tiles_metadata.csv "
                         "(same argument reconstruct.py takes).")
    ap.add_argument("--he_dir", type=Path, default=None,
                    help="Reconstructed H&E WSIs. Without it the two floor-free "
                         "geometric descriptors are NaN.")
    ap.add_argument("--roi_dir", type=Path, default=None,
                    help="Per-WSI binary masks (<stem>.tif) restricting the "
                         "analysis to an anatomical compartment — cortex on the "
                         "kidney arm. Resized nearest-neighbour if it does not "
                         "match the reconstruction. A WSI with no mask here is "
                         "EXCLUDED, not passed through whole.")
    ap.add_argument("--min_roi_fraction", type=float, default=0.5,
                    help="Region coverage by --roi_dir needed to keep it. "
                         "Coverage, not centre: a region half in medulla is not "
                         "a cortex measurement. [%(default)s]")
    ap.add_argument("--outdir", type=Path, default=Path("phi_uncertainty"))

    ap.add_argument("--region_mm", type=float, default=1.5,
                    help="Region side in mm (section 4.2 recommends 1-2). [%(default)s]")
    ap.add_argument("--region_px", type=int, default=None,
                    help="Region side in PIXELS, overriding --region_mm. Sizes "
                         "are in mm by default because reconstructions sit at "
                         "the source resolution and a pixel count means a "
                         "different physical scale on a different cohort. Use "
                         "this only where the pixel grid itself matters, e.g. "
                         "2048 for a seamless heatmap overlay.")
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP,
                    help="Microns per pixel OF THE RECONSTRUCTION. [%(default)s]")
    ap.add_argument("--min_tissue_fraction", type=float, default=0.25,
                    help="Drop regions below this tissue coverage. [%(default)s]")
    ap.add_argument("--min_object_px", type=int, default=16,
                    help="Speckle removal before the topological terms. [%(default)s]")
    ap.add_argument("--closing_px", type=int, default=0,
                    help="Morphological closing before the topological terms. [%(default)s]")
    ap.add_argument("--qc_dir", type=Path, default=None,
                    help="Write one region per WSI here as a TIF pair — the "
                         "label mask (0 outside, 1 tissue, 2 lumen) and the "
                         "matching H&E crop — for inspecting in Fiji. "
                         "lumen_fraction is thresholded and has no plateau on "
                         "some cohorts, so the number alone cannot tell you "
                         "whether it found lumens or pale tissue.")
    ap.add_argument("--qc_max_px", type=int, default=0,
                    help="Cap the long side of the QC crops, 0 = full "
                         "resolution. The mask compresses to almost nothing, "
                         "but a full-resolution H&E crop of a 1.5 mm region is "
                         "~100 MB before compression. [%(default)s]")
    ap.add_argument("--white_thresh", type=float, default=WHITE_THRESH,
                    help="Whitespace cut for lumen_fraction and tissue_fraction. "
                         "EVERY channel must clear it, so the number to compare "
                         "against is the per-pixel channel minimum, not the grey "
                         "level an 8-bit conversion shows. A lumen_fraction near "
                         "1e-5 means this sits above the lumens. [%(default)s]")
    args = ap.parse_args()

    roots = [args.ensemble] if args.ensemble else list(args.fold)
    for r in roots:
        if not Path(r).is_dir():
            raise SystemExit(f"not a directory: {r}")

    blocks, regions, members, tissue_frac = _collect(roots, args)
    comps = decompose(blocks)

    # mu / Var pooled across every member of every fold
    pooled = np.concatenate(blocks, axis=0)
    mu, var = mean_and_variance(pooled)

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, region in enumerate(regions):
        row = {
            "region_stem": region.stem,
            "wsi": region.wsi,
            "region_index": region.index,
            "y0": region.y0, "y1": region.y1, "x0": region.x0, "x1": region.x1,
            "area_mm2": region_area_mm2(region, args.mpp),
            "var_total_descriptor_space": float(var[i]),
            # The ANOVA total summary.json reports — procedural + data, not the
            # pooled plug-in variance above, which ignores the fold structure and
            # so does not equal their sum. Written per region so a run split over
            # WSIs can be pooled afterwards without re-deriving it.
            "var_total_anova": float(comps.total[i]),
            "procedural": float(comps.procedural[i]),
            "data_exposure": None if comps.data is None else float(comps.data[i]),
            # H&E footprint coverage: no variance and no error, so it is not a
            # phi component, but it is the QC number for a thin region
            "tissue_fraction": (None if tissue_frac is None
                                else float(tissue_frac[i])),
        }
        for j, name in enumerate(PHI_NAMES):
            row[f"mu_{name}"] = float(mu[i, j])
            # Per-descriptor SDs are what calibration pairs with |error|; the
            # summed scalars above cannot be matched to a single descriptor.
            row[f"sd_total_{name}"] = _sd(comps.total_per_dim, i, j)
            row[f"sd_procedural_{name}"] = _sd(comps.procedural_per_dim, i, j)
            row[f"sd_data_{name}"] = _sd(comps.data_per_dim, i, j)
        for f, block in enumerate(blocks):
            # per-fold mu and sd: the subset-level prediction, whose error pairs
            # with procedural spread alone
            with np.errstate(invalid="ignore"):
                fold_mu = np.nanmean(block[:, i, :], axis=0)
                fold_sd = np.nanstd(block[:, i, :], axis=0, ddof=1)
            for j, name in enumerate(PHI_NAMES):
                row[f"fold{f + 1}_mu_{name}"] = _finite(fold_mu[j])
                row[f"fold{f + 1}_sd_{name}"] = _finite(fold_sd[j])
        rows.append(row)

    per_region = args.outdir / "per_region.csv"
    pd.DataFrame(rows).to_csv(per_region, index=False)

    summary = {
        "n_folds": comps.n_folds,
        "n_seeds_per_fold": comps.n_seeds_per_fold,
        "n_regions": len(regions),
        "n_wsis": len({r.wsi for r in regions}),
        "descriptors": list(PHI_NAMES),
        "reference_class": dict(zip(PHI_NAMES, PHI_REFERENCE)),
        "variance": comps.summary(),
        "bias": {
            "computed": False,
            "reason": (
                "bias^2 = observed^2 - d requires a floor covariance from "
                "uncertainty_phi.floor (section 7 go/no-go) and, on kidney, a "
                "segmenter validated out of distribution (section 6.2)"
            ),
        },
        "params": {
            "roots": [str(r) for r in roots],
            "members_per_fold": members,
            "tiles_metadata": str(args.tiles_metadata),
            "he_dir": str(args.he_dir) if args.he_dir else None,
            "roi_dir": str(args.roi_dir) if args.roi_dir else None,
            "qc_dir": str(args.qc_dir) if args.qc_dir else None,
            "qc_max_px": args.qc_max_px if args.qc_dir else None,
            "min_roi_fraction": args.min_roi_fraction if args.roi_dir else None,
            "region_mm": args.region_mm,
            "region_px": args.region_px,
            "mpp": args.mpp,
            "min_tissue_fraction": args.min_tissue_fraction,
            "min_object_px": args.min_object_px,
            "closing_px": args.closing_px,
            "white_thresh": args.white_thresh,
        },
    }
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== phi_struct uncertainty ===")
    print(f"folds     : {comps.n_folds}  (seeds per fold: {comps.n_seeds_per_fold})")
    print(f"regions   : {len(regions)} over {summary['n_wsis']} WSI")
    print(f"procedural: {comps.summary()['procedural_mean']}")
    dm = comps.summary()["data_mean"]
    print(f"data      : {dm if dm is not None else 'undefined (single fold)'}")
    print(f"\nwrote {per_region}")
    print(f"wrote {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
