#!/usr/bin/env python
"""Per-descriptor biological floor — the §7 go/no-go pilot.

Estimate the floor **before** building anything on top of it. If the observed
virtual-vs-real discrepancy lands near the floor there is no headroom and
`bias² = observed² − d` comes out at or below zero. Cheap to check on a handful
of slides, and a genuine stop condition.

The readout is deliberately **per descriptor**, not pooled. A single number hides
the question that decides the design: is this particular component stable enough
between levels to carry a bias signal? CPA averages over millions of pixels and
concentrates fast; β₀/β₁ count discrete events and behave Poisson-like, so a
region holding ~50 loops has a relative SD near 1/√50 ≈ 14% between levels before
threshold sensitivity is added. β may be a valid direction but a noisy one — an
empirical question, not something to assume.

Three estimators, in decreasing order of authority:

  --psr_level_b   direct cross-level measurement. Needs a second real PSR level
                  per case. Supersedes the bracket entirely (§9 open question 5).
  --real_he       cross-stain UPPER bound, stain-invariant terms only.
  (always)        split-half LOWER bound, from disjoint region sets of one slide.

Usage
-----
    python estimate_floor.py \\
        --real_psr /path/psr_masks/real/psr_masks_wsi_final \\
        --tiles_metadata /path/tiles/testB \\
        --real_he /path/reconstructed_he \\
        --outdir ./floor_pilot/

Outputs
-------
floor_per_descriptor.csv   one row per descriptor: floor SD, between-region SD,
                           floor/signal ratio, verdict
floor.json                 the covariances, provenance and parameter record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from uncertainty_phi.descriptors import phi_struct
from uncertainty_phi.ensemble import _stem_index, load_label_mask, load_rgb
from uncertainty_phi.floor import (
    cross_level_floor,
    variogram_floor,
    cross_stain_floor,
    per_descriptor_report,
    split_half_floor,
    split_regions,
)
from uncertainty_phi.regions import (
    SOURCE_MPP,
    region_centres_mm,
    filter_by_tissue,
    iter_metadata_csvs,
    region_grid,
    wsi_extent,
)
from uncertainty_phi.descriptors import he_tissue_footprint


def _phi_for_dir(mask_dir: Path, args, *, he_dir: Optional[Path] = None,
                 with_geometry: bool = False):
    """φ over every region of every WSI in a directory of real masks.

    With `with_geometry` also returns region centres (mm) and slide labels, which
    the variogram needs to bin pairs by separation and to keep pairs within a
    slide.
    """
    index = _stem_index(Path(mask_dir))
    he_index = _stem_index(Path(he_dir)) if he_dir else {}
    out: List[np.ndarray] = []
    coords: List[np.ndarray] = []
    labels_out: List[str] = []

    for csv_path in iter_metadata_csvs(Path(args.tiles_metadata)):
        source, _, _ = wsi_extent(csv_path)
        stem = Path(source).stem
        mpath = index.get(stem)
        if mpath is None:
            print(f"  [skip] no mask for {stem} in {mask_dir}")
            continue

        labels = load_label_mask(mpath)
        grid = filter_by_tissue(
            region_grid(csv_path, region_mm=args.region_mm, mpp=args.mpp),
            labels, min_tissue_fraction=args.min_tissue_fraction,
        )
        if not grid:
            continue

        he = load_rgb(he_index[stem]) if stem in he_index else None
        footprint = he_tissue_footprint(he) if he is not None else None

        for r in grid:
            out.append(phi_struct(
                r.crop(labels),
                r.crop(he) if he is not None else None,
                mpp=args.mpp,
                tissue_mask=r.crop(footprint) if footprint is not None else None,
                min_object_px=args.min_object_px,
                closing_px=args.closing_px,
            ))
        if with_geometry:
            coords.append(region_centres_mm(grid, args.mpp))
            labels_out.extend([stem] * len(grid))

    if not out:
        raise SystemExit(f"no regions produced from {mask_dir}")
    phi = np.vstack(out)
    if with_geometry:
        return phi, np.vstack(coords), labels_out
    return phi


def main() -> None:
    ap = argparse.ArgumentParser("Per-descriptor biological floor (go/no-go pilot)")
    ap.add_argument("--real_psr", type=Path, required=True,
                    help="Directory of REAL PSR WSI masks (level A).")
    ap.add_argument("--tiles_metadata", type=Path, required=True,
                    help="Dataset root with per-WSI tiles_metadata.csv.")
    ap.add_argument("--real_he", type=Path, default=None,
                    help="Reconstructed real H&E WSIs. Enables the cross-stain "
                         "UPPER bound on the two stain-invariant descriptors.")
    ap.add_argument("--psr_level_b", type=Path, default=None,
                    help="A SECOND real PSR level for the same cases. Gives the "
                         "direct cross-level floor and supersedes the bracket.")
    ap.add_argument("--outdir", type=Path, default=Path("floor_pilot"))

    ap.add_argument("--region_mm", type=float, default=1.5)
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP,
                    help="Microns per pixel OF THE RECONSTRUCTION. [%(default)s]")
    ap.add_argument("--min_tissue_fraction", type=float, default=0.25)
    ap.add_argument("--min_object_px", type=int, default=16)
    ap.add_argument("--closing_px", type=int, default=0)
    ap.add_argument("--no_variogram", action="store_true",
                    help="Skip the variogram sill. Leaves the collagen terms on "
                         "the split-half LOWER bound, which is anti-conservative "
                         "- bias will read too high.")
    ap.add_argument("--variogram_bins", type=int, default=12)
    ap.add_argument("--sill_quantile", type=float, default=0.5,
                    help="Lags above this quantile count as the sill. [%(default)s]")
    ap.add_argument("--max_lag_fraction", type=float, default=0.5,
                    help="Discard lags beyond this fraction of the largest "
                         "separation; edge effects dominate there. [%(default)s]")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the split-half partition. [%(default)s]")
    args = ap.parse_args()

    print(f"[1/4] phi over real PSR: {args.real_psr}")
    phi_real, coords, slide_ids = _phi_for_dir(
        args.real_psr, args, he_dir=args.real_he, with_geometry=True)
    print(f"      {phi_real.shape[0]} regions over {len(set(slide_ids))} WSI")

    # --- lower bound: split-half within the real slides ---
    ia, ib = split_regions(phi_real.shape[0], seed=args.seed)
    lower = split_half_floor(phi_real[ia], phi_real[ib]) if len(ia) >= 2 else None
    if lower is None:
        print("      [warn] too few regions for a split-half lower bound")

    # --- upper bound: stain-invariant terms, real H&E vs real PSR ---
    upper = None
    if args.real_he:
        print(f"[2/4] cross-stain upper bound from {args.real_he}")
        phi_he = _phi_for_dir(args.real_psr, args, he_dir=args.real_he)
        n = min(len(phi_he), len(phi_real))
        upper = cross_stain_floor(phi_he[:n], phi_real[:n])
    else:
        print("[2/4] no --real_he: collagen terms will have no upper bound")

    # --- direct: two real levels ---
    direct = None
    if args.psr_level_b:
        print(f"[3/4] direct cross-level floor from {args.psr_level_b}")
        phi_b = _phi_for_dir(args.psr_level_b, args)
        n = min(len(phi_b), len(phi_real))
        direct = cross_level_floor(phi_real[:n], phi_b[:n])
    else:
        print("[3/4] no --psr_level_b: floor is bracketed, not measured")

    # --- variogram sill: the only upper bound that reaches the collagen terms ---
    vg = None
    if not args.no_variogram:
        try:
            vg, curve = variogram_floor(
                phi_real, coords, slide_ids,
                n_bins=args.variogram_bins,
                sill_quantile=args.sill_quantile,
                max_lag_fraction=args.max_lag_fraction,
            )
            not_flat = [n for n, ok in curve["sill_reached"].items() if not ok]
            print(f"[4/4] variogram sill over {vg.n_samples} within-slide pairs")
            if not_flat:
                print(f"      [warn] sill not reached for: {', '.join(not_flat)}")
                print("             their bound is an UNDER-estimate; the slide is")
                print("             small relative to that descriptor's correlation length")
        except ValueError as e:
            print(f"[4/4] variogram unavailable: {e}")
    else:
        print("[4/4] --no_variogram: collagen terms fall back to the lower bound")

    rows = per_descriptor_report(phi_real, lower=lower, upper=upper,
                                 direct=direct, variogram=vg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir / "floor_per_descriptor.csv", index=False)

    payload = {
        "n_regions": int(phi_real.shape[0]),
        "per_descriptor": rows,
        "estimates": {
            k: (v.summary() if v is not None else None)
            for k, v in (("split_half_lower", lower),
                         ("cross_stain_upper", upper),
                         ("cross_level_direct", direct))
        },
        "verdict_thresholds": {"usable": "<0.5", "marginal": "0.5-0.9",
                               "floor_limited": ">=0.9"},
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
    }
    with open(args.outdir / "floor.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== per-descriptor floor ===")
    hdr = (f"{'descriptor':22s} {'floor_sd':>10s} {'region_sd':>10s} {'ratio':>7s} "
           f"{'source':>12s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = r["floor_sd_used"]
        s = r["between_region_sd"]
        q = r["floor_to_signal"]
        print(f"{r['descriptor']:22s} "
              f"{'    n/a' if f is None else f'{f:10.4f}'} "
              f"{'    n/a' if s is None else f'{s:10.4f}'} "
              f"{'    n/a' if q is None else f'{q:7.2f}'} "
              f"{str(r['floor_source']):>12s}  {r['verdict']}")

    limited = [r["descriptor"] for r in rows if r["verdict"] == "floor-limited"]
    if limited:
        print(f"\n[!] floor-limited: {', '.join(limited)}")
        print("    These carry no usable bias signal at this region size. If the")
        print("    topological terms are among them, CPA stands alone and the")
        print("    section 5.3 lumen-filler blind spot reopens.")
    print(f"\nwrote {args.outdir / 'floor_per_descriptor.csv'}")
    print(f"wrote {args.outdir / 'floor.json'}")


if __name__ == "__main__":
    main()
