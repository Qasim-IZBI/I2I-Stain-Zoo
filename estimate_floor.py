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
        --real_he /path/real_he_wsis \\
        --real_psr_rgb /path/real_sr_wsis \\
        --outdir ./floor_pilot/

`--tiles_metadata` is optional. Without it the region grid is sized from each
mask directly — the real SR is evaluated whole-slide and has no tiling, and
producing one purely for coordinates would cost hundreds of GB for numbers the
image already carries.

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

from uncertainty_phi.descriptors import PHI_NAMES, phi_struct
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
    region_grid_from_extent,
    region_centres_mm,
    filter_by_tissue,
    iter_metadata_csvs,
    region_grid,
    wsi_extent,
)
from uncertainty_phi.descriptors import WHITE_THRESH, he_tissue_footprint


def _phi_for_dir(mask_dir: Path, args, *, he_dir: Optional[Path] = None,
                 with_geometry: bool = False,
                 white_thresh: Optional[float] = None):
    """φ over every region of every WSI in a directory of real masks.

    With `with_geometry` also returns region centres (mm) and slide labels, which
    the variogram needs to bin pairs by separation and to keep pairs within a
    slide.
    """
    white_thresh = args.white_thresh if white_thresh is None else white_thresh
    index = _stem_index(Path(mask_dir))
    he_index = _stem_index(Path(he_dir)) if he_dir else {}
    out: List[np.ndarray] = []
    coords: List[np.ndarray] = []
    labels_out: List[str] = []

    # The region grid needs an extent and a stem, nothing else. With a tiled
    # dataset those come from the metadata; the real SR is evaluated whole-slide
    # and never tiled, so they come from the mask itself. Same construction.
    if args.tiles_metadata is not None:
        sources = []
        for csv_path in iter_metadata_csvs(Path(args.tiles_metadata)):
            source, _, _ = wsi_extent(csv_path)
            sources.append((Path(source).stem, csv_path))
    else:
        sources = [(stem, None) for stem in sorted(index)]

    for stem, csv_path in sources:
        mpath = index.get(stem)
        if mpath is None:
            print(f"  [skip] no mask for {stem} in {mask_dir}")
            continue

        labels = load_label_mask(mpath)
        grid = (region_grid(csv_path, region_mm=args.region_mm, mpp=args.mpp)
                if csv_path is not None else
                region_grid_from_extent(stem, labels.shape[0], labels.shape[1],
                                        region_mm=args.region_mm, mpp=args.mpp))
        grid = filter_by_tissue(grid, labels,
                                min_tissue_fraction=args.min_tissue_fraction)
        if not grid:
            continue

        he = load_rgb(he_index[stem]) if stem in he_index else None
        # one threshold for the footprint and the lumen count — see
        # uncertainty_phi.ensemble.phi_for_wsi
        footprint = (he_tissue_footprint(he, white_thresh=white_thresh)
                     if he is not None else None)

        for r in grid:
            out.append(phi_struct(
                r.crop(labels),
                r.crop(he) if he is not None else None,
                mpp=args.mpp,
                tissue_mask=r.crop(footprint) if footprint is not None else None,
                min_object_px=args.min_object_px,
                closing_px=args.closing_px,
                white_thresh=white_thresh,
            ))
        if with_geometry:
            coords.append(region_centres_mm(grid, args.mpp))
            labels_out.extend([stem] * len(grid))

    if not out:
        want = [s for s, _ in sources][:3]
        raise SystemExit(
            f"no regions produced from {mask_dir}.\n"
            f"  looked for : {', '.join(want)}{' ...' if len(sources) > 3 else ''}\n"
            f"  masks are  : {', '.join(sorted(index)[:3])}"
            f"{' ...' if len(index) > 3 else ''}\n"
            "If those are different slide sets, --tiles_metadata belongs to "
            "another cohort; drop it and the grid is sized from each mask.")
    phi = np.vstack(out)
    if with_geometry:
        return phi, np.vstack(coords), labels_out
    return phi



# Status palette — verdicts always ship with their text label, never colour alone.
C_GOOD, C_WARN, C_CRIT = "#0ca30c", "#fab219", "#d03b3b"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e3e3df"
# Categorical slots 1-4, for the variogram curves.
C_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")


def make_figure(rows: List[dict], curve: Optional[dict], outpath: Path) -> None:
    """Two panels: the verdict, and the assumption the verdict rests on.

    A is the go/no-go itself — floor SD over between-region SD, per descriptor,
    against the bands that define the verdicts. B is the variogram, because when
    a descriptor's floor comes from the sill, the number is only as good as the
    curve having actually flattened.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_curve = curve is not None and len(curve.get("lag_mm", [])) > 1
    fig, axes = plt.subplots(1, 2 if have_curve else 1,
                             figsize=(15 if have_curve else 8.5, 5.2),
                             squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)

    # --- A: the go/no-go -----------------------------------------------------
    ax = axes[0]
    names = [r["descriptor"] for r in rows]
    ys = np.arange(len(rows))[::-1]           # first descriptor at the top

    ratios = [r["floor_to_signal"] for r in rows]
    finite = [v for v in ratios if v is not None and np.isfinite(v)]
    xmax = max(1.3, min(3.0, (max(finite) * 1.25) if finite else 1.3))

    ax.axvspan(0, 0.5, color=C_GOOD, alpha=0.10, zorder=1, linewidth=0)
    ax.axvspan(0.5, 0.9, color=C_WARN, alpha=0.13, zorder=1, linewidth=0)
    ax.axvspan(0.9, xmax, color=C_CRIT, alpha=0.10, zorder=1, linewidth=0)
    for x in (0.5, 0.9):
        ax.axvline(x, color=C_MUTED, linewidth=1, linestyle=":", zorder=2)

    for y, r in zip(ys, rows):
        ratio, sig = r["floor_to_signal"], r["between_region_sd"]
        if ratio is None or not np.isfinite(ratio):
            ax.text(0.02, y, "no floor estimate for this component",
                    va="center", ha="left", fontsize=9, color=C_MUTED, style="italic")
            continue
        # the bracket, in the same units as the point
        lo, hi = r.get("floor_sd_lower"), r.get("floor_sd_upper")
        if sig and lo is not None and hi is not None and np.isfinite(sig) and sig > 0:
            ax.plot([lo / sig, hi / sig], [y, y], color=C_MUTED, linewidth=1.5,
                    alpha=0.6, zorder=3, solid_capstyle="butt")
        colour = C_GOOD if ratio < 0.5 else (C_WARN if ratio < 0.9 else C_CRIT)
        ax.plot([min(ratio, xmax)], [y], "o", markersize=10, color=colour, zorder=5)
        ax.text(min(ratio, xmax) + 0.03, y,
                f"{r['verdict']}  ·  {r['floor_source']}",
                va="center", ha="left", fontsize=9, color=C_MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10, color=C_INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("floor SD / between-region SD", color=C_MUTED, fontsize=10)
    ax.set_title("A · Is there headroom above the floor?",
                 color=C_INK, fontsize=11, loc="left", pad=10)
    ax.text(0.25, len(rows) - 0.45, "usable", ha="center", fontsize=9, color=C_GOOD)
    ax.text(0.70, len(rows) - 0.45, "marginal", ha="center", fontsize=9, color="#a06a00")
    ax.text((0.9 + xmax) / 2, len(rows) - 0.45, "floor-limited",
            ha="center", fontsize=9, color=C_CRIT)

    # --- B: has the sill actually been reached? ------------------------------
    if have_curve:
        ax = axes[1]
        lags = np.asarray(curve["lag_mm"], dtype=float)
        covs = np.asarray(curve["cov"], dtype=float)
        reached = curve.get("sill_reached", {})
        shown = 0
        for i, name in enumerate(PHI_NAMES):
            gamma = np.sqrt(np.clip(covs[:, i, i], 0, None))
            if not np.isfinite(gamma).any() or np.nanmax(gamma) <= 0:
                continue
            # each descriptor normalised by its own plateau, since the raw units
            # span orders of magnitude and would collapse onto one axis
            ax.plot(lags, gamma / np.nanmax(gamma), linewidth=2,
                    color=C_SERIES[shown % len(C_SERIES)], marker="o", markersize=4,
                    label=f"{name}{'' if reached.get(name, True) else '  (still climbing)'}")
            shown += 1
        ax.axhline(1.0, color=C_MUTED, linewidth=1, linestyle=":", zorder=2)
        # From zero, always. On an auto axis a curve that is already at its sill
        # fills the panel with a few percent of noise and reads as "climbing
        # steeply" — the opposite of what it means.
        ax.set_ylim(0, 1.12)
        ax.set_xlim(left=0)
        ax.set_xlabel("separation (mm)", color=C_MUTED, fontsize=10)
        ax.set_ylabel("floor SD / its own maximum", color=C_MUTED, fontsize=10)
        ax.set_title("B · The variogram: does it flatten?",
                     color=C_INK, fontsize=11, loc="left", pad=10)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for txt in leg.get_texts():
            txt.set_color(C_MUTED)

    fig.suptitle("Per-descriptor floor — the go/no-go",
                 color=C_INK, fontsize=13, x=0.011, ha="left", y=0.99)
    fig.text(0.011, 0.015,
             "A flat curve in B means the sill is the fully-decorrelated limit, so "
             "it over-estimates the floor and under-states bias — the safe direction. "
             "A curve still climbing means its bound is an under-estimate.",
             color=C_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(outpath, dpi=150, facecolor="white")
    print(f"wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser("Per-descriptor biological floor (go/no-go pilot)")
    ap.add_argument("--real_psr", type=Path, required=True,
                    help="Directory of REAL PSR WSI masks (level A).")
    ap.add_argument("--tiles_metadata", type=Path, default=None,
                    help="Dataset root with per-WSI tiles_metadata.csv. OPTIONAL: "
                         "without it the region grid is sized from each mask in "
                         "--real_psr, which is what the real SR arm wants since "
                         "it is evaluated whole-slide and never tiled. Grids from "
                         "the two routes differ by up to one region row/column, "
                         "so do not compare runs across them.")
    ap.add_argument("--real_he", type=Path, default=None,
                    help="Reconstructed real H&E WSIs. Enables the cross-stain "
                         "UPPER bound on the two stain-invariant descriptors.")
    ap.add_argument("--real_psr_rgb", type=Path, default=None,
                    help="Real PSR RGB WSIs. Required alongside --real_he for "
                         "the cross-stain bound: the stain-invariant terms have "
                         "to be measured on BOTH images, and only the masks are "
                         "read from --real_psr.")
    ap.add_argument("--psr_level_b", type=Path, default=None,
                    help="A SECOND real PSR level for the same cases. Gives the "
                         "direct cross-level floor and supersedes the bracket.")
    ap.add_argument("--outdir", type=Path, default=Path("floor_pilot"))

    ap.add_argument("--region_mm", type=float, default=1.5)
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP,
                    help="Microns per pixel OF THE RECONSTRUCTION. [%(default)s]")
    ap.add_argument("--min_tissue_fraction", type=float, default=0.25)
    ap.add_argument("--min_object_px", type=int, default=16)
    ap.add_argument("--white_thresh", type=float, default=WHITE_THRESH,
                    help="Whitespace cut for the H&E terms. Every channel must\n"
                         "clear it, so compare against the per-pixel channel\n"
                         "MINIMUM, not a Fiji 8-bit grey level. [%(default)s]")
    ap.add_argument("--white_thresh_psr", type=float, default=None,
                    help="Whitespace cut for --real_psr_rgb. PSR and H&E sit at "
                         "different whitespace levels, so they get their own "
                         "thresholds. Defaults to --white_thresh.")
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

    if args.white_thresh_psr is None:
        args.white_thresh_psr = args.white_thresh

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
    if args.real_he and args.real_psr_rgb:
        print(f"[2/4] cross-stain upper bound: H&E {args.real_he} vs "
              f"PSR {args.real_psr_rgb}")
        # phi_real already carries the H&E-referenced terms measured on the H&E.
        # The other side must be the SAME descriptors measured on the real PSR
        # RGB — passing --real_he twice makes delta identically zero.
        phi_he = phi_real
        phi_psr_rgb = _phi_for_dir(args.real_psr, args,
                                   he_dir=args.real_psr_rgb,
                                   white_thresh=args.white_thresh_psr)
        n = min(len(phi_he), len(phi_psr_rgb))
        upper = cross_stain_floor(phi_he[:n], phi_psr_rgb[:n])
    elif args.real_he:
        print("[2/4] --real_he without --real_psr_rgb: no cross-stain bound. "
              "Both sides of the comparison need their own image; one image "
              "twice gives a zero floor, which inflates bias.")
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
    vg, curve = None, None
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
    if vg is not None:
        payload["estimates"]["variogram_sill"] = vg.summary()
    if curve is not None:
        payload["variogram_curve"] = {
            "lag_mm": np.asarray(curve["lag_mm"]).tolist(),
            "n_pairs": np.asarray(curve["n_pairs"]).tolist(),
            "floor_sd_per_lag": {
                n: np.sqrt(np.clip(np.asarray(curve["cov"])[:, i, i], 0, None)).tolist()
                for i, n in enumerate(PHI_NAMES)
            },
            "sill_reached": curve.get("sill_reached"),
            "tail_drift": curve.get("tail_drift"),
        }

    make_figure(rows, curve, args.outdir / "floor.png")

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
