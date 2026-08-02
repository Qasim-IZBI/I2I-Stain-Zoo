#!/usr/bin/env python
"""Stain-perturbation sensitivity — how much "bias" is really the segmenter?

Applying one segmenter to both arms cancels *anatomy-driven* error but not
*appearance-driven* error, because appearance is precisely where real and virtual
PSR differ (kidney_ood_data_plan.md §6.2). This measures the second component
without needing any manual annotation.

Method — hold anatomy fixed, move only appearance:

    t = 0   a real PSR slide, untouched
    t = 1   the same real slide wearing the virtual stain's colour statistics

Segment every step. The tissue never changes, so any movement in CPA, β₀, β₁ or
dispersion is measurement artefact, and its size is the error bar you must put on
any bias number.

Two phases, with an nnU-Net run in between:

    1. make-series   real + virtual WSIs -> perturbed copies at each t
    2. (segment)     scripts/segment_psr_perturbation.sh
    3. analyse       masks at each t -> sensitivity table

Usage
-----
    python stain_sensitivity.py make-series \\
        --real_psr   /path/real_psr_wsis/ \\
        --virtual_psr /path/ensemble/.../reconstructed/model_01/ \\
        --outdir     /path/perturbation/

    sbatch I2I-Stain-Zoo/scripts/segment_psr_perturbation.sh

    python stain_sensitivity.py analyse \\
        --masks /path/perturbation/masks/ \\
        --tiles_metadata /path/tiles/testB \\
        --outdir /path/perturbation/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile

from uncertainty_phi.descriptors import PHI_NAMES, he_bright, phi_struct
from uncertainty_phi.ensemble import _stem_index, load_label_mask, load_rgb
from uncertainty_phi.perturb import (
    appearance_gap,
    clipped_fraction,
    interpolate_stats,
    pool_stats,
    reinhard_transfer,
    stain_stats,
)
from uncertainty_phi.regions import (
    SOURCE_MPP,
    filter_by_tissue,
    iter_metadata_csvs,
    region_grid,
    wsi_extent,
)

DEFAULT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _params(args) -> dict:
    """JSON-safe parameter record. argparse stashes the subcommand callable in
    `func`, which is not serialisable."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items() if k != "func" and not callable(v)}


def _tissue_of(rgb: np.ndarray) -> np.ndarray:
    """Crude tissue mask for statistics: anything not near-white."""
    return ~he_bright(rgb, white_thresh=0.85)


# ---------------------------------------------------------------- make-series

def cmd_make_series(args) -> None:
    real_idx = _stem_index(Path(args.real_psr))
    virt_idx = _stem_index(Path(args.virtual_psr))
    if not real_idx:
        raise SystemExit(f"no images under {args.real_psr}")
    if not virt_idx:
        raise SystemExit(f"no images under {args.virtual_psr}")

    print(f"[1/3] virtual stain statistics over {len(virt_idx)} WSI")
    virt_stats = pool_stats([
        stain_stats(load_rgb(p), _tissue_of(load_rgb(p))) for p in virt_idx.values()
    ])

    print(f"[2/3] real stain statistics over {len(real_idx)} WSI")
    real_each = {s: stain_stats(load_rgb(p), _tissue_of(load_rgb(p)))
                 for s, p in real_idx.items()}
    real_pooled = pool_stats(list(real_each.values()))

    gap = appearance_gap(real_pooled, virt_stats)
    print("      appearance gap (virtual - real), LAB:")
    print(f"        dL={gap['delta_mean_L']:+.2f}  da={gap['delta_mean_a']:+.2f}  "
          f"db={gap['delta_mean_b']:+.2f}")
    print("        in units of the real spread: "
          + ", ".join(f"{v:+.2f}" for v in gap["delta_mean_over_sd"]))

    fractions = tuple(args.fractions)
    outdir = Path(args.outdir)
    clip_report: Dict[str, Dict[str, float]] = {}

    print(f"[3/3] writing series at t = {list(fractions)}")
    for stem, path in real_idx.items():
        rgb = load_rgb(path)
        tissue = _tissue_of(rgb)
        src = real_each[stem]
        clip_report[stem] = {}
        for t in fractions:
            dst = interpolate_stats(src, virt_stats, t)
            out = reinhard_transfer(rgb, src, dst, tissue_mask=tissue)
            frac = clipped_fraction(rgb, src, dst, tissue_mask=tissue)
            clip_report[stem][f"{t:.2f}"] = frac

            d = outdir / f"t{t:.2f}"
            d.mkdir(parents=True, exist_ok=True)
            # _0000 suffix: nnU-Net single-channel input convention
            tifffile.imwrite(str(d / f"{stem}_0000.tif"),
                             (out * 255).astype(np.uint8))
        print(f"      {stem}: max clipped {max(clip_report[stem].values()):.3%}")

    worst = max((v for d in clip_report.values() for v in d.values()), default=0.0)
    meta = {
        "fractions": list(fractions),
        "real_stats_pooled": real_pooled.as_dict(),
        "virtual_stats_pooled": virt_stats.as_dict(),
        "appearance_gap": gap,
        "clipped_fraction": clip_report,
        "max_clipped_fraction": worst,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "series.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    if worst > 0.01:
        print(f"\n[!] up to {worst:.2%} of tissue pixels clip out of gamut.")
        print("    Clipping is non-invertible and breaks the fixed-anatomy premise;")
        print("    narrow --fractions if this is large at the t you care about.")
    print(f"\nwrote {outdir}/t*/ and {outdir / 'series.json'}")
    print("next: sbatch I2I-Stain-Zoo/scripts/segment_psr_perturbation.sh")


# -------------------------------------------------------------------- analyse

def _phi_at(mask_dir: Path, args) -> Dict[str, np.ndarray]:
    """{wsi_stem: [n_regions, d]} for one perturbation step."""
    idx = _stem_index(mask_dir)
    out: Dict[str, np.ndarray] = {}
    for csv_path in iter_metadata_csvs(Path(args.tiles_metadata)):
        source, _, _ = wsi_extent(csv_path)
        stem = Path(source).stem
        if stem not in idx:
            continue
        labels = load_label_mask(idx[stem])
        grid = filter_by_tissue(
            region_grid(csv_path, region_mm=args.region_mm, mpp=args.mpp),
            labels, min_tissue_fraction=args.min_tissue_fraction,
        )
        if grid:
            out[stem] = np.vstack([
                phi_struct(r.crop(labels), mpp=args.mpp,
                           min_object_px=args.min_object_px,
                           closing_px=args.closing_px) for r in grid
            ])
    return out


def cmd_analyse(args) -> None:
    root = Path(args.masks)
    steps = sorted(root.glob("t*"), key=lambda p: float(p.name[1:]))
    if not steps:
        raise SystemExit(f"no t*/ directories under {root}")
    ts = [float(p.name[1:]) for p in steps]
    print(f"found steps t = {ts}")

    per_t = {t: _phi_at(p, args) for t, p in zip(ts, steps)}
    base = per_t[ts[0]]
    if not base:
        raise SystemExit("no regions at the baseline step")

    rows: List[dict] = []
    for t in ts[1:]:
        cur = per_t[t]
        shared = sorted(set(base) & set(cur))
        for j, name in enumerate(PHI_NAMES):
            b = np.concatenate([base[s][:, j] for s in shared]) if shared else np.array([])
            c = np.concatenate([cur[s][:, j] for s in shared]) if shared else np.array([])
            n = min(len(b), len(c))
            b, c = b[:n], c[:n]
            ok = np.isfinite(b) & np.isfinite(c)
            if ok.sum() < 2:
                continue
            d = c[ok] - b[ok]
            spread = float(np.nanstd(b[ok], ddof=1))
            rows.append({
                "t": t,
                "descriptor": name,
                "n_regions": int(ok.sum()),
                "mean_shift": float(d.mean()),
                "abs_mean_shift": float(np.abs(d).mean()),
                "shift_sd": float(d.std(ddof=1)),
                "between_region_sd": spread,
                # Zero biological spread with a non-zero shift is the WORST case,
                # not an unknown one: the artefact exceeds all real variation.
                # Reporting it as n/a would let it slip past the flag below.
                "shift_over_region_sd": (
                    float(abs(d.mean()) / spread) if spread
                    else (float("inf") if abs(d.mean()) > 0 else 0.0)
                ),
            })

    df = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "stain_sensitivity.csv", index=False)

    print("\n=== artefact attributable to the real->virtual colour gap ===")
    at1 = df[df["t"] == max(ts)] if len(df) else df
    hdr = f"{'descriptor':22s} {'mean shift':>12s} {'region_sd':>11s} {'shift/sd':>9s}"
    print(hdr); print("-" * len(hdr))
    for _, r in at1.iterrows():
        q = r["shift_over_region_sd"]
        print(f"{r['descriptor']:22s} {r['mean_shift']:+12.5f} {r['between_region_sd']:11.5f} "
              f"{'      inf' if q is not None and np.isinf(q) else ('      n/a' if q is None else f'{q:9.3f}')}")

    safe_rows = [
        {k: ("inf" if isinstance(v, float) and np.isinf(v) else v) for k, v in r.items()}
        for r in rows
    ]
    with open(outdir / "stain_sensitivity.json", "w") as fh:
        json.dump({"steps": ts, "rows": safe_rows,
                   "params": _params(args)}, fh, indent=2)

    big = at1[at1["shift_over_region_sd"].fillna(0) > 0.25]["descriptor"].tolist()
    if big:
        print(f"\n[!] appearance-sensitive: {', '.join(big)}")
        print("    For these the segmenter reacts to colour at a scale comparable to")
        print("    real biological variation. Fold this shift into the floor before")
        print("    quoting any bias, and treat a bias of similar size as unproven.")
    else:
        print("\nAll descriptors move well below the biological spread — the")
        print("cancellation argument holds and bias numbers are not colour-driven.")
    print(f"\nwrote {outdir / 'stain_sensitivity.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser("Stain-perturbation sensitivity of the segmenter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-series", help="write perturbed copies of the real WSIs")
    m.add_argument("--real_psr", type=Path, required=True,
                   help="Real PSR WSIs (RGB, not masks).")
    m.add_argument("--virtual_psr", type=Path, required=True,
                   help="Virtual PSR WSIs, to supply the target colour statistics.")
    m.add_argument("--outdir", type=Path, required=True)
    m.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    m.set_defaults(func=cmd_make_series)

    a = sub.add_parser("analyse", help="descriptor drift across the segmented series")
    a.add_argument("--masks", type=Path, required=True,
                   help="Directory of t*/ mask dirs from the segmentation step.")
    a.add_argument("--tiles_metadata", type=Path, required=True)
    a.add_argument("--outdir", type=Path, required=True)
    a.add_argument("--region_mm", type=float, default=1.5)
    a.add_argument("--mpp", type=float, default=SOURCE_MPP)
    a.add_argument("--min_tissue_fraction", type=float, default=0.25)
    a.add_argument("--min_object_px", type=int, default=16)
    a.add_argument("--closing_px", type=int, default=0)
    a.set_defaults(func=cmd_analyse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
