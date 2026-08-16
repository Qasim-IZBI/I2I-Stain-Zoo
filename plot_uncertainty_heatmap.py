#!/usr/bin/env python
"""Per-region uncertainty painted back onto the slide.

`per_region.csv` already carries each region's box, so mapping σ back to WSI
geometry is a lookup rather than a recomputation. Two outputs per slide:

* a **PNG figure** — σ on the top row, σ/μ on the bottom, one column per
  descriptor, for reading
* a **float32 TIF** per descriptor per metric — the same raster, for overlaying
  in Fiji or QuPath against the slide it came from

    python plot_uncertainty_heatmap.py \\
        --phi_csv ./phi_uncertainty/per_region.csv \\
        --downsample 32 --outdir ./uncertainty_heatmaps/

Why both σ and σ/μ
------------------
σ_β₀ scales with how much collagen a region contains, so a raw σ map can be a
collagen-density map wearing an uncertainty label. The coefficient of variation
divides that out. Where the two disagree, the raw map was showing you tissue
content; where they agree, the model really is less certain there. Neither alone
answers it, which is why the figure carries both rather than making you choose.

Regions dropped by the tissue filter stay blank rather than being interpolated
over — an absent measurement is not a low one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile

from uncertainty_phi.descriptors import PHI_NAMES

# Same colormap as uncertainty.py's pixel-space heatmaps, so the two kinds of
# uncertainty map in this repository read alike.
CMAP = "magma"
C_INK, C_MUTED, C_ABSENT = "#0b0b0b", "#52514e", "#e8e8e4"


def raster_for(group: pd.DataFrame, column: str, downsample: int,
               shape: tuple) -> np.ndarray:
    """Paint one column's per-region values into a downsampled slide raster."""
    out = np.full(shape, np.nan, dtype=np.float32)
    if column not in group.columns:
        return out
    for r in group.itertuples():
        v = getattr(r, column)
        if v is None or not np.isfinite(v):
            continue
        out[r.y0 // downsample:max(r.y0 // downsample + 1, r.y1 // downsample),
            r.x0 // downsample:max(r.x0 // downsample + 1, r.x1 // downsample)] = v
    return out


def make_figure(rasters: dict, names: List[str], wsi: str, metric: str,
                outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(1, len(names))
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol + 1.2, 6.4),
                             squeeze=False)

    for col, name in enumerate(names):
        for row, (kind, label) in enumerate(((metric, "σ"), "cv σ/μ".split(" ", 1))):
            key = (name, "value" if row == 0 else "cv")
            ax = axes[row][col]
            arr = rasters.get(key)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(C_ABSENT)
            finite = arr[np.isfinite(arr)] if arr is not None else np.empty(0)
            if finite.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color=C_MUTED,
                        style="italic")
                continue
            if float(np.ptp(finite)) == 0.0:
                # A constant field is not a map. Rendered as a heatmap it comes
                # out a solid block with a decorative colourbar, which reads as
                # structure that is not there — most often a descriptor with no
                # ensemble spread at all.
                ax.text(0.5, 0.5, f"constant\n{finite[0]:.3g}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8,
                        color=C_MUTED, style="italic")
                continue
            # blank where a region was filtered out; an absent measurement is
            # not a low one, so it must not read as dark
            ax.set_facecolor(C_ABSENT)
            lo, hi = np.nanpercentile(arr, [1, 99])
            im = ax.imshow(arr, cmap=CMAP, vmin=lo, vmax=hi if hi > lo else lo + 1e-9,
                           interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6.5, colors=C_MUTED)
            cb.outline.set_visible(False)
            # plain ticks: matplotlib's offset notation ("1e-5+2.608e-1") is
            # unreadable beside a colour ramp, and a near-constant field is
            # exactly where it kicks in
            fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-3, 4))
            cb.ax.yaxis.set_major_formatter(fmt)
            if row == 0:
                ax.set_title(name, fontsize=8.5, color=C_INK, pad=6)
            if col == 0:
                ax.set_ylabel("σ" if row == 0 else "σ / μ", fontsize=9,
                              color=C_INK)

    fig.suptitle(f"{wsi} — regional uncertainty ({metric})",
                 fontsize=12, color=C_INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.012,
             "Top row is the spread itself; bottom divides by the regional mean. "
             "σ for a count-based descriptor rises with how much structure a "
             "region holds, so where the two rows disagree the top one is "
             "showing tissue content rather than uncertainty. Blank = region "
             "dropped by the tissue filter.",
             fontsize=7.5, color=C_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(outpath, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Per-region uncertainty heatmaps over the slide")
    ap.add_argument("--phi_csv", type=Path, required=True,
                    help="per_region.csv from compute_phi_uncertainty.py.")
    ap.add_argument("--outdir", type=Path, default=Path("uncertainty_heatmaps"))
    ap.add_argument("--metric", default="sd_total",
                    choices=("sd_total", "sd_procedural", "sd_data"),
                    help="Which variance component to map. [%(default)s]")
    ap.add_argument("--descriptors", nargs="*", default=None,
                    help="Subset of PHI_NAMES. Default: all present in the CSV.")
    ap.add_argument("--downsample", type=int, default=32,
                    help="Slide pixels per raster pixel. At 2048 px regions, 32 "
                         "gives 64-pixel blocks. [%(default)s]")
    ap.add_argument("--no_tif", action="store_true",
                    help="Skip the float32 rasters, write only the figures.")
    args = ap.parse_args()

    df = pd.read_csv(args.phi_csv)
    names = args.descriptors or [n for n in PHI_NAMES
                                 if f"{args.metric}_{n}" in df.columns]
    if not names:
        raise SystemExit(
            f"no '{args.metric}_<descriptor>' columns in {args.phi_csv}. "
            f"Re-run compute_phi_uncertainty.py — older runs wrote only the "
            f"summed scalars, which cannot be mapped per descriptor."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for wsi, group in df.groupby("wsi", sort=False):
        stem = Path(str(wsi)).stem
        # The regions cover the slide up to the last whole region; partial edge
        # regions were dropped when the grid was built, so this is the extent
        # that was measured, not the full slide.
        shape = (int(np.ceil(group["y1"].max() / args.downsample)),
                 int(np.ceil(group["x1"].max() / args.downsample)))

        rasters = {}
        for name in names:
            sd = raster_for(group, f"{args.metric}_{name}", args.downsample, shape)
            mu = raster_for(group, f"mu_{name}", args.downsample, shape)
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(np.abs(mu) > 0, sd / np.abs(mu), np.nan)
            rasters[(name, "value")] = sd
            rasters[(name, "cv")] = cv.astype(np.float32)

            if not args.no_tif:
                for kind, arr in (("", sd), ("_cv", cv.astype(np.float32))):
                    path = args.outdir / f"{stem}_{name}_{args.metric}{kind}.tif"
                    tifffile.imwrite(
                        str(path), arr, compression="zlib",
                        description=json.dumps({
                            "wsi": str(wsi), "descriptor": name,
                            "metric": args.metric + kind,
                            "downsample": args.downsample,
                            "note": "NaN = region dropped by the tissue filter",
                        }),
                    )

        make_figure(rasters, names, str(wsi), args.metric,
                    args.outdir / f"{stem}_uncertainty.png")
        manifest.append({"wsi": str(wsi), "n_regions": int(len(group)),
                         "raster_shape": list(shape)})
        print(f"{stem}: {len(group)} regions -> {shape[0]}x{shape[1]} raster")

    with open(args.outdir / "heatmaps.json", "w") as fh:
        json.dump({"per_wsi": manifest, "descriptors": names,
                   "params": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()}}, fh, indent=2)

    print(f"\nwrote {len(manifest)} slide(s) to {args.outdir}")
    print("Compare the two rows before reading the top one as uncertainty: for a "
          "count-based descriptor σ rises with how much structure a region holds.")


if __name__ == "__main__":
    main()
