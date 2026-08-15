#!/usr/bin/env python
"""Choose `--white_thresh` from the data instead of guessing it.

`lumen_fraction` and `tissue_fraction` are cut out of the H&E by a single
brightness threshold, and the default 0.85 is scanner-dependent: on the UC liver
cohort it sat above the lumens entirely and `mu_lumen_fraction` came back at
~1e-5. Lowering it is not the fix on its own — a threshold parked on the slope of
the tissue distribution gives a number that is an artefact of where you cut.

**Pick a plateau, not a value.** Between the tissue mode and the whitespace mode
there is a valley in the brightness histogram; a threshold there is insensitive
to small changes in itself, which is what makes it reproducible across slides and
across the two stains.

    python calibrate_white_thresh.py \\
        --he_dir /path/export_rgb/testA \\
        --tiles_metadata /path/tiles/testA \\
        --outdir ./white_thresh_liver/

Outputs
-------
white_thresh.csv     lumen/tissue fraction at every threshold, per WSI
white_thresh.png     the histogram, the sweep, and the sensitivity curve
white_thresh.json    the suggested threshold and how it was derived

Note on what is being thresholded
---------------------------------
`he_bright` requires EVERY channel to clear the cut, i.e. it thresholds the
per-pixel channel MINIMUM. An 8-bit conversion in Fiji shows a channel average,
which is always the larger of the two — so a grey level read off a converted
image is an upper bound on the threshold, never the threshold. This script works
on the channel minimum throughout, so its x-axis is directly comparable to
`--white_thresh`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from uncertainty_phi.descriptors import WHITE_THRESH
from uncertainty_phi.regions import (
    SOURCE_MPP,
    iter_metadata_csvs,
    region_grid,
    region_grid_from_extent,
    wsi_extent,
)

# Categorical slots 1 and 3 of the validated palette; slide lines are muted ink so
# the pooled mean stays the figure's subject.
C_MEAN = "#2a78d6"
C_ACCENT = "#1baf7a"
C_SLIDE = "#9a9a94"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#e3e3df"


def channel_min(path: Path, row_block: int = 4096) -> np.ndarray:
    """[H,W] uint8 per-pixel minimum over RGB, read in row blocks.

    A whole-slide RGB is several GB; `memmap` keeps it off the heap when the
    export is uncompressed, and the row-block loop bounds the peak either way.
    The channel minimum is the only statistic the threshold acts on, so reducing
    to it up front makes every subsequent threshold a cheap comparison.
    """
    try:
        arr = tifffile.memmap(str(path))
    except (ValueError, MemoryError):        # compressed or unmappable
        arr = tifffile.imread(str(path))

    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.uint8)

    h = arr.shape[0]
    out = np.empty(arr.shape[:2], dtype=np.uint8)
    for y0 in range(0, h, row_block):
        y1 = min(y0 + row_block, h)
        out[y0:y1] = np.asarray(arr[y0:y1, :, :3]).min(axis=2)
    return out


def sweep_slide(mn: np.ndarray, thresholds: np.ndarray) -> List[dict]:
    """lumen/tissue fraction at each threshold, on the WSI-level footprint.

    Deliberately not per-region: `binary_fill_holes` only fills background that
    is fully enclosed, so a lumen straddling a region border is not enclosed
    within that crop and would be missed. Doing the fill on the whole slide is
    what `phi_for_wsi` does, so these numbers are directly comparable to
    `mu_lumen_fraction` in per_region.csv.
    """
    rows = []
    scale = 255.0
    for t in thresholds:
        bright = mn > (t * scale)
        footprint = ndimage.binary_fill_holes(~bright)
        n_tissue = int(np.count_nonzero(footprint))
        if n_tissue == 0:
            rows.append({"white_thresh": float(t), "lumen_fraction": np.nan,
                         "tissue_fraction": 0.0, "n_tissue_px": 0})
            continue
        rows.append({
            "white_thresh": float(t),
            "lumen_fraction": float(np.count_nonzero(bright & footprint) / n_tissue),
            "tissue_fraction": float(n_tissue / footprint.size),
            "n_tissue_px": n_tissue,
        })
    return rows


def tissue_histogram(mn: np.ndarray, regions, bins: int) -> np.ndarray:
    """Counts of the channel minimum over tissue-bearing regions only.

    Whole-slide counts are swamped by slide background, which sits far to the
    right and tells you nothing about where tissue ends and lumen begins.
    """
    edges = np.linspace(0, 1, bins + 1)
    total = np.zeros(bins, dtype=np.int64)
    for r in regions:
        crop = mn[r.y0:r.y1, r.x0:r.x1]
        if crop.size == 0:
            continue
        counts, _ = np.histogram(crop.ravel() / 255.0, bins=edges)
        total += counts
    return total


def suggest_threshold(counts: np.ndarray, centres: np.ndarray,
                      lo: float = 0.35, hi: float = 0.97) -> Optional[dict]:
    """The valley between the tissue mode and the whitespace mode.

    Smoothed so single-bin noise cannot pass for a minimum. Returns None when the
    two modes are not separated — in which case there is no plateau to sit on and
    the threshold has to be justified some other way.
    """
    band = (centres >= lo) & (centres <= hi)
    if band.sum() < 5:
        return None
    dens = ndimage.gaussian_filter1d(counts[band].astype(float), sigma=1.5)
    x = centres[band]

    # local maxima, tallest two = tissue mode and whitespace mode
    peaks = [i for i in range(1, len(dens) - 1)
             if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1]]
    if len(peaks) < 2:
        return None
    peaks.sort(key=lambda i: dens[i], reverse=True)
    a, b = sorted(peaks[:2])
    if b - a < 2:
        return None

    valley = a + int(np.argmin(dens[a:b + 1]))
    return {
        "threshold": float(x[valley]),
        "tissue_mode": float(x[a]),
        "whitespace_mode": float(x[b]),
        "valley_depth_ratio": float(dens[valley] / max(dens[a], dens[b])),
    }


def footprint_breakdown(t: np.ndarray, tissue: np.ndarray,
                        max_jump: float = 0.005) -> Optional[float]:
    """Lowest threshold at which the footprint stops being the tissue footprint.

    Slide background is only excluded from the footprint while it reads as
    bright. Once the cut rises past it, `~bright` contains the background,
    `binary_fill_holes` absorbs it, and `tissue_fraction` jumps — on the UC liver
    H&E by 0.6% at 0.700 and then 21% at 0.725, against 0.05-0.09% per
    step while it is stable. Past that point lumen_fraction is not a
    high estimate of the same quantity, it is a measurement of a different
    object, and it goes non-monotonic to prove it.

    Returns the first threshold in the broken regime, or None if the footprint
    is stable across the whole sweep.
    """
    for i in range(1, len(t)):
        if tissue[i - 1] > 0 and abs(tissue[i] / tissue[i - 1] - 1.0) > max_jump:
            return float(t[i])
    return None


def find_plateau(t: np.ndarray, lumen: np.ndarray,
                 tissue: Optional[np.ndarray] = None,
                 tolerance: float = 2.0, abs_max: float = 1.0,
                 max_jump: float = 0.005) -> dict:
    """The stretch of thresholds over which lumen_fraction barely moves.

    Sensitivity is |d ln(lumen)/dt|: the relative change in the measurement per
    unit of threshold, scale-free because lumen_fraction spans orders of
    magnitude across a sweep. At the 0.025 step used here, sensitivity 1.0 means
    ~2.5% per step and sensitivity 5 means ~13%.

    Two conditions, and the second is the one that matters. Relative: within
    `tolerance` x the flattest point. **Absolute**: below `abs_max`. Without the
    absolute test a uniformly sloped curve reports its own middle as a plateau —
    which is exactly what the UC liver H&E does, sloping ~12% per step from 0.50
    to 0.675 with no flat spot anywhere.

    Thresholds at or above `footprint_breakdown` are excluded before any of this,
    since their lumen values describe a footprint that has swallowed the slide
    background.
    """
    out: dict = {"lo": None, "hi": None, "reason": None, "valid_hi": None,
                 "sensitivity_floor": None, "tolerance": tolerance,
                 "abs_max": abs_max}

    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.abs(np.gradient(np.log(np.where(lumen > 0, lumen, np.nan)), t))
    out["sensitivity"] = sens.tolist()

    valid = np.isfinite(sens)
    if tissue is not None:
        broken_at = footprint_breakdown(t, np.asarray(tissue, dtype=float), max_jump)
        out["breakdown"] = broken_at
        if broken_at is not None:
            # the step INTO the breakdown is already contaminated, and so is the
            # central difference on either side of it
            valid &= t < broken_at
            if valid.any():
                out["valid_hi"] = float(t[valid].max())
    if valid.sum() < 3:
        out["reason"] = "fewer than three usable thresholds below the breakdown"
        return out

    floor = float(np.nanmin(sens[valid]))
    out["sensitivity_floor"] = floor
    if floor > abs_max:
        pct = (np.exp(floor * float(np.median(np.diff(t)))) - 1) * 100
        out["reason"] = (
            f"no threshold is flat: the least sensitive point still moves "
            f"lumen_fraction by {pct:.0f}% per step. The tissue and whitespace "
            f"brightness distributions overlap, so any value here is a "
            f"convention, not a measurement."
        )
        return out

    inside = valid & (sens <= min(max(floor * tolerance, floor + 1e-12), abs_max))
    best, run = None, None
    for i, flag in enumerate(inside):
        if flag:
            run = (i, i) if run is None else (run[0], i)
            if best is None or (run[1] - run[0]) > (best[1] - best[0]):
                best = run
        else:
            run = None
    if best is None:
        out["reason"] = "no contiguous flat run"
        return out

    out["lo"], out["hi"] = float(t[best[0]]), float(t[best[1]])
    return out


def make_figure(df: pd.DataFrame, hist_counts: np.ndarray, hist_centres: np.ndarray,
                suggestion: Optional[dict], plateau: Optional[dict],
                current: float, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slides = sorted(df["wsi"].unique())
    mean = df.groupby("white_thresh")[["lumen_fraction", "tissue_fraction"]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)

    def mark(ax, shade: bool = True):
        """Invalid region, plateau band, then the two vertical rules."""
        if shade and plateau.get("breakdown") is not None:
            ax.axvspan(plateau["breakdown"], ax.get_xlim()[1], color=C_MUTED,
                       alpha=0.10, zorder=1, linewidth=0)
        if shade and plateau.get("lo") is not None:
            lo, hi = plateau["lo"], plateau["hi"]
            if hi - lo < 1e-9:              # single point: give it visible width
                half = 0.4 * float(np.median(np.diff(mean.index.to_numpy())))
                lo, hi = lo - half, hi + half
            ax.axvspan(lo, hi, color=C_ACCENT, alpha=0.12, zorder=1, linewidth=0)
        ax.axvline(current, color=C_MUTED, linewidth=1.5, linestyle=":", zorder=2)
        if suggestion:
            ax.axvline(suggestion["threshold"], color=C_ACCENT, linewidth=2, zorder=3)

    # --- A: where tissue ends and whitespace begins -------------------------
    ax = axes[0]
    ax.fill_between(hist_centres, hist_counts, step="mid",
                    color=C_MEAN, alpha=0.18, zorder=1)
    ax.plot(hist_centres, hist_counts, drawstyle="steps-mid",
            color=C_MEAN, linewidth=2, zorder=4)
    ax.set_yscale("log")
    mark(ax, shade=False)
    nz = np.nonzero(hist_counts)[0]
    if nz.size:
        pad = 0.03
        ax.set_xlim(max(0.0, hist_centres[nz[0]] - pad),
                    min(1.0, hist_centres[nz[-1]] + pad))
    ax.set_xlabel("channel minimum / 255", color=C_MUTED, fontsize=10)
    ax.set_ylabel("pixels (log)", color=C_MUTED, fontsize=10)
    ax.set_title("A · Brightness inside tissue regions",
                 color=C_INK, fontsize=11, loc="left", pad=10)
    if suggestion:
        # place the label on whichever side of the rule has room, so it never
        # sits on top of the "current" rule next to it
        x0, x1 = ax.get_xlim()
        right = suggestion["threshold"] < 0.5 * (x0 + x1)
        ax.annotate(f"valley {suggestion['threshold']:.3f}",
                    xy=(suggestion["threshold"], 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(6 if right else -6, -8), textcoords="offset points",
                    color=C_ACCENT, fontsize=9, fontweight="bold",
                    va="top", ha="left" if right else "right",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # --- B: the sweep --------------------------------------------------------
    ax = axes[1]
    for s in slides:
        d = df[df["wsi"] == s].sort_values("white_thresh")
        ax.plot(d["white_thresh"], d["lumen_fraction"],
                color=C_SLIDE, linewidth=1.2, alpha=0.75, zorder=2)
    ax.plot(mean.index, mean["lumen_fraction"], color=C_MEAN, linewidth=2.5,
            marker="o", markersize=5, zorder=5, label=f"mean of {len(slides)} slides")
    ax.plot([], [], color=C_SLIDE, linewidth=1.2, label="per slide")
    ax.set_yscale("log")
    # The collapse to zero past the lumen peak would otherwise stretch the axis
    # over decades nobody reads. Four decades below the maximum keeps the
    # collapse visible as a cliff without giving it most of the panel.
    top = float(np.nanmax(mean["lumen_fraction"].to_numpy()))
    if np.isfinite(top) and top > 0:
        ax.set_ylim(top * 1e-4, top * 2.0)
    mark(ax)
    ax.set_xlabel("--white_thresh", color=C_MUTED, fontsize=10)
    ax.set_ylabel("lumen_fraction (log)", color=C_MUTED, fontsize=10)
    ax.set_title("B · What the threshold buys you",
                 color=C_INK, fontsize=11, loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(C_MUTED)

    # --- C: the plateau, made objective -------------------------------------
    ax = axes[2]
    ts = mean.index.to_numpy()
    sens = np.asarray(plateau["sensitivity"])
    ax.plot(ts, sens, color=C_MEAN, linewidth=2.5, marker="o", markersize=5, zorder=5)
    mark(ax)
    if plateau.get("breakdown") is not None:
        ax.annotate("footprint broken\n(background absorbed)",
                    xy=(plateau["breakdown"], 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -8), textcoords="offset points",
                    color=C_MUTED, fontsize=8, va="top", ha="left")
    if plateau.get("lo") is not None:
        ax.annotate(f"plateau {plateau['lo']:.3f}–{plateau['hi']:.3f}",
                    xy=(0.5 * (plateau["lo"] + plateau["hi"]), 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -8), textcoords="offset points",
                    color=C_ACCENT, fontsize=9, fontweight="bold",
                    va="top", ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("--white_thresh", color=C_MUTED, fontsize=10)
    ax.set_ylabel("|d ln(lumen) / d threshold|", color=C_MUTED, fontsize=10)
    ax.set_title("C · Sensitivity — lower is a flatter plateau",
                 color=C_INK, fontsize=11, loc="left", pad=10)

    note = (f"dotted = current {current:g}"
            + (f"   ·   green line = histogram valley {suggestion['threshold']:.3f}"
               if suggestion else "   ·   no valley found: modes not separated")
            + (f"   ·   green band = plateau {plateau['lo']:.3f}–{plateau['hi']:.3f}"
               if plateau.get("lo") is not None
               else "   ·   NO PLATEAU: " + (plateau.get("reason") or "").split(".")[0])
            + (f"   ·   grey = footprint broken at {plateau['breakdown']:.3f}"
               if plateau.get("breakdown") is not None else ""))
    fig.text(0.011, 0.015, note, color=C_MUTED, fontsize=9)
    fig.suptitle("Choosing --white_thresh: sit in the valley, not on the slope",
                 color=C_INK, fontsize=13, x=0.011, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(outpath, dpi=150, facecolor="white")
    print(f"wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser("Calibrate --white_thresh from the H&E itself")
    ap.add_argument("--he_dir", type=Path, required=True,
                    help="Directory of H&E RGB WSIs (originals or reconstructions).")
    ap.add_argument("--tiles_metadata", type=Path, default=None,
                    help="Dataset root with per-WSI tiles_metadata.csv, for the "
                         "region grid the histogram is pooled over. OPTIONAL: "
                         "without it the grid is sized from each image, which is "
                         "what the real SR arm wants — it is evaluated "
                         "whole-slide and has no tiling.")
    ap.add_argument("--outdir", type=Path, default=Path("white_thresh"))
    ap.add_argument("--n_wsis", type=int, default=3,
                    help="How many slides to sweep. Each is a full-slide fill per "
                         "threshold, so this is the cost knob. [%(default)s]")
    ap.add_argument("--t_min", type=float, default=0.50)
    ap.add_argument("--t_max", type=float, default=0.90)
    ap.add_argument("--t_step", type=float, default=0.025)
    ap.add_argument("--region_mm", type=float, default=1.5)
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP)
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--current", type=float, default=WHITE_THRESH,
                    help="Threshold to mark as 'where you are now'. [%(default)s]")
    args = ap.parse_args()

    thresholds = np.arange(args.t_min, args.t_max + 1e-9, args.t_step)
    he_index = {p.stem: p for p in sorted(args.he_dir.iterdir())
                if p.suffix.lower() in (".tif", ".tiff", ".png")}
    if not he_index:
        raise SystemExit(f"no images in {args.he_dir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    hist_total = np.zeros(args.bins, dtype=np.int64)
    edges = np.linspace(0, 1, args.bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    done = 0

    # The grid only sizes the histogram pool, so the extent can come from the
    # image itself when the arm was never tiled.
    if args.tiles_metadata is not None:
        sources = [(Path(wsi_extent(c)[0]).stem, c)
                   for c in iter_metadata_csvs(args.tiles_metadata)]
    else:
        sources = [(stem, None) for stem in sorted(he_index)]

    for stem, csv_path in sources:
        if done >= args.n_wsis:
            break
        if stem not in he_index:
            print(f"[skip] no H&E for {stem}")
            continue

        print(f"[{done + 1}/{args.n_wsis}] {stem}")
        mn = channel_min(he_index[stem])
        print(f"        {mn.shape[0]}x{mn.shape[1]}, "
              f"{len(thresholds)} thresholds")

        regions = (region_grid(csv_path, region_mm=args.region_mm, mpp=args.mpp)
                   if csv_path is not None else
                   region_grid_from_extent(stem, mn.shape[0], mn.shape[1],
                                           region_mm=args.region_mm, mpp=args.mpp))
        hist_total += tissue_histogram(mn, regions, args.bins)

        for row in sweep_slide(mn, thresholds):
            row["wsi"] = stem
            rows.append(row)
            print(f"        t={row['white_thresh']:.3f}  "
                  f"lumen={row['lumen_fraction']:.5f}  "
                  f"tissue={row['tissue_fraction']:.4f}")
        del mn
        done += 1

    if not rows:
        raise SystemExit(
            "no slides processed — with --tiles_metadata the image stems must "
            "match its source_file entries; without it, check --he_dir has TIFs")

    df = pd.DataFrame(rows)[
        ["wsi", "white_thresh", "lumen_fraction", "tissue_fraction", "n_tissue_px"]
    ]
    csv_path = args.outdir / "white_thresh.csv"
    df.to_csv(csv_path, index=False)

    suggestion = suggest_threshold(hist_total, centres)
    mean_curve = df.groupby("white_thresh")[["lumen_fraction", "tissue_fraction"]].mean()
    plateau = find_plateau(mean_curve.index.to_numpy(),
                           mean_curve["lumen_fraction"].to_numpy(),
                           mean_curve["tissue_fraction"].to_numpy())
    make_figure(df, hist_total, centres, suggestion, plateau, args.current,
                args.outdir / "white_thresh.png")

    payload = {
        "suggested": suggestion,
        "plateau": {k: v for k, v in plateau.items() if k != "sensitivity"},
        "sensitivity": plateau["sensitivity"],
        "current": args.current,
        "n_wsis": int(df["wsi"].nunique()),
        "thresholds": [float(t) for t in thresholds],
        "histogram": {"centres": centres.tolist(), "counts": hist_total.tolist()},
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
    }
    with open(args.outdir / "white_thresh.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== --white_thresh ===")
    if suggestion:
        print(f"suggested : {suggestion['threshold']:.3f}  "
              f"(valley between modes at {suggestion['tissue_mode']:.3f} "
              f"and {suggestion['whitespace_mode']:.3f})")
        print(f"            valley depth {suggestion['valley_depth_ratio']:.3f} "
              f"of the taller mode — the smaller, the cleaner the separation")
    else:
        print("no valley found: the tissue and whitespace modes are not separated "
              "in this histogram, so no threshold here is stable. Read panel A "
              "before choosing one by hand.")
    if plateau.get("breakdown") is not None:
        print(f"BREAKDOWN : {plateau['breakdown']:.3f} — at and above this the "
              f"footprint has absorbed the slide background.")
        print(f"            Usable range is <= {plateau['valid_hi']:.3f}. A value "
              f"above it measures a different object, not a smaller lumen.")
    if plateau.get("lo") is not None:
        inside = plateau["lo"] <= args.current <= plateau["hi"]
        print(f"plateau   : {plateau['lo']:.3f} to {plateau['hi']:.3f}  "
              f"(lumen_fraction stops depending on the cut here)")
        print(f"current   : {args.current:g}  "
              f"{'INSIDE the plateau' if inside else 'OUTSIDE the plateau'}")
    else:
        print(f"plateau   : NONE. {plateau.get('reason')}")
        print(f"current   : {args.current:g}"
              + ("  — ABOVE the usable range" if plateau.get("valid_hi") is not None
                 and args.current > plateau["valid_hi"] else ""))
    print(f"\nwrote {csv_path}")
    print(f"wrote {args.outdir / 'white_thresh.json'}")


if __name__ == "__main__":
    main()
