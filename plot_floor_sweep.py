#!/usr/bin/env python
"""Pool several `estimate_floor.py` runs into one region-size figure.

The §7 go/no-go is decided per region size, and region size is the one knob that
moves it: the floor averages out as regions grow while the biology does not
(§4.2). One run answers "is there headroom at 1.5 mm"; a sweep answers "is there
headroom at any scale this cohort can support", which is the question that
closes or opens the bias branch.

    python plot_floor_sweep.py \\
        --runs ./floor_liver_075 ./floor_liver_150 ./floor_liver_250 \\
        --outdir ./floor_sweep/

Region size is read from each run's `floor.json` params, so the runs can be
given in any order.

Outputs
-------
floor_sweep.png   ratio vs region size, over the verdict bands, plus the
                  conditioning of the variogram each ratio rests on
floor_sweep.csv   the same numbers, tidy, one row per run x descriptor

Why the second panel exists
---------------------------
Ratios improve with region size, but so does the *fragility* of the estimate
behind them: fewer regions, fewer within-slide pairs, and a narrower span of
separations in which to observe a sill. Past a point the apparent improvement is
the floor estimate degrading rather than the floor shrinking, and an
under-estimated floor flatters the ratio. The panels have to be read together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Categorical slots 1-4 for the descriptors; status steps for the verdict bands.
C_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
C_GOOD, C_WARN, C_CRIT = "#0ca30c", "#fab219", "#d03b3b"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e3e3df"


def load_runs(dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        csv, js = d / "floor_per_descriptor.csv", d / "floor.json"
        if not csv.is_file() or not js.is_file():
            raise SystemExit(f"{d} is not an estimate_floor output directory")
        payload = json.load(open(js))
        curve = payload.get("variogram_curve") or {}
        lags = curve.get("lag_mm") or []
        vg = (payload.get("estimates") or {}).get("variogram_sill") or {}
        df = pd.read_csv(csv)
        for r in df.itertuples():
            rows.append({
                "run": d.name,
                "region_mm": payload["params"]["region_mm"],
                "descriptor": r.descriptor,
                "floor_sd": r.floor_sd_used,
                "between_region_sd": r.between_region_sd,
                "floor_to_signal": r.floor_to_signal,
                "floor_source": r.floor_source,
                "verdict": r.verdict,
                "sill_reached": curve.get("sill_reached", {}).get(r.descriptor),
                "n_regions": payload["n_regions"],
                "n_variogram_pairs": vg.get("n_samples"),
                "lag_min_mm": min(lags) if lags else np.nan,
                "lag_max_mm": max(lags) if lags else np.nan,
            })
    out = pd.DataFrame(rows).sort_values(["region_mm", "descriptor"])
    if out["region_mm"].nunique() < 2:
        raise SystemExit("need runs at two or more region sizes to sweep")
    return out


def make_figure(t: pd.DataFrame, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = t.dropna(subset=["floor_to_signal"])
    names = list(dict.fromkeys(scored["descriptor"]))
    sizes = sorted(t["region_mm"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)

    # --- A: does any region size clear the floor? ---------------------------
    ax = axes[0]
    ymax = max(1.35, scored["floor_to_signal"].max() * 1.1)
    ax.axhspan(0, 0.5, color=C_GOOD, alpha=0.10, zorder=1, linewidth=0)
    ax.axhspan(0.5, 0.9, color=C_WARN, alpha=0.13, zorder=1, linewidth=0)
    ax.axhspan(0.9, ymax, color=C_CRIT, alpha=0.10, zorder=1, linewidth=0)

    pad = (max(sizes) - min(sizes)) * 0.06
    for i, name in enumerate(names):
        g = scored[scored["descriptor"] == name].sort_values("region_mm")
        colour = C_SERIES[i % len(C_SERIES)]
        ax.plot(g["region_mm"], g["floor_to_signal"], color=colour, linewidth=2,
                marker="o", markersize=7, zorder=5, label=name)
        last = g.iloc[-1]
        ax.text(last["region_mm"] + pad * 0.4, last["floor_to_signal"], name,
                color=colour, fontsize=8.5, va="center", ha="left", zorder=6)

    ax.set_xlim(min(sizes) - pad, max(sizes) + pad * 5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}" for s in sizes])
    ax.set_xlabel("region size (mm)", color=C_MUTED, fontsize=10)
    ax.set_ylabel("floor SD / between-region SD", color=C_MUTED, fontsize=10)
    ax.set_title("A · Does any region size clear the floor?",
                 color=C_INK, fontsize=11, loc="left", pad=10)
    for y, label, colour in ((0.25, "usable", C_GOOD),
                             (0.70, "marginal", "#a06a00"),
                             ((0.9 + ymax) / 2, "floor-limited", C_CRIT)):
        ax.text(min(sizes) - pad * 0.75, y, label, color=colour, fontsize=9,
                va="center", ha="left")
    # lower right: the lines run top-left to bottom-right, and the band labels
    # own the left edge
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for txt in leg.get_texts():
        txt.set_color(C_MUTED)

    # --- B: what each ratio rests on ----------------------------------------
    ax = axes[1]
    per_run = t.drop_duplicates("region_mm").sort_values("region_mm")
    ys = np.arange(len(per_run))[::-1]
    for y, r in zip(ys, per_run.itertuples()):
        if not np.isfinite(r.lag_min_mm):
            continue
        span = r.lag_max_mm / r.lag_min_mm
        ax.plot([r.lag_min_mm, r.lag_max_mm], [y, y], color=C_SERIES[0],
                linewidth=7, solid_capstyle="butt", alpha=0.85, zorder=4)
        ax.text(r.lag_max_mm + 0.12, y,
                f"{span:.1f}x span · {int(r.n_variogram_pairs):,} pairs",
                va="center", ha="left", fontsize=8.5, color=C_MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.region_mm:g} mm\n{int(r.n_regions)} regions"
                        for r in per_run.itertuples()],
                       fontsize=9.5, color=C_INK)
    ax.set_ylim(-0.7, len(per_run) - 0.3)
    ax.set_xlim(0, t["lag_max_mm"].max() * 1.75)
    ax.set_xlabel("separations the variogram actually saw (mm)",
                  color=C_MUTED, fontsize=10)
    ax.set_title("B · How much evidence is each ratio resting on?",
                 color=C_INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("Biological floor against region size",
                 color=C_INK, fontsize=13, x=0.011, ha="left", y=0.99)
    fig.text(0.011, 0.045,
             "Read together. The ratio improves with region size because the floor averages out "
             "faster than the biology does — but the sill it rests on is measured over a narrower",
             color=C_MUTED, fontsize=8.5)
    fig.text(0.011, 0.017,
             "span of separations, with fewer pairs. Where the span is short, a flat variogram is "
             "the absence of evidence rather than evidence of a plateau, and an under-estimated "
             "floor flatters the ratio.",
             color=C_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(outpath, dpi=150, facecolor="white")
    print(f"wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser("Pool estimate_floor runs across region sizes")
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="estimate_floor output directories, any order.")
    ap.add_argument("--outdir", type=Path, default=Path("floor_sweep"))
    args = ap.parse_args()

    t = load_runs(args.runs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv = args.outdir / "floor_sweep.csv"
    t.to_csv(csv, index=False)
    make_figure(t, args.outdir / "floor_sweep.png")

    print("\n=== floor vs region size ===")
    scored = t.dropna(subset=["floor_to_signal"])
    for name, g in scored.groupby("descriptor", sort=False):
        g = g.sort_values("region_mm")
        trail = "  ".join(f"{r.region_mm:g}mm {r.floor_to_signal:.2f}"
                          for r in g.itertuples())
        print(f"  {name:22s} {trail}   -> {g.iloc[-1]['verdict']}")

    usable = scored[scored["floor_to_signal"] < 0.5]
    if usable.empty:
        print("\nNo descriptor reaches 'usable' (<0.5) at any region size swept.")
        print("The bias branch has no headroom on this cohort; a second real PSR")
        print("level (--psr_level_b) is the only estimator that would supersede")
        print("the variogram and could change that.")
    else:
        print("\nusable:")
        for r in usable.itertuples():
            print(f"  {r.descriptor} at {r.region_mm:g} mm ({r.floor_to_signal:.2f})")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
