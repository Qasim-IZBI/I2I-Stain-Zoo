#!/usr/bin/env python
"""The three figures the WACV manuscript asks for, at column width.

Reads finished runs — it re-plots, it never re-measures — and writes PDFs sized
for a 3.281 in column so nothing is rescaled after export.

    python make_paper_figures.py \\
        --sources   ./compare_sources \\
        --pixel     ./reliability \\
        --outdir    ~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/figures/

Why this is a separate script from the exploratory plotters
-----------------------------------------------------------
`calibrate_phi.py` and friends draw every descriptor, bake in a title, size for
a screen and emit PNG. The paper wants one descriptor, no title, 7 pt type at
3.281 in, greyscale-separable curves and PDF. Bolting all of that onto the
working figures as flags would leave neither doing its job well, so the paper
figures re-plot from the CSVs the runs already wrote.

Three requirements that are not cosmetic
----------------------------------------
* **Nothing may exceed the column.** `wacv.sty` loads `lineno` with `switch`,
  putting line numbers in the outer margin; anything wider prints underneath
  them. An overfull box is a layout bug in this template, not a warning.
* **Type must be legible at 3.281 in.** Set figure size and font size together —
  `\\includegraphics[width=\\columnwidth]` on an oversized export shrinks the
  text with the axes.
* **Greyscale-separable.** Line style and marker vary with colour, because four
  sources on one panel is where this stops being theoretical.

The calibration line is 0.80σ, not the diagonal: for a symmetric error of scale
σ, E|e| = σ√(2/π), so a diagonal would call a perfectly calibrated ensemble 20%
over-confident.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibrate_phi import HALF_NORMAL, HALF_NORMAL_PIXEL, risk_coverage, score

COLUMN_IN = 3.281          # WACV two-column text width, one column
DESCRIPTOR = "task_specific_value"

# Order, label, colour, line style, marker. Style and marker carry the identity
# in greyscale; colour is redundant encoding, not the only encoding.
SOURCES = [
    ("total",          "total σ",          "#2a78d6", "-",  "o"),
    ("data_exposure",  "data-exposure σ",  "#1baf7a", "--", "^"),
    ("procedural",     "procedural σ",     "#eb6834", "-.", "s"),
    ("regen_error",    "cycle error",      "#e34948", ":",  "X"),
]
PIXEL_SOURCES = [s for s in SOURCES if s[0] != "regen_error"]

# The baseline the paper's own text volunteers: rank by the point prediction,
# no uncertainty involved. On this cohort it BEATS sigma, so a risk-coverage
# figure without it says the opposite of what the manuscript says.
POINT = ("point_prediction", "predicted CPA (no uncertainty)", "#4a3aa7", (0, (3, 1, 1, 1, 1, 1)), "D")


def style(fontsize: float = 7.0) -> None:
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "legend.fontsize": fontsize - 0.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "pdf.fonttype": 42,        # TrueType, so the PDF is not bitmapped text
        "ps.fonttype": 42,
        "figure.dpi": 200,
    })


def tidy(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e3e3df", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", length=2.5, pad=1.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c4")


def add_point_baseline(t: pd.DataFrame) -> pd.DataFrame:
    """A source whose `sd` is the point prediction itself."""
    base = t[t["component"] == "total"].copy()
    base["sd"] = base["mu"]
    base["component"] = POINT[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        base["z"] = np.where(base["sd"] > 0, base["error"] / base["sd"], np.nan)
    return pd.concat([t, base], ignore_index=True)


def fig_reliability(rows: List[dict], sources, outpath: Path,
                    half_normal: float, xlabel: str, ylabel: str,
                    per_slide: Dict[str, list], height: float = 3.4,
                    absolute_only=None) -> None:
    """Reliability in two panels, because the units forbid one.

    (a) **absolute**, raw units, with the 0.80σ line. Only the ensemble sources
        can appear: σ and the error are both CPA there, which is what makes the
        line meaningful and what makes E|z| interpretable.
    (b) **the per-slide partial ρ**, all sources including cycle error. Cycle
        error is in 0–255 intensity units against a CPA fraction, so it shares
        no axis with the others — on a common linear x it crushes every ensemble
        curve into a dot at the origin, which is what "four sources, one panel"
        produced on the first attempt. What survives the unit mismatch is the
        RANK statistic, and (b) shows the twenty per-slide values behind it
        rather than a pooled curve. That choice matters: pooled bins mix within-
        and between-slide variation, and cycle error's pooled curve rises while
        its within-slide answer is −0.040.
    """
    by = {r["component"]: r for r in rows if r.get("bins")}
    abs_sources = absolute_only if absolute_only is not None else [
        s for s in sources if s[0] != "regen_error"]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(COLUMN_IN, height),
                                  gridspec_kw={"hspace": 0.62})

    # ---- (a) absolute, ensemble only ----
    xs = [d["mean_sd"] for k, *_ in abs_sources if k in by for d in by[k]["bins"]]
    ys = [d["mean_error"] for k, *_ in abs_sources if k in by for d in by[k]["bins"]]
    if not xs:
        raise SystemExit("no binned source to plot")
    xhi = max(xs) * 1.10
    yhi = max(max(ys) * 1.20, xhi * half_normal * 1.05)
    ax.plot([0, xhi], [0, xhi * half_normal], color="#52514e", linewidth=0.9,
            linestyle=(0, (4, 2)), zorder=3,
            label=f"calibrated ({half_normal:.2f}$\\sigma$)")
    for key, label, colour, ls, mk in abs_sources:
        if key not in by:
            continue
        b = by[key]["bins"]
        ax.errorbar([d["mean_sd"] for d in b], [d["mean_error"] for d in b],
                    yerr=[d.get("se_error_by_case", np.nan) for d in b],
                    color=colour, linestyle=ls, marker=mk, capsize=1.4,
                    elinewidth=0.7, zorder=5, label=label,
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlim(0, xhi)
    ax.set_ylim(0, yhi)
    ax.set_xlabel(xlabel, color="#52514e", labelpad=1.5)
    ax.set_ylabel(ylabel, color="#52514e")
    tidy(ax)
    leg = ax.legend(frameon=False, loc="upper left", handlelength=2.2,
                    borderaxespad=0.15, labelspacing=0.22, handletextpad=0.4)
    for txt in leg.get_texts():
        txt.set_color("#0b0b0b")
    ax.text(-0.20, 1.04, "(a)", transform=ax.transAxes, fontweight="bold",
            va="bottom", fontsize=plt.rcParams["font.size"])

    # ---- (b) the per-slide partial rho, which is the controlled statistic ----
    # NOT a pooled reliability curve for all four. Pooled bins mix within- and
    # between-slide variation, and cycle error's apparent trend there comes
    # entirely from the latter — the pooled curve rises 0.031 to 0.052 while the
    # within-slide answer is -0.040. This panel shows the twenty per-slide
    # values behind the claim, so the reader sees n and the spread rather than a
    # summary that flatters one source.
    axr.axhline(0, color="#52514e", linewidth=0.8, zorder=3)
    rng = np.random.default_rng(0)
    for i, (key, label, colour, ls, mk) in enumerate(sources):
        v = per_slide.get(key)
        if v is None:
            continue
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        jitter = (rng.random(v.size) - 0.5) * 0.26
        axr.scatter(np.full(v.size, i) + jitter, v, s=5.5, color=colour,
                    alpha=0.55, linewidths=0, zorder=4)
        bs = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        axr.errorbar([i], [v.mean()], yerr=[[v.mean() - lo], [hi - v.mean()]],
                     color=colour, marker=mk, markersize=4.2, capsize=2.2,
                     elinewidth=1.1, zorder=6, markeredgecolor="white",
                     markeredgewidth=0.5, linestyle="none")
        axr.text(i, hi + 0.045, f"{int((v > 0).sum())}/{v.size}", ha="center",
                 fontsize=plt.rcParams["font.size"] - 1.2, color=colour)
    axr.set_xticks(range(len(sources)))
    axr.set_xticklabels([s[1].replace(" \u03c3", "").replace(" ", "\n")
                         for s in sources])
    axr.set_xlim(-0.55, len(sources) - 0.45)
    axr.set_ylabel("within-slide $\\rho$\n(partialled on $\\mu$)", color="#52514e")
    tidy(axr)
    axr.grid(False, axis="x")
    axr.text(-0.20, 1.04, "(b)", transform=axr.transAxes, fontweight="bold",
             va="bottom", fontsize=plt.rcParams["font.size"])

    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_risk_coverage(rc: List[dict], outpath: Path,
                      height: float = 2.55) -> None:
    df = pd.DataFrame(rc)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    ax.axhline(0, color="#52514e", linewidth=0.8, linestyle=(0, (4, 2)),
               zorder=3, label="random / keep all")
    for key, label, colour, ls, mk in list(SOURCES) + [POINT]:
        g = df[df["component"] == key].sort_values("coverage")
        if g.empty:
            continue
        ax.plot(g["coverage"] * 100, g["rel_change"] * 100, color=colour,
                linestyle=ls, marker=mk, zorder=5, label=label,
                markeredgecolor="white", markeredgewidth=0.4)
    g = df[df["component"] == "total"].sort_values("coverage")
    ax.plot(g["coverage"] * 100, g["rel_change_oracle"] * 100, color="#0b0b0b",
            linestyle=(0, (1, 1.4)), linewidth=0.9, zorder=4, label="oracle")
    ax.invert_xaxis()
    ax.set_xlabel("coverage: regions kept (%)", color="#52514e")
    ax.set_ylabel("change in CPA MAE (%)", color="#52514e")
    tidy(ax)
    leg = ax.legend(frameon=False, loc="lower left", handlelength=2.4,
                    borderaxespad=0.2, labelspacing=0.24, handletextpad=0.5)
    for txt in leg.get_texts():
        txt.set_color("#0b0b0b")
    fig.tight_layout(pad=0.25)
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def report_numbers(rows: List[dict], label: str) -> None:
    print(f"\n--- {label}: normalised ECE and rho, per source ---")
    print(f"{'source':18s} {'n':>6s} {'rho':>8s} {'95% CI':>18s} {'ECE':>8s}")
    for r in rows:
        if "spearman_rho" not in r:
            continue
        ci = (f"[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "n/a")
        print(f"{r['component']:18s} {r['n']:>6d} {r['spearman_rho']:>+8.3f} "
              f"{ci:>18s} {r['ece_normalised']:>8.4f}")
    print("  ECE is min-max normalised on both axes, so it is comparable across")
    print("  sources here. Do NOT report E|z| for cycle error: it is in 0-255")
    print("  intensity units while CPA error is a fraction.")


def main() -> None:
    ap = argparse.ArgumentParser("WACV figures, at column width")
    ap.add_argument("--sources", type=Path, required=True,
                    help="compare_uncertainty_sources.py output directory.")
    ap.add_argument("--pixel", type=Path, default=None,
                    help="plot_pixel_reliability.py output, for figure S1.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Figures 1 and 2, from the region-level head-to-head ----
    t = pd.read_csv(args.sources / "per_region_sources.csv")
    t = t[t["descriptor"] == DESCRIPTOR].copy()
    if t.empty:
        raise SystemExit(f"no {DESCRIPTOR} rows in {args.sources}")
    t = add_point_baseline(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins, args.n_boot, args.seed)
        rc = risk_coverage(t, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                           args.n_boot, args.seed)

    ws = {r["component"]: r["per_slide_partial_mu"]
          for r in json.load(open(args.sources / "summary.json"))["within_slide"]
          if r["descriptor"] == DESCRIPTOR}
    fig_reliability(rows, SOURCES, args.outdir / "reliability_sources.pdf",
                    HALF_NORMAL, "ensemble $\\sigma$ (CPA)",
                    "|error| vs real SR (CPA)", per_slide=ws)
    fig_risk_coverage(rc, args.outdir / "risk_coverage.pdf")
    report_numbers(rows, "region level (CPA)")

    # the baseline the figure exists to show
    df = pd.DataFrame(rc)
    print("\n--- risk-coverage at 80% coverage ---")
    for key, label, *_ in list(SOURCES) + [POINT]:
        g = df[(df["component"] == key) & (np.isclose(df["coverage"], 0.8))]
        if g.empty:
            continue
        r = g.iloc[0]
        ci = (f"[{r['rel_ci_lo']:+.1%},{r['rel_ci_hi']:+.1%}]"
              if "rel_ci_lo" in r and np.isfinite(r["rel_ci_lo"]) else "")
        print(f"  {label:32s} {r['rel_change']:>+7.1%}  {ci}")

    # ---- Figure S1, the pixel-scale replication ----
    if args.pixel:
        tp = pd.read_csv(args.pixel / "per_tile_components.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = score(tp, args.n_bins, args.n_boot, args.seed,
                       half_normal=HALF_NORMAL_PIXEL)
        wp = {r["component"]: r["per_slide_partial_mu"]
              for r in json.load(open(args.pixel / "summary.json"))["within_slide"]}
        fig_reliability(rp, PIXEL_SOURCES,
                        args.outdir / "reliability_pixel.pdf",
                        HALF_NORMAL_PIXEL, "ensemble $\\sigma$ (intensity)",
                        "mean |cycle error| per tile", per_slide=wp)
        report_numbers(rp, "pixel level (cycle error)")

    print(f"\nAll figures are {COLUMN_IN} in wide — do not rescale in LaTeX.")
    print("Use \\includegraphics[width=\\columnwidth]{...} and nothing else.")


if __name__ == "__main__":
    main()
