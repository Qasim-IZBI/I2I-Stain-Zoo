#!/usr/bin/env python
"""The three figures the WACV manuscript asks for, at column width.

Reads finished runs — it re-plots, it never re-measures — and writes PDFs sized
for a 3.281 in column so nothing is rescaled after export.

    python make_paper_figures.py \\
        --sources   ./compare_sources \\
        --pixel     ./reliability \\
        --outdir    ~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/figures/

Writes five column-width PDFs:

    reliability_sources.pdf     the three ensemble components, x in CPA, 0.80 line
    reliability_residual.pdf    the cycle-recon. residual, x in intensity, no line
    within_slide_rho.pdf        per-slide rank correlation, ALL FOUR sources
    risk_coverage.pdf           mean e vs coverage, with the mu-alone baseline
    reliability_pixel.pdf       ensemble sigma vs the residual (supplementary)
    within_slide_rho_pixel.pdf  the same rank view at pixel scale (optional)

The first two are separate files rather than panels of one figure, because
their x axes are different quantities. They share a y RANGE, set from both
together, so heights stay comparable when they sit side by side. --half_width
exports them at half a column each for native-size subfigures.

The two reliability figures carry only sources whose sigma shares units with
their error, because the calibrated line means nothing otherwise. Cycle error
against CPA error is that case, so it appears in `within_slide_rho.pdf`, where
the statistic is a dimensionless rank.

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
# Labels follow the manuscript's symbol register (PAPER_FIGURES.md 2a): sigma,
# e, mu, z, rho — never "MAE", "std" or "|error|". The fourth source is the
# cycle-reconstruction RESIDUAL, which is the term Table 2 uses.
SOURCES = [
    ("total",          "total $\\sigma$",         "#2a78d6", "-",  "o"),
    ("data_exposure",  "data-exposure $\\sigma$", "#1baf7a", "--", "^"),
    ("procedural",     "procedural $\\sigma$",    "#eb6834", "-.", "s"),
    ("regen_error",    "cycle-recon. residual",  "#e34948", ":",  "X"),
]
PIXEL_SOURCES = [s for s in SOURCES if s[0] != "regen_error"]

# The baseline the paper's own text volunteers: rank by the point prediction,
# no uncertainty involved. On this cohort it BEATS sigma, so a risk-coverage
# figure without it says the opposite of what the manuscript says.
POINT = ("point_prediction", "$\\mu$ alone", "#4a3aa7",
         (0, (3, 1, 1, 1, 1, 1)), "D")


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


def fig_reliability_split(rows: List[dict], outdir: Path,
                          half_normal: float = HALF_NORMAL,
                          width: float = COLUMN_IN,
                          height: float = 2.45) -> None:
    """Two SEPARATE files, not two panels of one figure.

        reliability_sources.pdf   the three ensemble components, x in CPA,
                                  with the E|e| = 0.80 sigma line
        reliability_residual.pdf  the cycle-reconstruction residual, x in 0-255
                                  intensity, and NO reference line

    They are separate because the x units are not the same quantity, so no
    shared axis is honest. What IS shared is **the y range**, set from both
    together and applied to each: `e` is the same quantity for all four sources
    by construction, since only sigma moves between them, and a reader placing
    the two side by side must be able to compare heights directly. Letting each
    autoscale would silently rescale the comparison.

    The residual gets no scale reference and no E|z| annotation. Both would
    divide an intensity by a fraction.
    """
    by = {r["component"]: r for r in rows if r.get("bins")}
    ens = [s for s in SOURCES if s[0] != "regen_error"]

    # one y range for both files — the point of splitting is the x axis, not y
    ys = [d["mean_error"] + (d.get("se_error_by_case") or 0.0)
          for k, *_ in SOURCES if k in by for d in by[k]["bins"]]
    yhi = max(ys) * 1.28

    def panel(sources, outpath, xlabel, line: bool, annotate_scale: bool):
        fig, ax = plt.subplots(figsize=(width, height))
        xs = [d["mean_sd"] for k, *_ in sources if k in by for d in by[k]["bins"]]
        if not xs:
            raise SystemExit(f"nothing to plot for {outpath.name}")
        xhi = max(xs) * 1.10
        if line:
            ax.plot([0, xhi], [0, xhi * half_normal], color="#52514e",
                    linewidth=0.9, linestyle=(0, (4, 2)), zorder=3,
                    label=f"$\\mathbb{{E}}|e| = {half_normal:.2f}\\sigma$")
        for i, (key, label, colour, ls, mk) in enumerate(sources):
            if key not in by:
                continue
            b, r = by[key]["bins"], by[key]
            ax.errorbar([d["mean_sd"] for d in b], [d["mean_error"] for d in b],
                        yerr=[d.get("se_error_by_case", np.nan) for d in b],
                        color=colour, linestyle=ls, marker=mk, capsize=1.3,
                        elinewidth=0.6, zorder=5, label=label,
                        markeredgecolor="white", markeredgewidth=0.4)
            txt = f"$\\rho$ {r['spearman_rho']:+.2f}"
            if annotate_scale:
                txt += f"   $\\mathbb{{E}}|z|/0.80$ {r['calibration_ratio']:.2f}"
            ax.text(0.035, 0.97 - 0.082 * i, txt, transform=ax.transAxes,
                    color=colour, va="top",
                    fontsize=plt.rcParams["font.size"] - 1.2)
        ax.set_xlim(0, xhi)
        ax.set_ylim(0, yhi)
        ax.set_xlabel(xlabel, color="#52514e", labelpad=1.5)
        ax.set_ylabel("$e$", color="#52514e")
        tidy(ax)
        leg = ax.legend(frameon=False, loc="lower right", handlelength=2.0,
                        borderaxespad=0.2, labelspacing=0.22, handletextpad=0.4)
        for t_ in leg.get_texts():
            t_.set_color("#0b0b0b")
        fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print(f"wrote {outpath}")

    panel(ens, outdir / "reliability_sources.pdf", "$\\sigma$  (CPA)",
          line=True, annotate_scale=True)
    panel([SOURCES[3]], outdir / "reliability_residual.pdf",
          "cycle-recon. residual  (0-255)", line=False, annotate_scale=False)


def fig_reliability(rows: List[dict], sources, outpath: Path,
                    half_normal: float, xlabel: str, ylabel: str,
                    height: float = 2.45) -> None:
    """Single-panel reliability, for the supplementary pixel figure."""
    by = {r["component"]: r for r in rows if r.get("bins")}
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    xs = [d["mean_sd"] for k, *_ in sources if k in by for d in by[k]["bins"]]
    ys = [d["mean_error"] for k, *_ in sources if k in by for d in by[k]["bins"]]
    if not xs:
        raise SystemExit("no binned source to plot")
    xhi = max(xs) * 1.10
    yhi = max(max(ys) * 1.20, xhi * half_normal * 1.05)
    ax.plot([0, xhi], [0, xhi * half_normal], color="#52514e", linewidth=0.9,
            linestyle=(0, (4, 2)), zorder=3,
            label=f"$\\mathbb{{E}}|e| = {half_normal:.2f}\\sigma$")
    for key, label, colour, ls, mk in sources:
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
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_per_slide_rho(per_slide: Dict[str, list], sources, outpath: Path,
                      ylabel: str = "within-slide $\\rho$ (partialled on $\\mu$)",
                      height: float = 2.45, seed: int = 0) -> None:
    """One dot per slide: does this source rank the error inside that slide?

    The figure the claim actually rests on. Each dot is a slide, so the reader
    sees n and the spread rather than a summary; the large marker is the mean
    with its slide-clustered interval, and the count is how many slides came out
    positive.

    Deliberately NOT a pooled reliability curve for all sources. Pooled bins mix
    within- and between-slide variation: cycle error's pooled curve rises while
    its within-slide answer is negative, so a pooled panel would flatter the one
    source the paper argues against. This shows the controlled statistic.

    It also solves the unit problem. Sigma in CPA and sigma in intensity units
    cannot share an axis, but a rank correlation is dimensionless, so every
    source belongs on this one.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    ax.axhline(0, color="#52514e", linewidth=0.8, zorder=3)
    rng = np.random.default_rng(seed)
    present = [s for s in sources if s[0] in per_slide]
    for i, (key, label, colour, ls, mk) in enumerate(present):
        v = np.asarray(per_slide[key], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.scatter(np.full(v.size, i) + (rng.random(v.size) - 0.5) * 0.26, v,
                   s=5.5, color=colour, alpha=0.55, linewidths=0, zorder=4)
        bs = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.errorbar([i], [v.mean()], yerr=[[v.mean() - lo], [hi - v.mean()]],
                    color=colour, marker=mk, markersize=4.2, capsize=2.2,
                    elinewidth=1.1, zorder=6, markeredgecolor="white",
                    markeredgewidth=0.5, linestyle="none")
        ax.annotate(f"{int((v > 0).sum())}/{v.size}", (i, hi), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=colour,
                    fontsize=plt.rcParams["font.size"] - 1.0)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([s[1].replace(" \u03c3", "").replace("-", "-\n")
                        if len(s[1]) > 11 else s[1].replace(" \u03c3", "")
                        for s in present])
    ax.set_xlim(-0.55, len(present) - 0.45)
    ax.set_ylabel(ylabel, color="#52514e")
    tidy(ax)
    ax.grid(False, axis="x")
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_risk_coverage(rc: List[dict], outpath: Path,
                      height: float = 2.55) -> None:
    df = pd.DataFrame(rc)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    ax.axhline(0, color="#52514e", linewidth=0.8, linestyle=(0, (4, 2)),
               zorder=3, label="random")
    for key, label, colour, ls, mk in list(SOURCES) + [POINT]:
        g = df[df["component"] == key].sort_values("coverage")
        if g.empty:
            continue
        ax.plot(g["coverage"] * 100, g["rel_change"] * 100, color=colour,
                linestyle=ls, marker=mk, zorder=5, label=label,
                markeredgecolor="white", markeredgewidth=0.4)
    g = df[df["component"] == "total"].sort_values("coverage")
    ax.plot(g["coverage"] * 100, g["rel_change_oracle"] * 100, color="#0b0b0b",
            linestyle=(0, (1, 1.4)), linewidth=0.9, zorder=4,
            label="oracle (rank by $e$)")
    ax.invert_xaxis()
    ax.set_xlabel("coverage (% of regions kept)", color="#52514e")
    ax.set_ylabel("change in mean $e$ (%)", color="#52514e")
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
    ap.add_argument("--half_width", action="store_true",
                    help="Export the two reliability files at half a column "
                         "each, for placing side by side as subfigures at their "
                         "NATIVE size. Without it they are full column width, "
                         "which is right when they sit one above the other. "
                         "Scaling in LaTeX shrinks the type with the axes, so "
                         "the choice has to be made here.")
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
    # Two separate files, sharing a y range so they stay comparable.
    fig_reliability_split(rows, args.outdir,
                          width=COLUMN_IN / 2 - 0.04 if args.half_width
                          else COLUMN_IN)
    # The controlled view of the same comparison. See the note printed below.
    fig_per_slide_rho(ws, SOURCES, args.outdir / "within_slide_rho.pdf")
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
                        HALF_NORMAL_PIXEL, "$\\sigma$  (0-255)",
                        "mean residual per tile")
        fig_per_slide_rho(wp, PIXEL_SOURCES,
                          args.outdir / "within_slide_rho_pixel.pdf",
                          ylabel="within-slide $\\rho$ vs residual")
        report_numbers(rp, "pixel level (cycle error)")

    print(f"\nAll figures are {COLUMN_IN} in wide — do not rescale in LaTeX.")
    print("Use \\includegraphics[width=\\columnwidth]{...} and nothing else.")


if __name__ == "__main__":
    main()
