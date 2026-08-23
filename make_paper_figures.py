#!/usr/bin/env python
"""The five figures the WACV manuscript asks for, at column width.

Reads finished runs — it re-plots, it never re-measures — and writes PDFs sized
for a 3.281 in column so nothing is rescaled after export.

    python make_paper_figures.py \\
        --sources   ./compare_sources \\
        --pixel     ./reliability \\
        --fold      ./calibration_phi_fold \\
        --outdir    ~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/figures/

Writes the five files the manuscript's floats already reference
(PAPER_FIGURES.md §1, revised 2026-08-22):

    reliability_pixel.pdf       MAIN Fig 1 — pixel sigma vs the RESIDUAL, the
                                conventional protocol
    reliability_sources.pdf     MAIN Fig 2 — the three ensemble components vs e,
                                the proposed protocol, 0.80 line
    within_slide_rho.pdf        SUPP Fig 1 — per-slide rank correlation, raw and
                                partialled on mu, paired (§2c)
    risk_coverage.pdf           SUPP Fig 2 — mean e vs coverage, with the
                                mu-alone baseline
    data_exposure_share.pdf     SUPP Fig 3 — how much of the region-level CPA
                                variance is data exposure (§2d)

`within_slide_rho_pixel.pdf` is the same rank view at pixel scale. The
manuscript does not ask for it, so it is behind `--pixel_rank` rather than
written into a directory whose contents are meant to match the float list.

The cycle-reconstruction residual is not an uncertainty source in any of them.
It is one model's error magnitude rather than a dispersion across members, so it
belongs on the target side, which is what reliability_pixel.pdf shows.

Each reliability figure carries only sources whose sigma shares units with its
error, because the calibrated line means nothing otherwise — which is why the
pixel panel and the CPA panel are two files rather than two curves, and why
their reference lines differ by sqrt(3).

`within_slide_rho.pdf` shows both readings of the within-slide statistic, raw
and partialled on mu. The gap between a pair is the part of the association that
is collagen density rather than error, and the manuscript's caption describes
that gap, so the partialled bar alone is not enough.

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
from matplotlib.patches import Patch

from calibrate_phi import HALF_NORMAL, HALF_NORMAL_PIXEL, risk_coverage, score

COLUMN_IN = 3.281          # WACV two-column text width, one column
DESCRIPTOR = "task_specific_value"

# Order, label, colour, line style, marker. Style and marker carry the identity
# in greyscale; colour is redundant encoding, not the only encoding.
# Labels follow the manuscript's symbol register (PAPER_FIGURES.md 2a): sigma,
# e, mu, z, rho — never "MAE", "std" or "|error|".
#
# The cycle-reconstruction residual is NOT among them (2b, 2026-08-21). It is a
# single model's error magnitude, not a dispersion across members, so scoring it
# in the same column as sigma compared two different kinds of object — the empty
# scale column it always carried was the symptom. It survives in the paper only
# as the TARGET the conventional protocol validates against, which is what
# reliability_pixel.pdf shows.
SOURCES = [
    ("total",          "total $\\sigma$",         "#2a78d6", "-",  "o"),
    ("data_exposure",  "data-exposure $\\sigma$", "#1baf7a", "--", "^"),
    ("procedural",     "procedural $\\sigma$",    "#eb6834", "-.", "s"),
]
PIXEL_SOURCES = SOURCES

# The baseline the paper's own text volunteers: rank by the point prediction,
# no uncertainty involved. On this cohort it BEATS sigma, so a risk-coverage
# figure without it says the opposite of what the manuscript says.
POINT = ("point_prediction", "$\\mu$ alone", "#4a3aa7",
         (0, (3, 1, 1, 1, 1, 1)), "D")

# The two readings of the within-slide statistic, drawn as a PAIR of bars per
# component (2c, 2026-08-22). The gap between them is the part of the apparent
# ranking ability that is collagen density rather than error, which is the
# paper's answer to its strongest objection — so it has to be visible as a gap,
# not asserted in prose. Hatched vs solid, so the pair survives greyscale.
READINGS = [
    ("per_slide_raw",        "raw",                  "white", "////"),
    ("per_slide_partial_mu", "partialled on $\\mu$", None,    None),
]

INK = "#0b0b0b"
GREY = "#52514e"

# Sources with no meaningful E|z|: their sigma does not share units with the
# error, so the ratio would be a number in mixed units. Rank still works.
NO_SCALE = {"regen_error", POINT[0]}


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
        "hatch.linewidth": 0.45,   # the raw bars; heavier reads as a fill
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
                    height: float = 2.45, width: float = COLUMN_IN) -> None:
    """Reliability, one curve per source, in raw units.

    Every source here has a sigma in the same units as `e`, which is what makes
    the E|e| = half_normal * sigma line meaningful and E|z| interpretable. The
    cycle-reconstruction residual is not one of them and no longer appears as a
    source at all (PAPER_FIGURES.md 2b).

    **No per-curve annotations, on either figure** (2e, 2026-08-23). rho, E|z|
    and the ECE moved into the manuscript's Table 2, which puts both protocols
    side by side. Annotating them here would crowd the panel and, worse, put the
    two arms' summary statistics in matching corners of adjacent figures — which
    invites the reader to compare numbers the paper says are not comparable,
    since the two arms are normalised on different lines. `report_numbers`
    prints them instead, each with its own divisor written out.
    """
    by = {r["component"]: r for r in rows if r.get("bins")}
    fig, ax = plt.subplots(figsize=(width, height))
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
                    borderaxespad=0.2, labelspacing=0.22, handletextpad=0.4)
    for txt in leg.get_texts():
        txt.set_color("#0b0b0b")
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_per_slide_rho(per_slide: Dict[str, dict], sources, outpath: Path,
                      ylabel: str = "within-slide $\\rho$",
                      height: float = 2.6, seed: int = 0) -> None:
    """Per component, the within-slide rank correlation BEFORE and AFTER
    partialling on the point prediction.

    The figure the claim actually rests on, and since 2c it carries both
    readings rather than the partialled one alone. Each dot is a slide, so the
    reader sees n and the spread rather than a summary; the bar is the mean over
    slides, the whisker its slide-clustered 95% interval, and the count is how
    many slides came out positive.

    **The gap between the two bars of a pair is the result.** Sigma tracks how
    much structure a region holds \u2014 rho(sigma, mu_CPA) = +0.76 on this cohort \u2014
    and absolute error grows with the same thing, so a raw rho is largely the
    two sharing that dependence. What the pair shows is how much survives the
    control: about half, and the surviving component is data exposure, which
    only a crossed grid can measure. Asserting that in prose was the reason the
    manuscript asked for this rerun.

    Deliberately NOT a pooled reliability curve for all sources. Pooled bins mix
    within- and between-slide variation: cycle error's pooled curve rises while
    its within-slide answer is negative, so a pooled panel would flatter the one
    source the paper argues against. This shows the controlled statistic.

    It also solves the unit problem. Sigma in CPA and sigma in intensity units
    cannot share an axis, but a rank correlation is dimensionless, so every
    source belongs on this one.

    `per_slide[component]` is `{"per_slide_raw": [...],
    "per_slide_partial_mu": [...]}` \u2014 the two lists `within_slide()` writes into
    `summary.json`, one value per slide.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    rng = np.random.default_rng(seed)
    present = [s for s in sources if s[0] in per_slide]
    half, bw = 0.205, 0.36
    tops, bots, counts = [], [], []
    for i, (key, label, colour, ls, mk) in enumerate(present):
        for j, (reading, rlabel, face, hatch) in enumerate(READINGS):
            v = np.asarray(per_slide[key].get(reading, []), float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            x = i + (j * 2 - 1) * half
            ax.bar(x, v.mean(), width=bw, zorder=4,
                   facecolor=face if face else colour,
                   edgecolor=colour, linewidth=0.7, hatch=hatch)
            # resample the SLIDES: twenty values, one per case
            bs = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.scatter(x + (rng.random(v.size) - 0.5) * (bw * 0.78), v,
                       s=3.6, color=INK, alpha=0.38, linewidths=0, zorder=5)
            ax.errorbar([x], [v.mean()],
                        yerr=[[v.mean() - lo], [hi - v.mean()]],
                        color=INK, capsize=1.8, elinewidth=0.9, zorder=6,
                        linestyle="none")
            counts.append((x, f"{int((v > 0).sum())}/{v.size}"))
            tops.append(max(hi, v.max()))
            bots.append(min(lo, v.min()))
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([s[1].replace("-", "-\n") if len(s[1]) > 11 else s[1]
                        for s in present])
    ax.set_xlim(-0.55, len(present) - 0.45)
    if tops:
        # One row of counts rather than each perched on its own bar: the height
        # of a count would otherwise read as a quantity, which it is not.
        lo_y = min(min(bots), 0.0)
        span = max(tops) - lo_y
        for x, txt in counts:
            ax.annotate(txt, (x, max(tops) + 0.05 * span), ha="center",
                        va="bottom", color=INK,
                        fontsize=plt.rcParams["font.size"] - 1.4)
        ax.set_ylim(lo_y - 0.04 * span, max(tops) + 0.20 * span)
    ax.set_ylabel(ylabel, color=GREY)
    tidy(ax)
    ax.grid(False, axis="x")
    # Neutral swatches: the legend distinguishes the two READINGS, and giving
    # them one component's colour would read as naming a component.
    handles = [Patch(facecolor=f if f else "#8f8f8a", edgecolor="#5f5f5b",
                     linewidth=0.7, hatch=h, label=lab)
               for _, lab, f, h in READINGS]
    leg = ax.legend(handles=handles, frameon=False, loc="lower center",
                    ncol=len(READINGS), bbox_to_anchor=(0.5, 1.0),
                    handlelength=1.5, columnspacing=1.4, borderaxespad=0.1,
                    handletextpad=0.4)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


def fig_data_exposure_share(t: pd.DataFrame, outpath: Path,
                            height: float = 2.05, n_bins: int = 36) -> int:
    """How much of a region's CPA variance is data exposure rather than seed?

    The decomposition's own readout, and until 2d it was a pair of numbers in
    the text with no plot behind them. Per region, sigma_data^2 / sigma_total^2
    \u2014 the two are exactly additive in variance, so this is a share in [0, 1] and
    a median near 0.5 says the two components are of the same size.

    Regions where the ANOVA put the between-subset term at or below zero have no
    data-exposure SD at all and drop out here. That is a real outcome near zero
    rather than a missing measurement, so the count is returned and printed
    rather than silently filled in as a share of 0, which would drag the median
    down with regions that have no estimate.
    """
    p = t.pivot_table(index=["wsi", "region_index"], columns="component",
                      values="sd")
    need = {"total", "data_exposure"}
    if not need <= set(p.columns):
        raise SystemExit(f"per_region_sources.csv lacks {sorted(need - set(p.columns))}")
    n_all = len(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (p["data_exposure"] ** 2 / p["total"] ** 2)
    share = share.replace([np.inf, -np.inf], np.nan).dropna()
    if share.empty:
        raise SystemExit("no region had both a total and a data-exposure sigma")
    med = float(share.median())
    q1, q3 = (float(share.quantile(0.25)), float(share.quantile(0.75)))

    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    ax.hist(share, bins=np.linspace(0, 1, n_bins + 1), color="#1baf7a",
            alpha=0.75, edgecolor="white", linewidth=0.3, zorder=4)
    ax.axvspan(q1, q3, color="#1baf7a", alpha=0.28, linewidth=0, zorder=3)
    ax.axvline(med, color=INK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=6)
    # Top left, which the distribution leaves empty: a share below ~0.3 is rare
    # and the box would otherwise sit over the mode it is describing.
    ax.text(0.025, 0.97, f"median {med:.3f}\nIQR {q1:.2f}\u2013{q3:.2f}",
            transform=ax.transAxes, ha="left", va="top", color=INK,
            fontsize=plt.rcParams["font.size"] - 0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("data-exposure share of $\\sigma^2$", color=GREY, labelpad=1.5)
    ax.set_ylabel("regions", color=GREY)
    tidy(ax)
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")
    print(f"  data-exposure share of CPA variance: median {med:.4f}, "
          f"IQR {q1:.3f}-{q3:.3f}, n = {len(share)} regions"
          + (f" ({n_all - len(share)} without a data-exposure \u03c3)"
             if n_all > len(share) else ""))
    return len(share)


def fig_risk_coverage(rc: List[dict], outpath: Path,
                      height: float = 2.55) -> None:
    df = pd.DataFrame(rc)
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height))
    ax.axhline(0, color="#52514e", linewidth=0.8, linestyle=(0, (4, 2)),
               zorder=3, label="random")
    # SOURCES no longer holds the residual (2b); mu alone, the oracle and the
    # flat random line stay, since those are baselines rather than uncertainties.
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


def report_numbers(rows: List[dict], label: str,
                   half_normal: float = HALF_NORMAL) -> None:
    """The Table 2 row for one arm, as text — the figures no longer carry it.

    `half_normal` is that arm's calibrated line and it is printed in the column
    header rather than assumed, because the two arms do not share it: pixel
    sigma sums three colour channels while the residual is a per-channel mean,
    so the line there is sqrt(2/pi)/sqrt(3) = 0.46 against 0.80 for CPA.
    Dividing the pixel arm by 0.80 would report a perfectly calibrated ensemble
    as 74% over-confident, so the divisor has to travel with the number.
    """
    div = f"E|z|/{half_normal:.2f}"
    print(f"\n--- {label}: rho, scale and ECE, per source ---")
    print(f"  calibrated line E|e| = {half_normal:.2f}σ — quote the ratio as "
          f"{div}, with the divisor")
    print(f"{'source':18s} {'n':>6s} {'rho':>8s} {'95% CI':>18s} "
          f"{'E|z|':>8s} {div:>10s} {'ECE':>8s}")
    for r in rows:
        if "spearman_rho" not in r:
            continue
        ci = (f"[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "n/a")
        # rho to three decimals wherever it is printed (2e): the manuscript's
        # intervals are quoted to three, and +0.22 beside +0.217 makes a reader
        # check whether two numbers that look different are different.
        #
        # A scale is only defined where sigma shares units with the error, so
        # the residual (0-255 against a CPA fraction) and mu (a prediction, not
        # a dispersion) get a dash rather than a number someone could transcribe
        # into the table. Their RANK against the error is comparable; nothing
        # else about them is.
        z, ratio = (("       -", "         -") if r["component"] in NO_SCALE
                    else (f"{r['mean_abs_z']:>8.2f}",
                          f"{r['calibration_ratio']:>10.2f}"))
        print(f"{r['component']:18s} {r['n']:>6d} {r['spearman_rho']:>+8.3f} "
              f"{ci:>18s} {z} {ratio} {r['ece_normalised']:>8.4f}")
    print("  ECE is min-max normalised on both axes, so the two arms' ECEs are "
          "at least\n  constructed alike. The ratio > 1 is over-confident: "
          "errors exceed the spread.")


def report_per_subset(fold_dir: Path) -> None:
    """The five per-subset rows the manuscript is still missing (§3).

    Text, not a figure: the per-subset panels are not in the paper. Each subset
    is scored on its own because pooling the five enters every region five times
    against one shared target, and because the subsets sit at different sigma
    AND different error levels, which induces a between-subset trend present in
    none of them. Intervals rather than p-values, per §4 — the one place the
    resolved-by-omission contradiction could come back is here.
    """
    payload = json.load(open(fold_dir / "summary.json"))
    if payload.get("prediction") != "fold":
        raise SystemExit(f"{fold_dir}/summary.json is a "
                         f"'{payload.get('prediction')}' run, not --prediction fold")
    rows = [r for r in payload["per_descriptor"]
            if r["descriptor"] == DESCRIPTOR
            and r.get("component") == "procedural_within_fold"]
    if not rows:
        raise SystemExit(f"no {DESCRIPTOR} subset rows in {fold_dir}/summary.json")
    print(f"\n--- per subset, CPA, --prediction fold ({fold_dir}) ---")
    print("Each subset's own mean paired with its own procedural spread. Never "
          "pooled.")
    print(f"{'subset':>8s} {'n':>6s} {'rho':>8s} {'95% CI (by case)':>20s} "
          f"{'E|z|/0.80':>10s} {'shuf':>6s}")
    for r in rows:
        ci = (f"[{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "n/a")
        shuf = (f"{r['rho_shuffled']:.3f}"
                if np.isfinite(r.get("rho_shuffled", np.nan)) else "-")
        print(f"{r['prediction']:>8s} {r['n']:>6d} {r['spearman_rho']:>+8.3f} "
              f"{ci:>20s} {r['calibration_ratio']:>10.2f} {shuf:>6s}")
    for a in payload.get("fold_agreement", []):
        if a["descriptor"] != DESCRIPTOR:
            continue
        print(f"  agreement: median {a['rho_median']:+.3f}, range "
              f"{a['rho_min']:+.3f} to {a['rho_max']:+.3f} "
              f"({a['rho_range']:.3f} wide), sign consistent: "
              f"{'yes' if a['consistent_sign'] else 'NO'}")
    print("  The interval is a slide-clustered bootstrap. Report it rather than "
          "a p-value:\n  no p appears in either document, and a p from a "
          "different null beside a\n  percentile interval is what §4 is about.")


def main() -> None:
    ap = argparse.ArgumentParser("WACV figures, at column width")
    ap.add_argument("--sources", type=Path, required=True,
                    help="compare_uncertainty_sources.py output directory.")
    ap.add_argument("--pixel", type=Path, default=None,
                    help="plot_pixel_reliability.py output, for main Figure 1.")
    ap.add_argument("--fold", type=Path, default=None,
                    help="calibrate_phi.py --prediction fold output directory. "
                         "Prints the five per-subset rows §3 still owes the "
                         "manuscript; produces no figure, since the per-subset "
                         "panels are not in the paper.")
    ap.add_argument("--pixel_rank", action="store_true",
                    help="Also write within_slide_rho_pixel.pdf. Not requested "
                         "by the manuscript (§1), so off by default — --outdir "
                         "is the float list and an extra file there invites the "
                         "question of which figure it is.")
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

    # ---- The CPA-scale figures, from the region-level head-to-head ----
    raw = pd.read_csv(args.sources / "per_region_sources.csv")
    raw = raw[raw["descriptor"] == DESCRIPTOR].copy()
    if raw.empty:
        raise SystemExit(f"no {DESCRIPTOR} rows in {args.sources}")
    t = add_point_baseline(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins, args.n_boot, args.seed)
        rc = risk_coverage(t, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                           args.n_boot, args.seed)

    # Both readings per component (2c): the pair, not the partialled bar alone.
    ws = {r["component"]: r
          for r in json.load(open(args.sources / "summary.json"))["within_slide"]
          if r["descriptor"] == DESCRIPTOR}
    fig_reliability(rows, SOURCES, args.outdir / "reliability_sources.pdf",
                    HALF_NORMAL, "$\\sigma$  (CPA)", "$e$",
                    width=COLUMN_IN / 2 - 0.04 if args.half_width else COLUMN_IN)
    # The controlled view of the same comparison. See the note printed below.
    fig_per_slide_rho(ws, SOURCES, args.outdir / "within_slide_rho.pdf",
                      seed=args.seed)
    fig_risk_coverage(rc, args.outdir / "risk_coverage.pdf")
    # `raw`, not `t`: the point-prediction baseline is a source on the other
    # figures, but it is not a variance component and has no share of one.
    fig_data_exposure_share(raw, args.outdir / "data_exposure_share.pdf")
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

    # ---- Main Figure 1, the conventional protocol at pixel scale ----
    # Its E|z| and ECE are Table 2's four blank cells, and they have never
    # reached the manuscript — §7 currently describes this arm's magnitude
    # failure qualitatively because that is all a plot supports. They come out
    # of report_numbers below, normalised on 0.46 rather than 0.80 (2e).
    if args.pixel:
        tp = pd.read_csv(args.pixel / "per_tile_components.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rp = score(tp, args.n_bins, args.n_boot, args.seed,
                       half_normal=HALF_NORMAL_PIXEL)
        fig_reliability(rp, PIXEL_SOURCES,
                        args.outdir / "reliability_pixel.pdf",
                        HALF_NORMAL_PIXEL, "$\\sigma$  (0-255)",
                        "mean residual per tile")
        if args.pixel_rank:
            wp = {r["component"]: r for r in
                  json.load(open(args.pixel / "summary.json"))["within_slide"]}
            fig_per_slide_rho(wp, PIXEL_SOURCES,
                              args.outdir / "within_slide_rho_pixel.pdf",
                              ylabel="within-slide $\\rho$ vs residual",
                              seed=args.seed)
        report_numbers(rp, "pixel level (residual)", HALF_NORMAL_PIXEL)

    # ---- The numbers §3 still owes, as text ----
    if args.fold:
        report_per_subset(args.fold)

    print(f"\nAll figures are {COLUMN_IN} in wide — do not rescale in LaTeX.")
    print("Use \\includegraphics[width=\\columnwidth]{...} and nothing else.")


if __name__ == "__main__":
    main()
