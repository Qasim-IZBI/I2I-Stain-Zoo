#!/usr/bin/env python
"""Does the ensemble's spread predict its structural error?

Ensemble variance measures disagreement between members, not error. The BMVC
2026 result is that cycle-reconstruction error — the cheap self-consistency
proxy — does not calibrate it. This asks the same question against an external,
task-relevant target: φ_struct of the generated stain versus φ_struct of the
**real** tissue, per descriptor, per region.

The two measurements are separate stages, and this is only the third:

    compute_phi_uncertainty.py   ensemble masks -> per_region.csv     (mu, sd)
    compute_phi_reference.py     real masks     -> reference_phi.csv  (real_*)
    calibrate_phi.py             both CSVs      -> does sd predict the error?

    python calibrate_phi.py \\
        --phi_csv       ./phi_uncertainty/per_region.csv \\
        --reference_csv ./calibration_phi/reference_phi.csv \\
        --outdir        ./calibration_phi/

So this reads no masks and takes seconds: changing `--prediction`, `--n_bins`,
`--n_boot` or anything about the figure never re-measures tissue. Which arms are
scored is decided when the reference is built — `--real_psr` gives the four
collagen terms, `--real_lumen` the three H&E-referenced ones.

Both sides carry their region boxes, so a reference is proved to belong to this
grid rather than trusted: same region ids on different boxes exits.

Outputs
-------
per_region_calibration.csv   mu, sd, reference, error and z per region x descriptor
reliability_bins.csv         the reliability diagram's own data, one row per bin
summary.json                 Spearman rho, E|z|, ECE, reliability bins, provenance
calibration_phi.png          working panel: reliability per descriptor + rho summary
reliability_phi.png          the reliability diagram alone, at figure quality
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uncertainty_calibration import expected_calibration_error, reliability_bins
from uncertainty_phi.descriptors import PHI_NAMES, PHI_REFERENCE
from uncertainty_phi.reference import load_reference

# For a Gaussian error of scale sigma, E|e| = sigma * sqrt(2/pi). A reliability
# line of slope 1 would therefore call a perfectly calibrated ensemble 20%
# over-confident.
HALF_NORMAL = float(np.sqrt(2.0 / np.pi))

COLLAGEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "psr"]
LUMEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "he"]

C_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948")
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e3e3df"


def pair(df: pd.DataFrame, ref: pd.DataFrame, mode: str, n_folds: int) -> pd.DataFrame:
    """Long table of (prediction, uncertainty, reference) per region x descriptor.

    `grand` pairs the mean over all members with the total spread — the
    deployed prediction. `fold` pairs each subset's mean with that subset's
    procedural spread alone, giving one row per subset and asking whether
    procedural uncertainty suffices without the data-exposure term.
    """
    merged = df.merge(ref, on=["wsi", "region_index"], how="inner")
    if merged.empty:
        raise SystemExit("no regions matched between --phi_csv and the reference")

    out = []
    sources = ([("grand", "mu_{n}", "sd_total_{n}")] if mode == "grand"
               else [(f"fold{f}", f"fold{f}_mu_{{n}}", f"fold{f}_sd_{{n}}")
                     for f in range(1, n_folds + 1)])

    for label, mu_key, sd_key in sources:
        for name in PHI_NAMES:
            mu_col, sd_col, real_col = (mu_key.format(n=name),
                                        sd_key.format(n=name), f"real_{name}")
            if not {mu_col, sd_col, real_col} <= set(merged.columns):
                continue
            block = merged[["wsi", "region_index", mu_col, sd_col, real_col]].copy()
            block.columns = ["wsi", "region_index", "mu", "sd", "real"]
            block["descriptor"] = name
            block["prediction"] = label
            out.append(block)

    if not out:
        raise SystemExit(
            f"no descriptor had all of mu/sd/reference for mode '{mode}'. "
            f"For 'fold', --phi_csv must come from a multi-fold run."
        )
    t = pd.concat(out, ignore_index=True)
    t["error"] = (t["mu"] - t["real"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["z"] = np.where(t["sd"] > 0, t["error"] / t["sd"], np.nan)
    return t


def cluster_bootstrap_rho(g: pd.DataFrame, n_boot: int, seed: int) -> tuple:
    """95% CI for Spearman rho, resampling WHOLE SLIDES rather than regions.

    Regions inside a slide are spatially correlated and share one case's biology,
    so 2850 regions are nowhere near 2850 independent observations and the naive
    p-value is meaningless — it was 1e-31 on a cohort of twenty cases. The unit
    of replication is the case, so that is what gets resampled.

    A slide drawn twice contributes its regions twice, which is the point: the
    interval then reflects how much the answer depends on which twenty slides
    were collected.
    """
    wsis = g["wsi"].unique()
    if len(wsis) < 3:
        return float("nan"), float("nan"), 0
    by_wsi = {w: sub[["sd", "error"]].to_numpy() for w, sub in g.groupby("wsi")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(wsis, size=len(wsis), replace=True)
        block = np.concatenate([by_wsi[w] for w in pick])
        if np.ptp(block[:, 0]) == 0 or np.ptp(block[:, 1]) == 0:
            continue
        r = spearmanr(block[:, 0], block[:, 1]).statistic
        if np.isfinite(r):
            draws.append(r)
    if len(draws) < max(20, n_boot // 10):
        return float("nan"), float("nan"), len(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), len(draws)


def shuffled_rho(g: pd.DataFrame, n_perm: int, seed: int) -> float:
    """Negative control: break the sigma-error pairing, keep both marginals.

    If rho survives this, it was never measuring a relationship between the two —
    it was an artefact of their distributions or of the region ordering. A
    calibrated result must collapse to ~0 here.
    """
    sd, err = g["sd"].to_numpy(), g["error"].to_numpy()
    if np.ptp(sd) == 0 or np.ptp(err) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = [spearmanr(rng.permutation(sd), err).statistic for _ in range(n_perm)]
    draws = [d for d in draws if np.isfinite(d)]
    return float(np.mean(np.abs(draws))) if draws else float("nan")


def score(t: pd.DataFrame, n_bins: int, n_boot: int = 0, seed: int = 0) -> List[dict]:
    """Per descriptor: does sd rank error, and is its scale right?"""
    rows = []
    for name, g in t.groupby("descriptor", sort=False):
        g = g.dropna(subset=["sd", "error"])
        g = g[np.isfinite(g["sd"]) & np.isfinite(g["error"]) & (g["sd"] > 0)]
        if len(g) < 3:
            rows.append({"descriptor": name, "n": int(len(g)),
                         "note": "too few finite regions to score"})
            continue

        sd, err = g["sd"].to_numpy(), g["error"].to_numpy()
        rho, p = spearmanr(sd, err)

        # Absolute reliability: sd and error share units here, unlike the pixel
        # case, so the bins stay in raw units and the calibrated line is
        # E|e| = sd * sqrt(2/pi) rather than a normalised diagonal.
        edges = np.quantile(sd, np.linspace(0, 1, n_bins + 1))
        raw_edges = edges.copy()          # keep the finite ones for the record
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.digitize(sd, edges[1:-1])
        wsi = g["wsi"].to_numpy()
        bins = []
        for b in range(n_bins):
            sel = idx == b
            if not sel.any():
                continue
            # Error bar on the bin mean, clustered on the CASE. Regions inside a
            # slide are spatially correlated, so a plain SEM over regions would
            # be far too tight — often invisible at this scale, which is worse
            # than no error bar because it looks like precision.
            per_case = pd.Series(err[sel]).groupby(wsi[sel]).mean().to_numpy()
            n_case = int(per_case.size)
            se = (float(per_case.std(ddof=1) / np.sqrt(n_case))
                  if n_case > 1 else float("nan"))
            m_sd, m_err = float(sd[sel].mean()), float(err[sel].mean())
            bins.append({
                "bin": b,
                "sd_lo": float(raw_edges[b]),
                "sd_hi": float(raw_edges[b + 1]),
                "mean_sd": m_sd,
                "median_sd": float(np.median(sd[sel])),
                "mean_error": m_err,
                "median_error": float(np.median(err[sel])),
                # what a calibrated ensemble would show in this bin
                "expected_error": HALF_NORMAL * m_sd,
                "ratio_obs_over_expected": (m_err / (HALF_NORMAL * m_sd)
                                            if m_sd > 0 else float("nan")),
                "se_error_by_case": se,
                "n": int(sel.sum()),
                "n_wsi": n_case,
            })

        # and the normalised ECE, for continuity with uncertainty_calibration.py
        bu, be, bc = reliability_bins(sd, err, n_bins, sd.min(), sd.max(),
                                      err.min(), err.max())
        rows.append({
            "descriptor": name,
            "reference_class": (PHI_REFERENCE[PHI_NAMES.index(name)]
                                if name in PHI_NAMES else None),
            "n": int(len(g)),
            "n_wsi": int(g["wsi"].nunique()),
            "spearman_rho": float(rho),
            # Naive: it treats every region as independent. Kept for continuity,
            # but the cluster bootstrap below is the one to quote.
            "spearman_p_naive": float(p),
            "spearman_p": float(p),
            # Which side had no spread to rank, when rho is undefined. The two
            # mean opposite things: a constant sigma is an ensemble that agrees
            # everywhere, a constant error is a reference that does not
            # discriminate between regions.
            "undefined_because": (
                None if np.isfinite(rho)
                else "σ constant" if float(np.ptp(sd)) == 0.0
                else "error constant" if float(np.ptp(err)) == 0.0
                else "degenerate"
            ),
            "mean_abs_z": float(np.nanmean(g["z"])),
            "calibration_ratio": float(np.nanmean(g["z"]) / HALF_NORMAL),
            **(dict(zip(("rho_ci_lo", "rho_ci_hi", "n_boot_used"),
                        cluster_bootstrap_rho(g, n_boot, seed)))
               if n_boot else {}),
            **({"rho_shuffled": shuffled_rho(g, min(200, n_boot), seed + 1)}
               if n_boot else {}),
            "ece_normalised": float(expected_calibration_error(bu, be, bc)),
            "mean_sd": float(sd.mean()),
            "mean_error": float(err.mean()),
            "bins": bins,
        })
    return rows


def make_reliability_figure(rows: List[dict], outpath: Path, title: str) -> None:
    """The reliability diagram on its own, at figure quality.

    Separate from `calibration_phi.png`, which is a working panel. Three things
    it adds, each because the compact version can mislead:

    * **Error bars clustered on the case.** Without them a bin mean over ~285
      regions looks precise, when those regions come from twenty slides.
    * **The bin population**, as a bar strip underneath. Quantile bins hold equal
      counts by construction but *not* equal numbers of slides, and a bin drawn
      from three cases should not be read like one drawn from twenty.
    * **Raw units on both axes.** sigma_CPA and |dCPA| are both in CPA, so
      normalising — which `uncertainty_calibration.py` must do for pixels — would
      throw away exactly what makes this comparison strong.

    The reference line is E|e| = 0.80 sigma, not the diagonal: for Gaussian error
    the mean absolute deviation is sigma*sqrt(2/pi), so a diagonal would call a
    perfectly calibrated ensemble 20% over-confident.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    scored = [r for r in rows if r.get("bins")]
    if not scored:
        print("[note] no binned descriptor to plot — reliability figure skipped")
        return

    ncol = len(scored)
    fig = plt.figure(figsize=(4.0 * ncol + 0.6, 5.4))
    gs = GridSpec(2, ncol, height_ratios=[3.4, 1.0], hspace=0.32, wspace=0.28,
                  figure=fig)

    for k, r in enumerate(scored):
        b = r["bins"]
        x = np.array([d["mean_sd"] for d in b])
        y = np.array([d["mean_error"] for d in b])
        se = np.array([d.get("se_error_by_case", np.nan) for d in b], float)
        n = np.array([d["n"] for d in b])
        nw = np.array([d.get("n_wsi", 0) for d in b])
        colour = C_SERIES[PHI_NAMES.index(r["descriptor"]) % len(C_SERIES)]

        ax = fig.add_subplot(gs[0, k])
        # Axes are scaled to their OWN data, not forced equal. A shared scale is
        # the textbook reliability diagram, but it only works while sigma and
        # error are comparable: on an over-confident descriptor (beta0 here has
        # sigma ~40 against an error ~500) equal limits crush every point into a
        # corner and the panel shows nothing. The calibration line carries the
        # comparison instead — read the points against the line, not against 45
        # degrees, which is why the diagonal is not drawn at all.
        xhi = float(x.max()) * 1.15
        yhi = float(max(np.nanmax(y + np.nan_to_num(se)), xhi * HALF_NORMAL)) * 1.12
        ax.plot([0, xhi], [0, xhi * HALF_NORMAL], color=C_MUTED, linewidth=1.4,
                linestyle="--", zorder=3, label="calibrated   E|e| = 0.80 σ")
        ax.errorbar(x, y, yerr=se, color=colour, linewidth=2, marker="o",
                    markersize=6, capsize=3, elinewidth=1.2, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlim(0, xhi)
        ax.set_ylim(0, yhi)
        ax.set_xlabel("ensemble σ", color=C_MUTED, fontsize=9)
        if k == 0:
            ax.set_ylabel("|error| vs real tissue", color=C_MUTED, fontsize=9)
        ratio = r["calibration_ratio"]
        verdict = ("over-confident" if ratio > 1.1
                   else "under-confident" if ratio < 0.9 else "calibrated")
        rho_txt = (f"ρ = {r['spearman_rho']:+.3f}"
                   if np.isfinite(r["spearman_rho"]) else "ρ undefined")
        if np.isfinite(r.get("rho_ci_lo", np.nan)):
            rho_txt += f" [{r['rho_ci_lo']:+.2f}, {r['rho_ci_hi']:+.2f}]"
        ax.set_title(f"{r['descriptor']}\n{rho_txt}\n"
                     f"E|z|/0.80 = {ratio:.2f}  ({verdict})",
                     color=C_INK, fontsize=9.5, loc="left", pad=8)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8.5)
        if k == 0:
            leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            for txt in leg.get_texts():
                txt.set_color(C_MUTED)

        # bin population: equal region counts by construction, but NOT equal
        # numbers of slides, which is what the interval actually rests on
        axb = fig.add_subplot(gs[1, k])
        axb.bar(np.arange(len(n)), n, color=colour, alpha=0.32, zorder=3,
                edgecolor="white", linewidth=0.6)
        for i, (cnt, cases) in enumerate(zip(n, nw)):
            axb.text(i, cnt, f"{cases}", ha="center", va="bottom", fontsize=6.5,
                     color=C_MUTED)
        axb.set_xlabel("σ bin (low → high)", color=C_MUTED, fontsize=8.5)
        if k == 0:
            axb.set_ylabel("regions", color=C_MUTED, fontsize=8.5)
        axb.set_xticks(np.arange(len(n)))
        axb.set_xticklabels([str(i + 1) for i in range(len(n))], fontsize=7)
        axb.set_ylim(0, n.max() * 1.28)
        axb.grid(True, axis="y", color=C_GRID, linewidth=0.8, zorder=0)
        axb.set_axisbelow(True)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            axb.spines[s].set_color(C_GRID)
        axb.tick_params(colors=C_MUTED, labelsize=7.5)

    fig.suptitle(title, color=C_INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.005,
             "Points are bin means in raw descriptor units; error bars are ±1 SE "
             "clustered on the case, not on the region. Dashed line = calibration "
             "(E|e| = 0.80σ for Gaussian error); points above it are\n"
             "over-confident. Axes are scaled per panel, so the line's slope "
             "differs between them — read each panel against its own line. Bars "
             "below give each bin's region count, annotated with its slide count.",
             fontsize=7.5, color=C_MUTED, linespacing=1.5)
    # Not tight_layout: the reliability axes are set_aspect("equal"), which it
    # cannot solve for, and it says so on every run.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.155)
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_figure(t: pd.DataFrame, rows: List[dict], outpath: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in rows if "spearman_rho" in r]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8.5)

    for k, name in enumerate(PHI_NAMES):
        ax = axes[k]
        r = next((x for x in scored if x["descriptor"] == name), None)
        if r is None or not r["bins"]:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{name}\nno reference", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=C_MUTED, style="italic")
            continue

        x = np.array([b["mean_sd"] for b in r["bins"]])
        y = np.array([b["mean_error"] for b in r["bins"]])
        hi = max(x.max(), y.max()) * 1.1
        # the calibrated line, not the diagonal
        ax.plot([0, hi], [0, hi * HALF_NORMAL], color=C_MUTED, linewidth=1.5,
                linestyle="--", zorder=3, label="calibrated  E|e| = 0.80 σ")
        ax.plot(x, y, color=C_SERIES[k % len(C_SERIES)], linewidth=2, marker="o",
                markersize=6, zorder=5)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("ensemble σ", color=C_MUTED, fontsize=9)
        ax.set_ylabel("|error| vs real", color=C_MUTED, fontsize=9)
        ax.set_title(f"{name}\nρ = {r['spearman_rho']:.2f}   E|z|/0.80 = "
                     f"{r['calibration_ratio']:.2f}",
                     color=C_INK, fontsize=9.5, loc="left", pad=8)
        if k == 0:
            leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            for txt in leg.get_texts():
                txt.set_color(C_MUTED)

    # summary panel: rho per descriptor, the headline
    ax = axes[len(PHI_NAMES)]
    if scored:
        names = [r["descriptor"] for r in scored]
        # rho is undefined when either side has no spread to rank. That is a
        # finding, so it gets a bar of zero and a label naming WHICH side was
        # constant — rather than being dropped, or drawn at a NaN position, which
        # matplotlib turns into six "posx and posy should be finite" lines and an
        # unlabelled gap. The two cases mean opposite things: a constant sigma is
        # an ensemble that agrees everywhere, a constant error is a reference
        # that does not discriminate between regions.
        rhos = [r["spearman_rho"] if np.isfinite(r["spearman_rho"]) else 0.0
                for r in scored]
        ys = np.arange(len(names))[::-1]
        ax.axvline(0, color=C_MUTED, linewidth=1, zorder=2)
        ax.barh(ys, rhos, color=[C_SERIES[PHI_NAMES.index(n) % len(C_SERIES)]
                                 for n in names], height=0.6, zorder=4)
        for y, r, rho in zip(ys, scored, rhos):
            note = (f"p={r['spearman_p']:.1e}" if np.isfinite(r["spearman_rho"])
                    else r.get("undefined_because") or "ρ undefined")
            ax.text(rho + (0.02 if rho >= 0 else -0.02), y, note, va="center",
                    fontsize=7.5, ha="left" if rho >= 0 else "right",
                    color=C_MUTED)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(min(-0.1, min(rhos) * 1.3), max(0.6, max(rhos) * 1.45))
    ax.set_xlabel("Spearman ρ(σ, |error|)", color=C_MUTED, fontsize=9)
    ax.set_title("does σ rank the error?", color=C_INK, fontsize=9.5,
                 loc="left", pad=8)

    fig.suptitle(title, color=C_INK, fontsize=13, x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.012,
             "ρ is the claim that survives noise in the reference: a floor or a "
             "registration offset attenuates it toward zero, so a positive value "
             "is conservative. The dashed line is E|e| = 0.80 σ, not the diagonal — "
             "for Gaussian error the mean absolute deviation is σ·√(2/π).",
             color=C_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(outpath, dpi=150, facecolor="white")
    print(f"wrote {outpath}")


# Flags that moved to compute_phi_reference.py. argparse would reject them with
# "unrecognized arguments", which does not say where they went.
MOVED = ("--real_psr", "--real_lumen", "--he_masks", "--he_dir", "--strip_prefix",
         "--white_thresh", "--min_object_px", "--closing_px", "--mpp",
         "--tile_size", "--save_reference", "--reference_only")


def _check_moved_flags() -> None:
    import sys
    used = [f for f in MOVED if any(a == f or a.startswith(f + "=")
                                    for a in sys.argv[1:])]
    if used:
        raise SystemExit(
            f"{', '.join(used)} moved to compute_phi_reference.py — measuring "
            f"the real tissue is now its own stage, so it runs once and every "
            f"calibration reuses it.\n\n"
            f"  python compute_phi_reference.py --phi_csv <per_region.csv> \\\n"
            f"      --real_psr <real masks> --strip_prefix --he_masks <tissue> \\\n"
            f"      --outdir <dir>\n"
            f"  python calibrate_phi.py --phi_csv <per_region.csv> \\\n"
            f"      --reference_csv <dir>/reference_phi.csv --outdir <dir>"
        )


def main() -> None:
    _check_moved_flags()
    ap = argparse.ArgumentParser("Calibrate phi_struct uncertainty against real tissue")
    ap.add_argument("--phi_csv", type=Path, required=True,
                    help="per_region.csv from compute_phi_uncertainty.py "
                         "(or the pooled one from aggregate_phi_uncertainty.py).")
    ap.add_argument("--reference_csv", type=Path, required=True,
                    help="reference_phi.csv from compute_phi_reference.py — "
                         "phi of the real tissue on this same grid. Verified "
                         "against --phi_csv's region boxes; a mismatch exits "
                         "rather than pairing one grid's spread with another "
                         "grid's tissue.")
    ap.add_argument("--outdir", type=Path, default=Path("calibration_phi"))

    ap.add_argument("--prediction", choices=("grand", "fold"), default="grand",
                    help="'grand' pairs the mean of all members with the total "
                         "spread — the deployed prediction. 'fold' pairs each "
                         "subset's mean with its procedural spread alone, and "
                         "comparing the two is the data-exposure claim. "
                         "[%(default)s]")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="Bootstrap resamples for the rho CI, drawing WHOLE "
                         "SLIDES. Regions inside a slide are spatially "
                         "correlated, so the naive p-value over ~2850 regions "
                         "describes a cohort that does not exist. 0 disables "
                         "both this and the shuffled control. [%(default)s]")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the bootstrap and the shuffled control.")

    args = ap.parse_args()

    df = pd.read_csv(args.phi_csv)
    print(f"[1/2] {len(df)} regions over {df['wsi'].nunique()} WSI from {args.phi_csv}")
    ref, provenance = load_reference(args.reference_csv, df)

    t = pair(df, ref, args.prediction, args.n_folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins, args.n_boot, args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.outdir / "per_region_calibration.csv", index=False)

    # The reliability diagram's own data, flat, so the figure can be rebuilt or
    # restyled for the manuscript without re-running the calibration — and so
    # the numbers behind each point are quotable rather than only plotted.
    bin_rows = [{"descriptor": r["descriptor"],
                 "reference_class": r["reference_class"],
                 "prediction": args.prediction, **b}
                for r in rows if r.get("bins") for b in r["bins"]]
    if bin_rows:
        pd.DataFrame(bin_rows).to_csv(
            args.outdir / "reliability_bins.csv", index=False)

    make_figure(t, rows, args.outdir / "calibration_phi.png",
                f"φ_struct calibration — {args.prediction} prediction")
    make_reliability_figure(rows, args.outdir / "reliability_phi.png",
                            f"φ_struct reliability — {args.prediction} prediction")

    payload = {
        "prediction": args.prediction,
        "n_regions": int(df.shape[0]),
        "per_descriptor": rows,
        # How the reference was measured is not this script's decision, so it is
        # recorded rather than re-derived — otherwise a result carries no trace
        # of the thresholds and mask directories behind its target.
        "reference": {"path": str(args.reference_csv), **provenance},
        "conventions": {
            "sigma": "predictive SD (spread of members), not the standard error "
                     "of the mean — sigma/sqrt(M) would be tiny and the test "
                     "would collapse into a test of bias",
            "calibrated_mean_abs_z": HALF_NORMAL,
            "calibration_ratio": ">1 is over-confident: errors exceed the spread",
        },
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
    }
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== φ_struct calibration ===")
    print(f"{'descriptor':24s} {'ref':>5s} {'n':>6s} {'rho':>7s} "
          f"{'95% CI (by case)':>18s} {'shuf':>6s} {'E|z|/0.80':>10s}")
    for r in rows:
        if "spearman_rho" not in r:
            print(f"{r['descriptor']:24s} {'':>5s} {r['n']:>6d}   {r.get('note', '')}")
            continue
        ci = (f"[{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "naive p "
              f"{r['spearman_p']:.0e}")
        shuf = (f"{r['rho_shuffled']:.3f}"
                if np.isfinite(r.get("rho_shuffled", np.nan)) else "-")
        print(f"{r['descriptor']:24s} {r['reference_class']:>5s} {r['n']:>6d} "
              f"{r['spearman_rho']:>7.3f} {ci:>18s} {shuf:>6s} "
              f"{r['calibration_ratio']:>10.2f}"
              + (f"   [{r['undefined_because']}]"
                 if r.get("undefined_because") else ""))
    print("\nrho > 0 means uncertain regions are the wrong ones. E|z|/0.80 > 1")
    print("means the ensemble is over-confident: errors exceed its own spread.")
    print("The CI resamples SLIDES, not regions — quote it, not the naive p, "
          "which\ntreats every region as independent. 'shuf' is the negative "
          "control: mean\n|rho| with the pairing broken, which must sit near 0 "
          "for rho to mean anything.")
    print(f"\nwrote {args.outdir / 'per_region_calibration.csv'}")
    if bin_rows:
        print(f"wrote {args.outdir / 'reliability_bins.csv'}  "
              f"({len(bin_rows)} bins — the reliability diagram's own data)")
    print(f"wrote {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
