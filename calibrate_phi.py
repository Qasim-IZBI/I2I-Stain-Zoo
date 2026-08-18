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
reliability_bins.csv         the reliability diagram's own data, per bin x component
within_slide.csv             per-slide rho, raw and with the point prediction
                             partialled out — the confound-controlled result
risk_coverage.csv            error vs coverage: what discarding the least certain
                             regions buys, with the oracle ceiling
summary.json                 Spearman rho, E|z|, ECE, reliability bins, provenance
calibration_phi.png          working panel: reliability per descriptor + rho summary
reliability_phi.png          reliability per descriptor, total / procedural /
                             data-exposure sigma overlaid
risk_coverage.png            the selective-prediction curve — the headline figure
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

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


# The three spreads `compute_phi_uncertainty.py` writes per descriptor. In
# `grand` mode all three are scored against the SAME error — the prediction is
# the mean of all fifty either way — so the comparison isolates which variance
# component carries the predictive signal, rather than confounding it with a
# change of prediction.
COMPONENTS = (("total", "sd_total_{n}"),
              ("procedural", "sd_procedural_{n}"),
              ("data_exposure", "sd_data_{n}"))

COMPONENT_LABEL = {
    "total": "total σ",
    "procedural": "procedural σ (seed)",
    "data_exposure": "data-exposure σ (subset)",
    "procedural_within_fold": "procedural σ, per subset",
    **{f"fold{i}": f"subset {i}" for i in range(1, 21)},
}


def pair(df: pd.DataFrame, ref: pd.DataFrame, mode: str, n_folds: int) -> pd.DataFrame:
    """Long table of (prediction, uncertainty, reference) per region x descriptor.

    `grand` pairs the mean over all members with **each** variance component in
    turn — total, procedural and data-exposure. The prediction, and therefore the
    error, is identical across the three; only sigma changes. So the three
    reliability curves for a descriptor differ only in how far along the sigma
    axis each point sits, and comparing them answers directly which component
    the calibration rests on.

    `fold` pairs each subset's mean with that subset's procedural spread alone,
    giving one row per subset. That is a different *prediction*, not a different
    spread, which is why it is a separate mode rather than a fourth component.
    """
    merged = df.merge(ref, on=["wsi", "region_index"], how="inner")
    if merged.empty:
        raise SystemExit("no regions matched between --phi_csv and the reference")

    out = []
    if mode == "grand":
        sources = [("grand", comp, "mu_{n}", key) for comp, key in COMPONENTS]
    else:
        sources = [(f"fold{f}", "procedural_within_fold",
                    f"fold{f}_mu_{{n}}", f"fold{f}_sd_{{n}}")
                   for f in range(1, n_folds + 1)]

    for label, component, mu_key, sd_key in sources:
        for name in PHI_NAMES:
            mu_col, sd_col, real_col = (mu_key.format(n=name),
                                        sd_key.format(n=name), f"real_{name}")
            if not {mu_col, sd_col, real_col} <= set(merged.columns):
                continue
            block = merged[["wsi", "region_index", mu_col, sd_col, real_col]].copy()
            block.columns = ["wsi", "region_index", "mu", "sd", "real"]
            block["descriptor"] = name
            block["prediction"] = label
            block["component"] = component
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
    """Per descriptor x variance component: does sd rank error, and is its scale
    right?"""
    rows = []
    # `prediction` joins the keys so each subset is scored on its own. Pooling
    # the five would enter every region five times against ONE shared target,
    # which is not five observations of anything: it narrows the interval
    # without adding evidence. In `grand` mode the column is constant, so
    # including it changes nothing there.
    keys = ["descriptor"]
    if "component" in t.columns:
        keys.append("component")
    if "prediction" in t.columns:
        keys.append("prediction")
    for key, g in t.groupby(keys, sort=False):
        # groupby on a LIST of keys yields a tuple even when the list holds one
        # element, so this cannot unpack a fixed pair.
        key = key if isinstance(key, tuple) else (key,)
        name = key[0]
        component = key[1] if len(key) > 1 else "total"
        prediction = key[2] if len(key) > 2 else "grand"
        n_raw = int(len(g))
        g = g.dropna(subset=["sd", "error"])
        g = g[np.isfinite(g["sd"]) & np.isfinite(g["error"]) & (g["sd"] > 0)]
        # A negative ANOVA variance component is a real outcome near zero and is
        # reported rather than clipped, so it has no SD and its column is empty.
        # Those regions drop out here, and how many did is part of reading the
        # data-exposure panel: a component estimated as zero on half the regions
        # is a finding about the ensemble, not a missing measurement.
        dropped = n_raw - int(len(g))
        if len(g) < 3:
            rows.append({"descriptor": name, "component": component,
                         "prediction": prediction,
                         "n": int(len(g)), "n_dropped": dropped,
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
            "component": component,
            "prediction": prediction,
            "reference_class": (PHI_REFERENCE[PHI_NAMES.index(name)]
                                if name in PHI_NAMES else None),
            "n": int(len(g)),
            # regions with no SD for this component — see above
            "n_dropped": dropped,
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

    # Descriptor-major, components in their fixed order. `pair` emits
    # component-major, which would print each descriptor's three components
    # scattered down the table and make the one comparison this exists for
    # something the reader has to reassemble by eye.
    comp_order = [c for c, _ in COMPONENTS] + ["procedural_within_fold"]
    rows.sort(key=lambda r: (
        PHI_NAMES.index(r["descriptor"]) if r["descriptor"] in PHI_NAMES else 99,
        comp_order.index(r.get("component", "total"))
        if r.get("component", "total") in comp_order else 99,
        str(r.get("prediction", "")),
    ))
    return rows


def fold_agreement(rows: List[dict]) -> List[dict]:
    """Do the subsets agree, and what does pooling them cost?

    Five subsets give five estimates of the same quantity, so their SPREAD is
    the evidence — a descriptor whose rho swings sign between subsets has not
    been shown to calibrate, however tight the pooled interval looks. Pooling is
    reported alongside, explicitly labelled, because it is what a reader would
    otherwise compute and it is anti-conservative here: every region enters five
    times against one shared target.
    """
    out = []
    per_desc: Dict[str, list] = {}
    for r in rows:
        if r.get("component") != "procedural_within_fold" or "spearman_rho" not in r:
            continue
        per_desc.setdefault(r["descriptor"], []).append(r)
    for name, rs in per_desc.items():
        rhos = np.array([r["spearman_rho"] for r in rs], float)
        rhos = rhos[np.isfinite(rhos)]
        if rhos.size == 0:
            continue
        signs = set(np.sign(rhos[rhos != 0]).tolist())
        out.append({
            "descriptor": name,
            "n_folds": int(rhos.size),
            "rho_median": float(np.median(rhos)),
            "rho_min": float(rhos.min()),
            "rho_max": float(rhos.max()),
            "rho_range": float(rhos.max() - rhos.min()),
            # the readout: subsets disagreeing on the SIGN have not shown
            # anything, whatever a pooled interval says
            "consistent_sign": bool(len(signs) <= 1),
            "folds": {r["prediction"]: float(r["spearman_rho"]) for r in rs},
        })
    return out


# One colour per variance component, held fixed across every panel and every
# run — the component is the entity, so its colour must not depend on which
# descriptors happened to be scored.
C_COMPONENT = {
    "total": "#2a78d6",
    "procedural": "#eb6834",
    "data_exposure": "#1baf7a",
    "procedural_within_fold": "#4a3aa7",
}
M_COMPONENT = {"total": "o", "procedural": "s", "data_exposure": "^",
               "procedural_within_fold": "D"}

# In fold mode the curves are the five subsets, not the three components, so the
# series key changes. Sequential colours rather than categorical ones: the folds
# are five draws of one thing, where the components are three different things.
C_FOLD = ("#1b4a8f", "#2a78d6", "#5aa0e6", "#8fbfe8", "#b9d7f2")
M_FOLD = ("o", "s", "^", "D", "v")


def series_of(r: dict) -> str:
    """Which curve a scored row belongs to — component, or subset in fold mode."""
    comp = r.get("component", "total")
    return r.get("prediction", "grand") if comp == "procedural_within_fold" else comp


def series_style(label: str, idx: int) -> tuple:
    if label in C_COMPONENT:
        return C_COMPONENT[label], M_COMPONENT.get(label, "o")
    return C_FOLD[idx % len(C_FOLD)], M_FOLD[idx % len(M_FOLD)]


def _partial_spearman(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Spearman of a and b with z partialled out, on ranks.

    Rank-transform all three, regress the first two on the third, correlate the
    residuals. Rank-based so it inherits Spearman's robustness to the skew in
    CPA, and linear in ranks so it removes a monotone dependence on z rather
    than only a linear one.
    """
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    if np.ptp(rz) == 0 or np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    A = np.c_[np.ones_like(rz, dtype=float), rz]
    ra = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    rb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(spearmanr(ra, rb).statistic)


def within_slide(t: pd.DataFrame, min_regions: int, n_boot: int,
                 seed: int) -> List[dict]:
    """Per-slide rho, and per-slide rho with the point prediction partialled out.

    Two things this fixes that a pooled correlation cannot.

    **The unit of replication.** Pooling ~2850 regions from twenty cases and
    correlating once treats them as 2850 observations. Computing rho inside each
    slide and summarising the twenty values makes the slide the unit, which is
    what it is. `n_positive` out of `n_slides` is then a sign test anyone can
    read without trusting a bootstrap.

    **The level confound, which is the serious one.** sigma tracks how much
    structure a region holds — on the UC liver cohort rho(sigma, mu_CPA) = +0.76
    — and absolute error grows with the same thing. So a raw rho is mostly the
    two sharing a dependence on the amount of collagen, not the ensemble knowing
    where it is wrong. Partialling on `mu` asks the question that survives
    review: does the spread say anything the POINT PREDICTION does not already
    imply? `mu` and not `real`, because mu is available at inference and real is
    not — partialling on the reference would remove the very signal being
    tested.

    A raw rho that collapses under this is a structure-content map wearing an
    uncertainty label. Report both.
    """
    out = []
    keys = [k for k in ("descriptor", "component", "prediction") if k in t.columns]
    if "mu" not in t.columns:
        return out
    for key, g in t.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(keys, key))
        g = g.dropna(subset=["sd", "error", "mu"])
        g = g[np.isfinite(g["sd"]) & np.isfinite(g["error"]) & (g["sd"] > 0)]
        raw, par, sizes = [], [], []
        for _, sub in g.groupby("wsi"):
            if len(sub) < min_regions:
                continue
            a, b, m = (sub["sd"].to_numpy(), sub["error"].to_numpy(),
                       sub["mu"].to_numpy())
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            raw.append(float(spearmanr(a, b).statistic))
            par.append(_partial_spearman(a, b, m))
            sizes.append(len(sub))
        raw = np.array([v for v in raw if np.isfinite(v)], float)
        par = np.array([v for v in par if np.isfinite(v)], float)
        if raw.size < 3:
            continue
        row = {**meta, "n_slides": int(raw.size),
               "regions_per_slide_min": int(min(sizes)),
               "regions_per_slide_max": int(max(sizes))}
        rng = np.random.default_rng(seed)
        for tag, v in (("raw", raw), ("partial_mu", par)):
            if v.size < 3:
                continue
            # resampling the SLIDES themselves: twenty values, one per case
            bs = [float(np.mean(rng.choice(v, v.size, replace=True)))
                  for _ in range(max(n_boot, 1000))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update({
                f"rho_{tag}_mean": float(v.mean()),
                f"rho_{tag}_median": float(np.median(v)),
                f"rho_{tag}_ci_lo": float(lo),
                f"rho_{tag}_ci_hi": float(hi),
                f"n_positive_{tag}": int((v > 0).sum()),
            })
            if v.size > 5:
                try:
                    row[f"wilcoxon_p_{tag}"] = float(wilcoxon(v).pvalue)
                except ValueError:      # all-zero differences
                    row[f"wilcoxon_p_{tag}"] = float("nan")
        row["per_slide_raw"] = [float(v) for v in raw]
        row["per_slide_partial_mu"] = [float(v) for v in par]
        out.append(row)
    return out


def risk_coverage(t: pd.DataFrame, coverages, n_boot: int, seed: int) -> List[dict]:
    """Selective prediction: keep the most certain regions, measure what remains.

    The question a correlation coefficient invites and does not answer — rho =
    0.22 is real but a reader is entitled to ask what it buys. This answers in
    the units of the task: rank regions by sigma, discard the least certain,
    and report the error over what is left.

    Three reference points, and the figure needs all three:

    * **random** selection has expected MAE equal to the overall MAE, exactly —
      dropping a random subset changes nothing in expectation. So the curve's
      departure from flat *is* the effect, and no Monte-Carlo baseline is needed
      to establish it.
    * **oracle**, ranking by the true error, is the ceiling. The fraction of it
      achieved is the honest measure of how far this is from solved, and it
      belongs beside every headline number.
    * the **bootstrap CI**, resampling whole slides, decides whether the
      reduction survives a cohort of twenty cases.
    """
    out = []
    keys = [k for k in ("descriptor", "component", "prediction") if k in t.columns]
    for key, g in t.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(keys, key))
        g = g.dropna(subset=["sd", "error"])
        g = g[np.isfinite(g["sd"]) & np.isfinite(g["error"])]
        if len(g) < 20:
            continue
        sd, err, wsi = (g["sd"].to_numpy(), g["error"].to_numpy(),
                        g["wsi"].to_numpy())
        base = float(err.mean())
        order = np.argsort(sd, kind="stable")
        by_sd = err[order]
        by_err = np.sort(err)                      # the oracle ordering

        u = np.unique(wsi)
        idx = {x: np.where(wsi == x)[0] for x in u}
        rng = np.random.default_rng(seed)
        draws = None
        if n_boot and len(u) >= 3:
            draws = {c: [] for c in coverages}
            for _ in range(n_boot):
                pick = np.concatenate([idx[x] for x in
                                       rng.choice(u, len(u), replace=True)])
                s, e = sd[pick], err[pick]
                b = e.mean()
                if b <= 0:
                    continue
                es = e[np.argsort(s, kind="stable")]
                for c in coverages:
                    k = max(1, int(round(len(es) * c)))
                    draws[c].append(es[:k].mean() / b - 1.0)

        for c in coverages:
            k = max(1, int(round(len(by_sd) * c)))
            mae = float(by_sd[:k].mean())
            omae = float(by_err[:k].mean())
            rel = mae / base - 1.0 if base > 0 else float("nan")
            orel = omae / base - 1.0 if base > 0 else float("nan")
            row = {**meta, "coverage": float(c), "n_kept": int(k), "n": int(len(g)),
                   "mae": mae,
                   # random selection is unbiased, so this is exact rather than
                   # simulated: the departure from it is the whole effect
                   "mae_random": base,
                   "mae_oracle": omae,
                   "rel_change": float(rel),
                   "rel_change_oracle": float(orel),
                   # What fraction of the achievable gain the ensemble
                   # captures. At full coverage both gains are zero and the
                   # ratio is 0/0, which floating point renders as a confident
                   # 100% — the one number here a reader must not misread.
                   "capture_of_oracle": (float(rel / orel) if orel < -1e-9
                                         else float("nan"))}
            if draws is not None and len(draws[c]) >= max(20, n_boot // 10):
                lo, hi = np.percentile(draws[c], [2.5, 97.5])
                row["rel_ci_lo"], row["rel_ci_hi"] = float(lo), float(hi)
            out.append(row)
    return out


def make_risk_coverage_figure(rc: List[dict], outpath: Path, title: str) -> None:
    """Error against coverage, one panel per descriptor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rc:
        return
    df = pd.DataFrame(rc)
    order = [n for n in PHI_NAMES if n in set(df["descriptor"])]
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(4.3 * len(order) + 0.6, 4.3),
                             squeeze=False)
    for k, name in enumerate(order):
        ax = axes[0][k]
        sub = df[df.descriptor == name]
        series = list(dict.fromkeys(
            sub["prediction"] if (sub["component"] == "procedural_within_fold").all()
            else sub["component"]))
        col = ("prediction" if (sub["component"] == "procedural_within_fold").all()
               else "component")
        for gi, s in enumerate(series):
            gg = sub[sub[col] == s].sort_values("coverage")
            colour, marker = series_style(s, gi)
            ax.plot(gg["coverage"] * 100, gg["rel_change"] * 100, color=colour,
                    marker=marker, markersize=5, linewidth=1.9, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=COMPONENT_LABEL.get(s, s))
            if {"rel_ci_lo", "rel_ci_hi"} <= set(gg.columns):
                ax.fill_between(gg["coverage"] * 100, gg["rel_ci_lo"] * 100,
                                gg["rel_ci_hi"] * 100, color=colour, alpha=0.13,
                                linewidth=0, zorder=2)
        first = sub[sub[col] == series[0]].sort_values("coverage")
        ax.plot(first["coverage"] * 100, first["rel_change_oracle"] * 100,
                color=C_INK, linestyle=":", linewidth=1.5, zorder=4,
                label="oracle (rank by true error)")
        # random selection is unbiased, so its curve is exactly zero — the line
        # every other curve has to beat
        ax.axhline(0, color=C_MUTED, linewidth=1.3, linestyle="--", zorder=3,
                   label="random / keep all")
        ax.invert_xaxis()
        ax.set_xlabel("coverage: % of regions kept", color=C_MUTED, fontsize=9)
        if k == 0:
            ax.set_ylabel("change in MAE vs keeping all (%)", color=C_MUTED,
                          fontsize=9)
        ax.set_title(name, color=C_INK, fontsize=10, loc="left", pad=8)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8.5)
        if k == 0:
            leg = ax.legend(frameon=False, fontsize=7, loc="lower left")
            for txt in leg.get_texts():
                txt.set_color(C_MUTED)

    fig.suptitle(title, color=C_INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.012,
             "Discard the least certain regions and measure the error on what "
             "remains. Below zero is useful. Random selection is unbiased, so its "
             "curve is exactly the zero line; the gap to the dotted oracle is how "
             "much of the achievable gain the ensemble captures. Bands are 95% "
             "bootstrap, resampling slides.",
             fontsize=7.5, color=C_MUTED)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20)
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_reliability_figure(rows: List[dict], outpath: Path, title: str) -> None:
    """Reliability per descriptor, with the variance components overlaid.

    One panel per descriptor, one curve per component (total, procedural,
    data-exposure). The prediction is the same in all three, so the **error is
    identical** and only sigma moves — which is exactly what makes them
    comparable on one axis. Reading down a panel answers the question the crossed
    5x10 grid exists to pose: is it the seed spread or the data-exposure spread
    that tracks the error?

    Three things it does that the compact working panel does not, each because
    the compact one can mislead:

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

    # descriptor -> its components, in the fixed order, so panels line up
    order = [n for n in PHI_NAMES if any(r["descriptor"] == n for r in scored)]
    by_desc = {n: [r for r in scored if r["descriptor"] == n] for n in order}
    comp_order = [c for c, _ in COMPONENTS] + ["procedural_within_fold"]
    for n in by_desc:
        by_desc[n].sort(key=lambda r: (comp_order.index(r.get("component", "total"))
                                       if r.get("component") in comp_order else 99,
                                       str(r.get("prediction", ""))))

    ncol = len(order)
    fig = plt.figure(figsize=(4.6 * ncol + 0.8, 6.8))
    gs = GridSpec(2, ncol, height_ratios=[3.6, 1.0], hspace=0.34, wspace=0.30,
                  figure=fig)

    for k, name in enumerate(order):
        group = by_desc[name]
        ax = fig.add_subplot(gs[0, k])

        # Axes are scaled to their OWN data, not forced equal. A shared scale is
        # the textbook reliability diagram, but it only works while sigma and
        # error are comparable: on an over-confident descriptor (beta0 sits at
        # sigma ~40 against an error ~500) equal limits crush every point into a
        # corner and the panel shows nothing. The calibration line carries the
        # comparison instead — read the points against the line, not against 45
        # degrees, which is why the diagonal is not drawn at all.
        xs = [d["mean_sd"] for r in group for d in r["bins"]]
        ys = [d["mean_error"] + (d.get("se_error_by_case") or 0.0)
              for r in group for d in r["bins"]]
        xhi = float(max(xs)) * 1.15
        yhi = float(max(max(ys), xhi * HALF_NORMAL)) * 1.12

        ax.plot([0, xhi], [0, xhi * HALF_NORMAL], color=C_MUTED, linewidth=1.4,
                linestyle="--", zorder=3, label="calibrated  E|e| = 0.80σ")

        for gi, r in enumerate(group):
            comp = series_of(r)
            colour, marker = series_style(comp, gi)
            b = r["bins"]
            x = np.array([d["mean_sd"] for d in b])
            y = np.array([d["mean_error"] for d in b])
            se = np.array([d.get("se_error_by_case", np.nan) for d in b], float)
            ax.errorbar(x, y, yerr=se, color=colour, linewidth=1.9, marker=marker,
                        markersize=5.5, capsize=3, elinewidth=1.1, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=COMPONENT_LABEL.get(comp, comp))

        ax.set_xlim(0, xhi)
        ax.set_ylim(0, yhi)
        ax.set_xlabel("ensemble σ", color=C_MUTED, fontsize=9)
        if k == 0:
            ax.set_ylabel("|error| vs real tissue", color=C_MUTED, fontsize=9)
        # The per-component numbers go ABOVE the axes, colour-coded, rather than
        # into the legend. Four legend entries carrying a rho and a CI each is a
        # box large enough to cover the curves it labels, and where it lands
        # depends on the data — so on some runs it hides them and on others it
        # does not.
        ax.set_title(name, color=C_INK, fontsize=10, loc="left",
                     pad=10 + 10.5 * len(group))
        for gi, r in enumerate(group):
            comp = series_of(r)
            colour, _ = series_style(comp, gi)
            rho = r["spearman_rho"]
            txt = f"{COMPONENT_LABEL.get(comp, comp).split(' σ')[0]:<14s}"
            if np.isfinite(rho):
                txt += f" ρ={rho:+.3f}"
                if np.isfinite(r.get("rho_ci_lo", np.nan)):
                    txt += f" [{r['rho_ci_lo']:+.2f},{r['rho_ci_hi']:+.2f}]"
            else:
                txt += f" ρ {r.get('undefined_because') or 'undefined'}"
            txt += f"   E|z|/0.80={r['calibration_ratio']:.2f}"
            # Belongs beside that component's own numbers, not on the bar strip
            # below, where it lands on top of the bars it is describing.
            if r.get("n_dropped"):
                txt += f"   ({r['n_dropped']} of {r['n'] + r['n_dropped']} no σ)"
            ax.text(0.0, 1.005 + 0.042 * (len(group) - 1 - gi), txt,
                    transform=ax.transAxes, fontsize=6.6,
                    color=colour, family="monospace")
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8.5)

        # bin population, from the first component — the binning is per
        # component, but the region set behind it is the same except where a
        # negative variance component left no SD
        axb = fig.add_subplot(gs[1, k])
        width = 0.8 / max(1, len(group))
        for gi, r in enumerate(group):
            colour, _ = series_style(series_of(r), gi)
            n = np.array([d["n"] for d in r["bins"]])
            pos = np.arange(len(n)) + (gi - (len(group) - 1) / 2) * width
            axb.bar(pos, n, width=width, color=colour,
                    alpha=0.42, zorder=3, edgecolor="white", linewidth=0.5)
            if gi == 0:
                for i, (cnt, cases) in enumerate(
                        zip(n, [d.get("n_wsi", 0) for d in r["bins"]])):
                    axb.text(i, cnt, f"{cases}", ha="center", va="bottom",
                             fontsize=6.2, color=C_MUTED)
        axb.set_xlabel("σ bin (low → high)", color=C_MUTED, fontsize=8.5)
        if k == 0:
            axb.set_ylabel("regions", color=C_MUTED, fontsize=8.5)
        nb = max(len(r["bins"]) for r in group)
        axb.set_xticks(np.arange(nb))
        axb.set_xticklabels([str(i + 1) for i in range(nb)], fontsize=7)
        axb.grid(True, axis="y", color=C_GRID, linewidth=0.8, zorder=0)
        axb.set_axisbelow(True)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            axb.spines[s].set_color(C_GRID)
        axb.tick_params(colors=C_MUTED, labelsize=7.5)

    fig.suptitle(title, color=C_INK, fontsize=13, x=0.006, ha="left", y=0.992)
    # One legend for the whole figure: the components are the same everywhere,
    # so repeating it per panel spends space on nothing.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels),
                     loc="upper left", bbox_to_anchor=(0.005, 0.955))
    for txt in leg.get_texts():
        txt.set_color(C_MUTED)
    fig.text(0.006, 0.007,
             "One curve per variance component. The prediction is the same in all "
             "three, so the ERROR is identical and only σ moves — which component\n"
             "tracks the error is the question the crossed grid poses. Points are "
             "bin means in raw units; error bars are ±1 SE clustered on the case.\n"
             "Dashed line = calibration (E|e| = 0.80σ); points above it are "
             "over-confident. Axes are scaled per panel. Bars give each bin's\n"
             "region count, the first component annotated with its slide count.",
             fontsize=7.5, color=C_MUTED, linespacing=1.5)
    # Not tight_layout: it cannot solve for these, and says so on every run.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.185)
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_figure(t: pd.DataFrame, rows: List[dict], outpath: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Total only. This panel is one axes per descriptor and cannot carry three
    # curves; the component comparison lives in reliability_phi.png.
    scored = [r for r in rows if "spearman_rho" in r
              and r.get("component", "total") in ("total", "procedural_within_fold")]
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
    ap.add_argument("--min_regions_per_slide", type=int, default=15,
                    help="Slides with fewer regions are dropped from the "
                         "within-slide analysis: a rho over a handful of regions "
                         "is noise that the per-slide mean would weight equally "
                         "with a well-estimated one. [%(default)s]")
    ap.add_argument("--coverages", type=float, nargs="*",
                    default=[1.0, 0.9, 0.8, 0.7, 0.5],
                    help="Fractions of regions to KEEP, most certain first, for "
                         "the risk-coverage curve. This is what a correlation "
                         "coefficient does not answer: what discarding the "
                         "least certain regions actually buys. [%(default)s]")

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
    agreement = fold_agreement(rows)
    rc = risk_coverage(t, sorted(args.coverages, reverse=True), args.n_boot,
                       args.seed)
    ws = within_slide(t, args.min_regions_per_slide, args.n_boot, args.seed)

    bin_rows = [{"descriptor": r["descriptor"],
                 "component": r.get("component", "total"),
                 "fold": r.get("prediction", "grand"),
                 "reference_class": r["reference_class"],
                 "prediction": args.prediction, **b}
                for r in rows if r.get("bins") for b in r["bins"]]
    if bin_rows:
        pd.DataFrame(bin_rows).to_csv(
            args.outdir / "reliability_bins.csv", index=False)

    make_figure(t, rows, args.outdir / "calibration_phi.png",
                f"φ_struct calibration — {args.prediction} prediction")
    if ws:
        flat = [{k: v for k, v in r.items() if not k.startswith("per_slide_")}
                for r in ws]
        pd.DataFrame(flat).to_csv(args.outdir / "within_slide.csv", index=False)
    if rc:
        pd.DataFrame(rc).to_csv(args.outdir / "risk_coverage.csv", index=False)
        make_risk_coverage_figure(
            rc, args.outdir / "risk_coverage.png",
            f"Selective prediction — {args.prediction} prediction")
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
        "fold_agreement": agreement,
        "risk_coverage": rc,
        "within_slide": ws,
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
    fold_mode = any(r.get("component") == "procedural_within_fold" for r in rows)
    col = "subset" if fold_mode else "component"
    print(f"{'descriptor':24s} {col:>14s} {'n':>6s} {'rho':>7s} "
          f"{'95% CI (by case)':>18s} {'shuf':>6s} {'E|z|/0.80':>10s}")
    last = None
    last_desc_seen = None
    for r in rows:
        comp = series_of(r)
        # group by descriptor so the components/subsets read as one block
        if last is not None and r["descriptor"] != last:
            print()
        last = r["descriptor"]
        first = (last_desc_seen != r["descriptor"])
        last_desc_seen = r["descriptor"]
        shown = r["descriptor"] if first else ""
        if "spearman_rho" not in r:
            print(f"{shown:24s} {comp:>14s} {r['n']:>6d}   {r.get('note', '')}")
            continue
        ci = (f"[{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "naive p "
              f"{r['spearman_p']:.0e}")
        shuf = (f"{r['rho_shuffled']:.3f}"
                if np.isfinite(r.get("rho_shuffled", np.nan)) else "-")
        print(f"{shown:24s} {comp:>14s} {r['n']:>6d} "
              f"{r['spearman_rho']:>7.3f} {ci:>18s} {shuf:>6s} "
              f"{r['calibration_ratio']:>10.2f}"
              + (f"   [{r['undefined_because']}]"
                 if r.get("undefined_because") else "")
              + (f"   ({r['n_dropped']} without σ)" if r.get("n_dropped") else ""))
    print("\nrho > 0 means uncertain regions are the wrong ones. E|z|/0.80 > 1")
    print("means the ensemble is over-confident: errors exceed its own spread.")
    print("The CI resamples SLIDES, not regions — quote it, not the naive p, "
          "which\ntreats every region as independent. 'shuf' is the negative "
          "control: mean\n|rho| with the pairing broken, which must sit near 0 "
          "for rho to mean anything.")
    if ws:
        print("\n--- within slide, and is it just a structure-content map? ---")
        print("rho computed INSIDE each slide, then summarised over slides — the "
              "slide is the\nunit of replication, not the region. 'partial' "
              "additionally removes the point\nprediction mu: sigma tracks how "
              "much structure a region holds, and absolute error\ndoes too, so a "
              "raw rho is largely the two sharing that. The partial asks whether "
              "the\nspread says anything mu does not already imply. Report "
              "both.\n")
        print(f"{'descriptor':22s} {'comp':>12s} {'rho':>7s} {'95% CI':>17s} "
              f"{'+ve':>6s} {'partial':>8s} {'95% CI':>17s} {'+ve':>6s} {'p':>8s}")
        for r in ws:
            if r.get("component") not in ("total", "procedural", "data_exposure",
                                          "procedural_within_fold"):
                continue
            ci = f"[{r['rho_raw_ci_lo']:+.3f},{r['rho_raw_ci_hi']:+.3f}]"
            pci = (f"[{r['rho_partial_mu_ci_lo']:+.3f},"
                   f"{r['rho_partial_mu_ci_hi']:+.3f}]"
                   if "rho_partial_mu_ci_lo" in r else "")
            pm = (f"{r['rho_partial_mu_mean']:+.3f}"
                  if "rho_partial_mu_mean" in r else "")
            pp = (f"{r['wilcoxon_p_partial_mu']:.4f}"
                  if r.get("wilcoxon_p_partial_mu") == r.get("wilcoxon_p_partial_mu")
                  and "wilcoxon_p_partial_mu" in r else "")
            print(f"{r['descriptor'][:22]:22s} {str(r.get('component',''))[:12]:>12s} "
                  f"{r['rho_raw_mean']:>+7.3f} {ci:>17s} "
                  f"{r['n_positive_raw']:>3d}/{r['n_slides']:<2d} {pm:>8s} "
                  f"{pci:>17s} "
                  f"{r.get('n_positive_partial_mu', 0):>3d}/{r['n_slides']:<2d} {pp:>8s}")

    if rc:
        print("\n--- what does the uncertainty buy? (selective prediction) ---")
        print("Discard the least certain regions, measure the error on what "
              "remains. Random\nselection is unbiased, so 0% is exactly what "
              "chance gives; the oracle column is\nthe ceiling, and the fraction "
              "of it reached is how far this is from solved.\n")
        df = pd.DataFrame(rc)
        for (d, c), g in df.groupby(["descriptor", "component"], sort=False):
            if c not in ("total", "procedural"):
                continue
            print(f"{d}  ({c})")
            print(f"   {'keep':>6s} {'MAE':>9s} {'vs all':>18s} {'oracle':>8s} "
                  f"{'captured':>9s}")
            for r in g.sort_values("coverage", ascending=False).to_dict("records"):
                ci = (f"[{r['rel_ci_lo']:+.1%},{r['rel_ci_hi']:+.1%}]"
                      if "rel_ci_lo" in r and np.isfinite(r["rel_ci_lo"]) else "")
                cap = (f"{r['capture_of_oracle']:>8.0%}"
                       if np.isfinite(r.get("capture_of_oracle", np.nan)) else "     n/a")
                print(f"   {r['coverage']:>6.0%} {r['mae']:>9.4f} "
                      f"{r['rel_change']:>+7.1%} {ci:>18s} "
                      f"{r['rel_change_oracle']:>+8.1%} {cap}")
            print()

    if agreement:
        print("\n--- do the subsets agree? ---")
        print("Five subsets are five estimates of one quantity, so their SPREAD "
              "is the evidence.\nA descriptor whose rho changes sign between "
              "subsets has not been shown to calibrate,\nhowever tight a pooled "
              "interval would look — and pooling is anti-conservative here,\n"
              "since every region enters five times against one shared target.\n")
        print(f"{'descriptor':24s} {'median':>8s} {'min':>8s} {'max':>8s} "
              f"{'range':>8s}  verdict")
        for a in agreement:
            verdict = ("consistent sign" if a["consistent_sign"]
                       else "SIGN FLIPS between subsets")
            print(f"{a['descriptor']:24s} {a['rho_median']:>+8.3f} "
                  f"{a['rho_min']:>+8.3f} {a['rho_max']:>+8.3f} "
                  f"{a['rho_range']:>8.3f}  {verdict}")

    if any(r.get("component") in ("procedural", "data_exposure") for r in rows):
        print("\nThe three components share ONE prediction, so the error is "
              "identical across\nthem and only sigma changes. Whichever gives "
              "the higher rho is the component\nthe calibration rests on — the "
              "question the crossed 5x10 grid exists to pose.")
    print(f"\nwrote {args.outdir / 'per_region_calibration.csv'}")
    if bin_rows:
        print(f"wrote {args.outdir / 'reliability_bins.csv'}  "
              f"({len(bin_rows)} bins — the reliability diagram's own data)")
    print(f"wrote {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
