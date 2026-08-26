"""Is the 0.80 sigma reference line justified, or is it "Gaussian assumed"? (W-2)

`4_evaluation.tex:120` derives the calibrated line from the assumption that the
error is "symmetric with scale sigma", giving E|e| = sigma*sqrt(2/pi) ~ 0.80
sigma. **Symmetry does not give that factor** — it is Gaussian-specific. A
Laplace error of the same scale sits at 0.707, a Student-t with 5 df at 0.735, a
uniform one at 0.866. The factor sets both reference lines (0.80 in descriptor
space, 0.46 at pixel scale), the whole E|z| column of Table 2, `fig:reliability`,
`fig:kidney` and every supplement scale ratio, so it is worth an hour to find out
whether it is true here.

Why the obvious check does not answer it
----------------------------------------
Writing r for the signed residual,

    E|z| = E|r| / sigma = [ E|r| / sd(r) ]  x  [ sd(r) / sigma ]
                            ^ SHAPE            ^ SCALE (the calibration)

Comparing the measured E|e|/sigma against 0.80 tests the product, and the paper
currently attributes every deviation to the scale term. The shape term has to be
estimated on its own, which is what this does: standardise by sigma first, so any
constant miscalibration divides out, then ask only about the shape of what is
left.

    r  = mu - real            signed, per region
    u  = r / sigma            standardised, scale-invariant

    kappa_uncentred = mean|u| / sqrt(mean(u^2))
    kappa_centred   = mean|u - mean(u)| / sd(u, ddof=1)

**Computed inside each slide, then combined over slides.** Pooling regions across
cases of different fibrosis stage mixes scales, and a mixture of Gaussians with
different variances is leptokurtic even when every component is Gaussian — which
would drag kappa down and manufacture a "heavy tails" verdict out of nothing. The
pooled value is reported too, precisely so that gap is visible rather than
assumed; it is not the answer.

One confound the spec does not guard against, and it is large
--------------------------------------------------------------
Standardising by sigma buys immunity to a **constant** miscalibration. It buys
nothing against a sigma that varies *within* a slide without tracking the local
error scale: u is then a scale mixture, and a scale mixture is leptokurtic
whatever the error's own shape. Measured on Gaussian errors throughout, with cv
the within-slide coefficient of variation of the mis-tracking:

    cv = 0.2  ->  kappa 0.782
    cv = 0.4  ->  kappa 0.742      <- reads as Student-t
    cv = 0.6  ->  kappa 0.701      <- reads as Laplace

That is the entire distance from Gaussian to Laplace, manufactured by sigma
alone. So the run reports `cv_sd` (the within-slide spread of sigma) and
`kappa_centred_raw` (the same statistic on the UNSTANDARDISED residual) beside
every estimate. Neither alone identifies the cause; together they bracket it, and
a small gap between the two means standardising did not move the answer.

**kappa on u remains the correct multiplier for the reliability line either
way**, because that line is drawn in exactly those coordinates. What the gap
decides is whether section 4 may additionally claim the errors are non-Gaussian,
or only that 0.80 is the wrong slope.

Reference values for `kappa`, centred:

    Laplace            0.7071      = 1/sqrt(2)
    Student-t, 5 df    0.7352
    GAUSSIAN           0.7979      = sqrt(2/pi), the value the paper assumes
    Uniform            0.8660      = sqrt(3)/2

What the answer changes
-----------------------
Near 0.798: add "assuming Gaussian errors, supported at kappa-hat = ..." to
section 4 and a line to Limitations, and nothing else moves. Materially
different: both reference lines become kappa-hat*sigma and kappa-hat*sigma/sqrt3,
every E|z| renormalises, and three figures and two tables regenerate. The run
prints the renormalised column either way so the size of the rework is known
before anyone starts it. **Ranking claims are unaffected — rho is scale-free.**

Input
-----
`per_region_calibration.csv` (calibrate_phi.py) or `per_region_sources.csv`
(compare_uncertainty_sources.py). Both carry `mu` and `real` as separate columns,
so the SIGNED residual is recoverable even though only its absolute value is
stored as `error` — which is the one fact that makes W-2 a re-read rather than a
re-run. No masks, no `--phi_csv`, no reference directory.

**Not runnable on the pixel protocol.** `evaluation.py:633` computes
`torch.abs(x - x').mean(dim=1)` before saving, so the sign is destroyed at source
and the channel mean with it; `plot_pixel_reliability.py` then stores a per-tile
mean of those absolutes and no `real` column at all. The pixel arm needs a change
there and a re-run of the regen-error stage, and this script says so rather than
quietly scoring something else.

Outputs
-------
shape_factor.csv          per slide x component: both kappas, the bias share, n
shape_factor_summary.csv  per component: kappa with a case-clustered 95% CI, the
                          verdict against the four reference distributions, and
                          the renormalised E|z| column
summary.json              all of the above plus the acceptance checks
shape_factor.png          per-slide kappa against the four reference lines
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# E|X| / sqrt(E[X^2]) for a centred variable. The Gaussian value is the one the
# manuscript assumes; the others are what it would be if the assumption is wrong
# in the directions anyone would suspect for a residual.
REFERENCE_SHAPES = [
    ("Laplace",         float(1.0 / np.sqrt(2.0))),
    ("Student-t, 5 df", 0.735246),   # 2*sqrt(5)*G(3)/(4*sqrt(pi)*G(2.5)) / sqrt(5/3)
    ("Gaussian",        float(np.sqrt(2.0 / np.pi))),
    ("Uniform",         float(np.sqrt(3.0) / 2.0)),
]
GAUSSIAN = float(np.sqrt(2.0 / np.pi))

# The manuscript's own numbers, at the precision of the comment in
# sec/7_results.tex:70 rather than the rounded table. `calibration_ratio` is
# what calibrate_phi.py:308 computes, so it is the tightest available check that
# this script is reading the same table the paper was built from.
PUBLISHED = {
    "liver": {
        "calibration_ratio": {"total": 0.7135, "procedural": 1.0128,
                              "data_exposure": 1.0972},
        "n_regions_total": 2844,
        "bias_share_pooled": 0.49,
        "n_slides_under_predicting": 14,
        "n_slides": 20,
    },
    "kidney": {
        "calibration_ratio": {"total": 2.07},
        "n_regions_total": 1725,
        "bias_share_pooled": 0.98,
        "n_slides_under_predicting": 20,
        "n_slides": 20,
    },
}


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

def shape_uncentred(u: np.ndarray) -> float:
    """mean|u| / sqrt(mean(u^2)) — the shape INCLUDING any level offset.

    Approaches 1 as a constant offset comes to dominate, because a constant has
    mean|.| equal to its own root mean square. So a value near 1 is not a
    platykurtic error, it is a biased one, and the centred figure beside it is
    what to read.
    """
    d = float(np.sqrt(np.mean(u ** 2)))
    return float(np.mean(np.abs(u)) / d) if d > 0 else float("nan")


def shape_centred(u: np.ndarray) -> float:
    """mean|u - mean u| / sd(u, ddof=1) — the shape of the SPREAD alone.

    This is the number the reference line question turns on: the 0.80 comes from
    a distributional assumption about how the error scatters, not about where it
    is centred, and a slide-level offset is a bias finding rather than a shape
    finding.

    ddof=1 in the denominator, matching the spec, and note the asymmetry with
    `shape_uncentred`, whose second moment is the population one. On a slide of
    ~140 regions the difference is ~0.4%, which is why the two can cross by a
    hair when the offset is genuinely zero — see `check_ordering`.
    """
    if len(u) < 2:
        return float("nan")
    s = float(np.std(u, ddof=1))
    return float(np.mean(np.abs(u - np.mean(u))) / s) if s > 0 else float("nan")


def bias_share(r: np.ndarray) -> float:
    """|mean r| / mean|r| — the share of the mean absolute error that is a level
    offset rather than scatter.

    The `tab:ood` row "share of mean e that is a constant offset": 49% on liver,
    98% on kidney. It is a property of the residual alone, so it does NOT depend
    on which variance component is in the denominator — the run checks that it
    comes out constant across components, which is a direct test that `mu` and
    `real` were pulled correctly.
    """
    d = float(np.mean(np.abs(r)))
    return float(abs(np.mean(r)) / d) if d > 0 else float("nan")


def per_slide(g: pd.DataFrame, min_regions: int) -> List[dict]:
    """Both kappas, the bias share and the sign, one row per slide."""
    out = []
    for wsi, s in g.groupby("wsi", sort=True):
        s = s[np.isfinite(s["r"]) & np.isfinite(s["sd"]) & (s["sd"] > 0)]
        if len(s) < min_regions:
            continue
        r = s["r"].to_numpy(np.float64)
        sd = s["sd"].to_numpy(np.float64)
        u = r / sd
        out.append({
            "wsi": str(wsi), "n_regions": int(len(s)),
            "kappa_centred": shape_centred(u),
            "kappa_uncentred": shape_uncentred(u),
            # The confound diagnostics. See `shape_centred`'s note: standardising
            # by a sigma that varies within the slide turns u into a scale
            # mixture, which is leptokurtic whatever the error's own shape.
            # kappa on the RAW residual is the same statistic without that step,
            # and cv_sd says whether the step could have mattered at all.
            "kappa_centred_raw": shape_centred(r),
            "cv_sd": float(np.std(sd, ddof=1) / np.mean(sd)) if np.mean(sd) > 0
                     else float("nan"),
            "bias_share": bias_share(r),
            "mean_u": float(np.mean(u)), "sd_u": float(np.std(u, ddof=1)),
            "mean_abs_z": float(np.mean(np.abs(u))),
            "mean_r": float(np.mean(r)),
            # mu < real. "Cases where the model under-predicts", 14/20 on liver.
            "under_predicts": bool(np.mean(r) < 0),
        })
    return out


def combine(vals: Sequence[float], n_boot: int, seed: int) -> dict:
    """Mean over slides with a 95% CI from resampling WHOLE SLIDES.

    The unit of replication is the case, matching every other interval in the
    study. With twenty slides the interval is wide, and that is the honest
    width — a per-region interval here would be a statement about 2844
    spatially-correlated regions pretending to be independent.
    """
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    out = {"mean": float(np.mean(v)) if len(v) else float("nan"),
           "sd_across_slides": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
           "min": float(np.min(v)) if len(v) else float("nan"),
           "max": float(np.max(v)) if len(v) else float("nan"),
           "n_slides": int(len(v))}
    if n_boot and len(v) >= 3:
        rng = np.random.default_rng(seed)
        draws = [float(np.mean(rng.choice(v, len(v), replace=True)))
                 for _ in range(n_boot)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    return out


def verdict(k: dict) -> dict:
    """Which reference distribution the estimate is consistent with.

    Reported as "consistent with" rather than "is", because a CI covering two
    reference values does not choose between them — and on twenty cases it
    usually will. The decision the manuscript needs is narrower than
    identification anyway: does the interval cover the Gaussian value.
    """
    m, lo, hi = k.get("mean"), k.get("ci_lo"), k.get("ci_hi")
    if not np.isfinite(m):
        return {"nearest": None, "covers": [], "gaussian_covered": None}
    nearest = min(REFERENCE_SHAPES, key=lambda x: abs(x[1] - m))[0]
    covers = ([n for n, v in REFERENCE_SHAPES if lo <= v <= hi]
              if lo is not None else [])
    return {
        "nearest": nearest,
        "covers": covers,
        "gaussian_covered": (None if lo is None else bool(lo <= GAUSSIAN <= hi)),
        "distance_from_gaussian": float(m - GAUSSIAN),
        "relative_error_of_0_80_line": float(GAUSSIAN / m - 1.0) if m > 0 else float("nan"),
    }


# Under the null — u Gaussian, no offset — the per-slide difference
# (uncentred - centred) has mean ~0.58/n and SD ~0.58/n, so it is NEGATIVE about
# 17% of the time at every n. Measured on 4000 draws at n = 50, 150 and 500.
# The spec's "always" is therefore a statement about the population, not about
# individual slides, and a per-slide check with a tight tolerance fails on
# perfectly well-behaved data.
NULL_CROSSING_RATE = 0.166
_ORDER_SLACK_K = 1.75          # 3 x the null SD of the difference, i.e. 3 x 0.58/n


def check_ordering(rows: List[dict], summ: Optional[List[dict]] = None) -> dict:
    """Acceptance check 1: kappa_uncentred >= kappa_centred.

    **True of the means, not of every slide.** The two denominators differ in
    ddof and the numerators differ in where they are centred, and the resulting
    per-slide difference is itself noisy: under a pure Gaussian null it comes out
    negative for about one slide in six, at any n. Failing the run on that would
    reject correct data six times out of ten on a twenty-case cohort.

    So the pass/fail rests on the COMPONENT MEANS, which is where the ordering
    is a real property, and the per-slide crossings are reported as a rate
    against the ~17% expected under the null. A rate far above that, or a
    negative deficit in the means, is the signal the spec was reaching for: the
    standardisation or the slide grouping is wrong.
    """
    bad = []
    n_checked = 0
    for r in rows:
        c, u, n = r["kappa_centred"], r["kappa_uncentred"], r["n_regions"]
        if not (np.isfinite(c) and np.isfinite(u)):
            continue
        n_checked += 1
        if u < c - _ORDER_SLACK_K / max(1, n):
            bad.append({"wsi": r["wsi"], "component": r.get("component"),
                        "kappa_centred": c, "kappa_uncentred": u,
                        "deficit": float(c - u),
                        "slack": float(_ORDER_SLACK_K / max(1, n))})
    means = []
    for s in (summ or []):
        d = s["kappa_uncentred_mean"] - s["kappa_centred_mean"]
        means.append({"component": s["component"], "deficit_of_means": float(-d),
                      "passes": bool(d >= -1e-12)})
    rate = (len(bad) / n_checked) if n_checked else float("nan")
    passes = all(m["passes"] for m in means) if means else True
    return {"n_slide_crossings": len(bad), "n_slides_checked": n_checked,
            "crossing_rate": float(rate),
            "null_crossing_rate": NULL_CROSSING_RATE,
            "crossings_beyond_null": bool(np.isfinite(rate)
                                          and rate > 3 * NULL_CROSSING_RATE),
            "violations": bad, "by_component_mean": means, "passes": bool(passes),
            "note": ("the ordering holds in the means; per-slide crossings at "
                     f"~{NULL_CROSSING_RATE:.0%} are the Gaussian null"
                     if passes else
                     "VIOLATED IN THE MEANS — resolve before reading anything "
                     "else in this run")}


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def make_figure(rows: List[dict], summ: List[dict], outpath: Path,
                descriptor: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([r for r in rows if r.get("descriptor") == descriptor])
    if df.empty:
        return
    comps = list(dict.fromkeys(df["component"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, field, title in zip(
            axes, ("kappa_centred", "kappa_uncentred"),
            ("centred — the shape of the spread",
             "uncentred — shape plus level offset")):
        for name, v in REFERENCE_SHAPES:
            ax.axhline(v, color="#b3b1ac", linewidth=0.8,
                       linestyle="-" if name == "Gaussian" else (0, (3, 2)),
                       zorder=1)
            ax.text(len(comps) - 0.42, v, f" {name}", va="center", fontsize=6.5,
                    color="#52514e" if name != "Gaussian" else "#0b0b0b")
        for i, comp in enumerate(comps):
            v = df[df["component"] == comp][field].to_numpy()
            v = v[np.isfinite(v)]
            ax.scatter(i + rng.uniform(-0.11, 0.11, len(v)), v, s=13,
                       color="#2a78d6", alpha=0.55, zorder=4,
                       edgecolor="white", linewidth=0.3)
            s = next((x for x in summ if x["component"] == comp
                      and x["descriptor"] == descriptor), None)
            if s and np.isfinite(s.get(f"{field}_mean", np.nan)):
                lo = s.get(f"{field}_ci_lo", np.nan)
                hi = s.get(f"{field}_ci_hi", np.nan)
                yerr = ([[s[f"{field}_mean"] - lo], [hi - s[f"{field}_mean"]]]
                        if np.isfinite(lo) else None)
                ax.errorbar(i, s[f"{field}_mean"], yerr=yerr, fmt="D",
                            color="#eb6834", markersize=5.5, capsize=3.4,
                            zorder=6, markeredgecolor="white",
                            markeredgewidth=0.5, linewidth=1.2)
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels([c.replace("_", " ") for c in comps], fontsize=8)
        ax.set_xlim(-0.5, len(comps) - 0.5 + 0.55)
        ax.set_title(title, fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("$\\hat{\\kappa} = \\mathbb{E}|u| / \\mathrm{sd}(u)$")
    fig.suptitle(f"{descriptor} — one point per slide, diamond = mean with a "
                 f"case-clustered 95% CI", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="W-2: estimate the shape factor behind the 0.80 sigma line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--table", type=Path, required=True,
                    help="per_region_calibration.csv or per_region_sources.csv. "
                         "Must carry mu AND real — the signed residual is the "
                         "whole point.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--descriptor", default="task_specific_value",
                    help="'all' scores every descriptor present.")
    ap.add_argument("--components", nargs="*", default=None,
                    help="Default: every component present except the synthetic "
                         "point-prediction source, whose sd is mu.")
    ap.add_argument("--prediction", default="grand",
                    help="Filter on the `prediction` column where present; "
                         "'any' keeps every row.")
    ap.add_argument("--min_regions_per_slide", type=int, default=10,
                    help="A slide below this cannot support a shape estimate and "
                         "would carry equal weight in the mean over slides.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", default="liver", choices=["liver", "kidney", "none"],
                    help="Which published block to check against. 'none' skips "
                         "the acceptance checks.")
    args = ap.parse_args()

    t = pd.read_csv(args.table)
    if "real" not in t.columns:
        raise SystemExit(
            f"{args.table} has no 'real' column, so the SIGNED residual cannot be "
            f"recovered and W-2 cannot be answered from it.\n"
            f"  If this is the pixel protocol: evaluation.py:633 computes\n"
            f"  torch.abs(x - x_prime).mean(dim=1) BEFORE saving, so the sign is\n"
            f"  destroyed at source and the per-channel structure with it. The\n"
            f"  pixel arm needs that line changed to save the signed per-channel\n"
            f"  residual and the regen-error stage re-run — it is not a re-read.\n"
            f"  Found columns: {list(t.columns)}")
    for col in ("mu", "sd", "wsi", "component"):
        if col not in t.columns:
            raise SystemExit(f"{args.table} has no '{col}' column. "
                             f"Found: {list(t.columns)}")
    if "descriptor" not in t.columns:
        t["descriptor"] = "unknown"
    if "prediction" in t.columns and args.prediction != "any":
        t = t[t["prediction"] == args.prediction]
        if t.empty:
            raise SystemExit(f"no rows with prediction == '{args.prediction}'")
    if args.descriptor != "all":
        t = t[t["descriptor"] == args.descriptor]
        if t.empty:
            raise SystemExit(f"no rows for descriptor '{args.descriptor}'")

    present = list(dict.fromkeys(t["component"]))
    keep = args.components or [c for c in present if c != "point_prediction"]
    if not args.components and "point_prediction" in present:
        print("[note] dropped the synthetic 'point_prediction' component: its sd "
              "is mu, not a spread")
    t = t[t["component"].isin(keep)].copy()
    if t.empty:
        raise SystemExit(f"no rows for components {keep} (present: {present})")

    # The one line the whole analysis rests on. calibrate_phi.py stores only
    # |mu - real| as `error`; both operands survive, so the sign does too.
    t["r"] = t["mu"] - t["real"]

    rows: List[dict] = []
    summ: List[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for (desc, comp), g in t.groupby(["descriptor", "component"], sort=False):
            g = g.dropna(subset=["r", "sd"])
            g = g[np.isfinite(g["r"]) & np.isfinite(g["sd"])]
            n_dropped = int((g["sd"] <= 0).sum())
            g = g[g["sd"] > 0]
            if g["wsi"].nunique() < 3:
                print(f"[skip] {desc}/{comp}: fewer than 3 slides")
                continue
            slides = per_slide(g, args.min_regions_per_slide)
            for s in slides:
                s.update({"descriptor": desc, "component": comp})
            rows += slides

            u_all = (g["r"] / g["sd"]).to_numpy(np.float64)
            row = {"descriptor": desc, "component": comp,
                   "n_regions": int(len(g)), "n_slides": len(slides),
                   "n_regions_dropped_sd_nonpositive": n_dropped,
                   # Pooled over regions, which calibrate_phi.py:308 also does —
                   # this is the number Table 2's E|z| column reports.
                   "mean_abs_z_pooled": float(np.mean(np.abs(u_all))),
                   "calibration_ratio_pooled": float(np.mean(np.abs(u_all)) / GAUSSIAN),
                   # Deliberately reported and deliberately NOT the answer: a
                   # mixture of per-slide scales is leptokurtic even when every
                   # slide is Gaussian, so this sits below the within-slide
                   # figure and the gap is the size of that artefact.
                   "kappa_centred_pooled": shape_centred(u_all),
                   "kappa_uncentred_pooled": shape_uncentred(u_all),
                   "bias_share_pooled": bias_share(g["r"].to_numpy(np.float64))}
            for field in ("kappa_centred", "kappa_uncentred",
                          "kappa_centred_raw", "cv_sd", "bias_share"):
                c = combine([s[field] for s in slides], args.n_boot, args.seed)
                for k, v in c.items():
                    row[f"{field}_{k}" if k != "mean" else f"{field}_mean"] = v
            row["verdict"] = verdict(
                {"mean": row["kappa_centred_mean"],
                 "ci_lo": row.get("kappa_centred_ci_lo"),
                 "ci_hi": row.get("kappa_centred_ci_hi")})
            # What the reference lines become if the Gaussian value is wrong.
            k = row["kappa_centred_mean"]
            row["implied_line_descriptor_space"] = float(k)
            row["implied_line_pixel_space"] = float(k / np.sqrt(3.0))
            row["renormalised_scale_ratio"] = (
                float(row["mean_abs_z_pooled"] / k) if k > 0 else float("nan"))
            row["n_slides_under_predicting"] = sum(s["under_predicts"] for s in slides)
            summ.append(row)

    if not summ:
        raise SystemExit("nothing scored — check --descriptor and --components")

    ordering = check_ordering(rows, summ)

    # bias_share must not depend on the component: r has no component. A spread
    # here means mu or real was pulled per-component, which would invalidate
    # every kappa above.
    bs = {s["component"]: s["bias_share_pooled"] for s in summ}
    bs_spread = (float(np.ptp(list(bs.values()))) if len(bs) > 1 else 0.0)
    invariance = {"bias_share_by_component": bs, "spread": bs_spread,
                  "passes": bool(bs_spread < 1e-9),
                  "note": ("bias share is a property of the residual alone, so a "
                           "non-zero spread means mu or real differs by "
                           "component — which the grand prediction forbids")}

    checks: Optional[dict] = None
    if args.cohort != "none":
        p = PUBLISHED[args.cohort]
        by = {s["component"]: s for s in summ}
        ratio = []
        for comp, want in p["calibration_ratio"].items():
            got = by.get(comp, {}).get("calibration_ratio_pooled")
            ratio.append({"component": comp, "published": want,
                          "computed": None if got is None else float(got),
                          "matches": (None if got is None
                                      else bool(abs(got - want) <= 0.005))})
        tot = by.get("total", {})
        checks = {
            "cohort": args.cohort,
            "calibration_ratio": ratio,
            "n_regions": {"published": p["n_regions_total"],
                          "computed": tot.get("n_regions"),
                          "matches": tot.get("n_regions") == p["n_regions_total"]},
            "bias_share_pooled": {
                "published": p["bias_share_pooled"],
                "computed": tot.get("bias_share_pooled"),
                # The table rounds to a whole percent, so 0.005 is the tolerance
                # the manuscript's own precision allows.
                "matches": (None if tot.get("bias_share_pooled") is None else
                            bool(abs(tot["bias_share_pooled"]
                                     - p["bias_share_pooled"]) <= 0.005))},
            "bias_share_mean_over_slides": tot.get("bias_share_mean"),
            "n_slides_under_predicting": {
                "published": p["n_slides_under_predicting"],
                "computed": tot.get("n_slides_under_predicting"),
                "matches": (tot.get("n_slides_under_predicting")
                            == p["n_slides_under_predicting"])},
            "note": ("The published bias share is pooled over regions. The mean "
                     "over slides is reported beside it because the two answer "
                     "slightly different questions and the spec did not say "
                     "which the row is — if the pooled one matches, the row is "
                     "pooled."),
        }

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir / "shape_factor.csv", index=False)
    pd.DataFrame(summ).to_csv(args.outdir / "shape_factor_summary.csv", index=False)
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump({"table": str(args.table), "descriptor": args.descriptor,
                   "components": keep, "prediction": args.prediction,
                   "min_regions_per_slide": args.min_regions_per_slide,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "reference_shapes": dict(REFERENCE_SHAPES),
                   "gaussian": GAUSSIAN,
                   "per_slide": rows, "per_component": summ,
                   "check_ordering": ordering,
                   "check_bias_share_invariance": invariance,
                   "published_checks": checks}, fh, indent=2)

    for desc in dict.fromkeys(s["descriptor"] for s in summ):
        make_figure(rows, summ, args.outdir / f"shape_factor_{desc}.png", desc)

    # ---- what the run says ----
    print(f"\n=== kappa: the SHAPE of the standardised residual u = (mu - real)/sigma ===")
    print("Within slide, then combined over slides. Reference values: "
          "Laplace 0.707,\nStudent-t(5) 0.735, GAUSSIAN 0.798, uniform 0.866.")
    print(f"\n{'component':>15s} {'kappa_cent':>11s} {'95% CI':>17s} "
          f"{'kappa_unc':>10s} {'bias':>6s} {'pooled':>8s} {'nearest':>16s}")
    for s in summ:
        ci = (f"[{s['kappa_centred_ci_lo']:.3f},{s['kappa_centred_ci_hi']:.3f}]"
              if "kappa_centred_ci_lo" in s else "")
        print(f"{s['component']:>15s} {s['kappa_centred_mean']:11.3f} {ci:>17s} "
              f"{s['kappa_uncentred_mean']:10.3f} {s['bias_share_mean']:6.2f} "
              f"{s['kappa_centred_pooled']:8.3f} {str(s['verdict']['nearest']):>16s}")
    print("\n'pooled' is the same statistic computed over all regions at once. It "
          "sits BELOW\nthe within-slide figure because a mixture of per-slide "
          "scales is leptokurtic even\nwhen every slide is Gaussian — which is "
          "why the answer is the within-slide column.")

    print(f"\n--- is a low kappa heavy tails, or a sigma that mis-tracks? ---")
    print("Standardising by sigma makes kappa immune to a CONSTANT "
          "miscalibration, which is\nwhat the spec relies on. It does NOT make "
          "it immune to a sigma that varies\nwithin a slide without tracking the "
          "local error scale: u is then a scale mixture,\nand a scale mixture is "
          "leptokurtic whatever the error's own shape. Measured, with\nGaussian "
          "errors throughout: cv 0.2 -> kappa 0.782, cv 0.4 -> 0.742, cv 0.6 -> "
          "0.701.\nThat is the whole distance from Gaussian to Laplace, "
          "manufactured by sigma alone.")
    print(f"\n{'component':>15s} {'kappa(u)':>9s} {'kappa(r)':>9s} {'gap':>7s} "
          f"{'cv(sigma)':>10s}  reading")
    for s_ in summ:
        gap = s_["kappa_centred_mean"] - s_["kappa_centred_raw_mean"]
        cv = s_["cv_sd_mean"]
        if cv < 0.05:
            read = "sigma is flat in-slide; kappa is the error's own shape"
        elif abs(gap) < 0.01:
            read = "standardising barely moved it; shape reading holds"
        else:
            read = "standardising moved it — do NOT name a distribution"
        print(f"{s_['component']:>15s} {s_['kappa_centred_mean']:9.3f} "
              f"{s_['kappa_centred_raw_mean']:9.3f} {gap:+7.3f} {cv:10.3f}  {read}")
    print("\nEither way kappa(u) is the CORRECT MULTIPLIER for the reliability "
          "line, which is\ndrawn in exactly those coordinates. What the gap "
          "governs is whether section 4 may\nalso say the errors are "
          "non-Gaussian, or only that 0.80 is the wrong slope.")

    print(f"\n--- what this does to the reference lines ---")
    for s in summ:
        v = s["verdict"]
        cov = ("covers the Gaussian value" if v["gaussian_covered"]
               else "EXCLUDES the Gaussian value" if v["gaussian_covered"] is False
               else "no interval")
        print(f"  {s['component']:>15s}: {cov}. 0.80 would be off by "
              f"{v['relative_error_of_0_80_line'] * 100:+.1f}%; the line would "
              f"become {s['implied_line_descriptor_space']:.3f} sigma "
              f"({s['implied_line_pixel_space']:.3f} at pixel scale), and E|z| "
              f"{s['mean_abs_z_pooled'] / GAUSSIAN:.2f} -> "
              f"{s['renormalised_scale_ratio']:.2f}")
    if all(s["verdict"]["gaussian_covered"] for s in summ
           if s["verdict"]["gaussian_covered"] is not None):
        print("\nEvery interval covers 0.798. The manuscript keeps its lines and "
              "adds\n\"assuming Gaussian errors, supported at kappa-hat = ...\" "
              "to section 4 plus a\nline to Limitations. Nothing else moves.")
    else:
        print("\nAt least one interval excludes 0.798. Both reference lines "
              "become kappa-hat*sigma\nand kappa-hat*sigma/sqrt(3), every E|z| "
              "renormalises, and three figures and two\ntables regenerate. "
              "Ranking claims are unaffected — rho is scale-free.")

    print(f"\n--- acceptance check 1: kappa_uncentred >= kappa_centred ---")
    print(f"  {'PASS' if ordering['passes'] else 'FAIL'} — judged on the "
          f"component means, where the ordering is a real property.")
    for m in ordering["by_component_mean"]:
        print(f"    {m['component']:>15s}: deficit "
              f"{m['deficit_of_means']:+.5f} {'ok' if m['passes'] else 'NEGATIVE'}")
    print(f"  per-slide crossings {ordering['n_slide_crossings']}/"
          f"{ordering['n_slides_checked']} "
          f"({ordering['crossing_rate']:.0%}); the Gaussian null is "
          f"~{ordering['null_crossing_rate']:.0%}, so this is "
          f"{'ABOVE expectation' if ordering['crossings_beyond_null'] else 'unremarkable'}")

    print(f"\n--- bias share is component-invariant ---")
    print(f"  {'PASS' if invariance['passes'] else 'FAIL'} — spread "
          f"{invariance['spread']:.2e} across {len(bs)} component(s)")

    if checks:
        print(f"\n--- acceptance checks against the published {args.cohort} numbers ---")
        for c in checks["calibration_ratio"]:
            ok = "-" if c["matches"] is None else ("yes" if c["matches"] else "NO")
            got = "n/a" if c["computed"] is None else f"{c['computed']:.4f}"
            print(f"  E|z|/0.80  {c['component']:>15s}  published "
                  f"{c['published']:.4f}  computed {got:>7s}  {ok}")
        for key in ("n_regions", "bias_share_pooled", "n_slides_under_predicting"):
            c = checks[key]
            ok = "-" if c["matches"] is None else ("yes" if c["matches"] else "NO")
            got = c["computed"]
            got = ("n/a" if got is None else
                   f"{got:.4f}" if isinstance(got, float) else str(got))
            print(f"  {key:>28s}  published {c['published']}  "
                  f"computed {got}  {ok}")
        print(f"  {'bias share, mean over slides':>28s}  "
              f"{checks['bias_share_mean_over_slides']:.4f}  "
              f"(the published row is pooled — see summary.json)")
        bad = [c["component"] for c in checks["calibration_ratio"]
               if c["matches"] is False]
        if bad or checks["n_regions"]["matches"] is False:
            print("\n[warn] the table does not reproduce the published numbers. "
                  "Resolve that\n       first: it means this is not the run "
                  "behind Table 2.")

    print(f"\nwrote {args.outdir}/shape_factor.csv, shape_factor_summary.csv "
          f"and summary.json")


if __name__ == "__main__":
    main()
