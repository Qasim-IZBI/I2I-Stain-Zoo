"""Does the spread add anything to the point prediction? (W-30)

`calibrate_phi.py`'s `risk_coverage()` ranks regions by sigma and nothing else,
so the paper can say what sigma buys but not whether it buys anything *beyond*
mu. That gap matters because the manuscript already concedes the uncomfortable
half of the answer: on this cohort ranking by the point prediction alone beats
ranking by sigma, and it needs no ensemble to do it. Section 8 nevertheless
claims the two "carry different information and are worth reading together",
resting on a partialled correlation, which is indirect evidence for an operational
claim.

This makes the comparison direct. Same regions, same coverage grid, same
case-clustered bootstrap; the only thing that changes is the score the regions
are ranked by.

    rule            score                       needs an ensemble?
    -----------------------------------------------------------------
    random          -                           no   (exactly flat)
    mu              mu                          no   (the baseline that wins)
    sd              sd                          yes  (the existing curve)
    mu_rank         within-slide rank of mu     no   (matched baseline, see below)
    sd_rank         within-slide rank of sd     yes  (matched baseline)
    ranksum         rank(mu) + rank(sd)         yes  (no fitting to overfit)
    fitted          E[rank e | mu, sd], LOCO    yes  (cross-validated by case)
    oracle          the true error              -    (the ceiling)

**The claim is the paired difference, not two point estimates.** Reporting
"combined gives -16.4%" beside "mu gives -15.1%" with overlapping intervals
settles nothing, so every bootstrap replicate evaluates *all* rules on the *same*
resampled cohort and the difference is taken inside the replicate. That is what
`rank_rule_deltas.csv` holds.

Three things that are easy to get wrong and are handled here:

* **The fitted rule is cross-validated by case.** Fitting error on (mu, sd) and
  then ranking the same regions is optimistic and a reviewer will say so. One
  case is held out, the model is fit on the other nineteen, the held-out case is
  scored, and it rotates. The features are within-slide rank transforms, which
  touch no target, so the held-out slide's own errors never enter its score.
* **Matched baselines.** `ranksum` and `fitted` are within-slide-normalised by
  construction, so comparing them against a `mu` that is ranked on raw values
  pooled across slides confounds the combination with the normalisation.
  `mu_rank` and `sd_rank` are the same rules under the same normalisation, and
  the honest read of the combined rules is against those.
* **Two scopes, because the paper reports two.** `pooled` sorts every region
  together; `within` discards a fixed fraction inside each slide and weights
  slides equally, which is the scope that matches "the slide is the unit of
  replication". `calibrate_phi.py` only ever computed the pooled one.

Input is the flat per-region table either calibration script writes — the same
object `risk_coverage()` already consumes:

    compare_uncertainty_sources.py  ->  per_region_sources.csv   (what the paper's
                                        risk_coverage.pdf is built from)
    calibrate_phi.py                ->  per_region_calibration.csv

Both carry `mu`, `sd`, `error`, `wsi`, `component`, `descriptor`. Nothing else is
read: no masks, no `--phi_csv`, no reference.

Outputs
-------
rank_rules.csv        one row per component x scope x rule x coverage: the curve,
                      with the oracle ceiling and a case-clustered CI
rank_rule_deltas.csv  the paired difference against a reference rule, per
                      replicate — the actual W-30 claim
fitted_coefficients.csv  the LOCO fit per held-out case, so the combination is
                      inspectable rather than a black box
summary.json          provenance, the reconciliation block, everything above
rank_rules.png        the curves, one panel per scope
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# The order rules are reported and drawn in. Label, colour, line style, marker —
# style and marker carry the identity in greyscale, as in make_paper_figures.py.
RULES = [
    ("mu",      "$\\mu$ alone",              "#4a3aa7", (0, (3, 1, 1, 1)), "D"),
    ("sd",      "$\\sigma$ alone",           "#2a78d6", "-",               "o"),
    ("mu_rank", "$\\mu$ (slide rank)",       "#7a6fd0", (0, (1, 1.4)),     "v"),
    ("sd_rank", "$\\sigma$ (slide rank)",    "#6fa8e8", (0, (1, 1.4)),     "^"),
    ("ranksum", "$\\mu + \\sigma$ rank sum", "#1baf7a", "--",              "s"),
    ("fitted",  "$\\mu + \\sigma$ fitted",   "#eb6834", "-.",              "P"),
]
RULE_NAMES = [r[0] for r in RULES]

# Components that are not uncertainties and must not be ranked as one.
# `point_prediction` is make_paper_figures.py's synthetic source, whose `sd` IS
# `mu` — leaving it in would duplicate the `mu` rule under another name and
# invite the reader to read a baseline as a result.
SYNTHETIC_COMPONENTS = {"point_prediction"}

# The liver values printed in the manuscript, for --check_published. Kept here
# rather than in the text of a docstring so the comparison is machine-made.
PUBLISHED_LIVER = {
    "mae_all_regions":            0.0397,
    "mae_kept_80_total_sd":       0.0366,
    "mae_random_80":              0.0398,
    "rel_oracle_80":             -0.412,
    "rel_mu_80_pooled":          -0.151,
    "rel_mu_80_within":          -0.159,
    "rel_sd_80_pooled":          -0.078,
    "rel_sd_80_within":          -0.115,
}


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------

def pct_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank in the open interval (0, 1), ties averaged.

    Dividing by n+1 rather than n keeps the top region off exactly 1.0, so a
    rank sum has no atom at its maximum and the fitted design matrix stays
    away from a degenerate column when a slide holds few regions.
    """
    return rankdata(v, method="average") / (len(v) + 1.0)


def by_slide_pct_rank(v: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """`pct_rank` computed inside each slide.

    Within-slide, because the alternative folds between-slide differences into
    the feature: a sham case's regions all carry little collagen, so a pooled
    rank of mu is largely a rank of which slide the region came from. The
    combination is supposed to be about ordering regions for triage, and triage
    happens inside a case.
    """
    out = np.empty(len(v), dtype=np.float64)
    for g in groups:
        out[g] = pct_rank(v[g])
    return out


def loco_fitted_score(mu_r: np.ndarray, sd_r: np.ndarray, err: np.ndarray,
                      groups: Sequence[np.ndarray],
                      slide_ids: Sequence[str]) -> Tuple[np.ndarray, List[dict]]:
    """Leave-one-case-out fitted combination of mu and sd.

    Fits the within-slide rank of the error on the within-slide ranks of mu and
    sd — a rank-based linear model, which is all the claim needs: W-30 asks about
    *ordering*, not about calibrated error magnitudes, and a rank fit cannot be
    dragged around by the handful of high-collagen regions that dominate a
    least-squares fit in raw units.

    **Nothing about a slide's own errors enters its own score.** The two features
    are rank transforms of mu and sd computed inside the slide, which involve no
    target at all; the coefficients come only from the other nineteen cases. So
    the returned score is genuinely out-of-fold and can be ranked against the
    same regions' errors without the optimism a reviewer would object to.

    The rank-sum rule is this model with the coefficients pinned at (1, 1), so
    reading the fitted coefficients tells you directly whether the data wanted a
    different weighting or simply wanted mu.
    """
    score = np.full(len(err), np.nan, dtype=np.float64)
    coefs: List[dict] = []
    if len(groups) < 3:
        # With two cases the "fit" is one case predicting the other, which is a
        # statement about two slides rather than a model. Better to return NaN
        # and have the rule drop out visibly than to publish it.
        return score, coefs
    err_r = by_slide_pct_rank(err, groups)
    for i, g in enumerate(groups):
        train = np.concatenate([h for j, h in enumerate(groups) if j != i])
        X = np.column_stack([mu_r[train], sd_r[train], np.ones(len(train))])
        b, *_ = np.linalg.lstsq(X, err_r[train], rcond=None)
        score[g] = np.column_stack([mu_r[g], sd_r[g], np.ones(len(g))]) @ b
        coefs.append({"held_out_wsi": str(slide_ids[i]), "n_train": int(len(train)),
                      "n_held_out": int(len(g)), "b_mu": float(b[0]),
                      "b_sd": float(b[1]), "intercept": float(b[2])})
    return score, coefs


def build_scores(mu: np.ndarray, sd: np.ndarray, err: np.ndarray,
                 groups: Sequence[np.ndarray],
                 slide_ids: Sequence[str]) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Every ranking rule, as a dict of risk scores — low means keep."""
    mu_r = by_slide_pct_rank(mu, groups)
    sd_r = by_slide_pct_rank(sd, groups)
    fitted, coefs = loco_fitted_score(mu_r, sd_r, err, groups, slide_ids)
    return {
        "mu": mu.astype(np.float64),
        "sd": sd.astype(np.float64),
        "mu_rank": mu_r,
        "sd_rank": sd_r,
        "ranksum": mu_r + sd_r,
        "fitted": fitted,
    }, coefs


# --------------------------------------------------------------------------
# the curves
# --------------------------------------------------------------------------

def _keep_mean(err: np.ndarray, score: np.ndarray, cov: float) -> float:
    """Mean error over the most-certain `cov` fraction, ranked by `score`."""
    k = max(1, int(round(len(err) * cov)))
    return float(err[np.argsort(score, kind="stable")][:k].mean())


def curve_pooled(err: np.ndarray, scores: Dict[str, np.ndarray],
                 coverages: Sequence[float]) -> Dict[str, Dict[float, float]]:
    """Sort every region together and discard the least certain.

    Random selection is unbiased, so its curve is exactly the base MAE at every
    coverage — no Monte-Carlo baseline is needed and the departure from flat is
    the whole effect.
    """
    out: Dict[str, Dict[float, float]] = {}
    for name, s in scores.items():
        if not np.isfinite(s).all():
            continue
        out[name] = {c: _keep_mean(err, s, c) for c in coverages}
    out["oracle"] = {c: _keep_mean(err, err, c) for c in coverages}
    base = float(err.mean())
    out["random"] = {c: base for c in coverages}
    return out


def curve_within(err: np.ndarray, scores: Dict[str, np.ndarray],
                 groups: Sequence[np.ndarray],
                 coverages: Sequence[float]) -> Dict[str, Dict[float, float]]:
    """Discard the same fraction inside every slide, weight slides equally.

    The scope the rest of the paper's statistics use. Pooling instead lets one
    large slide decide the answer, and worse, lets a rule score well by
    discarding whole cases rather than the worst regions of each — which is not
    what a triage rule is for.
    """
    out: Dict[str, Dict[float, float]] = {}
    for name, s in scores.items():
        if not np.isfinite(s).all():
            continue
        out[name] = {c: float(np.mean([_keep_mean(err[g], s[g], c) for g in groups]))
                     for c in coverages}
    out["oracle"] = {c: float(np.mean([_keep_mean(err[g], err[g], c) for g in groups]))
                     for c in coverages}
    base = float(np.mean([err[g].mean() for g in groups]))
    out["random"] = {c: base for c in coverages}
    return out


def evaluate(err: np.ndarray, scores: Dict[str, np.ndarray],
             groups: Sequence[np.ndarray], coverages: Sequence[float],
             scope: str) -> Tuple[Dict[str, Dict[float, float]], float]:
    """One scope's curves plus its own base, which is the random line."""
    if scope == "pooled":
        maes = curve_pooled(err, scores, coverages)
    else:
        maes = curve_within(err, scores, groups, coverages)
    return maes, maes["random"][coverages[0]]


# --------------------------------------------------------------------------
# case-clustered bootstrap, paired across rules
# --------------------------------------------------------------------------

def bootstrap(err: np.ndarray, scores: Dict[str, np.ndarray],
              idx: Dict[str, np.ndarray], slide_ids: Sequence[str],
              coverages: Sequence[float], scope: str, n_boot: int,
              seed: int) -> List[Dict[str, Dict[float, float]]]:
    """Resample WHOLE SLIDES and re-evaluate every rule on each replicate.

    Regions inside a slide are spatially correlated and share one case's biology,
    so an interval built by resampling regions describes a cohort of twenty cases
    as if it held two thousand. A slide drawn twice contributes its regions
    twice, which is the point.

    All rules are evaluated on the *same* replicate, so differences between them
    can be taken inside it. That pairing is what makes the W-30 claim testable:
    two marginal intervals that overlap say nothing about the difference.

    The fitted rule's out-of-fold scores are computed once on the full cohort and
    then carried through the resampling rather than refit inside each replicate.
    The optimism W-30 is worried about is handled by the leave-one-case-out
    split, not by the bootstrap, and refitting 20 x n_boot times to widen the
    interval by a little would cost hours for no change in the verdict.
    """
    rng = np.random.default_rng(seed)
    uniq = np.asarray(slide_ids)
    draws: List[Dict[str, Dict[float, float]]] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        if scope == "pooled":
            sel = np.concatenate([idx[x] for x in pick])
            maes = curve_pooled(err[sel], {k: v[sel] for k, v in scores.items()},
                                coverages)
        else:
            # Each *draw* of a slide is its own group: a slide drawn twice is two
            # groups, matching how the pooled arm counts it twice.
            g = [idx[x] for x in pick]
            maes = curve_within(err, scores, g, coverages)
        base = maes["random"][coverages[0]]
        if base <= 0:
            continue
        draws.append({r: {c: v / base - 1.0 for c, v in cur.items()}
                      for r, cur in maes.items()})
    return draws


def ci(vals: Sequence[float]) -> Tuple[float, float]:
    lo, hi = np.percentile(np.asarray(vals, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def run_component(g: pd.DataFrame, coverages: Sequence[float], n_boot: int,
                  seed: int, reference_rule: str,
                  min_regions_per_slide: int) -> Tuple[List[dict], List[dict], List[dict]]:
    """Every scope, rule and coverage for one (descriptor, component)."""
    g = g.dropna(subset=["mu", "sd", "error", "wsi"])
    g = g[np.isfinite(g["mu"]) & np.isfinite(g["sd"]) & np.isfinite(g["error"])]
    # A slide with a handful of regions cannot support a within-slide rank, and
    # in the within scope it would carry the same weight as a slide with two
    # hundred. Dropped from both scopes, so the two are computed on one cohort.
    counts = g.groupby("wsi")["error"].transform("size")
    g = g[counts >= min_regions_per_slide]
    if g["wsi"].nunique() < 3 or len(g) < 20:
        return [], [], []

    mu = g["mu"].to_numpy(np.float64)
    sd = g["sd"].to_numpy(np.float64)
    err = g["error"].to_numpy(np.float64)
    wsi = g["wsi"].to_numpy()
    slide_ids = list(pd.unique(wsi))
    idx = {s: np.where(wsi == s)[0] for s in slide_ids}
    groups = [idx[s] for s in slide_ids]

    scores, coefs = build_scores(mu, sd, err, groups, slide_ids)
    meta = {k: g[k].iloc[0] for k in ("descriptor", "component") if k in g.columns}
    for c in coefs:
        c.update(meta)

    rows: List[dict] = []
    deltas: List[dict] = []
    for scope in ("pooled", "within"):
        maes, base = evaluate(err, scores, groups, coverages, scope)
        draws = bootstrap(err, scores, idx, slide_ids, coverages, scope,
                          n_boot, seed) if n_boot else []
        for rule, cur in maes.items():
            for c in coverages:
                rel = cur[c] / base - 1.0 if base > 0 else float("nan")
                orel = maes["oracle"][c] / base - 1.0 if base > 0 else float("nan")
                row = {**meta, "scope": scope, "rule": rule, "coverage": float(c),
                       "n": int(len(g)), "n_slides": len(slide_ids),
                       "n_kept": int(max(1, round(len(g) * c))),
                       "mae": cur[c], "mae_random": base,
                       "mae_oracle": maes["oracle"][c],
                       "rel_change": float(rel),
                       "rel_change_oracle": float(orel),
                       # At full coverage both gains are zero and the ratio is
                       # 0/0, which floating point renders as a confident 100%.
                       "capture_of_oracle": (float(rel / orel) if orel < -1e-9
                                             else float("nan"))}
                d = [x[rule][c] for x in draws if rule in x]
                if len(d) >= max(20, n_boot // 10):
                    row["rel_ci_lo"], row["rel_ci_hi"] = ci(d)
                    row["n_boot_used"] = len(d)
                rows.append(row)

        # The claim: paired difference against the reference rule, taken inside
        # each replicate so the two curves share a cohort.
        if reference_rule in maes:
            for rule in maes:
                if rule == reference_rule:
                    continue
                for c in coverages:
                    obs = ((maes[rule][c] - maes[reference_rule][c]) / base
                           if base > 0 else float("nan"))
                    d = [x[rule][c] - x[reference_rule][c]
                         for x in draws if rule in x and reference_rule in x]
                    row = {**meta, "scope": scope, "rule": rule,
                           "reference_rule": reference_rule,
                           "coverage": float(c),
                           "delta_rel_change": float(obs)}
                    if len(d) >= max(20, n_boot // 10):
                        row["delta_ci_lo"], row["delta_ci_hi"] = ci(d)
                        # One-sided evidence that the rule beats the reference.
                        # Reported as a bootstrap fraction rather than a p-value
                        # because that is what it is.
                        #
                        # A delta that is identically zero in every replicate is
                        # not a tie the data produced, it is the same ordering
                        # under two names: inside a slide, ranking by mu and by
                        # the within-slide rank of mu cannot differ. Reporting
                        # P(better) = 0 there would read as "reliably worse".
                        arr = np.asarray(d)
                        degenerate = bool(np.all(arr == 0.0))
                        row["frac_better"] = (float("nan") if degenerate
                                              else float(np.mean(arr < 0)))
                        if degenerate:
                            row["note"] = "identical ordering by construction"
                        row["n_boot_used"] = len(d)
                    deltas.append(row)
    return rows, deltas, coefs


def reconciliation(rows: List[dict], descriptor: str) -> dict:
    """The 8.1% / 7.8% question in `sec/supp_data.tex`, at full precision.

    The text reduces 0.0397 to 0.0366 and calls it 8.1%, then gives 7.8% for the
    same quantity three sentences later. From the two printed values it is
    7.808%; against the random baseline of 0.0398 it is 8.040%. Only the
    unrounded numbers can say which, so they are written out here rather than
    left for someone to recompute from a figure.
    """
    sel = [r for r in rows if r.get("descriptor") == descriptor
           and r["scope"] == "pooled" and r["rule"] == "sd"
           and abs(r["coverage"] - 0.8) < 1e-9 and r.get("component") == "total"]
    if not sel:
        return {"note": "no pooled total-sigma row at 80% coverage to reconcile"}
    r = sel[0]
    mae, base = r["mae"], r["mae_random"]
    out = {
        # `mae_random` at any coverage IS the full-coverage MAE: random
        # selection is unbiased, so the two are the same number.
        "mae_all_regions_full_precision": base,
        "mae_kept_80_total_sd_full_precision": mae,
        "reduction_from_full_precision": mae / base - 1.0,
        "reduction_from_rounded_mae": (round(mae, 4) - round(base, 4)) / round(base, 4),
        "published_8_1_percent": -0.081,
        "published_7_8_percent": -0.078,
    }
    out["note"] = (
        "The manuscript's 8.1% and 7.8% describe the same quantity. "
        f"At full precision it is {out['reduction_from_full_precision'] * 100:+.3f}%. "
        "Correct the text to this value and delete whichever of the two "
        "does not match."
    )
    return out


def check_published(rows: List[dict], descriptor: str) -> List[dict]:
    """Acceptance checks: the eight liver numbers this must reproduce."""
    def get(rule, scope, cov=0.8, field="rel_change", comp="total"):
        for r in rows:
            if (r.get("descriptor") == descriptor and r.get("component") == comp
                    and r["scope"] == scope and r["rule"] == rule
                    and abs(r["coverage"] - cov) < 1e-9):
                return r.get(field)
        return None

    got = {
        "mae_all_regions":      get("random", "pooled", 1.0, "mae"),
        "mae_kept_80_total_sd": get("sd", "pooled", 0.8, "mae"),
        "mae_random_80":        get("random", "pooled", 0.8, "mae"),
        "rel_oracle_80":        get("sd", "pooled", 0.8, "rel_change_oracle"),
        "rel_mu_80_pooled":     get("mu", "pooled"),
        "rel_mu_80_within":     get("mu", "within"),
        "rel_sd_80_pooled":     get("sd", "pooled"),
        "rel_sd_80_within":     get("sd", "within"),
    }
    out = []
    for k, published in PUBLISHED_LIVER.items():
        v = got.get(k)
        # Tolerance is the rounding the manuscript itself prints to: MAEs to
        # four decimals, percentages to one.
        tol = 0.00005 if k.startswith("mae") else 0.0005
        out.append({"quantity": k, "published": published,
                    "computed": None if v is None else float(v),
                    "abs_diff": None if v is None else abs(float(v) - published),
                    "matches": None if v is None else bool(abs(float(v) - published) <= tol)})
    return out


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def make_figure(rows: List[dict], outpath: Path, descriptor: str,
                component: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([r for r in rows if r.get("descriptor") == descriptor
                       and r.get("component") == component])
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, scope in zip(axes, ("pooled", "within")):
        s = df[df["scope"] == scope]
        if s.empty:
            continue
        ax.axhline(0, color="#52514e", linewidth=0.8, linestyle=(0, (4, 2)),
                   zorder=3, label="random")
        for key, label, colour, ls, mk in RULES:
            gg = s[s["rule"] == key].sort_values("coverage")
            if gg.empty:
                continue
            ax.plot(gg["coverage"] * 100, gg["rel_change"] * 100, color=colour,
                    linestyle=ls, marker=mk, markersize=4.2, zorder=5,
                    label=label, markeredgecolor="white", markeredgewidth=0.4)
        gg = s[s["rule"] == "sd"].sort_values("coverage")
        ax.plot(gg["coverage"] * 100, gg["rel_change_oracle"] * 100,
                color="#0b0b0b", linestyle=(0, (1, 1.4)), linewidth=0.9,
                zorder=4, label="oracle (rank by $e$)")
        ax.invert_xaxis()
        ax.set_xlabel("coverage (% of regions kept)")
        ax.set_title(scope, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("change in mean $e$ (%)")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="lower left",
                    handlelength=2.6, labelspacing=0.24)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="W-30: does mu + sigma rank error better than mu alone?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--table", type=Path, required=True,
                    help="per_region_sources.csv (compare_uncertainty_sources.py) "
                         "or per_region_calibration.csv (calibrate_phi.py). The "
                         "paper's risk_coverage.pdf is built from the first, so "
                         "that is the one whose numbers the acceptance checks "
                         "expect.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--descriptor", default="task_specific_value",
                    help="CPA. 'all' scores every descriptor in the table.")
    ap.add_argument("--components", nargs="*", default=None,
                    help="Which uncertainty components to rank by. Default: "
                         "every one present except the synthetic point-prediction "
                         "source, whose sd IS mu and which is the 'mu' rule here.")
    ap.add_argument("--prediction", default="grand",
                    help="Filter on the `prediction` column where the table has "
                         "one. 'any' keeps every row.")
    ap.add_argument("--coverages", type=float, nargs="+",
                    default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--reference_rule", default="mu", choices=RULE_NAMES,
                    help="The rule the paired differences are taken against. "
                         "'mu' is the deployment baseline W-30 asks about.")
    ap.add_argument("--min_regions_per_slide", type=int, default=10,
                    help="Slides below this are dropped from BOTH scopes, so the "
                         "two are computed on one cohort.")
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="Case-clustered bootstrap replicates. 0 disables the "
                         "intervals, which are the point of the exercise.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check_published", action="store_true",
                    help="Compare against the eight published liver numbers. "
                         "Meaningless on the kidney arm.")
    args = ap.parse_args()

    t = pd.read_csv(args.table)
    for col in ("mu", "sd", "error", "wsi", "component"):
        if col not in t.columns:
            raise SystemExit(
                f"{args.table} has no '{col}' column. Expected the flat "
                f"per-region table from compare_uncertainty_sources.py or "
                f"calibrate_phi.py; found columns: {list(t.columns)}")
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
    if args.components:
        keep = list(args.components)
    else:
        keep = [c for c in present if c not in SYNTHETIC_COMPONENTS]
        dropped = [c for c in present if c in SYNTHETIC_COMPONENTS]
        if dropped:
            print(f"[note] dropped synthetic component(s) {dropped}: their sd is "
                  f"the point prediction, which is the 'mu' rule here")
    t = t[t["component"].isin(keep)]
    if t.empty:
        raise SystemExit(f"no rows left after selecting components {keep} "
                         f"(present: {present})")

    coverages = sorted(set(args.coverages), reverse=True)
    if coverages[0] != 1.0:
        # Full coverage is the base every relative change is measured against,
        # and its absence would silently rescale every number in the table.
        coverages = [1.0] + coverages

    rows: List[dict] = []
    deltas: List[dict] = []
    coefs: List[dict] = []
    keys = [k for k in ("descriptor", "component") if k in t.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, g in t.groupby(keys, sort=False):
            r, d, c = run_component(g, coverages, args.n_boot, args.seed,
                                    args.reference_rule,
                                    args.min_regions_per_slide)
            if not r:
                print(f"[skip] {key}: fewer than 3 slides or 20 regions after "
                      f"filtering")
            rows += r
            deltas += d
            coefs += c

    if not rows:
        raise SystemExit("nothing scored — check --descriptor and --components")

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir / "rank_rules.csv", index=False)
    pd.DataFrame(deltas).to_csv(args.outdir / "rank_rule_deltas.csv", index=False)
    if coefs:
        pd.DataFrame(coefs).to_csv(args.outdir / "fitted_coefficients.csv",
                                   index=False)

    desc = (args.descriptor if args.descriptor != "all"
            else rows[0].get("descriptor", "unknown"))
    recon = reconciliation(rows, desc)
    checks = check_published(rows, desc) if args.check_published else None

    summary = {
        "table": str(args.table),
        "descriptor": args.descriptor,
        "components": keep,
        "prediction": args.prediction,
        "coverages": coverages,
        "reference_rule": args.reference_rule,
        "min_regions_per_slide": args.min_regions_per_slide,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "rules": RULE_NAMES,
        "rank_rules": rows,
        "deltas": deltas,
        "fitted_coefficients": coefs,
        "reconciliation_8_1_vs_7_8": recon,
        "published_checks": checks,
    }
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    for comp in dict.fromkeys(r["component"] for r in rows
                              if r.get("descriptor") == desc):
        make_figure(rows, args.outdir / f"rank_rules_{comp}.png", desc, comp,
                    f"{desc} — ranked by {comp} $\\sigma$")

    # ---- what the run says ----
    df = pd.DataFrame(rows)
    for comp in dict.fromkeys(df["component"]):
        for scope in ("pooled", "within"):
            s = df[(df["component"] == comp) & (df["scope"] == scope)
                   & (np.abs(df["coverage"] - 0.8) < 1e-9)
                   & (df["descriptor"] == desc)]
            if s.empty:
                continue
            print(f"\n--- {desc} / {comp} / {scope}, 80% coverage ---")
            print(f"{'rule':>10s} {'MAE':>9s} {'change':>9s} {'95% CI':>19s} "
                  f"{'of oracle':>10s}")
            for _, r in s.iterrows():
                cistr = (f"[{r['rel_ci_lo'] * 100:+.1f}%,{r['rel_ci_hi'] * 100:+.1f}%]"
                         if "rel_ci_lo" in s.columns and np.isfinite(r.get("rel_ci_lo", np.nan))
                         else "")
                cap = (f"{r['capture_of_oracle'] * 100:8.1f}%"
                       if np.isfinite(r["capture_of_oracle"]) else "")
                print(f"{r['rule']:>10s} {r['mae']:9.5f} "
                      f"{r['rel_change'] * 100:+8.2f}% {cistr:>19s} {cap:>10s}")

    dd = pd.DataFrame(deltas)
    if not dd.empty:
        print(f"\n=== THE CLAIM: paired difference against '{args.reference_rule}' "
              f"at 80% coverage ===")
        print("Both rules are evaluated on the SAME resampled cohort, so this "
              "interval is\nabout the difference — which two overlapping "
              "marginal intervals are not.")
        print(f"{'component':>14s} {'scope':>7s} {'rule':>10s} {'delta':>9s} "
              f"{'95% CI':>19s} {'P(better)':>10s}")
        s = dd[np.abs(dd["coverage"] - 0.8) < 1e-9]
        for _, r in s.iterrows():
            if r["rule"] in ("random", "oracle"):
                continue
            cistr = (f"[{r['delta_ci_lo'] * 100:+.1f}%,{r['delta_ci_hi'] * 100:+.1f}%]"
                     if "delta_ci_lo" in s.columns and np.isfinite(r.get("delta_ci_lo", np.nan))
                     else "")
            fb = (f"{r['frac_better']:9.3f}"
                  if "frac_better" in s.columns and np.isfinite(r.get("frac_better", np.nan))
                  else "")
            print(f"{r['component']:>14s} {r['scope']:>7s} {r['rule']:>10s} "
                  f"{r['delta_rel_change'] * 100:+8.2f}% {cistr:>19s} {fb:>10s}")
        print("\nNegative delta = the rule beats the reference. The section 8 "
              "claim is\ndemonstrated only where the CI excludes zero.")

    if coefs:
        c = pd.DataFrame(coefs)
        print(f"\n--- LOCO fitted combination, {len(c)} held-out cases ---")
        print("Rank of e on within-slide ranks of mu and sd. The rank-sum rule "
              "is this\nmodel with both coefficients pinned at 1, so a b_sd near "
              "zero means the\ndata wanted mu and nothing else.")
        blocks = (c.groupby("component") if "component" in c.columns
                  else [("", c)])
        for comp, gc in blocks:
            print(f"{comp:>14s}  b_mu {gc['b_mu'].mean():+.3f} "
                  f"[{gc['b_mu'].min():+.3f},{gc['b_mu'].max():+.3f}]   "
                  f"b_sd {gc['b_sd'].mean():+.3f} "
                  f"[{gc['b_sd'].min():+.3f},{gc['b_sd'].max():+.3f}]")

    print(f"\n--- 8.1% vs 7.8% in sec/supp_data.tex ---")
    for k, v in recon.items():
        print(f"  {k}: {v}" if not isinstance(v, float) else f"  {k}: {v:.6f}")

    if checks:
        print("\n--- acceptance checks against the published liver numbers ---")
        print(f"{'quantity':>24s} {'published':>11s} {'computed':>11s} {'ok':>4s}")
        for c in checks:
            got = "n/a" if c["computed"] is None else f"{c['computed']:11.5f}"
            ok = "-" if c["matches"] is None else ("yes" if c["matches"] else "NO")
            print(f"{c['quantity']:>24s} {c['published']:11.5f} {got:>11s} {ok:>4s}")
        bad = [c["quantity"] for c in checks if c["matches"] is False]
        if bad:
            print(f"\n[warn] {len(bad)} check(s) did not reproduce: {bad}\n"
                  f"       Resolve these before quoting anything above — a "
                  f"mismatch means this\n       table is not the one behind the "
                  f"published figure.")

    print(f"\nwrote {args.outdir}/rank_rules.csv, rank_rule_deltas.csv, "
          f"fitted_coefficients.csv and summary.json")


if __name__ == "__main__":
    main()
