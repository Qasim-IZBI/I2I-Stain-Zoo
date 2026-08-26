"""How stable is the data-exposure component with K = 5? (W-29)

With five disjoint training subsets the between-group variance rests on **four
degrees of freedom**, and the paper shows no sensitivity analysis behind its
claim that roughly half the region-level variance is data exposure (median share
0.508). A real sensitivity study over K means retraining and is out of scope.
What is affordable is a stability estimate on the existing grid — plus honest
disclosure of the asymmetry, which is itself most of the answer:

    procedural      K(S-1) = 45 df      relative SE ~ sqrt(2/45) = 21%
    data exposure   K-1    =  4 df      relative SE ~ sqrt(2/4)  = 71%

Those two numbers need no simulation and no assumption, and they are confirmed
against the actual grid rather than assumed: K and S are read from the phi run's
`summary.json`, not hard-coded.

What this reads, and the one thing it cannot
--------------------------------------------
`per_region.csv` carries `fold{f}_mu_{name}` and `fold{f}_sd_{name}` for every
subset, which is exactly enough to rerun the decomposition with a subset left
out. It does **not** carry per-member phi: `compute_phi_uncertainty.py` holds the
member block in memory and writes only the fold summaries. So the subset
jackknife (steps 1, 2, 4) is exact, and the seed-dimension contrast (step 3) is
not directly computable from the CSVs.

Two routes are offered for it, and the run says which one it took:

* `--member_npz` — exact, if a member-level dump exists. The format is one array
  per fold, `[n_seeds, n_regions]` for the descriptor, saved under keys
  `fold1`..`foldK`. Producing it is a one-line addition to
  `compute_phi_uncertainty.py`, which already has the block as `blocks`.
* the default **parametric** route, which draws subsample means and variances
  from the stored per-fold mean and SD under a Gaussian-within-fold assumption,
  with the finite-population correction for drawing s of S seeds without
  replacement. Labelled as parametric everywhere it appears, because it is a
  model of the seed spread rather than a resample of it.

The analytic df statement above does not depend on either route, and it is the
disclosure the supervisor actually asked for.

A note on the jackknife SE
--------------------------
The delete-one jackknife is **inconsistent for the median** — it is a
non-smooth statistic, and this is a textbook failure case rather than a
technicality. It is computed because the spec asks for it, and reported beside
two things that do not share the problem: the LOSO **range**, which is
assumption-free, and the same jackknife on the **mean** share, where the
estimator is smooth and the SE is on firm ground. If the three disagree, quote
the range.

Outputs
-------
loso_shares.csv        the five leave-one-subset-out replicates
seed_subsample.csv     the seed-dimension contrast, per draw
stability_summary.json everything, including the reconstruction check
stability_data_exposure.png   panel A the contrast, panel B the df asymmetry
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# fig:dataexposure and the supplement. Checked, not assumed — check 2 in the spec
# is the test that the data pull is right, and everything else rests on it.
PUBLISHED = {
    "median_share": 0.508,
    "iqr_lo": 0.44,
    "iqr_hi": 0.57,
    "n_regions": 2844,
    "n_regions_with_data_component": 2838,
}


# --------------------------------------------------------------------------
# the estimator, reproduced from fold summaries
# --------------------------------------------------------------------------

def components_from_folds(fold_mu: np.ndarray, fold_var: np.ndarray,
                          counts: np.ndarray,
                          procedural_override: Optional[np.ndarray] = None
                          ) -> Dict[str, np.ndarray]:
    """`uncertainty_phi.decompose` for one descriptor, from mean and SD per fold.

    Reproduces both corrections, because the spec is explicit that a corrected
    estimate must not be compared against an uncorrected one:

    * within-fold variance is already ddof=1 — `fold{f}_sd_*` is written with
      `np.nanstd(..., ddof=1)`, so squaring it is the unbiased term;
    * the spread of fold means is contaminated by procedural noise, since each
      fold mean averages S noisy members. `Var(fold means) = sigma^2_data +
      sigma^2_proc / n0`, so `sigma^2_proc / n0` is subtracted.

    `n0` is the ANOVA effective group size, equal to S for a balanced design and
    degrading gracefully when seed counts differ — recomputed here from whichever
    folds are in play, which is what makes leave-one-subset-out legitimate rather
    than an approximation.

    `procedural_override` holds the procedural term at the full-grid estimate
    while the between term is jackknifed. That isolates the component the
    supervisor asked about: procedural carries 45 df and barely moves, so
    recomputing it from four folds would mix a large, stable change into a small,
    unstable one. It is used in the /n0 correction too, since the correction
    describes the same noise.
    """
    K = len(fold_mu)
    if K < 2:
        raise ValueError("need at least two folds for a between-subset term")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        procedural = (np.nanmean(fold_var, axis=0)
                      if procedural_override is None else procedural_override)
        total_n = counts.sum()
        n0 = (total_n - (counts ** 2).sum() / total_n) / (K - 1)
        between = np.nanvar(fold_mu, axis=0, ddof=1)
        data = between - procedural / n0
        total = procedural + data
    return {"procedural": procedural, "data": data, "total": total,
            "between": between, "n0": float(n0), "n_folds": K}


def share(comp: Dict[str, np.ndarray]) -> np.ndarray:
    """data / (procedural + data), per region.

    Undefined where the ANOVA put the between-subset term at or below zero:
    there is no data-exposure SD, and filling in a share of 0 would drag the
    median down with regions that carry no estimate rather than an estimate of
    nothing. Returned as NaN and counted, matching `fig_data_exposure_share`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        s = comp["data"] / comp["total"]
    s = np.where((comp["data"] > 0) & (comp["total"] > 0), s, np.nan)
    return s


def summarise_share(s: np.ndarray, n_all: int) -> Dict[str, float]:
    v = s[np.isfinite(s)]
    if not len(v):
        return {"median_share": float("nan"), "mean_share": float("nan"),
                "iqr_lo": float("nan"), "iqr_hi": float("nan"),
                "n_with_data": 0, "n_regions": int(n_all),
                "n_dropped_no_data_component": int(n_all)}
    return {"median_share": float(np.median(v)), "mean_share": float(np.mean(v)),
            "iqr_lo": float(np.percentile(v, 25)),
            "iqr_hi": float(np.percentile(v, 75)),
            "n_with_data": int(len(v)), "n_regions": int(n_all),
            "n_dropped_no_data_component": int(n_all - len(v))}


# --------------------------------------------------------------------------
# 1-2. leave one subset out, and the jackknife on it
# --------------------------------------------------------------------------

def leave_one_subset_out(fold_mu: np.ndarray, fold_var: np.ndarray,
                         counts: np.ndarray, procedural_full: np.ndarray,
                         hold_procedural: bool
                         ) -> Tuple[List[dict], List[np.ndarray]]:
    """Five replicates of the headline share, each missing one training subset.

    Returns the per-region share arrays as well as the summaries, because the
    two readings of these replicates need the arrays (see `matched_summaries`).
    """
    out, shares = [], []
    K = len(fold_mu)
    for k in range(K):
        keep = [i for i in range(K) if i != k]
        comp = components_from_folds(
            fold_mu[keep], fold_var[keep], counts[keep],
            procedural_override=procedural_full if hold_procedural else None)
        sh = share(comp)
        shares.append(sh)
        row = {"dropped_fold": k + 1, "n_folds_used": K - 1,
               "df_between": K - 2, "n0": comp["n0"],
               "procedural_held_fixed": bool(hold_procedural)}
        row.update(summarise_share(sh, fold_mu.shape[1]))
        out.append(row)
    return out, shares


def matched_summaries(full_share: np.ndarray, loso_shares: List[np.ndarray]
                      ) -> Tuple[dict, List[dict], np.ndarray]:
    """The same medians on the regions that survive in EVERY replicate.

    Two mechanisms move a leave-one-subset-out median away from the full-grid
    value, they act in OPPOSITE directions, and neither is a mistake:

    * **selection.** With K=4 the between-subset term is noisier, so more regions
      fall at or below zero and drop out — and the ones that survive are the ones
      with the larger data estimates. That pushes the median UP.
    * **noise, on a fixed region set.** The share is `1/(1 + proc/data)`, concave
      in `data`, so a noisier `data` pulls the median DOWN by Jensen.

    Measured on a synthetic grid with 400 regions: unmatched, all five replicates
    landed above the full grid (0.177-0.185 against 0.174) while 19-27 more
    regions dropped out; restricted to the 229 common regions, all five landed
    below it (0.222-0.230 against 0.235).

    So **the spec's check that the five should bracket the full-grid value is not
    a valid diagnostic** — one-sidedness is the expected outcome in both readings,
    and it says nothing about whether the recomputation reproduces the estimator.
    `reconstruction_check` is what does that, exactly and unambiguously.

    Both readings are reported: the unmatched one is what a study actually run at
    K=4 would publish, selection included; the matched one isolates the noise.
    """
    common = np.isfinite(full_share)
    for sh in loso_shares:
        common &= np.isfinite(sh)
    n = int(common.sum())
    full = {"median_share_matched": float(np.median(full_share[common])) if n else float("nan"),
            "mean_share_matched": float(np.mean(full_share[common])) if n else float("nan"),
            "n_matched": n}
    rows = [{"median_share_matched": float(np.median(sh[common])) if n else float("nan"),
             "mean_share_matched": float(np.mean(sh[common])) if n else float("nan"),
             "n_matched": n} for sh in loso_shares]
    return full, rows, common


def jackknife(theta_full: float, theta_loo: Sequence[float]) -> Dict[str, float]:
    """Delete-one jackknife bias and SE over the K subsets.

    **Inconsistent for a median**, which is a known failure of the delete-one
    jackknife on non-smooth statistics rather than a caveat invented here. It is
    reported because the spec asks for it, and always beside the assumption-free
    range. On the mean share the same formulae are sound.
    """
    v = np.asarray([x for x in theta_loo if np.isfinite(x)], dtype=np.float64)
    K = len(v)
    if K < 2 or not np.isfinite(theta_full):
        return {"jackknife_se": float("nan"), "jackknife_bias": float("nan"),
                "n": K}
    bar = float(np.mean(v))
    return {
        "mean_of_replicates": bar,
        "jackknife_bias": float((K - 1) * (bar - theta_full)),
        "jackknife_se": float(np.sqrt((K - 1) / K * np.sum((v - bar) ** 2))),
        "bias_corrected": float(K * theta_full - (K - 1) * bar),
        "range_lo": float(np.min(v)), "range_hi": float(np.max(v)),
        "spread": float(np.ptp(v)), "n": K,
    }


# --------------------------------------------------------------------------
# 3. the seed-dimension contrast
# --------------------------------------------------------------------------

def seed_subsample_parametric(fold_mu: np.ndarray, fold_var: np.ndarray,
                              counts: np.ndarray, s_sub: int, n_draws: int,
                              seed: int) -> List[float]:
    """Median share when only s of S seeds per fold are used — PARAMETRIC.

    Per-member phi is not on disk, so this models the seed spread rather than
    resampling it. For a simple random sample of s from a finite population of S
    members, with S^2 the population's ddof=1 variance:

        Var(subsample mean) = (S^2 / s) * (1 - s/S)          [FPC]
        E[subsample ddof=1 variance] = S^2,  spread ~ S^2 * chi2_{s-1}/(s-1)

    The fold mean is treated as the finite population mean, so a draw perturbs it
    by exactly the sampling error of taking s of the S members that exist. The
    Gaussian and chi-square forms are the assumption; the FPC is not.

    The point of this is the CONTRAST with the subset jackknife, and that
    contrast is driven by degrees of freedom, which the model gets right by
    construction. Read the analytic df figures as the primary statement and this
    as its illustration.
    """
    rng = np.random.default_rng(seed)
    K, R = fold_mu.shape
    S = counts.astype(np.float64)
    out = []
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_draws):
            mu_d = np.empty_like(fold_mu)
            var_d = np.empty_like(fold_var)
            for k in range(K):
                s_k = min(s_sub, int(S[k]))
                se = np.sqrt(np.maximum(fold_var[k], 0.0) / s_k
                             * max(0.0, 1.0 - s_k / S[k]))
                mu_d[k] = fold_mu[k] + rng.normal(0.0, 1.0, R) * se
                var_d[k] = (fold_var[k] * rng.chisquare(max(1, s_k - 1), R)
                            / max(1, s_k - 1))
            comp = components_from_folds(mu_d, var_d,
                                         np.full(K, float(s_sub)))
            v = share(comp)
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if len(v) else float("nan"))
    return out


def seed_subsample_exact(members: List[np.ndarray], s_sub: int, n_draws: int,
                         seed: int) -> List[float]:
    """The same, resampling actual members. `members[f]` is [n_seeds, n_regions]."""
    rng = np.random.default_rng(seed)
    out = []
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_draws):
            mu, var, cnt = [], [], []
            for f in members:
                pick = rng.choice(len(f), min(s_sub, len(f)), replace=False)
                sub = f[pick]
                mu.append(np.nanmean(sub, axis=0))
                var.append(np.nanvar(sub, axis=0, ddof=1))
                cnt.append(len(pick))
            comp = components_from_folds(np.stack(mu), np.stack(var),
                                         np.asarray(cnt, dtype=np.float64))
            v = share(comp)
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if len(v) else float("nan"))
    return out


# --------------------------------------------------------------------------
# 4. case-clustered bootstrap on the headline
# --------------------------------------------------------------------------

def cluster_bootstrap_median(s: np.ndarray, wsi: np.ndarray, n_boot: int,
                             seed: int) -> Dict[str, float]:
    """95% CI for the median share, resampling WHOLE SLIDES.

    The interval convention used everywhere else in the study: regions inside a
    slide are spatially correlated, so an interval over ~2844 of them describes a
    cohort of twenty cases as if it held 2844.
    """
    uniq = np.unique(wsi)
    if len(uniq) < 3 or not n_boot:
        return {}
    idx = {u: np.where(wsi == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq),
                                                          replace=True)])
        v = s[pick]
        v = v[np.isfinite(v)]
        if len(v):
            draws.append(float(np.median(v)))
    if len(draws) < max(20, n_boot // 10):
        return {}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"ci_lo": float(lo), "ci_hi": float(hi), "n_boot_used": len(draws),
            "n_slides": int(len(uniq))}


# --------------------------------------------------------------------------
# 5. the df asymmetry — the disclosure that needs no simulation
# --------------------------------------------------------------------------

def df_asymmetry(K: int, counts: Sequence[float]) -> dict:
    """Degrees of freedom behind each component, from the actual grid.

    The relative SE of a variance estimate on nu df is sqrt(2/nu) — a chi-square
    result, exact under normality and the right order of magnitude without it.
    This is the whole of SH's point, quantified: the two components of one
    decomposition are not estimated to remotely comparable precision, and no
    amount of resampling the existing grid can change that.
    """
    S = np.asarray(counts, dtype=np.float64)
    df_proc = float(np.sum(S - 1.0))
    df_data = float(K - 1)
    return {
        "K_subsets": int(K), "seeds_per_fold": [int(x) for x in S],
        "df_procedural": df_proc, "df_data_exposure": df_data,
        "relative_se_procedural": float(np.sqrt(2.0 / df_proc)) if df_proc else float("nan"),
        "relative_se_data_exposure": float(np.sqrt(2.0 / df_data)) if df_data else float("nan"),
        "ratio": float(np.sqrt(df_proc / df_data)) if df_data else float("nan"),
        "note": ("relative SE of a variance component on nu df is sqrt(2/nu). "
                 "The asymmetry is a property of the design, not of the "
                 "estimator, and only more subsets can change it."),
    }


# --------------------------------------------------------------------------

def load_folds(t: pd.DataFrame, descriptor: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """`fold{f}_mu_{name}` / `fold{f}_sd_{name}` -> [K, R] arrays."""
    mus, sds, names = [], [], []
    f = 1
    while f"fold{f}_mu_{descriptor}" in t.columns:
        mus.append(t[f"fold{f}_mu_{descriptor}"].to_numpy(np.float64))
        sd_col = f"fold{f}_sd_{descriptor}"
        if sd_col not in t.columns:
            raise SystemExit(f"per_region.csv has {f'fold{f}_mu_{descriptor}'} "
                             f"but no {sd_col}")
        sds.append(t[sd_col].to_numpy(np.float64))
        names.append(f"fold{f}")
        f += 1
    if len(mus) < 2:
        raise SystemExit(
            f"found {len(mus)} fold column set(s) for '{descriptor}'. This needs "
            f"the CROSSED grid (compute_phi_uncertainty.py with one --fold per "
            f"subset); a single --ensemble run has no data-exposure term at all.")
    return np.stack(mus), np.stack(sds) ** 2, names


def reconstruction_check(t: pd.DataFrame, descriptor: str,
                         comp: Dict[str, np.ndarray]) -> dict:
    """Does the reconstruction reproduce the columns the phi run already wrote?

    The gate for everything else. If `sd_procedural_*`, `sd_data_*` and
    `sd_total_*` come back from the fold summaries to floating point, then the
    leave-one-subset-out recomputation is running the same estimator the paper
    ran, and check 1 in the spec ("the five values should bracket 0.508") is
    testing stability rather than testing whether this script is right.
    """
    out = {}
    for key, col in (("procedural", f"sd_procedural_{descriptor}"),
                     ("data", f"sd_data_{descriptor}"),
                     ("total", f"sd_total_{descriptor}")):
        if col not in t.columns:
            out[key] = {"column": col, "present": False}
            continue
        want = t[col].to_numpy(np.float64)
        with np.errstate(invalid="ignore"):
            got = np.sqrt(np.where(comp[key] > 0, comp[key], np.nan))
        m = np.isfinite(want) & np.isfinite(got)
        d = float(np.max(np.abs(want[m] - got[m]))) if m.any() else float("nan")
        scale = float(np.nanmax(np.abs(want[m]))) if m.any() else 1.0
        out[key] = {"column": col, "present": True, "n_compared": int(m.sum()),
                    "max_abs_diff": d,
                    "max_rel_diff": float(d / scale) if scale else float("nan"),
                    # 1e-6 relative is CSV round-trip, not a different estimator.
                    "matches": bool(np.isfinite(d) and d <= 1e-6 * max(scale, 1e-12))}
    out["passes"] = all(v.get("matches", False) for v in out.values()
                        if isinstance(v, dict) and v.get("present"))
    return out


def make_figure(loso: List[dict], full: dict, boot: dict, seeds: Dict[int, List[float]],
                dfa: dict, outpath: Path, descriptor: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax = axes[0]
    med = full["median_share"]
    if boot:
        ax.axhspan(boot["ci_lo"], boot["ci_hi"], color="#2a78d6", alpha=0.14,
                   linewidth=0, zorder=1,
                   label=f"case bootstrap 95% CI")
    ax.axhline(med, color="#0b0b0b", linewidth=1.0, linestyle=(0, (4, 2)),
               zorder=3, label=f"full grid  {med:.3f}")
    K = dfa["K_subsets"]
    S = int(max(dfa["seeds_per_fold"]))
    ax.scatter(np.linspace(-0.10, 0.10, len(loso)),
               [r["median_share"] for r in loso], s=34, color="#eb6834",
               zorder=5, edgecolor="white", linewidth=0.5,
               label=f"leave one subset out ({K} replicates)")
    # One column per perturbation size rather than one shared column: the sizes
    # are different perturbations and overplotting them reads as a single cloud.
    ticks = [0]
    labels = [f"drop 1 of {K}\nsubsets\n({K - 1} df between)"]
    greens = ["#1baf7a", "#7fcfae"]
    for i, (s_sub, vals) in enumerate(sorted(seeds.items(), reverse=True)):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if not len(v):
            continue
        xpos = i + 1
        ax.scatter(np.full(len(v), float(xpos))
                   + np.random.default_rng(0).uniform(-0.16, 0.16, len(v)),
                   v, s=8, color=greens[i % len(greens)], alpha=0.5, zorder=4,
                   edgecolor="none", label=f"seeds S={S}→{s_sub}")
        ticks.append(xpos)
        labels.append(f"drop to {s_sub} of {S}\nseeds\n"
                      f"({100 * (1 - s_sub / S):.0f}% cut)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.45, max(ticks) + 0.45)
    ax.set_ylabel("median data-exposure share")
    ax.set_title("how far the headline moves under each cut", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.6, loc="best")

    ax = axes[1]
    nus = np.array([dfa["df_data_exposure"], dfa["df_procedural"]])
    labs = [f"data exposure\nK−1 = {int(nus[0])} df",
            f"procedural\nK(S−1) = {int(nus[1])} df"]
    vals = np.sqrt(2.0 / nus) * 100
    ax.bar(range(2), vals, color=["#1baf7a", "#eb6834"], width=0.55, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("relative SE of the variance component")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("a property of the design, not the estimator", fontsize=8.5)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(f"{descriptor} — stability of the data-exposure component",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="W-29: stability of the data-exposure component at K = 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--per_region", type=Path, required=True,
                    help="per_region.csv from the CROSSED grid run of "
                         "compute_phi_uncertainty.py — it must carry "
                         "fold{f}_mu_* columns.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--descriptor", default="task_specific_value")
    ap.add_argument("--member_npz", type=Path, default=None,
                    help="Optional exact route for the seed contrast: one array "
                         "per fold, [n_seeds, n_regions], under keys fold1..foldK. "
                         "Without it the contrast is parametric and labelled so.")
    ap.add_argument("--seeds_per_fold", type=int, nargs="*", default=None,
                    help="Overrides summary.json. Only needed if that file is "
                         "missing — S is not recoverable from per_region.csv.")
    ap.add_argument("--subsample_sizes", type=int, nargs="*", default=None,
                    help="s in 'use s of S seeds per fold'. Default: the size "
                         "that removes the SAME FRACTION as dropping one of K "
                         "subsets, plus S/2 (the spec's S->5). A matched "
                         "fraction is the only way the two arms answer the same "
                         "question.")
    ap.add_argument("--n_draws", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute_procedural", action="store_true",
                    help="Recompute the procedural term inside each "
                         "leave-one-out replicate instead of holding it at the "
                         "full-grid value. Reported either way; the default "
                         "isolates the 4-df component the question is about.")
    args = ap.parse_args()

    t = pd.read_csv(args.per_region)
    fold_mu, fold_var, fold_names = load_folds(t, args.descriptor)
    K, R = fold_mu.shape

    # S is not in per_region.csv. summary.json beside it records n_seeds_per_fold,
    # which is also what confirms the 45 df rather than assuming a balanced grid.
    counts = None
    sj = args.per_region.parent / "summary.json"
    if args.seeds_per_fold:
        counts = np.asarray(args.seeds_per_fold, dtype=np.float64)
    elif sj.exists():
        try:
            v = json.load(open(sj)).get("variance", {}).get("n_seeds_per_fold")
            v = v or json.load(open(sj)).get("n_seeds_per_fold")
            if v:
                counts = np.asarray(v, dtype=np.float64)
        except (json.JSONDecodeError, OSError):
            counts = None
    if counts is None:
        raise SystemExit(
            f"cannot determine the seeds per fold. {sj} has no "
            f"n_seeds_per_fold, and per_region.csv does not record S. Pass "
            f"--seeds_per_fold (e.g. --seeds_per_fold 10 10 10 10 10).")
    if len(counts) != K:
        raise SystemExit(f"{len(counts)} seed counts for {K} folds")

    full = components_from_folds(fold_mu, fold_var, counts)
    check = reconstruction_check(t, args.descriptor, full)
    s_full = share(full)
    summ_full = summarise_share(s_full, R)
    boot = cluster_bootstrap_median(
        s_full, t["wsi"].to_numpy() if "wsi" in t.columns
        else np.zeros(R), args.n_boot, args.seed)
    dfa = df_asymmetry(K, counts)

    loso, loso_shares = leave_one_subset_out(
        fold_mu, fold_var, counts, full["procedural"],
        hold_procedural=not args.recompute_procedural)
    m_full, m_rows, common = matched_summaries(s_full, loso_shares)
    summ_full.update(m_full)
    for r, m in zip(loso, m_rows):
        r.update(m)
    jk_med = jackknife(summ_full["median_share"],
                       [r["median_share"] for r in loso])
    jk_mean = jackknife(summ_full["mean_share"], [r["mean_share"] for r in loso])
    jk_med_matched = jackknife(summ_full["median_share_matched"],
                               [r["median_share_matched"] for r in loso])

    members = None
    if args.member_npz:
        z = np.load(args.member_npz)
        members = [np.asarray(z[f"fold{i + 1}"], dtype=np.float64)
                   for i in range(K)]
        if any(m.shape[1] != R for m in members):
            raise SystemExit(f"member dump has {members[0].shape[1]} regions, "
                             f"per_region.csv has {R}")
    # Matched removal fraction: dropping one of K subsets removes 1/K of that
    # dimension, so the comparable seed perturbation is S*(1 - 1/K), not S/2.
    s_matched = max(2, int(round(float(np.min(counts)) * (1.0 - 1.0 / K))))
    sizes = args.subsample_sizes
    if not sizes:
        sizes = sorted({s_matched, max(2, int(np.min(counts)) // 2)}, reverse=True)
    seeds: Dict[int, List[float]] = {}
    for s_sub in sizes:
        seeds[s_sub] = (seed_subsample_exact(members, s_sub, args.n_draws, args.seed)
                        if members is not None else
                        seed_subsample_parametric(fold_mu, fold_var, counts,
                                                  s_sub, args.n_draws, args.seed))

    checks = {
        "median_share": {"published": PUBLISHED["median_share"],
                         "computed": summ_full["median_share"],
                         "matches": bool(abs(summ_full["median_share"]
                                             - PUBLISHED["median_share"]) <= 0.0005)},
        "iqr": {"published": [PUBLISHED["iqr_lo"], PUBLISHED["iqr_hi"]],
                "computed": [summ_full["iqr_lo"], summ_full["iqr_hi"]],
                "matches": bool(abs(summ_full["iqr_lo"] - PUBLISHED["iqr_lo"]) <= 0.005
                                and abs(summ_full["iqr_hi"] - PUBLISHED["iqr_hi"]) <= 0.005)},
        "n_regions": {"published": PUBLISHED["n_regions"], "computed": R,
                      "matches": R == PUBLISHED["n_regions"]},
        "n_with_data_component": {
            "published": PUBLISHED["n_regions_with_data_component"],
            "computed": summ_full["n_with_data"],
            "matches": (summ_full["n_with_data"]
                        == PUBLISHED["n_regions_with_data_component"])},
        "n_dropped": {"published": PUBLISHED["n_regions"]
                      - PUBLISHED["n_regions_with_data_component"],
                      "computed": summ_full["n_dropped_no_data_component"]},
        # REPORTED, NOT A PASS/FAIL. The spec expects the five replicates to
        # bracket the full-grid value and reads one-sidedness as a broken
        # recomputation. They are not expected to bracket it: selection pushes
        # the unmatched medians up and Jensen pushes the matched ones down, both
        # by construction. See `matched_summaries`. The gate is
        # `reconstruction_check`, which is exact.
        "loso_vs_full_grid": {
            "unmatched": {"lo": jk_med.get("range_lo"), "hi": jk_med.get("range_hi"),
                          "full": summ_full["median_share"],
                          "brackets": bool(
                              np.isfinite(jk_med.get("range_lo", np.nan))
                              and jk_med["range_lo"] <= summ_full["median_share"]
                              <= jk_med["range_hi"])},
            "matched": {"lo": jk_med_matched.get("range_lo"),
                        "hi": jk_med_matched.get("range_hi"),
                        "full": summ_full["median_share_matched"],
                        "n_matched": summ_full["n_matched"],
                        "brackets": bool(
                            np.isfinite(jk_med_matched.get("range_lo", np.nan))
                            and jk_med_matched["range_lo"]
                            <= summ_full["median_share_matched"]
                            <= jk_med_matched["range_hi"])},
            "note": ("not a diagnostic: dropping a subset makes the between term "
                     "noisier, which BOTH drops more regions (selection, pushes "
                     "the unmatched median up) and concaves the surviving shares "
                     "(Jensen, pushes the matched median down). One-sidedness in "
                     "either reading is expected. The reconstruction check is the "
                     "gate.")},
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(loso).to_csv(args.outdir / "loso_shares.csv", index=False)
    pd.DataFrame([{"s_sub": s, "draw": i, "median_share": v,
                   "route": "exact" if members is not None else "parametric"}
                  for s, vals in seeds.items() for i, v in enumerate(vals)]
                 ).to_csv(args.outdir / "seed_subsample.csv", index=False)
    summary = {
        "per_region": str(args.per_region), "descriptor": args.descriptor,
        "fold_columns": fold_names, "seeds_per_fold": [int(c) for c in counts],
        "n_regions": R, "n0_full_grid": full["n0"],
        "full_grid": summ_full, "bootstrap": boot,
        "df_asymmetry": dfa,
        "leave_one_subset_out": loso,
        "jackknife_median": jk_med, "jackknife_mean": jk_mean,
        "jackknife_median_matched": jk_med_matched,
        "seed_subsample_matched_size": s_matched,
        "seed_subsample": {
            str(s): {"route": "exact" if members is not None else "parametric",
                     "n_draws": len(v),
                     "median": float(np.nanmedian(v)) if len(v) else float("nan"),
                     "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
                     "range": [float(np.nanmin(v)), float(np.nanmax(v))] if len(v) else None}
            for s, v in seeds.items()},
        "reconstruction_check": check,
        "published_checks": checks,
        "procedural_held_fixed": not args.recompute_procedural,
    }
    with open(args.outdir / "stability_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    make_figure(loso, summ_full, boot, seeds, dfa,
                args.outdir / "stability_data_exposure.png", args.descriptor)

    # ---- what the run says ----
    print(f"\n--- gate: does the reconstruction reproduce the phi run's own columns? ---")
    for key in ("procedural", "data", "total"):
        c = check.get(key, {})
        if not c.get("present"):
            print(f"  {key:>11s}: column {c.get('column')} absent")
            continue
        print(f"  {key:>11s}: max |diff| {c['max_abs_diff']:.3e} over "
              f"{c['n_compared']} regions  "
              f"{'ok' if c['matches'] else 'MISMATCH'}")
    if not check["passes"]:
        print("\n[warn] the reconstruction does NOT match the stored columns, so "
              "the\n       leave-one-out replicates below are not the paper's "
              "estimator.\n       Resolve this before reading anything else.")

    print(f"\n=== the disclosure: degrees of freedom behind each component ===")
    print(f"  procedural     K(S-1) = {int(dfa['df_procedural']):3d} df   "
          f"relative SE ~ {dfa['relative_se_procedural'] * 100:.0f}%")
    print(f"  data exposure  K-1    = {int(dfa['df_data_exposure']):3d} df   "
          f"relative SE ~ {dfa['relative_se_data_exposure'] * 100:.0f}%")
    print(f"  The two components of one decomposition are estimated to precisions "
          f"differing\n  by {dfa['ratio']:.1f}x. That is a property of the design, "
          f"not of the estimator, and no\n  resampling of the existing grid can "
          f"change it — only more subsets can.")

    print(f"\n--- the headline, and what happens when a subset is removed ---")
    print(f"  full grid   median {summ_full['median_share']:.3f}  "
          f"IQR {summ_full['iqr_lo']:.2f}-{summ_full['iqr_hi']:.2f}  "
          f"n {summ_full['n_with_data']}/{summ_full['n_regions']}"
          + (f"  95% CI [{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]" if boot else ""))
    print(f"  matched on the {summ_full['n_matched']} regions surviving every "
          f"replicate, the full grid reads {summ_full['median_share_matched']:.3f}")
    print(f"  {'dropped':>8s} {'median':>8s} {'matched':>8s} {'mean':>8s} "
          f"{'n with data':>12s}")
    for r in loso:
        print(f"  {r['dropped_fold']:>8d} {r['median_share']:>8.3f} "
              f"{r['median_share_matched']:>8.3f} {r['mean_share']:>8.3f} "
              f"{r['n_with_data']:>12d}")
    print(f"  unmatched range {jk_med['range_lo']:.3f}-{jk_med['range_hi']:.3f} "
          f"(spread {jk_med['spread']:.3f}), jackknife SE "
          f"{jk_med['jackknife_se']:.3f}, bias {jk_med['jackknife_bias']:+.3f}")
    print(f"  matched   range {jk_med_matched['range_lo']:.3f}-"
          f"{jk_med_matched['range_hi']:.3f} "
          f"(spread {jk_med_matched['spread']:.3f}), jackknife SE "
          f"{jk_med_matched['jackknife_se']:.3f}")
    print("  The two readings move in OPPOSITE directions and both are correct: "
          "dropping a\n  subset makes the between term noisier, which drops more "
          "regions (selection lifts\n  the unmatched median) and concaves the "
          "surviving shares (Jensen lowers the\n  matched one). Quote the "
          "unmatched one as what a K=4 study would report.")
    print(f"  on the MEAN share, where the jackknife is sound: SE "
          f"{jk_mean['jackknife_se']:.3f}, bias {jk_mean['jackknife_bias']:+.3f}")
    print("  The delete-one jackknife is inconsistent for a median. If the SE and "
          "the range\n  disagree, quote the range — it assumes nothing.")

    print(f"\n--- the contrast: dropping a SUBSET vs dropping SEEDS ---")
    sd_subset = float(np.std([r["median_share"] for r in loso], ddof=1))
    print(f"  drop 1 of {K} subsets ({100.0 / K:.0f}% of that dimension):")
    print(f"      raw SD across the {K} replicates {sd_subset:.4f}  "
          f"-> jackknife SE {jk_med['jackknife_se']:.4f}")
    for s_sub, v in sorted(seeds.items(), reverse=True):
        v = np.asarray([x for x in v if np.isfinite(x)])
        if not len(v):
            continue
        route = "exact" if members is not None else "PARAMETRIC"
        frac = 100.0 * (1.0 - s_sub / float(np.min(counts)))
        tag = "  <- matched fraction" if s_sub == s_matched else ""
        print(f"  drop to {s_sub} of {int(np.min(counts))} seeds ({frac:.0f}%) "
              f"[{route}, {len(v)} draws]: SD {np.std(v, ddof=1):.4f}, median "
              f"{np.median(v):.3f}{tag}")
    print("\n  Two things make the naive version of this comparison wrong, and "
          "the spec's\n  expectation that the seed arm is 'far tighter' does not "
          "survive either:")
    print(f"  * The {K} leave-one-out replicates each share {K - 1} of {K} folds, "
          f"so their raw SD\n    is a CORRELATED dispersion that under-states the "
          f"sampling error. The jackknife\n    inflation by (K-1)/sqrt(K) = "
          f"{(K - 1) / np.sqrt(K):.2f} is what makes it an SE, and only the "
          f"inflated\n    figure belongs beside an independent bootstrap SD.")
    print("  * Removing half the seeds is a bigger perturbation than removing a "
          "fifth of the\n    subsets. Matched at the same fraction, the two are "
          "the comparison to quote.")
    print("\n  The analytic df figures above need none of this and are the "
          "statement to lead\n  with: 45 df against 4 is the finding, and it is "
          "a fact about the design.")
    if members is None:
        print("  The seed route is a MODEL of the seed spread, not a resample of "
              "it: per-member\n  phi is not on disk. Pass --member_npz for the "
              "exact version. The analytic df\n  figures above need neither.")

    print(f"\n--- acceptance checks ---")
    for k in ("median_share", "iqr", "n_regions", "n_with_data_component"):
        c = checks[k]
        ok = "yes" if c["matches"] else "NO"
        print(f"  {k:>22s}: published {c['published']}  computed {c['computed']}  {ok}")
    print(f"  {'regions dropped':>22s}: published "
          f"{checks['n_dropped']['published']}  computed "
          f"{checks['n_dropped']['computed']}")
    b = checks["loso_vs_full_grid"]
    for tag in ("unmatched", "matched"):
        v = b[tag]
        print(f"  {'LOSO vs full, ' + tag:>22s}: {v['lo']:.3f}-{v['hi']:.3f} "
              f"around {v['full']:.3f} "
              f"({'brackets' if v['brackets'] else 'one-sided'})")
    print("      Reported, NOT a pass/fail. The spec reads one-sidedness as a "
          "broken\n      recomputation; it is the expected outcome of selection "
          "and of Jensen, in\n      opposite directions. The reconstruction gate "
          "at the top is what proves the\n      estimator, and it is exact.")

    print(f"\n--- coupling to W-20, which this does not measure ---")
    print("  The training subsets are BALANCED, 3 sham and 4 BDL+vehicle each, so "
          "section 5's\n  justification that subsets \"differ in fibrosis "
          "composition\" is contradicted by the\n  design. Whatever the numbers "
          "above say, that sentence has to be rewritten to say\n  data exposure "
          "comes from WHICH individual cases a member saw. If the balancing\n  "
          "extended to time points too, say so — it tightens the claim further.")

    print(f"\nwrote {args.outdir}/loso_shares.csv, seed_subsample.csv and "
          f"stability_summary.json")


if __name__ == "__main__":
    main()
