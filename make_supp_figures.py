"""The two supplement figures of FIGURE_REQUESTS.md — F-1 and F-2.

Neither changes a result. F-1 re-renders a plot that already exists so it can sit
beside the other paper assets; F-2 is new but illustrative, showing on real
tissue what section 5 already asserts.

    python make_supp_figures.py stability      --per_region ... --outdir figures/
    python make_supp_figures.py region-mapping --per_region ... --he ... --sr ...

Conventions both obey, each of which is a decision the manuscript already made
rather than a style preference (FIGURE_REQUESTS.md section 1):

* **PDF**, `bbox_inches="tight"`, `pad_inches=0.01`, `style(7.0)` — the same
  helpers `make_paper_figures.py` uses, imported rather than copied so the two
  cannot drift.
* **Authored at the width it is published at.** A figure drawn at 9 in and
  scaled into a 3.281 in column loses roughly two thirds of its label size, so
  F-1 is drawn at `COLUMN_IN` and F-2 at the full text width.
* **No title inside the figure.** LaTeX captions the float; a figure carrying its
  own title double-prints it. Panel labels are plain noun phrases.
* **The paper's symbols.** CPA, never `task_specific_value`; sigma, never
  "uncertainty (SD)"; e, never "MAE".
* A **draft caption** is written beside each figure, because the caption is also
  where a value with no float of its own gets anchored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from make_paper_figures import COLUMN_IN, style, tidy

# WACV two-column text width. F-2 is allowed it: the supplement has no page
# limit and legibility beats compactness there.
TEXT_IN = 6.875

INK = "#0b0b0b"
GREY = "#52514e"


# ==========================================================================
# F-1 — the data-exposure stability figure, for publication
# ==========================================================================

# Section 8's Limitations already quotes these, and they are committed. A
# re-render that does not reproduce them is not a re-render of the same run.
EXPECT = {
    "median_share": 0.508,
    "loso_lo": 0.282,
    "loso_hi": 0.562,
    "n_replicates": 5,
    "seed_spread_max": 0.005,
}


def _stability_numbers(per_region: Path, descriptor: str, n_boot: int,
                       n_draws: int, seed: int) -> dict:
    """Rerun the W-29 computation. Imported, not reimplemented.

    The science is done and must not change here, so every number comes from
    `stability_data_exposure.py` — including the reconstruction gate, which is
    what proves the recomputation is the paper's estimator.
    """
    from stability_data_exposure import (cluster_bootstrap_median,
                                         components_from_folds, df_asymmetry,
                                         jackknife, leave_one_subset_out,
                                         load_folds, matched_summaries,
                                         reconstruction_check,
                                         seed_subsample_parametric, share,
                                         summarise_share)

    t = pd.read_csv(per_region)
    fold_mu, fold_var, fold_names = load_folds(t, descriptor)
    K, R = fold_mu.shape

    counts = None
    sj = per_region.parent / "summary.json"
    if sj.exists():
        try:
            j = json.load(open(sj))
            v = (j.get("variance", {}) or {}).get("n_seeds_per_fold") \
                or j.get("n_seeds_per_fold")
            if v:
                counts = np.asarray(v, dtype=np.float64)
        except (json.JSONDecodeError, OSError):
            counts = None
    if counts is None:
        raise SystemExit(
            f"cannot determine the seeds per fold: {sj} has no "
            f"n_seeds_per_fold and per_region.csv does not record S.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = components_from_folds(fold_mu, fold_var, counts)
        check = reconstruction_check(t, descriptor, full)
        s_full = share(full)
        summ = summarise_share(s_full, R)
        boot = cluster_bootstrap_median(
            s_full, t["wsi"].to_numpy() if "wsi" in t.columns else np.zeros(R),
            n_boot, seed)
        loso, loso_shares = leave_one_subset_out(fold_mu, fold_var, counts,
                                                 full["procedural"], True)
        m_full, m_rows, _ = matched_summaries(s_full, loso_shares)
        summ.update(m_full)
        for r, m in zip(loso, m_rows):
            r.update(m)
        jk = jackknife(summ["median_share"], [r["median_share"] for r in loso])
        S = int(np.min(counts))
        sizes = sorted({max(2, int(round(S * (1 - 1.0 / K)))), max(2, S // 2)},
                       reverse=True)
        seeds = {s: seed_subsample_parametric(fold_mu, fold_var, counts, s,
                                              n_draws, seed) for s in sizes}
    return {"full": summ, "boot": boot, "loso": loso, "jk": jk, "seeds": seeds,
            "df": df_asymmetry(K, counts), "check": check, "K": K, "S": S,
            "n_regions": R}


def _stability_checks(res: dict) -> List[dict]:
    """FIGURE_REQUESTS F-1's acceptance table, as pass/fail rows."""
    med = res["full"]["median_share"]
    lo, hi = res["jk"].get("range_lo"), res["jk"].get("range_hi")
    spread = max((float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
                 for v in res["seeds"].values()) if res["seeds"] else float("nan")
    return [
        {"quantity": "full-grid median share", "must_be": EXPECT["median_share"],
         "got": med,
         # The paper prints three decimals, so that is the tolerance its own
         # precision allows.
         "ok": bool(abs(med - EXPECT["median_share"]) <= 0.0005)},
        {"quantity": "LOSO range low", "must_be": EXPECT["loso_lo"], "got": lo,
         "ok": bool(lo is not None and abs(lo - EXPECT["loso_lo"]) <= 0.0005)},
        {"quantity": "LOSO range high", "must_be": EXPECT["loso_hi"], "got": hi,
         "ok": bool(hi is not None and abs(hi - EXPECT["loso_hi"]) <= 0.0005)},
        {"quantity": "n LOSO replicates", "must_be": EXPECT["n_replicates"],
         "got": len(res["loso"]),
         "ok": len(res["loso"]) == EXPECT["n_replicates"]},
        {"quantity": "seed-subsample spread", "must_be": f"< {EXPECT['seed_spread_max']}",
         "got": spread,
         "ok": bool(np.isfinite(spread) and spread < EXPECT["seed_spread_max"])},
    ]


def fig_stability(res: dict, outpath: Path, panels: str,
                  height: float = 2.30) -> None:
    """One column, no suptitle, CPA rather than the internal descriptor key.

    `panels="left"` ships the left panel alone, which FIGURE_REQUESTS F-1 item 6
    asks for when two will not fit legibly at column width — and at 3.281 in they
    do not: two panels leave about 1.5 in each, and the left one alone carries
    three x-tick groups with two-line labels. The right panel's degrees-of-freedom
    point is already stated in words in section 8 and stays there.

    Three things are kept exactly as they were, because they are the finding:
    the low leave-one-subset-out replicate is drawn unhighlighted and the y-axis
    is **not** clipped to hide it; the case-bootstrap band and the full-grid line
    stay; and both seed-subsample groups stay.
    """
    K, S = res["K"], res["S"]
    loso, full, boot, seeds = res["loso"], res["full"], res["boot"], res["seeds"]
    two = panels == "both"
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(COLUMN_IN, height), squeeze=False)
    ax = axes[0][0]

    med = full["median_share"]
    if boot:
        ax.axhspan(boot["ci_lo"], boot["ci_hi"], color="#2a78d6", alpha=0.16,
                   linewidth=0, zorder=2, label="case bootstrap 95% CI")
    ax.axhline(med, color=INK, linewidth=0.9, linestyle=(0, (4, 2)), zorder=5,
               label=f"full grid {med:.3f}")
    ax.scatter(np.linspace(-0.13, 0.13, len(loso)),
               [r["median_share"] for r in loso], s=17, color="#eb6834",
               zorder=6, edgecolor="white", linewidth=0.35,
               label=f"leave one subset out")
    ticks, labels = [0], [f"drop 1 of {K}\nsubsets"]
    greens = ["#1baf7a", "#8ad3b6"]
    for i, (s_sub, vals) in enumerate(sorted(seeds.items(), reverse=True)):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if not len(v):
            continue
        ax.scatter(np.full(len(v), float(i + 1))
                   + np.random.default_rng(0).uniform(-0.17, 0.17, len(v)),
                   v, s=2.6, color=greens[i % len(greens)], alpha=0.45,
                   zorder=4, edgecolor="none",
                   label=f"seeds {S}$\\rightarrow${s_sub}")
        ticks.append(i + 1)
        labels.append(f"drop to {s_sub}\nof {S} seeds")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, max(ticks) + 0.5)
    ax.set_ylabel("data-exposure share of $\\sigma^2$", color=GREY, labelpad=1.5)
    tidy(ax)
    # Above the axes, not in a corner: the leave-one-out replicates span the
    # full y-range on the real data, so every in-axes corner collides with the
    # finding. Only the line and the band need naming — the three point groups
    # are already named by the x-axis.
    h, l = ax.get_legend_handles_labels()
    keep = [(hh, ll) for hh, ll in zip(h, l)
            if "seeds" not in ll and "leave one" not in ll]
    leg = ax.legend([k[0] for k in keep], [k[1] for k in keep],
                    frameon=False, loc="lower center", ncol=2,
                    bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
                    borderaxespad=0.0, columnspacing=1.1, handletextpad=0.4,
                    fontsize=plt.rcParams["font.size"] - 1.0)
    for txt in leg.get_texts():
        txt.set_color(INK)

    if two:
        ax = axes[0][1]
        d = res["df"]
        nus = np.array([d["df_data_exposure"], d["df_procedural"]])
        ax.bar(range(2), np.sqrt(2.0 / nus) * 100,
               color=["#1baf7a", "#eb6834"], width=0.55, zorder=4)
        ax.set_xticks(range(2))
        ax.set_xticklabels([f"data\nexposure\n{int(nus[0])} df",
                            f"procedural\n{int(nus[1])} df"])
        ax.set_ylabel("relative SE (%)", color=GREY, labelpad=1.5)
        tidy(ax)

    fig.tight_layout(pad=0.25)
    fig.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {outpath}")


# LaTeX braces and str.format do not mix — one unescaped `\\caption{` is enough
# to raise, and the failure lands after the figure is already written. Explicit
# <<TOKEN>> substitution has no escaping rules at all.
STABILITY_CAPTION = """\
\\caption{Stability of the data-exposure component of the CPA variance. Each
point is the median data-exposure share of $\\sigma^2$ over the <<NREG>>
regions, recomputed under a resampling of the ensemble grid. The five orange
points leave out one training subset at a time and spread from <<LO>> to <<HI>>;
the green points redraw the seeds within every subset and stay within
<<SEEDSPREAD>> of one another. The dashed line is the full-grid value of <<MED>>
and the band its case-clustered 95\\% confidence interval. Read the two groups
against each other: the share is stable to which seeds a member was trained with
and not to which subset was withheld, because the between-subset term rests on
$K-1=<<DFDATA>>$ degrees of freedom against the procedural term's
$K(S-1)=<<DFPROC>>$. The low replicate is shown as measured, not treated as an
outlier.}
\\label{fig:dataexposure}"""


def fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace(f"<<{k}>>", str(v))
    left = re.findall(r"<<[A-Z_]+>>", out)
    if left:
        raise ValueError(f"caption placeholders not filled: {sorted(set(left))}")
    return out


def run_stability(args) -> int:
    style(7.0)
    res = _stability_numbers(args.per_region, args.descriptor, args.n_boot,
                             args.n_draws, args.seed)
    print(f"\n--- gate: reconstruction against the phi run's own columns ---")
    for k in ("procedural", "data", "total"):
        c = res["check"].get(k, {})
        if c.get("present"):
            print(f"  {k:>11s}: max |diff| {c['max_abs_diff']:.3e}  "
                  f"{'ok' if c['matches'] else 'MISMATCH'}")
    checks = _stability_checks(res)
    print(f"\n--- acceptance checks (FIGURE_REQUESTS F-1) ---")
    for c in checks:
        got = c["got"]
        got = f"{got:.4f}" if isinstance(got, float) else str(got)
        print(f"  {c['quantity']:>24s}  must be {str(c['must_be']):>8s}  "
              f"got {got:>8s}  {'ok' if c['ok'] else 'NO'}")
    bad = [c["quantity"] for c in checks if not c["ok"]]
    if bad and not args.force:
        print(f"\n[STOP] {len(bad)} acceptance check(s) failed: {bad}\n"
              f"  The paper quotes these numbers and they are committed, so a "
              f"figure that\n  disagrees with them is not shipped. Either "
              f"--per_region points at a different\n  run, or the numbers moved "
              f"and section 8 needs updating first.\n"
              f"  Re-run with --force to render anyway (for inspection only).")
        return 1
    if bad:
        print(f"\n[warn] rendering despite {len(bad)} failed check(s) because "
              f"--force was given. Do NOT ship this file.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "stability_data_exposure.pdf"
    fig_stability(res, out, args.panels)

    seed_spread = max((float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)
                      for v in res["seeds"].values())
    cap = fill(STABILITY_CAPTION,
               NREG=res["full"]["n_with_data"],
               LO=f"{res['jk']['range_lo']:.3f}",
               HI=f"{res['jk']['range_hi']:.3f}",
               SEEDSPREAD=f"{seed_spread:.3f}",
               MED=f"{res['full']['median_share']:.3f}",
               DFDATA=int(res["df"]["df_data_exposure"]),
               DFPROC=int(res["df"]["df_procedural"]))
    (args.outdir / "stability_data_exposure.caption.tex").write_text(cap + "\n")
    print(f"wrote {args.outdir / 'stability_data_exposure.caption.tex'} "
          f"(draft caption — the manuscript will adapt it)")
    if args.panels != "both":
        print("\n[note] left panel only. At one column two panels leave ~1.5 in "
              "each and the\n  labels stop being legible; the df point is "
              "already in section 8's words.\n  --panels both overrides.")
    return 0


# ==========================================================================
# F-2 — the spatial region-mapping figure
# ==========================================================================

def strip_prefix(stem: str) -> str:
    """`SR_d31_BDL+A_M2` and `HE_d31_BDL+A_M2` are one case.

    The same rule `apply_he_mask.py`, `compare_psr.py` and
    `compute_phi_reference.py` use: phi is gridded on the H&E while the SR
    slides carry their own prefix.
    """
    return stem.split("_", 1)[1] if "_" in stem else stem


def find_slide(directory: Path, wsi_stem: str) -> Path:
    """Match by stem, then by stem with the first token stripped from either side."""
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    cands = [p for p in sorted(directory.iterdir()) if p.suffix in exts]
    if not cands:
        raise SystemExit(f"no TIFs in {directory}")
    key = strip_prefix(wsi_stem)
    for p in cands:
        if p.stem == wsi_stem:
            return p
    hits = [p for p in cands if strip_prefix(p.stem) == key]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Two files collapsing to one key is fatal, not last-one-wins: the wrong
        # slide beside the right one is invisible in the output.
        raise SystemExit(f"{len(hits)} slides in {directory} match '{key}': "
                         f"{[p.name for p in hits]}")
    raise SystemExit(f"no slide in {directory} matches '{wsi_stem}' (key "
                     f"'{key}'). Present: {[p.stem for p in cands[:6]]}...")


def read_slide(path: Path, level: Optional[int]) -> np.ndarray:
    """RGB array. `level` uses a pyramid level if the file has one.

    A UC slide is ~35k x 40k, so the full read is ~4 GB before any copy. Prefer a
    level where the file is pyramidal; the fallback is the full read, and the run
    says which it took so a memory failure is diagnosable from the log.
    """
    import tifffile
    if level is not None:
        try:
            a = tifffile.imread(str(path), level=level)
            print(f"  read {path.name} at pyramid level {level}: {a.shape}")
            return a
        except (TypeError, ValueError, IndexError, KeyError) as e:
            print(f"  [note] level={level} unavailable for {path.name} ({e}); "
                  f"reading full resolution")
    a = tifffile.imread(str(path))
    print(f"  read {path.name} at full resolution: {a.shape}")
    return a


def as_rgb(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    if a.shape[0] in (3, 4) and a.ndim == 3 and a.shape[0] < a.shape[-1]:
        a = np.moveaxis(a, 0, -1)          # channel-first pages
    return a[:, :, :3]


def thumbnail(a: np.ndarray, max_px: int) -> Tuple[np.ndarray, int]:
    """Stride-downsample to at most `max_px` on the long side. Returns the stride."""
    step = max(1, int(np.ceil(max(a.shape[:2]) / max_px)))
    return a[::step, ::step], step


def fit_for_print(a: np.ndarray, inches: float, dpi: int) -> np.ndarray:
    """Resample to the pixels the printed panel can actually show.

    Embedding a 2048 px crop in a 3.4 in panel is ~600 dpi of detail no printer
    or screen renders, and it cost 12 MB in the first version of this figure —
    on the edge of what a submission system accepts, for nothing visible. The
    panel width and the target dpi fix the only pixel count that matters.
    """
    target = int(round(inches * dpi))
    long_side = max(a.shape[:2])
    if long_side <= target:
        return a
    from PIL import Image
    sc = target / float(long_side)
    size = (max(1, int(round(a.shape[1] * sc))),
            max(1, int(round(a.shape[0] * sc))))
    return np.asarray(Image.fromarray(a).resize(size, Image.LANCZOS))


def kept_regions(t: pd.DataFrame, wsi: str) -> Tuple[pd.DataFrame, int, int]:
    g = t[t["wsi"].astype(str).str.contains(wsi, regex=False)] \
        if wsi not in set(t["wsi"].astype(str)) else t[t["wsi"].astype(str) == wsi]
    if g.empty:
        raise SystemExit(f"no rows for wsi '{wsi}' in the per-region table. "
                         f"Present: {sorted(set(t['wsi'].astype(str)))[:6]}...")
    need = {"y0", "y1", "x0", "x1", "wsi_h", "wsi_w"}
    if not need <= set(g.columns):
        raise SystemExit(f"per_region.csv lacks {sorted(need - set(g.columns))}; "
                         f"the region boxes and the frame travel in those columns")
    return g, int(g["wsi_h"].iloc[0]), int(g["wsi_w"].iloc[0])


def check_frame(name: str, arr: np.ndarray, h: int, w: int,
                tile_size: int) -> None:
    """The SR must share the H&E frame, or the grid is not the same tissue.

    Same rule as `compute_phi_reference.py`: phi is gridded on a reconstruction,
    which `utils.reconstruct_wsi` truncates to a whole number of tiles, so the
    original is larger by up to one tile at the same origin and scale. An excess
    below `--tile_size` is the expected truncation and is accepted with a note;
    anything larger, or any shortfall, means the two are not the same frame and
    the figure would contradict section 6.
    """
    ah, aw = arr.shape[:2]
    dh, dw = ah - h, aw - w
    if dh < 0 or dw < 0:
        raise SystemExit(
            f"{name} is {ah}x{aw}, SHORTER than the phi frame {h}x{w}. The grid "
            f"would index pixels the image does not have.")
    if dh >= tile_size or dw >= tile_size:
        raise SystemExit(
            f"{name} is {ah}x{aw} against the phi frame {h}x{w} — an excess of "
            f"{dh}x{dw}, at or beyond one tile ({tile_size}). Truncation cannot "
            f"lose a whole tile, so these are different frames and the grid does "
            f"not index the same tissue in both.")
    if dh or dw:
        print(f"  [note] {name} exceeds the phi frame by {dh}x{dw} px, below one "
              f"tile — the expected reconstruction truncation, same origin and "
              f"scale.")


def draw_grid(ax, g: pd.DataFrame, side: int, h: int, w: int, step: int,
              lw: float = 0.35) -> Tuple[int, int]:
    """Every grid cell, kept ones outlined and dropped ones ghosted.

    The kept set is read from `per_region.csv` rather than recomputed, so
    "dropped regions are the ones below 25% coverage" holds by construction
    rather than by reimplementing the rule and hoping it agrees.
    """
    kept = {(int(r.y0), int(r.x0)) for r in g.itertuples()}
    n_kept = n_drop = 0
    for y0 in range(0, h, side):
        for x0 in range(0, w, side):
            if y0 + side > h or x0 + side > w:
                continue                     # drop_partial, as the grid was built
            is_kept = (y0, x0) in kept
            n_kept += is_kept
            n_drop += not is_kept
            ax.add_patch(Rectangle(
                (x0 / step, y0 / step), side / step, side / step,
                fill=not is_kept,
                facecolor="#ffffff" if is_kept else "#6f6e6a",
                # Light: a dropped cell is context, not a finding, and a heavy
                # veil hides the tissue the figure exists to show.
                alpha=1.0 if is_kept else 0.20,
                edgecolor="#1baf7a" if is_kept else "#9a9994",
                linewidth=lw if is_kept else lw * 0.7, zorder=5))
    return n_kept, n_drop


# Nice round bar lengths in micrometres, largest first.
BAR_UM = (5000, 2000, 1000, 500, 250, 200, 100, 50, 20)


def scale_bar(ax, n_px: int, mpp: float, frac: float = 0.35) -> float:
    """Longest round bar that fits in `frac` of the crop. Returns the length in um.

    Fixing the bar at 500 um is what the first version did, and at 0.221 um/px
    that is 2262 px against a 2048 px region — a bar wider than the thing it
    measures. The length has to come from the crop.
    """
    um = next((u for u in BAR_UM if u / mpp <= n_px * frac), BAR_UM[-1])
    length = um / mpp
    y, x0 = n_px * 0.945, n_px * 0.045
    ax.plot([x0, x0 + length], [y, y], color=INK, linewidth=1.8,
            solid_capstyle="butt", zorder=9)
    ax.text(x0 + length / 2, y - n_px * 0.022, f"{um:g} $\\mu$m", ha="center",
            va="bottom", fontsize=plt.rcParams["font.size"] - 0.5, color=INK,
            zorder=9)
    return float(um)


REGION_CAPTION = """\
\\caption{The analysis grid on one case (<<CASE>>), drawn in the same
coordinates on both stains. Top: the H\\&E with the <<SIDE>>\\,px
($\\approx$<<SIDEMM>>\\,mm) region grid overlaid, and beside it the registered
Sirius Red for the same case under the identical grid. Cells outlined in green
are the <<NKEPT>> regions that clear the <<MINTISSUE>>\\,\\% tissue-coverage
rule and enter the results; shaded cells are the <<NDROP>> that do not. Bottom:
region <<RIDX>> of the same case at full resolution in both stains. The pair is
visibly the same tissue area and visibly not the same structures, which is the
correspondence the region scale is chosen to provide: the grid is defined once in
the H\\&E frame and both stains are read in it, so a region means the same piece
of tissue in both while nothing below region scale does. This case was chosen as
an ordinary one rather than the best registered.}
\\label{fig:regionmapping}"""


def run_region_mapping(args) -> int:
    style(7.0)
    t = pd.read_csv(args.per_region)
    if args.list_cases:
        c = (t.groupby("wsi").size().sort_values()
             .rename("n_regions").reset_index())
        print("\ncases by region count — pick one near the MEDIAN, not the top:\n")
        for i, r in c.iterrows():
            mark = "  <- median" if i == len(c) // 2 else ""
            print(f"  {r['wsi']:<40s} {r['n_regions']:>5d}{mark}")
        return 0
    if not args.wsi:
        raise SystemExit("--wsi is required (or --list_cases to choose one)")

    g, h, w = kept_regions(t, args.wsi)
    wsi = str(g["wsi"].iloc[0])
    side = int(g["y1"].iloc[0] - g["y0"].iloc[0])
    print(f"case {wsi}: frame {h}x{w}, region side {side} px "
          f"({side * args.mpp / 1000.0:.2f} mm), {len(g)} kept regions")

    he_path = find_slide(args.he, wsi)
    sr_path = find_slide(args.sr, wsi)
    print(f"reading slides (a UC case is several GB at full resolution):")
    he = as_rgb(read_slide(he_path, args.level))
    sr = as_rgb(read_slide(sr_path, args.level))
    check_frame("H&E", he, h, w, args.tile_size)
    check_frame("Sirius Red", sr, h, w, args.tile_size)

    if args.region_index is None:
        # The middle of the kept list: an ordinary region, for the same reason
        # the case should be an ordinary one.
        row = g.iloc[len(g) // 2]
    else:
        m = g[g["region_index"] == args.region_index]
        if m.empty:
            raise SystemExit(f"region {args.region_index} is not among the kept "
                             f"regions for {wsi}")
        row = m.iloc[0]
    y0, y1, x0, x1 = (int(row.y0), int(row.y1), int(row.x0), int(row.x1))
    ridx = int(row.region_index)
    print(f"magnifying region {ridx} at y {y0}:{y1}, x {x0}:{x1} — the SAME box "
          f"in both stains")

    panel_in = TEXT_IN / 2.0
    # Stride first — it is the cheap operation on a multi-GB array — then
    # resample the small result to what the panel prints.
    he_t, step = thumbnail(he[:h, :w], args.thumb_px)
    sr_t, step_sr = thumbnail(sr[:h, :w], args.thumb_px)
    if step != step_sr:
        raise SystemExit(f"thumbnail strides differ ({step} vs {step_sr}); the "
                         f"two grids would not be drawn at the same scale")

    panel_w = TEXT_IN / 2.0
    fig_h = panel_w * (h / float(w)) + panel_w + 0.55
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_IN, fig_h),
                             gridspec_kw={"height_ratios":
                                          [panel_w * h / float(w), panel_w]})
    for ax, img, label in ((axes[0][0], he_t, "H&E, analysis grid"),
                           (axes[0][1], sr_t, "Sirius Red, same grid")):
        # The grid is drawn in `step` units, so the displayed image must stay on
        # that scale: `extent` maps the resampled pixels back onto it rather
        # than shifting every rectangle.
        shown = fit_for_print(img, panel_in, args.dpi)
        ax.imshow(shown, interpolation="nearest",
                  extent=(0, img.shape[1], img.shape[0], 0))
        n_kept, n_drop = draw_grid(ax, g, side, h, w, step)
        ax.add_patch(Rectangle((x0 / step, y0 / step), side / step, side / step,
                               fill=False, edgecolor="#eb6834", linewidth=1.4,
                               zorder=8))
        ax.set_title(label, fontsize=plt.rcParams["font.size"], color=INK,
                     pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c9c9c4")
    for ax, img, label in (
            (axes[1][0], he[y0:y1, x0:x1], f"region {ridx}, H&E"),
            (axes[1][1], sr[y0:y1, x0:x1], f"region {ridx}, Sirius Red")):
        shown = fit_for_print(img, panel_in, args.dpi)
        # Same trick: the scale bar is placed in SOURCE pixels, so the axis keeps
        # source coordinates and the bar stays honest whatever the resample did.
        ax.imshow(shown, interpolation="nearest",
                  extent=(0, img.shape[1], img.shape[0], 0))
        bar_um = scale_bar(ax, min(img.shape[:2]), args.mpp)
        ax.set_title(label, fontsize=plt.rcParams["font.size"], color=INK, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#eb6834")
            s.set_linewidth(1.2)

    fig.tight_layout(pad=0.3)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "region_mapping.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.01,
                dpi=args.dpi)
    print(f"  {out.stat().st_size / 1e6:.1f} MB at {args.dpi} dpi "
          f"(--dpi lower if the submission caps figure size)")
    plt.close(fig)
    print(f"wrote {out}")

    cap = fill(REGION_CAPTION,
               CASE=wsi.replace("_", "\\_"), SIDE=side,
               SIDEMM=f"{side * args.mpp / 1000.0:.1f}",
               NKEPT=n_kept, NDROP=n_drop,
               MINTISSUE=f"{args.min_tissue_fraction * 100.0:.0f}",
               RIDX=ridx)
    (args.outdir / "region_mapping.caption.tex").write_text(cap + "\n")
    print(f"wrote {args.outdir / 'region_mapping.caption.tex'}")

    print(f"\n--- acceptance checks (FIGURE_REQUESTS F-2) ---")
    print(f"  grid identical in both stains        : one geometry, one draw_grid "
          f"call per panel")
    print(f"  dropped regions are the <{args.min_tissue_fraction:.0%} ones : kept "
          f"set read from per_region.csv, not recomputed")
    print(f"  magnified pair is the same region    : region {ridx}, box "
          f"y {y0}:{y1} x {x0}:{x1}, both stains")
    print(f"  no identifying text in any panel     : the case name is in the "
          f"CAPTION only")
    print(f"\n[check by eye before shipping] Whole-slide formats embed a label "
          f"image that\n  routinely photographs a handwritten case identifier. "
          f"This reads the main image,\n  not the label, but open "
          f"{out.name} and confirm no barcode, scanner overlay or\n  handwriting "
          f"survived into either thumbnail. The review copy is double-blind.")
    return 0


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("stability", help="F-1: re-render the W-29 figure as PDF")
    a.add_argument("--per_region", type=Path, required=True)
    a.add_argument("--outdir", type=Path, required=True)
    a.add_argument("--descriptor", default="task_specific_value")
    a.add_argument("--panels", default="left", choices=["left", "both"],
                   help="'left' is the finding; 'both' adds the df bars, which "
                        "at one column are cramped and already stated in words.")
    a.add_argument("--n_boot", type=int, default=2000)
    a.add_argument("--n_draws", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--force", action="store_true",
                   help="Render even if an acceptance check fails. For "
                        "inspection only — the paper quotes those numbers.")

    b = sub.add_parser("region-mapping", help="F-2: the grid on both stains")
    b.add_argument("--per_region", type=Path, required=True)
    b.add_argument("--he", type=Path, help="directory of H&E WSIs")
    b.add_argument("--sr", type=Path, help="directory of registered SR WSIs")
    b.add_argument("--outdir", type=Path, required=True)
    b.add_argument("--wsi", default=None,
                   help="case stem as it appears in per_region.csv")
    b.add_argument("--list_cases", action="store_true",
                   help="List cases by region count and stop. Pick one near the "
                        "median: an unrepresentative figure is worse than a "
                        "plain one.")
    b.add_argument("--region_index", type=int, default=None,
                   help="Which region to magnify. Default: the middle of the "
                        "kept list.")
    b.add_argument("--mpp", type=float, default=0.221)
    b.add_argument("--min_tissue_fraction", type=float, default=0.25,
                   help="Reported in the caption. The kept set comes from "
                        "per_region.csv, so this does not re-filter anything.")
    b.add_argument("--tile_size", type=int, default=512,
                   help="tile.py --tile_size, for the frame check. NOT "
                        "--resize_to: reconstructions sit at source resolution.")
    b.add_argument("--thumb_px", type=int, default=1600,
                   help="Stride target before the print resample. "
                        "Only affects speed and memory.")
    b.add_argument("--level", type=int, default=None,
                   help="Pyramid level for the read, if the TIFs have one. "
                        "Falls back to full resolution with a note.")
    b.add_argument("--dpi", type=int, default=300,
                   help="Print resolution the rasters are resampled to. 300 is "
                        "standard for print; the panels are ~3.4 in wide, so "
                        "higher only grows the file.")

    args = ap.parse_args()
    if args.cmd == "stability":
        return run_stability(args)
    return run_region_mapping(args)


if __name__ == "__main__":
    sys.exit(main())
