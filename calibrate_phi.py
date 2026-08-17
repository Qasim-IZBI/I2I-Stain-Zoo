#!/usr/bin/env python
"""Does the ensemble's spread predict its structural error?

Ensemble variance measures disagreement between members, not error. The BMVC
2026 result is that cycle-reconstruction error — the cheap self-consistency
proxy — does not calibrate it. This asks the same question against an external,
task-relevant target: φ_struct of the generated stain versus φ_struct of the
**real** tissue, per descriptor, per region.

    python calibrate_phi.py \\
        --phi_csv    ./phi_uncertainty/per_region.csv \\
        --real_psr   /path/psr_masks/real/psr_masks_wsi_final \\
        --real_lumen /path/lumen_masks_real \\
        --he_masks   /path/HE_tissue \\
        --outdir     ./calibration_phi/

The two reference arms are independent, and either may be omitted:

* `--real_lumen` scores lumen_fraction and the two lumen Betti numbers against
  the **real H&E** — the same physical section the model generated from, so
  there is no level offset and no biological floor. Its regions are on the H&E
  frame, the same frame the virtual run used, so pairing is exact.
* `--real_psr` scores CPA and the collagen Betti/dispersion terms against the
  real SR, which sits at a different section level. Pairing region *r* across
  the two requires the SR to share the H&E's coordinate frame; the run checks
  the geometry and refuses rather than measuring different tissue.

Regions come from `--phi_csv` verbatim — the same y0/y1/x0/x1 boxes the virtual
run used — rather than rebuilding a grid, so the two sides cannot drift apart
through a parameter that differs by one.

Outputs
-------
per_region_calibration.csv   mu, sd, reference, error and z per region x descriptor
summary.json                 Spearman rho, E|z|, ECE and reliability bins
calibration_phi.png          reliability per descriptor, plus a rho summary
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
from uncertainty_phi.descriptors import (
    PHI_NAMES,
    PHI_REFERENCE,
    WHITE_THRESH,
    he_tissue_footprint,
    lumen_descriptors,
    phi_struct,
    tissue_footprint_from_mask,
)
from uncertainty_phi.ensemble import (
    _stem_index,
    load_label_mask,
    load_rgb,
    load_roi_mask,
)
from uncertainty_phi.regions import SOURCE_MPP
from apply_he_mask import normalize_stem

# For a Gaussian error of scale sigma, E|e| = sigma * sqrt(2/pi). A reliability
# line of slope 1 would therefore call a perfectly calibrated ensemble 20%
# over-confident.
HALF_NORMAL = float(np.sqrt(2.0 / np.pi))

COLLAGEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "psr"]
LUMEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "he"]

C_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948")
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e3e3df"


def _indexed(directory: Optional[Path], strip_prefix: bool, what: str) -> dict:
    """stem -> path, optionally keyed on the stem minus its first token.

    The real PSR masks are named after the SR slides while φ is gridded on the
    H&E, so `SR_slide` has to reach `HE_slide` — the rule `apply_he_mask.py` and
    `compare_psr.py` already carry. A collision is fatal rather than
    last-one-wins: two files differing only in their prefix collapse to one key,
    and picking either would score one slide's regions against another's tissue.
    """
    if directory is None:
        return {}
    raw = _stem_index(directory)
    if not strip_prefix:
        return raw
    out: dict = {}
    for stem, path in raw.items():
        key = normalize_stem(stem, True)
        if key in out:
            raise SystemExit(
                f"--strip_prefix collapses two files in {what} to the key "
                f"'{key}': {out[key].name} and {path.name}. Scoring a region "
                f"against the wrong slide's tissue is invisible in the output, "
                f"so this is refused rather than resolved arbitrarily."
            )
        out[key] = path
    return out


def reference_phi(df: pd.DataFrame, args) -> pd.DataFrame:
    """φ of the real tissue, on the exact region boxes the virtual run used."""
    sp = args.strip_prefix
    psr_index = _indexed(args.real_psr, sp, "--real_psr")
    lum_index = _indexed(args.real_lumen, sp, "--real_lumen")
    he_index = _indexed(args.he_dir, sp, "--he_dir")
    he_mask_index = _indexed(args.he_masks, sp, "--he_masks")

    rows: List[dict] = []
    missing: List[str] = []
    for wsi, group in df.groupby("wsi", sort=False):
        # The φ side is keyed the same way, so HE_x still reaches HE_x while
        # SR_x also reaches it.
        stem = normalize_stem(Path(str(wsi)).stem, sp)

        labels = load_label_mask(psr_index[stem]) if stem in psr_index else None
        lumen = (load_label_mask(lum_index[stem]) > 0) if stem in lum_index else None
        # The footprint MUST be built the same way the virtual side built it,
        # or the two sides of the comparison divide by different denominators.
        # --he_masks is that way; thresholding is the fallback for a phi run
        # that predates it.
        footprint = None
        if stem in he_mask_index:
            shape = (labels.shape if labels is not None
                     else lumen.shape if lumen is not None else None)
            if shape is not None:
                footprint = tissue_footprint_from_mask(
                    load_roi_mask(he_mask_index[stem], shape))
        elif stem in he_index:
            footprint = he_tissue_footprint(load_rgb(he_index[stem]),
                                            white_thresh=args.white_thresh)

        if labels is None and lumen is None:
            print(f"[skip] no reference for {stem!r}")
            missing.append(stem)
            continue

        # The boxes were built on one frame. A reference of a different size is
        # a different frame, and cropping it at these coordinates scores
        # different tissue under the same region id.
        #
        # "Large enough" is NOT the test. A slide can exceed the region extent
        # and still be a different crop — one UC case is 34794x27942 against the
        # H&E's 32521x23201, which covers every box while aligning with none of
        # them. So compare against the recorded frame where the phi run wrote
        # one, and fall back to the extent bound only for older CSVs.
        want = None
        if {"wsi_h", "wsi_w"} <= set(group.columns):
            h, w = group["wsi_h"].iloc[0], group["wsi_w"].iloc[0]
            if pd.notna(h) and pd.notna(w):
                want = (int(h), int(w))
        need = (int(group["y1"].max()), int(group["x1"].max()))

        for name, arr in (("--real_psr", labels), ("--real_lumen", lumen),
                          ("--he_masks/--he_dir", footprint)):
            if arr is None:
                continue
            got = (int(arr.shape[0]), int(arr.shape[1]))
            if want is not None and got != want:
                raise SystemExit(
                    f"{stem}: {name} is {got[0]}x{got[1]} but the phi run was "
                    f"gridded on {want[0]}x{want[1]}. Different frames — region r "
                    f"is different tissue on each side. Note it is not enough to "
                    f"be larger than the regions: this checks the frame, not the "
                    f"bound. Run scripts/check_frame_alignment.sh; for the "
                    f"collagen arm the SR must be RESAMPLED onto the H&E grid, "
                    f"not merely registered to it."
                )
            if want is None and (got[0] < need[0] or got[1] < need[1]):
                raise SystemExit(
                    f"{stem}: {name} is {got[0]}x{got[1]} but the regions run to "
                    f"{need[0]}x{need[1]}. Different frames. (This CSV predates "
                    f"the wsi_h/wsi_w columns, so only the bound could be "
                    f"checked — re-run compute_phi_uncertainty for an exact "
                    f"frame check.)"
                )

        for row in group.itertuples():
            ys, xs = slice(row.y0, row.y1), slice(row.x0, row.x1)
            out: Dict[str, float] = {"wsi": wsi, "region_index": row.region_index}

            if labels is not None:
                v = phi_struct(labels[ys, xs], None, mpp=args.mpp,
                               min_object_px=args.min_object_px,
                               closing_px=args.closing_px)
                for j in COLLAGEN:
                    out[f"real_{PHI_NAMES[j]}"] = float(v[j])

            if lumen is not None and footprint is not None:
                frac, b0, b1 = lumen_descriptors(lumen[ys, xs], footprint[ys, xs],
                                                 args.mpp)
                out[f"real_{PHI_NAMES[LUMEN[0]]}"] = frac
                out[f"real_{PHI_NAMES[LUMEN[1]]}"] = b0
                out[f"real_{PHI_NAMES[LUMEN[2]]}"] = b1

            rows.append(out)

    if not rows:
        # "check the stems match" is not enough to act on. Show both sides, and
        # test whether dropping the first token would have bridged them — the
        # SR_/HE_ case is the expected one on the collagen arm.
        have = sorted(set(psr_index) | set(lum_index))
        lines = ["no reference regions produced — no WSI in --phi_csv had a "
                 "matching reference mask.",
                 f"  phi_csv     ({len(missing)}): "
                 f"{', '.join(repr(s) for s in missing[:3])}"
                 f"{' ...' if len(missing) > 3 else ''}",
                 f"  reference   ({len(have)}): "
                 f"{', '.join(repr(s) for s in have[:3])}"
                 f"{' ...' if len(have) > 3 else ''}"]
        if not sp and have and missing:
            bridged = ({normalize_stem(s, True) for s in missing}
                       & {normalize_stem(s, True) for s in have})
            if bridged:
                lines.append(
                    f"  => --strip_prefix would match {len(bridged)} of them "
                    f"(e.g. {sorted(bridged)[0]!r}). The real PSR masks are "
                    f"named after the SR slides while phi is gridded on the "
                    f"H&E; add --strip_prefix, as apply_he_mask.py and "
                    f"compare_psr.py do."
                )
        elif sp:
            lines.append("  --strip_prefix is already on, so the two sides "
                         "differ by more than a leading token.")
        raise SystemExit("\n".join(lines))
    return pd.DataFrame(rows)


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


def score(t: pd.DataFrame, n_bins: int) -> List[dict]:
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
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.digitize(sd, edges[1:-1])
        bins = []
        for b in range(n_bins):
            sel = idx == b
            if sel.any():
                bins.append({"mean_sd": float(sd[sel].mean()),
                             "mean_error": float(err[sel].mean()),
                             "n": int(sel.sum())})

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
            "ece_normalised": float(expected_calibration_error(bu, be, bc)),
            "mean_sd": float(sd.mean()),
            "mean_error": float(err.mean()),
            "bins": bins,
        })
    return rows


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


def main() -> None:
    ap = argparse.ArgumentParser("Calibrate phi_struct uncertainty against real tissue")
    ap.add_argument("--phi_csv", type=Path, required=True,
                    help="per_region.csv from compute_phi_uncertainty.py "
                         "(or the pooled one from aggregate_phi_uncertainty.py).")
    ap.add_argument("--real_psr", type=Path, default=None,
                    help="Real SR collagen masks. Scores the four PSR-referenced "
                         "descriptors — needs the SR on the H&E frame.")
    ap.add_argument("--real_lumen", type=Path, default=None,
                    help="Lumen masks of the real H&E, from make_lumen_masks.py. "
                         "Scores the three H&E-referenced descriptors. Same "
                         "physical section, so no floor and no frame question.")
    ap.add_argument("--he_masks", type=Path, default=None,
                    help="H&E tissue masks, for the footprint the lumen "
                         "densities are divided by. Use whatever the phi run "
                         "used: a footprint built differently on the two sides "
                         "means the comparison divides by different denominators.")
    ap.add_argument("--he_dir", type=Path, default=None,
                    help="Real H&E WSIs — the fallback footprint source, by "
                         "thresholding, for a phi run that predates --he_masks.")
    ap.add_argument("--strip_prefix", action="store_true",
                    help="Drop the first '_'-delimited token from every stem "
                         "before matching, so SR_slide reaches HE_slide. Needed "
                         "with --real_psr, whose masks are named after the SR "
                         "slides while phi is gridded on the H&E. Same rule as "
                         "apply_he_mask.py and compare_psr.py.")
    ap.add_argument("--outdir", type=Path, default=Path("calibration_phi"))

    ap.add_argument("--prediction", choices=("grand", "fold"), default="grand",
                    help="'grand' pairs the mean of all members with the total "
                         "spread — the deployed prediction. 'fold' pairs each "
                         "subset's mean with its procedural spread alone, and "
                         "comparing the two is the data-exposure claim. "
                         "[%(default)s]")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_bins", type=int, default=10)

    # must match the run that produced --phi_csv
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP)
    ap.add_argument("--min_object_px", type=int, default=16)
    ap.add_argument("--closing_px", type=int, default=0)
    ap.add_argument("--white_thresh", type=float, default=WHITE_THRESH)
    args = ap.parse_args()

    if args.real_lumen and not (args.he_masks or args.he_dir):
        ap.error("--real_lumen needs --he_masks (or --he_dir): the lumen "
                 "densities are per mm2 of the H&E footprint, and without it "
                 "they are not comparable to the virtual side's")
    if not args.real_psr and not args.real_lumen:
        ap.error("give --real_psr, --real_lumen, or both — there is nothing to "
                 "calibrate against otherwise")

    df = pd.read_csv(args.phi_csv)
    print(f"[1/3] {len(df)} regions over {df['wsi'].nunique()} WSI from {args.phi_csv}")

    ref = reference_phi(df, args)
    print(f"[2/3] reference phi for {len(ref)} regions")

    t = pair(df, ref, args.prediction, args.n_folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins)

    args.outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.outdir / "per_region_calibration.csv", index=False)
    make_figure(t, rows, args.outdir / "calibration_phi.png",
                f"φ_struct calibration — {args.prediction} prediction")

    payload = {
        "prediction": args.prediction,
        "n_regions": int(df.shape[0]),
        "per_descriptor": rows,
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
    print(f"{'descriptor':24s} {'ref':>5s} {'n':>6s} {'rho':>7s} {'p':>9s} "
          f"{'E|z|/0.80':>10s}")
    for r in rows:
        if "spearman_rho" not in r:
            print(f"{r['descriptor']:24s} {'':>5s} {r['n']:>6d}   {r.get('note', '')}")
            continue
        print(f"{r['descriptor']:24s} {r['reference_class']:>5s} {r['n']:>6d} "
              f"{r['spearman_rho']:>7.3f} {r['spearman_p']:>9.2e} "
              f"{r['calibration_ratio']:>10.2f}"
              + (f"   [{r['undefined_because']}]"
                 if r.get("undefined_because") else ""))
    print("\nrho > 0 means uncertain regions are the wrong ones. E|z|/0.80 > 1")
    print("means the ensemble is over-confident: errors exceed its own spread.")
    print(f"\nwrote {args.outdir / 'per_region_calibration.csv'}")
    print(f"wrote {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
