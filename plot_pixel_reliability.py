#!/usr/bin/env python
"""Reliability of the pixel variance components against cycle error.

The pixel-scale counterpart of `calibrate_phi.py`: one panel per unit, three
curves — total, procedural and data-exposure σ — scored against the **same**
regen error, so a difference between them is a difference between the SPREADS
and not between the errors.

    python plot_pixel_reliability.py \\
        --components ./pixel_components/wsi001 \\
        --error_dirs /path/{block}/model_small/regen_error/wsi001/error_npy \\
        --mask_dir /path/tiles/testA/001/masks \\
        --outdir ./pixel_reliability/

`--components` is a `decompose_pixel_uncertainty.py` output root, holding
`total/`, `procedural/` and `data_exposure/` — and `mean_rgb/` too if that run
used `--save_mean_rgb`, which is picked up automatically and is what enables the
partial on the point prediction. Repeat `--components` and `--error_dirs` to pool
WSIs into one figure, which is what makes the slide-clustered interval
meaningful.

Unit of analysis
----------------
The **tile**, clustered on the **slide** — mean σ against mean error per tile,
exactly as `uncertainty_calibration.py`'s across-tile panel, but with all three
components on shared axes and the statistics `calibrate_phi.py` uses: a
bootstrap that resamples slides rather than tiles, and ρ additionally partialled
on the point prediction.

The calibration line is **E|e| = 0.461σ**, not 0.80σ
----------------------------------------------------
The two sides are built differently. `uncertainty.py`'s σ is
√(Σ per-channel variance) = √3·σ_c, while `evaluation.py`'s regen error is the
**mean** over channels of |Δ|. For Gaussian per-channel error
E|e| = σ_c·√(2/π), so the line is σ_map·√(2/π)/√3 = 0.461·σ_map. Using the
descriptor-space 0.80 here would call a perfectly calibrated ensemble 74%
over-confident. It assumes the three channels are exchangeable, which is why the
raw axes are kept rather than normalised — `uncertainty_calibration.py`
normalises precisely because it does not make that assumption.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from calibrate_phi import (
    HALF_NORMAL_PIXEL,
    make_reliability_figure,
    make_risk_coverage_figure,
    risk_coverage,
    score,
    within_slide,
)
from uncertainty_calibration import (
    build_stem_to_wsi,
    find_mask_path,
    load_error_map,
    load_tissue_mask,
)

COMPONENTS = ("total", "procedural", "data_exposure")


def tile_table(component_roots: List[Path], error_dirs: List[List[Path]],
               mask_dirs: List[Optional[Path]],
               stem_to_wsi: Dict[str, str], min_tissue_pixels: int,
               common_mask: bool = True) -> pd.DataFrame:
    """Per tile: mean sigma per component, mean error, mean predicted intensity.

    One row per (tile, component). `mu` is the ensemble-mean intensity, which is
    the pixel analogue of the point prediction — the thing σ is largely a proxy
    for, and therefore the thing the partial has to remove.
    """
    rows: List[dict] = []
    for root, errs, mask_dir in zip(component_roots, error_dirs, mask_dirs):
        # mean_rgb always sits beside the components, written by
        # --save_mean_rgb, and is indexed by the SAME relative path as raw_npy.
        # Taking it as a separate argument invited passing the leaf directory,
        # which doubled the {NNN}/images/ component, found nothing, and left mu
        # NaN — silently dropping the partial rather than failing.
        mean_dir = root / "mean_rgb"
        if not mean_dir.is_dir():
            mean_dir = None
        avail = [c for c in COMPONENTS if (root / c / "raw_npy").is_dir()]
        if not avail:
            raise SystemExit(
                f"{root} holds none of {COMPONENTS}. It should be a "
                f"decompose_pixel_uncertainty.py output root."
            )
        # npy files sit at raw_npy/{NNN}/images/<stem>.npy
        first = root / avail[0] / "raw_npy"
        stems = sorted(p.relative_to(first).with_suffix("")
                       for p in first.rglob("*.npy"))
        if not stems:
            raise SystemExit(f"no .npy under {first}")

        for rel in stems:
            stem = rel.name
            e_full = load_error_map(errs, stem)
            mask = None
            if mask_dir is not None:
                mpath = find_mask_path(mask_dir, stem)
                if mpath is None:
                    continue
                mask = load_tissue_mask(mpath, e_full.shape)
            else:
                mask = np.ones(e_full.shape, bool)

            # One pixel set for ALL components, not one per component. The
            # ANOVA data term is NaN where it came out negative, and masking
            # per component would then average the ERROR over a different set
            # for that curve — on the liver run about 4.7% fewer pixels, which
            # is small but means the three curves are not scored against a
            # strictly identical target, which is the whole point of the figure.
            # Intersecting costs those pixels in every curve and buys an exact
            # comparison.
            common = mask & np.isfinite(e_full)
            if common_mask:
                for comp in avail:
                    u_c = np.load(root / comp / "raw_npy" / rel.with_suffix(".npy"))
                    if u_c.shape == e_full.shape:
                        common = common & np.isfinite(u_c)

            mu = np.nan
            if mean_dir is not None:
                mpath = mean_dir / rel.with_suffix(".tif")
                if mpath.is_file():
                    import tifffile
                    mu = float(tifffile.imread(str(mpath))[common].mean())

            for comp in avail:
                u_full = np.load(
                    root / comp / "raw_npy" / rel.with_suffix(".npy")
                ).astype(np.float64)
                if u_full.shape != e_full.shape:
                    continue
                # A pixel with no finite component cannot enter a mean any more
                # than it can enter a correlation: the ANOVA data term is NaN
                # wherever it came out negative.
                sel = common & np.isfinite(u_full)
                if int(sel.sum()) < min_tissue_pixels:
                    continue
                rows.append({
                    "descriptor": "pixel_intensity",
                    "component": comp,
                    "prediction": "grand",
                    "wsi": stem_to_wsi.get(stem, str(rel.parent)),
                    "region_index": stem,
                    "sd": float(u_full[sel].mean()),
                    "error": float(e_full[sel].mean()),
                    "mu": mu,
                    "n_pixels": int(sel.sum()),
                })
    if not rows:
        raise SystemExit(
            "no tile survived — check --mask_dir matches the tiles, and that "
            "--min_tissue_pixels is not above the tissue in every tile"
        )
    t = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        t["z"] = np.where(t["sd"] > 0, t["error"] / t["sd"], np.nan)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(
        "Reliability of pixel variance components against cycle error")
    ap.add_argument("--components", type=Path, action="append", required=True,
                    help="decompose_pixel_uncertainty.py output root holding "
                         "total/ procedural/ data_exposure/. Repeat per WSI.")
    ap.add_argument("--error_dirs", action="append", required=True,
                    help="Regen error_npy dir(s) for the matching --components. "
                         "Repeat once per --components; give several per WSI as "
                         "a comma-separated list and they are averaged.")
    ap.add_argument("--mask_dir", type=Path, action="append", default=None,
                    help="Tissue masks for the matching --components. Without "
                         "them background dominates every tile mean.")
    ap.add_argument("--tiles_metadata", type=Path, default=None,
                    help="Dataset root, to map tile stems to slides. Without it "
                         "the WSI folder is used, which is enough when each "
                         "--components root is one slide.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--min_tissue_pixels", type=int, default=256)
    ap.add_argument("--per_component_mask", action="store_true",
                    help="Let each component keep its own finite pixels instead "
                         "of intersecting. Retains ~5%% more pixels for total and "
                         "procedural, at the cost of averaging the ERROR over a "
                         "different set per curve — which is exactly what the "
                         "figure is comparing.")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_tiles_per_slide", type=int, default=15)
    ap.add_argument("--coverages", type=float, nargs="*",
                    default=[1.0, 0.9, 0.8, 0.7, 0.5])
    args = ap.parse_args()

    n = len(args.components)
    def _align(v, name):
        if v is None:
            return [None] * n
        if len(v) != n:
            raise SystemExit(
                f"--{name} given {len(v)} time(s) but --components {n}; they are "
                f"positional pairs, so give one of each per WSI"
            )
        return v
    errs = [[Path(x) for x in spec.split(",")] for spec in _align(args.error_dirs, "error_dirs")]
    masks = _align(args.mask_dir, "mask_dir")

    stem_to_wsi: Dict[str, str] = {}
    if args.tiles_metadata:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stem_to_wsi = build_stem_to_wsi(args.tiles_metadata)

    t = tile_table(args.components, errs, masks, stem_to_wsi,
                   args.min_tissue_pixels, common_mask=not args.per_component_mask)
    # by (slide, tile): stems repeat across slides — every WSI has a 0000001 —
    # so counting stems alone under-reports by the slide count
    n_tiles = int(t.groupby(["wsi", "region_index"]).ngroups)
    print(f"[1/2] {n_tiles} tile(s) over {t['wsi'].nunique()} slide(s), "
          f"{t['component'].nunique()} components")
    if not t["mu"].notna().any():
        print("[note] no mean_rgb/ beside the components, so mu is unavailable "
              "and only the raw rho is reported. Re-run "
              "decompose_pixel_uncertainty.py with --save_mean_rgb for the "
              "partial, which is what separates a real signal from sigma "
              "tracking how bright the tile is.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins, args.n_boot, args.seed,
                     half_normal=HALF_NORMAL_PIXEL)
        ws = within_slide(t, args.min_tiles_per_slide, args.n_boot, args.seed)
        rc = risk_coverage(t, sorted(args.coverages, reverse=True),
                           args.n_boot, args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.outdir / "per_tile_components.csv", index=False)
    bins = [{"component": r["component"], **b}
            for r in rows if r.get("bins") for b in r["bins"]]
    if bins:
        pd.DataFrame(bins).to_csv(args.outdir / "reliability_bins.csv", index=False)
    if ws:
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("per_slide_")}
                      for r in ws]).to_csv(args.outdir / "within_slide.csv", index=False)
    if rc:
        pd.DataFrame(rc).to_csv(args.outdir / "risk_coverage.csv", index=False)
        make_risk_coverage_figure(rc, args.outdir / "risk_coverage.png",
                                  "Selective prediction — pixel components")
    make_reliability_figure(
        rows, args.outdir / "reliability_pixel.png",
        "Pixel uncertainty vs cycle error — variance components",
        half_normal=HALF_NORMAL_PIXEL, y_label="mean |cycle error| per tile",
        unit_label="tiles")

    with open(args.outdir / "summary.json", "w") as fh:
        json.dump({"per_component": rows, "within_slide": ws, "risk_coverage": rc,
                   "n_tiles": n_tiles,
                   "n_slides": int(t["wsi"].nunique()),
                   "calibration_line": {
                       "half_normal": HALF_NORMAL_PIXEL,
                       "why": "sigma is sqrt(sum of 3 channel variances); the "
                              "error is the mean over channels, so the Gaussian "
                              "line is sqrt(2/pi)/sqrt(3), not sqrt(2/pi)"},
                   "params": {k: (str(v) if isinstance(v, Path)
                                  else [str(x) for x in v] if isinstance(v, list)
                                  else v) for k, v in vars(args).items()}}, fh,
                  indent=2, default=str)

    print("\n=== pixel components vs cycle error ===")
    print(f"{'component':16s} {'n':>6s} {'rho':>7s} {'95% CI':>18s} "
          f"{'shuf':>6s} {'E|z|/0.46':>10s}")
    for r in rows:
        if "spearman_rho" not in r:
            continue
        ci = (f"[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]"
              if np.isfinite(r.get("rho_ci_lo", np.nan)) else "n/a")
        print(f"{r['component']:16s} {r['n']:>6d} {r['spearman_rho']:>+7.3f} "
              f"{ci:>18s} {r.get('rho_shuffled', float('nan')):>6.3f} "
              f"{r['calibration_ratio']:>10.2f}")
    if ws:
        print(f"\n{'component':16s} {'within rho':>11s} {'+ve':>7s} {'partial':>9s} {'p':>8s}")
        for r in ws:
            pm = (f"{r['rho_partial_mu_mean']:+.3f}"
                  if "rho_partial_mu_mean" in r else "n/a")
            print(f"{r['component']:16s} {r['rho_raw_mean']:>+11.3f} "
                  f"{r['n_positive_raw']:>3d}/{r['n_slides']:<3d} {pm:>9s} "
                  f"{r.get('wilcoxon_p_partial_mu', float('nan')):>8.4f}")
    print(f"\nwrote {args.outdir}/reliability_pixel.png and four tables")


if __name__ == "__main__":
    main()
