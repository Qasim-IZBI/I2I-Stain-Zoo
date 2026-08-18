#!/usr/bin/env python
"""Cycle-reconstruction error against ensemble spread, on the same regions.

The BMVC 2026 result is that cycle error does not calibrate, and the phi study
shows an external structural target does. Those two live on different data, so
the contrast is currently a citation beside a measurement. This puts both on the
**same regions, the same target and the same statistics**, so only the
uncertainty source differs.

    compute_phi_uncertainty.py   ensemble masks -> per_region.csv    (mu, sd)
    compute_phi_reference.py     real masks     -> reference_phi.csv (real_*)
    evaluation.py --metric regen_error --save_error_npy  -> error_npy/
    THIS                         all three      -> which source predicts the error?

    python compare_uncertainty_sources.py \\
        --phi_csv       ./phi_uncertainty/per_region.csv \\
        --reference_csv ./calibration_phi/reference_phi.csv \\
        --regen_root    /path/ensemble/regen_error/model_01 \\
        --regen_root    /path/ensemble/regen_error/model_02 \\
        --tiles_metadata /path/tiles/testA \\
        --outdir        ./compare_sources/

Each `--regen_root` holds `wsi{NNN}/error_npy/<tile>.npy`, the layout
`scripts/compute_ensemble_regen_error.sh` already writes. Repeat it per ensemble
member; the per-tile means are averaged across members, which is the same number
as averaging the maps first because the mean is linear in the pixels — and far
cheaper, since it never holds more than one tile at a time.

Regen error is a **per-tile pixel quantity** and phi is a **per-region
descriptor**, so the tiles are aggregated into the region boxes `--phi_csv`
already defines. Tiling is non-overlapping with stride = tile size from origin
(0,0), so tiles nest exactly inside the region grid: a 2048 px region holds
sixteen 512 px tiles and no tile straddles a boundary.

The comparison is deliberately asymmetric in regen error's favour where a choice
existed. It gets the same regions, the same target, the same slide-clustered
statistics, and the same partialling on the point prediction — so a null result
cannot be blamed on the protocol.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from calibrate_phi import (
    fold_agreement,
    load_reference,
    make_reliability_figure,
    make_risk_coverage_figure,
    pair,
    phi_run_provenance,
    risk_coverage,
    score,
    within_slide,
)
from uncertainty_phi.descriptors import PHI_NAMES

# The sources being compared, as `component` values so every existing analysis
# and figure treats them as curves on one axis.
SIGMA_SOURCES = ("total", "procedural", "data_exposure")
REGEN = "regen_error"


def tile_table(tiles_metadata: Path) -> pd.DataFrame:
    """tile_name -> WSI, folder and pixel position, from the per-WSI CSVs.

    The folder (`001/`, `002/`, …) is kept because tile ids REPEAT across WSIs —
    every slide has a `0000001` — so the name alone does not identify a tile.
    `regen_error/wsi{NNN}/` is keyed on the same folder, which is why that layout
    is the one this reads.
    """
    csvs = sorted(Path(tiles_metadata).glob("*/tiles_metadata.csv"))
    if not csvs and Path(tiles_metadata).is_file():
        csvs = [Path(tiles_metadata)]
    if not csvs:
        raise SystemExit(f"no */tiles_metadata.csv under {tiles_metadata}")
    frames = []
    for csv in csvs:
        # tile_name MUST be read as a string. Tile names are zero-padded numerics
        # ("0000001"), which pandas parses as int64 1 — the lookup then asks for
        # 1.npy, every tile "has no error map", and the run dies claiming the
        # layout is wrong when only the padding was lost.
        df = pd.read_csv(csv, dtype={"tile_name": str})
        if df.empty:
            continue
        need = {"tile_name", "source_file", "x", "y", "tile_size"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"{csv} lacks column(s) {sorted(missing)}")
        df = df[list(need)].copy()
        df["folder"] = csv.parent.name
        df["wsi"] = df["source_file"].map(lambda s: Path(str(s)).stem)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["tile_name"] = out["tile_name"].astype(str).map(lambda s: Path(s).stem)
    out["folder"] = out["folder"].astype(str)
    return out


def assign_tiles_to_regions(tiles: pd.DataFrame,
                            regions: pd.DataFrame) -> pd.DataFrame:
    """Tiles wholly inside each region box.

    Containment, not centres: a tile straddling a boundary would contribute its
    whole error to one region and none to its neighbour. With non-overlapping
    tiling from (0,0) and a region size that is a multiple of the tile size this
    never happens, and the count per region is exact — which the caller checks.
    """
    out = []
    tiles_by_wsi = {w: g for w, g in tiles.groupby("wsi")}
    for r in regions.itertuples():
        stem = Path(str(r.wsi)).stem
        g = tiles_by_wsi.get(stem)
        if g is None:
            continue
        sel = g[(g["x"] >= r.x0) & (g["x"] + g["tile_size"] <= r.x1) &
                (g["y"] >= r.y0) & (g["y"] + g["tile_size"] <= r.y1)]
        if sel.empty:
            continue
        out.append(pd.DataFrame({
            "wsi": r.wsi, "region_index": r.region_index,
            "folder": sel["folder"].to_numpy(),
            "tile_name": sel["tile_name"].to_numpy()}))
    if not out:
        raise SystemExit(
            "no tiles fell inside any region — check that --tiles_metadata is the "
            "cohort --phi_csv was gridded on, and that both are at the same "
            "resolution (reconstructions sit at source mpp, not the model's)."
        )
    return pd.concat(out, ignore_index=True)


def tile_mean_error(roots: List[Path], folder: str, tile: str,
                    mask_dir: Optional[Path]) -> float:
    """Mean regen error for one tile, averaged over ensemble members.

    Averaging the per-tile means across members equals averaging the maps and
    then taking the mean — the mean is linear — so this never holds more than one
    tile at a time. With fifty members and ~48k tiles the map-first route would
    read hundreds of GB.
    """
    vals = []
    for root in roots:
        p = Path(root) / f"wsi{folder}" / "error_npy" / f"{tile}.npy"
        if not p.is_file():
            p = Path(root) / "error_npy" / f"{tile}.npy"   # flat fallback
            if not p.is_file():
                continue
        arr = np.load(p).astype(np.float64)
        if mask_dir is not None:
            m = None
            for ext in (".tif", ".tiff", ".png"):
                q = Path(mask_dir) / f"{tile}{ext}"
                if q.is_file():
                    import tifffile
                    m = tifffile.imread(str(q)) if ext != ".png" else None
                    if m is None:
                        from PIL import Image
                        m = np.array(Image.open(q))
                    break
            if m is not None:
                if m.ndim > 2:
                    m = m[..., 0]
                m = m != 0
                if m.shape == arr.shape and m.any():
                    arr = arr[m]
        vals.append(float(np.mean(arr)) if arr.size else np.nan)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def regen_per_region(regions: pd.DataFrame, tiles: pd.DataFrame,
                     roots: List[Path], mask_dir: Optional[Path],
                     cache: Optional[Path] = None) -> pd.DataFrame:
    """Mean cycle-reconstruction error per region, from its tiles."""
    pairs = assign_tiles_to_regions(tiles, regions)
    per_region_counts = pairs.groupby(["wsi", "region_index"]).size()
    print(f"[tiles] {len(pairs)} tile-region assignments, "
          f"{per_region_counts.min()}-{per_region_counts.max()} tiles per region "
          f"(median {int(per_region_counts.median())})")
    if per_region_counts.nunique() > 1:
        # Uneven counts mean the region grid and the tile grid disagree, which
        # makes a region's regen error an average over a different area than its
        # phi. Worth seeing rather than silently averaging.
        print(f"[WARN] regions hold different tile counts "
              f"({per_region_counts.min()}-{per_region_counts.max()}). Region "
              f"size is probably not a whole multiple of the tile size.")

    seen: Dict[Tuple[str, str], float] = {}
    if cache is not None and Path(cache).is_file():
        # same padding trap on the way back in
        c = pd.read_csv(cache, dtype={"folder": str, "tile_name": str})
        seen = {(str(a), str(b)): float(v) for a, b, v in
                zip(c["folder"], c["tile_name"], c["mean_error"])}
        print(f"[cache] {len(seen)} tile means from {cache}")

    todo = [(f, t) for f, t in
            pairs[["folder", "tile_name"]].drop_duplicates().itertuples(index=False)
            if (str(f), str(t)) not in seen]
    for i, (f, t) in enumerate(todo, 1):
        seen[(str(f), str(t))] = tile_mean_error(roots, str(f), str(t), mask_dir)
        if i % 2000 == 0:
            print(f"   {i}/{len(todo)} tiles")
    if cache is not None and todo:
        pd.DataFrame([{"folder": f, "tile_name": t, "mean_error": v}
                      for (f, t), v in seen.items()]).to_csv(cache, index=False)
        print(f"[cache] wrote {cache}")

    pairs["tile_error"] = [seen.get((str(f), str(t)), np.nan) for f, t in
                           zip(pairs["folder"], pairs["tile_name"])]
    miss = int(pairs["tile_error"].isna().sum())
    if miss:
        print(f"[WARN] {miss}/{len(pairs)} tiles had no error map and are skipped")
    agg = (pairs.dropna(subset=["tile_error"])
                .groupby(["wsi", "region_index"], as_index=False)["tile_error"]
                .mean().rename(columns={"tile_error": "regen"}))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        "Compare cycle-reconstruction error against ensemble spread")
    ap.add_argument("--phi_csv", type=Path, required=True)
    ap.add_argument("--reference_csv", type=Path, required=True)
    ap.add_argument("--regen_root", type=Path, action="append", required=True,
                    help="Directory holding wsi{NNN}/error_npy/<tile>.npy, as "
                         "compute_ensemble_regen_error.sh writes. Repeat once "
                         "per ensemble member; per-tile means are averaged.")
    ap.add_argument("--tiles_metadata", type=Path, required=True,
                    help="Dataset root with per-WSI tiles_metadata.csv — maps "
                         "tiles to pixel positions so they can be aggregated "
                         "into the region boxes.")
    ap.add_argument("--mask_dir", type=Path, default=None,
                    help="Flat per-tile tissue masks. Without it, background "
                         "pixels dilute the regen error of edge regions.")
    ap.add_argument("--tile_cache", type=Path, default=None,
                    help="CSV of per-tile mean errors. Written if absent, reused "
                         "if present — the tile pass is the slow half. "
                         "[<outdir>/tile_errors.csv]")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_regions_per_slide", type=int, default=15)
    ap.add_argument("--coverages", type=float, nargs="*",
                    default=[1.0, 0.9, 0.8, 0.7, 0.5])
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phi_csv)
    print(f"[1/4] {len(df)} regions over {df['wsi'].nunique()} WSI")
    phi_prov = phi_run_provenance(args.phi_csv)
    ref, ref_prov = load_reference(args.reference_csv, df)

    # The ensemble side, exactly as calibrate_phi computes it
    t = pair(df, ref, "grand", 5)

    print(f"[2/4] aggregating regen error over {len(args.regen_root)} member(s)")
    tiles = tile_table(args.tiles_metadata)
    regions = df[["wsi", "region_index", "y0", "y1", "x0", "x1"]].drop_duplicates()
    regen = regen_per_region(
        regions, tiles, args.regen_root, args.mask_dir,
        cache=args.tile_cache or (args.outdir / "tile_errors.csv"))

    # Regen error enters as another `component`, so every downstream analysis —
    # scoring, the slide-clustered bootstrap, the mu partial, risk-coverage and
    # both figures — treats it as one more curve rather than a special case.
    blocks = [t]
    for name in PHI_NAMES:
        base = t[(t["descriptor"] == name) & (t["component"] == "total")]
        if base.empty:
            continue
        b = base.merge(regen, on=["wsi", "region_index"], how="inner").copy()
        if b.empty:
            continue
        b["sd"] = b["regen"]
        b["component"] = REGEN
        with np.errstate(divide="ignore", invalid="ignore"):
            b["z"] = np.where(b["sd"] > 0, b["error"] / b["sd"], np.nan)
        blocks.append(b.drop(columns=["regen"]))
    t = pd.concat(blocks, ignore_index=True)
    n_regen = int((t["component"] == REGEN).sum())
    if not n_regen:
        raise SystemExit(
            "no region got a regen error — the tiles matched no region, or the "
            "error maps are missing. Check --regen_root's wsi{NNN}/error_npy/ "
            "layout against --tiles_metadata's folder names."
        )

    print(f"[3/4] scoring {t['component'].nunique()} uncertainty sources")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = score(t, args.n_bins, args.n_boot, args.seed)
        ws = within_slide(t, args.min_regions_per_slide, args.n_boot, args.seed)
        rc = risk_coverage(t, sorted(args.coverages, reverse=True), args.n_boot,
                           args.seed)

    t.to_csv(args.outdir / "per_region_sources.csv", index=False)
    if ws:
        pd.DataFrame([{k: v for k, v in r.items()
                       if not k.startswith("per_slide_")} for r in ws]).to_csv(
            args.outdir / "within_slide.csv", index=False)
    if rc:
        pd.DataFrame(rc).to_csv(args.outdir / "risk_coverage.csv", index=False)
        make_risk_coverage_figure(rc, args.outdir / "risk_coverage.png",
                                  "Selective prediction — uncertainty sources")
    make_reliability_figure(rows, args.outdir / "reliability_sources.png",
                            "Reliability — cycle error vs ensemble spread")

    with open(args.outdir / "summary.json", "w") as fh:
        json.dump({"sources": sorted(t["component"].unique()),
                   "n_regions_with_regen": n_regen,
                   "per_descriptor": rows, "within_slide": ws,
                   "risk_coverage": rc, "fold_agreement": fold_agreement(rows),
                   "reference": {"path": str(args.reference_csv), **ref_prov},
                   "phi_run": phi_prov,
                   "params": {k: (str(v) if isinstance(v, Path) else
                                  [str(x) for x in v]
                                  if isinstance(v, list) and v
                                  and isinstance(v[0], Path) else v)
                              for k, v in vars(args).items()}}, fh, indent=2)

    print("\n[4/4] === which uncertainty source predicts the error? ===")
    print("Same regions, same target, same slide-clustered statistics — only the")
    print("source differs. 'partial' removes the point prediction, without which")
    print("both sources are largely reporting how much structure a region holds.\n")
    if not ws:
        # An empty table with no explanation is the worst outcome: it looks like
        # the sources tied.
        print(f"[none] no slide had >= --min_regions_per_slide "
              f"({args.min_regions_per_slide}) regions, or fewer than three "
              f"slides qualified, so the within-slide comparison was withheld "
              f"rather than computed from a handful of regions. The pooled "
              f"scores in summary.json still stand; lower the threshold if the "
              f"cohort really is this small.")
    else:
        print(f"{'descriptor':22s} {'source':>15s} {'within rho':>11s} {'+ve':>7s} "
              f"{'partial':>9s} {'95% CI':>17s} {'p':>7s}")
        last = None
        for r in ws:
            if last and r["descriptor"] != last:
                print()
            shown = "" if r["descriptor"] == last else r["descriptor"]
            last = r["descriptor"]
            # The partial is absent, not zero, when mu had no spread to
            # partial on. Printing "+nan" reads as a computed result.
            has = "rho_partial_mu_mean" in r
            pm = f"{r['rho_partial_mu_mean']:+.3f}" if has else "n/a"
            pci = (f"[{r['rho_partial_mu_ci_lo']:+.3f},"
                   f"{r['rho_partial_mu_ci_hi']:+.3f}]" if has else "μ constant")
            pp = (f"{r['wilcoxon_p_partial_mu']:.4f}"
                  if has and np.isfinite(r.get("wilcoxon_p_partial_mu", np.nan))
                  else "")
            print(f"{shown[:22]:22s} "
                  f"{r['component']:>15s} {r['rho_raw_mean']:>+11.3f} "
                  f"{r['n_positive_raw']:>3d}/{r['n_slides']:<3d} "
                  f"{pm:>9s} {pci:>17s} {pp:>7s}")
    print(f"\nwrote {args.outdir}/per_region_sources.csv, within_slide.csv, "
          f"risk_coverage.csv, summary.json and two figures")


if __name__ == "__main__":
    main()
