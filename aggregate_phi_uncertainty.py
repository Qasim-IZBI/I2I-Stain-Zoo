#!/usr/bin/env python
"""Pool per-WSI phi_struct runs into one cohort-level result.

`compute_phi_uncertainty.py` can be split over WSIs (one array task each,
`scripts/compute_phi_uncertainty_grid_array.sh`) because `decompose()` works
region by region and regions never cross slide boundaries. The per-region
numbers are therefore already final; only the three means in `summary.json` are
cohort-level, and this recovers them by pooling the rows.

    python aggregate_phi_uncertainty.py \\
        --indir  /path/phi_uncertainty/per_wsi \\
        --outdir /path/phi_uncertainty/

Output is byte-for-byte the analysis a single whole-cohort job would have
produced: `per_region.csv` concatenated in WSI order, `summary.json` with means
taken over the pooled regions.

Refuses to pool runs whose parameters disagree. A region grid built at a
different `--region_mm`, or scored against a different set of folds, is not
comparable, and averaging across the difference would produce a number that
describes no experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Parameters that must agree for pooling to mean anything. `tiles_metadata`
# is deliberately absent — it is the per-WSI CSV, and differing is the point.
POOL_CRITICAL = (
    "roots",
    "he_dir",
    "roi_dir",
    "min_roi_fraction",
    "region_mm",
    "mpp",
    "min_tissue_fraction",
    "min_object_px",
    "closing_px",
)


def _nanmean_or_none(values: np.ndarray):
    if values.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        v = float(np.nanmean(values))
    return None if not np.isfinite(v) else v


def main() -> None:
    ap = argparse.ArgumentParser("Pool per-WSI phi_struct runs")
    ap.add_argument("--indir", type=Path, required=True,
                    help="Directory of per-WSI run directories (wsi001/, wsi002/, "
                         "...), each holding per_region.csv and summary.json.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--expect", type=int, default=None,
                    help="Fail unless exactly this many per-WSI runs are found. "
                         "Set it to the cohort size — a silently short pool is "
                         "the failure mode worth catching.")
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.indir.iterdir()
                      if d.is_dir() and (d / "per_region.csv").is_file())
    if not run_dirs:
        raise SystemExit(f"no per-WSI runs under {args.indir}")

    incomplete = [d.name for d in sorted(args.indir.iterdir())
                  if d.is_dir() and not (d / "per_region.csv").is_file()]
    if incomplete:
        print(f"[WARN] ignoring {len(incomplete)} directory(ies) with no "
              f"per_region.csv: {', '.join(incomplete)}")

    if args.expect is not None and len(run_dirs) != args.expect:
        raise SystemExit(
            f"found {len(run_dirs)} per-WSI runs, expected {args.expect}. "
            f"Re-run the missing array tasks before pooling."
        )

    frames: List[pd.DataFrame] = []
    summaries = []
    reference = None

    for d in run_dirs:
        with open(d / "summary.json") as fh:
            s = json.load(fh)
        params = {k: s["params"].get(k) for k in POOL_CRITICAL}
        if reference is None:
            reference, reference_name = params, d.name
        elif params != reference:
            differing = [k for k in POOL_CRITICAL if params[k] != reference[k]]
            raise SystemExit(
                f"{d.name} disagrees with {reference_name} on: {', '.join(differing)}. "
                f"These runs are not poolable — regions scored under different "
                f"parameters do not describe one experiment."
            )
        summaries.append(s)
        frames.append(pd.read_csv(d / "per_region.csv"))

    seeds = [tuple(s["n_seeds_per_fold"]) for s in summaries]
    if len(set(seeds)) != 1:
        raise SystemExit(f"per-WSI runs saw different member counts: {sorted(set(seeds))}")

    per_region = pd.concat(frames, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "per_region.csv"
    per_region.to_csv(out_csv, index=False)

    first = summaries[0]
    has_data = per_region["data_exposure"].notna().any()

    summary = {
        "n_folds": first["n_folds"],
        "n_seeds_per_fold": first["n_seeds_per_fold"],
        "n_regions": int(len(per_region)),
        "n_wsis": int(per_region["wsi"].nunique()),
        "descriptors": first["descriptors"],
        "reference_class": first["reference_class"],
        "variance": {
            "n_folds": first["n_folds"],
            "n_seeds_per_fold": first["n_seeds_per_fold"],
            "total_mean": _nanmean_or_none(per_region["var_total_anova"].to_numpy()),
            "procedural_mean": _nanmean_or_none(per_region["procedural"].to_numpy()),
            "data_mean": (_nanmean_or_none(per_region["data_exposure"].to_numpy())
                          if has_data else None),
            "data_component": "estimated" if has_data else "undefined (single fold)",
        },
        "mu_mean": {
            name: _nanmean_or_none(per_region[f"mu_{name}"].to_numpy())
            for name in first["descriptors"] if f"mu_{name}" in per_region.columns
        },
        "regions_per_wsi": {
            str(k): int(v) for k, v in per_region["wsi"].value_counts().sort_index().items()
        },
        "bias": first["bias"],
        "params": {
            **{k: reference[k] for k in POOL_CRITICAL},
            "pooled_from": [str(d) for d in run_dirs],
            "tiles_metadata": [s["params"]["tiles_metadata"] for s in summaries],
        },
    }

    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    v = summary["variance"]
    print("\n=== pooled phi_struct uncertainty ===")
    print(f"runs      : {len(run_dirs)}")
    print(f"regions   : {summary['n_regions']} over {summary['n_wsis']} WSI")
    print(f"total     : {v['total_mean']}")
    print(f"procedural: {v['procedural_mean']}")
    print(f"data      : {v['data_mean'] if v['data_mean'] is not None else 'undefined (single fold)'}")
    print(f"\nwrote {out_csv}")
    print(f"wrote {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
