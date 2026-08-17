#!/usr/bin/env python
"""φ_struct of the REAL tissue, on the region grid a φ run already defined.

The reference half of the calibration. It is a measurement in its own right and
nothing about it depends on the ensemble, so it runs once and is reused:

    compute_phi_uncertainty.py   ensemble masks -> per_region.csv     (mu, sd)
    compute_phi_reference.py     real masks     -> reference_phi.csv  (real_*)
    calibrate_phi.py             both CSVs      -> does sd predict the error?

    # collagen arm — the real SR, named after the SR slides, hence --strip_prefix
    python compute_phi_reference.py \\
        --phi_csv  ./phi_uncertainty/per_region.csv \\
        --real_psr /path/psr_masks/real/psr_masks_wsi_final --strip_prefix \\
        --he_masks /path/HE_tissue \\
        --outdir   ./calibration_phi/

    # lumen arm — the real H&E, same physical section (see CLAUDE.md for why
    # this is not computable on the UC liver cohort)
    python compute_phi_reference.py \\
        --phi_csv    ./phi_uncertainty/per_region.csv \\
        --real_lumen /path/lumen_masks_real --he_masks /path/HE_tissue \\
        --outdir     ./calibration_phi_lumen/

Regions come from `--phi_csv` **verbatim** — the same y0/y1/x0/x1 boxes, never a
rebuilt grid — and they travel into the output, so `calibrate_phi.py` can prove a
reference belongs to the grid it is being used on rather than trusting it.

Outputs `reference_phi.csv` (one row per region, `real_<descriptor>` columns plus
the box) and `reference_phi.json` (the parameter record).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uncertainty_phi.descriptors import WHITE_THRESH
from uncertainty_phi.reference import reference_phi, save_reference
from uncertainty_phi.regions import SOURCE_MPP

# Measurement parameters that must agree between this run and the φ run it is
# scored against. A component-size or resolution that differs between the two
# sides makes part of the "error" a parameter choice.
MUST_MATCH_PHI_RUN = ("mpp", "min_object_px", "closing_px")


def check_against_phi_run(args) -> dict:
    """Compare the measurement parameters against the φ run's own record.

    `compute_phi_uncertainty.py` writes `summary.json` beside `per_region.csv`.
    If it is there, the two sides can be checked automatically — and they must
    agree, because the error is a difference between two measurements and a
    `min_object_px` that differs between them puts part of that difference into
    the parameters rather than the tissue.
    """
    summary = args.phi_csv.parent / "summary.json"
    if not summary.is_file():
        print(f"[note] no summary.json beside {args.phi_csv.name}, so the phi "
              f"run's parameters cannot be checked automatically. Confirm "
              f"{', '.join(MUST_MATCH_PHI_RUN)} match it by hand.")
        return {}
    with open(summary) as fh:
        phi_params = json.load(fh).get("params", {})

    differ = {k: (phi_params[k], getattr(args, k)) for k in MUST_MATCH_PHI_RUN
              if k in phi_params and phi_params[k] != getattr(args, k)}
    if differ and not args.allow_param_mismatch:
        raise SystemExit(
            "the reference would be measured differently from the ensemble it "
            "is scored against, so part of the 'error' would be the parameter "
            "rather than the tissue:\n"
            + "\n".join(f"  {k}: phi run {a!r} vs this run {b!r}"
                        for k, (a, b) in differ.items())
            + f"\n  Match them, or pass --allow_param_mismatch if the difference "
              f"is deliberate.\n  (phi run: {summary})"
        )
    if differ:
        print("[WARN] --allow_param_mismatch: measuring the reference "
              "differently from the ensemble — "
              + "; ".join(f"{k} {a!r} vs {b!r}" for k, (a, b) in differ.items()))
    else:
        print(f"[ok] measurement parameters match the phi run ({summary.name})")
    return phi_params


def main() -> None:
    ap = argparse.ArgumentParser(
        "phi_struct of the real tissue, on a phi run's region grid")
    ap.add_argument("--phi_csv", type=Path, required=True,
                    help="per_region.csv from compute_phi_uncertainty.py (or "
                         "aggregate_phi_uncertainty.py). Supplies the region "
                         "boxes and nothing else — the grid is never rebuilt.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--out_name", default="reference_phi.csv",
                    help="Output filename inside --outdir. [%(default)s]")

    ap.add_argument("--real_psr", type=Path, default=None,
                    help="Real SR collagen masks. Supplies the four "
                         "PSR-referenced descriptors — needs the SR on the H&E "
                         "frame; the geometry is checked per slide.")
    ap.add_argument("--real_lumen", type=Path, default=None,
                    help="Lumen masks of the real H&E, from make_lumen_masks.py. "
                         "Supplies the three H&E-referenced descriptors. Same "
                         "physical section, so no floor and no frame question.")
    ap.add_argument("--he_masks", type=Path, default=None,
                    help="H&E tissue masks — the footprint the lumen densities "
                         "divide by. Pass whatever the phi run used: a footprint "
                         "built differently on the two sides means the "
                         "comparison divides by different denominators.")
    ap.add_argument("--he_dir", type=Path, default=None,
                    help="Real H&E WSIs — the fallback footprint source, by "
                         "thresholding, for a phi run that predates --he_masks.")
    ap.add_argument("--strip_prefix", action="store_true",
                    help="Drop the first '_'-delimited token from every stem "
                         "before matching, so SR_slide reaches HE_slide. Needed "
                         "with --real_psr, whose masks are named after the SR "
                         "slides while phi is gridded on the H&E. Same rule as "
                         "apply_he_mask.py and compare_psr.py.")

    # must match the run that produced --phi_csv; checked automatically against
    # its summary.json where one exists
    ap.add_argument("--mpp", type=float, default=SOURCE_MPP)
    ap.add_argument("--min_object_px", type=int, default=16)
    ap.add_argument("--closing_px", type=int, default=0)
    ap.add_argument("--white_thresh", type=float, default=WHITE_THRESH,
                    help="Only used by --he_dir's thresholded footprint and the "
                         "lumen terms. Inert with --he_masks and no "
                         "--real_lumen. [%(default)s]")
    ap.add_argument("--tile_size", type=int, default=512,
                    help="Tile size the cohort was tiled at (tile.py "
                         "--tile_size, NOT --resize_to: reconstructions sit at "
                         "source resolution). A reconstruction is the original "
                         "truncated to whole tiles, so a reference may exceed "
                         "the phi frame by up to this much and still be the same "
                         "frame. Above it they are different crops. [%(default)s]")
    ap.add_argument("--allow_param_mismatch", action="store_true",
                    help="Proceed even if the measurement parameters disagree "
                         "with the phi run's. Only for a deliberate difference.")
    args = ap.parse_args()

    if not args.real_psr and not args.real_lumen:
        ap.error("give --real_psr, --real_lumen, or both — there is no reference "
                 "to measure otherwise")
    if args.real_lumen and not (args.he_masks or args.he_dir):
        ap.error("--real_lumen needs --he_masks (or --he_dir): the lumen "
                 "densities divide by the H&E footprint, and it must be the same "
                 "footprint the phi run used")

    df = pd.read_csv(args.phi_csv)
    print(f"[1/2] {len(df)} regions over {df['wsi'].nunique()} WSI from "
          f"{args.phi_csv}")
    check_against_phi_run(args)

    ref = reference_phi(df, args)
    out = args.outdir / args.out_name
    save_reference(ref, out, args)

    cols = [c for c in ref.columns if c.startswith("real_")]
    print(f"\n[2/2] reference phi for {len(ref)} regions over "
          f"{ref['wsi'].nunique()} WSI")
    print(f"      descriptors: {', '.join(c[5:] for c in cols)}")
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    print(f"\nCalibrate with:\n"
          f"  python calibrate_phi.py --phi_csv {args.phi_csv} \\\n"
          f"      --reference_csv {out} --outdir <outdir>")


if __name__ == "__main__":
    main()
