"""φ_struct of the REAL tissue, on a given region grid.

The reference half of the calibration, kept apart from the calibration itself
for the same reason the ensemble half is: it is a measurement, it is expensive,
and nothing about it depends on the ensemble. Twenty full-slide masks, `betti`
and a structure tensor over every region — hours — against a calibration that is
seconds once both sides exist.

    compute_phi_uncertainty.py   ensemble masks  -> per_region.csv     (mu, sd)
    compute_phi_reference.py     real masks      -> reference_phi.csv  (real_*)
    calibrate_phi.py             both CSVs       -> does sd predict the error?

Separating them means a change to the figure, the binning or the bootstrap never
re-measures tissue, and `reference_phi.csv` stands on its own as what the
descriptors are on real tissue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from apply_he_mask import normalize_stem
from uncertainty_phi.descriptors import (
    PHI_NAMES,
    PHI_REFERENCE,
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

COLLAGEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "psr"]
LUMEN = [i for i, r in enumerate(PHI_REFERENCE) if r == "he"]


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


# Everything that changes what a reference φ IS. A cached reference computed
# under different values measures a different quantity, so reuse is refused
# rather than silently mixing them.
REFERENCE_PARAMS = ("mpp", "min_object_px", "closing_px", "white_thresh",
                    "real_psr", "real_lumen", "he_masks", "he_dir",
                    "strip_prefix")

BOX_COLS = ["y0", "y1", "x0", "x1"]


def save_reference(ref: pd.DataFrame, path: Path, args) -> None:
    """Write the reference φ plus the parameter record that makes it reusable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(path, index=False)
    meta = {k: (str(getattr(args, k)) if isinstance(getattr(args, k), Path)
                else getattr(args, k))
            for k in REFERENCE_PARAMS}
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump({"params": meta, "n_regions": int(len(ref)),
                   "n_wsi": int(ref["wsi"].nunique()),
                   "descriptors": [c for c in ref.columns
                                   if c.startswith("real_")]}, fh, indent=2)


def load_reference(path: Path, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Reuse a cached reference φ, after proving it belongs to THIS grid.

    Recomputing the reference is nearly all of this script's runtime — twenty
    full-slide masks, `betti` and a structure tensor over every region — so a
    cache turns a re-plot from hours into seconds. What it must never do is
    quietly pair one grid's σ with another grid's reference.

    Two independent checks, because either alone leaves a hole. The parameters
    catch a reference measured differently (a different `--white_thresh`, a
    different mask directory) on the same boxes. The **boxes themselves** catch a
    regrid — `--region_px 1024` against a cache built at 2048 keeps every
    parameter identical while every region moves.
    """
    ref = pd.read_csv(path)
    meta_path = path.with_suffix(".json")
    provenance = {}
    if meta_path.is_file():
        with open(meta_path) as fh:
            provenance = json.load(fh)
    else:
        print(f"[WARN] no {meta_path.name} beside {path.name}, so there is no "
              f"record of how the reference was measured. It will not appear in "
              f"summary.json. The region boxes are still checked.")

    if not set(BOX_COLS) <= set(ref.columns):
        raise SystemExit(
            f"{path} carries no y0/y1/x0/x1 columns, so it cannot be checked "
            f"against this grid. It predates --save_reference; recompute it."
        )

    key = ["wsi", "region_index"]
    merged = df[key + BOX_COLS].merge(ref[key + BOX_COLS], on=key, how="inner",
                                      suffixes=("_phi", "_ref"))
    if len(merged) != len(df):
        raise SystemExit(
            f"{path} covers {len(merged)} of the {len(df)} regions in --phi_csv. "
            f"It was built for a different grid — recompute it, or point "
            f"--phi_csv at the run it came from."
        )
    moved = merged[[f"{c}_phi" for c in BOX_COLS]].to_numpy() != \
        merged[[f"{c}_ref" for c in BOX_COLS]].to_numpy()
    if moved.any():
        bad = merged[moved.any(axis=1)].iloc[0]
        # Print the WHOLE box. Two grids at different region sizes share the same
        # top-left corner for region 0, so reporting only y0/x0 shows two
        # identical pairs and reads as a spurious failure.
        def _box(suffix):
            return (f"y {bad['y0' + suffix]}-{bad['y1' + suffix]}  "
                    f"x {bad['x0' + suffix]}-{bad['x1' + suffix]}")
        raise SystemExit(
            f"{path} has the same region ids on DIFFERENT boxes — e.g. "
            f"{bad['wsi']} region {bad['region_index']}:\n"
            f"  cache    {_box('_ref')}\n"
            f"  phi_csv  {_box('_phi')}\n"
            f"  Reusing it would score one grid's spread against another grid's "
            f"tissue. Recompute the reference for this grid."
        )
    print(f"[ref] {len(ref)} regions over {ref['wsi'].nunique()} WSI from "
          f"{path} (boxes verified against --phi_csv)")
    return ref, provenance


def reference_phi(df: pd.DataFrame, args) -> pd.DataFrame:
    """φ of the real tissue, on the exact region boxes the virtual run used.

    `df` supplies the grid and nothing else: this never rebuilds a region grid,
    so the two sides of the comparison cannot drift apart through a parameter
    that differs by one. `args` carries the mask directories and the measurement
    parameters — see `compute_phi_reference.py`, which is the entry point.
    """
    sp = args.strip_prefix
    psr_index = _indexed(args.real_psr, sp, "--real_psr")
    lum_index = _indexed(args.real_lumen, sp, "--real_lumen")
    he_index = _indexed(args.he_dir, sp, "--he_dir")
    he_mask_index = _indexed(args.he_masks, sp, "--he_masks")

    rows: List[dict] = []
    missing: List[str] = []
    edge_noted: set = set()
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
                # One benign difference: the reference is the UNTRUNCATED
                # original while phi was gridded on a reconstruction, which
                # utils.reconstruct_wsi truncates to a whole number of tiles at
                # the same origin and scale. Then the phi frame is a prefix of
                # the reference in both axes and the boxes index identical
                # pixels — the edge strip the reconstruction dropped is simply
                # never addressed.
                #
                # The bound is what separates it from a genuinely different
                # frame: truncation cannot lose a whole tile, so an excess below
                # one tile means the reference truncates to exactly this frame,
                # while the UC M3 case is over by 2273x4741 px and aligns with
                # nothing. "Larger" alone is not the test.
                slack = (got[0] - want[0], got[1] - want[1])
                truncates_to_frame = (
                    0 <= slack[0] < args.tile_size and 0 <= slack[1] < args.tile_size
                )
                if truncates_to_frame:
                    if stem not in edge_noted:
                        aligned = (want[0] % args.tile_size == 0
                                   and want[1] % args.tile_size == 0)
                        print(f"[note] {stem}: reference is {got[0]}x{got[1]}, phi "
                              f"frame {want[0]}x{want[1]} (+{slack[0]}x{slack[1]} "
                              f"px, under one {args.tile_size}px tile"
                              f"{'' if aligned else '; phi frame is NOT tile-aligned'})"
                              f" — the reference is the untruncated original, "
                              f"cropped to the reconstruction's frame.")
                        edge_noted.add(stem)
                    continue
                raise SystemExit(
                    f"{stem}: {name} is {got[0]}x{got[1]} but the phi run was "
                    f"gridded on {want[0]}x{want[1]} (off by "
                    f"{slack[0]}x{slack[1]} px, at least one {args.tile_size}px "
                    f"tile). Different frames — region r is different tissue on "
                    f"each side. Note it is not enough to be larger than the "
                    f"regions: this checks the frame, not the bound. Run "
                    f"scripts/check_frame_alignment.sh; for the collagen arm the "
                    f"SR must be RESAMPLED onto the H&E grid, not merely "
                    f"registered to it. If the excess really is only tiling "
                    f"truncation, --tile_size must match the tiling."
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
            # The boxes travel WITH the reference, so a cached reference can be
            # checked against the grid it is being reused on rather than trusted.
            out: Dict[str, float] = {
                "wsi": wsi, "region_index": row.region_index,
                "y0": row.y0, "y1": row.y1, "x0": row.x0, "x1": row.x1,
            }
            box = (row.y1 - row.y0, row.x1 - row.x0)

            # Numpy slicing past an edge returns a SHORT array rather than
            # raising, and every descriptor is a density — a short crop divides
            # by the wrong area and comes back plausible. The frame checks above
            # should make this unreachable; it is here because the failure is
            # invisible if they ever do not.
            for name, arr in (("--real_psr", labels), ("--real_lumen", lumen),
                              ("--he_masks/--he_dir", footprint)):
                if arr is None:
                    continue
                crop = arr[ys, xs]
                if crop.shape[:2] != box:
                    raise SystemExit(
                        f"{stem} region {row.region_index}: {name} cropped to "
                        f"{crop.shape[0]}x{crop.shape[1]} but the box is "
                        f"{box[0]}x{box[1]}. The region runs past the reference's "
                        f"edge, so every density would divide by the wrong area."
                    )

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


