"""Per-member φ over an ensemble, then μ and Var in descriptor space.

Deliberately provenance-agnostic: it consumes any directory laid out as

    <root>/model_01/<wsi>.tif
    <root>/model_02/<wsi>.tif
    ...

which is what `fill_tissue_holes_ensemble.sh` (vanilla) and the UGAC equivalent
both produce. Whether those masks came from a 10-seed deep ensemble or one fold
of the 5×10 UGAC grid is not this module's concern — `decompose.py` handles the
fold structure a level up.

The ensemble mean here is the mean of the **descriptor vectors**, never of the
images (uncertainty_strategy.md §2.1).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from uncertainty_phi.descriptors import (
    PHI_DIM,
    WHITE_THRESH,
    he_bright,
    he_tissue_footprint,
    phi_struct,
    tissue_footprint_from_mask,
)
from uncertainty_phi.regions import (
    Region,
    filter_by_roi,
    filter_by_tissue,
    region_grid,
    wsi_extent,
)

# Reused from uncertainty.py, which already globs model_* for the pixel-space path.
from uncertainty import discover_ensemble_dirs  # noqa: F401


def _to_channels_last(arr: np.ndarray, axes: Optional[str] = None) -> np.ndarray:
    """Normalise a 3D array to H,W,C.

    Some exports store planar (C,H,W) rather than interleaved (H,W,C). Both
    readers below index the LAST axis — `arr[..., :3]`, `arr[..., 0]` — so on a
    planar file they slice along width and return silent garbage rather than
    failing. One SR slide in the UC cohort is stored this way.

    `axes` from `tifffile`'s series is authoritative when present ("SYX"/"CYX"
    against "YXS"). Without it, fall back to shape: a channel axis is 1, 3 or 4
    and on a whole slide is always the shortest by orders of magnitude.
    """
    if arr.ndim != 3:
        return arr
    if axes and len(axes) == 3:
        return np.moveaxis(arr, 0, -1) if axes[0] in "SCP" else arr
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        return np.moveaxis(arr, 0, -1)
    return arr


def _imread_hwc(path: Path) -> np.ndarray:
    """`tifffile.imread` with the channel axis put last if it was first."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        return _to_channels_last(series.asarray(), series.axes)


def load_label_mask(path: Path) -> np.ndarray:
    """Read an nnU-Net label TIF as a 2D array.

    Masks stitched by `reconstruct.py --mode rgb` come back 3-channel; the first
    channel carries the labels. Same convention as `apply_he_mask.py:load_mask`
    and `compare_psr.py:compute_psr_fraction`.
    """
    arr = _imread_hwc(path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def load_rgb(path: Path) -> np.ndarray:
    arr = _imread_hwc(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def load_roi_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """Read a binary region-of-interest mask, resized to `shape` if needed.

    Any non-zero value is inside the ROI. Cortex masks are normally annotated at
    thumbnail magnification rather than at the 0.221 um/px of a reconstruction,
    so a size mismatch is the expected case, not an error. Nearest-neighbour
    keeps it binary — the same convention as `apply_he_mask.py`.
    """
    arr = _imread_hwc(path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    roi = arr != 0
    if roi.shape != shape:
        from PIL import Image
        roi = np.array(
            Image.fromarray(roi.astype(np.uint8) * 255).resize(
                (shape[1], shape[0]), Image.NEAREST
            )
        ) > 0
    return roi


def _stem_index(directory: Path) -> Dict[str, Path]:
    """Map file stem -> path for every image in a directory."""
    out: Dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in (".tif", ".tiff", ".png"):
            out[p.stem] = p
    return out


def save_lumen_qc(
    he: np.ndarray,
    footprint: np.ndarray,
    regions: List[Region],
    wsi_stem: str,
    outdir: Path,
    *,
    white_thresh: float,
    max_px: int = 0,
) -> Optional[Path]:
    """One region per WSI written as TIFs, so the lumen call can be inspected.

    `lumen_fraction` is a thresholded quantity with no plateau on some cohorts,
    so the number alone cannot distinguish "found the lumens" from "found pale
    tissue". Two files go out, at full resolution unless `max_px` caps them:

        <stem>_r<idx>_y<y0>_x<x0>_lumen.tif   0 outside, 1 tissue, 2 lumen
        <stem>_r<idx>_y<y0>_x<x0>_he.tif      the matching H&E crop

    Same label convention as the nnU-Net masks, so Fiji opens the pair and
    overlays them directly. The measured fractions and the threshold go in the
    TIFF description, where Image > Show Info surfaces them.

    The region is the one with the highest footprint coverage — fully interior,
    where lumens are unambiguous — with the lowest index breaking ties, so
    re-running picks the same region and two thresholds are comparable.
    """
    if not regions:
        return None

    coverage = [(float(r.crop(footprint).mean()), -r.index, r) for r in regions]
    _, _, region = max(coverage, key=lambda c: (c[0], c[1]))

    he_crop = region.crop(he)
    fp_crop = region.crop(footprint)
    lumen = he_bright(he_crop, white_thresh) & fp_crop

    n_tissue = int(np.count_nonzero(fp_crop))
    lumen_frac = float(np.count_nonzero(lumen) / n_tissue) if n_tissue else float("nan")
    tissue_frac = float(n_tissue / fp_crop.size) if fp_crop.size else float("nan")

    step = 1
    if max_px and max(he_crop.shape[:2]) > max_px:
        step = int(np.ceil(max(he_crop.shape[:2]) / max_px))
        he_crop, fp_crop, lumen = (he_crop[::step, ::step],
                                   fp_crop[::step, ::step],
                                   lumen[::step, ::step])

    label = np.where(lumen, 2, np.where(fp_crop, 1, 0)).astype(np.uint8)

    meta = json.dumps({
        "wsi": wsi_stem, "region_index": region.index,
        "y0": region.y0, "y1": region.y1, "x0": region.x0, "x1": region.x1,
        "white_thresh": white_thresh,
        "lumen_fraction": lumen_frac, "tissue_fraction": tissue_frac,
        "labels": {"0": "outside footprint", "1": "tissue", "2": "lumen"},
        "downsample": step,
    })

    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{wsi_stem}_r{region.index:04d}_y{region.y0}_x{region.x0}"
    mask_path = outdir / f"{base}_lumen.tif"
    tifffile.imwrite(str(mask_path), label, compression="zlib", description=meta)
    tifffile.imwrite(str(outdir / f"{base}_he.tif"), he_crop,
                     compression="zlib", description=meta)

    print(f"[qc] {mask_path.name}  lumen {lumen_frac:.4f}  tissue {tissue_frac:.4f}"
          + (f"  (1/{step})" if step > 1 else ""))
    return mask_path


def phi_for_wsi(
    member_dirs: Sequence[Path],
    wsi_stem: str,
    regions: List[Region],
    *,
    he_path: Optional[Path] = None,
    he_mask_path: Optional[Path] = None,
    lumen_dirs: Optional[Sequence[Path]] = None,
    mpp: float,
    qc_dir: Optional[Path] = None,
    qc_max_px: int = 0,
    **phi_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """φ for one WSI across members, plus the per-region tissue fraction.

    Returns ([n_members, n_regions, PHI_DIM], [n_regions]).

    A member missing this WSI yields an all-NaN slab rather than being dropped,
    so the member axis stays aligned and the gap is visible downstream.

    tissue_fraction rides along rather than sitting in φ: it is the H&E
    footprint's coverage, shared by every member and by the reference, so it has
    no variance to decompose and no error to calibrate. It is still worth
    reporting, and this is the only place the footprint exists.
    """
    he = load_rgb(he_path) if he_path is not None and Path(he_path).exists() else None

    # WSI-level footprint: a lumen straddling a region boundary is only enclosed
    # (and therefore only counted) when the fill is done on the whole slide.
    #
    # Preferred source is the H&E tissue mask — the same boundary
    # apply_he_mask.py applies to the collagen masks, and independent of
    # white_thresh, which the footprint is otherwise the most sensitive thing to.
    # Falls back to thresholding the H&E, in which case the footprint and the
    # lumen count must use ONE threshold: `lumen_fraction` is `bright &
    # footprint`, so two cuts would intersect two definitions of whitespace.
    footprint = None
    if he_mask_path is not None and Path(he_mask_path).exists():
        shape = he.shape[:2] if he is not None else load_label_mask(
            _stem_index(Path(member_dirs[0])).get(wsi_stem)).shape
        footprint = tissue_footprint_from_mask(load_roi_mask(he_mask_path, shape))
    elif he is not None:
        footprint = he_tissue_footprint(
            he, white_thresh=phi_kwargs.get("white_thresh", WHITE_THRESH))

    if qc_dir is not None and he is not None and footprint is not None:
        # Once per WSI, not per member: both H&E terms are member-independent.
        save_lumen_qc(
            he, footprint, regions, wsi_stem, Path(qc_dir),
            white_thresh=phi_kwargs.get("white_thresh", WHITE_THRESH),
            max_px=qc_max_px,
        )

    tissue_frac = np.array(
        [float(region.crop(footprint).mean()) if footprint is not None else np.nan
         for region in regions],
        dtype=np.float64,
    )

    out = np.full((len(member_dirs), len(regions), PHI_DIM), np.nan, dtype=np.float64)
    for m, mdir in enumerate(member_dirs):
        idx = _stem_index(Path(mdir))
        mpath = idx.get(wsi_stem)
        if mpath is None:
            continue
        labels = load_label_mask(mpath)

        # Per-member lumen, precomputed by make_lumen_masks.py. Reading a mask
        # rather than thresholding the member's RGB here is the whole point of
        # that stage: the lumen is member-specific, so the RGB path would load
        # several GB fifty times per slide.
        member_lumen = None
        if lumen_dirs is not None:
            lpath = _stem_index(Path(lumen_dirs[m])).get(wsi_stem)
            if lpath is not None:
                member_lumen = load_label_mask(lpath) > 0
        for r, region in enumerate(regions):
            lab_crop = region.crop(labels)
            fp_crop = region.crop(footprint) if footprint is not None else None
            if member_lumen is not None:
                out[m, r] = phi_struct(
                    lab_crop, None, mpp=mpp, tissue_mask=fp_crop,
                    lumen=region.crop(member_lumen), **phi_kwargs
                )
            else:
                out[m, r] = phi_struct(
                    lab_crop, region.crop(he) if he is not None else None,
                    mpp=mpp, tissue_mask=fp_crop, **phi_kwargs
                )
    return out, tissue_frac


def phi_over_ensemble(
    ensemble_root: Path,
    tiles_metadata_root: Path,
    *,
    he_dir: Optional[Path] = None,
    he_masks_dir: Optional[Path] = None,
    lumen_root: Optional[Path] = None,
    roi_dir: Optional[Path] = None,
    min_roi_fraction: float = 0.5,
    qc_dir: Optional[Path] = None,
    qc_max_px: int = 0,
    region_mm: float = 1.5,
    region_px: Optional[int] = None,
    mpp: float,
    min_tissue_fraction: float = 0.25,
    **phi_kwargs,
) -> Tuple[np.ndarray, List[Region], List[Path]]:
    """φ for every WSI and member under one ensemble root.

    Returns (phi, regions, member_dirs, tissue_fraction, shapes) with phi
    [n_members, n_regions_total, d], `regions` flattened across WSIs in the same
    order as the second axis, tissue_fraction [n_regions_total], and `shapes`
    mapping WSI stem to the (h, w) frame the regions were cut from — recorded so
    a reference can be checked for *matching* that frame rather than merely
    covering it.

    The region grid is built once per WSI, so all members are scored on identical
    boxes — a per-member grid would make the variance meaningless.

    With `he_masks_dir` the tissue filter reads the **H&E** footprint, which is
    the same on every fold. Without it the filter falls back to the first
    available member's collagen mask, and then the grid depends on which
    ensemble it was cut from: a region near `min_tissue_fraction` is kept by one
    fold and dropped by another, and `compute_phi_uncertainty.py` rejects the
    run for a fold-count mismatch. Pass the H&E masks for any crossed grid.
    """
    member_dirs = discover_ensemble_dirs(Path(ensemble_root))
    if not member_dirs:
        raise FileNotFoundError(f"no model_* directories under {ensemble_root}")

    lumen_dirs = None
    if lumen_root is not None:
        lumen_dirs = discover_ensemble_dirs(Path(lumen_root))
        if len(lumen_dirs) != len(member_dirs):
            raise FileNotFoundError(
                f"{lumen_root} holds {len(lumen_dirs)} model_* directories but "
                f"{ensemble_root} holds {len(member_dirs)} — member m must be "
                f"the same model in both, so a mismatch would pair one member's "
                f"collagen with another's lumen"
            )

    he_index = _stem_index(Path(he_dir)) if he_dir else {}
    he_mask_index = _stem_index(Path(he_masks_dir)) if he_masks_dir else {}
    roi_index = _stem_index(Path(roi_dir)) if roi_dir else {}

    all_phi: List[np.ndarray] = []
    all_tissue: List[np.ndarray] = []
    shapes: Dict[str, Tuple[int, int]] = {}
    all_regions: List[Region] = []
    roi_missing: List[str] = []
    fold_dependent_grid: List[str] = []

    from uncertainty_phi.regions import iter_metadata_csvs

    for csv_path in iter_metadata_csvs(Path(tiles_metadata_root)):
        wsi_stem, _, _ = wsi_extent(csv_path)
        wsi_stem = Path(wsi_stem).stem

        grid = region_grid(csv_path, region_mm=region_mm, mpp=mpp,
                           region_px=region_px)
        if not grid:
            continue

        # One member's mask, for the frame the regions are cut from. Also the
        # fallback tissue reference — see below.
        reference = None
        for mdir in member_dirs:
            p = _stem_index(Path(mdir)).get(wsi_stem)
            if p is not None:
                reference = load_label_mask(p)
                break
        if reference is None:
            continue

        # The tissue filter must not depend on WHICH ensemble it ran on, or two
        # folds get different grids and the decomposition has nothing to
        # decompose. A member's collagen mask is a model output, so a region
        # near min_tissue_fraction falls on either side of it by fold. The H&E
        # footprint is a property of the slide and identical everywhere — and is
        # already the denominator for every density, so filtering on it is the
        # same definition of tissue rather than a second one.
        he_mask_path = he_mask_index.get(wsi_stem)
        if he_mask_path is not None:
            tissue_reference = load_roi_mask(he_mask_path, reference.shape)
        else:
            tissue_reference = reference
            fold_dependent_grid.append(wsi_stem)
        grid = filter_by_tissue(grid, tissue_reference,
                                min_tissue_fraction=min_tissue_fraction)
        if not grid:
            continue

        # Anatomical restriction, e.g. cortex on the kidney arm. A WSI with no
        # ROI mask is DROPPED, not passed through: falling back to the whole
        # slide would quietly mix medulla into a cortex-only result, and a
        # missing number is recoverable where a contaminated one is not.
        if roi_dir:
            roi_path = roi_index.get(wsi_stem)
            if roi_path is None:
                roi_missing.append(wsi_stem)
                print(f"[WARN] no ROI mask for {wsi_stem} — WSI excluded")
                continue
            grid = filter_by_roi(
                grid, load_roi_mask(roi_path, reference.shape),
                min_roi_fraction=min_roi_fraction,
            )
            if not grid:
                continue

        block, tissue_frac = phi_for_wsi(
            member_dirs, wsi_stem, grid,
            he_path=he_index.get(wsi_stem),
            he_mask_path=he_mask_index.get(wsi_stem),
            lumen_dirs=lumen_dirs, mpp=mpp,
            qc_dir=qc_dir, qc_max_px=qc_max_px, **phi_kwargs,
        )
        all_phi.append(block)
        all_tissue.append(tissue_frac)
        all_regions.extend(grid)
        shapes[wsi_stem] = (int(reference.shape[0]), int(reference.shape[1]))

    if not all_phi:
        hint = (
            f"; every WSI was excluded for want of an ROI mask under {roi_dir}"
            if roi_dir and roi_missing else ""
        )
        raise RuntimeError(
            f"no regions produced — check that {ensemble_root} holds masks whose "
            f"stems match the WSIs described by {tiles_metadata_root}{hint}"
        )
    if roi_missing:
        print(f"[WARN] {len(roi_missing)} WSI(s) excluded for a missing ROI mask: "
              f"{', '.join(roi_missing)}")
    if fold_dependent_grid:
        print(f"[WARN] {len(fold_dependent_grid)} WSI(s) had no H&E tissue mask, so "
              f"the tissue filter used a member's collagen mask: "
              f"{', '.join(fold_dependent_grid[:3])}"
              f"{' ...' if len(fold_dependent_grid) > 3 else ''}\n"
              f"       The resulting grid depends on which ensemble it was cut "
              f"from. Fine for a single ensemble; for a crossed grid the folds "
              f"will disagree on the region count. Pass --he_masks.")

    return (np.concatenate(all_phi, axis=1), all_regions, member_dirs,
            np.concatenate(all_tissue), shapes)


def mean_and_variance(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(μ, Var) across members, in descriptor space.

    μ = E_m[φ_m]                         -> [n_regions, d]
    Var = E_m‖φ_m − μ‖²                  -> [n_regions]

    Var is the squared Euclidean spread summed over descriptors, matching the
    `Var(x)` of the §2.1 identity. Whiten it first if you need it commensurable
    with a whitened bias² (see `decompose.decompose_whitened`).
    """
    p = np.asarray(phi, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"expected [n_members, n_regions, d], got {p.shape}")
    # an all-NaN region is background, not an error worth warning about
    warnings.simplefilter("ignore", RuntimeWarning)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(p, axis=0)
        var = np.nanmean(np.nansum((p - mu[None]) ** 2, axis=2), axis=0)
    return mu, var
