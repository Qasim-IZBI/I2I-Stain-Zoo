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

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from uncertainty_phi.descriptors import (
    PHI_DIM,
    WHITE_THRESH,
    he_tissue_footprint,
    phi_struct,
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


def load_label_mask(path: Path) -> np.ndarray:
    """Read an nnU-Net label TIF as a 2D array.

    Masks stitched by `reconstruct.py --mode rgb` come back 3-channel; the first
    channel carries the labels. Same convention as `apply_he_mask.py:load_mask`
    and `compare_psr.py:compute_psr_fraction`.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def load_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
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
    arr = tifffile.imread(str(path))
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


def phi_for_wsi(
    member_dirs: Sequence[Path],
    wsi_stem: str,
    regions: List[Region],
    *,
    he_path: Optional[Path] = None,
    mpp: float,
    **phi_kwargs,
) -> np.ndarray:
    """φ for one WSI across members. Returns [n_members, n_regions, PHI_DIM].

    A member missing this WSI yields an all-NaN slab rather than being dropped,
    so the member axis stays aligned and the gap is visible downstream.
    """
    he = load_rgb(he_path) if he_path is not None and Path(he_path).exists() else None
    # WSI-level footprint: a lumen straddling a region boundary is only enclosed
    # (and therefore only counted) when the fill is done on the whole slide.
    #
    # The footprint and the lumen count must use ONE threshold — `lumen_fraction`
    # is `bright & footprint`, so a footprint built at a different cut would
    # intersect two different definitions of whitespace. Hence reading it out of
    # phi_kwargs rather than letting each default independently.
    footprint = (
        he_tissue_footprint(he, white_thresh=phi_kwargs.get("white_thresh", WHITE_THRESH))
        if he is not None else None
    )

    out = np.full((len(member_dirs), len(regions), PHI_DIM), np.nan, dtype=np.float64)
    for m, mdir in enumerate(member_dirs):
        idx = _stem_index(Path(mdir))
        mpath = idx.get(wsi_stem)
        if mpath is None:
            continue
        labels = load_label_mask(mpath)
        for r, region in enumerate(regions):
            lab_crop = region.crop(labels)
            he_crop = region.crop(he) if he is not None else None
            fp_crop = region.crop(footprint) if footprint is not None else None
            out[m, r] = phi_struct(
                lab_crop, he_crop, mpp=mpp, tissue_mask=fp_crop, **phi_kwargs
            )
    return out


def phi_over_ensemble(
    ensemble_root: Path,
    tiles_metadata_root: Path,
    *,
    he_dir: Optional[Path] = None,
    roi_dir: Optional[Path] = None,
    min_roi_fraction: float = 0.5,
    region_mm: float = 1.5,
    mpp: float,
    min_tissue_fraction: float = 0.25,
    **phi_kwargs,
) -> Tuple[np.ndarray, List[Region], List[Path]]:
    """φ for every WSI and member under one ensemble root.

    Returns (phi, regions, member_dirs) with phi [n_members, n_regions_total, d]
    and `regions` flattened across WSIs in the same order as the second axis.

    The region grid is built once per WSI from the *first available* member's
    mask, so all members are scored on identical boxes — a per-member grid would
    make the variance meaningless.
    """
    member_dirs = discover_ensemble_dirs(Path(ensemble_root))
    if not member_dirs:
        raise FileNotFoundError(f"no model_* directories under {ensemble_root}")

    he_index = _stem_index(Path(he_dir)) if he_dir else {}
    roi_index = _stem_index(Path(roi_dir)) if roi_dir else {}

    all_phi: List[np.ndarray] = []
    all_regions: List[Region] = []
    roi_missing: List[str] = []

    from uncertainty_phi.regions import iter_metadata_csvs

    for csv_path in iter_metadata_csvs(Path(tiles_metadata_root)):
        wsi_stem, _, _ = wsi_extent(csv_path)
        wsi_stem = Path(wsi_stem).stem

        grid = region_grid(csv_path, region_mm=region_mm, mpp=mpp)
        if not grid:
            continue

        # tissue filter uses the first member that has this WSI
        reference = None
        for mdir in member_dirs:
            p = _stem_index(Path(mdir)).get(wsi_stem)
            if p is not None:
                reference = load_label_mask(p)
                break
        if reference is None:
            continue
        grid = filter_by_tissue(grid, reference, min_tissue_fraction=min_tissue_fraction)
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

        block = phi_for_wsi(
            member_dirs, wsi_stem, grid,
            he_path=he_index.get(wsi_stem), mpp=mpp, **phi_kwargs,
        )
        all_phi.append(block)
        all_regions.extend(grid)

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

    return np.concatenate(all_phi, axis=1), all_regions, member_dirs


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
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(p, axis=0)
        var = np.nanmean(np.nansum((p - mu[None]) ** 2, axis=2), axis=0)
    return mu, var
