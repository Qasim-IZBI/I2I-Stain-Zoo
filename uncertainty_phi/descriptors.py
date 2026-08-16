"""φ_struct — the structural descriptor vector (kidney_ood_data_plan.md §5.4).

Seven marginal statistics of a region, split into two reference classes (§6.0):

    task_specific_value   collagen proportionate area          } referenced to
    beta0_per_mm2         connected components of collagen     } the real PSR at
    beta1_per_mm2         loops in collagen                    } level B — these
    regional_dispersion   spread of collagen orientation       } pay the floor

    lumen_fraction        enclosed whitespace within tissue    } referenced to the
    beta0_lumen_per_mm2   connected components of lumen        } real H&E at level A,
    beta1_lumen_per_mm2   loops in the lumen space             } the SAME physical
                                                               } section — no floor

Two rules the whole design rests on:

1. **Mask, never intensity.** Every collagen term is computed from the thresholded
   binary mask. A global colour offset in the virtual stain would otherwise
   masquerade as genuine model bias (§6.2); mask-derived shape statistics are
   immune to it.
2. **Counts are densities.** β₀/β₁ scale with region area, so they are reported
   per mm² of tissue and regions of differing size stay comparable (§5.5.2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

# Order is load-bearing: whiten.py and decompose.py index by position.
PHI_NAMES: Tuple[str, ...] = (
    "task_specific_value",
    "beta0_per_mm2",
    "beta1_per_mm2",
    "regional_dispersion",
    "lumen_fraction",
    "beta0_lumen_per_mm2",
    "beta1_lumen_per_mm2",
)

# tissue_fraction is deliberately NOT in the vector. It is read from the H&E
# footprint, which is shared by every ensemble member and by the reference, so
# it has zero variance and zero error — nothing to decompose and nothing to
# calibrate. `phi_struct` still returns it separately, and the pipeline writes
# it to per_region.csv as a QC column.

# Which image each component is referenced against — decides whether it pays the
# biological floor. "psr" = real PSR at level B (floor); "he" = H&E input at
# level A, same physical section, pixel-aligned (no floor).
PHI_REFERENCE: Tuple[str, ...] = ("psr", "psr", "psr", "psr", "he", "he", "he")

PHI_DIM = len(PHI_NAMES)

# nnU-Net Dataset314_SR_light convention, matching compare_psr.py
LABEL_TISSUE = 1
LABEL_PSR = 2

# Brightness above which a pixel counts as whitespace. Scanner-dependent, and
# `he_bright` requires EVERY channel to clear it, so the number to compare
# against is the per-pixel channel MINIMUM — not the grey level Fiji shows after
# an 8-bit conversion, which is a channel average and always the larger of the
# two. Measure it per cohort: if lumen_fraction comes back at 1e-5 the threshold
# is above the lumens and they are being counted as tissue.
WHITE_THRESH = 0.85


# ---------------------------------------------------------------------------
# mask preparation
# ---------------------------------------------------------------------------

def clean_mask(
    mask: np.ndarray,
    min_object_px: int = 16,
    closing_px: int = 0,
) -> np.ndarray:
    """Remove speckle and optionally close sub-resolution gaps.

    Topology is far more sensitive to mask noise than area is — β₀ counts every
    speck as a component — so this must run before the Betti numbers and the
    parameters must be identical for real and virtual (§5.4.4).
    """
    m = np.asarray(mask, dtype=bool)

    if closing_px > 0:
        structure = np.ones((2 * closing_px + 1, 2 * closing_px + 1), dtype=bool)
        m = ndimage.binary_closing(m, structure=structure)

    if min_object_px > 1 and m.any():
        lab, n = ndimage.label(m)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_object_px
            keep[0] = False          # background label
            m = keep[lab]

    return m


# ---------------------------------------------------------------------------
# individual descriptors
# ---------------------------------------------------------------------------

def collagen_fraction(
    labels: np.ndarray,
    label_tissue: int = LABEL_TISSUE,
    label_psr: int = LABEL_PSR,
) -> float:
    """Positive area fraction — CPA for PSR (same definition as
    `compare_psr.py:compute_psr_fraction`, evaluated on a region rather than a WSI).

    Denominator is tissue only; background is excluded, per
    `evaluation_metrics.tex`. Returns NaN for a region with no tissue.
    """
    labels = np.asarray(labels)
    tissue = int(np.count_nonzero(labels == label_tissue))
    psr = int(np.count_nonzero(labels == label_psr))
    denom = tissue + psr
    return float(psr / denom) if denom else float("nan")


def betti(mask: np.ndarray) -> Tuple[int, int]:
    """(β₀, β₁) of a 2D binary mask.

    β₀ = connected components. β₁ = loops, i.e. connected components of the
    background enclosed by the foreground.

    β₁ is the component that catches the lumen-filler failure of §5.3: collagen
    rendered as solid blobs where it should form rings around lumens has the same
    area and the same β₀, but β₁ collapses to zero.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0, 0
    b0 = int(ndimage.label(m)[1])
    holes = ndimage.binary_fill_holes(m) & ~m
    b1 = int(ndimage.label(holes)[1]) if holes.any() else 0
    return b0, b1


def regional_dispersion(mask: np.ndarray, sigma: float = 2.0) -> float:
    """Spread of the dominant orientation across the region, in [0, 1].

    Structure tensor on the binary mask; orientation is mod π so circular
    statistics use the doubled angle:  1 − |mean(exp(2iθ))|.

    0 = every structure points the same way, 1 = no preferred direction. This is
    deliberately *not* "local coherence" (are structures elongated at all?), which
    is near-identical for aligned and scrambled fibres and so cannot separate them
    — see §5.4.3.
    """
    m = np.asarray(mask, dtype=float)
    if m.sum() < 2:
        return float("nan")

    gy, gx = np.gradient(m)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)

    energy = jxx + jyy
    if not np.isfinite(energy).any() or energy.max() <= 0:
        return float("nan")

    # only where there is an edge to orient; below that it is noise
    sel = energy > np.percentile(energy[energy > 0], 90) if (energy > 0).any() else None
    if sel is None or not sel.any():
        return float("nan")

    theta = 0.5 * np.arctan2(2.0 * jxy[sel], jxx[sel] - jyy[sel])
    resultant = np.abs(np.mean(np.exp(2j * theta)))
    return float(1.0 - resultant)


def he_bright(he_rgb: np.ndarray, white_thresh: float = WHITE_THRESH) -> np.ndarray:
    """Near-white mask of an H&E image. Accepts uint8 or float [0,1]."""
    img = np.asarray(he_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected H&E RGB [H,W,>=3], got shape {img.shape}")
    if img.max() > 1.5:                       # uint8-style input
        img = img / 255.0
    return np.all(img[..., :3] > white_thresh, axis=2)


def he_tissue_footprint(he_rgb: np.ndarray,
                        white_thresh: float = WHITE_THRESH) -> np.ndarray:
    """Tissue outline *including* its internal lumens, for a whole WSI.

    Compute this once per slide and crop it per region — never per region
    directly. `binary_fill_holes` only fills background that is fully enclosed,
    so a lumen straddling a region boundary is not enclosed within that crop, is
    not filled, and would be silently counted as outside the tissue. At 1.5 mm
    regions with vessels of 100–500 µm that is a large fraction of all lumens.

    At WSI level the problem disappears: genuine slide background is contiguous
    with the image border and correctly stays unfilled, while every real lumen is
    enclosed.
    """
    return ndimage.binary_fill_holes(~he_bright(he_rgb, white_thresh))


def lumen_mask(
    rgb: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    white_thresh: float = WHITE_THRESH,
) -> Tuple[np.ndarray, np.ndarray]:
    """(lumen, footprint) — enclosed whitespace, and the tissue it sits inside.

    Split out of `lumen_tissue_fraction` so the same mask can feed `betti`: the
    topology of the lumen space is what distinguishes vessels from any pale
    patch, and it is the direct test of the §5.3 lumen-filler failure — a model
    that paints collagen over vessels keeps the area and loses the loops.

    `rgb` is whichever image the whitespace is being read from. On the reference
    side that is the real H&E; on the virtual side it is the **generated SR**,
    with `tissue_mask` still the H&E footprint. Taking the footprint from the
    H&E rather than by thresholding the SR is deliberate: the SR footprint is
    unstable at both ends of the threshold sweep (it erodes into tissue below
    ~0.60 and swallows the slide background above ~0.70), while the H&E one is
    stable across 0.500-0.675.
    """
    bright = he_bright(rgb, white_thresh)

    if tissue_mask is not None:
        footprint = np.asarray(tissue_mask, dtype=bool)
    else:
        # pad the border as tissue so a lumen cut by the crop edge is still
        # enclosed, then strip the pad back off
        padded = np.pad(~bright, 1, mode="constant", constant_values=True)
        footprint = ndimage.binary_fill_holes(padded)[1:-1, 1:-1]

    return bright & footprint, footprint


def lumen_tissue_fraction(
    he_rgb: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    white_thresh: float = WHITE_THRESH,
) -> Tuple[float, float]:
    """(lumen_fraction, tissue_fraction) from the H&E input.

    These are the floor-free terms of §6.0: the virtual stain depicts the *same
    physical section* as the H&E it was generated from, tile-for-tile aligned, so
    the H&E is an exact reference with no level offset. Any deviation is model
    error — the structure was visible in the input.

    `tissue_mask` should be the WSI-level footprint from `he_tissue_footprint`,
    cropped to this region. Omitting it falls back to a per-region estimate that
    pads the border as foreground so border-touching lumens still register — an
    approximation, adequate for a standalone call but not for the pipeline.

    Caveat (§6.0): tears, folds and processing artefacts also read as whitespace
    and do differ between sections — threshold conservatively.
    """
    lumen, footprint = lumen_mask(he_rgb, tissue_mask, white_thresh)

    area = int(footprint.size)
    n_tissue = int(np.count_nonzero(footprint))
    if area == 0 or n_tissue == 0:
        return float("nan"), 0.0

    return float(np.count_nonzero(lumen) / n_tissue), float(n_tissue / area)


# ---------------------------------------------------------------------------
# the vector
# ---------------------------------------------------------------------------

def phi_struct(
    psr_labels: np.ndarray,
    he_rgb: Optional[np.ndarray] = None,
    *,
    mpp: float = 0.442,
    min_object_px: int = 16,
    closing_px: int = 0,
    sigma: float = 2.0,
    white_thresh: float = WHITE_THRESH,
    tissue_mask: Optional[np.ndarray] = None,
    label_tissue: int = LABEL_TISSUE,
    label_psr: int = LABEL_PSR,
) -> np.ndarray:
    """Compute the 7-vector for one region.

    Parameters
    ----------
    psr_labels : [H,W] nnU-Net label map (0 background, 1 tissue, 2 PSR-positive).
    he_rgb     : [H,W,3] for the same region, whichever image the whitespace is
                 read from — the real H&E on the reference side, the generated
                 SR on the virtual side. If None the three lumen components come
                 back NaN and the four collagen terms are still valid.
    tissue_mask: the H&E footprint, cropped to this region. Supplies the
                 denominator for the lumen densities as well as the enclosure.
    mpp        : microns per pixel, for the count → density conversion.

    Returns
    -------
    np.ndarray, shape (7,), ordered as PHI_NAMES. NaN marks "not computable
    here", never zero — a zero would be silently absorbed by downstream means.
    `tissue_fraction` is not part of the vector; it is `tissue_mask.mean()`,
    which the caller already has.
    """
    labels = np.asarray(psr_labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label map, got shape {labels.shape}")

    out = np.full(PHI_DIM, np.nan, dtype=np.float64)

    out[0] = collagen_fraction(labels, label_tissue, label_psr)

    collagen = clean_mask(labels == label_psr, min_object_px, closing_px)

    # densities are per mm^2 of TISSUE, not of the region: a region that is half
    # background must not read as half the component density.
    n_tissue_px = int(np.count_nonzero((labels == label_tissue) | (labels == label_psr)))
    tissue_mm2 = n_tissue_px * (mpp ** 2) / 1e6

    if tissue_mm2 > 0:
        b0, b1 = betti(collagen)
        out[1] = b0 / tissue_mm2
        out[2] = b1 / tissue_mm2

    out[3] = regional_dispersion(collagen, sigma=sigma)

    if he_rgb is not None:
        lum, footprint = lumen_mask(he_rgb, tissue_mask, white_thresh)
        n_footprint = int(np.count_nonzero(footprint))
        if n_footprint:
            out[4] = float(np.count_nonzero(lum) / n_footprint)

            # Lumen densities are per mm2 of the H&E FOOTPRINT, not of the
            # label mask's tissue. The footprint is the one denominator
            # available on both sides: the virtual side has generated collagen
            # labels, the reference side is the real H&E with no labels at all,
            # and a density is only comparable if its denominator is.
            footprint_mm2 = n_footprint * (mpp ** 2) / 1e6
            if footprint_mm2 > 0:
                lb0, lb1 = betti(clean_mask(lum, min_object_px, closing_px))
                out[5] = lb0 / footprint_mm2
                out[6] = lb1 / footprint_mm2

    return out
