"""Stain-perturbation sensitivity — bounding the segmenter's measurement artefact.

kidney_ood_data_plan.md §6.2 warns that an identical segmentation rule can behave
differently on real and virtual PSR if the virtual stain's colour statistics are
slightly off, and that the resulting artefact is **indistinguishable from genuine
model bias**. Applying the same segmenter to both arms does not fix this: it
cancels *anatomy-driven* error, which is common to both, but not
*appearance-driven* error, because appearance is exactly where the two arms
differ.

The test here isolates that component by holding anatomy fixed and moving only
appearance. Take a **real** PSR slide and transform its colour statistics toward
those of the **virtual** PSR, in steps t ∈ [0, 1]:

    t = 0   the real slide, untouched
    t = 1   real anatomy wearing the virtual stain's colour statistics

Re-segment at each t. Because the underlying tissue never changes, *any* movement
in CPA, β₀, β₁ or dispersion is pure measurement artefact — a direct estimate of
how much of a measured "bias" is really the segmenter reacting to colour.

Calibrating to the observed real-vs-virtual gap is what makes the magnitude
meaningful; an arbitrary perturbation size would answer a question nobody asked.

Transfer is Reinhard in CIE LAB, the same formulation the pruned `--color_ref`
path used. At t = 0 it is the identity by construction (transferring an image to
its own statistics is a no-op), which the tests pin down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# CIE LAB (recovered from the pre-prune inference.py --color_ref implementation)
# --------------------------------------------------------------------------

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """[H,W,3] float [0,1] -> CIE LAB."""
    img = np.asarray(img, dtype=np.float32).clip(0, 1)
    lin = np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = (lin.reshape(-1, 3) @ M.T).reshape(img.shape)
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(xyz.clip(0)), (kappa * xyz + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE LAB -> [H,W,3] float [0,1]."""
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps, kappa = 0.008856, 903.3
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > eps * kappa, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([x, y, z], axis=-1) * np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    M_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    lin = (xyz.reshape(-1, 3) @ M_inv.T).reshape(xyz.shape).clip(0, None)
    rgb = np.where(lin > 0.0031308, 1.055 * lin ** (1.0 / 2.4) - 0.055, 12.92 * lin)
    return rgb.clip(0, 1)


# --------------------------------------------------------------------------
# stain statistics and interpolation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StainStats:
    """Per-channel LAB mean and sd over tissue pixels."""
    mean: np.ndarray          # (3,)
    std: np.ndarray           # (3,)
    n_pixels: int

    def as_dict(self) -> dict:
        return {"mean": [float(v) for v in self.mean],
                "std": [float(v) for v in self.std],
                "n_pixels": int(self.n_pixels)}


def _to_unit_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected RGB [H,W,>=3], got {a.shape}")
    a = a[..., :3]
    return a / 255.0 if a.max() > 1.5 else a


def stain_stats(img: np.ndarray, tissue_mask: Optional[np.ndarray] = None) -> StainStats:
    """LAB statistics over tissue only.

    Background is excluded deliberately: glass is the same in both arms, and
    including it would dilute the very difference being measured.
    """
    lab = rgb_to_lab(_to_unit_rgb(img))
    if tissue_mask is None:
        flat = lab.reshape(-1, 3)
    else:
        m = np.asarray(tissue_mask, dtype=bool)
        if m.shape != lab.shape[:2]:
            raise ValueError(f"mask {m.shape} does not match image {lab.shape[:2]}")
        flat = lab[m]
    if flat.shape[0] == 0:
        raise ValueError("no tissue pixels to compute stain statistics from")
    return StainStats(mean=flat.mean(axis=0), std=flat.std(axis=0), n_pixels=flat.shape[0])


def pool_stats(stats: list) -> StainStats:
    """Pool per-slide statistics into one reference, weighted by pixel count."""
    if not stats:
        raise ValueError("nothing to pool")
    w = np.array([s.n_pixels for s in stats], dtype=np.float64)
    w = w / w.sum()
    mean = np.sum([s.mean * wi for s, wi in zip(stats, w)], axis=0)
    # pooled variance: within-slide plus between-slide spread of the means
    within = np.sum([(s.std ** 2) * wi for s, wi in zip(stats, w)], axis=0)
    between = np.sum([((s.mean - mean) ** 2) * wi for s, wi in zip(stats, w)], axis=0)
    return StainStats(mean=mean, std=np.sqrt(within + between),
                      n_pixels=int(sum(s.n_pixels for s in stats)))


def interpolate_stats(src: StainStats, dst: StainStats, t: float) -> StainStats:
    """Target statistics a fraction `t` of the way from src to dst.

    t = 0 returns src exactly, which makes the transfer at t = 0 an identity.
    """
    t = float(t)
    return StainStats(
        mean=(1.0 - t) * src.mean + t * dst.mean,
        std=(1.0 - t) * src.std + t * dst.std,
        n_pixels=src.n_pixels,
    )


def reinhard_transfer(
    img: np.ndarray,
    src: StainStats,
    dst: StainStats,
    tissue_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Move `img` from `src` statistics to `dst`, in LAB.

    Returns float [0,1] RGB. Background is left untouched when a mask is given —
    perturbing glass would change the tissue/background contrast and confound the
    very thing under test.
    """
    rgb = _to_unit_rgb(img)
    lab = rgb_to_lab(rgb)
    scale = dst.std / np.maximum(src.std, eps)
    out = (lab - src.mean) * scale + dst.mean
    rgb_out = lab_to_rgb(out)
    if tissue_mask is not None:
        m = np.asarray(tissue_mask, dtype=bool)[..., None]
        rgb_out = np.where(m, rgb_out, rgb)
    return rgb_out


def perturbation_series(
    img: np.ndarray,
    src: StainStats,
    dst: StainStats,
    fractions: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    tissue_mask: Optional[np.ndarray] = None,
) -> dict:
    """{t: perturbed RGB} moving `img` from its own statistics toward `dst`.

    Anatomy is identical across the series by construction — only appearance
    moves — so any descriptor change measured downstream is measurement artefact,
    not biology and not model bias.
    """
    return {
        float(t): reinhard_transfer(img, src, interpolate_stats(src, dst, t), tissue_mask)
        for t in fractions
    }


def clipped_fraction(
    img: np.ndarray,
    src: StainStats,
    dst: StainStats,
    tissue_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> float:
    """Fraction of tissue pixels driven out of gamut by the transfer.

    Matters for the validity of the test, not just for tidiness. Clipping is
    non-invertible: it flattens highlights and can genuinely alter relative
    structure, so a series with heavy clipping no longer holds anatomy fixed and
    its sensitivity estimate is contaminated. Report it alongside every t; if it
    climbs into the percent range at the fractions you care about, shrink the
    range rather than trusting the number.
    """
    rgb = _to_unit_rgb(img)
    lab = rgb_to_lab(rgb)
    scale = dst.std / np.maximum(src.std, eps)
    out = (lab - src.mean) * scale + dst.mean

    # replicate lab_to_rgb without the final clamp, to see what would be lost
    L, a, b = out[..., 0], out[..., 1], out[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    e, kappa = 0.008856, 903.3
    x = np.where(fx ** 3 > e, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > e * kappa, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > e, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([x, y, z], axis=-1) * np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    M_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    lin = (xyz.reshape(-1, 3) @ M_inv.T).reshape(xyz.shape)
    srgb = np.where(lin > 0.0031308,
                    1.055 * np.abs(lin).clip(1e-12) ** (1 / 2.4) - 0.055,
                    12.92 * lin)
    out_of_gamut = (lin < 0).any(axis=-1) | (srgb > 1.0).any(axis=-1) | (srgb < 0.0).any(axis=-1)
    if tissue_mask is not None:
        m = np.asarray(tissue_mask, dtype=bool)
        if not m.any():
            return 0.0
        return float(out_of_gamut[m].mean())
    return float(out_of_gamut.mean())


def appearance_gap(src: StainStats, dst: StainStats) -> dict:
    """How far apart two image sets are in LAB, for reporting.

    `delta_mean_over_sd` expresses the mean shift in units of the source's own
    spread — the scale-free way to say whether the virtual stain sits inside or
    outside the natural variation of the real one.
    """
    d_mean = dst.mean - src.mean
    return {
        "delta_mean_L": float(d_mean[0]),
        "delta_mean_a": float(d_mean[1]),
        "delta_mean_b": float(d_mean[2]),
        "std_ratio": [float(v) for v in (dst.std / np.maximum(src.std, 1e-6))],
        "delta_mean_over_sd": [float(v) for v in (d_mean / np.maximum(src.std, 1e-6))],
    }
