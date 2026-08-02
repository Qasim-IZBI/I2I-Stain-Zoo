"""Region grid over a reconstructed WSI (kidney_ood_data_plan.md §4.2).

Bias is computed on regions **~1–2 mm across**, not on 256² tiles: the floor
shrinks with region size as structure displacements average out, while bias does
not. Liver lobular architecture is ~1–2 mm; kidney glomeruli are ~200 µm at
~300–500 µm spacing — both sit inside that window.

Regions are also the only scale at which β₀/β₁ and dispersion are meaningful,
because components and loops cross tile boundaries: the topology of a region is
not a function of its tiles' topologies. That is why the pipeline consumes
stitched WSI masks rather than per-tile predictions.

Scale trap
----------
`utils.reconstruct_wsi` resizes each tile back up from `saved_size` to
`tile_size` and places it at the original `x, y` (utils.py:83-84). A
reconstructed WSI is therefore at the **source** resolution — 0.221 µm/px for
this cohort — even though the model only ever saw 0.442 µm/px. Sizing regions in
millimetres rather than pixels keeps that from silently halving every region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

# Source acquisition resolution: 20x, downsampled 2x to the 0.442 um/px the
# model consumes. Reconstructions live at this scale.
SOURCE_MPP = 0.221


@dataclass(frozen=True)
class Region:
    """An axis-aligned crop box in reconstructed-WSI pixel coordinates."""
    wsi: str
    index: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> Tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def crop(self, arr: np.ndarray) -> np.ndarray:
        ys, xs = self.slices
        return arr[ys, xs]

    @property
    def stem(self) -> str:
        return f"{self.wsi}_r{self.index:04d}"


def wsi_extent(metadata_csv: Path) -> Tuple[str, int, int]:
    """(source_file, height, width) of the reconstruction this CSV describes.

    Mirrors the canvas sizing in `utils.reconstruct_wsi` (utils.py:61-64) so the
    grid always matches the array it will be cropping.
    """
    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"empty metadata: {metadata_csv}")
    sources = df["source_file"].unique()
    if len(sources) != 1:
        raise ValueError(f"expected one source_file in {metadata_csv}, got {list(sources)}")
    h = int((df["y"] + df["tile_size"]).max())
    w = int((df["x"] + df["tile_size"]).max())
    return str(sources[0]), h, w


def region_grid(
    metadata_csv: Path,
    *,
    region_mm: float = 1.5,
    mpp: float = SOURCE_MPP,
    drop_partial: bool = True,
) -> List[Region]:
    """Non-overlapping grid of ~`region_mm` square regions over one WSI.

    `drop_partial` discards edge regions smaller than the nominal size, so every
    region has the same area and the per-mm² densities are not skewed by slivers.
    Tissue filtering is a separate step (`filter_by_tissue`) because it needs the
    mask, which the caller loads.
    """
    wsi, h, w = wsi_extent(metadata_csv)
    side = int(round(region_mm * 1000.0 / mpp))
    if side < 1:
        raise ValueError(f"region_mm={region_mm} at mpp={mpp} gives a {side}px region")

    regions: List[Region] = []
    idx = 0
    for y0 in range(0, h, side):
        for x0 in range(0, w, side):
            y1, x1 = min(y0 + side, h), min(x0 + side, w)
            if drop_partial and (y1 - y0 < side or x1 - x0 < side):
                continue
            regions.append(Region(wsi=wsi, index=idx, y0=y0, y1=y1, x0=x0, x1=x1))
            idx += 1
    return regions


def filter_by_tissue(
    regions: List[Region],
    labels: np.ndarray,
    *,
    min_tissue_fraction: float = 0.25,
    label_tissue: int = 1,
    label_psr: int = 2,
) -> List[Region]:
    """Keep regions whose tissue coverage clears a threshold.

    Background-dominated regions produce meaningless densities (tiny denominator)
    and noisy dispersion, and they inflate the apparent n while contributing no
    information.
    """
    tissue = (labels == label_tissue) | (labels == label_psr)
    keep = []
    for r in regions:
        patch = r.crop(tissue)
        if patch.size and patch.mean() >= min_tissue_fraction:
            keep.append(r)
    return keep


def iter_metadata_csvs(tiles_root: Path) -> Iterator[Path]:
    """Yield per-WSI tiles_metadata.csv paths under a dataset root.

    Same layout convention as `reconstruct.py --metadata <dataset dir>`:
    `<root>/<NNN>/tiles_metadata.csv`.
    """
    root = Path(tiles_root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.glob("*/tiles_metadata.csv")):
        yield p


def region_area_mm2(region: Region, mpp: float = SOURCE_MPP) -> float:
    """Physical area of a region, for sanity checks and reporting."""
    return (region.y1 - region.y0) * (region.x1 - region.x0) * (mpp ** 2) / 1e6


def region_centres_mm(regions: List[Region], mpp: float = SOURCE_MPP) -> np.ndarray:
    """[n_regions, 2] centre coordinates in millimetres, for variogram lags."""
    return np.array(
        [[(r.y0 + r.y1) / 2.0, (r.x0 + r.x1) / 2.0] for r in regions],
        dtype=np.float64,
    ) * mpp / 1000.0
