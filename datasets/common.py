# datasets/common.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_images(root: str) -> List[str]:
    """Recursively list all image files under root, sorted."""
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


def list_images_from_range(root: str, start: int, end: int) -> List[str]:
    """
    Collect images from numbered subfolders root/001/images/ ... root/NNN/images/.
    Raises FileNotFoundError if any expected folder is missing or no images are found.
    """
    paths: List[str] = []
    root_path = Path(root)
    for i in range(start, end + 1):
        folder = root_path / f"{i:03d}" / "images"
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {folder}. "
                f"Check --data_range or re-run tiling."
            )
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(str(folder / fn))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found in range {start:03d}–{end:03d} under: {root}"
        )
    return paths
