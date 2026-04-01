# datasets/single_domain_dataset.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _list_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


def _list_images_from_range(root: str, start: int, end: int) -> List[str]:
    """
    Collect images from numbered subfolders root/001/images/ ... root/006/images/.
    Raises FileNotFoundError if any expected folder is missing.
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


class SingleDomainDataset(Dataset):
    """
    Loads images from a single folder (used for inference).

    data_range : tuple (start, end), optional
        When provided, loads tiles only from numbered subfolders
        root/001/images/ through root/006/images/ (inclusive).
        When None, walks the entire root directory.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        data_range: Optional[Tuple[int, int]] = None,
    ):
        if data_range is not None:
            self.paths = _list_images_from_range(root, *data_range)
        else:
            self.paths = _list_images(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is None:
            raise ValueError("SingleDomainDataset requires a transform (PIL->Tensor normalized to [-1,1]).")

        x = self.transform(img)
        if not torch.is_tensor(x):
            raise TypeError("Transform must return torch.Tensor.")
        return x, path
