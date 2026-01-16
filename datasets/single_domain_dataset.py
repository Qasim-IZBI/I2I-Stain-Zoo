# datasets/single_domain_dataset.py
from __future__ import annotations

import os
from typing import Callable, List, Optional

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


class SingleDomainDataset(Dataset):
    """
    Loads images from a single folder (used for inference).
    Returns tensor only, plus path if you want it.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
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
