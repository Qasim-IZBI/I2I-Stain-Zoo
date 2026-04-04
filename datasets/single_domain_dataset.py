# datasets/single_domain_dataset.py
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.common import list_images, list_images_from_range


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
            self.paths = list_images_from_range(root, *data_range)
        else:
            self.paths = list_images(root)
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
