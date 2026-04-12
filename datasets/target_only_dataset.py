# datasets/target_only_dataset.py
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.common import list_images, list_images_from_range


class TargetOnlyDataset(Dataset):
    """
    Returns {"B": tensor, "path_B": str} for pretraining diffusion on target domain.

    data_range : tuple (start, end), optional
        When provided, loads tiles only from numbered subfolders
        root/001/images/ through root/006/images/ (inclusive).
        When None, walks the entire root directory.
    """
    def __init__(
        self,
        root_B: str,
        transform: Optional[Callable] = None,
        data_range: Optional[Tuple[int, int]] = None,
    ):
        if data_range is not None:
            self.paths = list_images_from_range(root_B, *data_range)
        else:
            self.paths = list_images(root_B)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self.transform is None:
            raise ValueError("TargetOnlyDataset requires a transform (PIL->Tensor normalized to [-1,1]).")

        for attempt in range(5):
            try:
                p = self.paths[idx % len(self.paths)]
                img = Image.open(p).convert("RGB")
                break
            except OSError as e:
                print(f"[Dataset] I/O error reading tile (attempt {attempt + 1}/5), skipping: {e}")
                idx = (idx + 7919) % len(self.paths)
        else:
            raise RuntimeError("[Dataset] Failed to load a valid tile after 5 attempts.")

        x = self.transform(img)
        if not torch.is_tensor(x):
            raise TypeError("Transform must return torch.Tensor.")
        return {"B": x, "path_B": p}
