# datasets/unpaired_dataset.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.common import list_images, list_images_from_range


class UnpairedDataset(Dataset):
    """
    Unpaired dataset for domain A and B.

    Returns:
      {
        "A": tensor [C,H,W],
        "B": tensor [C,H,W],
        "path_A": str,
        "path_B": str,
      }

    data_range : tuple (start, end), optional
        When provided, loads tiles only from numbered subfolders
        root/001/images/ through root/006/images/ (inclusive).
        When None, walks the entire root directory.

    pairing:
      - "random": pseudo-random but worker-safe (deterministic function of idx)
      - "serial": B index = idx % len(B)
    """
    def __init__(
        self,
        root_A: str,
        root_B: str,
        transform: Optional[Callable] = None,
        seed: int = 0,
        pairing: str = "random",
        data_range: Optional[Tuple[int, int]] = None,
    ):
        if data_range is not None:
            self.A_paths = list_images_from_range(root_A, *data_range)
            self.B_paths = list_images_from_range(root_B, *data_range)
        else:
            self.A_paths = list_images(root_A)
            self.B_paths = list_images(root_B)
        self.transform = transform
        self.seed = int(seed)
        if pairing not in ("random", "serial"):
            raise ValueError(f"pairing must be 'random' or 'serial', got: {pairing}")
        self.pairing = pairing

    def __len__(self) -> int:
        return len(self.A_paths)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _choose_b_index(self, idx: int) -> int:
        if self.pairing == "serial":
            return idx % len(self.B_paths)
        return (idx * 9973 + self.seed) % len(self.B_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path_A = self.A_paths[idx % len(self.A_paths)]
        path_B = self.B_paths[self._choose_b_index(idx)]

        img_A = self._load_rgb(path_A)
        img_B = self._load_rgb(path_B)

        if self.transform is None:
            raise ValueError("UnpairedDataset requires a transform (PIL->Tensor normalized to [-1,1]).")

        A = self.transform(img_A)
        B = self.transform(img_B)

        if not torch.is_tensor(A) or not torch.is_tensor(B):
            raise TypeError("Transform must return torch.Tensor for both domains.")

        return {"A": A, "B": B, "path_A": path_A, "path_B": path_B}
