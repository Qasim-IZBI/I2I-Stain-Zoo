# datasets/unpaired_dataset.py
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

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
    ):
        self.A_paths = _list_images(root_A)
        self.B_paths = _list_images(root_B)
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

        # Worker-safe "random": deterministic mix of idx and seed
        # 9973 is just a prime to scramble indices.
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