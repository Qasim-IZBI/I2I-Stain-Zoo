# datasets/target_only_dataset.py
from __future__ import annotations
import os
from typing import Callable, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

def _list_images(root: str) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths

class TargetOnlyDataset(Dataset):
    """Returns {"B": tensor, "path_B": str} for pretraining diffusion on target domain."""
    def __init__(self, root_B: str, transform: Optional[Callable] = None):
        self.paths = _list_images(root_B)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is None:
            raise ValueError("TargetOnlyDataset requires a transform (PIL->Tensor normalized to [-1,1]).")
        x = self.transform(img)
        if not torch.is_tensor(x):
            raise TypeError("Transform must return torch.Tensor.")
        return {"B": x, "path_B": p}
