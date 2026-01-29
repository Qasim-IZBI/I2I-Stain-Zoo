# evaluation.py
"""
Compute distribution-level distance between TWO UNPAIRED image sets using FID-style Fréchet distance.

Feature backends:
  - inception: torchvision InceptionV3 pool3 (2048-d), classic FID
  - dino: DINOv2 ViT features (typically 768/1024-d depending on model)

Example (classic FID):
  python evaluation.py --path_real data/CD13 --path_fake results/he_to_cd13 --backend inception --device cuda

Example (DINO features + Fréchet distance):
  python evaluation.py --path_real data/CD13 --path_fake results/he_to_cd13 --backend dino --dino_model dinov2_vits14 --device cuda

Notes:
- For DINO backend we still compute the same Fréchet distance formula; it's "FID-like" but not the canonical Inception FID.
- Images are treated as RGB.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths


class ImageFolderList(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x


# ============================================================
# Inception backend (classic FID)
# ============================================================

class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 -> 2048-d features from final avgpool (pool3)."""
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT

        # IMPORTANT: torchvision may enforce aux_logits=True when weights are used.
        m = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        m.eval()

        # We do NOT use AuxLogits; we only build the feature trunk up to avgpool.
        self.features = nn.Sequential(
            m.Conv2d_1a_3x3,
            m.Conv2d_2a_3x3,
            m.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            m.Conv2d_3b_1x1,
            m.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            m.Mixed_5b,
            m.Mixed_5c,
            m.Mixed_5d,
            m.Mixed_6a,
            m.Mixed_6b,
            m.Mixed_6c,
            m.Mixed_6d,
            m.Mixed_6e,
            m.Mixed_7a,
            m.Mixed_7b,
            m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)     # [B,2048,1,1]
        return h.flatten(1)      # [B,2048]



def inception_transform():
    return transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ============================================================
# DINO backend (DINOv2 via torch.hub)
# ============================================================

class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 ViT -> global embedding.
    Uses torch.hub 'facebookresearch/dinov2' models.

    Supported model names commonly include:
      dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    """
    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        # NOTE: requires internet the first time to download weights.
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [B, D]
        return self.model(x)


def dino_transform(image_size: int = 224):
    # DINOv2 expects ImageNet normalization as well
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ============================================================
# FID computation
# ============================================================

def compute_activations(
    folder: str,
    extractor: nn.Module,
    device: torch.device,
    transform,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    paths = list_images(folder)
    ds = ImageFolderList(paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    acts = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        a = extractor(x).detach().cpu().numpy()
        acts.append(a)

    return np.concatenate(acts, axis=0)  # [N,D]


def compute_stats(acts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mat = (mat + mat.T) * 0.5
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, a_min=0.0, a_max=None)
    return (v * np.sqrt(w + eps)) @ v.T


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    diff = mu1 - mu2
    diff_sq = float(diff @ diff)

    covmean = sqrtm_psd(s1 @ s2)
    covmean = (covmean + covmean.T) * 0.5

    tr = float(np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    return max(diff_sq + tr, 0.0)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("FID-style distance for unpaired image sets (Inception or DINO features)")
    ap.add_argument("--path_real", required=True, type=str, help="Folder with real target-domain images")
    ap.add_argument("--path_fake", required=True, type=str, help="Folder with generated images")
    ap.add_argument("--backend", choices=["inception", "dino"], default="inception")

    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    # DINO options
    ap.add_argument("--dino_model", default="dinov2_vits14", type=str)
    ap.add_argument("--dino_image_size", default=224, type=int)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if args.backend == "inception":
        extractor = InceptionFeatureExtractor().to(device).eval()
        tfm = inception_transform()
    else:
        extractor = DINOv2FeatureExtractor(model_name=args.dino_model).to(device).eval()
        tfm = dino_transform(image_size=args.dino_image_size)

    acts_real = compute_activations(args.path_real, extractor, device, tfm, args.batch_size, args.num_workers)
    acts_fake = compute_activations(args.path_fake, extractor, device, tfm, args.batch_size, args.num_workers)

    mu_r, sig_r = compute_stats(acts_real)
    mu_f, sig_f = compute_stats(acts_fake)

    fid_like = frechet_distance(mu_r, sig_r, mu_f, sig_f)

    label = "FID" if args.backend == "inception" else "Fréchet(DINO)"
    print(f"{label} (real={args.path_real} vs fake={args.path_fake}): {fid_like:.4f}")
    print(f"N_real={acts_real.shape[0]}, N_fake={acts_fake.shape[0]}, feat_dim={acts_real.shape[1]}")


if __name__ == "__main__":
    main()
