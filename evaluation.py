# evaluation.py
"""
Compute evaluation metrics between two image sets.

Metrics:
  - fid: Distribution-level Fréchet distance (unpaired, Inception or DINO features)
  - ssim: Structural Similarity Index (paired, matched by filename)
  - patch_ssim: Patch-based SSIM — extract random patches and compute SSIM per patch (paired)
  - lpips: Learned Perceptual Image Patch Similarity using VGG16 features (paired, lower=better)
  - regen_error: Cycle reconstruction MAE — translate A→B'→A' and measure |A−A'| per pixel;
                 requires --path_A, --model, --ckpt; optionally saves heatmap/overlay images
                 via --overlay_dir. Not supported for MIUDiff (no B→A generator).

Feature backends (FID only):
  - inception: torchvision InceptionV3 pool3 (2048-d), classic FID
  - dino: DINOv2 ViT features (typically 768/1024-d depending on model)

Example (classic FID):
  python evaluation.py --metric fid --path_real data/CD13 --path_fake results/he_to_cd13 --backend inception --device cuda

Example (SSIM):
  python evaluation.py --metric ssim --path_real data/CD13 --path_fake results/he_to_cd13

Example (Patch SSIM):
  python evaluation.py --metric patch_ssim --path_real data/CD13 --path_fake results/he_to_cd13 --patch_size 64 --patches_per_image 16

Example (Cycle reconstruction error with overlays):
  python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \\
      --direction A2B --overlay_dir ./regen_overlays/ --device cuda

Notes:
- For DINO backend we still compute the same Fréchet distance formula; it's "FID-like" but not the canonical Inception FID.
- SSIM and patch_ssim require paired images matched by filename.
- Images are treated as RGB.
- regen_error MAE is reported in [0, 255] pixel scale.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


from datasets.common import IMG_EXTS, list_images


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
# SSIM computation (paired)
# ============================================================

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # [size, size]
    k2d = k2d.expand(channels, 1, size, size).contiguous()
    return k2d


def compute_ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = (0.01 * 255) ** 2,
    C2: float = (0.03 * 255) ** 2,
) -> torch.Tensor:
    """
    Compute per-pixel SSIM map between two [B, C, H, W] tensors in [0, 255] range.
    Returns SSIM map of shape [B, 1, H', W'].
    """
    channels = img1.shape[1]
    kernel = _gaussian_kernel_2d(window_size, 1.5, channels).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = nn.functional.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = nn.functional.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den  # [B, C, H, W]
    return ssim_map.mean(dim=1, keepdim=True)  # average over channels -> [B, 1, H, W]


def list_paired_images(path_real: str, path_fake: str) -> List[Tuple[str, str]]:
    """Match images by filename between two folders."""
    real_map = {}
    for p in list_images(path_real):
        real_map[os.path.basename(p)] = p
    fake_map = {}
    for p in list_images(path_fake):
        fake_map[os.path.basename(p)] = p

    common = sorted(set(real_map) & set(fake_map))
    if len(common) == 0:
        raise FileNotFoundError(
            f"No matching filenames between {path_real} and {path_fake}. "
            "SSIM requires paired images with the same filenames."
        )
    return [(real_map[k], fake_map[k]) for k in common]


def compute_ssim(
    path_real: str,
    path_fake: str,
    image_size: int = 256,
) -> Tuple[float, List[float]]:
    """
    Compute vanilla SSIM between paired images (matched by filename).
    Returns (mean_ssim, list of per-image ssim values).
    """
    pairs = list_paired_images(path_real, path_fake)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # [0, 1]
    ])

    scores = []
    for real_path, fake_path in pairs:
        img_r = tfm(Image.open(real_path).convert("RGB")).unsqueeze(0) * 255.0
        img_f = tfm(Image.open(fake_path).convert("RGB")).unsqueeze(0) * 255.0
        ssim_map = compute_ssim_map(img_r, img_f)
        scores.append(float(ssim_map.mean()))

    return float(np.mean(scores)), scores


def compute_patch_ssim(
    path_real: str,
    path_fake: str,
    image_size: int = 256,
    patch_size: int = 64,
    patches_per_image: int = 16,
    seed: int = 42,
) -> Tuple[float, List[float]]:
    """
    Compute patch-based SSIM: extract random patches from paired images,
    compute SSIM per patch, and average.
    Returns (mean_patch_ssim, list of per-image mean patch ssim values).
    """
    rng = np.random.RandomState(seed)
    pairs = list_paired_images(path_real, path_fake)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    per_image_scores = []
    for real_path, fake_path in pairs:
        img_r = tfm(Image.open(real_path).convert("RGB")) * 255.0  # [C, H, W]
        img_f = tfm(Image.open(fake_path).convert("RGB")) * 255.0

        H, W = img_r.shape[1], img_r.shape[2]
        max_y = H - patch_size
        max_x = W - patch_size
        if max_y < 0 or max_x < 0:
            raise ValueError(
                f"patch_size={patch_size} exceeds image dimensions {H}x{W}. "
                "Use a smaller patch_size or larger image_size."
            )

        patch_scores = []
        tops = rng.randint(0, max_y + 1, size=patches_per_image)
        lefts = rng.randint(0, max_x + 1, size=patches_per_image)
        for t, l in zip(tops, lefts):
            pr = img_r[:, t:t + patch_size, l:l + patch_size].unsqueeze(0)
            pf = img_f[:, t:t + patch_size, l:l + patch_size].unsqueeze(0)
            # Use window_size that fits the patch (must be odd, <= patch_size)
            win = min(11, patch_size if patch_size % 2 == 1 else patch_size - 1)
            ssim_map = compute_ssim_map(pr, pf, window_size=win)
            patch_scores.append(float(ssim_map.mean()))

        per_image_scores.append(float(np.mean(patch_scores)))

    return float(np.mean(per_image_scores)), per_image_scores


# ============================================================
# LPIPS computation (paired)
# ============================================================

class _VGGFeatures(nn.Module):
    """VGG16 feature extractor at conv layers 1_2, 2_2, 3_3, 4_3, 5_3."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # Slice indices for relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slice1 = nn.Sequential(*list(vgg[:4]))    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16]))  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23])) # relu4_3
        self.slice5 = nn.Sequential(*list(vgg[23:30])) # relu5_3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


def _normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / norm


def compute_lpips(
    path_real: str,
    path_fake: str,
    device: torch.device,
    image_size: int = 256,
) -> Tuple[float, List[float]]:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between paired images
    using VGG16 features. Lower = more similar.
    Returns (mean_lpips, list of per-image lpips values).
    """
    pairs = list_paired_images(path_real, path_fake)
    vgg = _VGGFeatures().to(device).eval()

    # ImageNet normalization (LPIPS expects [0,1] input, normalized to ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # [0, 1]
    ])

    scores = []
    with torch.no_grad():
        for real_path, fake_path in pairs:
            img_r = tfm(Image.open(real_path).convert("RGB")).unsqueeze(0).to(device)
            img_f = tfm(Image.open(fake_path).convert("RGB")).unsqueeze(0).to(device)

            # Normalize to ImageNet stats
            img_r = (img_r - mean) / std
            img_f = (img_f - mean) / std

            feats_r = vgg(img_r)
            feats_f = vgg(img_f)

            # Cosine distance per layer, spatially averaged, then summed across layers
            dist = 0.0
            for fr, ff in zip(feats_r, feats_f):
                fr = _normalize_tensor(fr)
                ff = _normalize_tensor(ff)
                dist += torch.mean((fr - ff) ** 2, dim=[1, 2, 3]).item()

            scores.append(dist)

    return float(np.mean(scores)), scores


# ============================================================
# Cycle reconstruction error  (A → B' → A')
# ============================================================

def _load_model_for_regen(model_name: str, ckpt_path: str, device: torch.device, style_dim: int = 8):
    """Load a translation model from checkpoint for cycle reconstruction."""
    from models.cyclegan import CycleGAN, CycleGANConfig
    from models.unit import UNIT, UNITConfig
    from models.munit import MUNIT, MUNITConfig
    from models.dclgan import DCLGAN, DCLGANConfig
    from models.uvcgan import UVCGAN, UVCGANConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config")

    if model_name == "cyclegan":
        cfg = CycleGANConfig(**saved_cfg) if saved_cfg else CycleGANConfig()
        model = CycleGAN(cfg)
    elif model_name == "unit":
        cfg = UNITConfig(**saved_cfg) if saved_cfg else UNITConfig()
        model = UNIT(cfg)
    elif model_name == "munit":
        cfg = MUNITConfig(**saved_cfg) if saved_cfg else MUNITConfig(style_dim=style_dim)
        model = MUNIT(cfg)
    elif model_name == "dclgan":
        cfg = DCLGANConfig(**saved_cfg) if saved_cfg else DCLGANConfig()
        model = DCLGAN(cfg)
    elif model_name == "uvcgan":
        cfg = UVCGANConfig(**saved_cfg) if saved_cfg else UVCGANConfig()
        model = UVCGAN(cfg)
    elif model_name == "miudiff":
        raise ValueError(
            "MIUDiff does not support regen_error: it has no B→A generator. "
            "Use a GAN model (cyclegan, unit, munit, dclgan, uvcgan) instead."
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    if saved_cfg:
        print(f"Restored {model_name} config from checkpoint")
    return model


def _cycle_forward(
    model,
    model_name: str,
    x: torch.Tensor,
    direction: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one cycle: A → B' → A'  (or B → A' → B' when direction='B2A').
    Returns (b_prime, a_prime) both in [-1, 1].
    """
    with torch.no_grad():
        if model_name == "cyclegan":
            fwd = model.forward_A2B if direction == "A2B" else model.forward_B2A
            bwd = model.forward_B2A if direction == "A2B" else model.forward_A2B
            b_prime = fwd(x)
            a_prime = bwd(b_prime)

        elif model_name == "unit":
            fwd = model.forward_A2B if direction == "A2B" else model.forward_B2A
            bwd = model.forward_B2A if direction == "A2B" else model.forward_A2B
            b_prime, _ = fwd(x)
            a_prime, _ = bwd(b_prime)

        elif model_name == "munit":
            if direction == "A2B":
                c, _ = model.encode_A(x)
                s = torch.randn(x.shape[0], model.cfg.style_dim, device=device)
                b_prime = model.decode_B(c, s)
                c2, _ = model.encode_B(b_prime)
                s2 = torch.randn(x.shape[0], model.cfg.style_dim, device=device)
                a_prime = model.decode_A(c2, s2)
            else:
                c, _ = model.encode_B(x)
                s = torch.randn(x.shape[0], model.cfg.style_dim, device=device)
                b_prime = model.decode_A(c, s)
                c2, _ = model.encode_A(b_prime)
                s2 = torch.randn(x.shape[0], model.cfg.style_dim, device=device)
                a_prime = model.decode_B(c2, s2)

        elif model_name == "dclgan":
            if direction == "A2B":
                b_prime, _ = model.G_A2B(x)
                a_prime, _ = model.G_B2A(b_prime)
            else:
                b_prime, _ = model.G_B2A(x)
                a_prime, _ = model.G_A2B(b_prime)

        elif model_name == "uvcgan":
            fwd = model.forward_A2B if direction == "A2B" else model.forward_B2A
            bwd = model.forward_B2A if direction == "A2B" else model.forward_A2B
            b_prime = fwd(x)
            a_prime = bwd(b_prime)

        else:
            raise ValueError(f"Unsupported model for cycle forward: {model_name}")

    return b_prime, a_prime


def compute_regen_error(
    path_A: str,
    model,
    model_name: str,
    direction: str,
    device: torch.device,
    image_size: int = 256,
    overlay_dir: Optional[str] = None,
    save_error_npy: bool = False,
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compute per-image cycle reconstruction MAE: A → B' → A', error = |A − A'|.

    MAE values are in [0, 255] pixel scale.
    If overlay_dir is given, saves:
      overlay_dir/heatmaps/<stem>.png  — error heatmap with colorbar (hot colormap)
      overlay_dir/overlays/<stem>.png  — 50/50 blend of heatmap and original A
    If save_error_npy is True (requires overlay_dir), additionally saves:
      overlay_dir/error_npy/<stem>.npy — raw [H, W] error map in [0, 255]

    Returns (mean_mae, [(stem, mae), ...]).
    """
    from torchvision.utils import save_image as tv_save_image

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0,1] → [-1,1]
    ])

    paths = list_images(path_A)

    # --- Pass 1: compute all error maps ---
    stems: List[str] = []
    error_maps: List[np.ndarray] = []   # each [H, W] in [0, 255]
    orig_tensors: List[torch.Tensor] = []  # each [3, H, W] in [-1, 1] on CPU

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        _, a_prime = _cycle_forward(model, model_name, x, direction, device)

        x_255 = (x + 1.0) * 127.5
        a_prime_255 = (a_prime.clamp(-1.0, 1.0) + 1.0) * 127.5
        err = torch.abs(x_255 - a_prime_255).mean(dim=1).squeeze(0)  # [H, W]

        stems.append(os.path.splitext(os.path.basename(p))[0])
        error_maps.append(err.cpu().numpy())
        orig_tensors.append(x.squeeze(0).cpu())

    # --- Global percentile normalization for consistent colormap ---
    all_vals = np.concatenate([e.flatten() for e in error_maps])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))

    # --- Pass 2: save overlays (optional) ---
    if overlay_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        heatmap_dir = os.path.join(overlay_dir, "heatmaps")
        overlay_out_dir = os.path.join(overlay_dir, "overlays")
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(overlay_out_dir, exist_ok=True)

        cmap = cm.get_cmap("hot")

        for stem, err_map, orig_t in zip(stems, error_maps, orig_tensors):
            # Heatmap with colorbar
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(err_map, cmap="hot", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label="MAE [0–255]")
            ax.axis("off")
            ax.set_title(f"Regen error: {stem}", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(heatmap_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Overlay: blend heatmap with original A
            err_norm = np.clip((err_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
            heat_rgb = torch.from_numpy(cmap(err_norm)[:, :, :3]).permute(2, 0, 1).float()  # [3,H,W] in [0,1]
            orig_rgb = (orig_t + 1.0) / 2.0  # [-1,1] → [0,1]
            blend = (0.5 * heat_rgb + 0.5 * orig_rgb).clamp(0.0, 1.0)
            tv_save_image(blend, os.path.join(overlay_out_dir, f"{stem}.png"))

        print(f"[regen_error] Saved heatmaps → {heatmap_dir}")
        print(f"[regen_error] Saved overlays → {overlay_out_dir}")

    if save_error_npy:
        if not overlay_dir:
            raise ValueError("save_error_npy=True requires overlay_dir to be set")
        npy_dir = os.path.join(overlay_dir, "error_npy")
        os.makedirs(npy_dir, exist_ok=True)
        for stem, err_map in zip(stems, error_maps):
            np.save(os.path.join(npy_dir, f"{stem}.npy"), err_map.astype(np.float32))
        print(f"[regen_error] Saved error .npy maps → {npy_dir}")

    results = [(s, float(e.mean())) for s, e in zip(stems, error_maps)]
    mean_mae = float(np.mean([v for _, v in results]))
    return mean_mae, results


# ============================================================
# CSV output
# ============================================================

def _save_csv(path: str, fieldnames: List[str], rows: List[dict]):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser("Evaluation metrics for image-to-image translation")
    ap.add_argument("--metric", choices=["fid", "ssim", "patch_ssim", "lpips", "regen_error"], default="fid",
                    help="Metric to compute: fid (unpaired), ssim/patch_ssim/lpips (paired), "
                         "regen_error (cycle A→B'→A', requires --path_A --model --ckpt)")
    ap.add_argument("--path_real", required=True, type=str, help="Folder with real target-domain images")
    ap.add_argument("--path_fake", required=True, type=str, help="Folder with generated images")
    ap.add_argument("--backend", choices=["inception", "dino"], default="inception",
                    help="Feature backend for FID")

    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    # DINO options
    ap.add_argument("--dino_model", default="dinov2_vits14", type=str)
    ap.add_argument("--dino_image_size", default=224, type=int)

    # SSIM / patch SSIM options
    ap.add_argument("--ssim_image_size", default=256, type=int,
                    help="Resize images to this size before computing SSIM")
    ap.add_argument("--patch_size", default=64, type=int,
                    help="Patch size for patch_ssim")
    ap.add_argument("--patches_per_image", default=16, type=int,
                    help="Number of random patches per image for patch_ssim")

    # regen_error options
    ap.add_argument("--path_A", type=str, default=None,
                    help="Folder of source-domain A images (required for regen_error)")
    ap.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "uvcgan"], default=None,
                    help="Model type (required for regen_error)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Model checkpoint path (required for regen_error)")
    ap.add_argument("--direction", choices=["A2B", "B2A"], default="A2B",
                    help="Translation direction for regen_error cycle (default: A2B)")
    ap.add_argument("--overlay_dir", type=str, default=None,
                    help="Directory to save error heatmaps and overlays (regen_error only)")
    ap.add_argument("--save_error_npy", action="store_true",
                    help="Save raw per-pixel error maps as .npy under "
                         "<overlay_dir>/error_npy/ (regen_error only; requires --overlay_dir)")
    ap.add_argument("--style_dim", type=int, default=8,
                    help="Style dimension for MUNIT (regen_error only, ignored if config in checkpoint)")

    # Output
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Save results to a CSV file (summary + per-image scores when available)")

    args = ap.parse_args()

    if args.metric == "fid":
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

        if args.save_csv:
            _save_csv(args.save_csv, ["metric", "value", "backend", "n_real", "n_fake"],
                      [{"metric": label, "value": f"{fid_like:.6f}", "backend": args.backend,
                        "n_real": acts_real.shape[0], "n_fake": acts_fake.shape[0]}])

    elif args.metric == "ssim":
        mean_ssim, per_image = compute_ssim(args.path_real, args.path_fake, image_size=args.ssim_image_size)
        pairs = list_paired_images(args.path_real, args.path_fake)
        print(f"SSIM (real={args.path_real} vs fake={args.path_fake}): {mean_ssim:.4f}")
        print(f"N_pairs={len(per_image)}, image_size={args.ssim_image_size}")

        if args.save_csv:
            rows = [{"filename": os.path.basename(p[0]), "ssim": f"{s:.6f}"} for p, s in zip(pairs, per_image)]
            rows.append({"filename": "MEAN", "ssim": f"{mean_ssim:.6f}"})
            _save_csv(args.save_csv, ["filename", "ssim"], rows)

    elif args.metric == "patch_ssim":
        mean_pssim, per_image = compute_patch_ssim(
            args.path_real, args.path_fake,
            image_size=args.ssim_image_size,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
        )
        pairs = list_paired_images(args.path_real, args.path_fake)
        print(f"Patch-SSIM (real={args.path_real} vs fake={args.path_fake}): {mean_pssim:.4f}")
        print(f"N_pairs={len(per_image)}, patch_size={args.patch_size}, patches_per_image={args.patches_per_image}")

        if args.save_csv:
            rows = [{"filename": os.path.basename(p[0]), "patch_ssim": f"{s:.6f}"} for p, s in zip(pairs, per_image)]
            rows.append({"filename": "MEAN", "patch_ssim": f"{mean_pssim:.6f}"})
            _save_csv(args.save_csv, ["filename", "patch_ssim"], rows)

    elif args.metric == "lpips":
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        mean_lpips, per_image = compute_lpips(
            args.path_real, args.path_fake, device, image_size=args.ssim_image_size,
        )
        pairs = list_paired_images(args.path_real, args.path_fake)
        print(f"LPIPS (real={args.path_real} vs fake={args.path_fake}): {mean_lpips:.4f}")
        print(f"N_pairs={len(per_image)}, image_size={args.ssim_image_size}")

        if args.save_csv:
            rows = [{"filename": os.path.basename(p[0]), "lpips": f"{s:.6f}"} for p, s in zip(pairs, per_image)]
            rows.append({"filename": "MEAN", "lpips": f"{mean_lpips:.6f}"})
            _save_csv(args.save_csv, ["filename", "lpips"], rows)

    elif args.metric == "regen_error":
        for flag, name in [(args.path_A, "--path_A"), (args.model, "--model"), (args.ckpt, "--ckpt")]:
            if flag is None:
                ap.error(f"regen_error requires {name}")
        if args.save_error_npy and not args.overlay_dir:
            ap.error("--save_error_npy requires --overlay_dir to be set")

        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        model = _load_model_for_regen(args.model, args.ckpt, device, style_dim=args.style_dim)

        mean_mae, per_image = compute_regen_error(
            path_A=args.path_A,
            model=model,
            model_name=args.model,
            direction=args.direction,
            device=device,
            image_size=args.ssim_image_size,
            overlay_dir=args.overlay_dir,
            save_error_npy=args.save_error_npy,
        )
        print(f"Regen Error MAE ({args.direction} cycle, path={args.path_A}): {mean_mae:.4f}")
        print(f"N_images={len(per_image)}, image_size={args.ssim_image_size}")

        if args.save_csv:
            rows = [{"filename": f"{s}.tif", "regen_mae": f"{v:.6f}"} for s, v in per_image]
            rows.append({"filename": "MEAN", "regen_mae": f"{mean_mae:.6f}"})
            _save_csv(args.save_csv, ["filename", "regen_mae"], rows)


if __name__ == "__main__":
    main()
