# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Init helpers
# ============================================================

def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights."""
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(f"init method {init_type} not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net: nn.Module, device: torch.device, init_type: str = "normal", init_gain: float = 0.02) -> nn.Module:
    net.to(device)
    init_weights(net, init_type=init_type, init_gain=init_gain)
    return net


# ============================================================
# Replay buffer (used by CycleGAN; also handy for others)
# ============================================================

class ImagePool:
    """History of generated images to stabilize discriminator training."""
    def __init__(self, pool_size: int = 50):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: tensor [B, C, H, W]
        returns: tensor [B, C, H, W]
        """
        if self.pool_size <= 0:
            return images

        out: List[torch.Tensor] = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(img)
                out.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = img
                    out.append(old)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


# ============================================================
# Losses
# ============================================================

class GANLoss(nn.Module):
    """
    LSGAN by default (MSE to 1/0). If you want vanilla GAN, swap to BCEWithLogitsLoss.
    """
    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("mode must be 'lsgan' or 'vanilla'")

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.mode == "lsgan":
            return torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        # vanilla:
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        tgt = self._target(pred, is_real)
        return self.loss(pred, tgt)



# ============================================================
# Building blocks
# ============================================================

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: nn.Module = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        p = 0
        if padding_type == "reflect":
            pad = nn.ReflectionPad2d(1)
        elif padding_type == "replicate":
            pad = nn.ReplicationPad2d(1)
        elif padding_type == "zero":
            pad = nn.Identity()
            p = 1
        else:
            raise NotImplementedError(padding_type)

        layers: List[nn.Module] = [
            pad,
            nn.Conv2d(dim, dim, 3, padding=p, bias=True),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [
            pad,
            nn.Conv2d(dim, dim, 3, padding=p, bias=True),
            norm_layer(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):
    """
    Encoder trunk (CycleGAN-style) WITHOUT ResBlocks.
    Typical for 256x256 input:
      - stem (k7)
      - 2 downsamples (stride 2)
    Output: [B, ngf*4, H/4, W/4]
    """
    def __init__(
        self,
        input_nc: int,
        ngf: int = 64,
        n_down: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_layers = feature_layers if feature_layers is not None else []

        layers: List[nn.Module] = []
        # Stem
        layers += [
            nn.ReflectionPad2d(3),                              # idx 0
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=True),  # idx 1
            norm_layer(ngf),                                    # idx 2
            nn.ReLU(True),                                      # idx 3
        ]

        # Downsamples
        mult = 1
        for _ in range(n_down):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        self.layers = nn.ModuleList(layers)
        self.out_channels = ngf * mult
        self.out_stride = 2 ** n_down  # spatial reduction factor

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.return_features and (i in self.feature_layers):
                feats[f"enc_layer_{i}"] = h
        return (h, feats) if self.return_features else h


# ============================================================
# Bottle neck
# ============================================================

class ResnetBottleneck(nn.Module):
    """
    The ONLY module containing ResnetBlocks.
    Keeps shape: [B, C, H, W] -> [B, C, H, W]
    This is what you can share across domains for UNIT.
    """
    def __init__(
        self,
        channels: int,
        n_blocks: int = 9,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_blocks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_blocks = feature_blocks if feature_blocks is not None else []

        self.blocks = nn.ModuleList(
            [ResnetBlock(channels, padding_type="reflect", norm_layer=norm_layer) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.return_features and (i in self.feature_blocks):
                feats[f"bottleneck_block_{i}"] = h
        return (h, feats) if self.return_features else h


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    """
    Decoder trunk (CycleGAN-style) WITHOUT ResBlocks.
    Typical:
      - 2 upsample (ConvTranspose)
      - output head (k7 + tanh)
    """
    def __init__(
        self,
        in_channels: int,
        output_nc: int,
        ngf: int = 64,
        n_up: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_layers = feature_layers if feature_layers is not None else []

        # infer "mult" assuming classic pattern: in_channels == ngf * (2**n_up)
        # e.g. ngf=64, n_up=2 => in_channels should be 256
        mult = in_channels // ngf
        if mult < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= ngf ({ngf})")

        layers: List[nn.Module] = []

        # Upsamples
        for _ in range(n_up):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2

        # Output head
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0, bias=True),
            nn.Tanh(),
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.return_features and (i in self.feature_layers):
                feats[f"dec_layer_{i}"] = h
        return (h, feats) if self.return_features else h


class Decoder3Head(nn.Module):
    """Decoder with three output heads for UGAC (Upadhyay et al., NeurIPS 2021).

    Shares the upsampling trunk with `Decoder`, then splits into three heads that
    parameterise a zero-mean generalized Gaussian over the per-pixel residual:

      mu        the image itself          (tanh, matches `Decoder`)
      inv_alpha 1 / scale                 (softplus + floor)
      beta      shape                     (softplus + floor, clamped)

    Following the paper the network predicts 1/alpha rather than alpha for
    numerical stability. Positivity is enforced with softplus rather than the
    paper's ReLU: ReLU can emit exactly zero, which makes log(inv_alpha) and the
    1/beta in lgamma diverge.

    With (alpha, beta) = (1, 1) the GGD NLL reduces to the L1 cycle loss, so a
    UGAC model is a strict generalisation of the vanilla one.
    """

    def __init__(
        self,
        in_channels: int,
        output_nc: int,
        ngf: int = 64,
        n_up: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        min_inv_alpha: float = 1e-2,
        min_beta: float = 0.2,
        max_beta: float = 4.0,
    ):
        super().__init__()
        self.min_inv_alpha = min_inv_alpha
        self.min_beta = min_beta
        self.max_beta = max_beta

        mult = in_channels // ngf
        if mult < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= ngf ({ngf})")

        trunk: List[nn.Module] = []
        for _ in range(n_up):
            trunk += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2
        self.trunk = nn.Sequential(*trunk)

        def _head(out_nc: int) -> nn.Module:
            return nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, out_nc, 7, padding=0, bias=True),
            )

        # mu keeps the vanilla head exactly (conv + tanh)
        self.head_mu = _head(output_nc)
        # alpha / beta are single-channel: one scale and one shape per pixel
        self.head_inv_alpha = _head(1)
        self.head_beta = _head(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = torch.tanh(self.head_mu(h))
        inv_alpha = F.softplus(self.head_inv_alpha(h)) + self.min_inv_alpha
        beta = F.softplus(self.head_beta(h)) + self.min_beta
        beta = beta.clamp(max=self.max_beta)
        return mu, inv_alpha, beta


def ggd_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    inv_alpha: torch.Tensor,
    beta: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Generalized Gaussian negative log-likelihood — UGAC Eq. 8.

        L = mean[ (|pred - target| / alpha)^beta - log(beta / alpha) + lgamma(1/beta) ]

    Written in terms of the predicted `inv_alpha` = 1/alpha:

        (|r| * inv_alpha)^beta - log(beta) - log(inv_alpha) + lgamma(1/beta)

    `inv_alpha` and `beta` are [N,1,H,W] and broadcast over the residual's colour
    channels. `mask` (if given) restricts the mean to selected pixels, which
    UVCGAN's masked cycle loss needs.

    The power term is evaluated in log space and clamped: |r| can reach 2.0 on
    [-1,1] images and beta up to 4, so a naive pow overflows in fp16.
    """
    r = (pred - target).abs()
    log_term = torch.log(r * inv_alpha + eps)
    power = torch.exp((beta * log_term).clamp(max=20.0))

    nll = power - torch.log(beta) - torch.log(inv_alpha) + torch.lgamma(1.0 / beta)

    if mask is not None:
        mask = mask.expand_as(nll)
        denom = mask.sum().clamp(min=1.0)
        return (nll * mask).sum() / denom
    return nll.mean()


def ggd_aleatoric_var(inv_alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Closed-form aleatoric variance of a generalized Gaussian (UGAC Sec. 3.2).

        sigma^2 = alpha^2 * Gamma(3/beta) / Gamma(1/beta)

    Needs no sampling — one forward pass gives the full map. Returns the same
    shape as `inv_alpha`, i.e. [N,1,H,W].
    """
    alpha_sq = (1.0 / inv_alpha) ** 2
    ratio = torch.exp(torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    return alpha_sq * ratio


# ============================================================
# Resnet Generator (Encoder + Bottleneck + Decoder)
# ============================================================

class ResnetGenerator(nn.Module):
    """
    Convenience wrapper to reproduce the old behavior, but with explicit modules:
      y = dec(bottleneck(enc(x)))

    This keeps your CycleGAN code simple while letting UNIT/MUNIT reuse enc/dec.
    """
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        n_down: int = 2,
        n_blocks: int = 9,
        n_up: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ):
        super().__init__()
        self.enc = Encoder(input_nc, ngf=ngf, n_down=n_down, norm_layer=norm_layer)
        self.bottleneck = ResnetBottleneck(self.enc.out_channels, n_blocks=n_blocks, norm_layer=norm_layer)
        self.dec = Decoder(self.enc.out_channels, output_nc, ngf=ngf, n_up=n_up, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.bottleneck(z)
        y = self.dec(z)
        return y



class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (no sigmoid by default)."""
    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ):
        super().__init__()
        kw = 4
        padw = 1
        seq: List[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            seq += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        seq += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        seq += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# Small image helpers
# ============================================================

def denorm01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x + 1.0) / 2.0


# ============================================================
# Patch sampling & contrastive utilities
# ============================================================

def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss with one-to-one positives by index."""
    q = F.normalize(q.float(), dim=1)
    k = F.normalize(k.float(), dim=1)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


class PatchSampler:
    """Sample random spatial patches from [B,C,H,W] feature maps."""

    @staticmethod
    def sample(feat: torch.Tensor, n_patches: int, patch_ids: torch.Tensor | None = None):
        """
        Returns:
          vecs: [B*K, C] gathered vectors
          ids:  [B, K] linear indices into H*W
        """
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        K = min(n_patches, H * W)
        if patch_ids is None:
            ids = torch.randint(0, H * W, (B, K), device=feat.device)
        else:
            ids = patch_ids
        gathered = torch.gather(flat, 1, ids.unsqueeze(-1).expand(-1, -1, C))
        return gathered.reshape(-1, C), ids


# ============================================================
# Shared GAN loss utilities
# ============================================================

def discriminator_loss(gan, D_A, D_B, real_A, real_B, fake_A, fake_B):
    """Standard symmetric PatchGAN discriminator loss.

    fake_A / fake_B must already be detached (and optionally pool-queried) by the caller.
    Returns (loss_D tensor, logs dict).
    """
    loss_D_A = 0.5 * (gan(D_A(real_A), True) + gan(D_A(fake_A), False))
    loss_D_B = 0.5 * (gan(D_B(real_B), True) + gan(D_B(fake_B), False))
    loss_D = loss_D_A + loss_D_B
    return loss_D, {
        "loss_D":   float(loss_D.detach().cpu()),
        "loss_D_A": float(loss_D_A.detach().cpu()),
        "loss_D_B": float(loss_D_B.detach().cpu()),
    }


def identity_loss(l1, forward_A2B, forward_B2A, real_A, real_B, lam):
    """Optional CycleGAN-style identity regularisation loss.

    Returns a zero tensor when lam <= 0.
    forward_A2B / forward_B2A must return plain tensors (not tuples).
    """
    if lam <= 0:
        return torch.tensor(0.0, device=real_A.device)
    idt_A = forward_B2A(real_A)
    idt_B = forward_A2B(real_B)
    return 0.5 * (l1(idt_A, real_A) + l1(idt_B, real_B))


# ============================================================
# Diffusion components (shared by CycleDiffusion)
# ============================================================

# =========================
# DDPM schedule
# =========================

@dataclass
class DiffusionSchedule:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def make(self, device):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
        return betas, alphas, alphas_cumprod, alphas_cumprod_prev


# ============================================================
# DDPM / guided-diffusion style UNet (2D)
# - ResBlocks with time embedding
# - Attention at selected resolutions
# - Down/Up sampling with conv
# ============================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    t: [B] float in [0,1] (we will scale to [0, max_period] internally)
    returns: [B, dim]
    """
    # Scale continuous t into "timesteps" space
    # (works fine for both discrete or continuous time)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = (t.float().unsqueeze(1) * max_period) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _gn_groups(n_channels: int) -> int:
    """Return the largest of {32, 16, 8, 4} that evenly divides n_channels."""
    for g in (32, 16, 8, 4):
        if n_channels % g == 0:
            return g
    return 1


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ZeroModule(nn.Module):
    """Wraps a module and initializes its weights to zero (stable residual starts)."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        return self.module(x)


class AttentionBlock(nn.Module):
    """
    Self-attention over spatial positions for [B, C, H, W].
    This is the classic diffusion attention block (single-head or multi-head).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = GroupNorm32(_gn_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = ZeroModule(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # [B,C,HW]
        qkv = self.qkv(h)                   # [B,3C,HW]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # reshape heads: [B, heads, C//heads, HW]
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale  # [B,heads,HW,HW]
        attn = attn.float().softmax(dim=-1).to(q.dtype)        # fp32 softmax for stability
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)               # [B,heads,head_dim,HW]
        out = out.reshape(B, C, H * W)                               # [B,C,HW]
        out = self.proj_out(out).view(B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    """
    WideResNet-style ResBlock with time embedding conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        gn_groups_in = _gn_groups(in_channels)
        gn_groups_out = _gn_groups(out_channels)

        self.norm1 = GroupNorm32(gn_groups_in, in_channels)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = GroupNorm32(gn_groups_out, out_channels)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ZeroModule(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            if use_conv_shortcut:
                self.skip = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, 1)

        # Output norm: normalises the block output before it enters the next block.
        # Prevents time_proj additions from compounding across the ~19 ResBlocks in
        # the UNet and eventually overflowing fp32 during long training runs.
        self.norm_out = GroupNorm32(gn_groups_out, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Pre-norm on skip path: normalise x once, feed to both branches so the
        # skip contribution is also bounded.
        x_n = self.norm1(x)
        h = self.conv1(self.act1(x_n))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return self.norm_out(self.skip(x_n) + h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


@dataclass
class UNetConfig:
    in_channels: int
    out_channels: int = 3

    # “model_channels” in DDPM/OpenAI code
    base_channels: int = 64

    # for 256x256, DDPM uses 6 resolutions (256→128→64→32→16→8)
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)

    num_res_blocks: int = 2
    dropout: float = 0.0

    # attention at given downsample rates (1 means 256x256, 2 means 128x128, 16 means 16x16)
    attention_resolutions: Tuple[int, ...] = (16,)
    num_heads: int = 4

    time_emb_mult: int = 4  # time embedding dim = base_channels * time_emb_mult


class DDPMUNet(nn.Module):
    """
    DDPM-style UNet backbone (Ho et al. / OpenAI guided-diffusion).
    Forward signature matches your eps-model: eps_theta(x, t_frac).

    - x: [B, in_channels, H, W]
    - t_frac: [B] float in [0,1]
    returns: [B, out_channels, H, W]
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels

        time_dim = cfg.base_channels * cfg.time_emb_mult
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.base_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        ch = cfg.base_channels
        ds = 1  # downsample rate relative to input
        for level, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch, num_heads=cfg.num_heads))
                self.skip_channels.append(ch)

            # downsample except last level
            if level != len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                ds *= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim, dropout=cfg.dropout),
            AttentionBlock(ch, num_heads=cfg.num_heads) if (ds in cfg.attention_resolutions) else nn.Identity(),
            ResBlock(ch, ch, time_dim, dropout=cfg.dropout),
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch, num_heads=cfg.num_heads))

            if level != 0:
                self.upsamples.append(Upsample(ch))
                ds //= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = GroupNorm32(_gn_groups(ch), ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        # Run entirely in fp32 — partial fp16 fixes are insufficient because any large
        # intermediate value cast back to fp16 produces inf and corrupts all downstream blocks
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(x.float(), t_frac)

    def _forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t_frac, self.cfg.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)
        hs: List[torch.Tensor] = []

        # Encoder
        di = 0
        for level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[di](h, t_emb); di += 1
                # optional attention block right after resblock
                if di < len(self.down_blocks) and isinstance(self.down_blocks[di], AttentionBlock):
                    h = self.down_blocks[di](h); di += 1
                hs.append(h)
            h = self.downsamples[level](h)

        # Middle
        h = self.mid[0](h, t_emb)
        h = self.mid[1](h) if not isinstance(self.mid[1], nn.Identity) else h
        h = self.mid[2](h, t_emb)

        # Decoder
        ui = 0
        for level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[ui](h, t_emb); ui += 1
                if ui < len(self.up_blocks) and isinstance(self.up_blocks[ui], AttentionBlock):
                    h = self.up_blocks[ui](h); ui += 1
            h = self.upsamples[level](h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
