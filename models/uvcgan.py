# uvcgan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from base_models import GANLoss, NLayerDiscriminator, ImagePool
from base_models import discriminator_loss, identity_loss


# ============================================================
# Config
# ============================================================

@dataclass
class UVCGANConfig:
    # generator
    ngf: int = 64
    n_down: int = 4
    vit_features: int = 192
    vit_n_heads: int = 6
    vit_n_blocks: int = 6
    vit_rezero: bool = True

    # discriminator
    ndf: int = 64
    n_layers_D: int = 3

    # losses
    gan_mode: str = "lsgan"
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    pool_size: int = 50

    # pre-training
    pretrain: bool = False
    mask_patch_size: int = 32
    n_masks: int = 4

    # expected spatial input size (controls ViT pos_embed dimensions)
    image_size: int = 256


# ============================================================
# Transformer components
# ============================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional ReZero residual scaling."""

    def __init__(self, dim: int, n_heads: int, rezero: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        if rezero:
            self.alpha1 = nn.Parameter(torch.zeros(1))
            self.alpha2 = nn.Parameter(torch.zeros(1))
        else:
            self.alpha1 = None
            self.alpha2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + (h * self.alpha1 if self.alpha1 is not None else h)

        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + (h * self.alpha2 if self.alpha2 is not None else h)
        return x


class ViTBottleneck(nn.Module):
    """Treats spatial feature maps as token sequences for transformer processing."""

    def __init__(self, in_channels: int, spatial_size: int, vit_features: int,
                 n_heads: int, n_blocks: int, rezero: bool = True):
        super().__init__()
        self.spatial_size = spatial_size
        n_tokens = spatial_size * spatial_size

        self.proj_in = nn.Linear(in_channels, vit_features)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, vit_features) * 0.02)
        self.blocks = nn.Sequential(
            *[TransformerBlock(vit_features, n_heads, rezero) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(vit_features)
        self.proj_out = nn.Linear(vit_features, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.proj_in(tokens) + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        tokens = self.proj_out(tokens)
        # (B, H*W, C) -> (B, C, H, W)
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ============================================================
# UNet Encoder / Decoder
# ============================================================

class UNetEncoder(nn.Module):
    """Progressive downsampling encoder returning skip features."""

    def __init__(self, input_nc: int = 3, ngf: int = 64, n_down: int = 4):
        super().__init__()
        # Stem: ReflectionPad + Conv(3->ngf, k=7) + IN + ReLU
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )

        # Down blocks
        self.downs = nn.ModuleList()
        ch = ngf
        for i in range(n_down):
            ch_out = min(ch * 2, ngf * 8)
            self.downs.append(nn.Sequential(
                nn.Conv2d(ch, ch_out, 3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(True),
            ))
            ch = ch_out

        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        h = self.stem(x)
        skips.append(h)  # skip 0: after stem
        for down in self.downs[:-1]:
            h = down(h)
            skips.append(h)
        # Last down — no skip (goes to ViT)
        h = self.downs[-1](h)
        return h, skips


class UNetDecoder(nn.Module):
    """Upsampling decoder with skip concatenation."""

    def __init__(self, output_nc: int = 3, ngf: int = 64, n_down: int = 4):
        super().__init__()
        # Build channel list matching encoder (reverse order for decoder)
        enc_channels = [ngf]  # stem output
        ch = ngf
        for i in range(n_down):
            ch = min(ch * 2, ngf * 8)
            enc_channels.append(ch)
        # enc_channels: [64, 128, 256, 512, 512] for ngf=64, n_down=4

        self.upsample = nn.ModuleList()
        self.merge = nn.ModuleList()
        # We go from bottleneck backwards through skips
        # Bottleneck channels = enc_channels[-1]
        in_ch = enc_channels[-1]
        for i in range(n_down):
            skip_ch = enc_channels[-(i + 2)]  # matching skip
            out_ch = skip_ch
            # Step 1: upsample h
            self.upsample.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(in_ch),
                nn.ReLU(True),
            ))
            # Step 2: conv to merge concatenated (upsampled h + skip)
            self.merge.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=True),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True),
            ))
            in_ch = out_ch

        # Final output head
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        h = x
        for i in range(len(self.upsample)):
            h = self.upsample[i](h)              # upsample to match skip spatial size
            skip = skips[-(i + 1)]                # reverse order
            h = torch.cat([h, skip], dim=1)       # concat along channels
            h = self.merge[i](h)                  # conv to merge
        return self.head(h)


# ============================================================
# Generator
# ============================================================

class UNetViTGenerator(nn.Module):
    """UNet encoder -> ViT bottleneck -> UNet decoder."""

    def __init__(self, cfg: UVCGANConfig, image_size: int = 256):
        super().__init__()
        self.encoder = UNetEncoder(input_nc=3, ngf=cfg.ngf, n_down=cfg.n_down)
        spatial = image_size // (2 ** cfg.n_down)
        self.vit = ViTBottleneck(
            in_channels=self.encoder.out_channels,
            spatial_size=spatial,
            vit_features=cfg.vit_features,
            n_heads=cfg.vit_n_heads,
            n_blocks=cfg.vit_n_blocks,
            rezero=cfg.vit_rezero,
        )
        self.decoder = UNetDecoder(output_nc=3, ngf=cfg.ngf, n_down=cfg.n_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self.encoder(x)
        z = self.vit(z)
        return self.decoder(z, skips)


# ============================================================
# UVCGAN Model
# ============================================================

class UVCGAN(nn.Module):
    """
    UVCGAN: UNet-ViT cycle-consistent GAN.

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()
      - compute_generator_loss(batch) -> (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) -> (loss, logs)
    """

    def __init__(self, cfg: UVCGANConfig):
        super().__init__()
        self.cfg = cfg

        # Generators
        self.G_A2B = UNetViTGenerator(cfg, image_size=cfg.image_size)
        self.G_B2A = UNetViTGenerator(cfg, image_size=cfg.image_size)

        # Discriminators (not used during pretrain)
        self.D_A = NLayerDiscriminator(3, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(3, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # Buffers & losses
        self.pool_A = ImagePool(cfg.pool_size)
        self.pool_B = ImagePool(cfg.pool_size)
        self.gan = GANLoss(cfg.gan_mode)
        self.l1 = nn.L1Loss()

    # ---------------- BaseTrainer interface ----------------

    def generator_parameters(self):
        return list(self.G_A2B.parameters()) + list(self.G_B2A.parameters())

    def discriminator_parameters(self):
        if self.cfg.pretrain:
            return []
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    # ---------------- Forward helpers ----------------

    def forward_A2B(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_A2B(x)

    def forward_B2A(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_B2A(x)

    # ---------------- Masking (pretrain) ----------------

    def _create_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create a binary mask: 1 = masked (to reconstruct), 0 = visible."""
        B, C, H, W = x.shape
        mask = torch.zeros(B, 1, H, W, device=x.device)
        ps = self.cfg.mask_patch_size
        for _ in range(self.cfg.n_masks):
            top = torch.randint(0, H - ps + 1, (B,))
            left = torch.randint(0, W - ps + 1, (B,))
            for b in range(B):
                mask[b, :, top[b]:top[b] + ps, left[b]:left[b] + ps] = 1.0
        return mask

    # ---------------- Losses ----------------

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        if self.cfg.pretrain:
            return self._compute_pretrain_loss(batch)
        return self._compute_cycle_loss(batch)

    def _compute_pretrain_loss(self, batch: Dict[str, torch.Tensor]):
        real_A = batch["A"]
        real_B = batch["B"]

        # Mask and reconstruct
        mask_A = self._create_mask(real_A)
        mask_B = self._create_mask(real_B)

        masked_A = real_A * (1 - mask_A)
        masked_B = real_B * (1 - mask_B)

        rec_A = self.G_B2A(masked_A)  # G_B2A pretrained on domain A
        rec_B = self.G_A2B(masked_B)  # G_A2B pretrained on domain B

        # L1 loss on masked regions only
        loss_A = self.l1(rec_A * mask_A, real_A * mask_A)
        loss_B = self.l1(rec_B * mask_B, real_B * mask_B)
        loss_G = loss_A + loss_B

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_pretrain_A": float(loss_A.detach().cpu()),
            "loss_pretrain_B": float(loss_B.detach().cpu()),
        }
        visuals = {
            "real_A": real_A,
            "rec_A": rec_A,
            "real_B": real_B,
            "rec_B": rec_B,
        }
        return loss_G, logs, visuals

    def _compute_cycle_loss(self, batch: Dict[str, torch.Tensor]):
        real_A = batch["A"]
        real_B = batch["B"]

        # Translate
        fake_B = self.forward_A2B(real_A)
        fake_A = self.forward_B2A(real_B)

        # Cycle
        rec_A = self.forward_B2A(fake_B)
        rec_B = self.forward_A2B(fake_A)

        # Identity (optional)
        loss_idt = identity_loss(self.l1, self.forward_A2B, self.forward_B2A,
                                 real_A, real_B, self.cfg.lambda_identity)

        # GAN
        loss_gan = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)

        # Cycle
        loss_cycle = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)

        loss_G = loss_gan + self.cfg.lambda_cycle * loss_cycle + self.cfg.lambda_identity * loss_idt

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_gan": float(loss_gan.detach().cpu()),
            "loss_cycle": float(loss_cycle.detach().cpu()),
            "loss_idt": float(loss_idt.detach().cpu()),
        }
        visuals = {
            "real_A": real_A,
            "fake_B": fake_B,
            "rec_A": rec_A,
            "real_B": real_B,
            "fake_A": fake_A,
            "rec_B": rec_B,
        }
        return loss_G, logs, visuals

    def compute_discriminator_loss(self, batch: Dict[str, torch.Tensor], visuals: Dict[str, torch.Tensor]):
        real_A, real_B = batch["A"], batch["B"]
        fake_A = self.pool_A.query(visuals["fake_A"].detach())
        fake_B = self.pool_B.query(visuals["fake_B"].detach())
        return discriminator_loss(self.gan, self.D_A, self.D_B, real_A, real_B, fake_A, fake_B)
