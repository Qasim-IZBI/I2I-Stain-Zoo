# munit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GANLoss, NLayerDiscriminator
from models import Encoder, Decoder, ResnetBottleneck


# ============================================================
# Config
# ============================================================

@dataclass
class MUNITConfig:
    input_nc: int = 3
    output_nc: int = 3

    ngf: int = 64
    n_down: int = 2
    n_up: int = 2

    # Content bottleneck (ResBlocks-only, domain specific)
    n_content_blocks: int = 4

    # Style
    style_dim: int = 8

    # Decoder style injection (AdaIN ResBlocks)
    n_adain_blocks: int = 4
    mlp_dim: int = 256  # hidden dim for style MLP

    # Losses
    gan_mode: str = "lsgan"
    lambda_gan: float = 1.0
    lambda_recon_img: float = 10.0
    lambda_recon_content: float = 1.0
    lambda_recon_style: float = 1.0

    # Discriminator
    ndf: int = 64
    n_layers_D: int = 3


# ============================================================
# AdaIN blocks (style injection)
# ============================================================

def adain(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: [B,C,H,W]
    gamma,beta: [B,C]
    """
    B, C, _, _ = x.shape
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)
    return gamma * x_hat + beta


class AdaINResBlock(nn.Module):
    """
    Residual block where both convs are followed by AdaIN + ReLU (first) and AdaIN (second).
    Style params are provided externally by an MLP.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=0, bias=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=0, bias=True)

    def forward(self, x: torch.Tensor, gamma1: torch.Tensor, beta1: torch.Tensor, gamma2: torch.Tensor, beta2: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.pad1(x))
        h = adain(h, gamma1, beta1)
        h = F.relu(h, inplace=True)

        h = self.conv2(self.pad2(h))
        h = adain(h, gamma2, beta2)
        return x + h


class StyleMLP(nn.Module):
    """
    Maps style code s:[B,style_dim] -> AdaIN params for n blocks.
    For each block we need (gamma1,beta1,gamma2,beta2), each [B,C].
    Total params per block = 4*C.
    """
    def __init__(self, style_dim: int, channels: int, n_blocks: int, hidden_dim: int = 256):
        super().__init__()
        self.style_dim = style_dim
        self.channels = channels
        self.n_blocks = n_blocks
        self.out_dim = n_blocks * 4 * channels

        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # [B, n_blocks*4*C]


class AdaINBottleneck(nn.Module):
    """
    Stack of AdaINResBlocks. This is where style is injected.
    """
    def __init__(self, channels: int, style_dim: int, n_blocks: int, mlp_dim: int):
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([AdaINResBlock(channels) for _ in range(n_blocks)])
        self.mlp = StyleMLP(style_dim, channels, n_blocks, hidden_dim=mlp_dim)

    def forward(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        params = self.mlp(s)  # [B, n_blocks*4*C]
        B = c.size(0)
        C = self.channels

        # reshape into [B, n_blocks, 4, C]
        params = params.view(B, self.n_blocks, 4, C)

        h = c
        for i, blk in enumerate(self.blocks):
            gamma1 = params[:, i, 0, :]
            beta1  = params[:, i, 1, :]
            gamma2 = params[:, i, 2, :]
            beta2  = params[:, i, 3, :]
            h = blk(h, gamma1, beta1, gamma2, beta2)
        return h


# ============================================================
# Style encoder (domain-specific)
# ============================================================

class StyleEncoder(nn.Module):
    """
    Simple style encoder:
      Conv stride-2 stack -> global pooling -> linear to style_dim
    """
    def __init__(self, in_nc: int = 3, style_dim: int = 8, base: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, base, 7, padding=3), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base * 4, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).view(x.size(0), -1)
        return self.fc(h)


# ============================================================
# MUNIT model (BaseTrainer-compatible)
# ============================================================

class MUNIT(nn.Module):
    """
    Proper MUNIT-style model using:
      - ResnetEncoder (no ResBlocks) for content trunk
      - ResnetBottleneck (ResBlocks-only) for content refinement
      - AdaINBottleneck for style injection
      - ResnetDecoder (no ResBlocks) for upsampling/output head

    Domain A:
      cA = bottleneck_A(enc_A(xA))
      sA = styleEnc_A(xA)

    Translate A->B:
      xA2B = dec_B( adain_B(cA, sB_rand or sB) )
    """
    def __init__(self, cfg: MUNITConfig):
        super().__init__()
        self.cfg = cfg

        # ---------- Content encoders (no ResBlocks) ----------
        self.Ec_A = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)
        self.Ec_B = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)

        Cc = self.Ec_A.out_channels  # typically 256 for ngf=64, n_down=2

        # ---------- Content bottlenecks (ResBlocks-only; domain-specific) ----------
        self.Bn_A = ResnetBottleneck(Cc, n_blocks=cfg.n_content_blocks)
        self.Bn_B = ResnetBottleneck(Cc, n_blocks=cfg.n_content_blocks)

        # ---------- Style encoders ----------
        self.Es_A = StyleEncoder(cfg.input_nc, style_dim=cfg.style_dim, base=cfg.ngf)
        self.Es_B = StyleEncoder(cfg.input_nc, style_dim=cfg.style_dim, base=cfg.ngf)

        # ---------- AdaIN bottlenecks (style injection; domain-specific) ----------
        self.AdaIN_A = AdaINBottleneck(Cc, cfg.style_dim, n_blocks=cfg.n_adain_blocks, mlp_dim=cfg.mlp_dim)
        self.AdaIN_B = AdaINBottleneck(Cc, cfg.style_dim, n_blocks=cfg.n_adain_blocks, mlp_dim=cfg.mlp_dim)

        # ---------- Decoders (no ResBlocks) ----------
        self.Dec_A = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)
        self.Dec_B = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)

        # ---------- Discriminators ----------
        self.D_A = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # ---------- Losses ----------
        self.gan = GANLoss(cfg.gan_mode)
        self.l1 = nn.L1Loss()

    # ---------------- BaseTrainer interface ----------------

    def generator_parameters(self):
        params = []
        params += list(self.Ec_A.parameters()) + list(self.Ec_B.parameters())
        params += list(self.Bn_A.parameters()) + list(self.Bn_B.parameters())
        params += list(self.Es_A.parameters()) + list(self.Es_B.parameters())
        params += list(self.AdaIN_A.parameters()) + list(self.AdaIN_B.parameters())
        params += list(self.Dec_A.parameters()) + list(self.Dec_B.parameters())
        return params

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    # ---------------- encoding/decoding helpers ----------------

    def encode_A(self, xA: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.Bn_A(self.Ec_A(xA))
        s = self.Es_A(xA)
        return c, s

    def encode_B(self, xB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.Bn_B(self.Ec_B(xB))
        s = self.Es_B(xB)
        return c, s

    def decode_A(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.AdaIN_A(c, s)
        return self.Dec_A(h)

    def decode_B(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.AdaIN_B(c, s)
        return self.Dec_B(h)

    # ---------------- training losses ----------------

    def compute_generator_loss(self, batch):
        real_A = batch["A"]
        real_B = batch["B"]
        B = real_A.size(0)

        # Encode
        cA, sA = self.encode_A(real_A)
        cB, sB = self.encode_B(real_B)

        # Sample random style codes (multimodal)
        sA_rand = torch.randn(B, self.cfg.style_dim, device=real_A.device)
        sB_rand = torch.randn(B, self.cfg.style_dim, device=real_A.device)

        # Within-domain reconstructions
        rec_A = self.decode_A(cA, sA)
        rec_B = self.decode_B(cB, sB)

        # Cross-domain translations (use random style for multimodality)
        fake_B = self.decode_B(cA, sB_rand)
        fake_A = self.decode_A(cB, sA_rand)

        # Re-encode translated images for content/style consistency
        cA_hat, sB_hat = self.encode_B(fake_B)   # content should match cA, style should match sB_rand
        cB_hat, sA_hat = self.encode_A(fake_A)

        # GAN losses
        loss_gan = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)

        # Image recon loss
        loss_recon_img = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)

        # Content recon (cross)
        loss_recon_content = self.l1(cA_hat, cA.detach()) + self.l1(cB_hat, cB.detach())

        # Style recon (cross)
        loss_recon_style = self.l1(sB_hat, sB_rand.detach()) + self.l1(sA_hat, sA_rand.detach())

        loss_G = (
            self.cfg.lambda_gan * loss_gan
            + self.cfg.lambda_recon_img * loss_recon_img
            + self.cfg.lambda_recon_content * loss_recon_content
            + self.cfg.lambda_recon_style * loss_recon_style
        )

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_gan": float(loss_gan.detach().cpu()),
            "loss_recon_img": float(loss_recon_img.detach().cpu()),
            "loss_recon_content": float(loss_recon_content.detach().cpu()),
            "loss_recon_style": float(loss_recon_style.detach().cpu()),
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

    def compute_discriminator_loss(self, batch, visuals):
        real_A = batch["A"]
        real_B = batch["B"]
        fake_A = visuals["fake_A"].detach()
        fake_B = visuals["fake_B"].detach()

        loss_D_A = 0.5 * (self.gan(self.D_A(real_A), True) + self.gan(self.D_A(fake_A), False))
        loss_D_B = 0.5 * (self.gan(self.D_B(real_B), True) + self.gan(self.D_B(fake_B), False))
        loss_D = loss_D_A + loss_D_B

        logs = {
            "loss_D": float(loss_D.detach().cpu()),
            "loss_D_A": float(loss_D_A.detach().cpu()),
            "loss_D_B": float(loss_D_B.detach().cpu()),
        }
        return loss_D, logs
