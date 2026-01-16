# unit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from base_models import GANLoss, NLayerDiscriminator
from base_models import Encoder, Decoder, ResnetBottleneck


# ============================================================
# Config
# ============================================================

@dataclass
class UNITConfig:
    input_nc: int = 3
    output_nc: int = 3

    ngf: int = 64
    n_down: int = 2
    n_up: int = 2

    # Bottleneck structure (your requested pattern)
    n_blocks_total: int = 9              # total residual blocks in bottleneck
    n_blocks_shared: int = 3             # blocks 4-6
    n_blocks_private_pre: int = 3        # blocks 1-3
    n_blocks_private_post: int = 3       # blocks 7-9

    # latent channels (must match encoder out_channels unless you add adapters)
    z_dim: int = 256

    gan_mode: str = "lsgan"
    lambda_gan: float = 1.0
    lambda_recon: float = 10.0
    lambda_kl: float = 0.01

    ndf: int = 64
    n_layers_D: int = 3


# ============================================================
# Helpers
# ============================================================

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class LatentHeads(nn.Module):
    """Spatial mu/logvar heads: [B,C,H,W] -> [B,z_dim,H,W]."""
    def __init__(self, in_channels: int, z_dim: int):
        super().__init__()
        self.mu = nn.Conv2d(in_channels, z_dim, 1, bias=True)
        self.logvar = nn.Conv2d(in_channels, z_dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.logvar(h)


# ============================================================
# UNIT (partial shared bottleneck: blocks 4-5 shared)
# ============================================================

class UNIT(nn.Module):
    def __init__(self, cfg: UNITConfig):
        super().__init__()
        self.cfg = cfg

        # Validate the requested partition
        if cfg.n_blocks_private_pre + cfg.n_blocks_shared + cfg.n_blocks_private_post != cfg.n_blocks_total:
            raise ValueError(
                "Bottleneck partition must sum to n_blocks_total. "
                f"Got pre={cfg.n_blocks_private_pre}, shared={cfg.n_blocks_shared}, post={cfg.n_blocks_private_post}, "
                f"total={cfg.n_blocks_total}"
            )
        # blocks 4-5 shared corresponds to: pre=3, shared=2, post=3 (for total=8)

        # ---------- Encoders (NO ResBlocks) ----------
        self.E_A = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)
        self.E_B = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)

        if self.E_A.out_channels != cfg.z_dim:
            raise ValueError(
                f"cfg.z_dim ({cfg.z_dim}) must match encoder out_channels ({self.E_A.out_channels}). "
                "Set z_dim accordingly or add 1x1 adapters."
            )

        # ---------- Latent heads ----------
        self.latent_A = LatentHeads(cfg.z_dim, cfg.z_dim)
        self.latent_B = LatentHeads(cfg.z_dim, cfg.z_dim)

        # ---------- Bottleneck: private-pre, shared, private-post ----------
        self.bn_pre_A = ResnetBottleneck(cfg.z_dim, n_blocks=cfg.n_blocks_private_pre)
        self.bn_pre_B = ResnetBottleneck(cfg.z_dim, n_blocks=cfg.n_blocks_private_pre)

        self.bn_shared = ResnetBottleneck(cfg.z_dim, n_blocks=cfg.n_blocks_shared)  # shared weights

        self.bn_post_A = ResnetBottleneck(cfg.z_dim, n_blocks=cfg.n_blocks_private_post)
        self.bn_post_B = ResnetBottleneck(cfg.z_dim, n_blocks=cfg.n_blocks_private_post)

        # ---------- Decoders (NO ResBlocks) ----------
        self.Dec_A = Decoder(cfg.z_dim, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)
        self.Dec_B = Decoder(cfg.z_dim, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)

        # ---------- Discriminators ----------
        self.D_A = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # ---------- Losses ----------
        self.gan = GANLoss(cfg.gan_mode)
        self.l1 = nn.L1Loss()

    # ---------------- BaseTrainer interface ----------------

    def generator_parameters(self):
        params = []
        params += list(self.E_A.parameters()) + list(self.E_B.parameters())
        params += list(self.latent_A.parameters()) + list(self.latent_B.parameters())

        # bottlenecks
        params += list(self.bn_pre_A.parameters()) + list(self.bn_pre_B.parameters())
        params += list(self.bn_shared.parameters())
        params += list(self.bn_post_A.parameters()) + list(self.bn_post_B.parameters())

        params += list(self.Dec_A.parameters()) + list(self.Dec_B.parameters())
        return params

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    # ---------------- internals ----------------

    def _encode(self, which: str, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if which == "A":
            h = self.E_A(x)
            mu, lv = self.latent_A(h)
        elif which == "B":
            h = self.E_B(x)
            mu, lv = self.latent_B(h)
        else:
            raise ValueError(which)
        z = reparameterize(mu, lv)
        return z, mu, lv

    def _bottleneck(self, which: str, z: torch.Tensor) -> torch.Tensor:
        if which == "A":
            z = self.bn_pre_A(z)
            z = self.bn_shared(z)
            z = self.bn_post_A(z)
        elif which == "B":
            z = self.bn_pre_B(z)
            z = self.bn_shared(z)
            z = self.bn_post_B(z)
        else:
            raise ValueError(which)
        return z

    def forward_A2B(self, xA: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zA, muA, lvA = self._encode("A", xA)
        zA = self._bottleneck("A", zA)
        fakeB = self.Dec_B(zA)
        return fakeB, {"mu": muA, "logvar": lvA}

    def forward_B2A(self, xB: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zB, muB, lvB = self._encode("B", xB)
        zB = self._bottleneck("B", zB)
        fakeA = self.Dec_A(zB)
        return fakeA, {"mu": muB, "logvar": lvB}

    # ---------------- losses ----------------

    def compute_generator_loss(self, batch):
        real_A = batch["A"]
        real_B = batch["B"]

        fake_B, latA = self.forward_A2B(real_A)
        fake_A, latB = self.forward_B2A(real_B)

        # Cycle-like consistency (often used in UNIT codebases)
        rec_A, _ = self.forward_B2A(fake_B)
        rec_B, _ = self.forward_A2B(fake_A)

        loss_gan = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)
        loss_recon = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)
        loss_kl = kl_loss(latA["mu"], latA["logvar"]) + kl_loss(latB["mu"], latB["logvar"])

        loss_G = (
            self.cfg.lambda_gan * loss_gan
            + self.cfg.lambda_recon * loss_recon
            + self.cfg.lambda_kl * loss_kl
        )

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_gan": float(loss_gan.detach().cpu()),
            "loss_recon": float(loss_recon.detach().cpu()),
            "loss_kl": float(loss_kl.detach().cpu()),
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
