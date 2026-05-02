# models/unsb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models import GANLoss, NLayerDiscriminator, ImagePool
from models.miudiff import DDPMUNet, UNetConfig, DiffusionSchedule


@dataclass
class UNSBConfig:
    # diffusion schedule
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # UNet architecture
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2

    # discriminator
    ndf: int = 64
    n_layers_D: int = 3
    gan_mode: str = "lsgan"

    # loss weights
    lambda_adv: float = 0.1   # adversarial loss on predicted x0
    lambda_score: float = 1.0  # score matching (MSE) loss

    # replay buffer
    pool_size: int = 50

    # sampling
    sample_steps: int = 200


class UNSB(nn.Module):
    """
    UNSB-inspired model: Unpaired translation via Schrödinger Bridge.

    The forward SB process starts from xA (not standard Gaussian noise):
        x_t = sqrt(ᾱ_t)*xA + sqrt(1-ᾱ_t)*eps

    A score network z_theta(cat[x_t, xA], t) learns to predict eps such that
    denoising x_t conditioned on xA produces domain-B images.

    An adversarial discriminator D_adv pushes the predicted clean image
    x0_pred = (x_t - sqrt(1-ᾱ_t)*eps_pred) / sqrt(ᾱ_t) toward the B distribution.

    Compared to the full UNSB paper (which uses per-timestep Markovian discriminators),
    this is a practical single-discriminator simplification that operates on x0_pred.

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()
      - compute_generator_loss(batch) → (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) → (loss, logs)
    """

    def __init__(self, cfg: UNSBConfig):
        super().__init__()
        self.cfg = cfg

        sched = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end)
        _betas, _alphas, _a_bar, _a_bar_prev = sched.make("cpu")
        self.register_buffer("_betas", _betas)
        self.register_buffer("_alphas", _alphas)
        self.register_buffer("_a_bar", _a_bar)
        self.register_buffer("_a_bar_prev", _a_bar_prev)

        # Score network: 6-channel input [x_t (3ch), xA (3ch)]
        unet_kw = dict(
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
        )
        self.z_theta = DDPMUNet(UNetConfig(in_channels=6, **unet_kw))

        # Discriminator on predicted clean images
        self.D_adv = NLayerDiscriminator(3, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.pool_B = ImagePool(cfg.pool_size)
        self.gan = GANLoss(cfg.gan_mode)

    # ---- BaseTrainer interface ----

    def generator_parameters(self):
        return list(self.z_theta.parameters())

    def discriminator_parameters(self):
        return list(self.D_adv.parameters())

    # ---- helpers ----

    def _q_sample_from_A(self, xA: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """SB forward: x_t = sqrt(ᾱ_t)*xA + sqrt(1-ᾱ_t)*eps  (starts from xA, not xB)."""
        a = self._a_bar[t_idx].view(-1, 1, 1, 1)
        return a.sqrt() * xA + (1.0 - a).sqrt() * eps

    # ---- training ----

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device
        xA = batch["A"].to(device).float()
        B = xA.size(0)

        t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
        t_frac = t_idx.float() / (self.cfg.T - 1)

        eps = torch.randn_like(xA)
        x_t = self._q_sample_from_A(xA, t_idx, eps)

        eps_pred = self.z_theta(torch.cat([x_t, xA], dim=1), t_frac)

        # Score matching: predict the added noise
        loss_score = F.mse_loss(eps_pred, eps)

        # Predicted clean target image
        a_bar_t = self._a_bar[t_idx].view(-1, 1, 1, 1)
        x0_pred = (x_t - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
        x0_pred = x0_pred.clamp(-1, 1)

        # Adversarial loss: push x0_pred toward domain B
        loss_adv = self.gan(self.D_adv(x0_pred), True)

        loss_G = self.cfg.lambda_score * loss_score + self.cfg.lambda_adv * loss_adv

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_score": float(loss_score.detach().cpu()),
            "loss_adv": float(loss_adv.detach().cpu()),
        }
        visuals = {"real_A": xA, "fake_B": x0_pred}
        return loss_G, logs, visuals

    def compute_discriminator_loss(self, batch: Dict[str, torch.Tensor], visuals: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device
        real_B = batch["B"].to(device).float()

        fake_B = self.pool_B.query(visuals["fake_B"].detach())

        loss_real = self.gan(self.D_adv(real_B), True)
        loss_fake = self.gan(self.D_adv(fake_B), False)
        loss_D = (loss_real + loss_fake) * 0.5

        logs = {
            "loss_D": float(loss_D.detach().cpu()),
            "loss_D_real": float(loss_real.detach().cpu()),
            "loss_D_fake": float(loss_fake.detach().cpu()),
        }
        return loss_D, logs

    # ---- inference ----

    def forward_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        """
        DDIM sampling conditioned on xA.
        Initialises from noisy xA (SB-style) and denoises toward domain B.
        """
        self.eval()
        device = xA.device
        B, _, H, W = xA.shape
        T = self.cfg.T
        steps = self.cfg.sample_steps

        # Start from heavily-noised xA (SB initial distribution at t=T-1)
        t_start = T - 1
        a_bar_T = self._a_bar[t_start].view(1, 1, 1, 1)
        y = a_bar_T.sqrt() * xA + (1.0 - a_bar_T).sqrt() * torch.randn_like(xA)

        t_list = torch.linspace(t_start, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)

        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t = t_list[i]
                t_next = t_list[i + 1]
                t_frac = t.repeat(B).float() / (T - 1)

                eps_pred = self.z_theta(torch.cat([y, xA], dim=1), t_frac)
                a_bar_t = self._a_bar[t].view(1, 1, 1, 1)
                a_bar_next = self._a_bar[t_next].view(1, 1, 1, 1)

                x0_pred = (y - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

                y = a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).sqrt() * eps_pred

        return y.clamp(-1, 1)
