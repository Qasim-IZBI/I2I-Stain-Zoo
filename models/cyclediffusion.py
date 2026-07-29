# models/cyclediffusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models import DDPMUNet, UNetConfig, DiffusionSchedule


@dataclass
class CycleDiffusionConfig:
    # diffusion schedule
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # UNet architecture (shared for both domains)
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2

    # sampling
    sample_steps: int = 200


class CycleDiffusion(nn.Module):
    """
    CycleDiffusion: two unconditional DDPMs for unpaired image-to-image translation.

    One DDPM per domain (eps_A, eps_B).  Both are trained jointly with standard
    DDPM MSE losses on their respective domains.

    Translation A→B at inference:
      1. DDIM-invert xA forward in time using eps_A → shared noise code z.
      2. DDIM-decode z backward in time using eps_B → translated image yB.

    The shared noise space acts as the cross-domain latent: cycles are implicit
    because the same z is used in both directions without a discriminator.

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()   → empty
      - compute_generator_loss(batch) → (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) → (0, {})
    """

    def __init__(self, cfg: CycleDiffusionConfig):
        super().__init__()
        self.cfg = cfg

        sched = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end)
        _betas, _alphas, _a_bar, _a_bar_prev = sched.make("cpu")
        self.register_buffer("_betas", _betas)
        self.register_buffer("_alphas", _alphas)
        self.register_buffer("_a_bar", _a_bar)
        self.register_buffer("_a_bar_prev", _a_bar_prev)

        unet_kw = dict(
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
        )
        self.eps_A = DDPMUNet(UNetConfig(in_channels=3, **unet_kw))
        self.eps_B = DDPMUNet(UNetConfig(in_channels=3, **unet_kw))

    # ---- BaseTrainer interface ----

    def generator_parameters(self):
        return list(self.eps_A.parameters()) + list(self.eps_B.parameters())

    def discriminator_parameters(self):
        return []

    def compute_discriminator_loss(self, batch, visuals):
        return torch.tensor(0.0, device=next(self.parameters()).device), {}

    # ---- helpers ----

    def _q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self._a_bar[t_idx].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * eps

    # ---- training ----

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device
        xA = batch["A"].to(device).float()
        xB = batch["B"].to(device).float()
        B = xA.size(0)

        t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
        t_frac = t_idx.float() / (self.cfg.T - 1)

        eps_a = torch.randn_like(xA)
        eps_b = torch.randn_like(xB)

        xA_t = self._q_sample(xA, t_idx, eps_a)
        xB_t = self._q_sample(xB, t_idx, eps_b)

        eps_pred_A = self.eps_A(xA_t, t_frac)
        eps_pred_B = self.eps_B(xB_t, t_frac)

        loss_A = F.mse_loss(eps_pred_A, eps_a)
        loss_B = F.mse_loss(eps_pred_B, eps_b)
        loss = loss_A + loss_B

        logs = {
            "loss_G": float(loss.detach().cpu()),
            "loss_A": float(loss_A.detach().cpu()),
            "loss_B": float(loss_B.detach().cpu()),
        }
        visuals = {"real_A": xA, "real_B": xB}
        return loss, logs, visuals

    # ---- DDIM helpers ----

    def _ddim_invert(self, x0: torch.Tensor, eps_net: nn.Module) -> torch.Tensor:
        """
        DDIM inversion: encode x0 → noise code z by running the forward diffusion
        deterministically using the given score network.

        Returns z at t = T-1.
        """
        T = self.cfg.T
        steps = self.cfg.sample_steps
        B = x0.size(0)
        device = x0.device

        # ascending t_list: 0 → T-1
        t_list = torch.linspace(0, T - 1, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)

        y = x0.clone()
        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t = t_list[i]
                t_next = t_list[i + 1]
                t_frac = t.repeat(B).float() / (T - 1)

                eps_pred = eps_net(y, t_frac)
                a_bar_t = self._a_bar[t].view(1, 1, 1, 1)
                a_bar_next = self._a_bar[t_next].view(1, 1, 1, 1)

                # DDIM forward step (inverts the reverse process)
                x0_pred = (y - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)
                y = a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).sqrt() * eps_pred

        return y

    def _ddim_decode(self, z: torch.Tensor, eps_net: nn.Module) -> torch.Tensor:
        """DDIM decoding: noise code z → clean image using the given score network."""
        T = self.cfg.T
        steps = self.cfg.sample_steps
        B = z.size(0)
        device = z.device

        t_list = torch.linspace(T - 1, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)

        y = z.clone()
        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t = t_list[i]
                t_next = t_list[i + 1]
                t_frac = t.repeat(B).float() / (T - 1)

                eps_pred = eps_net(y, t_frac)
                a_bar_t = self._a_bar[t].view(1, 1, 1, 1)
                a_bar_next = self._a_bar[t_next].view(1, 1, 1, 1)

                x0_pred = (y - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)
                y = a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).sqrt() * eps_pred

        return y.clamp(-1, 1)

    # ---- inference ----

    def forward_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        """DDIM-invert xA with eps_A, then DDIM-decode the noise code with eps_B."""
        self.eval()
        z = self._ddim_invert(xA, self.eps_A)
        return self._ddim_decode(z, self.eps_B)

    def forward_B2A(self, xB: torch.Tensor) -> torch.Tensor:
        """DDIM-invert xB with eps_B, then DDIM-decode the noise code with eps_A."""
        self.eval()
        z = self._ddim_invert(xB, self.eps_B)
        return self._ddim_decode(z, self.eps_A)
