# models/unitddpm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.miudiff import DDPMUNet, UNetConfig, DiffusionSchedule, to_gray, sobel_grad


@dataclass
class UNITDDPMConfig:
    # diffusion schedule
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # UNet architecture
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2

    # stage
    stage: str = "pretrain"  # "pretrain" | "finetune"

    # conditioning: full RGB source (3ch) concatenated to noisy target (3ch) = 6ch
    # Alternative: set cond_type="gray" to use 1ch grayscale → eps_cond has 4ch input
    cond_type: str = "rgb"  # "rgb" (3ch) | "gray" (1ch) | "sobel" (1ch)

    # structural regularisation (finetune)
    lambda_struct: float = 0.0

    # sampling
    sample_steps: int = 200


class UNITDDPM(nn.Module):
    """
    UNIT-DDPM: two-stage conditional diffusion for unpaired image translation.

    Stage 1 (pretrain): eps_uncond — unconditional DDPM on domain B only.
                        Learns the target domain prior.
    Stage 2 (finetune): eps_cond — DDPM conditioned on source image xA.
                        Input = [y_t (3ch), cond(xA)].
                        Optional structural L1 loss (lambda_struct).

    Inference: DDIM from noise conditioned on xA.

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()   → empty (no discriminator)
      - compute_generator_loss(batch) → (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) → (0, {})
    """

    def __init__(self, cfg: UNITDDPMConfig):
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
        self.eps_uncond = DDPMUNet(UNetConfig(in_channels=3, **unet_kw))

        cond_ch = 3 if cfg.cond_type == "rgb" else 1
        self.eps_cond = DDPMUNet(UNetConfig(in_channels=3 + cond_ch, **unet_kw))

    # ---- BaseTrainer interface ----

    def generator_parameters(self):
        return list(self.eps_uncond.parameters()) + list(self.eps_cond.parameters())

    def discriminator_parameters(self):
        return []

    def compute_discriminator_loss(self, batch, visuals):
        return torch.tensor(0.0, device=next(self.parameters()).device), {}

    # ---- helpers ----

    def _extract_cond(self, xA: torch.Tensor) -> torch.Tensor:
        if self.cfg.cond_type == "rgb":
            return xA
        if self.cfg.cond_type == "gray":
            return to_gray(xA)
        if self.cfg.cond_type == "sobel":
            return sobel_grad(to_gray(xA))
        raise ValueError(f"Unknown cond_type {self.cfg.cond_type!r}")

    def _q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self._a_bar[t_idx].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * eps

    # ---- training ----

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device

        # Stage 1 — unconditional DDPM on domain B
        if self.cfg.stage == "pretrain":
            y0 = batch["B"].to(device).float()
            B = y0.size(0)
            t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
            t_frac = t_idx.float() / (self.cfg.T - 1)
            eps = torch.randn_like(y0)
            y_t = self._q_sample(y0, t_idx, eps)
            eps_pred = self.eps_uncond(y_t, t_frac)
            loss = F.mse_loss(eps_pred, eps)
            logs = {"loss_G": float(loss.detach().cpu()), "loss_eps": float(loss.detach().cpu())}
            visuals = {"real_B": y0, "noisy_B": y_t}
            return loss, logs, visuals

        # Stage 2 — conditional DDPM (A→B)
        xA = batch["A"].to(device).float()
        y0 = batch["B"].to(device).float()
        cond = self._extract_cond(xA)
        B = y0.size(0)

        t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
        t_frac = t_idx.float() / (self.cfg.T - 1)
        eps = torch.randn_like(y0)
        y_t = self._q_sample(y0, t_idx, eps)

        eps_pred = self.eps_cond(torch.cat([y_t, cond], dim=1), t_frac)
        loss_eps = F.mse_loss(eps_pred, eps)
        loss = loss_eps

        loss_struct = torch.tensor(0.0, device=device)
        if self.cfg.lambda_struct > 0:
            a_bar_t = self._a_bar[t_idx].view(-1, 1, 1, 1)
            x0_pred = (y_t - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)
            # predicted clean image should have same gray structure as source
            struct_pred = to_gray(x0_pred)
            struct_src = to_gray(xA)
            loss_struct = F.l1_loss(struct_pred, struct_src)
            loss = loss + self.cfg.lambda_struct * loss_struct

        logs = {
            "loss_G": float(loss.detach().cpu()),
            "loss_eps": float(loss_eps.detach().cpu()),
            "loss_struct": float(loss_struct.detach().cpu()),
        }
        visuals = {"real_A": xA, "real_B": y0, "noisy_B": y_t}
        return loss, logs, visuals

    # ---- inference ----

    def forward_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        """DDIM sampling conditioned on xA."""
        self.eval()
        device = xA.device
        B, _, H, W = xA.shape
        cond = self._extract_cond(xA)

        y = torch.randn(B, 3, H, W, device=device)
        T = self.cfg.T
        steps = self.cfg.sample_steps

        t_list = torch.linspace(T - 1, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)

        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t = t_list[i]
                t_next = t_list[i + 1]
                t_frac = t.repeat(B).float() / (T - 1)

                eps_pred = self.eps_cond(torch.cat([y, cond], dim=1), t_frac)
                a_bar_t = self._a_bar[t].view(1, 1, 1, 1)
                a_bar_next = self._a_bar[t_next].view(1, 1, 1, 1)

                x0_pred = (y - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

                y = a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).sqrt() * eps_pred

        return y.clamp(-1, 1)

    def forward_B2A(self, xB: torch.Tensor) -> torch.Tensor:
        """Unconditional sampling from domain B prior (ignores xB content)."""
        self.eval()
        device = xB.device
        B, _, H, W = xB.shape
        y = torch.randn(B, 3, H, W, device=device)
        T = self.cfg.T
        steps = self.cfg.sample_steps

        t_list = torch.linspace(T - 1, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)

        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t = t_list[i]
                t_next = t_list[i + 1]
                t_frac = t.repeat(B).float() / (T - 1)

                eps_pred = self.eps_uncond(y, t_frac)
                a_bar_t = self._a_bar[t].view(1, 1, 1, 1)
                a_bar_next = self._a_bar[t_next].view(1, 1, 1, 1)

                x0_pred = (y - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

                y = a_bar_next.sqrt() * x0_pred + (1.0 - a_bar_next).sqrt() * eps_pred

        return y.clamp(-1, 1)
