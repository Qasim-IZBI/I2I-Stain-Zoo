# cyclegan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from base_models import GANLoss, NLayerDiscriminator, ImagePool
from base_models import Encoder, Decoder, Decoder3Head, ResnetBottleneck
from base_models import discriminator_loss, identity_loss, ggd_nll, ggd_aleatoric_var


@dataclass
class CycleGANConfig:
    # architecture
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    n_layers_D: int = 3
    n_down: int = 2
    n_up: int = 2

    # losses
    gan_mode: str = "lsgan"         # "lsgan" or "vanilla"
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5    # typical CycleGAN uses 0.5; set 0 to disable

    # UGAC (Upadhyay et al., NeurIPS 2021) — aleatoric uncertainty heads.
    # False reproduces vanilla CycleGAN exactly; True swaps the decoders for
    # 3-head variants and replaces the L1 cycle loss with the GGD NLL.
    ugac: bool = False

    # misc
    pool_size: int = 50


class CycleGAN(nn.Module):
    """
    CycleGAN using split generator components:
      G_A2B = Enc_A -> Bn_A -> Dec_B
      G_B2A = Enc_B -> Bn_B -> Dec_A

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()
      - compute_generator_loss(batch) -> (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) -> (loss, logs)
    """
    def __init__(self, cfg: CycleGANConfig):
        super().__init__()
        self.cfg = cfg

        # ----- Generators -----
        self.Enc_A = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)
        self.Enc_B = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)
        Cc = self.Enc_A.out_channels

        self.Bn_A = ResnetBottleneck(Cc, n_blocks=cfg.n_blocks)
        self.Bn_B = ResnetBottleneck(Cc, n_blocks=cfg.n_blocks)

        dec_cls = Decoder3Head if cfg.ugac else Decoder
        self.Dec_A = dec_cls(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)
        self.Dec_B = dec_cls(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)

        # ----- Discriminators -----
        self.D_A = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # ----- Buffers -----
        self.pool_A = ImagePool(cfg.pool_size)
        self.pool_B = ImagePool(cfg.pool_size)

        # ----- Losses -----
        self.gan = GANLoss(cfg.gan_mode)
        self.l1 = nn.L1Loss()

    # ---------------- BaseTrainer interface ----------------

    def generator_parameters(self):
        params = []
        params += list(self.Enc_A.parameters()) + list(self.Bn_A.parameters()) + list(self.Dec_B.parameters())
        params += list(self.Enc_B.parameters()) + list(self.Bn_B.parameters()) + list(self.Dec_A.parameters())
        return params

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    # ---------------- Forward helpers ----------------

    def _decode_A2B(self, xA: torch.Tensor):
        z = self.Bn_A(self.Enc_A(xA))
        return self.Dec_B(z)

    def _decode_B2A(self, xB: torch.Tensor):
        z = self.Bn_B(self.Enc_B(xB))
        return self.Dec_A(z)

    def forward_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        """Translated image only — unchanged contract in both modes."""
        out = self._decode_A2B(xA)
        return out[0] if self.cfg.ugac else out

    def forward_B2A(self, xB: torch.Tensor) -> torch.Tensor:
        out = self._decode_B2A(xB)
        return out[0] if self.cfg.ugac else out

    # ---------------- UGAC inference ----------------

    @torch.no_grad()
    def forward_A2B_uncertainty(self, xA: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (translated image, per-pixel aleatoric variance [N,1,H,W]).

        Closed form — a single forward pass, no sampling. Requires cfg.ugac.
        """
        if not self.cfg.ugac:
            raise RuntimeError("forward_A2B_uncertainty() requires a model trained with ugac=True")
        mu, inv_alpha, beta = self._decode_A2B(xA)
        return mu, ggd_aleatoric_var(inv_alpha, beta)

    @torch.no_grad()
    def forward_B2A_uncertainty(self, xB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.ugac:
            raise RuntimeError("forward_B2A_uncertainty() requires a model trained with ugac=True")
        mu, inv_alpha, beta = self._decode_B2A(xB)
        return mu, ggd_aleatoric_var(inv_alpha, beta)

    # ---------------- Losses ----------------

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        real_A = batch["A"]
        real_B = batch["B"]

        if self.cfg.ugac:
            # UGAC Eq. 9: the cycle pass emits the GGD parameters that are supervised.
            fake_B = self.forward_A2B(real_A)
            fake_A = self.forward_B2A(real_B)
            rec_A, inv_alpha_A, beta_A = self._decode_B2A(fake_B)
            rec_B, inv_alpha_B, beta_B = self._decode_A2B(fake_A)
        else:
            fake_B = self.forward_A2B(real_A)
            fake_A = self.forward_B2A(real_B)
            rec_A = self.forward_B2A(fake_B)
            rec_B = self.forward_A2B(fake_A)

        # Identity (optional) — always on mu; forward_* returns the image in both modes
        loss_idt = identity_loss(self.l1, self.forward_A2B, self.forward_B2A,
                                 real_A, real_B, self.cfg.lambda_identity)

        # GAN
        loss_gan = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)

        # Cycle — GGD NLL (UGAC Eq. 7/8) or plain L1
        if self.cfg.ugac:
            loss_cycle = (ggd_nll(rec_A, real_A, inv_alpha_A, beta_A)
                          + ggd_nll(rec_B, real_B, inv_alpha_B, beta_B))
        else:
            loss_cycle = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)

        loss_G = loss_gan + self.cfg.lambda_cycle * loss_cycle + self.cfg.lambda_identity * loss_idt

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_gan": float(loss_gan.detach().cpu()),
            "loss_cycle": float(loss_cycle.detach().cpu()),
            "loss_idt": float(loss_idt.detach().cpu()),
        }
        if self.cfg.ugac:
            # L1 cycle is logged alongside so UGAC and vanilla runs stay comparable
            with torch.no_grad():
                logs["cycle_l1"] = float((self.l1(rec_A, real_A) + self.l1(rec_B, real_B)).cpu())
                logs["alpha_mean"] = float((1.0 / inv_alpha_A).mean().cpu())
                logs["beta_mean"] = float(beta_A.mean().cpu())

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

