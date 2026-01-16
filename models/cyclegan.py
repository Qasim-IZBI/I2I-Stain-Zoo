# cyclegan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models import GANLoss, NLayerDiscriminator, ImagePool
from models import Encoder, Decoder, ResnetBottleneck


@dataclass
class CycleGANConfig:
    # architecture
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    n_layers_D: int = 3

    # losses
    gan_mode: str = "lsgan"         # "lsgan" or "vanilla"
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5    # typical CycleGAN uses 0.5; set 0 to disable

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

        self.Dec_A = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)
        self.Dec_B = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)

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

    def forward_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        z = self.Enc_A(xA)
        z = self.Bn_A(z)
        return self.Dec_B(z)

    def forward_B2A(self, xB: torch.Tensor) -> torch.Tensor:
        z = self.Enc_B(xB)
        z = self.Bn_B(z)
        return self.Dec_A(z)

    # ---------------- Losses ----------------

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        real_A = batch["A"]
        real_B = batch["B"]

        # Translate
        fake_B = self.forward_A2B(real_A)
        fake_A = self.forward_B2A(real_B)

        # Cycle
        rec_A = self.forward_B2A(fake_B)
        rec_B = self.forward_A2B(fake_A)

        # Identity (optional)
        loss_idt = torch.tensor(0.0, device=real_A.device)
        if self.cfg.lambda_identity > 0:
            idt_A = self.forward_B2A(real_A)
            idt_B = self.forward_A2B(real_B)
            loss_idt = 0.5 * (self.l1(idt_A, real_A) + self.l1(idt_B, real_B))

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
        real_A = batch["A"]
        real_B = batch["B"]

        fake_A = self.pool_A.query(visuals["fake_A"].detach())
        fake_B = self.pool_B.query(visuals["fake_B"].detach())

        loss_D_A = 0.5 * (self.gan(self.D_A(real_A), True) + self.gan(self.D_A(fake_A), False))
        loss_D_B = 0.5 * (self.gan(self.D_B(real_B), True) + self.gan(self.D_B(fake_B), False))
        loss_D = loss_D_A + loss_D_B

        logs = {
            "loss_D": float(loss_D.detach().cpu()),
            "loss_D_A": float(loss_D_A.detach().cpu()),
            "loss_D_B": float(loss_D_B.detach().cpu()),
        }
        return loss_D, logs

