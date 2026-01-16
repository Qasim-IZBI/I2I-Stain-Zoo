# dclgan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GANLoss, NLayerDiscriminator, ImagePool
from models import Encoder, Decoder, ResnetBottleneck


# ============================================================
# Config
# ============================================================

@dataclass
class DCLGANConfig:
    input_nc: int = 3
    output_nc: int = 3

    ngf: int = 64
    n_down: int = 2
    n_up: int = 2
    n_blocks: int = 9  # bottleneck ResBlocks (all live here)

    # Discriminator
    ndf: int = 64
    n_layers_D: int = 3

    # Losses
    gan_mode: str = "lsgan"
    lambda_gan: float = 1.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.0  # optional, default off for VRAM
    lambda_dcl: float = 1.0       # dual contrastive weight

    # DCL / contrastive
    n_patches: int = 256
    proj_dim: int = 256
    temperature: float = 0.07

    # Replay buffer
    pool_size: int = 50


# ============================================================
# Patch sampling + projection (CUT-style)
# ============================================================

class PatchSampleF(nn.Module):
    """
    Samples patch features from a list of feature maps and projects them into proj_dim.
    Returns:
      - projected features: list of [B*K, proj_dim]
      - patch ids: list of [B, K] linear indices (so you can sample the same patches from another map)
    """
    def __init__(self, in_dims: List[int], proj_dim: int = 256):
        super().__init__()
        self.proj_dim = proj_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim),
                nn.ReLU(True),
                nn.Linear(proj_dim, proj_dim),
            ) for d in in_dims
        ])

    @staticmethod
    def _flatten_hw(feat: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B, H*W, C]
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int,
        patch_ids: List[torch.Tensor] | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        out_feats: List[torch.Tensor] = []
        out_ids: List[torch.Tensor] = []

        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            flat = self._flatten_hw(f)  # [B, HW, C]
            HW = flat.shape[1]

            if patch_ids is None:
                # sample K locations per image
                K = min(num_patches, HW)
                ids = torch.randint(0, HW, (B, K), device=f.device)
            else:
                ids = patch_ids[i]

            # gather: [B,K,C]
            gathered = torch.gather(
                flat,
                dim=1,
                index=ids.unsqueeze(-1).expand(-1, -1, C),
            )

            # [B*K, C]
            gathered = gathered.reshape(-1, C)

            # project
            proj = self.mlps[i](gathered)  # [B*K, proj_dim]
            proj = F.normalize(proj, dim=1)

            out_feats.append(proj)
            out_ids.append(ids)

        return out_feats, out_ids


def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    q,k: [N, D] with one-to-one positives by index.
    """
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


# ============================================================
# DCLGAN model
# ============================================================

class DCLGAN(nn.Module):
    """
    DCLGAN = CycleGAN backbone + Dual Contrastive Learning.

    Uses split modules:
      G_A2B = Enc_A -> Bn_A -> Dec_B
      G_B2A = Enc_B -> Bn_B -> Dec_A

    Dual contrastive:
      - compare patch features between real and translated within each direction.
      - we extract features from encoder outputs + bottleneck outputs (configurable).

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()
      - compute_generator_loss(batch) -> (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) -> (loss, logs)
    """
    def __init__(self, cfg: DCLGANConfig):
        super().__init__()
        self.cfg = cfg

        # ----- Generators (split) -----
        self.Enc_A = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down, return_features=True, feature_layers=[3, 6, 9])
        self.Enc_B = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down, return_features=True, feature_layers=[3, 6, 9])

        Cc = self.Enc_A.out_channels  # typically 256
        self.Bn_A = ResnetBottleneck(Cc, n_blocks=cfg.n_blocks, return_features=True, feature_blocks=[0, cfg.n_blocks // 2, cfg.n_blocks - 1])
        self.Bn_B = ResnetBottleneck(Cc, n_blocks=cfg.n_blocks, return_features=True, feature_blocks=[0, cfg.n_blocks // 2, cfg.n_blocks - 1])

        self.Dec_A = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)
        self.Dec_B = Decoder(Cc, cfg.output_nc, ngf=cfg.ngf, n_up=cfg.n_up)

        # ----- Discriminators -----
        self.D_A = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # ----- Losses / buffers -----
        self.gan = GANLoss(cfg.gan_mode)
        self.l1 = nn.L1Loss()

        self.pool_A = ImagePool(cfg.pool_size)
        self.pool_B = ImagePool(cfg.pool_size)

        # ----- Patch sampler/projection -----
        # We will use:
        # - encoder output (bottleneck input): Cc
        # - bottleneck early/mid/late outputs: Cc each
        # Total list dims: 1 + 3 = 4 feature maps, each Cc channels
        in_dims = [Cc, Cc, Cc, Cc]
        self.F = PatchSampleF(in_dims=in_dims, proj_dim=cfg.proj_dim)

    # ---------------- BaseTrainer interface ----------------

    def generator_parameters(self):
        params = []
        params += list(self.Enc_A.parameters()) + list(self.Enc_B.parameters())
        params += list(self.Bn_A.parameters()) + list(self.Bn_B.parameters())
        params += list(self.Dec_A.parameters()) + list(self.Dec_B.parameters())
        params += list(self.F.parameters())
        return params

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    # ---------------- generator forward helpers ----------------

    def G_A2B(self, xA: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zA, f_enc = self.Enc_A(xA)                # zA: [B,C,64,64]
        zA2, f_bn = self.Bn_A(zA)                 # refined
        yB = self.Dec_B(zA2)
        feats = {"enc_out": zA}
        feats.update(f_enc)
        feats.update(f_bn)
        return yB, feats

    def G_B2A(self, xB: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zB, f_enc = self.Enc_B(xB)
        zB2, f_bn = self.Bn_B(zB)
        yA = self.Dec_A(zB2)
        feats = {"enc_out": zB}
        feats.update(f_enc)
        feats.update(f_bn)
        return yA, feats

    # ---------------- feature list for DCL ----------------

    def _collect_dcl_feats(self, feat_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Return a fixed list of feature maps (length 4) for PatchSampleF.
        We include:
          1) encoder output (enc_out)      : [B,C,64,64]
          2) bottleneck block 0            : [B,C,64,64]
          3) bottleneck middle block       : [B,C,64,64]
          4) bottleneck last block         : [B,C,64,64]
        """
        feats = [feat_dict["enc_out"]]

        # These keys come from ResnetBottleneck with return_features=True:
        # "bottleneck_block_0", "bottleneck_block_mid", etc.
        # We used indices: 0, n//2, n-1
        # ResnetBottleneck names: bottleneck_block_{i}
        n = self.cfg.n_blocks
        feats.append(feat_dict[f"bottleneck_block_{0}"])
        feats.append(feat_dict[f"bottleneck_block_{n // 2}"])
        feats.append(feat_dict[f"bottleneck_block_{n - 1}"])
        return feats

    def _dcl_loss(
        self,
        feats_q: List[torch.Tensor],
        feats_k: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Dual contrastive loss between two lists of feature maps.
        Uses the SAME patch ids across q and k for each layer.
        """
        q_proj, patch_ids = self.F(feats_q, num_patches=self.cfg.n_patches, patch_ids=None)
        k_proj, _ = self.F(feats_k, num_patches=self.cfg.n_patches, patch_ids=patch_ids)

        loss = 0.0
        for q, k in zip(q_proj, k_proj):
            loss = loss + info_nce(q, k, temperature=self.cfg.temperature)
        return loss / len(q_proj)

    # ---------------- losses ----------------

    def compute_generator_loss(self, batch):
        real_A = batch["A"]
        real_B = batch["B"]

        # Forward translations + feature capture
        fake_B, feats_A = self.G_A2B(real_A)
        fake_A, feats_B = self.G_B2A(real_B)

        # Cycle
        rec_A, _ = self.G_B2A(fake_B)
        rec_B, _ = self.G_A2B(fake_A)

        # Optional identity (off by default)
        loss_idt = torch.tensor(0.0, device=real_A.device)
        if self.cfg.lambda_identity > 0:
            idt_A, _ = self.G_B2A(real_A)
            idt_B, _ = self.G_A2B(real_B)
            loss_idt = 0.5 * (self.l1(idt_A, real_A) + self.l1(idt_B, real_B))

        # GAN losses
        loss_gan = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)

        # Cycle losses
        loss_cycle = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)

        # Dual contrastive:
        # We need features from real and translated; DCLGAN usually contrasts real vs translated features.
        # We'll compute features for fake images as well (through the SAME direction network).
        # A direction: compare Enc/Bn features from real_A vs from fake_B (produced by A2B).
        _, feats_A_fake = self.G_A2B(fake_A.detach())  # note: alternative choices exist; keep consistent in your experiments
        _, feats_B_fake = self.G_B2A(fake_B.detach())

        dcl_A = self._dcl_loss(self._collect_dcl_feats(feats_A), self._collect_dcl_feats(feats_A_fake))
        dcl_B = self._dcl_loss(self._collect_dcl_feats(feats_B), self._collect_dcl_feats(feats_B_fake))
        loss_dcl = dcl_A + dcl_B

        loss_G = (
            self.cfg.lambda_gan * loss_gan
            + self.cfg.lambda_cycle * loss_cycle
            + self.cfg.lambda_identity * loss_idt
            + self.cfg.lambda_dcl * loss_dcl
        )

        logs = {
            "loss_G": float(loss_G.detach().cpu()),
            "loss_gan": float(loss_gan.detach().cpu()),
            "loss_cycle": float(loss_cycle.detach().cpu()),
            "loss_idt": float(loss_idt.detach().cpu()),
            "loss_dcl": float(loss_dcl.detach().cpu()),
            "loss_dcl_A": float(dcl_A.detach().cpu()),
            "loss_dcl_B": float(dcl_B.detach().cpu()),
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
