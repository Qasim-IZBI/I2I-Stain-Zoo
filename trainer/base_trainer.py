# trainer/base_trainer.py
from __future__ import annotations

import os
import torch
from torch import nn, optim
from torchvision.utils import save_image
from typing import Dict, Any, Optional


class BaseTrainer:
    """
    Generic trainer for unpaired image-to-image translation models:
      CycleGAN, UNIT, MUNIT, DCLGAN

    The model is responsible for:
      - defining its networks
      - computing losses
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        *,
        device: torch.device,
        lr: float = 2e-4,
        betas=(0.5, 0.999),
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        save_dir: str = "checkpoints",
        sample_dir: str = "samples",
        sample_every: int = 5000,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.sample_every = sample_every

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        self.save_dir = save_dir
        self.sample_dir = sample_dir

        # --- optimizers ---
        self.opt_G = optim.Adam(
            self.model.generator_parameters(),
            lr=lr,
            betas=betas,
        )

        self.opt_D = None
        if hasattr(self.model, "discriminator_parameters"):
            params_D = self.model.discriminator_parameters()
            if params_D:
                self.opt_D = optim.Adam(params_D, lr=lr, betas=betas)

        # --- AMP ---
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.global_step = 0
        self.epoch = 0

    # ============================================================
    # Core training loop
    # ============================================================

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self._train_epoch()
            self.save_checkpoint(f"epoch_{epoch}.pt")

    def _train_epoch(self):
        self.model.train()

        for batch in self.dataloader:
            self.global_step += 1

            batch = self._to_device(batch)

            # -------------------------
            # Generator step
            # -------------------------
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss_G, logs_G, visuals = self.model.compute_generator_loss(batch)

            self.scaler.scale(loss_G / self.grad_accum_steps).backward()

            if self.global_step % self.grad_accum_steps == 0:
                self.scaler.step(self.opt_G)
                self.scaler.update()
                self.opt_G.zero_grad(set_to_none=True)

            # -------------------------
            # Discriminator step
            # -------------------------
            if self.opt_D is not None:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss_D, logs_D = self.model.compute_discriminator_loss(batch, visuals)

                self.scaler.scale(loss_D / self.grad_accum_steps).backward()

                if self.global_step % self.grad_accum_steps == 0:
                    self.scaler.step(self.opt_D)
                    self.scaler.update()
                    self.opt_D.zero_grad(set_to_none=True)
            else:
                logs_D = {}

            # -------------------------
            # Logging / sampling
            # -------------------------
            if self.global_step % self.sample_every == 0:
                self.save_samples(visuals)

            self.log({**logs_G, **logs_D})

    # ============================================================
    # Utilities
    # ============================================================

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def save_samples(self, visuals: Dict[str, torch.Tensor]):
        """
        visuals: dict of tensors in [-1,1], e.g.
          real_A, fake_B, rec_A, ...
        """
        def denorm(x):
            return (x + 1.0) / 2.0

        imgs = []
        for _, v in visuals.items():
            if torch.is_tensor(v):
                imgs.append(denorm(v[:4]))

        if not imgs:
            return

        grid = torch.cat(imgs, dim=0)
        path = os.path.join(self.sample_dir, f"step_{self.global_step}.png")
        save_image(grid, path, nrow=3)

    def save_checkpoint(self, name: str):
        path = os.path.join(self.save_dir, name)
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "opt_G": self.opt_G.state_dict(),
        }
        if self.opt_D is not None:
            state["opt_D"] = self.opt_D.state_dict()
        torch.save(state, path)
        print(f"[Checkpoint] Saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        if self.opt_D is not None and "opt_D" in ckpt:
            self.opt_D.load_state_dict(ckpt["opt_D"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Checkpoint] Loaded: {path}")

    def log(self, logs: Dict[str, float]):
        """
        Minimal logging; override or extend (TensorBoard / WandB).
        """
        if self.global_step % 50 == 0:
            msg = f"[E{self.epoch:03d} | S{self.global_step:06d}] "
            msg += " ".join([f"{k}:{v:.4f}" for k, v in logs.items()])
            print(msg)
