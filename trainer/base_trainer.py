# trainer/base_trainer.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

import torch
from torch import nn, optim
from torchvision.utils import save_image
from typing import Dict, Any, List, Optional


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
        model_name: str = "",
        lr: float = 2e-4,
        betas=(0.5, 0.999),
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        save_dir: str = "checkpoints",
        sample_dir: str = "samples",
        save_epochs: int = 5,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.model_name = model_name

        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        self.save_dir = save_dir
        self.sample_dir = sample_dir
        self.save_epochs = save_epochs

        # --- optimizers ---
        g_params = list(self.model.generator_parameters())
        d_params = list(self.model.discriminator_parameters()) if hasattr(self.model, "discriminator_parameters") else []

        self.opt_G = torch.optim.Adam(g_params, lr=lr, betas=betas)

        self.opt_D = None
        if len(d_params) > 0:
            self.opt_D = torch.optim.Adam(d_params, lr=lr, betas=betas)

        # --- AMP ---
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.global_step = 0
        self.epoch = 0

        # --- training metadata ---
        self.lr = lr
        self.betas = betas

        # --- loss logging ---
        self.log_path = os.path.join(os.path.dirname(save_dir), "loss_log.csv")
        self._log_header_written = os.path.exists(self.log_path)
        self._epoch_losses: List[Dict[str, float]] = []

    # ============================================================
    # Core training loop
    # ============================================================

    def train(self, num_epochs: int):
        self._save_training_meta(num_epochs)
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self._train_epoch()
            if epoch % self.save_epochs  == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

    def _train_epoch(self):
        self.model.train()
        total_steps = len(self.dataloader)
        half_step = total_steps // 2

        for step_in_epoch, batch in enumerate(self.dataloader, 1):
            self.global_step += 1

            batch = self._to_device(batch)

            # -------------------------
            # Generator step
            # -------------------------

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss_G, logs_G, visuals = self.model.compute_generator_loss(batch)

            # Skip bad steps early (prevents poisoning the optimizer state)
            if not torch.isfinite(loss_G):
                print(f"[WARN] Non-finite loss_G at step {self.global_step}, skipping step.")
                self.opt_G.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss_G / self.grad_accum_steps).backward()

            if self.global_step % self.grad_accum_steps == 0:
                # unscale first so we can check true grads
                self.scaler.unscale_(self.opt_G)

                # If any grad is non-finite, skip the optimizer step (prevents poisoning)
                found_inf = False
                for p in self.model.generator_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        found_inf = True
                        break

                if found_inf:
                    print(f"[WARN] Non-finite gradients at step {self.global_step}, skipping optimizer step.")
                    self.opt_G.zero_grad(set_to_none=True)
                    self.scaler.update()  # still update scaler to reduce scale
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), 1.0)

                self.scaler.step(self.opt_G)
                self.scaler.update()
                self.opt_G.zero_grad(set_to_none=True)

            # if self.global_step % self.grad_accum_steps == 0:
            #     # IMPORTANT: unscale before clipping
            #     self.scaler.unscale_(self.opt_G)
            #     torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), 1.0)

            #     self.scaler.step(self.opt_G)
            #     self.scaler.update()
            #     self.opt_G.zero_grad(set_to_none=True)

            # # -------------------------
            # # Generator step
            # # -------------------------
            # with torch.cuda.amp.autocast(enabled=self.use_amp):
            #     loss_G, logs_G, visuals = self.model.compute_generator_loss(batch)

            # self.scaler.scale(loss_G / self.grad_accum_steps).backward()

            # if self.global_step % self.grad_accum_steps == 0:
            #     self.scaler.step(self.opt_G)
            #     self.scaler.update()
            #     self.opt_G.zero_grad(set_to_none=True)

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
            # Accumulate losses
            # -------------------------
            self._epoch_losses.append({**logs_G, **logs_D})

            # -------------------------
            # Logging / sampling (twice per epoch: at half and end)
            # -------------------------
            if step_in_epoch == half_step or step_in_epoch == total_steps:
                self.save_samples(visuals)
                avg_losses = self._flush_losses()
                self.log(avg_losses, save=True)

    # ============================================================
    # Utilities
    # ============================================================

    def _save_training_meta(self, num_epochs: int):
        """Save training metadata at the start of training for later analysis."""
        dataset = self.dataloader.dataset
        data_info = {"total_images": len(dataset)}
        if hasattr(dataset, "A_paths"):
            data_info["trainA_images"] = len(dataset.A_paths)
        if hasattr(dataset, "B_paths"):
            data_info["trainB_images"] = len(dataset.B_paths)

        g_params = sum(p.numel() for p in self.model.generator_parameters())
        d_params = sum(p.numel() for p in self.model.discriminator_parameters()) if hasattr(self.model, "discriminator_parameters") else 0

        meta = {
            "model_name": self.model_name,
            "config": _make_serializable(asdict(self.model.cfg)) if hasattr(self.model, "cfg") else None,
            "training": {
                "num_epochs": num_epochs,
                "batch_size": self.dataloader.batch_size,
                "learning_rate": self.lr,
                "betas": list(self.betas),
                "amp": self.use_amp,
                "grad_accum_steps": self.grad_accum_steps,
            },
            "dataset": data_info,
            "parameters": {
                "generator": g_params,
                "discriminator": d_params,
                "total": g_params + d_params,
            },
        }

        meta_path = os.path.join(os.path.dirname(self.save_dir), "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Meta] Saved training metadata to {meta_path}")

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
            "config": asdict(self.model.cfg),
            "model_name": self.model_name,
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

    def _flush_losses(self) -> Dict[str, float]:
        """Average accumulated losses and reset the buffer."""
        if not self._epoch_losses:
            return {}
        keys = self._epoch_losses[0].keys()
        avg = {}
        for k in keys:
            vals = [d[k] for d in self._epoch_losses if k in d]
            avg[k] = sum(vals) / len(vals) if vals else 0.0
        self._epoch_losses = []
        return avg

    def log(self, logs: Dict[str, float], save: bool = False):
        """Print and optionally save losses to CSV."""
        msg = f"[E{self.epoch:03d} | S{self.global_step:06d}] "
        msg += " ".join([f"{k}:{v:.4f}" for k, v in logs.items()])
        print(msg)

        if save and logs:
            row = {"epoch": self.epoch, "global_step": self.global_step, **logs}
            fieldnames = list(row.keys())
            write_header = not self._log_header_written
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    self._log_header_written = True
                writer.writerow(row)


def _make_serializable(obj):
    """Convert tuples and other non-JSON types for json.dump."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj
