# trainer/base_trainer.py
from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import asdict

import torch
from torch import nn, optim
from torchvision.utils import save_image
from typing import Dict, Any, List, Optional

from i2i_stain_zoo.utils import make_serializable

# matplotlib is optional — only imported when plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


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
        save_steps: int = 250_000,
        log_steps: int = 1_000,
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
        self.save_steps = save_steps
        self.log_steps = log_steps

        # --- optimizers ---
        g_params = list(self.model.generator_parameters())
        d_params = list(self.model.discriminator_parameters()) if hasattr(self.model, "discriminator_parameters") else []

        self.opt_G = torch.optim.Adam(g_params, lr=lr, betas=betas)

        self.opt_D = None
        if len(d_params) > 0:
            self.opt_D = torch.optim.Adam(d_params, lr=lr, betas=betas)

        # --- AMP ---
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp, growth_interval=500)

        self.global_step = 0
        self._internal_epoch = 0  # counts full passes through the dataloader (internal only)

        # --- training metadata ---
        self.lr = lr
        self.betas = betas

        # --- timing ---
        self.accumulated_training_seconds: float = 0.0
        self._session_start: Optional[float] = None
        self._log_interval_start: Optional[float] = None

        # --- loss logging ---
        self.log_path = os.path.join(os.path.dirname(save_dir), "loss_log.csv")
        self._log_header_written = os.path.exists(self.log_path)
        self._step_losses: List[Dict[str, float]] = []

        # --- loss plots ---
        self.plot_dir = os.path.join(os.path.dirname(save_dir), "loss_plots")

    # ============================================================
    # Core training loop
    # ============================================================

    def train(self, total_steps: int):
        self.resume_if_exists()

        if self.global_step >= total_steps:
            print(f"[Trainer] Already completed {self.global_step}/{total_steps} steps — nothing to do.")
            return

        if self.global_step == 0:
            self._save_training_meta(total_steps)
        else:
            print(f"[Trainer] Resuming from step {self.global_step} → target {total_steps} steps.")

        self._session_start = time.time()
        self._log_interval_start = time.time()

        self.model.train()

        while self.global_step < total_steps:
            self._internal_epoch += 1
            for batch in self.dataloader:
                if self.global_step >= total_steps:
                    break

                self.global_step += 1
                batch = self._to_device(batch)

                # -------------------------
                # Generator step
                # -------------------------
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss_G, logs_G, visuals = self.model.compute_generator_loss(batch)

                if not torch.isfinite(loss_G):
                    print(f"[WARN] Non-finite loss_G at step {self.global_step}, skipping step.")
                    self.opt_G.zero_grad(set_to_none=True)
                    continue

                self.scaler.scale(loss_G / self.grad_accum_steps).backward()

                if self.global_step % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.opt_G)

                    found_inf = False
                    for p in self.model.generator_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            found_inf = True
                            break

                    if found_inf:
                        print(f"[WARN] Non-finite gradients at step {self.global_step}, skipping optimizer step.")
                        self.opt_G.zero_grad(set_to_none=True)
                        self.scaler.update()
                        continue

                    torch.nn.utils.clip_grad_norm_(self.model.generator_parameters(), 1.0)
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
                # Accumulate losses
                # -------------------------
                self._step_losses.append({**logs_G, **logs_D})

                # -------------------------
                # Log every log_steps
                # -------------------------
                if self.global_step % self.log_steps == 0:
                    self.save_samples(visuals)
                    avg_losses = self._flush_losses()
                    self.log(avg_losses, save=True)
                    self._plot_losses()
                    self.save_checkpoint("step_latest.pt")

                # -------------------------
                # Checkpoint every save_steps
                # -------------------------
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

        # Save final state if the last step wasn't already a checkpoint boundary
        if self.global_step % self.save_steps != 0:
            self.save_checkpoint(f"step_{self.global_step}.pt")

        self._update_timing_meta(total_steps)

    # ============================================================
    # Utilities
    # ============================================================

    def _save_training_meta(self, total_steps: int):
        """Save training metadata at the start of a fresh training run."""
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
            "config": make_serializable(asdict(self.model.cfg)) if hasattr(self.model, "cfg") else None,
            "training": {
                "total_steps": total_steps,
                "batch_size": self.dataloader.batch_size,
                "learning_rate": self.lr,
                "betas": list(self.betas),
                "amp": self.use_amp,
                "grad_accum_steps": self.grad_accum_steps,
                "save_steps": self.save_steps,
                "log_steps": self.log_steps,
            },
            "dataset": data_info,
            "parameters": {
                "generator": g_params,
                "discriminator": d_params,
                "total": g_params + d_params,
            },
            "timing": {
                "accumulated_seconds": 0.0,
                "human_readable": "0h 00m 00s",
                "last_updated_step": 0,
                "latest_checkpoint_step": 0,
                "avg_seconds_per_1k_steps": None,
            },
        }

        meta_path = os.path.join(os.path.dirname(self.save_dir), "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Meta] Saved training metadata to {meta_path}")

    def _update_timing_meta(self, total_steps: Optional[int]):
        """Update the timing block in training_meta.json with current elapsed time."""
        meta_path = os.path.join(os.path.dirname(self.save_dir), "training_meta.json")
        if not os.path.exists(meta_path):
            return

        session_elapsed = time.time() - self._session_start if self._session_start else 0.0
        total_seconds = self.accumulated_training_seconds + session_elapsed

        steps_done = self.global_step
        avg = (total_seconds / steps_done * 1000) if steps_done > 0 else None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            return

        meta["timing"] = {
            "accumulated_seconds": round(total_seconds, 2),
            "human_readable": _seconds_to_hms(total_seconds),
            "last_updated_step": steps_done,
            "latest_checkpoint_step": steps_done,
            "avg_seconds_per_1k_steps": round(avg, 2) if avg is not None else None,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

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
        session_elapsed = time.time() - self._session_start if self._session_start else 0.0
        accumulated = self.accumulated_training_seconds + session_elapsed
        state = {
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "config": asdict(self.model.cfg),
            "model_name": self.model_name,
            "accumulated_training_seconds": accumulated,
        }
        if self.opt_D is not None:
            state["opt_D"] = self.opt_D.state_dict()
        if hasattr(self.model, "aux_optimizers"):
            for k, opt in self.model.aux_optimizers.items():
                state[f"aux_opt_{k}"] = opt.state_dict()
        torch.save(state, path)
        print(f"[Checkpoint] Saved: {path}")
        self._update_timing_meta(None)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        if self.opt_D is not None and "opt_D" in ckpt:
            self.opt_D.load_state_dict(ckpt["opt_D"])
        if hasattr(self.model, "aux_optimizers"):
            for k, opt in self.model.aux_optimizers.items():
                key = f"aux_opt_{k}"
                if key in ckpt:
                    opt.load_state_dict(ckpt[key])
        self.global_step = ckpt.get("global_step", 0)
        self.accumulated_training_seconds = ckpt.get("accumulated_training_seconds", 0.0)
        print(f"[Checkpoint] Loaded: {path}  (step {self.global_step})")

    def resume_if_exists(self):
        """Scan save_dir for checkpoints and resume from the furthest-ahead one.

        Considers both numbered step_N.pt files and the rolling step_latest.pt.
        If step_latest.pt is ahead of the highest numbered checkpoint (e.g. training
        crashed between two save_steps boundaries), it is preferred.  When tied,
        the numbered checkpoint wins as it is permanent.
        """
        pattern = re.compile(r"^step_(\d+)\.pt$")
        best_num_step, best_num_fname = -1, None
        for fname in os.listdir(self.save_dir):
            m = pattern.match(fname)
            if m:
                n = int(m.group(1))
                if n > best_num_step:
                    best_num_step, best_num_fname = n, fname

        latest_path = os.path.join(self.save_dir, "step_latest.pt")
        latest_step = -1
        if os.path.exists(latest_path):
            try:
                tmp = torch.load(latest_path, map_location="cpu")
                latest_step = tmp.get("global_step", -1)
            except Exception:
                pass

        if best_num_step == -1 and latest_step == -1:
            return

        if latest_step > best_num_step:
            chosen_path = latest_path
            chosen_name = "step_latest.pt"
        else:
            chosen_path = os.path.join(self.save_dir, best_num_fname)
            chosen_name = best_num_fname

        print(f"[Trainer] Found existing checkpoint: {chosen_name}")
        try:
            ckpt = torch.load(chosen_path, map_location=self.device)
        except Exception as e:
            print(f"[Trainer] Could not load {chosen_name} ({e}), starting fresh.")
            return

        # Config mismatch check — catches accidental reuse of an output directory
        # with different architecture or training hyperparameters.
        if "config" in ckpt and hasattr(self.model, "cfg"):
            saved_cfg = ckpt["config"]
            current_cfg = asdict(self.model.cfg)
            mismatches = {
                k: (saved_cfg.get(k), current_cfg.get(k))
                for k in set(saved_cfg) | set(current_cfg)
                if saved_cfg.get(k) != current_cfg.get(k)
            }
            if mismatches:
                lines = "\n".join(
                    f"  {k}: checkpoint={v[0]!r}  current={v[1]!r}"
                    for k, v in sorted(mismatches.items())
                )
                raise ValueError(
                    f"[Trainer] Config mismatch between checkpoint '{chosen_name}' "
                    f"and current run — refusing to resume.\n"
                    f"Differing keys:\n{lines}\n"
                    f"To start fresh in a new directory, change --output. "
                    f"To intentionally load these weights, use --init_ckpt."
                )
        elif "config" not in ckpt:
            print(f"[Trainer] Warning: checkpoint '{chosen_name}' has no config — skipping mismatch check.")

        self.model.load_state_dict(ckpt["model"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        if self.opt_D is not None and "opt_D" in ckpt:
            self.opt_D.load_state_dict(ckpt["opt_D"])
        if hasattr(self.model, "aux_optimizers"):
            for k, opt in self.model.aux_optimizers.items():
                key = f"aux_opt_{k}"
                if key in ckpt:
                    opt.load_state_dict(ckpt[key])
        self.global_step = ckpt.get("global_step", 0)
        self.accumulated_training_seconds = ckpt.get("accumulated_training_seconds", 0.0)
        print(f"[Checkpoint] Loaded: {chosen_path}  (step {self.global_step})")

    def _flush_losses(self) -> Dict[str, float]:
        """Average accumulated losses and reset the buffer."""
        if not self._step_losses:
            return {}
        keys = self._step_losses[0].keys()
        avg = {}
        for k in keys:
            vals = [d[k] for d in self._step_losses if k in d]
            avg[k] = sum(vals) / len(vals) if vals else 0.0
        self._step_losses = []
        return avg

    def _plot_losses(self):
        """Regenerate per-loss and combined loss plots from loss_log.csv."""
        if not _PLOT_AVAILABLE or not os.path.exists(self.log_path):
            return
        try:
            df = pd.read_csv(self.log_path)
        except Exception:
            return
        if df.empty or "global_step" not in df.columns:
            return

        loss_cols = [c for c in df.columns if c not in ("global_step", "elapsed_s")]
        if not loss_cols:
            return

        os.makedirs(self.plot_dir, exist_ok=True)
        x = df["global_step"]

        # Individual plot per loss
        for col in loss_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, df[col], linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel(col)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_dir, f"{col}.png"), dpi=120)
            plt.close(fig)

        # Combined plot (all losses, one subplot each)
        n = len(loss_cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
        for idx, col in enumerate(loss_cols):
            ax = axes[idx, 0]
            ax.plot(x, df[col], linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel(col)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "all_losses.png"), dpi=120)
        plt.close(fig)

    def log(self, logs: Dict[str, float], save: bool = False):
        """Print losses with elapsed time since the last log, and optionally save to CSV."""
        now = time.time()
        elapsed = now - self._log_interval_start if self._log_interval_start else 0.0
        self._log_interval_start = now

        msg = f"[S{self.global_step:08d} | {elapsed:6.1f}s] "
        msg += " ".join([f"{k}:{v:.4f}" for k, v in logs.items()])
        print(msg)

        if save and logs:
            row = {"global_step": self.global_step, "elapsed_s": f"{elapsed:.1f}", **logs}
            fieldnames = list(row.keys())
            write_header = not self._log_header_written
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    self._log_header_written = True
                writer.writerow(row)


def _seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m:02d}m {s:02d}s"
