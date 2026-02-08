# models/miudiff.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utilities
# =========================

def to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def channel_entropy_proxy(y: torch.Tensor) -> torch.Tensor:
    var = y.var(dim=(2,3), unbiased=False) + 1e-6
    return torch.log(var).sum(dim=1).mean()


# =========================
# DDPM schedule
# =========================

@dataclass
class DiffusionSchedule:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def make(self, device):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
        return betas, alphas, alphas_cumprod, alphas_cumprod_prev


# =========================
# UNet eps-model (small/stable)
# =========================

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, c: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.tproj = nn.Linear(tdim, c)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.tproj(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, c): super().__init__(); self.conv = nn.Conv2d(c, c, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, c): super().__init__(); self.conv = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class EpsUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, tdim: int = 128):
        super().__init__()
        self.tdim = tdim
        self.temb = TimeEmbedding(tdim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.rb1 = ResBlock(base, tdim)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, tdim)
        self.down2 = Down(base)

        self.mid1 = ResBlock(base, tdim)
        self.mid2 = ResBlock(base, tdim)

        self.up2 = Up(base)
        self.rb_up2 = ResBlock(base, tdim)

        self.up1 = Up(base)
        self.rb_up1 = ResBlock(base, tdim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.temb(t)

        h = self.in_conv(x)
        h = self.rb1(h, te)
        h = self.down1(h)

        h = self.rb2(h, te)
        h = self.down2(h)

        h = self.mid1(h, te)
        h = self.mid2(h, te)

        h = self.up2(h)
        h = self.rb_up2(h, te)

        h = self.up1(h)
        h = self.rb_up1(h, te)

        return self.out_conv(F.silu(self.out_norm(h)))


# =========================
# MI estimator (MINE-style)
# =========================

class MIEstimator(nn.Module):
    def __init__(self, patch: int = 32, hidden: int = 256):
        super().__init__()
        in_dim = (1 + 3) * patch * patch
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 1),
        )

    def forward(self, g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = g.size(0)
        v = torch.cat([g, y], dim=1).reshape(b, -1)
        return self.net(v).squeeze(1)


def sample_patches(a: torch.Tensor, b: torch.Tensor, patch: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = a.shape
    ys = torch.randint(0, H - patch + 1, (B, n), device=a.device)
    xs = torch.randint(0, W - patch + 1, (B, n), device=a.device)

    aps, bps = [], []
    for i in range(B):
        for k in range(n):
            y0 = int(ys[i, k])
            x0 = int(xs[i, k])
            aps.append(a[i:i+1, :, y0:y0+patch, x0:x0+patch])
            bps.append(b[i:i+1, :, y0:y0+patch, x0:x0+patch])
    return torch.cat(aps, dim=0), torch.cat(bps, dim=0)


# =========================
# Patch-wise contrastive (PatchNCE-like)
# =========================

class SmallFeatNet(nn.Module):
    """
    Light feature extractor for contrastive:
      1ch -> C or 3ch -> C
    Produces feature map [B,C,H,W] at same resolution (no strides).
    """
    def __init__(self, in_ch: int, c: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )

    def forward(self, x):  # [B,in,H,W] -> [B,C,H,W]
        return self.net(x)

class PatchProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, dim=1)

def sample_hw_patches(feat: torch.Tensor, n_patches: int, patch_ids: Optional[torch.Tensor] = None):
    """
    feat: [B,C,H,W]
    returns:
      vecs: [B*n, C]
      ids : [B, n] indices into HW
    """
    B, C, H, W = feat.shape
    flat = feat.permute(0,2,3,1).reshape(B, H*W, C)  # [B,HW,C]
    HW = H * W
    K = min(n_patches, HW)

    if patch_ids is None:
        ids = torch.randint(0, HW, (B, K), device=feat.device)
    else:
        ids = patch_ids

    gathered = torch.gather(flat, 1, ids.unsqueeze(-1).expand(-1, -1, C))  # [B,K,C]
    return gathered.reshape(-1, C), ids

def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


# =========================
# MIU-Diff config
# =========================

@dataclass
class MIUDiffConfig:
    # diffusion
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # networks
    base_channels: int = 64
    tdim: int = 128

    # conditioning
    cond_channels: int = 1

    # MI estimator
    mi_patch: int = 32
    mi_patches_per_img: int = 8
    lambda_mi: float = 1.0

    # guidance
    guidance_scale: float = 1.0

    # stage
    stage: str = "pretrain"  # "pretrain" or "finetune"

    # sampling
    sample_steps: int = 200
    t0_prime: int = 40  # apply local refinement for t <= t0_prime

    # ---- Patch-wise contrastive (optional) ----
    miu_pcl: bool = False
    lambda_pcl: float = 0.1
    pcl_n_patches: int = 256
    pcl_proj_dim: int = 128
    pcl_temp: float = 0.07

    # inference-time refinement (optional)
    pcl_refine_steps: int = 0   # 0 disables
    pcl_refine_lr: float = 0.05


# =========================
# MIU-Diff model
# =========================

class MIUDiff(nn.Module):
    def __init__(self, cfg: MIUDiffConfig):
        super().__init__()
        self.cfg = cfg
        self.sched = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end)

        self.eps_uncond = EpsUNet(in_ch=3, base=cfg.base_channels, tdim=cfg.tdim)
        self.eps_cond = EpsUNet(in_ch=3 + cfg.cond_channels, base=cfg.base_channels, tdim=cfg.tdim)
        self.mi = MIEstimator(patch=cfg.mi_patch)

        # PCL modules (optional)
        if cfg.miu_pcl:
            # Compare structure features of x (grad) vs structure features of generated y (grad)
            self.feat_x = SmallFeatNet(in_ch=1, c=cfg.base_channels)
            self.feat_y = SmallFeatNet(in_ch=1, c=cfg.base_channels)
            self.proj = PatchProjector(in_dim=cfg.base_channels, proj_dim=cfg.pcl_proj_dim)
        else:
            self.feat_x = None
            self.feat_y = None
            self.proj = None

    # ---- BaseTrainer interface ----
    def generator_parameters(self):
        params = list(self.eps_uncond.parameters()) + list(self.eps_cond.parameters()) + list(self.mi.parameters())
        if self.cfg.miu_pcl:
            params += list(self.feat_x.parameters()) + list(self.feat_y.parameters()) + list(self.proj.parameters())
        return params

    def discriminator_parameters(self):
        return []

    def compute_discriminator_loss(self, batch, visuals):
        return torch.tensor(0.0, device=next(self.parameters()).device), {}

    # ---- q sample ----
    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
        a = a_bar[t_idx].view(-1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise

    # ---- MI lower bound ----
    def mi_lower_bound(self, g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gp, yp = sample_patches(g, y, patch=self.cfg.mi_patch, n=self.cfg.mi_patches_per_img)
        T_joint = self.mi(gp, yp)
        yp_shuf = yp[torch.randperm(yp.size(0))]
        T_marg = self.mi(gp, yp_shuf)
        return T_joint.mean() - torch.log(torch.exp(T_marg).mean() + 1e-8)

    # ---- energy ----
    def energy_M(self, y: torch.Tensor, x_struct: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        B = y.size(0)
        device = y.device
        betas, alphas, a_bar, _ = self.sched.make(device)

        t_idx = (t_frac * (self.cfg.T - 1)).long().clamp(0, self.cfg.T - 1)
        noise = torch.randn_like(x_struct)
        x_t = self.q_sample(x_struct, t_idx, noise, a_bar)

        g_xt = sobel_grad(x_t)
        I_xt_y = self.mi_lower_bound(g_xt, y)
        H_y = channel_entropy_proxy(y)
        U_y = H_y - I_xt_y

        lam = t_frac.mean().clamp(0, 1)
        return -lam * U_y - (1.0 - lam) * I_xt_y

    # ---- Patch-wise contrastive loss ----
    def pcl_loss(self, x_struct: torch.Tensor, y_img: torch.Tensor) -> torch.Tensor:
        """
        x_struct: [B,1,H,W] (gray or grad map)
        y_img:    [B,3,H,W] (predicted clean image or current sample)
        Uses grad maps so it focuses on structure.
        """
        assert self.cfg.miu_pcl and self.feat_x is not None

        gx = sobel_grad(x_struct)          # [B,1,H,W]
        gy = sobel_grad(to_gray(y_img))    # [B,1,H,W]

        fx = self.feat_x(gx)  # [B,C,H,W]
        fy = self.feat_y(gy)  # [B,C,H,W]

        q, ids = sample_hw_patches(fx, self.cfg.pcl_n_patches, patch_ids=None)
        k, _   = sample_hw_patches(fy, self.cfg.pcl_n_patches, patch_ids=ids)

        q = self.proj(q)
        k = self.proj(k)
        return info_nce(q, k, temperature=self.cfg.pcl_temp)

    # ---- training ----
    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        device = next(self.parameters()).device
        betas, alphas, a_bar, _ = self.sched.make(device)

        if self.cfg.stage == "pretrain":
            y0 = batch["B"].to(device)
            B = y0.size(0)

            t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
            t_frac = t_idx.float() / (self.cfg.T - 1)

            eps = torch.randn_like(y0)
            y_t = self.q_sample(y0, t_idx, eps, a_bar)

            eps_pred = self.eps_uncond(y_t, t_frac)
            loss_eps = F.mse_loss(eps_pred, eps)

            g_y = sobel_grad(to_gray(y0))
            loss_mi = -self.mi_lower_bound(g_y, y0)

            loss = loss_eps + self.cfg.lambda_mi * loss_mi

            logs = {
                "loss_G": float(loss.detach().cpu()),
                "loss_eps": float(loss_eps.detach().cpu()),
                "loss_mi": float(loss_mi.detach().cpu()),
            }
            visuals = {"real_B": y0, "noisy_B": y_t}
            return loss, logs, visuals

        # finetune
        xA = batch["A"].to(device)
        y0 = batch["B"].to(device)
        x_struct = to_gray(xA)  # [B,1,H,W]
        B = y0.size(0)

        t_idx = torch.randint(0, self.cfg.T, (B,), device=device)
        t_frac = t_idx.float() / (self.cfg.T - 1)

        eps = torch.randn_like(y0)
        y_t = self.q_sample(y0, t_idx, eps, a_bar)

        eps_pred = self.eps_cond(torch.cat([y_t, x_struct], dim=1), t_frac)
        loss_eps = F.mse_loss(eps_pred, eps)

        g_y = sobel_grad(to_gray(y0))
        loss_mi = -self.mi_lower_bound(g_y, y0)

        loss = loss_eps + self.cfg.lambda_mi * loss_mi

        # Optional: Patch-wise contrastive loss only for late timesteps (t <= t0_prime)
        loss_pcl = torch.tensor(0.0, device=device)
        if self.cfg.miu_pcl and self.cfg.lambda_pcl > 0:
            late_mask = (t_idx <= self.cfg.t0_prime)
            if late_mask.any():
                # predict x0 from y_t (one-step clean estimate)
                a_bar_t = a_bar[t_idx].view(-1, 1, 1, 1)
                x0_pred = (y_t - torch.sqrt(1.0 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

                loss_pcl = self.pcl_loss(x_struct[late_mask], x0_pred[late_mask])
                loss = loss + self.cfg.lambda_pcl * loss_pcl

        logs = {
            "loss_G": float(loss.detach().cpu()),
            "loss_eps": float(loss_eps.detach().cpu()),
            "loss_mi": float(loss_mi.detach().cpu()),
            "loss_pcl": float(loss_pcl.detach().cpu()),
        }
        visuals = {"real_A": xA, "real_B": y0, "noisy_B": y_t}
        return loss, logs, visuals

    # =========================
    # Sampling (A -> B) with optional PCL refinement
    # =========================

    @torch.no_grad()
    def sample_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = xA.device
        betas, alphas, a_bar, a_bar_prev = self.sched.make(device)

        B, _, H, W = xA.shape
        y = torch.randn(B, 3, H, W, device=device)
        x_struct = to_gray(xA)

        steps = self.cfg.sample_steps
        t_list = torch.linspace(self.cfg.T - 1, 0, steps, device=device).long()

        for t in t_list:
            t_idx = t.repeat(B)
            t_frac = t_idx.float() / (self.cfg.T - 1)

            eps = self.eps_cond(torch.cat([y, x_struct], dim=1), t_frac)

            beta_t = betas[t_idx].view(-1, 1, 1, 1)
            alpha_t = alphas[t_idx].view(-1, 1, 1, 1)
            a_bar_t = a_bar[t_idx].view(-1, 1, 1, 1)
            a_bar_prev_t = a_bar_prev[t_idx].view(-1, 1, 1, 1)

            x0_pred = (y - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)

            coef1 = torch.sqrt(a_bar_prev_t) * beta_t / (1 - a_bar_t + 1e-8)
            coef2 = torch.sqrt(alpha_t) * (1 - a_bar_prev_t) / (1 - a_bar_t + 1e-8)
            mu = coef1 * x0_pred + coef2 * y

            # MI guidance
            y_req = y.detach().requires_grad_(True)
            M = self.energy_M(y_req, x_struct, t_frac)
            grad = torch.autograd.grad(M, y_req, retain_graph=False, create_graph=False)[0]
            mu = mu - self.cfg.guidance_scale * grad

            # sample
            if t.item() > 0:
                z = torch.randn_like(y)
                sigma = torch.sqrt(beta_t)
                y = mu + sigma * z
            else:
                y = mu

            # Optional: late-stage PCL refinement (gradient descent on y)
            if self.cfg.miu_pcl and self.cfg.pcl_refine_steps > 0 and t.item() <= self.cfg.t0_prime:
                y = self._pcl_refine(y, x_struct)

        return y.clamp(-1, 1)

    def _pcl_refine(self, y: torch.Tensor, x_struct: torch.Tensor) -> torch.Tensor:
        """
        Runs a few GD steps on y to reduce patch contrastive loss vs x_struct.
        This is optional, and only used late in sampling.
        """
        if not (self.cfg.miu_pcl and self.feat_x is not None):
            return y

        y_ref = y.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([y_ref], lr=self.cfg.pcl_refine_lr)

        for _ in range(self.cfg.pcl_refine_steps):
            opt.zero_grad(set_to_none=True)
            loss = self.pcl_loss(x_struct, y_ref)
            loss.backward()
            opt.step()
            y_ref.data.clamp_(-1, 1)

        return y_ref.detach()
