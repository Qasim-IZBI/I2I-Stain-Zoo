# models/miudiff.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from base_models import info_nce, PatchSampler, PatchProjector


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

def _has_bad_weights(state_dict):
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():
                print(f"[BAD WEIGHTS] {k} has non-finite values.")
                return True
    return False

def init_miudiff_from_stage1(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if _has_bad_weights(sd):
        raise RuntimeError("Checkpoint contains NaN/Inf weights. Use an earlier stage-1 checkpoint.")

    # 1) Load what matches directly (mi and eps_uncond will match)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # 2) Initialize eps_cond from eps_uncond (most layers match)
    # Copy all eps_uncond params into eps_cond when shapes match.
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k.startswith("eps_uncond."):
            k2 = "eps_cond." + k[len("eps_uncond."):]
            if k2 in model_sd and model_sd[k2].shape == v.shape:
                model_sd[k2] = v.clone()

    # Special handling for eps_cond.in_conv.weight if it has extra channels
    # eps_uncond.in_conv.weight: [Cout, 3, 3, 3]
    # eps_cond.in_conv.weight:   [Cout, 4, 3, 3] (if cond_channels=1)
    w_un = sd.get("eps_uncond.input_conv.weight", None)
    if w_un is not None and "eps_cond.input_conv.weight" in model_sd:
        w_c = model_sd["eps_cond.input_conv.weight"]
        if w_c.shape[0] == w_un.shape[0] and w_c.shape[2:] == w_un.shape[2:] and w_un.shape[1] == 3:
            # copy RGB weights
            w_c[:, :3, :, :] = w_un
            # initialize extra cond channel to mean of RGB filters
            if w_c.shape[1] > 3:
                w_c[:, 3:, :, :] = w_un.mean(dim=1, keepdim=True).repeat(1, w_c.shape[1] - 3, 1, 1)
            model_sd["eps_cond.input_conv.weight"] = w_c

    model.load_state_dict(model_sd, strict=False)

    print("MIU-Diff init from stage1:")
    print("  missing keys:", len(missing))
    print("  unexpected keys:", len(unexpected))

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


# ============================================================
# DDPM / guided-diffusion style UNet (2D)
# - ResBlocks with time embedding
# - Attention at selected resolutions
# - Down/Up sampling with conv
# ============================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    t: [B] float in [0,1] (we will scale to [0, max_period] internally)
    returns: [B, dim]
    """
    # Scale continuous t into "timesteps" space
    # (works fine for both discrete or continuous time)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = (t.float().unsqueeze(1) * max_period) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ZeroModule(nn.Module):
    """Wraps a module and initializes its weights to zero (stable residual starts)."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        return self.module(x)


class AttentionBlock(nn.Module):
    """
    Self-attention over spatial positions for [B, C, H, W].
    This is the classic diffusion attention block (single-head or multi-head).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = GroupNorm32(32 if channels >= 32 else 8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = ZeroModule(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # [B,C,HW]
        qkv = self.qkv(h)                   # [B,3C,HW]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # reshape heads: [B, heads, C//heads, HW]
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale  # [B,heads,HW,HW]
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)               # [B,heads,head_dim,HW]
        out = out.reshape(B, C, H * W)                               # [B,C,HW]
        out = self.proj_out(out).view(B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    """
    WideResNet-style ResBlock with time embedding conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        gn_groups_in = 32 if in_channels >= 32 else 8
        gn_groups_out = 32 if out_channels >= 32 else 8

        self.norm1 = GroupNorm32(gn_groups_in, in_channels)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = GroupNorm32(gn_groups_out, out_channels)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ZeroModule(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            if use_conv_shortcut:
                self.skip = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


@dataclass
class UNetConfig:
    in_channels: int
    out_channels: int = 3

    # “model_channels” in DDPM/OpenAI code
    base_channels: int = 64

    # for 256x256, DDPM uses 6 resolutions (256→128→64→32→16→8)
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)

    num_res_blocks: int = 2
    dropout: float = 0.0

    # attention at given downsample rates (1 means 256x256, 2 means 128x128, 16 means 16x16)
    attention_resolutions: Tuple[int, ...] = (16,)
    num_heads: int = 4

    time_emb_mult: int = 4  # time embedding dim = base_channels * time_emb_mult


class DDPMUNet(nn.Module):
    """
    DDPM-style UNet backbone (Ho et al. / OpenAI guided-diffusion).
    Forward signature matches your eps-model: eps_theta(x, t_frac).

    - x: [B, in_channels, H, W]
    - t_frac: [B] float in [0,1]
    returns: [B, out_channels, H, W]
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels

        time_dim = cfg.base_channels * cfg.time_emb_mult
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.base_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        ch = cfg.base_channels
        ds = 1  # downsample rate relative to input
        for level, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch, num_heads=cfg.num_heads))
                self.skip_channels.append(ch)

            # downsample except last level
            if level != len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                ds *= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim, dropout=cfg.dropout),
            AttentionBlock(ch, num_heads=cfg.num_heads) if (ds in cfg.attention_resolutions) else nn.Identity(),
            ResBlock(ch, ch, time_dim, dropout=cfg.dropout),
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout=cfg.dropout))
                ch = out_ch
                if ds in cfg.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch, num_heads=cfg.num_heads))

            if level != 0:
                self.upsamples.append(Upsample(ch))
                ds //= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = GroupNorm32(32 if ch >= 32 else 8, ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        # time embedding
        t_emb = timestep_embedding(t_frac, self.cfg.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)
        hs: List[torch.Tensor] = []

        # Encoder
        di = 0
        for level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[di](h, t_emb); di += 1
                # optional attention block right after resblock
                if di < len(self.down_blocks) and isinstance(self.down_blocks[di], AttentionBlock):
                    h = self.down_blocks[di](h); di += 1
                hs.append(h)
            h = self.downsamples[level](h)

        # Middle
        h = self.mid[0](h, t_emb)
        h = self.mid[1](h) if not isinstance(self.mid[1], nn.Identity) else h
        h = self.mid[2](h, t_emb)

        # Decoder
        ui = 0
        for level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[ui](h, t_emb); ui += 1
                if ui < len(self.up_blocks) and isinstance(self.up_blocks[ui], AttentionBlock):
                    h = self.up_blocks[ui](h); ui += 1
            h = self.upsamples[level](h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

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
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)  # 4 levels: 64→128→128→256
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

    # MI estimator learning rate (trained with its own optimizer)
    mi_lr: float = 1e-4


# =========================
# MIU-Diff model
# =========================

class MIUDiff(nn.Module):
    def __init__(self, cfg: MIUDiffConfig):
        super().__init__()
        self.cfg = cfg
        self.sched = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end)

        # self.eps_uncond = EpsUNet(in_ch=3, base=cfg.base_channels, tdim=cfg.tdim)
        # self.eps_cond = EpsUNet(in_ch=3 + cfg.cond_channels, base=cfg.base_channels, tdim=cfg.tdim)
        self.eps_uncond = DDPMUNet(UNetConfig(in_channels=3, base_channels=cfg.base_channels, channel_mult=cfg.channel_mult))
        self.eps_cond   = DDPMUNet(UNetConfig(in_channels=3 + cfg.cond_channels, base_channels=cfg.base_channels, channel_mult=cfg.channel_mult))
        self.mi = MIEstimator(patch=cfg.mi_patch)

        # MI estimator gets its own optimizer to prevent its large/volatile
        # gradients from starving the eps network via shared gradient clipping.
        self._mi_opt = torch.optim.Adam(self.mi.parameters(), lr=cfg.mi_lr)

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
        # MI estimator is excluded — it is trained with its own optimizer
        # inside compute_generator_loss to avoid gradient clipping interference.
        params = list(self.eps_uncond.parameters()) + list(self.eps_cond.parameters())
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

        # If anything is non-finite, skip MI for this batch
        if (not torch.isfinite(gp).all()) or (not torch.isfinite(yp).all()):
            return gp.new_tensor(0.0)

        T_joint = self.mi(gp, yp)
        yp_shuf = yp[torch.randperm(yp.size(0))]
        T_marg = self.mi(gp, yp_shuf)

        # Clamp BOTH to prevent runaway -> inf
        T_joint = T_joint.clamp(-20, 20)
        T_marg  = T_marg.clamp(-20, 20)

        # log(mean(exp(T_marg))) computed stably
        logN = T_marg.new_tensor(T_marg.numel()).log()
        log_mean_exp = torch.logsumexp(T_marg, dim=0) - logN

        out = T_joint.mean() - log_mean_exp
        if not torch.isfinite(out):
            return out.new_tensor(0.0)
        return out
    
    # def mi_lower_bound(self, g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     gp, yp = sample_patches(g, y, patch=self.cfg.mi_patch, n=self.cfg.mi_patches_per_img)

    #     T_joint = self.mi(gp, yp)
    #     yp_shuf = yp[torch.randperm(yp.size(0))]
    #     T_marg = self.mi(gp, yp_shuf)

    #     # Clamp BOTH to prevent runaway -> inf
    #     T_joint = T_joint.clamp(-20, 20)
    #     T_marg  = T_marg.clamp(-20, 20)

    #     # log(mean(exp(T_marg))) computed stably
    #     logN = T_marg.new_tensor(T_marg.numel()).log()
    #     log_mean_exp = torch.logsumexp(T_marg, dim=0) - logN

    #     return T_joint.mean() - log_mean_exp

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
        gx = sobel_grad(x_struct)          # depends only on x_struct # [B,1,H,W]
        gy = sobel_grad(to_gray(y_img))    # MUST depend on y_img # [B,1,H,W]

        fx = self.feat_x(gx) # [B,C,H,W]
        fy = self.feat_y(gy) # [B,C,H,W]

        q, ids = PatchSampler.sample(fx, self.cfg.pcl_n_patches, patch_ids=None)
        k, _   = PatchSampler.sample(fy, self.cfg.pcl_n_patches, patch_ids=ids)

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

            # Train MI estimator with its own optimizer (separate from eps network)
            self._mi_opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                y0_f = y0.float().clamp(-1, 1)
                g_y = sobel_grad(to_gray(y0_f)).clamp(0, 5)
                mi_lb = self.mi_lower_bound(g_y, y0_f)
                loss_mi = -mi_lb

            if torch.isfinite(loss_mi):
                loss_mi.backward()
                torch.nn.utils.clip_grad_norm_(self.mi.parameters(), 1.0)
                self._mi_opt.step()
            else:
                loss_mi = loss_eps.new_tensor(0.0)

            # Only eps loss goes through the BaseTrainer's optimizer
            loss = loss_eps

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

        # Train MI estimator with its own optimizer (separate from eps network)
        self._mi_opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            y0_f = y0.float().clamp(-1, 1)
            g_y = sobel_grad(to_gray(y0_f)).clamp(0, 5)
            mi_lb = self.mi_lower_bound(g_y, y0_f)
            loss_mi = -mi_lb

        if torch.isfinite(loss_mi):
            loss_mi.backward()
            torch.nn.utils.clip_grad_norm_(self.mi.parameters(), 1.0)
            self._mi_opt.step()
        else:
            loss_mi = loss_eps.new_tensor(0.0)

        # Only eps loss (+ optional PCL) goes through the BaseTrainer's optimizer
        loss = loss_eps

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
    def sample_A2B(self, xA: torch.Tensor) -> torch.Tensor:
        """
        DDIM-style sampling over an arbitrary timestep schedule (works when sample_steps << T),
        with optional MI guidance and optional late-stage PCL refinement.

        Key fix vs your previous version:
        - Uses t_next from the strided schedule (NOT a_bar_prev[t]) to update.
            The old posterior formula assumes consecutive timesteps (t-1). When you stride,
            that breaks and you get pure noise.
        """
        self.eval()
        device = xA.device
        betas, alphas, a_bar, _ = self.sched.make(device)

        B, _, H, W = xA.shape
        y = torch.randn(B, 3, H, W, device=device)
        x_struct = to_gray(xA)  # [B,1,H,W]

        steps = int(self.cfg.sample_steps)

        # Build a strictly descending, unique timestep list
        t_list = torch.linspace(self.cfg.T - 1, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)  # removes duplicates from rounding
        if t_list.numel() < 2:
            # fallback to full schedule if steps too small
            t_list = torch.arange(self.cfg.T - 1, -1, -1, device=device)

        # DDIM sampling (eta=0 deterministic)
        for i in range(t_list.numel() - 1):
            t = t_list[i]
            t_next = t_list[i + 1]

            # time fraction for conditioning network
            t_idx = t.repeat(B)
            t_frac = t_idx.float() / (self.cfg.T - 1)

            # ---- predict eps at time t ----
            with torch.no_grad():
                eps = self.eps_cond(torch.cat([y, x_struct], dim=1), t_frac)

                a_bar_t = a_bar[t].view(1, 1, 1, 1)
                a_bar_next = a_bar[t_next].view(1, 1, 1, 1)

                # predict x0
                x0_pred = (y - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

            # ---- MI guidance (optional) ----
            # More stable to apply the guidance as a tweak to eps (instead of shifting mu of a wrong posterior)
            if getattr(self.cfg, "guidance_scale", 0.0) and self.cfg.guidance_scale != 0.0:
                with torch.enable_grad():
                    y_req = y.detach().requires_grad_(True)
                    M = self.energy_M(y_req, x_struct.detach(), t_frac)
                    # If this assert fails, energy_M is detaching y internally.
                    assert M.requires_grad, "M does not require grad - guidance will fail."
                    grad = torch.autograd.grad(
                        M, y_req,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                # apply guidance to eps and recompute x0_pred for consistency
                eps = (eps + self.cfg.guidance_scale * grad).detach()
                x0_pred = (y - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

            # ---- DDIM update: y_{t_next} ----
            with torch.no_grad():
                y = torch.sqrt(a_bar_next) * x0_pred + torch.sqrt(1.0 - a_bar_next) * eps

            # ---- Optional late-stage PCL refinement (uses grads internally) ----
            if self.cfg.miu_pcl and self.cfg.pcl_refine_steps > 0 and int(t_next.item()) <= int(self.cfg.t0_prime):
                y = self._pcl_refine(y, x_struct)

        # Final clamp
        return y.clamp(-1, 1)

    # =========================
    # Sampling — stage 1 (unconditional)
    # =========================
    def sample_uncond(self, batch_size: int, image_size: int = 256) -> torch.Tensor:
        """
        Pure DDIM sampling using only eps_uncond (stage 1 / pretrain).

        No source image, no MI guidance, no PCL refinement — just DDPM reverse
        diffusion on the target domain B manifold.  Identical timestep schedule
        to sample_A2B so --miu_steps controls the number of denoising steps here
        too (read from self.cfg.sample_steps).

        Returns: [batch_size, 3, image_size, image_size] in [-1, 1].
        """
        self.eval()
        device = next(self.eps_uncond.parameters()).device
        betas, alphas, a_bar, _ = self.sched.make(device)

        y = torch.randn(batch_size, 3, image_size, image_size, device=device)

        steps = int(self.cfg.sample_steps)
        t_list = torch.linspace(self.cfg.T - 1, 0, steps, device=device).long()
        t_list = torch.unique_consecutive(t_list)
        if t_list.numel() < 2:
            t_list = torch.arange(self.cfg.T - 1, -1, -1, device=device)

        with torch.no_grad():
            for i in range(t_list.numel() - 1):
                t      = t_list[i]
                t_next = t_list[i + 1]

                t_frac = t.repeat(batch_size).float() / (self.cfg.T - 1)

                eps = self.eps_uncond(y, t_frac)

                a_bar_t    = a_bar[t].view(1, 1, 1, 1)
                a_bar_next = a_bar[t_next].view(1, 1, 1, 1)

                x0_pred = (y - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
                x0_pred = x0_pred.clamp(-1, 1)

                y = torch.sqrt(a_bar_next) * x0_pred + torch.sqrt(1.0 - a_bar_next) * eps

        return y.clamp(-1, 1)

    # # =========================
    # # Sampling (A -> B) with optional PCL refinement
    # # =========================
    # def sample_A2B(self, xA: torch.Tensor) -> torch.Tensor:
    #     self.eval()
    #     device = xA.device
    #     betas, alphas, a_bar, a_bar_prev = self.sched.make(device)

    #     B, _, H, W = xA.shape
    #     y = torch.randn(B, 3, H, W, device=device)
    #     x_struct = to_gray(xA)

    #     steps = self.cfg.sample_steps
    #     t_list = torch.linspace(self.cfg.T - 1, 0, steps, device=device).long()

    #     for t in t_list:
    #         t_idx = t.repeat(B)
    #         t_frac = t_idx.float() / (self.cfg.T - 1)

    #         # Everything except MI guidance can be no_grad
    #         with torch.no_grad():
    #             eps = self.eps_cond(torch.cat([y, x_struct], dim=1), t_frac)

    #             beta_t = betas[t_idx].view(-1, 1, 1, 1)
    #             alpha_t = alphas[t_idx].view(-1, 1, 1, 1)
    #             a_bar_t = a_bar[t_idx].view(-1, 1, 1, 1)
    #             a_bar_prev_t = a_bar_prev[t_idx].view(-1, 1, 1, 1)

    #             x0_pred = (y - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
    #             x0_pred = x0_pred.clamp(-1, 1)

    #             coef1 = torch.sqrt(a_bar_prev_t) * beta_t / (1 - a_bar_t + 1e-8)
    #             coef2 = torch.sqrt(alpha_t) * (1 - a_bar_prev_t) / (1 - a_bar_t + 1e-8)
    #             mu = coef1 * x0_pred + coef2 * y

    #         # MI guidance MUST run with grad enabled
    #         with torch.enable_grad():
    #             y_req = y.detach().requires_grad_(True)
    #             # ensure x_struct does not require grad (it’s a condition)
    #             x_struct_ng = x_struct.detach()

    #             M = self.energy_M(y_req, x_struct_ng, t_frac)
    #             assert M.requires_grad, "M does not require grad - guidance will fail."

    #             grad = torch.autograd.grad(
    #                 M, y_req,
    #                 retain_graph=False,
    #                 create_graph=False,
    #                 allow_unused=False,
    #             )[0]

    #         mu = mu - self.cfg.guidance_scale * grad

    #         # sample step (no grad)
    #         with torch.no_grad():
    #             if t.item() > 0:
    #                 z = torch.randn_like(y)
    #                 sigma = torch.sqrt(beta_t)
    #                 y = mu + sigma * z
    #             else:
    #                 y = mu

    #             # Optional late-stage PCL refinement (this uses grads internally)
    #             # so do NOT put it under no_grad
    #         if self.cfg.miu_pcl and self.cfg.pcl_refine_steps > 0 and t.item() <= self.cfg.t0_prime:
    #             y = self._pcl_refine(y, x_struct)

    #     return y.clamp(-1, 1)
        
    def _pcl_refine(self, y: torch.Tensor, x_struct: torch.Tensor) -> torch.Tensor:
        """
        Runs a few GD steps on y to reduce patch contrastive loss vs x_struct.
        Ensures gradients are enabled even if caller is inside no_grad.
        """
        """
        Gradient-descent refinement on y to reduce PCL loss vs x_struct.
        Robust to being called from inside no_grad contexts.
        """
        if not (self.cfg.miu_pcl and self.feat_x is not None):
            return y

        x_struct_ng = x_struct.detach()

        with torch.enable_grad():
            y_ref = y.detach().clone().requires_grad_(True)

            for _ in range(self.cfg.pcl_refine_steps):
                # IMPORTANT: recompute loss each step on the current y_ref
                loss = self.pcl_loss(x_struct_ng, y_ref)

                # sanity: if this fails, pcl_loss isn't connected to y_ref
                if not loss.requires_grad:
                    raise RuntimeError(
                        "PCL loss does not require grad. "
                        "This means pcl_loss is not connected to y_ref. "
                        "Check for accidental .detach() / no_grad in pcl_loss."
                    )
                print("loss.requires_grad:", loss.requires_grad, "y_ref.requires_grad:", y_ref.requires_grad)

                # compute d(loss)/d(y_ref)
                grad = torch.autograd.grad(
                    loss, y_ref,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                print("pcl loss:", float(loss.detach().cpu()), "grad mean:", float(grad.abs().mean().detach().cpu()))
                # GD update (manual)
                y_ref = (y_ref - self.cfg.pcl_refine_lr * grad).clamp(-1, 1).detach().requires_grad_(True)

        return y_ref.detach()