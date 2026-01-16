# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import random
import torch
import torch.nn as nn


# ============================================================
# Init helpers
# ============================================================

def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights."""
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(f"init method {init_type} not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net: nn.Module, device: torch.device, init_type: str = "normal", init_gain: float = 0.02) -> nn.Module:
    net.to(device)
    init_weights(net, init_type=init_type, init_gain=init_gain)
    return net


# ============================================================
# Replay buffer (used by CycleGAN; also handy for others)
# ============================================================

class ImagePool:
    """History of generated images to stabilize discriminator training."""
    def __init__(self, pool_size: int = 50):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: tensor [B, C, H, W]
        returns: tensor [B, C, H, W]
        """
        if self.pool_size <= 0:
            return images

        out: List[torch.Tensor] = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(img)
                out.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = img
                    out.append(old)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


# ============================================================
# Losses
# ============================================================

class GANLoss(nn.Module):
    """
    LSGAN by default (MSE to 1/0). If you want vanilla GAN, swap to BCEWithLogitsLoss.
    """
    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("mode must be 'lsgan' or 'vanilla'")

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.mode == "lsgan":
            return torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        # vanilla:
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        tgt = self._target(pred, is_real)
        return self.loss(pred, tgt)


# # ============================================================
# # Building blocks
# # ============================================================

# class ResnetBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         padding_type: str = "reflect",
#         norm_layer: nn.Module = nn.InstanceNorm2d,
#         use_dropout: bool = False,
#     ):
#         super().__init__()
#         p = 0
#         if padding_type == "reflect":
#             pad1 = nn.ReflectionPad2d(1)
#         elif padding_type == "replicate":
#             pad1 = nn.ReplicationPad2d(1)
#         elif padding_type == "zero":
#             pad1 = nn.Identity()
#             p = 1
#         else:
#             raise NotImplementedError(f"padding [{padding_type}] is not implemented")

#         block: List[nn.Module] = [
#             pad1,
#             nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
#             norm_layer(dim),
#             nn.ReLU(True),
#         ]
#         if use_dropout:
#             block += [nn.Dropout(0.5)]
#         block += [
#             pad1,
#             nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
#             norm_layer(dim),
#         ]
#         self.block = nn.Sequential(*block)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.block(x)


# ============================================================
# Building blocks
# ============================================================

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer: nn.Module = nn.InstanceNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        p = 0
        if padding_type == "reflect":
            pad = nn.ReflectionPad2d(1)
        elif padding_type == "replicate":
            pad = nn.ReplicationPad2d(1)
        elif padding_type == "zero":
            pad = nn.Identity()
            p = 1
        else:
            raise NotImplementedError(padding_type)

        layers: List[nn.Module] = [
            pad,
            nn.Conv2d(dim, dim, 3, padding=p, bias=True),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [
            pad,
            nn.Conv2d(dim, dim, 3, padding=p, bias=True),
            norm_layer(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):
    """
    Encoder trunk (CycleGAN-style) WITHOUT ResBlocks.
    Typical for 256x256 input:
      - stem (k7)
      - 2 downsamples (stride 2)
    Output: [B, ngf*4, H/4, W/4]
    """
    def __init__(
        self,
        input_nc: int,
        ngf: int = 64,
        n_down: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_layers = feature_layers if feature_layers is not None else []

        layers: List[nn.Module] = []
        # Stem
        layers += [
            nn.ReflectionPad2d(3),                              # idx 0
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=True),  # idx 1
            norm_layer(ngf),                                    # idx 2
            nn.ReLU(True),                                      # idx 3
        ]

        # Downsamples
        mult = 1
        for _ in range(n_down):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        self.layers = nn.ModuleList(layers)
        self.out_channels = ngf * mult
        self.out_stride = 2 ** n_down  # spatial reduction factor

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.return_features and (i in self.feature_layers):
                feats[f"enc_layer_{i}"] = h
        return (h, feats) if self.return_features else h


# ============================================================
# Bottle neck
# ============================================================

class ResnetBottleneck(nn.Module):
    """
    The ONLY module containing ResnetBlocks.
    Keeps shape: [B, C, H, W] -> [B, C, H, W]
    This is what you can share across domains for UNIT.
    """
    def __init__(
        self,
        channels: int,
        n_blocks: int = 9,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_blocks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_blocks = feature_blocks if feature_blocks is not None else []

        self.blocks = nn.ModuleList(
            [ResnetBlock(channels, padding_type="reflect", norm_layer=norm_layer) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.return_features and (i in self.feature_blocks):
                feats[f"bottleneck_block_{i}"] = h
        return (h, feats) if self.return_features else h


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    """
    Decoder trunk (CycleGAN-style) WITHOUT ResBlocks.
    Typical:
      - 2 upsample (ConvTranspose)
      - output head (k7 + tanh)
    """
    def __init__(
        self,
        in_channels: int,
        output_nc: int,
        ngf: int = 64,
        n_up: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        return_features: bool = False,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.return_features = return_features
        self.feature_layers = feature_layers if feature_layers is not None else []

        # infer "mult" assuming classic pattern: in_channels == ngf * (2**n_up)
        # e.g. ngf=64, n_up=2 => in_channels should be 256
        mult = in_channels // ngf
        if mult < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= ngf ({ngf})")

        layers: List[nn.Module] = []

        # Upsamples
        for _ in range(n_up):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2

        # Output head
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0, bias=True),
            nn.Tanh(),
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.return_features and (i in self.feature_layers):
                feats[f"dec_layer_{i}"] = h
        return (h, feats) if self.return_features else h


# ============================================================
# Resnet Generator (Encoder + Bottleneck + Decoder) 
# ============================================================

class ResnetGenerator(nn.Module):
    """
    Convenience wrapper to reproduce the old behavior, but with explicit modules:
      y = dec(bottleneck(enc(x)))

    This keeps your CycleGAN code simple while letting UNIT/MUNIT reuse enc/dec.
    """
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        n_down: int = 2,
        n_blocks: int = 9,
        n_up: int = 2,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ):
        super().__init__()
        self.enc = Encoder(input_nc, ngf=ngf, n_down=n_down, norm_layer=norm_layer)
        self.bottleneck = ResnetBottleneck(self.enc.out_channels, n_blocks=n_blocks, norm_layer=norm_layer)
        self.dec = Decoder(self.enc.out_channels, output_nc, ngf=ngf, n_up=n_up, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.bottleneck(z)
        y = self.dec(z)
        return y


# class ResnetGenerator(nn.Module):
#     """
#     Classic CycleGAN generator:
#       c7s1-64, d128, d256, N*ResBlocks, u128, u64, c7s1-3
#     Optionally returns intermediate feature maps (useful for contrastive losses later).
#     """
#     def __init__(
#         self,
#         input_nc: int,
#         output_nc: int,
#         ngf: int = 64, # Number of generator base filters (32 if memory is tight)   
#         n_blocks: int = 9, # 9 for 256x256 images, 6 for 128x128, 4 for UNIT / MUNIT    
#         norm_layer: nn.Module = nn.InstanceNorm2d,
#         return_features: bool = False,
#         feature_layers: Optional[List[int]] = None,
#     ):
#         super().__init__()
#         assert n_blocks >= 0
#         self.return_features = return_features
#         self.feature_layers = feature_layers if feature_layers is not None else []

#         layers: List[nn.Module] = []
#         layers += [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
#             norm_layer(ngf),
#             nn.ReLU(True),
#         ]

#         # Downsample
#         mult = 1
#         for _ in range(2):
#             layers += [
#                 nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
#                 norm_layer(ngf * mult * 2),
#                 nn.ReLU(True),
#             ]
#             mult *= 2

#         # Res blocks
#         for _ in range(n_blocks):
#             layers += [ResnetBlock(ngf * mult, padding_type="reflect", norm_layer=norm_layer)]

#         # Upsample
#         for _ in range(2):
#             layers += [
#                 nn.ConvTranspose2d(
#                     ngf * mult, ngf * mult // 2,
#                     kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
#                 ),
#                 norm_layer(ngf * mult // 2),
#                 nn.ReLU(True),
#             ]
#             mult //= 2

#         layers += [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=True),
#             nn.Tanh(),
#         ]

#         self.layers = nn.ModuleList(layers)

#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
#         feats: Dict[str, torch.Tensor] = {}
#         h = x
#         for i, layer in enumerate(self.layers):
#             h = layer(h)
#             if self.return_features and (i in self.feature_layers):
#                 feats[f"layer_{i}"] = h
#         if self.return_features:
#             return h, feats
#         return h


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (no sigmoid by default)."""
    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ):
        super().__init__()
        kw = 4
        padw = 1
        seq: List[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            seq += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        seq += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        seq += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# Small image helpers
# ============================================================

def denorm01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x + 1.0) / 2.0
