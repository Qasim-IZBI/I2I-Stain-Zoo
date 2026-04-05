"""
Shared fixtures and model registry for the I2I-Stain-Zoo test suite.

All models are built with tiny configs designed for CPU-only unit tests.
Images are 64x64 so the UVCGAN bottleneck (n_down=4) produces 4x4=16 ViT tokens.
"""
import sys
import os

# Ensure repo root is on sys.path so model imports work from any working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import torch
from PIL import Image
import tempfile

from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig
from models.uvcgan import UVCGAN, UVCGANConfig
from models.miudiff import MIUDiff, MIUDiffConfig


# ------------------------------------------------------------------ #
#  Tiny configs — fast on CPU, architecturally valid                  #
# ------------------------------------------------------------------ #

TINY_CYCLEGAN_CFG = CycleGANConfig(ngf=16, n_blocks=2, ndf=16)

TINY_UNIT_CFG = UNITConfig(
    ngf=16,
    z_dim=64,
    n_blocks_total=4,
    n_blocks_shared=2,
    n_blocks_private_pre=1,
    n_blocks_private_post=1,
    ndf=16,
)

TINY_MUNIT_CFG = MUNITConfig(
    ngf=16,
    n_content_blocks=2,
    n_adain_blocks=2,
    style_dim=4,
    mlp_dim=32,
)

TINY_DCLGAN_CFG = DCLGANConfig(
    ngf=16,
    n_blocks=2,
    ndf=16,
    n_patches=4,
    proj_dim=16,
)

# vit_features=48, vit_n_heads=6 → 48 % 6 == 0 (valid)
# n_down=2 → with 64×64 input the bottleneck is 16×16 = 256 tokens (matches pos_embed default)
# n_down=4 would give 4×4=16 tokens which mismatches the hardcoded 256-token pos_embed
TINY_UVCGAN_CFG = UVCGANConfig(
    ngf=16,
    n_down=2,
    vit_n_blocks=2,
    vit_features=48,
    vit_n_heads=6,
    ndf=16,
    pretrain=False,
    image_size=64,  # 64×64 → spatial=16, n_tokens=256 (matches pos_embed)
)

# base_channels=32 ensures GroupNorm32 divisibility:
# With channel_mult=(1,2): channels are 32, 64; skip-concat in decoder gives 96, 128 — both % 32 == 0.
# base_channels=16 would produce 48-channel inputs (16+32) where 48 % 32 ≠ 0.
TINY_MIUDIFF_PRETRAIN_CFG = MIUDiffConfig(
    stage="pretrain",
    base_channels=32,
    channel_mult=(1, 2),
    sample_steps=2,
    T=10,
)

TINY_MIUDIFF_FINETUNE_CFG = MIUDiffConfig(
    stage="finetune",
    base_channels=32,
    channel_mult=(1, 2),
    sample_steps=2,
    T=10,
    guidance_scale=0.0,  # 0 = use only eps_cond; faster and numerically simpler
)


# ------------------------------------------------------------------ #
#  Registry of (name, model, cfg) tuples used for parametrized tests  #
# ------------------------------------------------------------------ #

MODEL_REGISTRY = [
    ("cyclegan",        CycleGAN,  TINY_CYCLEGAN_CFG),
    ("unit",            UNIT,      TINY_UNIT_CFG),
    ("munit",           MUNIT,     TINY_MUNIT_CFG),
    ("dclgan",          DCLGAN,    TINY_DCLGAN_CFG),
    ("uvcgan",          UVCGAN,    TINY_UVCGAN_CFG),
    ("miudiff_pretrain",MIUDiff,   TINY_MIUDIFF_PRETRAIN_CFG),
    ("miudiff_finetune",MIUDiff,   TINY_MIUDIFF_FINETUNE_CFG),
]


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

IMAGE_SIZE = 64  # 64×64 keeps UVCGAN bottleneck valid and tests fast


@pytest.fixture
def real_batch():
    """A minimal batch dict {"A": ..., "B": ...} with CPU tensors in [-1, 1]."""
    A = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    B = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    return {"A": A, "B": B}


@pytest.fixture
def tmp_image_dir():
    """Temporary directory containing a handful of small .tif images."""
    with tempfile.TemporaryDirectory() as d:
        img_dir = os.path.join(d, "images")
        os.makedirs(img_dir)
        for i in range(4):
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(i * 40, 100, 200))
            img.save(os.path.join(img_dir, f"{i+1:07d}.tif"))
        yield d  # yield the root (parent of images/)


@pytest.fixture
def tmp_unpaired_dirs():
    """Two temporary image directories (domain A and B)."""
    with tempfile.TemporaryDirectory() as dA:
        with tempfile.TemporaryDirectory() as dB:
            for d in (dA, dB):
                img_dir = os.path.join(d, "images")
                os.makedirs(img_dir)
                for i in range(4):
                    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(i * 30, 80, 160))
                    img.save(os.path.join(img_dir, f"{i+1:07d}.tif"))
            yield dA, dB
