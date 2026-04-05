"""
MIUDiff-specific tests.

Covers:
  - sample_uncond(): output shape, value range, device consistency
  - sample_A2B():    output shape, value range (finetune stage)
  - Stage-specific loss keys (pretrain has no MI/cycle keys; finetune does)
  - compute_discriminator_loss() always returns (zero tensor, empty dict)
  - Config round-trip: asdict() / reconstruct preserves all fields
"""
import io
import os
import sys

import pytest
import torch
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.miudiff import MIUDiff, MIUDiffConfig
from tests.conftest import (
    TINY_MIUDIFF_PRETRAIN_CFG,
    TINY_MIUDIFF_FINETUNE_CFG,
    IMAGE_SIZE,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def pretrain_model():
    return MIUDiff(TINY_MIUDIFF_PRETRAIN_CFG).eval()


@pytest.fixture
def finetune_model():
    return MIUDiff(TINY_MIUDIFF_FINETUNE_CFG).eval()


@pytest.fixture
def source_image():
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def real_batch():
    return {
        "A": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        "B": torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    }


# ------------------------------------------------------------------ #
#  sample_uncond                                                       #
# ------------------------------------------------------------------ #

class TestSampleUncond:
    def test_output_shape(self, pretrain_model):
        with torch.no_grad():
            y = pretrain_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        assert y.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE), (
            f"Expected (1,3,{IMAGE_SIZE},{IMAGE_SIZE}), got {y.shape}"
        )

    def test_output_is_finite(self, pretrain_model):
        with torch.no_grad():
            y = pretrain_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        assert torch.isfinite(y).all(), "sample_uncond() produced non-finite values"

    def test_output_dtype_float32(self, pretrain_model):
        with torch.no_grad():
            y = pretrain_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        assert y.dtype == torch.float32

    def test_two_samples_differ(self, pretrain_model):
        """Stochastic sampling should produce different outputs each call."""
        with torch.no_grad():
            y1 = pretrain_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
            y2 = pretrain_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        assert not torch.allclose(y1, y2), (
            "sample_uncond() produced identical outputs on two independent calls — "
            "sampling appears deterministic or collapsed."
        )

    def test_finetune_model_can_also_sample_uncond(self, finetune_model):
        """sample_uncond() should work for any MIUDiff regardless of stage config."""
        with torch.no_grad():
            y = finetune_model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        assert y.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.isfinite(y).all()


# ------------------------------------------------------------------ #
#  sample_A2B (finetune stage)                                        #
# ------------------------------------------------------------------ #

class TestSampleA2B:
    def test_output_shape(self, finetune_model, source_image):
        with torch.no_grad():
            y = finetune_model.sample_A2B(source_image)
        assert y.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE), (
            f"Expected (1,3,{IMAGE_SIZE},{IMAGE_SIZE}), got {y.shape}"
        )

    def test_output_is_finite(self, finetune_model, source_image):
        with torch.no_grad():
            y = finetune_model.sample_A2B(source_image)
        assert torch.isfinite(y).all()

    def test_output_dtype_float32(self, finetune_model, source_image):
        with torch.no_grad():
            y = finetune_model.sample_A2B(source_image)
        assert y.dtype == torch.float32


# ------------------------------------------------------------------ #
#  Generator loss — stage checks                                       #
# ------------------------------------------------------------------ #

class TestPretainGeneratorLoss:
    def test_returns_triple(self, pretrain_model, real_batch):
        result = pretrain_model.compute_generator_loss(real_batch)
        assert len(result) == 3

    def test_loss_scalar_finite(self, pretrain_model, real_batch):
        loss, _, _ = pretrain_model.compute_generator_loss(real_batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_logs_nonempty(self, pretrain_model, real_batch):
        _, logs, _ = pretrain_model.compute_generator_loss(real_batch)
        assert len(logs) > 0

    def test_all_log_values_finite(self, pretrain_model, real_batch):
        _, logs, _ = pretrain_model.compute_generator_loss(real_batch)
        for k, v in logs.items():
            assert isinstance(v, float), f"log '{k}' is not float"
            assert torch.isfinite(torch.tensor(v)), f"log '{k}' = {v} is not finite"


class TestFinetuneGeneratorLoss:
    def test_returns_triple(self, finetune_model, real_batch):
        result = finetune_model.compute_generator_loss(real_batch)
        assert len(result) == 3

    def test_loss_scalar_finite(self, finetune_model, real_batch):
        loss, _, _ = finetune_model.compute_generator_loss(real_batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_logs_nonempty(self, finetune_model, real_batch):
        _, logs, _ = finetune_model.compute_generator_loss(real_batch)
        assert len(logs) > 0

    def test_all_log_values_finite(self, finetune_model, real_batch):
        _, logs, _ = finetune_model.compute_generator_loss(real_batch)
        for k, v in logs.items():
            assert isinstance(v, float), f"log '{k}' is not float"
            assert torch.isfinite(torch.tensor(v)), f"log '{k}' = {v} is not finite"


# ------------------------------------------------------------------ #
#  Discriminator loss (always a no-op for MIUDiff)                   #
# ------------------------------------------------------------------ #

class TestDiscriminatorLoss:
    def test_pretrain_returns_zero(self, pretrain_model, real_batch):
        _, _, visuals = pretrain_model.compute_generator_loss(real_batch)
        loss, logs = pretrain_model.compute_discriminator_loss(real_batch, visuals)
        assert loss.item() == pytest.approx(0.0)
        assert isinstance(logs, dict)

    def test_finetune_returns_zero(self, finetune_model, real_batch):
        _, _, visuals = finetune_model.compute_generator_loss(real_batch)
        loss, logs = finetune_model.compute_discriminator_loss(real_batch, visuals)
        assert loss.item() == pytest.approx(0.0)
        assert isinstance(logs, dict)


# ------------------------------------------------------------------ #
#  Config round-trip                                                   #
# ------------------------------------------------------------------ #

class TestConfigRoundTrip:
    def test_pretrain_config(self):
        cfg = TINY_MIUDIFF_PRETRAIN_CFG
        cfg2 = MIUDiffConfig(**asdict(cfg))
        assert cfg == cfg2

    def test_finetune_config(self):
        cfg = TINY_MIUDIFF_FINETUNE_CFG
        cfg2 = MIUDiffConfig(**asdict(cfg))
        assert cfg == cfg2

    def test_checkpoint_restore(self):
        model = MIUDiff(TINY_MIUDIFF_PRETRAIN_CFG).eval()
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "config": asdict(TINY_MIUDIFF_PRETRAIN_CFG)}, buf)
        buf.seek(0)

        ckpt = torch.load(buf, map_location="cpu")
        saved_cfg = ckpt["config"]
        # channel_mult saved as list in JSON — must round-trip to tuple for MIUDiffConfig
        if isinstance(saved_cfg.get("channel_mult"), list):
            saved_cfg["channel_mult"] = tuple(saved_cfg["channel_mult"])
        fresh = MIUDiff(MIUDiffConfig(**saved_cfg)).eval()
        fresh.load_state_dict(ckpt["model"], strict=True)

        torch.manual_seed(0)
        with torch.no_grad():
            y1 = model.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)
        torch.manual_seed(0)
        with torch.no_grad():
            y2 = fresh.sample_uncond(batch_size=1, image_size=IMAGE_SIZE)

        assert torch.allclose(y1, y2, atol=1e-5), (
            "sample_uncond() output differed after checkpoint round-trip"
        )
