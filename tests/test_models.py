"""
Model interface compliance tests.

Covers:
  - generator_parameters() / discriminator_parameters() are non-empty lists
  - compute_generator_loss() returns (scalar tensor, dict, dict)
  - compute_discriminator_loss() returns (scalar tensor, dict)
  - All log values are finite floats
  - Checkpoint save/restore round-trip: model output is identical after reload
"""
import io
import sys
import os

import pytest
import torch
from dataclasses import asdict

# conftest adds ROOT to sys.path; this import is redundant but harmless
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.conftest import MODEL_REGISTRY


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_model(model_cls, cfg):
    return model_cls(cfg).eval()


def _model_ids():
    return [name for name, _, _ in MODEL_REGISTRY]


# ------------------------------------------------------------------ #
#  Parametrized fixtures — one instance per registry entry            #
# ------------------------------------------------------------------ #

@pytest.fixture(params=MODEL_REGISTRY, ids=_model_ids())
def model_and_cfg(request):
    name, model_cls, cfg = request.param
    return name, _make_model(model_cls, cfg), cfg


# ------------------------------------------------------------------ #
#  Interface tests                                                     #
# ------------------------------------------------------------------ #

class TestGeneratorParameters:
    def test_returns_nonempty_list(self, model_and_cfg):
        _, model, _ = model_and_cfg
        params = model.generator_parameters()
        assert isinstance(params, list), "generator_parameters() must return a list"
        assert len(params) > 0, "generator_parameters() must not be empty"

    def test_all_params_are_tensors(self, model_and_cfg):
        _, model, _ = model_and_cfg
        for p in model.generator_parameters():
            assert isinstance(p, torch.Tensor)


class TestDiscriminatorParameters:
    def test_returns_list(self, model_and_cfg):
        name, model, _ = model_and_cfg
        params = model.discriminator_parameters()
        assert isinstance(params, list), "discriminator_parameters() must return a list"

    def test_nonempty_for_gan_models(self, model_and_cfg):
        name, model, _ = model_and_cfg
        # MIUDiff has an empty discriminator param list by design
        if "miudiff" in name:
            pytest.skip("MIUDiff has no discriminator")
        params = model.discriminator_parameters()
        assert len(params) > 0


class TestGeneratorLoss:
    # Note: torch.no_grad() is NOT used here because MIUDiff calls .backward()
    # internally on the MI estimator inside compute_generator_loss, which requires
    # the autograd graph to be alive.

    def test_returns_triple(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        result = model.compute_generator_loss(real_batch)
        assert len(result) == 3, "compute_generator_loss must return (loss, logs, visuals)"

    def test_loss_is_scalar_tensor(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        loss, _, _ = model.compute_generator_loss(real_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, "generator loss must be a scalar"

    def test_loss_is_finite(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        loss, _, _ = model.compute_generator_loss(real_batch)
        assert torch.isfinite(loss), f"generator loss is not finite: {loss.item()}"

    def test_logs_are_dict_of_floats(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, logs, _ = model.compute_generator_loss(real_batch)
        assert isinstance(logs, dict)
        for k, v in logs.items():
            assert isinstance(v, float), f"log value '{k}' is not a float (got {type(v)})"
            assert torch.isfinite(torch.tensor(v)), f"log value '{k}' = {v} is not finite"

    def test_logs_nonempty(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, logs, _ = model.compute_generator_loss(real_batch)
        assert len(logs) > 0, "compute_generator_loss must return at least one log entry"

    def test_visuals_are_dict_of_tensors(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, _, visuals = model.compute_generator_loss(real_batch)
        assert isinstance(visuals, dict)
        for k, v in visuals.items():
            assert isinstance(v, torch.Tensor), f"visual '{k}' is not a Tensor"


class TestDiscriminatorLoss:
    def test_returns_pair(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, _, visuals = model.compute_generator_loss(real_batch)
        with torch.no_grad():
            result = model.compute_discriminator_loss(real_batch, visuals)
        assert len(result) == 2, "compute_discriminator_loss must return (loss, logs)"

    def test_loss_is_scalar_tensor(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, _, visuals = model.compute_generator_loss(real_batch)
        with torch.no_grad():
            loss, _ = model.compute_discriminator_loss(real_batch, visuals)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_is_finite(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, _, visuals = model.compute_generator_loss(real_batch)
        with torch.no_grad():
            loss, _ = model.compute_discriminator_loss(real_batch, visuals)
        assert torch.isfinite(loss)

    def test_logs_are_dict(self, model_and_cfg, real_batch):
        _, model, _ = model_and_cfg
        _, _, visuals = model.compute_generator_loss(real_batch)
        with torch.no_grad():
            _, logs = model.compute_discriminator_loss(real_batch, visuals)
        assert isinstance(logs, dict)


# ------------------------------------------------------------------ #
#  Checkpoint round-trip                                              #
# ------------------------------------------------------------------ #

class TestCheckpointRoundTrip:
    def test_save_and_restore(self, model_and_cfg, real_batch):
        name, model, cfg = model_and_cfg

        # MIUDiff calls .backward() inside compute_generator_loss (MI estimator opt step),
        # making loss-based round-trip comparison unreliable. Its checkpoint fidelity is
        # tested separately in test_miudiff.py::TestConfigRoundTrip.
        if "miudiff" in name:
            pytest.skip("MIUDiff checkpoint round-trip tested in test_miudiff.py")

        # Run forward before save — fix random seed so stochastic ops (MUNIT style
        # sampling, DCLGAN patch sampling) produce the same draws on both calls.
        torch.manual_seed(0)
        out_before, _, _ = model.compute_generator_loss(real_batch)

        # Save to buffer
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "config": asdict(cfg)}, buf)
        buf.seek(0)

        # Load into fresh model
        ckpt = torch.load(buf, map_location="cpu")
        model_cls = type(model)
        cfg2 = type(cfg)(**ckpt["config"])
        fresh = model_cls(cfg2).eval()
        fresh.load_state_dict(ckpt["model"], strict=True)

        # Forward pass on fresh model with the same random seed
        torch.manual_seed(0)
        out_after, _, _ = fresh.compute_generator_loss(real_batch)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            f"Generator loss changed after checkpoint round-trip for {name}: "
            f"{out_before.item():.6f} vs {out_after.item():.6f}"
        )


# ------------------------------------------------------------------ #
#  Output shape tests                                                  #
# ------------------------------------------------------------------ #

class TestForwardOutputShape:
    """Check that the main translation output has the expected shape [B, 3, H, W]."""

    IMAGE_SIZE = 64

    def _expected_shape(self):
        return (1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE)

    def test_cyclegan_A2B(self, real_batch):
        from models.cyclegan import CycleGAN
        from tests.conftest import TINY_CYCLEGAN_CFG
        model = CycleGAN(TINY_CYCLEGAN_CFG).eval()
        with torch.no_grad():
            y = model.forward_A2B(real_batch["A"])
        assert y.shape == self._expected_shape()

    def test_cyclegan_B2A(self, real_batch):
        from models.cyclegan import CycleGAN
        from tests.conftest import TINY_CYCLEGAN_CFG
        model = CycleGAN(TINY_CYCLEGAN_CFG).eval()
        with torch.no_grad():
            y = model.forward_B2A(real_batch["B"])
        assert y.shape == self._expected_shape()

    def test_unit_A2B(self, real_batch):
        from models.unit import UNIT
        from tests.conftest import TINY_UNIT_CFG
        model = UNIT(TINY_UNIT_CFG).eval()
        with torch.no_grad():
            y, _ = model.forward_A2B(real_batch["A"])
        assert y.shape == self._expected_shape()

    def test_munit_A2B_random_style(self, real_batch):
        from models.munit import MUNIT
        from tests.conftest import TINY_MUNIT_CFG
        model = MUNIT(TINY_MUNIT_CFG).eval()
        with torch.no_grad():
            c, _ = model.encode_A(real_batch["A"])
            s = torch.randn(1, TINY_MUNIT_CFG.style_dim)
            y = model.decode_B(c, s)
        assert y.shape == self._expected_shape()

    def test_dclgan_A2B(self, real_batch):
        from models.dclgan import DCLGAN
        from tests.conftest import TINY_DCLGAN_CFG
        model = DCLGAN(TINY_DCLGAN_CFG).eval()
        with torch.no_grad():
            y = model.forward_A2B(real_batch["A"])
        assert y.shape == self._expected_shape()

    def test_uvcgan_A2B(self, real_batch):
        from models.uvcgan import UVCGAN
        from tests.conftest import TINY_UVCGAN_CFG
        model = UVCGAN(TINY_UVCGAN_CFG).eval()
        with torch.no_grad():
            y = model.forward_A2B(real_batch["A"])
        assert y.shape == self._expected_shape()
