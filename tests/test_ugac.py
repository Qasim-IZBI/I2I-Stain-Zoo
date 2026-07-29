"""Tests for the UGAC aleatoric-uncertainty heads on CycleGAN.

Covers the properties that make the estimator trustworthy:
  - the GGD NLL reduces to L1 at (alpha, beta) = (1, 1), so UGAC strictly
    generalises the vanilla cycle loss
  - the closed-form aleatoric variance matches the Laplace / Gaussian cases
  - the vanilla path is untouched when ugac=False
  - alpha/beta recover a known heteroscedastic noise scale
"""

import math

import pytest
import torch
import torch.nn.functional as F

from base_models import Decoder3Head, ggd_nll, ggd_aleatoric_var
from models.cyclegan import CycleGAN, CycleGANConfig


class TestGGDLoss:
    def test_reduces_to_l1_at_unit_params(self):
        """(alpha, beta) = (1, 1) must recover the plain L1 cycle loss."""
        torch.manual_seed(0)
        pred, tgt = torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16)
        ones = torch.ones(4, 1, 16, 16)
        nll = ggd_nll(pred, tgt, ones, ones)
        assert nll == pytest.approx(float((pred - tgt).abs().mean()), abs=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_variance_laplace(self, alpha):
        """beta = 1 is Laplace, whose variance is 2*alpha^2."""
        ia = torch.full((1, 1, 1, 1), 1.0 / alpha)
        b = torch.ones(1, 1, 1, 1)
        assert ggd_aleatoric_var(ia, b).item() == pytest.approx(2 * alpha ** 2, rel=1e-5)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_variance_gaussian(self, alpha):
        """beta = 2 is Gaussian with variance alpha^2 / 2."""
        ia = torch.full((1, 1, 1, 1), 1.0 / alpha)
        b = torch.full((1, 1, 1, 1), 2.0)
        assert ggd_aleatoric_var(ia, b).item() == pytest.approx(alpha ** 2 / 2, rel=1e-5)

    def test_finite_under_extreme_inputs(self):
        """Large residuals and large beta must not overflow (fp16 range)."""
        pred = torch.full((2, 3, 8, 8), 1.0)
        tgt = torch.full((2, 3, 8, 8), -1.0)          # residual 2.0, the [-1,1] max
        ia = torch.full((2, 1, 8, 8), 100.0)          # alpha = 0.01
        b = torch.full((2, 1, 8, 8), 4.0)             # max allowed shape
        assert torch.isfinite(ggd_nll(pred, tgt, ia, b))

    def test_zero_residual_is_finite(self):
        """|r| = 0 hits log(0) without the epsilon guard."""
        x = torch.zeros(2, 3, 8, 8)
        ia = torch.ones(2, 1, 8, 8)
        b = torch.ones(2, 1, 8, 8)
        assert torch.isfinite(ggd_nll(x, x, ia, b))

    def test_mask_restricts_to_selected_pixels(self):
        torch.manual_seed(0)
        pred, tgt = torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)
        ia, b = torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :] = 1.0
        masked = ggd_nll(pred, tgt, ia, b, mask=mask)
        half = ggd_nll(pred[..., :4, :], tgt[..., :4, :], ia[..., :4, :], b[..., :4, :])
        assert masked == pytest.approx(float(half), abs=1e-5)


class TestDecoder3Head:
    def test_output_shapes_and_ranges(self):
        dec = Decoder3Head(64, 3, ngf=16, n_up=2)
        mu, inv_alpha, beta = dec(torch.randn(2, 64, 16, 16))
        assert mu.shape == (2, 3, 64, 64)
        assert inv_alpha.shape == (2, 1, 64, 64)
        assert beta.shape == (2, 1, 64, 64)
        assert mu.abs().max() <= 1.0                      # tanh
        assert (inv_alpha > 0).all() and (beta > 0).all()  # positivity
        assert beta.max() <= dec.max_beta


class TestCycleGANUGAC:
    def test_vanilla_path_unchanged(self):
        """ugac=False must keep the plain Decoder and produce no UGAC logs."""
        m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16))
        from base_models import Decoder
        assert isinstance(m.Dec_A, Decoder) and not isinstance(m.Dec_A, Decoder3Head)
        batch = {"A": torch.randn(2, 3, 64, 64), "B": torch.randn(2, 3, 64, 64)}
        _, logs, _ = m.compute_generator_loss(batch)
        assert "alpha_mean" not in logs

    def test_ugac_path_uses_three_heads(self):
        m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16, ugac=True))
        assert isinstance(m.Dec_A, Decoder3Head)
        batch = {"A": torch.randn(2, 3, 64, 64), "B": torch.randn(2, 3, 64, 64)}
        _, logs, _ = m.compute_generator_loss(batch)
        for k in ("alpha_mean", "beta_mean", "cycle_l1"):
            assert k in logs

    def test_forward_returns_image_in_both_modes(self):
        """forward_A2B must keep its contract so inference/eval code is unaffected."""
        x = torch.randn(2, 3, 64, 64)
        for ugac in (False, True):
            m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16, ugac=ugac))
            y = m.forward_A2B(x)
            assert isinstance(y, torch.Tensor) and y.shape == (2, 3, 64, 64)

    def test_uncertainty_forward(self):
        m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16, ugac=True))
        img, var = m.forward_A2B_uncertainty(torch.randn(2, 3, 64, 64))
        assert img.shape == (2, 3, 64, 64)
        assert var.shape == (2, 1, 64, 64)
        assert (var > 0).all() and torch.isfinite(var).all()

    def test_uncertainty_rejected_without_ugac(self):
        m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16))
        with pytest.raises(RuntimeError, match="ugac=True"):
            m.forward_A2B_uncertainty(torch.randn(1, 3, 64, 64))

    def test_gradients_reach_all_three_heads(self):
        m = CycleGAN(CycleGANConfig(ngf=16, n_blocks=2, ndf=16, ugac=True))
        batch = {"A": torch.randn(2, 3, 64, 64), "B": torch.randn(2, 3, 64, 64)}
        loss, _, _ = m.compute_generator_loss(batch)
        loss.backward()
        for head in ("head_mu", "head_inv_alpha", "head_beta"):
            grad = getattr(m.Dec_A, head)[1].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    def test_param_overhead_is_small(self):
        """The extra heads must not disturb the S/M/L parameter budgets."""
        n_van = sum(p.numel() for p in CycleGAN(CycleGANConfig(ngf=64, n_blocks=9)).parameters())
        n_ug = sum(p.numel() for p in CycleGAN(CycleGANConfig(ngf=64, n_blocks=9, ugac=True)).parameters())
        assert (n_ug - n_van) / n_van < 0.001


class TestHeteroscedasticRecovery:
    def test_recovers_known_noise_scale(self):
        """Fitting the GGD NLL to Laplace noise of known, spatially varying scale
        must recover that scale — this is the property the whole method rests on."""
        torch.manual_seed(0)
        H = 16
        true_alpha = torch.ones(1, 1, H, H)
        true_alpha[..., : H // 2, :] = 0.2
        true_alpha[..., H // 2 :, :] = 1.0

        resid = torch.distributions.Laplace(0.0, 1.0).sample((256, 3, H, H)) * true_alpha
        pred = torch.zeros(256, 3, H, H)
        tgt = pred + resid

        raw_ia = torch.zeros(1, 1, H, H, requires_grad=True)
        raw_b = torch.zeros(1, 1, H, H, requires_grad=True)
        opt = torch.optim.Adam([raw_ia, raw_b], lr=0.05)
        for _ in range(400):
            ia = F.softplus(raw_ia) + 1e-2
            b = (F.softplus(raw_b) + 0.2).clamp(max=4.0)
            loss = ggd_nll(pred, tgt, ia, b)
            opt.zero_grad()
            loss.backward()
            opt.step()

        ia = F.softplus(raw_ia) + 1e-2
        b = (F.softplus(raw_b) + 0.2).clamp(max=4.0)
        sd = ggd_aleatoric_var(ia, b).detach().sqrt()

        # true Laplace sd = sqrt(2) * alpha
        lo = sd[..., : H // 2, :].mean().item()
        hi = sd[..., H // 2 :, :].mean().item()
        assert lo == pytest.approx(math.sqrt(2) * 0.2, rel=0.25)
        assert hi == pytest.approx(math.sqrt(2) * 1.0, rel=0.25)
        assert hi > 3 * lo          # the noisy half is clearly flagged
