# Adding a New Model to I2I-Stain-Zoo

This document is the authoritative reference for adding a new image-to-image translation model
to the repo. Follow every section in order. The goal is that any new model plugs into the shared
training, inference, and evaluation pipeline without touching `BaseTrainer` or the dataset code.

---

## 1. Repo Map (what you will touch)

```
models/
  <yourmodel>.py          ← create this (Config + Model class)

train.py                  ← register Config, Model, CLI args, param counter
inference.py              ← register model loading + inference forward pass
model_sizes.md            ← add small/medium/large A→B param counts
CLAUDE.md                 ← add training and inference command examples
```

Everything else — `trainer/base_trainer.py`, `datasets/`, `evaluation.py`,
`reconstruct.py` — is untouched for standard models.

---

## 2. The BaseTrainer Contract

`BaseTrainer` calls exactly four things on the model object. Your model **must** implement all of them.

| Method | Signature | Called by |
|--------|-----------|-----------|
| `generator_parameters()` | `() → Iterable[nn.Parameter]` | builds `opt_G` |
| `discriminator_parameters()` | `() → Iterable[nn.Parameter]` | builds `opt_D` (skip if no discriminator) |
| `compute_generator_loss(batch)` | `(Dict) → (Tensor, Dict[str,float], Dict[str,Tensor])` | generator step |
| `compute_discriminator_loss(batch, visuals)` | `(Dict, Dict) → (Tensor, Dict[str,float])` | discriminator step |

**`batch`** is always a dict produced by `UnpairedDataset`:
```python
{"A": Tensor[B,3,256,256], "B": Tensor[B,3,256,256]}   # [-1, 1] normalized
```

**Return values:**
- `loss_G` / `loss_D` — scalar `Tensor` with grad attached (used for `.backward()`)
- `logs` — `dict[str, float]` (use `.detach().cpu()` before casting to float); these keys become
  columns in `loss_log.csv` and are plotted live during training
- `visuals` — `dict[str, Tensor]` of images still in `[-1, 1]`; passed back into
  `compute_discriminator_loss` and saved as sample grids; first four images per key are shown

If your model has **no discriminator**, simply do not define `discriminator_parameters()` —
`BaseTrainer` checks `hasattr(model, "discriminator_parameters")` and skips `opt_D` entirely.
`compute_discriminator_loss` will never be called, but you should still return an empty
`logs_D = {}` (the trainer handles this automatically).

**`self.cfg`** — `BaseTrainer` calls `asdict(self.model.cfg)` when saving checkpoints.
Your model **must** have a `self.cfg` attribute that is a `dataclass` instance.

---

## 3. File Layout: `models/<yourmodel>.py`

### 3.1 Config dataclass

```python
from dataclasses import dataclass

@dataclass
class MyModelConfig:
    # architecture
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    n_down: int = 2
    n_up: int = 2
    n_layers_D: int = 3

    # losses
    gan_mode: str = "lsgan"
    lambda_cycle: float = 10.0

    # misc
    pool_size: int = 50
```

Rules:
- All fields must have defaults (so `MyModelConfig()` is valid with no args).
- All field values must be JSON-serializable primitives (int, float, str, bool, list).
  Use `list` not `tuple` if you need sequences — `asdict()` converts tuples to lists,
  but reconstruction from a checkpoint does `MyModelConfig(**saved_cfg)`, so the type
  the field accepts must match what comes back from JSON (i.e. a `list`, not a `tuple`).
  If you need a tuple at runtime, store as `list` in the config and convert in `__init__`.
- Do not store tensors or nn.Modules in the config.

### 3.2 Model class skeleton

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple

from base_models import (
    Encoder, ResnetBottleneck, Decoder,
    NLayerDiscriminator, ImagePool, GANLoss,
    discriminator_loss, identity_loss,
)


class MyModel(nn.Module):
    """
    One-line description of the method.

    BaseTrainer interface:
      - generator_parameters()
      - discriminator_parameters()
      - compute_generator_loss(batch) -> (loss, logs, visuals)
      - compute_discriminator_loss(batch, visuals) -> (loss, logs)
    """

    def __init__(self, cfg: MyModelConfig):
        super().__init__()
        self.cfg = cfg          # REQUIRED — used by checkpoint saving

        # --- generators ---
        self.Enc_A = Encoder(cfg.input_nc, ngf=cfg.ngf, n_down=cfg.n_down)
        self.Bn_A  = ResnetBottleneck(self.Enc_A.out_channels, n_blocks=cfg.n_blocks)
        self.Dec_B = Decoder(self.Enc_A.out_channels, cfg.output_nc,
                             ngf=cfg.ngf, n_up=cfg.n_up)
        # ... symmetric B→A path if needed ...

        # --- discriminators ---
        self.D_A = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)
        self.D_B = NLayerDiscriminator(cfg.input_nc, ndf=cfg.ndf, n_layers=cfg.n_layers_D)

        # --- replay buffers ---
        self.pool_A = ImagePool(cfg.pool_size)
        self.pool_B = ImagePool(cfg.pool_size)

        # --- loss functions ---
        self.gan = GANLoss(cfg.gan_mode)
        self.l1  = nn.L1Loss()

    # ---- BaseTrainer interface ----------------------------------------

    def generator_parameters(self):
        return (list(self.Enc_A.parameters()) +
                list(self.Bn_A.parameters())  +
                list(self.Dec_B.parameters()))
        # add B→A path if it exists

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    def compute_generator_loss(self, batch: Dict[str, torch.Tensor]):
        real_A, real_B = batch["A"], batch["B"]

        fake_B = self.forward_A2B(real_A)
        fake_A = self.forward_B2A(real_B)

        loss_gan   = self.gan(self.D_B(fake_B), True) + self.gan(self.D_A(fake_A), True)
        loss_cycle = self.l1(self.forward_B2A(fake_B), real_A) + \
                     self.l1(self.forward_A2B(fake_A), real_B)

        loss_G = loss_gan + self.cfg.lambda_cycle * loss_cycle

        logs = {
            "loss_G":     float(loss_G.detach().cpu()),
            "loss_gan":   float(loss_gan.detach().cpu()),
            "loss_cycle": float(loss_cycle.detach().cpu()),
        }
        visuals = {
            "real_A": real_A, "fake_B": fake_B,
            "real_B": real_B, "fake_A": fake_A,
        }
        return loss_G, logs, visuals

    def compute_discriminator_loss(self, batch: Dict[str, torch.Tensor],
                                   visuals: Dict[str, torch.Tensor]):
        real_A, real_B = batch["A"], batch["B"]
        fake_A = self.pool_A.query(visuals["fake_A"].detach())
        fake_B = self.pool_B.query(visuals["fake_B"].detach())
        return discriminator_loss(self.gan, self.D_A, self.D_B, real_A, real_B, fake_A, fake_B)

    # ---- Inference helpers -------------------------------------------
    # inference.py calls these directly — must be defined

    def forward_A2B(self, x: torch.Tensor) -> torch.Tensor:
        return self.Dec_B(self.Bn_A(self.Enc_A(x)))

    def forward_B2A(self, x: torch.Tensor) -> torch.Tensor:
        ...  # symmetric
```

**Naming conventions for `visuals`:**
Use `real_A`, `fake_B`, `rec_A`, `real_B`, `fake_A`, `rec_B` where applicable. These names
are cosmetic (only used for sample grid filenames) but keep them consistent across models.

**Naming conventions for `logs`:**
- Generator total: `loss_G`
- Discriminator total: `loss_D`
- Sub-losses: `loss_<component>` (e.g. `loss_cycle`, `loss_gan`, `loss_idt`, `loss_dcl`)

---

## 4. Available Building Blocks

### 4.1 `base_models.py` — GAN primitives

Import from `base_models` — do not reimplement these.

| Symbol | Description |
|--------|-------------|
| `Encoder(input_nc, ngf, n_down)` | Stem + strided convolutions. `enc.out_channels` gives the bottleneck channel count. |
| `ResnetBottleneck(channels, n_blocks)` | Stack of `ResnetBlock`. Shape-preserving. |
| `Decoder(in_channels, output_nc, ngf, n_up)` | ConvTranspose upsample + tanh head. |
| `ResnetGenerator(input_nc, output_nc, ngf, n_down, n_blocks, n_up)` | Convenience wrapper: Enc → Bn → Dec in one module (use when you don't need to split the path). |
| `NLayerDiscriminator(input_nc, ndf, n_layers)` | 70×70 PatchGAN discriminator. |
| `GANLoss(mode)` | `mode="lsgan"` (MSE) or `"vanilla"` (BCE). Call as `self.gan(pred, is_real: bool)`. |
| `ImagePool(pool_size)` | Replay buffer. Call `.query(fake.detach())` in discriminator step. |
| `PatchSampler` | Samples spatial feature vectors from `[B,C,H,W]` maps. Static `.sample(feat, n_patches)`. |
| `info_nce(q, k, temperature)` | InfoNCE loss with one-to-one positives. |
| `init_weights(net, init_type, init_gain)` | Normal/Xavier/Kaiming weight init. |
| `denorm01(x)` | `[-1,1] → [0,1]` helper. |
| `discriminator_loss(gan, D_A, D_B, real_A, real_B, fake_A, fake_B)` | Standard symmetric PatchGAN discriminator loss. Pass already-detached (and optionally pool-queried) fakes. Returns `(loss_D, logs_dict)`. |
| `identity_loss(l1, forward_A2B, forward_B2A, real_A, real_B, lam)` | Optional CycleGAN-style identity regularisation. Returns zero tensor when `lam <= 0`. `forward_A2B`/`forward_B2A` must return plain tensors. |

**Typical channel arithmetic:**
```python
enc = Encoder(input_nc=3, ngf=64, n_down=2)
# enc.out_channels == 64 * (2**2) == 256
bn  = ResnetBottleneck(enc.out_channels, n_blocks=9)
dec = Decoder(enc.out_channels, output_nc=3, ngf=64, n_up=2)
```
`n_down` and `n_up` must match, and `ngf` must match between `Encoder` and `Decoder`.

### 4.2 `models/miudiff.py` — Diffusion primitives

These are **not** in `base_models.py`. Import explicitly:

```python
from models.miudiff import DDPMUNet, UNetConfig, DiffusionSchedule, to_gray, sobel_grad
```

| Symbol | Description |
|--------|-------------|
| `UNetConfig` | Dataclass controlling DDPMUNet architecture: `in_channels`, `base_channels`, `channel_mult`, `num_res_blocks`, etc. |
| `DDPMUNet(cfg: UNetConfig)` | DDPM-style UNet. Forward: `(x: [B,C,H,W], t_frac: [B]) → [B,out_ch,H,W]`. Runs entirely in fp32 internally via `autocast(enabled=False)`. |
| `DiffusionSchedule(T, beta_start, beta_end)` | Linear β schedule. Call `.make(device)` → `(betas, alphas, alpha_bars, alpha_bars_prev)`. Register the tensors as buffers in your model. |
| `to_gray(x)` | `[B,3,H,W] → [B,1,H,W]` weighted grayscale. |
| `sobel_grad(x)` | `[B,1,H,W] → [B,1,H,W]` Sobel gradient magnitude. |

**DDPMUNet and AMP:** `DDPMUNet` wraps every forward in `autocast(enabled=False)` so it always runs fp32. Enabling PyTorch's `GradScaler` on top of this causes the loss scale to keep doubling until it overflows (~56k steps). For any model that uses `DDPMUNet`, you **must** add the model name to the `_ddpm_models` set in `train.py` to suppress AMP:

```python
# train.py — main()
_ddpm_models = {"miudiff", "unitddpm", "cyclediffusion", "unsb", "yourmodel"}
use_amp = args.amp and args.model not in _ddpm_models
```

**`channel_mult` CLI → config conversion:** Architecture CLI args for diffusion models store `channel_mult` as a comma-separated string (e.g. `"1,2,2,4"`). Convert in `build_model()`:

```python
channel_mult=tuple(int(x) for x in args.yourmodel_channel_mult.split(","))
```

---

## 5. Registering in `train.py`

Make **all five** of these changes:

### 5.1 Import
```python
from models.mymodel import MyModel, MyModelConfig
```

### 5.2 Add to `_CONFIG_CLS` and `_MODEL_CLS`
```python
_CONFIG_CLS = {
    ...
    "mymodel": MyModelConfig,
}
_MODEL_CLS = {
    ...
    "mymodel": MyModel,
}
```
Adding to these two dicts is sufficient for `--init_ckpt` checkpoint loading to work
automatically for standard single-stage models.

> **Multi-stage / complex models** (anything with staged training, custom weight transfer, or
> diffusion components) should **not** be added to `_CONFIG_CLS`/`_MODEL_CLS`. Handle them
> directly in `build_model()` with an `if args.model == "mymodel":` block instead — see
> Section 7.

### 5.3 Add to `_build_default_gan_config()`
```python
def _build_default_gan_config(args):
    ...
    elif args.model == "mymodel":
        return MyModelConfig(
            ngf=args.mymodel_ngf,
            n_blocks=args.mymodel_n_blocks,
            # ... any other CLI args
        )
```

### 5.4 Add to `_count_a2b_params()`
```python
def _count_a2b_params(model, model_name, ...):
    ...
    elif model_name == "mymodel":
        a2b = n(model.Enc_A) + n(model.Bn_A) + n(model.Dec_B)
```
This powers `python train.py --model mymodel --count_params`.

### 5.5 Register the CLI `--model` choice and add architecture args
```python
parser.add_argument("--model",
    choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan", "mymodel"],
    ...)

# Architecture knobs — prefix all args with the model name
parser.add_argument("--mymodel_ngf",      type=int, default=64)
parser.add_argument("--mymodel_n_blocks", type=int, default=9)
```

---

## 6. Registering in `inference.py`

### 6.1 Import
```python
from models.mymodel import MyModel, MyModelConfig
```

### 6.2 Add a branch inside `load_model()`
```python
elif args.model == "mymodel":
    cfg = MyModelConfig(**saved_cfg) if saved_cfg else MyModelConfig()
    model = MyModel(cfg)
```
Place this before the final `else: raise ValueError(args.model)`.

### 6.3 Add a branch inside the inference loop
```python
elif args.model == "mymodel":
    if args.direction == "A2B":
        y = model.forward_A2B(x)
    else:
        y = model.forward_B2A(x)
    save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)
```

> **Use `save_tile`, not `save_image` directly.** `save_tile` handles denormalisation
> (`[-1,1] → [0,1]`) and applies Reinhard LAB colour normalisation when `--color_ref` is given.
> It is defined at the top of `inference.py` — do not call `save_image` directly from the
> inference loop.

### 6.4 Register the CLI `--model` choice
```python
parser.add_argument("--model",
    choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan", "mymodel"],
    ...)
```

If your model has **inference-time hyperparameters** (e.g. number of diffusion steps, style
sampling), add model-specific CLI args to the `inference.py` parser the same way MIUDiff does,
and override the relevant config fields after loading `saved_cfg`.

### 6.5 `strict` flag in `load_model()`

By default `load_model()` uses `strict=True`. Use `strict=False` when:
- Your model has optional submodules that may not exist in some checkpoints (e.g. a pretrain
  checkpoint lacks the finetune-only `eps_cond` keys).
- Your model carries training-only modules (e.g. PCL feature nets) not needed at inference,
  which would otherwise cause "unexpected key" errors.

Add the model name to the `_non_strict` set:
```python
_non_strict = {"miudiff", "unitddpm", "yourmodel"}
strict = args.model not in _non_strict
```

---

## 7. Multi-Stage Models

Some models require staged training (e.g. pretrain → finetune). The pattern used by MIUDiff
and UVCGAN:

1. Add a `stage` (or equivalent) field to your config dataclass.
2. Handle stage-specific dataset selection in `train.py` (`main()`):
   - Stage 1 pretrain (target-domain only): use `TargetOnlyDataset(root_B=...)`
   - All other stages: use `UnpairedDataset(root_A=..., root_B=...)`
3. Add a `--mymodel_stage` CLI argument with `choices=["pretrain", "finetune"]`.
4. Add a `--mymodel_init_ckpt` argument and handle weight transfer logic inside `build_model()`
   (directly in `train.py`, like UVCGAN, or via a helper function, like `init_miudiff_from_stage1`).
5. Keep multi-stage models **out of** `_CONFIG_CLS` / `_MODEL_CLS` — handle them in the
   `if args.model == "mymodel":` block inside `build_model()` instead, because their
   init logic is more complex than a simple `Model(config)` call.

---

## 8. Image & Tensor Conventions

| Convention | Detail |
|------------|--------|
| Pixel range during training | `[-1, 1]` (normalized by `datasets/transforms.py`) |
| Pixel range when saving | `[0, 1]` — use `save_tile(y, path, color_ref_stats)` in `inference.py`; never call `save_image` directly from the loop |
| Spatial resolution | Always 256×256 after `default_train_transform` |
| Channels | 3-channel RGB throughout |
| Batch dict keys | `"A"` and `"B"` (unpaired dataset); `"B"` only (target-only dataset) |
| AMP | `BaseTrainer` wraps forward passes in `autocast`. Standard GAN models benefit from this. **DDPMUNet-based models must disable AMP** — add to `_ddpm_models` in `train.py` (see Section 4.2). |

---

## 9. Checkpoint Format

Checkpoints are saved by `BaseTrainer.save_checkpoint()` as:
```python
{
    "global_step": int,
    "model": model.state_dict(),
    "opt_G": opt_G.state_dict(),
    "opt_D": opt_D.state_dict(),        # only if discriminator exists
    "config": asdict(model.cfg),        # JSON-serializable dict
    "model_name": str,
    "accumulated_training_seconds": float,
}
```

Config is **automatically restored** on `--init_ckpt` (train) and `--ckpt` (inference) via
`MyModelConfig(**saved_cfg)`. This is why all config fields must be JSON-safe primitives with
defaults.

---

## 10. Dataset Variants

| Dataset class | When to use |
|---------------|-------------|
| `UnpairedDataset(root_A, root_B, ...)` | Default — all GAN models, finetune stages |
| `TargetOnlyDataset(root_B, ...)` | Pretrain stages that only need domain B (e.g. MIUDiff stage 1) |
| `SingleDomainDataset(root, ...)` | Inference only — single domain, preserves filenames |

All three support `data_range=(start, end)` to load from numbered subfolders `001/` … `00N/`.
Without `data_range`, they walk the full directory recursively.

Shared image-listing utilities (`IMG_EXTS`, `list_images()`, `list_images_from_range()`) live in
`datasets/common.py` and are imported by all three dataset classes. Do not redefine them in new
dataset files — import from there instead.

---

## 11. Complete Checklist

```
[ ] models/<mymodel>.py created
      [ ] @dataclass MyModelConfig — all fields have JSON-safe defaults
      [ ] self.cfg = cfg in __init__
      [ ] generator_parameters() returns all generator params
      [ ] discriminator_parameters() returns all discriminator params (or omit entirely)
      [ ] compute_generator_loss() returns (scalar Tensor, dict[str,float], dict[str,Tensor])
      [ ] compute_discriminator_loss() returns (scalar Tensor, dict[str,float])
      [ ] forward_A2B() / forward_B2A() defined (used by inference.py)

[ ] train.py
      [ ] Import MyModel, MyModelConfig
      [ ] Standard single-stage model: added to _CONFIG_CLS and _MODEL_CLS
            + Branch in _build_default_gan_config()
          Multi-stage / diffusion model: direct if-block in build_model() instead (Section 7)
      [ ] Branch in _count_a2b_params() — count ONLY inference-path params for A→B
      [ ] "mymodel" added to --model choices
      [ ] Model-specific CLI args added (prefixed --mymodel_*)
      [ ] Diffusion model: added to _ddpm_models set to disable AMP/GradScaler (Section 4.2)
      [ ] Multi-stage model: stage-1 pretrain routed to TargetOnlyDataset in main()

[ ] inference.py
      [ ] Import MyModel, MyModelConfig
      [ ] Branch in load_model() — override runtime-only fields (steps, stage) from CLI args
      [ ] Model with optional submodules: added to _non_strict set (Section 6.5)
      [ ] Branch in inference loop using save_tile (not save_image) (Section 6.3)
      [ ] "mymodel" added to --model choices

[ ] model_sizes.md
      [ ] Small / medium / large A→B param configs added
      [ ] Note explaining which submodules count toward A→B (if not the full model)
      [ ] Entry in CLI args reference table
      [ ] Full example command(s) in the examples section

[ ] Smoke test — training
      python train.py --model mymodel --count_params
      python train.py --model mymodel --dataA /tmp/A --dataB /tmp/B \
          --steps 10 --log_steps 5 --save_steps 10 --output /tmp/test_run

[ ] Smoke test — inference
      python inference.py --model mymodel --direction A2B \
          --data /tmp/A --ckpt /tmp/test_run/checkpoints/step_10.pt --outdir /tmp/out

[ ] Smoke test — resume
      python train.py --model mymodel --dataA /tmp/A --dataB /tmp/B \
          --steps 20 --log_steps 5 --save_steps 10 --output /tmp/test_run
      # Should resume from step 10 and train to step 20
```

---

## 12. Reference: Existing Model Summary

| Model | Config class | Generator structure | Discriminator | A→B inference params | Special |
|-------|-------------|---------------------|---------------|----------------------|---------|
| CycleGAN | `CycleGANConfig` | `Enc_A→Bn_A→Dec_B`, `Enc_B→Bn_B→Dec_A` | `D_A`, `D_B` | `Enc_A + Bn_A + Dec_B` | Identity loss, ImagePool |
| UNIT | `UNITConfig` | Shared `bn_shared` bottleneck, KL on latent | `D_A`, `D_B` | A-side enc + shared + B-side dec | VAE reparameterization |
| MUNIT | `MUNITConfig` | Content enc + style enc + AdaIN decoder | `D_A`, `D_B` | `Ec_A + Bn_A + AdaIN_B + Dec_B` | Style sampling at inference |
| DCLGAN | `DCLGANConfig` | `Enc_A→Bn_A→Dec_B` + contrastive heads | `D_A`, `D_B` | `Enc_A + Bn_A + Dec_B` | Dual patch contrastive loss; `G_A2B`/`G_B2A` return `(image, feats)` tuples — use `forward_A2B`/`forward_B2A` for plain tensor access |
| UVCGAN | `UVCGANConfig` | UNet-ViT hybrid `G_A2B`, `G_B2A` | `D_A`, `D_B` | `G_A2B` | 2-stage: masked pretrain → cycle finetune |
| MIUDiff | `MIUDiffConfig` | `eps_uncond` + `eps_cond` (DDPM UNets) | None | `eps_uncond + eps_cond` (both; classifier-free guidance) | 3-stage, diffusion sampling, PCL; AMP disabled |
| UNIT-DDPM | `UNITDDPMConfig` | `eps_uncond` + `eps_cond` (DDPM UNets) | None | `eps_cond` only (no CFG at inference) | 2-stage; full RGB or gray source conditioning; AMP disabled |
| CycleDiffusion | `CycleDiffusionConfig` | `eps_A` + `eps_B` (DDPM UNets) | None | `eps_A + eps_B` (both; DDIM invert + decode) | Single training stage; symmetric A↔B; AMP disabled |
| UNSB | `UNSBConfig` | `z_theta` (6-ch DDPM UNet) | `D_adv` | `z_theta` only (disc is training-only) | SB forward from xA; adversarial + score-matching; AMP disabled |
