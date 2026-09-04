# I2I-Stain-Zoo

Six unpaired image-to-image translation architectures behind one training and
inference interface, for virtual histological staining of whole-slide images
(H&E → Sirius Red on mouse liver).

**CycleGAN · UNIT · MUNIT · DCLGAN · UVCGAN · CycleDiffusion**

## What this gives you

- **An installable package.** `pip install -e .` gives you
  `from i2i_stain_zoo.models import CycleGAN` in any project, plus `i2i-train`
  and friends on your `PATH` — no `PYTHONPATH` juggling, no working-directory
  assumptions.
- **One interface for six families.** The same `i2i-train` / `i2i-inference`
  call drives every architecture — swap `--model` and the size flags. No
  per-model scripts, config trees or framework glue.
- **Generator capacity as a dial.** Each family exposes flags that scale its
  generator from ~10 M to ~100 M parameters, so capacity and data volume can be
  varied independently.
- **A WSI pipeline, not just a tile model.** Tiling with tissue filtering,
  step-based training that stays comparable across dataset sizes, tiled
  inference, and stitching back to whole slides.
- **Evaluation on three independent axes.** Perceptual (Patch-SSIM, LPIPS),
  distributional (FID), and task-specific (collagen proportionate area via a
  frozen nnU-Net segmenter) — plus deep-ensemble epistemic uncertainty and the
  tooling to check whether that uncertainty actually tracks error.
- **From-scratch implementations.** No external GAN or diffusion libraries;
  DDPM/DDIM sampling included. Shared building blocks live in
  `i2i_stain_zoo/base_models.py`, so adding a seventh family means implementing
  four methods.

Everything runs as a single command — no scheduler, no cluster assumptions.

## Paper

> **Towards Reliable AI-Based Histological Staining: A Systematic Study of
> Scaling and Uncertainty in Unpaired Generative Models**
> Qasim Siddiqui, Adrian Friebel, Maiju Myllys, Zaynab Hobloss, Daniela Gonzalez,
> Ahmed Ghallab, Stefan Hoehme
> **BMVC 2026** · [arXiv:2608.24626](https://arxiv.org/abs/2608.24626)

The paper benchmarks all six families across 54 configurations (6 models × 3
generator sizes × 3 data fractions) and retrains the best per family as a deep
ensemble. Its finding: perceptual quality, task-specific error and ensemble
agreement measure largely independent axes of model fitness, so no single metric
is sufficient for model selection.

The exact settings behind those numbers — step counts, data ranges, the best
configuration per family — are recorded under **Study Parameters** in
[`CLAUDE.md`](CLAUDE.md), if you want to match or extend them.

---

## Install

Python ≥ 3.9, and a CUDA-capable GPU for training.

```bash
conda create -n i2i-stain-zoo python=3.11
conda activate i2i-stain-zoo

# PyTorch first — pick the build matching your CUDA version (see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# This package, in editable mode — your edits take effect immediately
pip install -e .

pytest tests/ -q          # 100 tests, ~2 s, CPU only
```

`pip install -e ".[cpa]"` adds `nnunetv2`, needed only for the
collagen-segmentation metric; `".[dev]"` adds pytest.

To reuse the code from another project, install it from wherever you cloned it:

```bash
pip install -e /path/to/I2I-Stain-Zoo
```

### Two ways to call it

Every command below is a console script installed on your `PATH` and runnable
from any directory. Each is also available as a module, which is handy when you
want a specific interpreter:

```bash
i2i-train --model cyclegan …
python -m i2i_stain_zoo.cli.train --model cyclegan …    # equivalent
```

### As a library

```python
from i2i_stain_zoo.models import CycleGAN, CycleGANConfig
from i2i_stain_zoo.datasets.unpaired_dataset import UnpairedDataset
from i2i_stain_zoo.trainer.base_trainer import BaseTrainer
from i2i_stain_zoo.base_models import Encoder, Decoder, ResnetBottleneck

model = CycleGAN(CycleGANConfig(ngf=128, n_blocks=10))
```

The `base_models` module is the useful one to build on: the
encoder/bottleneck/decoder stack, a 70×70 PatchGAN discriminator, `ImagePool`,
`GANLoss`, InfoNCE and patch sampling, plus the DDPM UNet and noise schedule.

---

## Data

Tile your WSIs into numbered per-slide folders. Domain A is the source stain
(H&E), domain B the target (Sirius Red); training is unpaired, so the two are
never matched tile-for-tile.

```bash
i2i-tile --rgb /path/to/wsi/ --mask /path/to/tissue_masks/ \
    --output /path/to/tiles/ --image_type trainA \
    --tile_size 512 --resize_to 256 --tissue_threshold 0.5
```

Repeat for `trainB`, `testA`, `testB` (use `--tissue_threshold 0` on test data
to keep every tile). The result:

```
tiles/trainA/
  001/
    images/              0000001.tif, 0000002.tif, …
    masks/               tissue masks (if --mask was given)
    tiles_metadata.csv   tile coordinates, read by reconstruct.py
  002/
  …
```

| Flag | Default | Meaning |
|---|---|---|
| `--rgb` | *required* | Root containing an `{image_type}/` subfolder of WSI `.tif`s |
| `--output` | *required* | Root for the tiles |
| `--mask` | `None` | Root containing an `{image_type}/` subfolder of tissue masks |
| `--image_type` | `trainA` | Split name: `trainA`, `trainB`, `testA`, `testB` |
| `--tile_size` | `256` | Tile side extracted from the WSI |
| `--resize_to` | `None` | Downsample tiles to this size before saving |
| `--tissue_threshold` | `0.5` | Minimum tissue fraction to keep a tile (`0` = keep all) |
| `--num_workers` | all CPUs | Worker processes |

Tiling is non-overlapping. Re-running on an existing output directory resumes at
the next free slide index.

Use `--data_range START,END` in training and inference to select a contiguous
range of these numbered folders — this is how you vary the amount of training
data without changing anything else.

---

## Training

```bash
i2i-train \
    --model  cyclegan \
    --dataA  /path/to/tiles/trainA \
    --dataB  /path/to/tiles/trainB \
    --steps  1000000 \
    --amp \
    --output ./runs/cyclegan/
```

Training is **step-based**, not epoch-based, so runs stay comparable across
datasets of different sizes. Checkpoints land in
`./runs/cyclegan/checkpoints/step_{N}.pt`, and training **auto-resumes** from the
newest one it finds — re-running the same command after an interruption always
does the right thing, and exits immediately once `--steps` is reached.

Check a configuration's size before spending GPU time on it:

```bash
i2i-train --model cyclegan --cyclegan_ngf 128 --cyclegan_n_blocks 10 --count_params
```

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *required* | `cyclegan` \| `unit` \| `munit` \| `dclgan` \| `uvcgan` \| `cyclediffusion` |
| `--dataA` / `--dataB` | *required* | Source and target domain tile roots |
| `--output` | *required* | Run directory for checkpoints and logs |
| `--steps` | `5000000` | Total optimiser steps |
| `--data_range` | all | `START,END` — train on slide folders `001`…`NNN` only |
| `--amp` | off | Mixed precision (auto-disabled for CycleDiffusion, see below) |
| `--seed` | `None` | Seeds torch, numpy and random — the only thing that varies between ensemble members |
| `--save_steps` | `250000` | Checkpoint frequency |
| `--log_steps` | `1000` | Log frequency; lines look like `[S00001000 \|   12.3s] loss_G:0.4017 …` |
| `--init_ckpt` | `None` | Initialise weights from an existing checkpoint |
| `--count_params` | off | Print the A→B generator parameter count and exit |
| `--batch_size` | `1` | |
| `--lr` | `2e-4` | Adam, β = (0.5, 0.999), linear decay from the halfway point |
| `--num_workers` | `4` | Dataloader workers |

Checkpoints store their own config, so architecture flags do **not** need to be
repeated on `--init_ckpt` or at inference.

### Per-model flags

Every family scales through two or three flags. The S / M / L columns are the
~10 M / ~50 M / ~100 M A→B parameter settings used in the paper.

**CycleGAN** — cycle-consistency over paired Enc → Bottleneck → Dec generators.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--cyclegan_ngf` | `64` | Generator base channels | 64 / 128 / 192 |
| `--cyclegan_n_blocks` | `9` | ResNet blocks in the bottleneck | 8 / 10 / 9 |

**DCLGAN** — dual patch-level contrastive (InfoNCE) feature matching.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--dclgan_ngf` | `64` | Generator base channels | 64 / 128 / 192 |
| `--dclgan_n_blocks` | `9` | ResNet blocks in the bottleneck | 8 / 10 / 9 |
| `--lambda_dcl` | `1.0` | Weight of the contrastive loss | |
| `--dclgan_lambda_cycle` | `10.0` | Cycle-consistency weight | |
| `--dclgan_lambda_identity` | `0.0` | Identity loss weight (off by default) | |
| `--n_patches` | `256` | Patches sampled per image for matching | |
| `--proj_dim` | `256` | Projection-head dimensionality | |

**UNIT** — shared bottleneck plus a KL term on a variational latent.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--unit_ngf` | `64` | Generator base channels | 64 / 128 / 192 |
| `--unit_n_blocks` | `9` | Total bottleneck blocks (pre + shared + post) | 8 / 10 / 9 |
| `--unit_n_blocks_shared` | `3` | Of those, how many are shared between domains | 2 / 2 / 3 |

**MUNIT** — content/style decomposition with AdaIN; style is sampled at inference.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--munit_ngf` | `64` | Generator base channels | 64 / 128 / 192 |
| `--munit_n_content_blocks` | `4` | Content-encoder ResNet blocks | 3 / 5 / 4 |
| `--munit_n_adain_blocks` | `4` | AdaIN ResNet blocks in the decoder | |
| `--style_dim` | `8` | Style-code dimensionality | |

**UVCGAN** — UNet–ViT hybrid, trained in two stages.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--uvcgan_ngf` | `64` | UNet encoder/decoder base channels | 48 / 96 / 128 |
| `--uvcgan_vit_features` | `192` | ViT hidden dimension | 96 / 384 / 384 |
| `--uvcgan_vit_blocks` | `6` | Transformer blocks in the bottleneck | 6 / 6 / 17 |
| `--uvcgan_stage` | `finetune` | `pretrain` (masked-image modelling) or `finetune` (cycle-consistent) | |
| `--uvcgan_init_ckpt` | `None` | Stage-1 checkpoint to start the finetune stage from | |

UVCGAN needs both stages:

```bash
i2i-train --model uvcgan --uvcgan_stage pretrain \
    --dataA … --dataB … --steps 250000 --amp --output ./runs/uvcgan/stage1/

i2i-train --model uvcgan --uvcgan_stage finetune \
    --uvcgan_init_ckpt ./runs/uvcgan/stage1/checkpoints/step_250000.pt \
    --dataA … --dataB … --steps 750000 --amp --output ./runs/uvcgan/stage2/
```

**CycleDiffusion** — two unconditional DDPMs; DDIM-invert the source into a
shared noise code, then decode with the target UNet.

| Flag | Default | Meaning | S / M / L |
|---|---|---|---|
| `--cd_base_channels` | `64` | UNet base channel count | 48 / 84 / 128 |
| `--cd_channel_mult` | `1,2,2,4` | Per-resolution channel multipliers | `1,2,2,4` / `1,2,2,4` / `1,2,4` |
| `--cd_num_res_blocks` | `2` | Residual blocks per resolution | 1 / 2 / 2 |
| `--cd_steps` | `200` | DDIM steps — stored for inference, unused in training | |

> `--amp` is silently disabled for CycleDiffusion: its UNet runs fp32 internally
> and `GradScaler` overflows around 56 k steps. The script prints a notice. The
> parameter count covers both `eps_A` and `eps_B`; both are needed at inference.

---

## Inference

```bash
i2i-inference \
    --model     cyclegan \
    --direction A2B \
    --data      /path/to/tiles/testA \
    --ckpt      ./runs/cyclegan/checkpoints/step_1000000.pt \
    --outdir    ./out/cyclegan/
```

Output mirrors the input tile structure, so `./out/cyclegan/001/images/0000001.tif`
corresponds to `testA/001/images/0000001.tif`.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *required* | Must match the checkpoint |
| `--direction` | *required* | `A2B` or `B2A` (symmetric for every family) |
| `--data` | *required* | Tile root to translate |
| `--ckpt` | *required* | Checkpoint path; architecture is read from it |
| `--outdir` | `results` | Output root |
| `--data_range` | all | `START,END` — translate only these slide folders |
| `--resume` | off | Skip already-written tiles, redoing the most recent one |
| `--seed` | `None` | Deterministic sampling |
| `--cd_steps` | `200` | CycleDiffusion: DDIM inversion + decode steps each way |
| `--num_samples` | `1` | MUNIT: number of random style codes per tile |
| `--style_image` | `None` | MUNIT: take the style code from a reference image instead |
| `--style_dim` | `8` | MUNIT: used only if the checkpoint stores no config |

CycleDiffusion is by far the slowest — `2 × --cd_steps` UNet evaluations per
tile — and is a good candidate for splitting one job per slide with
`--data_range $i,$i`.

Stitch tiles back into whole slides:

```bash
i2i-reconstruct --metadata /path/to/tiles/testA \
    --tile_dir ./out/cyclegan/ --output ./wsi/cyclegan/
```

`--mode` is `rgb` | `mask` | `rgb_and_mask` | `auto`; `--blend` is `average` (the
default) or `overwrite`.

---

## Evaluation

Three axes, deliberately reported together — the paper's central result is that
they do not predict one another.

```bash
# Patch-based SSIM (paired by filename, higher is better)
i2i-evaluate --metric patch_ssim \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ \
    --patch_size 64 --patches_per_image 16 --save_csv patch_ssim.csv

# LPIPS — VGG16 perceptual distance (paired, lower is better)
i2i-evaluate --metric lpips \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ --device cuda

# FID — InceptionV3 pool3 (unpaired, distribution-level, lower is better)
i2i-evaluate --metric fid \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ --device cuda

# Cycle-reconstruction error: A→B'→A', per-pixel MAE in [0,255]
i2i-evaluate --metric regen_error \
    --path_A /path/to/tiles/testA \
    --model cyclegan --ckpt ./runs/cyclegan/checkpoints/step_1000000.pt \
    --overlay_dir ./regen/ --save_error_npy --device cuda
```

Add `--min_tissue_fraction 0.1` to skip background-only tiles; masks are
auto-detected by swapping the `images/` path component for `masks/`, or set
explicitly with `--mask_dir`. `--save_error_npy` writes raw `[H,W]` float32 maps
for downstream analysis. If B→A tiles already exist on disk, `--path_A_regen`
computes the cycle error from them directly — CPU-only, and no `--model` or
`--ckpt` needed.

SSIM and LPIPS assume co-registered pairs. Since the two stains come from
non-adjacent sections, the test slides were aligned with
[VALIS](https://github.com/MathOnco/valis) first; residual misregistration
inflates both.

### Task-specific: collagen proportionate area

The biologically grounded metric. A frozen nnU-Net collagen segmenter is applied
identically to real and virtual Sirius Red **whole slides**, and the collagen
fraction is compared per specimen. Stitch with `i2i-reconstruct`, segment with
`nnUNetv2_predict -d Dataset314_SR_light`, clean up with `i2i-apply-he-mask` and
`i2i-fill-tissue-holes`, then:

```bash
i2i-compare-psr --masks_real ./psr/real/ \
    --masks_generated ./psr/cyclegan/ ./psr/unit/ \
    --labels cyclegan unit --outdir ./psr_comparison/
```

Mask labels are `0` background, `1` tissue, `2` collagen-positive. The headline
number is paired CPA MAE; Pearson r and Spearman ρ over matched slides are
reported alongside it.

---

## Uncertainty

There is no paired ground truth in the unpaired setting, so epistemic
uncertainty is estimated by **deep ensembles**: train one configuration K times
with different `--seed` values into `model_01/ … model_NN/`, run inference per
member, then

```bash
i2i-uncertainty --model cyclegan --data ./ensemble_out/cyclegan/ \
    --output ./uncertainty/ --mask_dir /path/to/tiles/testA --min_tissue_fraction 0.1
```

Members are discovered by globbing `model_*`, so K is however many exist. The
per-pixel value is √(Σ per-channel sample variance) across members, in 0–255
intensity units. Outputs: `raw_npy/` (the input to everything downstream),
`heatmaps/` (qualitative), `mean_rgb/` (the ensemble mean — this is the virtual
stain to report), `summary.json`.

Variance measures *disagreement*, not correctness — a model can be confidently
wrong. `i2i-uncertainty-calibrate` tests how far the two coincide, pairing the
uncertainty maps with cycle-reconstruction error and reporting within-tile
Spearman ρ (are the uncertain pixels the wrong pixels?), across-tile correlation
(are the uncertain tiles the wrong tiles?) and a reliability diagram with ECE.

```bash
i2i-uncertainty-calibrate \
    --uncertainty_dir ./uncertainty/cyclegan/raw_npy/ \
    --error_dirs      ./regen/error_npy/ \
    --mask_dir        /path/to/tiles/testA/001/masks/ \
    --outdir          ./calibration/
```

Cycle error is a **proxy**, not ground truth: when the forward and inverse
generators share a bias, both may ignore the same feature and the round trip
still reconstructs the source despite a poor forward translation.

`i2i-aggregate-uncertainty` reduces maps to a per-tile scalar,
`i2i-aggregate-calibration` pools tiles per model, and
`i2i-plot-uncertainty` compares the distributions across families.

---

## Repository layout

```
i2i_stain_zoo/
  base_models.py             shared blocks (ResNet, PatchGAN, DDPM UNet)
  utils.py                   tiling, reconstruction, device helpers
  models/                    the six architectures
  datasets/                  unpaired / single-domain loaders, transforms
  trainer/                   step-based loop with auto-resume
  cli/                       one module per command, each with a main()
    train.py                 i2i-train
    inference.py             i2i-inference
    evaluation.py            i2i-evaluate    FID, patch-SSIM, LPIPS, cycle error
    tile.py  reconstruct.py  i2i-tile, i2i-reconstruct
    uncertainty*.py          ensemble variance, calibration, aggregation
    compare_psr.py           i2i-compare-psr    collagen area vs. real SR
    apply_he_mask.py         i2i-apply-he-mask
    fill_tissue_holes.py     i2i-fill-tissue-holes
    plot_*.py                metric overview, rank agreement, distributions

pyproject.toml               dependencies and console-script entry points
tests/                       pytest suite (100 tests, CPU-only)
CLAUDE.md                    conventions, study parameters, full flag reference
```

### Adding a model

Implement four methods and register the class — nothing else in the pipeline
changes:

```python
generator_parameters()                        -> params for the generator optimiser
discriminator_parameters()                    -> params for the discriminator optimiser
compute_generator_loss(batch)                 -> (loss, log_dict, visuals)
compute_discriminator_loss(batch, visuals)    -> (loss, log_dict)
```

Drop the module in `i2i_stain_zoo/models/`, export it from that package's
`__init__.py`, and register it in the `--model` choices in
`i2i_stain_zoo/cli/train.py` and `inference.py`.

`BaseTrainer` handles stepping, logging, AMP, checkpointing, auto-resume and
timing. `i2i_stain_zoo/base_models.py` already provides the
encoder/bottleneck/decoder stack, a 70×70 PatchGAN discriminator, `ImagePool`,
`GANLoss`, InfoNCE and patch sampling, plus the DDPM UNet and noise schedule.

---

## Citation

```bibtex
@inproceedings{siddiqui2026staining,
  title     = {Towards Reliable AI-Based Histological Staining: A Systematic
               Study of Scaling and Uncertainty in Unpaired Generative Models},
  author    = {Siddiqui, Qasim and Friebel, Adrian and Myllys, Maiju and
               Hobloss, Zaynab and Gonzalez, Daniela and Ghallab, Ahmed and
               Hoehme, Stefan},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```
