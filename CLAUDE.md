# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I2I-Stain-Zoo is an image-to-image translation research codebase for virtual staining of histopathology images (H&E ↔ IHC). It implements 5 models with a unified training/inference interface: CycleGAN, UNIT, MUNIT, DCLGAN, and MIUDiff (diffusion-based).

## Commands

### Training
```bash
# GAN models (cyclegan, unit, munit, dclgan)
python train.py --model cyclegan --dataA path/to/trainA --dataB path/to/trainB --epochs 100 --amp --output ./results/

# MIUDiff (3-stage): pretrain → finetune → finetune+PCL
python train.py --model miudiff --dataA ... --dataB ... --epochs 5 --amp --miu_stage pretrain --output ./stage1/
python train.py --model miudiff --dataA ... --dataB ... --epochs 5 --amp --miu_stage finetune --output ./stage1/
python train.py --model miudiff --miu_stage finetune --miu_pcl --lambda_pcl 0.1 --dataA ... --dataB ... --epochs 5 --amp --output ./stage3/
```

### Inference
```bash
python inference.py --model cyclegan --direction A2B --data path/to/images --ckpt model.pt --outdir ./output/
# MUNIT adds --num_samples; MIUDiff adds --miu_pcl --pcl_refine_steps 3 --miu_steps 200 --miu_guidance 1.0
```

### Evaluation
```bash
python evaluation.py --path_real real_images/ --path_fake generated_images/ --backend inception --device cuda
# Backends: inception (InceptionV3 pool3 2048-d) or dino (DINOv2 768/1024-d)
```

## Architecture

### Model Interface

All models implement a common interface consumed by `BaseTrainer` (`trainer/base_trainer.py`):
- `generator_parameters()` → params for generator optimizer
- `discriminator_parameters()` → params for discriminator optimizer (optional)
- `compute_generator_loss(batch)` → `(loss, log_dict, visuals_dict)`
- `compute_discriminator_loss(batch, visuals)` → `(loss, log_dict)`

### Shared Components (`base_models.py`)

Reusable building blocks across all GAN models:
- `Encoder` → `ResnetBottleneck` → `Decoder` pipeline (with `ResnetBlock` internals)
- `NLayerDiscriminator` — PatchGAN (70×70 receptive field)
- `ImagePool` — replay buffer for discriminator stability
- Diffusion components: `DiffusionSchedule`, `DDPMUNet`, `AttentionBlock`, `ResBlock`

### Model-Specific Notes

| Model | Key Mechanism |
|-------|--------------|
| CycleGAN | Cycle-consistency loss, paired Enc→Bn→Dec generators |
| UNIT | Shared bottleneck + KL divergence on variational latent |
| MUNIT | Content/style decomposition with AdaIN; style sampling at inference |
| DCLGAN | Patch-level contrastive feature matching |
| MIUDiff | Conditional DDPM with MI guidance, 3-stage training, optional PCL refinement |

### Data Pipeline

- `datasets/unpaired_dataset.py` — training (A+B folders, pseudo-random pairing)
- `datasets/single_domain_dataset.py` — inference (single folder)
- `datasets/target_only_dataset.py` — MIUDiff stage 1 pretraining
- `datasets/transforms.py` — resize to 256×256, normalize to [-1, 1]
- Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp

### Key Conventions

- All images normalized to [-1, 1] during training; denormalized to [0, 1] for saving
- AMP (automatic mixed precision) supported via `--amp` flag
- Checkpoints saved as `{"model": state_dict}` in `output/checkpoints/`
- No external diffusion libraries — DDPM/DDIM sampling implemented from scratch in `models/miudiff.py`
- No requirements.txt — core deps: torch, torchvision, numpy, PIL, tifffile, tqdm, pandas
