# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I2I-Stain-Zoo is an image-to-image translation research codebase for virtual staining of histopathology images (H&E ↔ IHC). It implements 6 models with a unified training/inference interface: CycleGAN, UNIT, MUNIT, DCLGAN, UVCGAN, and MIUDiff (diffusion-based).

## Commands

### Tiling
Tiles are saved per-WSI into numbered subfolders (`001/`, `002/`, ...) under the output directory.
Each folder contains `images/` (RGB tiles) and `masks/` (tissue mask tiles, if provided).
Tile filenames are `{tile_id:07d}.tif`. Running tiling again on the same output directory
automatically resumes from the next available index.

```bash
# Basic tiling (256×256, no overlap, no mask filtering)
python tile.py --rgb path/to/wsi --output path/to/tiles --image_type trainA --tile_size 256

# Extract 512×512 tiles, resize to 256×256, with tissue masks and 25% overlap
python tile.py --rgb path/to/wsi --output path/to/tiles --mask path/to/masks \
    --tile_size 512 --resize_to 256 --image_type trainA --overlap 0.25

# Test set tiling with overlap (all tiles kept, no tissue filtering)
python tile.py --rgb path/to/wsi --output path/to/tiles --image_type testA \
    --tile_size 256 --overlap 0.25
```

Output structure:
```
path/to/tiles/
  trainA/
    001/
      images/   ← 0000001.tif, 0000002.tif, ...
      masks/    ← 0000001.tif, ... (only if --mask provided)
    002/
      images/
      masks/
    tiles_metadata.csv   ← stride, overlap, x/y positions; appended on re-run
```

### Training
Training is step-based (not epoch-based) so comparisons across different dataset sizes are fair.
Logs are printed every `--log_steps` steps with wall time for that interval. Checkpoints are saved every `--save_steps` steps.

```bash
# GAN models (cyclegan, unit, munit, dclgan) — all tiles under trainA/trainB
python train.py --model cyclegan --dataA path/to/tiles/trainA --dataB path/to/tiles/trainB \
    --steps 5000000 --amp --output ./results/

# Train on a subset of WSIs (folders 001–006 only)
python train.py --model cyclegan --dataA path/to/tiles/trainA --dataB path/to/tiles/trainB \
    --data_range 1,6 --steps 5000000 --amp --output ./results/

# Custom log and checkpoint frequency
python train.py --model cyclegan --dataA ... --dataB ... --steps 5000000 --amp \
    --log_steps 500 --save_steps 100000 --output ./results/

# Resume/initialise any model from a pretrained checkpoint
python train.py --model cyclegan --dataA ... --dataB ... --steps 5000000 \
    --init_ckpt ./prev_run/checkpoints/step_250000.pt --output ./new_run/

# MIUDiff (3-stage): pretrain → finetune → finetune+PCL
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp --miu_stage pretrain --output ./stage1/
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp --miu_stage finetune --output ./stage1/
python train.py --model miudiff --miu_stage finetune --miu_pcl --lambda_pcl 0.1 --dataA ... --dataB ... --steps 500000 --amp --output ./stage3/

# MIUDiff UNet architecture controls (original: base_channels=128, channel_mult=1,1,2,2,4,4)
python train.py --model miudiff --miu_base_channels 64 --miu_channel_mult 1,2,2,4 \
    --dataA ... --dataB ... --steps 500000 --amp --miu_stage pretrain --output ./out/

# UVCGAN (2-stage): optional pretrain → finetune
python train.py --model uvcgan --uvcgan_stage pretrain --dataA ... --dataB ... --steps 1000000 --amp --output ./uvcgan_pt/
python train.py --model uvcgan --uvcgan_stage finetune --uvcgan_init_ckpt ./uvcgan_pt/checkpoints/step_1000000.pt \
    --dataA ... --dataB ... --steps 5000000 --amp --output ./uvcgan/

# UVCGAN ViT architecture controls (original: vit_n_blocks=12, vit_features=384)
python train.py --model uvcgan --uvcgan_vit_blocks 6 --uvcgan_vit_features 192 \
    --dataA ... --dataB ... --steps 5000000 --amp --output ./uvcgan/
```

### Inference
```bash
# All tiles under testA/
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --outdir ./output/

# Subset of WSIs (folders 001–003 only)
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --data_range 1,3 --ckpt model.pt --outdir ./output/

# MUNIT with random style sampling
python inference.py --model munit --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --num_samples 3

# MUNIT with style extracted from a reference image
python inference.py --model munit --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --style_image ref.png

# MIUDiff adds --miu_pcl --pcl_refine_steps 3 --miu_steps 200 --miu_guidance 1.0
```

### Evaluation
```bash
# FID (unpaired, distribution-level)
python evaluation.py --metric fid --path_real real_images/ --path_fake generated_images/ --backend inception --device cuda
# Backends: inception (InceptionV3 pool3 2048-d) or dino (DINOv2 768/1024-d)

# SSIM (paired, matched by filename)
python evaluation.py --metric ssim --path_real real_images/ --path_fake generated_images/

# Patch-based SSIM (paired)
python evaluation.py --metric patch_ssim --path_real real_images/ --path_fake generated_images/ --patch_size 64 --patches_per_image 16

# LPIPS (paired, VGG16 perceptual distance, lower=better)
python evaluation.py --metric lpips --path_real real_images/ --path_fake generated_images/ --device cuda

# Cycle reconstruction error A→B'→A' (MAE in [0,255]; correlates with uncertainty)
python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \
    --direction A2B --device cuda

# Regen error with error heatmaps and overlays saved
python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \
    --direction A2B --overlay_dir ./regen_overlays/ --save_csv regen.csv --device cuda

# Save results to CSV (works with any metric)
python evaluation.py --metric ssim --path_real real_images/ --path_fake generated_images/ --save_csv results.csv
```

### Reconstruction
Reconstructed files are saved with the **original WSI filename** (e.g. `slide_001.tif`).
Mask outputs are saved as `{stem}_mask.tif`. Overlapping tiles are averaged by default.

```bash
# Reconstruct WSI from original tiles (uses image_path from metadata CSV)
python reconstruct.py --metadata path/to/tiles/trainA/tiles_metadata.csv --output ./reconstructed/

# Reconstruct from translated tiles (e.g. inference output directory)
# Tiles are matched by tile_name (0000001.tif, etc.) inside --tile_dir
python reconstruct.py --metadata path/to/tiles/testA/tiles_metadata.csv \
    --tile_dir ./inference_output/ --output ./reconstructed/

# Reconstruct both RGB and mask, with average blending for overlapping tiles
python reconstruct.py --metadata path/to/tiles_metadata.csv --output ./reconstructed/ \
    --mode rgb_and_mask --blend average
```

### Training Summary & Loss Plots
```bash
# Plot losses and save hyperparameters from a training run
python plot_training.py --run ./results/
# Outputs: ./results/losses.png, ./results/training_summary.json
```

### Uncertainty Maps
```bash
# Compute epistemic uncertainty from deep ensemble outputs
python uncertainty.py --model cyclegan --data /path/to/cyclegan/output --output ./uncertainty_out

# With log compression, overlays, and custom percentile bounds
python uncertainty.py --model cyclegan --data /path/to/cyclegan/output --output ./uncertainty_out \
    --log-compress --overlays --lower-percentile 1 --upper-percentile 99
```
- Expects ensemble member directories named `model_01/`, `model_02/`, etc. under `--data`
- Computes per-pixel variance (ddof=1) across ensemble RGB predictions, summed across channels
- Global percentile-based normalisation ensures comparable maps across images
- Outputs: `raw_npy/`, `norm_npy/`, `heatmaps/` (magma colormap with colorbar), optional `overlays/`, `summary.json`

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
| UVCGAN | UNet-ViT hybrid with cycle-consistency; optional masked image pretrain |
| MIUDiff | Conditional DDPM with MI guidance, 3-stage training, optional PCL refinement |

### Data Pipeline

- `datasets/unpaired_dataset.py` — training (A+B folders, pseudo-random pairing); supports `data_range`
- `datasets/single_domain_dataset.py` — inference (single folder); supports `data_range`
- `datasets/target_only_dataset.py` — MIUDiff stage 1 pretraining; supports `data_range`
- `datasets/transforms.py` — resize to 256×256, normalize to [-1, 1]
- Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp
- When `data_range=(start, end)` is given, datasets load from `root/{i:03d}/images/` for `i` in `[start, end]`
- Without `data_range`, datasets walk the entire root directory (backward-compatible)

### Key Conventions

- All images normalized to [-1, 1] during training; denormalized to [0, 1] for saving
- AMP (automatic mixed precision) supported via `--amp` flag
- Checkpoints saved as `{"model": state_dict, "config": asdict(cfg), "model_name": str, ...}` in `output/checkpoints/step_<N>.pt`
- Config is auto-restored from checkpoint on `--init_ckpt` (train) and `--ckpt` (inference); old checkpoints without `"config"` fall back to CLI args/defaults
- **Step-based training**: `--steps` (default 5,000,000) controls total optimiser updates; `--save_steps` (default 250,000) controls checkpoint frequency; `--log_steps` (default 1,000) controls loss logging frequency. This keeps model updates constant across different dataset sizes.
- **Auto-resume**: `BaseTrainer.train()` automatically scans `output/checkpoints/` for `step_*.pt` files at startup and resumes from the latest — no extra flags needed. If the target step count is already reached, training exits immediately.
- **Training time tracking**: elapsed time is accumulated across resume sessions and stored in each checkpoint. After every checkpoint save and at the end of training, `output/training_meta.json` is updated with `accumulated_seconds`, `human_readable` (e.g. `2h 15m 30s`), `last_updated_step`, and `avg_seconds_per_1k_steps`.
- **Log format**: `[S00001000 |   12.3s] loss_G:0.4017 ...` — step number and wall time elapsed since the previous log.
- No external diffusion libraries — DDPM/DDIM sampling implemented from scratch in `models/miudiff.py`
- No requirements.txt — core deps: torch, torchvision, numpy, PIL, tifffile, tqdm, pandas, matplotlib
