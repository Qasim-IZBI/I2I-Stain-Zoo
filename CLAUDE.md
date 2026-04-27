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
      images/            ← 0000001.tif, 0000002.tif, ...
      masks/             ← 0000001.tif, ... (only if --mask provided)
      tiles_metadata.csv ← stride, overlap, x/y positions for this WSI
    002/
      images/
      masks/
      tiles_metadata.csv
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

# MIUDiff (3-stage): each stage must be a separate --output directory.
# Stage 1 — train eps_uncond: unconditional DDPM on domain B only (no domain A needed,
#            but --dataA is still required by the parser; it is not read during pretrain).
#            Builds a prior over target domain appearance.
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp \
    --miu_stage pretrain --output ./stage1/

# Stage 2 — train eps_cond: conditional translation A→B with MI guidance.
#            --miu_init_ckpt copies the stage-1 eps_uncond weights into eps_cond
#            as a warm start (all except the extra conditioning input channel).
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp \
    --miu_stage finetune \
    --miu_init_ckpt ./stage1/checkpoints/step_500000.pt \
    --output ./stage2/

# Stage 3 — finetune with patch contrastive loss (PCL) for structural sharpness.
#            --miu_init_ckpt loads the fully-trained stage-2 checkpoint directly
#            (does NOT re-copy eps_uncond → eps_cond).
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp \
    --miu_stage finetune --miu_pcl --lambda_pcl 0.1 \
    --miu_init_ckpt ./stage2/checkpoints/step_500000.pt \
    --output ./stage3/

# MIUDiff UNet architecture controls
# Option A (default): original channel multipliers, 2 ResBlocks per level
python train.py --model miudiff --miu_base_channels 64 --miu_channel_mult 1,2,2,4 --miu_num_res_blocks 2 \
    --dataA ... --dataB ... --steps 500000 --amp --miu_stage pretrain --output ./out/

# Option B: simpler 3-level, 1 ResBlock per level (more stable, faster per step)
python train.py --model miudiff --miu_base_channels 112 --miu_channel_mult 1,2,4 --miu_num_res_blocks 1 \
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

# ---- MIUDiff inference (3 modes) ----

# Stage 1 — unconditional sampling from domain B (pretrain checkpoint).
# Reads tiles from a domain B directory; each real tile provides a filename anchor.
# Output: {outdir}/{stem}_uncond.tif  (one generated sample per B tile found)
python inference.py --model miudiff --miu_stage pretrain --direction A2B \
    --data path/to/tiles/testB --ckpt stage1.pt --miu_steps 200 --outdir ./uncond_out/

# Stage 1 — unconditional sampling, count-based (no domain B directory needed).
# Output: {outdir}/uncond_0000.tif, uncond_0001.tif, ...
python inference.py --model miudiff --miu_stage pretrain --direction A2B \
    --data path/to/tiles/testB --ckpt stage1.pt --num_uncond_samples 50 \
    --miu_steps 200 --outdir ./uncond_out/

# Stage 2/3 — conditional A→B translation (finetune checkpoint, default behaviour).
# Uses eps_cond with optional MI guidance and PCL refinement.
# --miu_stage finetune is the default and can be omitted.
python inference.py --model miudiff --miu_stage finetune --direction A2B \
    --data path/to/tiles/testA --ckpt stage2.pt --miu_steps 200 \
    --miu_guidance 1.0 --outdir ./cond_out/

# Stage 2/3 with PCL latent refinement enabled
python inference.py --model miudiff --miu_stage finetune --direction A2B \
    --data path/to/tiles/testA --ckpt stage3.pt --miu_steps 200 \
    --miu_pcl --pcl_refine_steps 3 --outdir ./cond_pcl_out/
```

**MIUDiff inference notes:**
- `--miu_stage pretrain` uses only `eps_uncond` (unconditional DDPM on target domain B). No source image is fed into the network; samples are drawn from pure noise. MI guidance and PCL refinement are not applicable and are silently ignored.
- `--miu_stage finetune` (default) uses `eps_cond` conditioned on the grayscale source image, with optional MI guidance (`--miu_guidance`) and optional PCL latent refinement (`--miu_pcl --pcl_refine_steps`).
- `--miu_steps` controls the number of DDIM denoising steps for both stages (fewer = faster but lower quality; 200–300 is typical).
- Pretrain checkpoints contain random-weight `eps_cond` parameters. Passing `--miu_stage finetune` to a pretrain checkpoint will produce garbage — a warning is printed if this is detected.
- Finetune checkpoints have a fully-trained `eps_uncond` (updated alongside `eps_cond`). Running `--miu_stage pretrain` on a finetune checkpoint is valid and samples from the updated unconditional model.

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
# Reconstruct WSI from original tiles — pass the dataset directory; all per-WSI CSVs are found automatically
python reconstruct.py --metadata path/to/tiles/trainA --output ./reconstructed/

# Reconstruct from translated tiles (e.g. inference output directory)
python reconstruct.py --metadata path/to/tiles/testA \
    --tile_dir ./inference_output/ --output ./reconstructed/

# Or pass a single per-WSI CSV directly
python reconstruct.py --metadata path/to/tiles/trainA/001/tiles_metadata.csv --output ./reconstructed/

# Reconstruct both RGB and mask, with average blending for overlapping tiles
python reconstruct.py --metadata path/to/tiles_metadata.csv --output ./reconstructed/ \
    --mode rgb_and_mask --blend average
```

### Visual Inference Grid (all 54 runs)
Runs A→B inference on a sample of testA tiles for every combination of model,
model size, and data size, then saves one 6-row (model type) × 3-col figure per
size level. Checkpoints are located automatically from the standard output tree.

```bash
# Default: 3 figures (one per model size), columns = data sizes, 3 tiles/cell
python vis_inference.py

# Flip axes: 3 figures (one per data size), columns = model sizes
python vis_inference.py --group_by model_size

# Show source image row above translated row in each cell
python vis_inference.py --show_source

# More tiles per cell and faster MIUDiff diffusion
python vis_inference.py --num_images 5 --miu_steps 30

# Generate only specific size levels (model sizes when --group_by data_size)
python vis_inference.py --sizes small large

# Dry-run: print which checkpoints would be used without running inference
python vis_inference.py --dry_run

# Custom paths
python vis_inference.py --base /path/to/Outputs --data /path/to/testA --outdir ./vis_out/
```

Output files: `{outdir}/vis_{group_by}_{size}.png` — 3 files total (one per small/medium/large).
Multi-stage models: miudiff uses `stage3/`, uvcgan uses `stage2/` checkpoints.
Missing checkpoints render as a grey placeholder so the grid always completes.

### Loss Plots — All 54 Runs
Reads `loss_log.csv` files from every combination of model, model size, and data
size and produces a single 2×3 figure (one subplot per model type). Color encodes
model size; line style encodes data size. Multi-stage models (miudiff, uvcgan)
have their stage losses concatenated on a single x-axis.

```bash
# Plot all 54 runs with default base path
python plot_all_losses.py

# Custom output path
python plot_all_losses.py --out /path/to/all_losses.png
```

Output: one PNG at `--out` (default: `$BASE/all_losses.png`).
Legend: color = model size (blue/orange/green), line style = data size (solid/dashed/dotted).

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
