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

# Stage 2 with structural reconstruction loss (recommended for structure preservation).
# --miu_lambda_struct adds L1(extract_struct(x0_pred), x_struct) at every timestep,
# giving eps_cond a direct gradient to follow the HE conditioning input.
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp \
    --miu_stage finetune --miu_lambda_struct 1.0 \
    --miu_init_ckpt ./stage1/checkpoints/step_500000.pt \
    --output ./stage2/

# Stage 3 — finetune with patch contrastive loss (PCL) for structural sharpness.
#            PCL is now applied at ALL timesteps during training (t0_prime is
#            inference-only). --miu_lambda_struct can be combined with PCL.
#            --miu_init_ckpt loads the fully-trained stage-2 checkpoint directly
#            (does NOT re-copy eps_uncond → eps_cond).
python train.py --model miudiff --dataA ... --dataB ... --steps 500000 --amp \
    --miu_stage finetune --miu_pcl --lambda_pcl 0.1 --miu_lambda_struct 1.0 \
    --miu_init_ckpt ./stage2/checkpoints/step_500000.pt \
    --output ./stage3/

# MIUDiff UNet architecture controls
# Option A (default): original channel multipliers, 2 ResBlocks per level
python train.py --model miudiff --miu_base_channels 64 --miu_channel_mult 1,2,2,4 --miu_num_res_blocks 2 \
    --dataA ... --dataB ... --steps 500000 --amp --miu_stage pretrain --output ./out/

# Option B: simpler 3-level, 1 ResBlock per level (more stable, faster per step)
python train.py --model miudiff --miu_base_channels 112 --miu_channel_mult 1,2,4 --miu_num_res_blocks 1 \
    --dataA ... --dataB ... --steps 500000 --amp --miu_stage pretrain --output ./out/

# MIUDiff conditioning feature type (--miu_cond_type, finetune stages only)
# Controls what structure map is extracted from the source image xA and fed to eps_cond.
# 'gray'  (default) — grayscale of xA;  1 output channel
# 'sobel'           — Sobel gradient magnitude of grayscale xA;  1 output channel
# cond_type is saved in the checkpoint and restored automatically at inference — no
# inference flag needed.  To add a new type: add a branch in MIUDiff._extract_struct
# (models/miudiff.py) and update the --miu_cond_type help string in train.py.
python train.py --model miudiff --miu_stage finetune --miu_cond_type sobel \
    --miu_init_ckpt ./stage1/checkpoints/step_700000.pt \
    --dataA ... --dataB ... --steps 900000 --amp --output ./stage2/

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

# Deterministic output — same input always produces the same tile (all models)
python inference.py --model miudiff --miu_stage finetune --direction A2B \
    --data path/to/tiles/testA --ckpt stage3.pt --miu_steps 200 \
    --seed 42 --outdir ./cond_out/
```

**MIUDiff inference notes:**
- `--miu_stage pretrain` uses only `eps_uncond` (unconditional DDPM on target domain B). No source image is fed into the network; samples are drawn from pure noise. MI guidance and PCL refinement are not applicable and are silently ignored.
- `--miu_stage finetune` (default) uses `eps_cond` conditioned on the grayscale source image, with optional MI guidance (`--miu_guidance`) and optional PCL latent refinement (`--miu_pcl --pcl_refine_steps`).
- `--miu_steps` controls the number of DDIM denoising steps for both stages (fewer = faster but lower quality; 200–300 is typical).
- `--seed INT` fixes the global torch RNG before sampling, making all `torch.randn` calls (initial noise + MI guidance) deterministic. Identical runs produce pixel-identical outputs.
- `--miu_noise_level FLOAT` (default 1.0) initialises the starting noise from a partially-noised source image (SDEdit-style) instead of pure Gaussian noise. Not recommended for H&E→SR: values below 1.0 cause HE colour bleed into the output. Use `--color_ref` for colour consistency instead.
- `--miu_cond_type` is not needed at inference — `cond_type` is saved in the checkpoint and restored automatically.
- Pretrain checkpoints contain random-weight `eps_cond` parameters. Passing `--miu_stage finetune` to a pretrain checkpoint will produce garbage — a warning is printed if this is detected.
- Finetune checkpoints have a fully-trained `eps_uncond` (updated alongside `eps_cond`). Running `--miu_stage pretrain` on a finetune checkpoint is valid and samples from the updated unconditional model.

**Colour normalisation (all models)**

Applies Reinhard LAB colour transfer to every output tile so its colour statistics
match a reference target-domain image or dataset. Useful when the model produces
structurally correct but colour-inconsistent outputs (e.g. MIUDiff colour-mode drift).

```bash
# Single reference tile
python inference.py --model miudiff ... \
    --color_ref path/to/trainB/001/images/0000001.tif --outdir ./out/

# Entire trainB dataset as reference (macro-average of per-tile LAB stats)
python inference.py --model miudiff ... \
    --color_ref path/to/trainB/ --outdir ./out/

# Limit reference to a subset of WSIs
python inference.py --model miudiff ... \
    --color_ref path/to/trainB/ --color_ref_data_range 1,10 --outdir ./out/
```

- Works with all 6 models; combine freely with `--seed` and other flags.
- Directory mode: walks `images/` subdirectories (excludes binary `masks/` tiles); falls back to a flat directory scan if no `images/` subdirs are found.
- `--color_ref_data_range` uses the same `start,end` format as `--data_range` and is only relevant when `--color_ref` is a directory.

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

# Regen error with raw per-pixel error maps as .npy (consumed by uncertainty_calibration.py)
python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \
    --direction A2B --overlay_dir ./regen_overlays/ --save_error_npy --device cuda
# Writes <overlay_dir>/error_npy/<stem>.npy alongside heatmaps/ and overlays/.
# --save_error_npy requires --overlay_dir.

# Judge regen error |A − judge(B')| (model-independent error proxy; required for MIUDiff)
# Loads precomputed B' tiles from --path_B_generated (no forward inference here),
# runs an external judge model in --judge_direction (typically B2A) to produce A_judge,
# and computes |A − A_judge|. Pair to A by relative path under each root.
python evaluation.py --metric judge_regen_error \
    --path_A data/HE \
    --path_B_generated ./inference_miudiff/ \
    --judge_model cyclegan --judge_ckpt judge_cyclegan.pt --judge_direction B2A \
    --overlay_dir ./judge_err_miudiff/ --save_error_npy --device cuda
# The judge must be a GAN (cyclegan, unit, munit, dclgan, uvcgan); MIUDiff cannot judge.
# For paper symmetry, use the SAME judge across all 6 architectures under evaluation.

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

### Batch Inference — All 54 Runs (SLURM)
Runs `inference.py` for every combination of model, model size, and data size as
a 54-job SLURM array. Each job finds the latest checkpoint automatically and
writes translated tiles to `{MODEL_DIR}/inference/`.

```bash
sbatch infer_all_54.sh

# Run only a subset of the array (e.g. first 6 jobs, one per model at small/small)
sbatch --array=0-5 infer_all_54.sh
```

Checkpoint resolution:
- Single-stage models (cyclegan, unit, munit, dclgan): highest `step_*.pt` under `checkpoints/`
- miudiff: highest `step_*.pt` under `stage3/checkpoints/`
- uvcgan:  highest `step_*.pt` under `stage2/checkpoints/`

Output per run: `{BASE}/{model}/results/data_{datasize}/model_{size}/inference/`
Logs: `logs_infer/infer_{jobid}_{taskid}.out / .err`
MIUDiff DDIM steps are set by `MIU_STEPS=200` at the top of the script.

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

### Uncertainty Calibration
Pairs ensemble uncertainty maps with cycle-reconstruction error and reduces both
heatmaps to numerical calibration scores: within-tile Spearman ρ, across-tile
Pearson/Spearman, AUSE + sparsification curve, reliability diagram + ECE.
See `uncertainty_notes.md` for the full methodology and paper-writing notes.

**Prerequisite:** the test set should be tiled with `--overlap 0` to avoid
pixel double-counting in pooled metrics (ECE, across-tile ρ). See
`uncertainty_notes.md` §6 for details.

End-to-end workflow (per architecture):

```bash
# (a) Train N ensemble members with different --seed.
python train.py --model cyclegan --dataA path/to/tiles/trainA --dataB path/to/tiles/trainB \
    --steps 5000000 --amp --seed 1 --output ./ensemble/cyclegan/model_01/
# Repeat for --seed 2 ... N into model_02/, model_03/, ...

# (b) Run inference for each member, mirroring the model_01/, model_02/, ... layout.
for i in 01 02 03 04 05; do
  python inference.py --model cyclegan --direction A2B \
      --data path/to/tiles/testA --ckpt ./ensemble/cyclegan/model_${i}/checkpoints/step_5000000.pt \
      --outdir ./ensemble_outputs/cyclegan/model_${i}/
done

# (c) Compute per-pixel ensemble variance (uncertainty maps).
python uncertainty.py --model cyclegan \
    --data ./ensemble_outputs/cyclegan/ \
    --output ./uncertainty_out/

# (d) Compute per-pixel error maps. Two variants:
#
#  (d-self) Self-cycle (only for models with both directions: cyclegan, unit, munit,
#           dclgan, uvcgan). Each member judges its own translation.
python evaluation.py --metric regen_error \
    --path_A path/to/tiles/testA \
    --model cyclegan --ckpt ./ensemble/cyclegan/model_01/checkpoints/step_5000000.pt \
    --direction A2B \
    --overlay_dir ./regen_cyclegan_m01/ --save_error_npy --device cuda
# For ensemble-mean error, repeat (d-self) per member with distinct --overlay_dir.
#
#  (d-judge) External judge — required for MIUDiff (no inverse generator) and
#            recommended for cross-model symmetry: every model judged by the same
#            fixed inverter. Loads pre-translated B' tiles from inference output.
python evaluation.py --metric judge_regen_error \
    --path_A path/to/tiles/testA \
    --path_B_generated ./ensemble_outputs/cyclegan/model_01/ \
    --judge_model cyclegan --judge_ckpt ./judge_cyclegan.pt --judge_direction B2A \
    --overlay_dir ./judge_err_cyclegan_m01/ --save_error_npy --device cuda
# Use the SAME judge checkpoint across all 6 architectures so error maps are
# directly comparable. The judge must be a GAN (any of cyclegan/unit/munit/
# dclgan/uvcgan); pick one trained model and freeze it.

# (e) Run calibration analysis (tissue-only).
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --error_dirs     ./regen_cyclegan_m01/error_npy/ \
    --mask_dir       ./tissue_masks_flat/ \
    --tiles_metadata path/to/tiles/testA \
    --outdir         ./calibration_cyclegan/

# (e′) Ensemble-mean error variant: pass all member error dirs to --error_dirs.
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --error_dirs     ./regen_cyclegan_m01/error_npy/ ./regen_cyclegan_m02/error_npy/ \
                     ./regen_cyclegan_m03/error_npy/ ./regen_cyclegan_m04/error_npy/ \
                     ./regen_cyclegan_m05/error_npy/ \
    --mask_dir       ./tissue_masks_flat/ \
    --tiles_metadata path/to/tiles/testA \
    --outdir         ./calibration_cyclegan/
```

Inputs to `uncertainty_calibration.py`:
- `--uncertainty_dir` — flat directory of `<stem>.npy` from `uncertainty.py` (use `raw_npy/`, **not** `norm_npy/` which is clipped at p1/p99).
- `--error_dirs` — one or more flat directories of `<stem>.npy` from `evaluation.py --save_error_npy`. Multiple dirs are averaged per-pixel before calibration.
- `--mask_dir` — flat directory of tissue mask `<stem>.tif` files (any non-zero pixel = tissue). Required unless `--no_mask` is passed; background inflates Spearman/ECE spuriously.
- `--tiles_metadata` *(optional)* — dataset root containing per-WSI `tiles_metadata.csv` files. Enables `per_wsi.csv` rollup via the `source_file` column.

Key flags:
- `--n_bins 10` — quantile bins for the reliability diagram and ECE.
- `--ause_steps 100` — fraction-removed steps in the sparsification curve.
- `--reliability_sample 4096` — pixels sampled per tile into the global ECE pool (caps memory; set 0 for all pixels).
- `--min_tissue_pixels 256` — skip tiles with fewer tissue pixels than this.

Outputs in `--outdir`:
- `per_tile.csv` — tile_stem, source_wsi, n_tissue_pixels, spearman_rho, pearson_rho_within, mean_u, mean_e, ause
- `per_wsi.csv` — per-WSI rollup (only when `--tiles_metadata` provided)
- `summary.json` — dataset aggregates: within-tile Spearman mean/std/median, across-tile Pearson/Spearman, AUSE mean/std, ECE, reliability bins, sparsification curve arrays, parameter record
- `calibration.png` — 2×2 figure: reliability diagram, sparsification curve, within-tile ρ histogram, across-tile mean(U) vs mean(E) scatter

Interpretation cheatsheet:
- Within-tile Spearman ρ → +1 = uncertain pixels are wrong pixels; ≈0 = uninformative; <0 = anti-calibrated.
- AUSE → 0 = optimal; > 0 = predicted ranking misses high-error pixels.
- ECE → 0 = bins lie on the y=x diagonal after p1–p99 normalisation.
- Across-tile ρ → catches the case where uncertainty is locally calibrated but flat at tile level (useless for triage).

### PSR Positive Area Segmentation
Runs a nnUNet v2 segmentation model on pre-reconstructed Sirius Red WSIs to produce
per-WSI tissue and PSR-positive area masks. Intended for task-based evaluation:
compare PSR segmentation results between real SR WSIs and generated SR WSIs.

**Prerequisite:** reconstruct WSI TIFs from tiles first using `reconstruct.py`.

```bash
# Segment real SR WSIs
python segment_psr.py \
    --data ./reconstructed_real/ \
    --outdir ./psr_masks_real/ \
    --nnunet_results /path/to/nnunet/results \
    --nnunet_dataset 1 \
    --nnunet_config 2d \
    --nnunet_folds all

# Segment generated SR WSIs (e.g. from CycleGAN inference)
python reconstruct.py \
    --metadata /path/to/tiles/testB \
    --tile_dir /path/to/cyclegan/inference/ \
    --output ./reconstructed_generated/
python segment_psr.py \
    --data ./reconstructed_generated/ \
    --outdir ./psr_masks_generated/ \
    --nnunet_results /path/to/nnunet/results \
    --nnunet_dataset 1 --nnunet_config 2d

# Stream nnUNet output live and use specific folds
python segment_psr.py --data ./wsis/ --outdir ./masks/ \
    --nnunet_results /path/to/nnunet/results --nnunet_dataset 1 \
    --nnunet_folds "0 1 2" --verbose
```

Output mask TIF label convention:
- `0` — background
- `1` — tissue (Tissue class)
- `2` — PSR-positive area

Key flags:
- `--nnunet_results` sets `NNUNET_RESULTS`; can be omitted if already set in the environment
- `--nnunet_trainer` overrides the nnUNet trainer class (uses nnUNet default if omitted)
- `--device cuda|cpu|mps` (default: cuda)

### PSR Distribution Comparison
Compares PSR-positive area fraction distributions between real SR and one or more sets of
generated SR masks (output of `segment_psr.py`). Computes Wasserstein-1 distance with
bootstrap 95% CI, KS test, mean difference, and std ratio — all vs. real SR as reference.

```bash
# Compare one generated condition against real SR
python compare_psr.py \
    --masks_real ./psr_masks_real/ \
    --masks_generated ./psr_masks_cyclegan/ \
    --labels cyclegan \
    --outdir ./psr_comparison/

# Compare multiple models at once (produces a single combined plot)
python compare_psr.py \
    --masks_real ./psr_masks_real/ \
    --masks_generated ./masks_cyclegan/ ./masks_unit/ ./masks_munit/ ./masks_dclgan/ \
    --labels cyclegan unit munit dclgan \
    --outdir ./psr_comparison/
```

Outputs in `--outdir`:
- `per_wsi.csv` — one row per WSI: wsi stem, condition, psr_fraction
- `summary.json` — per-condition stats (mean, std, median, min, max) and pairwise metrics vs real
- `comparison.png` — box + individual data point plot, one column per condition

Key flags:
- `--label_tissue INT` / `--label_psr INT` — nnUNet label indices (default: 1 / 2)
- `--n_bootstrap INT` — bootstrap iterations for Wasserstein CI (default: 1000)

Pairwise metrics reported (each generated condition vs. real SR):
- Wasserstein-1 distance + bootstrap 95% CI (WSI-level resampling)
- KS test statistic and p-value
- Mean difference (generated − real): positive = over-estimates PSR
- Std ratio (generated / real): <1 = collapsed variance (mode failure)

### Cross-stain Consistency
Measures spatial agreement between acellular eosinophilic regions in the H&E input
(collagen proxy, via colour deconvolution) and PSR-positive regions in the generated SR
mask. Because H&E and generated SR tiles are pixel-aligned by construction, no
registration is needed. Requires no real SR images.

```bash
# Reconstruct H&E testA tiles into WSIs first
python reconstruct.py --metadata /path/to/tiles/testA --output ./wsis_he/

# Then run cross-stain consistency (PSR masks from segment_psr.py on generated SR)
python cross_stain_consistency.py \
    --he_wsis    ./wsis_he/ \
    --psr_masks  ./psr_masks_cyclegan/ \
    --outdir     ./cross_stain_cyclegan/

# Save H&E collagen proxy masks for visual threshold inspection
python cross_stain_consistency.py \
    --he_wsis ./wsis_he/ --psr_masks ./psr_masks_cyclegan/ \
    --outdir ./cross_stain_cyclegan/ --save_collagen_masks

# Tune deconvolution thresholds if collagen proxy looks wrong
python cross_stain_consistency.py \
    --he_wsis ./wsis_he/ --psr_masks ./psr_masks_cyclegan/ \
    --outdir ./cross_stain_cyclegan/ \
    --eosin_thresh 0.08 --haem_thresh 0.05 --nuclear_dilation 10
```

Outputs in `--outdir`:
- `per_wsi.csv` — wsi stem, Dice, IoU, collagen fraction, PSR+ fraction
- `summary.json` — mean/std Dice and IoU, parameter record
- `consistency.png` — Dice per WSI (dot plot) + H&E collagen fraction vs PSR+ fraction (scatter)
- `collagen_masks/` — H&E collagen proxy TIFs (only with `--save_collagen_masks`)

Collagen proxy pipeline (colour deconvolution + nuclear exclusion):
1. Macenko colour deconvolution → haematoxylin (H) and eosin (E) channels
2. `E > eosin_thresh` → eosinophilic mask (collagen + cytoplasm)
3. Dilate `H > haem_thresh` → nuclear+cytoplasm exclusion mask
4. Subtract exclusion mask → acellular ECM (collagen proxy)

Key tuning flags:
- `--eosin_thresh` — higher = stricter, fewer regions classified as eosinophilic [default: 0.05]
- `--nuclear_dilation` — larger = more cytoplasm excluded around nuclei [default: 8 px]
- `--min_area` — minimum blob size retained in collagen mask [default: 100 px]

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
