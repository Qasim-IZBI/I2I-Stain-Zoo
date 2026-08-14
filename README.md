# I2I-Stain-Zoo

Unpaired image-to-image translation for virtual histological staining
(H&E → Sirius Red) on mouse liver whole-slide images.

Six unsupervised architectures share a single training / inference / evaluation
interface: **CycleGAN**, **UNIT**, **MUNIT**, **DCLGAN**, **UVCGAN**, and
**CycleDiffusion**.

The repository accompanies *Towards Reliable AI-Based Histological Staining: A
Systematic Study of Scaling and Uncertainty in Unpaired Generative Models*, and
reproduces both the 54-configuration scaling study and the deep-ensemble
uncertainty analysis.

---

## Table of contents

1. [Environment and installation](#1-environment-and-installation)
2. [Data preparation](#2-data-preparation)
3. [Training](#3-training)
4. [Inference](#4-inference)
5. [Evaluation](#5-evaluation)
6. [Uncertainty: ensemble training and evaluation](#6-uncertainty-ensemble-training-and-evaluation)
7. [Repository layout](#7-repository-layout)

---

## 1. Environment and installation

Python ≥ 3.10 and a CUDA-capable GPU are recommended. Training was run on
NVIDIA A100 GPUs; inference and all CPU-only evaluation steps run on a laptop.

```bash
conda create -n i2i-stain-zoo python=3.11
conda activate i2i-stain-zoo

# PyTorch — pick the build matching your CUDA version (see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install numpy scipy pandas matplotlib pillow tifffile tqdm pytest
```

There is no `requirements.txt`; the full dependency set is exactly:

| Package | Used for |
|---|---|
| `torch`, `torchvision` | all models, FID (InceptionV3), LPIPS (VGG16) |
| `numpy`, `scipy` | metrics, correlations, uncertainty maps |
| `pandas` | tile metadata, per-tile/per-WSI CSVs |
| `matplotlib` | all figures |
| `pillow`, `tifffile` | tile and WSI I/O |
| `tqdm` | progress bars |
| `pytest` | test suite |

Verify the install:

```bash
pytest tests/ -q          # 239 tests, ~5 s on CPU
```

**Optional — collagen segmentation.** The task-specific CPA metric
(Section 5.3) calls a frozen [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
model (`Dataset314_SR_light`) via `nnUNetv2_predict`. It is only needed for
CPA evaluation:

```bash
pip install nnunetv2
export nnUNet_results=/path/to/nnunet/results
```

---

## 2. Data preparation

Tile the whole-slide images into non-overlapping 512×512 patches, downsampled
to 256×256 (the protocol used throughout the study):

```bash
python tile.py \
    --rgb        /path/to/wsi/ \
    --mask       /path/to/tissue_masks/ \
    --output     /path/to/tiles/ \
    --image_type trainA \
    --tile_size  512 \
    --resize_to  256 \
    --tissue_threshold 0.5
```

Repeat for `trainB`, `testA`, `testB`. Use `--tissue_threshold 0` for test
sets (no tissue filtering is applied to held-out data).

Resulting layout — one numbered subfolder per WSI:

```
tiles/
  trainA/
    001/
      images/            0000001.tif, 0000002.tif, …
      masks/             tissue masks (if --mask given)
      tiles_metadata.csv tile coordinates, used by reconstruct.py
    002/
  trainB/  testA/  testB/
```

`--data_range START,END` selects a contiguous range of these folders, which is
how the 25 % / 50 % / 100 % training fractions are formed (`1,7`, `1,15`,
`1,30`). Subsets are nested, so data-fraction differences are not confounded by
slide selection.

---

## 3. Training

Training is **step-based**, not epoch-based, so runs are comparable across data
fractions with different tile counts. Checkpoints land in
`{output}/checkpoints/step_{N}.pt`, and `BaseTrainer` auto-resumes from the
latest checkpoint found there — no resume flag needed.

### Single-stage models

CycleGAN, UNIT, MUNIT, DCLGAN, CycleDiffusion:

```bash
python train.py \
    --model   cyclegan \
    --dataA   /path/to/tiles/trainA \
    --dataB   /path/to/tiles/trainB \
    --steps   750000 \
    --amp \
    --output  ./runs/cyclegan/
```

Train on a data fraction with `--data_range 1,7` (25 %) or `1,15` (50 %).

> `--amp` is ignored for CycleDiffusion: its UNet runs in fp32 internally and
> `GradScaler` would overflow. The script prints a notice when this happens.

### UVCGAN (two stages)

```bash
# Stage 1 — masked-image pretraining
python train.py --model uvcgan --uvcgan_stage pretrain \
    --dataA /path/to/tiles/trainA --dataB /path/to/tiles/trainB \
    --steps 250000 --amp --output ./runs/uvcgan/stage1/

# Stage 2 — cycle-consistent finetuning from the stage-1 checkpoint
python train.py --model uvcgan --uvcgan_stage finetune \
    --uvcgan_init_ckpt ./runs/uvcgan/stage1/checkpoints/step_250000.pt \
    --dataA /path/to/tiles/trainA --dataB /path/to/tiles/trainB \
    --steps 500000 --amp --output ./runs/uvcgan/stage2/
```

### Generator capacity

Each family is trained at ~10 M (S), ~50 M (M) and ~100 M (L) A→B parameters.
Check a configuration before committing GPU time:

```bash
python train.py --model cyclegan --cyclegan_ngf 128 --cyclegan_n_blocks 10 --count_params
```

| Model | Small | Medium | Large |
|---|---|---|---|
| CycleGAN / DCLGAN | `ngf 64 n_blocks 8` | `ngf 128 n_blocks 10` | `ngf 192 n_blocks 9` |
| UNIT | `ngf 64 n_blocks 8` (shared 2) | `ngf 128 n_blocks 10` (shared 2) | `ngf 192 n_blocks 9` (shared 3) |
| MUNIT | `ngf 64` content 3 | `ngf 128` content 5 | `ngf 192` content 4 |
| UVCGAN | `ngf 48` ViT 96×6 | `ngf 96` ViT 384×6 | `ngf 128` ViT 384×17 |
| CycleDiffusion | `base 48 mult 1,2,2,4 blocks 1` | `base 84 mult 1,2,2,4 blocks 2` | `base 128 mult 1,2,4 blocks 2` |

Shared hyperparameters: Adam, lr `2e-4`, β = (0.5, 0.999), batch size 1, linear
LR decay from the halfway point.

---

## 4. Inference

All GAN models do a single forward pass per tile. CycleDiffusion runs DDIM
inversion with the source-domain UNet followed by DDIM sampling with the
target-domain UNet (`--cd_steps` each way).

```bash
# GAN models
python inference.py \
    --model     cyclegan \
    --direction A2B \
    --data      /path/to/tiles/testA \
    --ckpt      ./runs/cyclegan/checkpoints/step_750000.pt \
    --outdir    ./out/cyclegan/

# CycleDiffusion — 200 inversion + 200 sampling steps
python inference.py --model cyclediffusion --direction A2B \
    --data /path/to/tiles/testA --ckpt ./runs/cyclediffusion/checkpoints/step_750000.pt \
    --cd_steps 200 --outdir ./out/cyclediffusion/

# MUNIT — sample N style codes per tile
python inference.py --model munit --direction A2B \
    --data /path/to/tiles/testA --ckpt ./runs/munit/checkpoints/step_750000.pt \
    --num_samples 3 --outdir ./out/munit/
```

Useful flags: `--data_range 1,5` (subset of WSIs), `--seed 42` (deterministic
output), `--resume` (skip already-written tiles after an interrupted job).

Stitch translated tiles back into whole-slide images:

```bash
python reconstruct.py \
    --metadata /path/to/tiles/testA \
    --tile_dir ./out/cyclegan/ \
    --output   ./wsi/cyclegan/
```

---

## 5. Evaluation

Three independent axes; the paper's conclusion is that they must be reported
together, since none predicts the others.

### 5.1 Perceptual and distributional

```bash
# Patch-based SSIM (higher is better)
python evaluation.py --metric patch_ssim \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ \
    --patch_size 64 --patches_per_image 16 --save_csv patch_ssim.csv

# LPIPS — VGG16 perceptual distance (lower is better)
python evaluation.py --metric lpips \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ \
    --device cuda --save_csv lpips.csv

# FID — InceptionV3 pool3 features (lower is better)
python evaluation.py --metric fid \
    --path_real /path/to/tiles/testB --path_fake ./out/cyclegan/ \
    --device cuda --save_csv fid.csv
```

Add `--min_tissue_fraction 0.1` to skip background-only tiles. Masks are
auto-detected by swapping `images/` → `masks/` in the tile path, or set
explicitly with `--mask_dir`.

SSIM and LPIPS require co-registered H&E/SR pairs. Because the two stains come
from non-adjacent sections, the test WSIs are aligned with
[VALIS](https://github.com/MathOnco/valis) before these metrics are computed;
residual misregistration inflates both.

### 5.2 Cycle-reconstruction error

A per-pixel error proxy for the unpaired setting: translate A→B′, invert with
the model's own B→A generator, and measure |A − A′|.

```bash
python evaluation.py --metric regen_error \
    --path_A /path/to/tiles/testA \
    --model  cyclegan --ckpt ./runs/cyclegan/checkpoints/step_750000.pt \
    --direction A2B \
    --overlay_dir ./regen/cyclegan/ --save_error_npy --device cuda
```

`--save_error_npy` writes raw `[H,W]` float32 maps to
`{overlay_dir}/error_npy/`, which the calibration analysis consumes
(Section 6.3). If B→A tiles already exist on disk, pass `--path_A_regen` to
skip re-running the model.

### 5.3 Task-specific: collagen proportionate area (CPA)

The primary biologically grounded metric. A frozen nnU-Net collagen segmenter
is applied identically to real and virtual SR whole-slide images, and the
CPA (fraction of tissue pixels labelled collagen) is compared per specimen.

Segmentation runs on **reconstructed whole slides**, not tiles, using
`Dataset314_SR_light` via a direct `nnUNetv2_predict` call.

```bash
# 1. Stitch translated tiles into whole slides
python reconstruct.py --metadata /path/to/tiles/testA \
    --tile_dir ./out/cyclegan/ --output ./recon/cyclegan/

# 2. Segment the reconstructed WSIs
export nnUNet_results=/path/to/nnunet/nnUNet_results
export nnUNet_raw=/path/to/nnunet/nnUNet_raw
nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i ./recon/cyclegan/ \
    -o ./psr/cyclegan/wsi_masks/ \
    -f 0 -tr nnUNetTrainer -c 2d -p nnUNetPlans \
    -npp 1 -nps 1 -device cpu

# 3. Compare against real SR
python compare_psr.py \
    --masks_real      ./psr/real/psr_masks_wsi_final/ \
    --masks_generated ./psr/cyclegan/psr_masks_wsi_final/ \
    --labels          cyclegan \
    --outdir          ./psr_comparison/
```

On SLURM this is `scripts/recon_all_configs.sh` followed by
`scripts/segment_psr_nn_light_all_configs.sh`.

Mask labels: `0` background, `1` tissue, `2` collagen-positive.
`compare_psr.py` reports paired CPA MAE (the headline number), Pearson r and
Spearman ρ over WSIs matched by filename stem.

Two post-processing steps run between (2) and (3) —
`apply_he_mask.py` (zeroes predictions outside the H&E tissue boundary) and
`fill_tissue_holes.py` (fills enclosed background inside the tissue footprint).
Both materially affect CPA; see `scripts/apply_he_mask_all_configs.sh` and
`scripts/fill_tissue_holes_all_configs.sh`.

### 5.4 Combined figures

```bash
python plot_combined_metrics.py \
    --eval_indir /path/to/Eval --psr_indir /path/to/psr_comparison \
    --outdir ./figures/
python plot_ranking_correlation.py \
    --csv ./figures/combined_metrics.csv --outdir ./figures/
```

---

## 6. Uncertainty: ensemble training and evaluation

No paired ground truth exists in the unpaired setting, so epistemic uncertainty
is estimated by **deep ensembles**: train the same configuration K times with
different seeds and measure pixel-wise disagreement. High variance marks inputs
where the unsupervised objective does not pin down a unique answer.

Read it alongside CPA MAE — variance measures disagreement, not correctness.
A model can be confidently wrong.

### 6.1 Train the ensemble

Select the best configuration per family (lowest CPA MAE at 100 % data, FID as
tiebreak), then retrain it K times with `--seed 1 … K`:

```bash
for SEED in $(seq 1 10); do
  MEMBER=$(printf "%02d" $SEED)
  python train.py --model cyclegan \
      --dataA /path/to/tiles/trainA --dataB /path/to/tiles/trainB \
      --cyclegan_ngf 128 --cyclegan_n_blocks 10 \
      --steps 750000 --amp --seed $SEED \
      --output ./ensemble/cyclegan/models/model_${MEMBER}/
done
```

On SLURM, one array job per family — the array size sets K:

```bash
sbatch scripts/train_ensemble_cyclegan.sh
```

> **Note:** the committed array sizes are inconsistent — `dclgan` and
> `cyclediffusion` use `--array=0-9` (K = 10, as reported in the paper) while
> `cyclegan`, `unit`, `munit` and `uvcgan` use `--array=0-4` (K = 5). Set the
> array range deliberately before submitting.

The `model_01/ … model_NN/` naming matters: `uncertainty.py` discovers members
by globbing `model_*` under the ensemble root.

### 6.2 Ensemble inference and uncertainty maps

```bash
# A→B for every member (mirror the model_NN layout in the output)
sbatch scripts/infer_ensemble_cyclegan.sh

# Per-pixel uncertainty across members
python uncertainty.py \
    --model  cyclegan \
    --data   ./ensemble_out/cyclegan/ \
    --output ./uncertainty/ \
    --mask_dir /path/to/tiles/testA --min_tissue_fraction 0.001
```

Per-pixel uncertainty is the square root of the summed per-channel sample
variance (ddof = 1) across members, in 0–255 intensity units. Outputs:

| Output | Contents |
|---|---|
| `raw_npy/` | `[H,W]` float32 maps — **the input to every downstream step** |
| `heatmaps/` | magma PNGs with colourbar (qualitative only) |
| `mean_rgb/` | ensemble-mean prediction, used as the virtual stain |
| `summary.json` | per-image statistics and global normalisation bounds |

Reduce to a per-tile scalar σ̄ (tissue-masked spatial mean) and plot the
per-family distribution:

```bash
python aggregate_uncertainty.py \
    --uncertainty_dir ./uncertainty/cyclegan/raw_npy/ \
    --tiles_metadata  /path/to/tiles/testA \
    --mask_dir        /path/to/tiles/testA \
    --outdir          ./uncertainty/cyclegan/per_wsi_csv/

python plot_uncertainty_boxplot.py --base ./ensemble/ --outdir ./figures/
```

### 6.3 Calibration against cycle error

Does uncertainty actually predict error? Pair the uncertainty maps with the
cycle-reconstruction error maps from Section 5.2.

```bash
# Per-member error maps (SLURM: 6 models × 5 WSIs)
sbatch scripts/compute_ensemble_regen_error.sh

# Calibration, per model per WSI
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty/cyclegan/raw_npy/001/images/ \
    --error_dirs      ./regen/cyclegan/wsi001/error_npy/ \
    --mask_dir        /path/to/tiles/testA/001/masks/ \
    --tiles_metadata  /path/to/tiles/testA \
    --outdir          ./calibration/cyclegan/wsi001/

# Or the whole grid, then pool tiles per model
sbatch scripts/run_calibration_all.sh
python aggregate_calibration.py --base ./ensemble/ --outdir ./calibration_combined/
```

Pass several `--error_dirs` to average error across members before calibrating.
Use `raw_npy/`, never `heatmaps/`.

Reported metrics:

| Metric | Question |
|---|---|
| Within-tile Spearman ρ | Are the uncertain *pixels* the wrong pixels? |
| Across-tile Pearson / Spearman | Are the uncertain *tiles* the wrong tiles? |
| Reliability diagram + ECE | Does binned error track binned uncertainty? |

ρ ≈ +1 means uncertainty co-locates with error; ≈ 0 means it is uninformative;
< 0 means anti-calibrated. Cycle error is a **proxy**, not ground truth: when
the forward and inverse generators share a bias, both may ignore the same
feature and the round trip still reconstructs the source.

The test set should be re-tiled with **zero overlap** before this step, so
border pixels are not double-counted in the pooled ECE and across-tile
statistics.

---

## 7. Repository layout

```
train.py                    unified training entry point
inference.py                unified inference entry point
evaluation.py               FID, patch-SSIM, LPIPS, regen error
tile.py  reconstruct.py     WSI ↔ tile conversion
utils.py  base_models.py    shared blocks (ResNet, PatchGAN, DDPM UNet)

models/                     6 architectures
datasets/                   unpaired / single-domain loaders
trainer/                    step-based training loop, auto-resume
tests/                      pytest suite (239 tests)

uncertainty.py              ensemble variance maps
aggregate_uncertainty.py    per-tile σ̄ → per-WSI CSVs
uncertainty_calibration.py  uncertainty vs. cycle error
aggregate_calibration.py    pool per-WSI results per model
plot_uncertainty_boxplot.py per-family σ̄ distribution

uncertainty_phi/            descriptor-space package (φ_struct)
compute_phi_uncertainty.py  procedural vs. data-exposure split
aggregate_phi_uncertainty.py pool per-WSI φ runs into one cohort result
estimate_floor.py           per-descriptor floor, the go/no-go pilot
stain_sensitivity.py        is the "bias" really the segmenter?

compare_psr.py              CPA agreement vs. real SR
apply_he_mask.py            mask cleanup inside the CPA pipeline
fill_tissue_holes.py        hole filling inside the CPA pipeline

plot_combined_metrics.py    2×2 metric overview
plot_ranking_correlation.py cross-metric rank agreement

scripts/                    63 SLURM job scripts for the full study
```

SLURM scripts resolve Python via `PROJECT_ROOT=I2I-Stain-Zoo`, a path relative
to the **submit directory** — submit from the parent of the repository:

```bash
sbatch I2I-Stain-Zoo/scripts/train_ensemble_cyclegan.sh
```

For the complete flag reference of every entry point, see `CLAUDE.md`.
