# I2I-Stain-Zoo

Unpaired image-to-image translation for virtual histological staining
(H&E → Sirius Red) on mouse liver whole-slide images.

Six unsupervised architectures share a single training / inference / evaluation
interface: **CycleGAN**, **UNIT**, **MUNIT**, **DCLGAN**, **UVCGAN** and
**CycleDiffusion**.

This repository accompanies

> **Towards Reliable AI-Based Histological Staining: A Systematic Study of
> Scaling and Uncertainty in Unpaired Generative Models**
> Qasim Siddiqui, Adrian Friebel, Maiju Myllys, Zaynab Hobloss, Daniela Gonzalez,
> Ahmed Ghallab, Stefan Hoehme — [arXiv:2608.24626](https://arxiv.org/abs/2608.24626)

and contains everything needed to reproduce its two experiments: the
**54-configuration scaling study** and the **deep-ensemble uncertainty study**.

Every step is a plain `python <script>.py` command. There are no cluster job
scripts and no scheduler dependency — where the paper ran a job array, this
README gives the equivalent shell loop, which you can run serially, split across
GPUs, or wrap in whatever scheduler you have.

---

## Table of contents

1. [What reproduces what](#1-what-reproduces-what)
2. [Install](#2-install)
3. [Dataset and tiling](#3-dataset-and-tiling)
4. [Study 1 — the 54-configuration scaling study](#4-study-1--the-54-configuration-scaling-study)
5. [Study 2 — deep-ensemble uncertainty](#5-study-2--deep-ensemble-uncertainty)
6. [CLI reference](#6-cli-reference)
7. [Repository layout](#7-repository-layout)
8. [Notes and caveats](#8-notes-and-caveats)

---

## 1. What reproduces what

| Paper artefact | Produced by |
|---|---|
| Patch-SSIM / LPIPS / FID per configuration | [§4.4](#44-perceptual-and-distributional-metrics) — `evaluation.py` |
| CPA MAE per configuration | [§4.5](#45-task-specific-metric-collagen-proportionate-area) — `reconstruct.py` → nnU-Net → `apply_he_mask.py` → `fill_tissue_holes.py` → `compare_psr.py` |
| 2×2 metric overview figure | [§4.6](#46-summary-figures) — `plot_combined_metrics.py` |
| Cross-metric rank-agreement matrix | [§4.6](#46-summary-figures) — `plot_ranking_correlation.py` |
| Per-pixel epistemic uncertainty maps | [§5.3](#53-per-pixel-uncertainty-maps) — `uncertainty.py` |
| Per-family uncertainty distributions | [§5.4](#54-per-tile-uncertainty-and-the-family-comparison) — `aggregate_uncertainty.py`, `plot_uncertainty_boxplot.py` |
| Calibration (within-tile ρ, across-tile ρ, ECE) | [§5.6](#56-calibration-against-cycle-error) — `uncertainty_calibration.py`, `aggregate_calibration.py` |

The blinded expert reader study reported in the paper was run outside this
codebase and has no entry point here.

**Compute budget.** The full grid is large: 54 training runs of 1 M optimiser
steps each (UVCGAN: 250 k + 750 k over two stages), plus 60 ensemble runs of
750 k steps. Training used NVIDIA A100 GPUs; a single configuration takes on the
order of a day or more. Inference and *all* evaluation, uncertainty and
calibration steps are cheap and several are CPU-only — if you have the released
checkpoints or generated tiles, everything from [§4.4](#44-perceptual-and-distributional-metrics)
onward reproduces on a laptop.

---

## 2. Install

Python ≥ 3.9 (the study ran on 3.9; 3.10+ is fine) and, for training, a
CUDA-capable GPU.

```bash
conda create -n i2i-stain-zoo python=3.11
conda activate i2i-stain-zoo

# PyTorch first — pick the build matching your CUDA version (see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `torch`, `torchvision` | all models, FID (InceptionV3), LPIPS (VGG16) |
| `numpy`, `scipy` | metrics, correlations, uncertainty maps |
| `pandas` | tile metadata, per-tile / per-WSI CSVs |
| `matplotlib` | all figures |
| `pillow`, `tifffile` | tile and WSI I/O |
| `tqdm` | progress bars |
| `pytest` | test suite |

Verify the install:

```bash
pytest tests/ -q          # 100 tests, ~2 s, CPU only
```

**Optional — collagen segmentation.** The CPA metric ([§4.5](#45-task-specific-metric-collagen-proportionate-area))
calls a frozen [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) model
(`Dataset314_SR_light`) through `nnUNetv2_predict`. It is needed only for CPA:

```bash
pip install nnunetv2
export nnUNet_results=/path/to/nnunet/nnUNet_results
export nnUNet_raw=/path/to/nnunet/nnUNet_raw
```

---

## 3. Dataset and tiling

The paired H&E → Sirius Red mouse liver WSI dataset released with the paper is
organised as four splits: `trainA` / `testA` (H&E, domain A) and `trainB` /
`testB` (Sirius Red, domain B). Training pairs are *unpaired* — the two stains
come from different sections and are never matched during training.

Tile each split into non-overlapping 512 × 512 patches downsampled to 256 × 256
(the protocol used throughout the study):

```bash
# Training splits — drop tiles that are mostly background
for SPLIT in trainA trainB; do
  python tile.py \
      --rgb        /path/to/wsi/ \
      --mask       /path/to/tissue_masks/ \
      --output     /path/to/tiles/ \
      --image_type $SPLIT \
      --tile_size  512 \
      --resize_to  256 \
      --tissue_threshold 0.5
done

# Test splits — keep every tile, no tissue filtering on held-out data
for SPLIT in testA testB; do
  python tile.py \
      --rgb        /path/to/wsi/ \
      --mask       /path/to/tissue_masks/ \
      --output     /path/to/tiles/ \
      --image_type $SPLIT \
      --tile_size  512 \
      --resize_to  256 \
      --tissue_threshold 0
done
```

`--rgb` and `--mask` point at roots that each contain an `{image_type}/`
subfolder of WSI `.tif` files. Tiling is non-overlapping (stride = tile size);
this is the study protocol and is not configurable.

Resulting layout — one numbered subfolder per WSI:

```
tiles/
  trainA/
    001/
      images/              0000001.tif, 0000002.tif, …
      masks/               tissue masks (only if --mask was given)
      tiles_metadata.csv   tile coordinates, read by reconstruct.py
    002/
    …
  trainB/  testA/  testB/
```

Re-running `tile.py` on an existing output directory resumes at the next free
WSI index rather than overwriting.

Throughout the rest of this README:

```bash
TILES=/path/to/tiles          # output of the commands above
TRAIN_A=$TILES/trainA
TRAIN_B=$TILES/trainB
TEST_A=$TILES/testA
TEST_B=$TILES/testB
```

### Data fractions

`--data_range START,END` selects a contiguous range of numbered WSI folders.
The three training fractions are nested subsets of the same slide ordering, so
fraction differences are not confounded by slide selection:

| Data size | `--data_range` | WSIs | Fraction |
|---|---|---|---|
| `small` | `1,7` | 001–007 | 25 % |
| `medium` | `1,15` | 001–015 | 50 % |
| `large` | `1,30` | 001–030 | 100 % |

The test set is 5 WSIs (`--data_range 1,5`).

> **Important for the uncertainty study:** the test set must be tiled with
> **zero overlap** before calibration, so border pixels are not double-counted
> in the pooled ECE and across-tile statistics. The commands above already do
> this; if you re-tile with a smaller stride for qualitative figures, keep the
> non-overlapping tiling as a separate directory for [§5](#5-study-2--deep-ensemble-uncertainty).

---

## 4. Study 1 — the 54-configuration scaling study

6 architectures × 3 generator sizes × 3 data fractions = **54 configurations**,
each evaluated on Patch-SSIM, LPIPS, FID and CPA MAE.

### 4.1 The configuration grid

Generator sizes target ~10 M (S), ~50 M (M) and ~100 M (L) A→B parameters.
Check any configuration before committing GPU time:

```bash
python train.py --model cyclegan --cyclegan_ngf 128 --cyclegan_n_blocks 10 --count_params
```

| Model | S (~10 M) | M (~50 M) | L (~100 M) |
|---|---|---|---|
| CycleGAN | `--cyclegan_ngf 64 --cyclegan_n_blocks 8` | `--cyclegan_ngf 128 --cyclegan_n_blocks 10` | `--cyclegan_ngf 192 --cyclegan_n_blocks 9` |
| DCLGAN | `--dclgan_ngf 64 --dclgan_n_blocks 8` | `--dclgan_ngf 128 --dclgan_n_blocks 10` | `--dclgan_ngf 192 --dclgan_n_blocks 9` |
| UNIT | `--unit_ngf 64 --unit_n_blocks 8 --unit_n_blocks_shared 2` | `--unit_ngf 128 --unit_n_blocks 10 --unit_n_blocks_shared 2` | `--unit_ngf 192 --unit_n_blocks 9 --unit_n_blocks_shared 3` |
| MUNIT | `--munit_ngf 64 --munit_n_content_blocks 3` | `--munit_ngf 128 --munit_n_content_blocks 5` | `--munit_ngf 192 --munit_n_content_blocks 4` |
| UVCGAN | `--uvcgan_ngf 48 --uvcgan_vit_features 96 --uvcgan_vit_blocks 6` | `--uvcgan_ngf 96 --uvcgan_vit_features 384 --uvcgan_vit_blocks 6` | `--uvcgan_ngf 128 --uvcgan_vit_features 384 --uvcgan_vit_blocks 17` |
| CycleDiffusion | `--cd_base_channels 48 --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 1` | `--cd_base_channels 84 --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 2` | `--cd_base_channels 128 --cd_channel_mult 1,2,4 --cd_num_res_blocks 2` |

The CycleDiffusion count covers both `eps_A` and `eps_B`; both UNets are needed
at inference.

Shared hyperparameters (all families): Adam, lr `2e-4`, β = (0.5, 0.999), batch
size 1, linear LR decay from the halfway point to zero. Loss weights —
CycleGAN / UVCGAN λ_cyc 10, λ_id 0.5; UNIT λ_GAN 1, λ_recon 10, λ_KL 0.01;
MUNIT λ_img 10, λ_c 1, λ_s 1; DCLGAN λ_cyc 10, λ_id 0, λ_DCL 1; CycleDiffusion
ε-prediction only.

### 4.2 Training

Training is **step-based**, not epoch-based, so runs are comparable across data
fractions with different tile counts. Checkpoints are written to
`{output}/checkpoints/step_{N}.pt` every `--save_steps` (default 250 000), and
`BaseTrainer` **auto-resumes** from the newest checkpoint it finds there — no
resume flag, just re-run the same command. If the target step count is already
reached the run exits immediately.

Study setting: **1 000 000 steps** for CycleGAN, UNIT, MUNIT, DCLGAN and
CycleDiffusion; UVCGAN uses 250 000 pretrain + 750 000 finetune steps.

A single configuration:

```bash
python train.py \
    --model   cyclegan \
    --dataA   $TRAIN_A \
    --dataB   $TRAIN_B \
    --data_range 1,30 \
    --cyclegan_ngf 128 --cyclegan_n_blocks 10 \
    --steps   1000000 \
    --amp \
    --output  ./runs/cyclegan/data_large/model_medium/
```

The whole grid, as a loop — run it as-is to go serially, or dispatch the body to
whatever scheduler you have. (This and the UVCGAN loop below use associative
arrays, so they need **bash ≥ 4**; macOS ships bash 3.2, where `brew install bash`
or a plain `case` statement will do instead.)

```bash
declare -A RANGE=( [small]=1,7 [medium]=1,15 [large]=1,30 )

declare -A ARCH=(
  [cyclegan:small]="--cyclegan_ngf 64  --cyclegan_n_blocks 8"
  [cyclegan:medium]="--cyclegan_ngf 128 --cyclegan_n_blocks 10"
  [cyclegan:large]="--cyclegan_ngf 192 --cyclegan_n_blocks 9"
  [dclgan:small]="--dclgan_ngf 64  --dclgan_n_blocks 8"
  [dclgan:medium]="--dclgan_ngf 128 --dclgan_n_blocks 10"
  [dclgan:large]="--dclgan_ngf 192 --dclgan_n_blocks 9"
  [unit:small]="--unit_ngf 64  --unit_n_blocks 8  --unit_n_blocks_shared 2"
  [unit:medium]="--unit_ngf 128 --unit_n_blocks 10 --unit_n_blocks_shared 2"
  [unit:large]="--unit_ngf 192 --unit_n_blocks 9  --unit_n_blocks_shared 3"
  [munit:small]="--munit_ngf 64  --munit_n_content_blocks 3"
  [munit:medium]="--munit_ngf 128 --munit_n_content_blocks 5"
  [munit:large]="--munit_ngf 192 --munit_n_content_blocks 4"
  [cyclediffusion:small]="--cd_base_channels 48  --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 1"
  [cyclediffusion:medium]="--cd_base_channels 84  --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 2"
  [cyclediffusion:large]="--cd_base_channels 128 --cd_channel_mult 1,2,4   --cd_num_res_blocks 2"
)

for MODEL in cyclegan unit munit dclgan cyclediffusion; do
  for SIZE in small medium large; do
    for DATA in small medium large; do
      python train.py \
          --model   $MODEL \
          --dataA   $TRAIN_A --dataB $TRAIN_B \
          --data_range ${RANGE[$DATA]} \
          ${ARCH[$MODEL:$SIZE]} \
          --steps   1000000 \
          --amp \
          --output  ./runs/$MODEL/data_$DATA/model_$SIZE/
    done
  done
done
```

**UVCGAN is two-stage** — masked-image pretraining, then cycle-consistent
finetuning from the stage-1 checkpoint:

```bash
declare -A UVC=(
  [small]="--uvcgan_ngf 48  --uvcgan_vit_features 96  --uvcgan_vit_blocks 6"
  [medium]="--uvcgan_ngf 96  --uvcgan_vit_features 384 --uvcgan_vit_blocks 6"
  [large]="--uvcgan_ngf 128 --uvcgan_vit_features 384 --uvcgan_vit_blocks 17"
)

for SIZE in small medium large; do
  for DATA in small medium large; do
    OUT=./runs/uvcgan/data_$DATA/model_$SIZE

    python train.py --model uvcgan --uvcgan_stage pretrain \
        --dataA $TRAIN_A --dataB $TRAIN_B --data_range ${RANGE[$DATA]} \
        ${UVC[$SIZE]} --steps 250000 --amp --output $OUT/stage1/

    python train.py --model uvcgan --uvcgan_stage finetune \
        --uvcgan_init_ckpt $OUT/stage1/checkpoints/step_250000.pt \
        --dataA $TRAIN_A --dataB $TRAIN_B --data_range ${RANGE[$DATA]} \
        ${UVC[$SIZE]} --steps 750000 --amp --output $OUT/stage2/
  done
done
```

> `--amp` is silently disabled for CycleDiffusion: its UNet runs fp32 internally
> and `GradScaler` overflows around 56 k steps. The script prints a notice.

Each run also writes `{output}/training_meta.json` with accumulated wall-clock
time across resumes.

### 4.3 Inference

Translate the 5 test WSIs A→B for every configuration. Output mirrors the input
tile structure, so `$OUT/001/images/0000001.tif` corresponds to
`$TEST_A/001/images/0000001.tif`.

```bash
for MODEL in cyclegan unit munit dclgan; do
  for SIZE in small medium large; do
    for DATA in small medium large; do
      python inference.py \
          --model      $MODEL \
          --direction  A2B \
          --data       $TEST_A \
          --data_range 1,5 \
          --ckpt       ./runs/$MODEL/data_$DATA/model_$SIZE/checkpoints/step_1000000.pt \
          --outdir     ./inference/$MODEL/data_$DATA/model_$SIZE/
    done
  done
done

# UVCGAN — stage-2 checkpoint
for SIZE in small medium large; do
  for DATA in small medium large; do
    python inference.py --model uvcgan --direction A2B \
        --data $TEST_A --data_range 1,5 \
        --ckpt ./runs/uvcgan/data_$DATA/model_$SIZE/stage2/checkpoints/step_750000.pt \
        --outdir ./inference/uvcgan/data_$DATA/model_$SIZE/
  done
done

# CycleDiffusion — DDIM inversion with eps_A, decode with eps_B, 200 steps each way
for SIZE in small medium large; do
  for DATA in small medium large; do
    python inference.py --model cyclediffusion --direction A2B \
        --data $TEST_A --data_range 1,5 \
        --ckpt ./runs/cyclediffusion/data_$DATA/model_$SIZE/checkpoints/step_1000000.pt \
        --cd_steps 200 \
        --outdir ./inference/cyclediffusion/data_$DATA/model_$SIZE/
  done
done
```

`--resume` skips already-written tiles (and redoes the most recent one, in case
it was half-written) if a job is interrupted. `--seed N` makes sampling
deterministic. CycleDiffusion is by far the slowest — 400 UNet evaluations per
tile — and is a reasonable candidate for splitting one job per WSI with
`--data_range $i,$i`.

### 4.4 Perceptual and distributional metrics

Real Sirius Red tiles (`$TEST_B`) are the reference for all three. Patch-SSIM
and LPIPS are **paired**, matched by filename; FID is distribution-level.
`--min_tissue_fraction 0.1` (the study setting) skips tiles that are ≥ 90 %
background; masks are auto-detected by swapping the `images/` path component for
`masks/`.

```bash
for MODEL in cyclegan unit munit dclgan uvcgan cyclediffusion; do
  for SIZE in small medium large; do
    for DATA in small medium large; do
      FAKE=./inference/$MODEL/data_$DATA/model_$SIZE
      OUT=./evaluation/$MODEL/data_$DATA/model_$SIZE
      mkdir -p $OUT

      python evaluation.py --metric fid \
          --path_real $TEST_B --path_fake $FAKE \
          --device cuda --min_tissue_fraction 0.1 --save_csv $OUT/fid.csv

      python evaluation.py --metric patch_ssim \
          --path_real $TEST_B --path_fake $FAKE \
          --patch_size 64 --patches_per_image 16 \
          --device cuda --min_tissue_fraction 0.1 --save_csv $OUT/patch_ssim.csv

      python evaluation.py --metric lpips \
          --path_real $TEST_B --path_fake $FAKE \
          --device cuda --min_tissue_fraction 0.1 --save_csv $OUT/lpips.csv
    done
  done
done
```

This `{model}/data_{data}/model_{size}/{metric}.csv` layout is what
`plot_combined_metrics.py` expects in [§4.6](#46-summary-figures).

SSIM and LPIPS need co-registered H&E/SR pairs. Because the two stains come from
non-adjacent sections, the test WSIs are aligned with
[VALIS](https://github.com/MathOnco/valis) before these metrics are computed;
residual misregistration inflates both.

### 4.5 Task-specific metric: collagen proportionate area

The primary biologically grounded metric. A frozen nnU-Net collagen segmenter is
applied identically to real and virtual SR **whole slides** (not tiles), and the
CPA — the fraction of tissue pixels labelled collagen — is compared per specimen.
Because `testA` and `testB` are serial sections of the same blocks, the
comparison is paired, and **paired CPA MAE is the headline number**.

Mask label convention: `0` background, `1` tissue, `2` PSR-positive.

**Step 1 — stitch tiles into whole slides.**

```bash
for MODEL in cyclegan unit munit dclgan uvcgan cyclediffusion; do
  for SIZE in small medium large; do
    for DATA in small medium large; do
      python reconstruct.py \
          --metadata $TEST_A \
          --tile_dir ./inference/$MODEL/data_$DATA/model_$SIZE/ \
          --output   ./recon/$MODEL/data_$DATA/model_$SIZE/ \
          --mode rgb --blend average
    done
  done
done

# Real SR reference (no --tile_dir: stitches the original testB tiles)
python reconstruct.py --metadata $TEST_B --output ./recon/real/ \
    --mode rgb --blend average

# H&E tissue masks at WSI level — needed in step 3
python reconstruct.py --metadata $TEST_A --output ./he_tissue/ \
    --mode mask --blend average
```

Reconstructed files keep the original WSI filename; mask outputs are
`{stem}_mask.tif`.

**Step 2 — segment collagen.** A direct `nnUNetv2_predict` call with the frozen
`Dataset314_SR_light` model — the same model for generated and real slides.

```bash
export nnUNet_results=/path/to/nnunet/nnUNet_results
export nnUNet_raw=/path/to/nnunet/nnUNet_raw

for IN in ./recon/*/data_*/model_*/ ./recon/real/; do
  OUT=./psr/${IN#./recon/}wsi_masks/
  mkdir -p $OUT
  nnUNetv2_predict \
      -d Dataset314_SR_light \
      -i "$IN" -o "$OUT" \
      -f 0 -tr nnUNetTrainer -c 2d -p nnUNetPlans \
      -npp 1 -nps 1 -device cpu
done
```

`-npp` / `-nps` are nnU-Net worker counts; keep them at 1 on memory-constrained
machines (the study ran this step CPU-only with 256 GB).

**Step 3 — post-process.** Both steps materially affect CPA and are applied to
the real reference as well as to every generated set.

```bash
for D in ./psr/*/data_*/model_*/ ./psr/real/; do
  # Zero out PSR predictions outside the H&E tissue boundary
  python apply_he_mask.py \
      --psr_masks $D/wsi_masks/ \
      --he_masks  ./he_tissue/ \
      --outdir    $D/psr_masks_wsi_cleaned/

  # Fill enclosed background inside the tissue footprint (labels 1+2 as foreground)
  python fill_tissue_holes.py \
      --masks  $D/psr_masks_wsi_cleaned/ \
      --outdir $D/psr_masks_wsi_final/
done
```

**Step 4 — compare CPA.** One `compare_psr.py` call per model, passing all nine
of its configurations at once. The label format
`{model_size}_model/{data_size}_data` is what `plot_combined_metrics.py` parses
in [§4.6](#46-summary-figures).

```bash
for MODEL in cyclegan unit munit dclgan uvcgan cyclediffusion; do
  MASKS=(); LABELS=()
  for SIZE in small medium large; do
    for DATA in small medium large; do
      MASKS+=("./psr/$MODEL/data_$DATA/model_$SIZE/psr_masks_wsi_final/")
      LABELS+=("${SIZE}_model/${DATA}_data")
    done
  done
  python compare_psr.py \
      --masks_real      ./psr/real/psr_masks_wsi_final/ \
      --masks_generated "${MASKS[@]}" \
      --labels          "${LABELS[@]}" \
      --outdir          ./psr_comparison/$MODEL/ \
      --strip_prefix
done
```

`--strip_prefix` drops the first `_`-delimited filename token before matching, so
`SR_slide.tif` and `HE_slide.tif` both pair as `slide`. Outputs per model:
`per_wsi.csv`, `summary.json` (containing `mae_paired`, `pearson_r`,
`spearman_rho` and the signed bias `mean_paired_diff_generated_minus_real`),
`comparison.png`, `paired_scatter.png`, `paired_metrics.png`.

### 4.6 Summary figures

```bash
python plot_combined_metrics.py \
    --eval_indir ./evaluation/ \
    --psr_indir  ./psr_comparison/ \
    --outdir     ./figures/
```

Writes `combined_metrics.png` (2 × 2: Patch-SSIM, LPIPS, FID, CPA MAE — colour =
data size, marker = generator size, error bars = ±1 std across WSIs, star = best
configuration per model) and `combined_metrics.csv` (one row per
model × generator size × data size).

```bash
python plot_ranking_correlation.py \
    --csv ./figures/combined_metrics.csv --outdir ./figures/
```

Ranks all 54 configurations by each metric independently (rank 1 = best,
direction-aware) and computes the pairwise Spearman matrix — this is the
analysis behind the paper's claim that the metrics measure largely independent
axes, so image-quality scores cannot substitute for the CPA pipeline when
selecting a model. Writes `ranking_correlation.png`, `ranking_correlation.csv`
and `ranking_correlation_pvalues.csv`.

### 4.7 Cycle-reconstruction error (optional here, required for §5)

An error proxy for the unpaired setting: translate A→B′, invert with the model's
own B→A generator, and measure |A − A′| per pixel.

```bash
# Directly, re-running the model
python evaluation.py --metric regen_error \
    --path_A $TEST_A \
    --model cyclegan --ckpt ./runs/cyclegan/data_large/model_medium/checkpoints/step_1000000.pt \
    --direction A2B \
    --overlay_dir ./regen/cyclegan/ --save_error_npy --device cuda

# Or from precomputed A′ tiles, with no model inference at all
python inference.py --model cyclegan --direction B2A \
    --data ./inference/cyclegan/data_large/model_medium/ --data_range 1,5 \
    --ckpt ./runs/cyclegan/data_large/model_medium/checkpoints/step_1000000.pt \
    --outdir ./inference_B2A/cyclegan/data_large/model_medium/

python evaluation.py --metric regen_error \
    --path_A $TEST_A --path_A_regen ./inference_B2A/cyclegan/data_large/model_medium/ \
    --overlay_dir ./regen/cyclegan/ --save_error_npy
```

The `--path_A_regen` form is CPU-only, needs neither `--model` nor `--ckpt`, and
works for every family including CycleDiffusion. `--save_error_npy` writes raw
`[H, W]` float32 maps to `{overlay_dir}/error_npy/` — the input to
[§5.6](#56-calibration-against-cycle-error).

---

## 5. Study 2 — deep-ensemble uncertainty

No paired ground truth exists in the unpaired setting, so epistemic uncertainty
is estimated by **deep ensembles**: train the same configuration K times with
different seeds and measure pixel-wise disagreement. High variance marks inputs
where the unsupervised objective does not pin down a unique answer.

Read it alongside CPA MAE — variance measures *disagreement*, not correctness.
A model can be confidently wrong.

### 5.1 Best configuration per family

The ensemble uses the best configuration per family from Study 1 (lowest CPA MAE
at 100 % data, FID as tiebreak). As selected in the paper:

| Model | Generator size | Data | Architecture flags |
|---|---|---|---|
| CycleGAN | M | 100 % | `--cyclegan_ngf 128 --cyclegan_n_blocks 10` |
| UNIT | M | 100 % | `--unit_ngf 128 --unit_n_blocks 10 --unit_n_blocks_shared 2` |
| MUNIT | M | 100 % | `--munit_ngf 128 --munit_n_content_blocks 5` |
| DCLGAN | S | 100 % | `--dclgan_ngf 64 --dclgan_n_blocks 8` |
| UVCGAN | S | 100 % | `--uvcgan_ngf 48 --uvcgan_vit_features 96 --uvcgan_vit_blocks 6` |
| CycleDiffusion | S | 100 % | `--cd_base_channels 48 --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 1` |

### 5.2 Train the ensemble and run inference

**K = 10 members per family**, seeds 1…10, 750 000 steps each on the full
training set. Members differ *only* by `--seed`.

```bash
ENS=./ensemble

for SEED in $(seq 1 10); do
  M=$(printf "%02d" $SEED)
  python train.py --model cyclegan \
      --dataA $TRAIN_A --dataB $TRAIN_B --data_range 1,30 \
      --cyclegan_ngf 128 --cyclegan_n_blocks 10 \
      --steps 750000 --seed $SEED \
      --output $ENS/cyclegan/models/model_${M}/
done
```

UVCGAN members keep the two-stage recipe at 250 000 + 500 000 steps:

```bash
for SEED in $(seq 1 10); do
  M=$(printf "%02d" $SEED)
  D=$ENS/uvcgan/models/model_${M}
  python train.py --model uvcgan --uvcgan_stage pretrain \
      --dataA $TRAIN_A --dataB $TRAIN_B --data_range 1,30 \
      --uvcgan_ngf 48 --uvcgan_vit_features 96 --uvcgan_vit_blocks 6 \
      --steps 250000 --seed $SEED --output $D/stage1/
  python train.py --model uvcgan --uvcgan_stage finetune \
      --uvcgan_init_ckpt $D/stage1/checkpoints/step_250000.pt \
      --dataA $TRAIN_A --dataB $TRAIN_B --data_range 1,30 \
      --uvcgan_ngf 48 --uvcgan_vit_features 96 --uvcgan_vit_blocks 6 \
      --steps 500000 --seed $SEED --amp --output $D/stage2/
done
```

The `model_01/ … model_10/` naming matters: `uncertainty.py` discovers members
by globbing `model_*`, so K is simply however many member directories exist.

Then run **both directions** per member — A→B for the uncertainty maps, B→A for
the cycle-error maps:

```bash
for SEED in $(seq 1 10); do
  M=$(printf "%02d" $SEED)
  CKPT=$ENS/cyclegan/models/model_${M}/checkpoints/step_750000.pt

  python inference.py --model cyclegan --direction A2B \
      --data $TEST_A --data_range 1,5 \
      --ckpt $CKPT --outdir $ENS/cyclegan/inference/model_${M}/

  python inference.py --model cyclegan --direction B2A \
      --data $ENS/cyclegan/inference/model_${M}/ --data_range 1,5 \
      --ckpt $CKPT --outdir $ENS/cyclegan/inference_B2A/model_${M}/
done
```

### 5.3 Per-pixel uncertainty maps

```bash
python uncertainty.py \
    --model  cyclegan \
    --data   $ENS/cyclegan/inference/ \
    --output $ENS/cyclegan/uncertainty/ \
    --mask_dir $TEST_A --min_tissue_fraction 0.1 \
    --lower-percentile 1 --upper-percentile 99
```

The per-pixel value is **√(Σ per-channel sample variance)** (ddof = 1) across
members, in 0–255 intensity units — already a standard deviation, not a variance.

Outputs under `{output}/{model}/`:

| Output | Contents |
|---|---|
| `raw_npy/` | `[H, W]` float32 maps — **the input to every downstream step** |
| `heatmaps/` | magma PNGs with colourbar (qualitative only) |
| `mean_rgb/` | ensemble-mean prediction — this is the virtual stain to report |
| `summary.json` | per-image statistics and the global normalisation bounds |

The percentile flags affect `heatmaps/` only; `raw_npy/` is never rescaled. Each
output mirrors the input tile structure, e.g. `raw_npy/001/images/0000001.npy`.
Add `--data_range $i,$i` to process one WSI at a time (the summary is then named
`summary_wsi{NNN}.json`).

Run the same command over `inference_B2A/` to get the **ensemble-mean A′**, which
[§5.5](#55-cycle-error-maps-for-the-ensemble) needs:

```bash
python uncertainty.py --model cyclegan \
    --data   $ENS/cyclegan/inference_B2A/ \
    --output $ENS/cyclegan/regen_stats/ \
    --mask_dir $TEST_A --min_tissue_fraction 0.1
```

### 5.4 Per-tile uncertainty and the family comparison

Reduce each map to a scalar σ̄ (tissue-masked spatial mean), one CSV per WSI:

```bash
python aggregate_uncertainty.py \
    --uncertainty_dir $ENS/cyclegan/uncertainty/cyclegan/raw_npy/ \
    --tiles_metadata  $TEST_A \
    --mask_dir        $TEST_A \
    --min_tissue_fraction 0.1 \
    --outdir          $ENS/cyclegan/uncertainty/cyclegan/per_wsi_csv/
```

WSI membership is derived from the `NNN/` component of the `.npy` path, so tile
IDs repeating across WSIs do not collide. Each CSV has columns
`tile_name, mean_uncertainty`.

Then the cross-family figure. `--base` is a directory holding one subdirectory
per model; every `per_wsi_csv/*.csv` beneath each is loaded, whatever the
intermediate layout:

```bash
python plot_uncertainty_boxplot.py --base $ENS --outdir ./figures/
```

### 5.5 Cycle error maps for the ensemble

Per-pixel |A − A′| against the ensemble-mean A′ from [§5.3](#53-per-pixel-uncertainty-maps),
one directory per WSI so the error maps line up with the uncertainty maps:

```bash
for W in 001 002 003 004 005; do
  python evaluation.py --metric regen_error \
      --path_A       $TEST_A/$W/images \
      --path_A_regen $ENS/cyclegan/regen_stats/cyclegan/mean_rgb/$W/images \
      --mask_dir     $TEST_A/$W/masks \
      --min_tissue_fraction 0.1 \
      --overlay_dir  $ENS/cyclegan/regen_error/wsi$W/ \
      --save_error_npy
done
```

Both paths are leaf image directories, so tiles are matched by basename. This
step is CPU-only.

To calibrate against *per-member* error instead of ensemble-mean error, run
`evaluation.py --metric regen_error` once per member and pass all the resulting
`error_npy/` directories to `--error_dirs` below; they are averaged per-pixel.

### 5.6 Calibration against cycle error

Does uncertainty actually predict error? Pair the two map sets and reduce to
scalar scores.

```bash
for W in 001 002 003 004 005; do
  python uncertainty_calibration.py \
      --uncertainty_dir $ENS/cyclegan/uncertainty/cyclegan/raw_npy/$W/images/ \
      --error_dirs      $ENS/cyclegan/regen_error/wsi$W/error_npy/ \
      --mask_dir        $TEST_A/$W/masks/ \
      --tiles_metadata  $TEST_A \
      --min_tissue_pixels 256 \
      --title           cyclegan \
      --outdir          $ENS/cyclegan/calibration/wsi$W/
done
```

Always use `raw_npy/`, never `heatmaps/`. `--mask_dir` is required unless you
pass `--no_mask`: background inflates both ρ and ECE spuriously.

Per-WSI outputs: `per_tile.csv` (tile_stem, source_wsi, n_tissue_pixels,
spearman_rho, pearson_rho_within, mean_u, mean_e), `per_wsi.csv`,
`summary.json`, `calibration.png`.

Then pool every tile per model and recompute — this gives the correct per-model
statistics rather than an average of per-WSI summaries. `--base` holds one
subdirectory per model; every `per_tile.csv` beneath each is pooled:

```bash
python aggregate_calibration.py --base $ENS --outdir ./calibration_combined/
```

Writes `{outdir}/{model}/summary.json`, `{outdir}/{model}/calibration.png`,
`{outdir}/all_models.csv` and the combined 2 × 3 panels.

| Reported metric | Question it answers |
|---|---|
| Within-tile Spearman ρ | Are the uncertain *pixels* the wrong pixels? |
| Across-tile Pearson / Spearman | Are the uncertain *tiles* the wrong tiles? |
| Reliability diagram + ECE | Does binned error track binned uncertainty? |

ρ → +1 means uncertainty co-locates with error; ≈ 0 means uninformative; < 0
means anti-calibrated. The across-tile statistic catches the case where
uncertainty is locally calibrated but flat at tile level, which is useless for
triage. ECE → 0 means the bins lie on y = x after p1–p99 normalisation.

> **Cycle error is a proxy, not ground truth.** When the forward and inverse
> generators share a bias, both may ignore the same feature and the round trip
> still reconstructs the source despite a poor forward translation.

### 5.7 Ensemble CPA

The ensemble mean is the virtual stain, so its CPA goes through exactly the same
pipeline as [§4.5](#45-task-specific-metric-collagen-proportionate-area) — stitch
`mean_rgb/` (or each member's tiles, to get the CPA spread across members),
segment with `Dataset314_SR_light`, apply the H&E mask, fill holes, and compare
against the real SR reference:

```bash
for SEED in $(seq 1 10); do
  M=$(printf "%02d" $SEED)
  python reconstruct.py --metadata $TEST_A \
      --tile_dir $ENS/cyclegan/inference/model_${M}/ \
      --output   $ENS/cyclegan/reconstructed/model_${M}/ \
      --mode rgb --blend average
done
# … nnUNetv2_predict → apply_he_mask.py → fill_tissue_holes.py as in §4.5 …

MASKS=(); LABELS=()
for SEED in $(seq 1 10); do
  M=$(printf "%02d" $SEED)
  MASKS+=("$ENS/cyclegan/wsi_masks_final/model_${M}/"); LABELS+=("member_${M}")
done
python compare_psr.py --masks_real ./psr/real/psr_masks_wsi_final/ \
    --masks_generated "${MASKS[@]}" --labels "${LABELS[@]}" \
    --outdir ./psr_comparison_ensemble/cyclegan/ --strip_prefix
```

---

## 6. CLI reference

Every entry point also responds to `--help`.

### `tile.py` — WSI → tiles

| Flag | Default | Meaning |
|---|---|---|
| `--rgb` | *required* | Root folder containing an `{image_type}/` subfolder of RGB `.tif` WSIs |
| `--output` | *required* | Root output folder; tiles land in `{output}/{image_type}/{NNN}/` |
| `--mask` | `None` | Root folder containing an `{image_type}/` subfolder of tissue-mask `.tif`s. Required for `--tissue_threshold > 0` and for writing `masks/` |
| `--image_type` | `trainA` | Split name — `trainA`, `trainB`, `testA`, `testB` |
| `--tile_size` | `256` | Side length of square tiles extracted from the WSI (study: `512`) |
| `--resize_to` | `None` | Downsample each tile to this size before saving (study: `256`) |
| `--tissue_threshold` | `0.5` | Minimum tissue fraction to keep a tile. Study: `0.5` for train splits, `0` for test splits |
| `--num_workers` | all CPUs | Worker processes |

Tiling is non-overlapping. Re-running on an existing output directory resumes at
the next free WSI index. Each WSI folder gets a `tiles_metadata.csv`; its
`stride`/`overlap` columns are retained for compatibility with already-released
tilings and are not read by `reconstruct.py`.

### `train.py` — training

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *required* | `cyclegan` \| `unit` \| `munit` \| `dclgan` \| `uvcgan` \| `cyclediffusion` |
| `--dataA` / `--dataB` | *required* | Domain A (H&E) and domain B (SR) tile roots |
| `--data_range` | `None` (all) | `START,END` — load `{START:03d}/images/` … `{END:03d}/images/`. Sets the data fraction |
| `--steps` | `5000000` | Total optimiser steps. Study: `1000000` (UVCGAN 250 000 + 750 000) |
| `--output` | *required* | Run directory; checkpoints go to `{output}/checkpoints/step_{N}.pt` |
| `--save_steps` | `250000` | Checkpoint frequency |
| `--log_steps` | `1000` | Logging frequency. Log line: `[S00001000 |   12.3s] loss_G:0.4017 …` |
| `--amp` | off | Mixed precision. Silently disabled for `cyclediffusion` (fp32 UNet; `GradScaler` overflows ≈ 56 k steps) |
| `--seed` | `None` | Global RNG seed (torch, numpy, random). Ensemble members differ only by this |
| `--init_ckpt` | `None` | Initialise weights from a checkpoint (any model) |
| `--count_params` | off | Print the A→B generator parameter count and exit — no data or output needed |
| `--batch_size` | `1` | Study setting is 1 |
| `--lr` | `2e-4` | Adam learning rate, β = (0.5, 0.999), linear decay from the halfway point |
| `--num_workers` | `4` | Dataloader workers |

Per-architecture flags:

| Flag | Default | Model | Meaning |
|---|---|---|---|
| `--cyclegan_ngf` | `64` | CycleGAN | Generator base channels |
| `--cyclegan_n_blocks` | `9` | CycleGAN | ResNet blocks in the bottleneck |
| `--unit_ngf` | `64` | UNIT | Generator base channels |
| `--unit_n_blocks` | `9` | UNIT | Total bottleneck blocks (pre + shared + post) |
| `--unit_n_blocks_shared` | `3` | UNIT | Shared blocks; pre = post = (total − shared) / 2 |
| `--munit_ngf` | `64` | MUNIT | Generator base channels |
| `--munit_n_content_blocks` | `4` | MUNIT | Content-encoder ResNet blocks |
| `--munit_n_adain_blocks` | `4` | MUNIT | AdaIN ResNet blocks in the decoder |
| `--style_dim` | `8` | MUNIT | Style-code dimensionality |
| `--dclgan_ngf` | `64` | DCLGAN | Generator base channels |
| `--dclgan_n_blocks` | `9` | DCLGAN | ResNet blocks in the bottleneck |
| `--lambda_dcl` | `1.0` | DCLGAN | Weight of the dual contrastive (InfoNCE) loss |
| `--dclgan_lambda_cycle` | `10.0` | DCLGAN | Cycle-consistency weight |
| `--dclgan_lambda_identity` | `0.0` | DCLGAN | Identity loss weight (off by default) |
| `--n_patches` | `256` | DCLGAN | Patches sampled per image for contrastive matching |
| `--proj_dim` | `256` | DCLGAN | Projection-head dimensionality |
| `--uvcgan_stage` | `finetune` | UVCGAN | `pretrain` (masked-image modelling) or `finetune` (cycle-consistent) |
| `--uvcgan_init_ckpt` | `None` | UVCGAN | Stage-1 checkpoint to start the finetune stage from |
| `--uvcgan_ngf` | `64` | UVCGAN | UNet encoder/decoder base channels |
| `--uvcgan_vit_blocks` | `6` | UVCGAN | Transformer blocks in the bottleneck |
| `--uvcgan_vit_features` | `192` | UVCGAN | ViT hidden dimension |
| `--cd_base_channels` | `64` | CycleDiffusion | DDPM UNet base channel count |
| `--cd_channel_mult` | `1,2,2,4` | CycleDiffusion | Per-resolution channel multipliers |
| `--cd_num_res_blocks` | `2` | CycleDiffusion | Residual blocks per resolution |
| `--cd_steps` | `200` | CycleDiffusion | Stored in the config for inference; unused during training |

**Auto-resume:** at startup `BaseTrainer` scans `{output}/checkpoints/` for
`step_*.pt` and resumes from the newest. Re-running an identical command is
always safe, and exits immediately once `--steps` is reached. Elapsed time
accumulates across resumes into `{output}/training_meta.json`.

**Checkpoint contents:** `{"model": state_dict, "config": asdict(cfg),
"model_name": str, …}`. The config auto-restores on `--init_ckpt` (training) and
`--ckpt` (inference), so architecture flags need not be repeated; checkpoints
without a `"config"` key fall back to the CLI args and defaults.

### `inference.py` — translation

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *required* | Architecture; must match the checkpoint |
| `--direction` | *required* | `A2B` or `B2A`. For CycleDiffusion, `A2B` inverts with `eps_A` and decodes with `eps_B`; `B2A` is symmetric |
| `--data` | *required* | Tile root to translate |
| `--data_range` | `None` (all) | `START,END` over numbered WSI folders |
| `--ckpt` | *required* | Checkpoint path |
| `--outdir` | `results` | Output root; the input's relative tile paths are mirrored |
| `--resume` | off | Skip tiles whose output exists, re-doing the most recent one (guards against a half-written tile) |
| `--seed` | `None` | Fix the RNG for deterministic sampling |
| `--cd_steps` | `200` | CycleDiffusion DDIM inversion + decode steps, each way. More is more faithful at proportional cost; 200 is the study setting |
| `--num_samples` | `1` | MUNIT — number of random style codes per tile. `> 1` appends a suffix per sample |
| `--style_image` | `None` | MUNIT — take the style code from this reference image instead of sampling |
| `--style_dim` | `8` | MUNIT — used only if the checkpoint has no stored config |

### `evaluation.py` — metrics

Available metrics are `fid`, `patch_ssim`, `lpips` and `regen_error` only.
Full-image SSIM, the DINOv2 FID backend and the external-judge error proxy were
removed: the paper uses patch-SSIM, InceptionV3 and each model's own inverse
generator.

| Flag | Default | Meaning |
|---|---|---|
| `--metric` | `fid` | `fid` (unpaired, distribution-level, InceptionV3 pool3 2048-d) \| `patch_ssim` (paired) \| `lpips` (paired, VGG16, lower is better) \| `regen_error` (cycle MAE in [0, 255]) |
| `--path_real` | `None` | Real target-domain images — required for `fid` / `patch_ssim` / `lpips` |
| `--path_fake` | `None` | Generated images — required for the same three. Paired metrics match by filename |
| `--patch_size` | `64` | `patch_ssim` patch side |
| `--patches_per_image` | `16` | `patch_ssim` random patches per image |
| `--ssim_image_size` | `256` | Resize before computing SSIM |
| `--path_A` | `None` | Source-domain A images — required for `regen_error` |
| `--path_A_regen` | `None` | Precomputed A′ tiles (B→A inference output). Skips internal inference, works for every family including CycleDiffusion, and makes `--model` / `--ckpt` unnecessary |
| `--model` / `--ckpt` | `None` | Required for `regen_error` *without* `--path_A_regen` |
| `--direction` | `A2B` | Cycle direction for `regen_error` |
| `--overlay_dir` | `None` | Where to write error heatmaps and overlays (`regen_error` only) |
| `--save_error_npy` | off | Also write raw `[H, W]` float32 maps to `{overlay_dir}/error_npy/`. Requires `--overlay_dir` |
| `--mask_dir` | `None` | Tissue-mask root, walked recursively and matched by stem. If omitted, masks are auto-detected by replacing the `images` path component with `masks` |
| `--min_tissue_fraction` | `0.0` | Minimum fraction of non-zero mask pixels for a tile to count. Study setting: `0.1`. Tiles with no matching mask are always included |
| `--save_csv` | `None` | Write results (summary plus per-image scores where available) to CSV |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--batch_size` | `32` | Feature-extraction batch size |
| `--num_workers` | `4` | Dataloader workers |
| `--style_dim` | `8` | MUNIT, `regen_error` only; ignored if the checkpoint stores a config |

### `reconstruct.py` — tiles → WSI

| Flag | Default | Meaning |
|---|---|---|
| `--metadata` | *required* | A dataset root holding per-WSI `tiles_metadata.csv` files (all are found automatically), or a single `tiles_metadata.csv` for one WSI |
| `--output` | *required* | Output directory. Reconstructions keep the original WSI filename; masks are `{stem}_mask.tif` |
| `--tile_dir` | `None` | Directory of tiles to stitch, e.g. inference output. If omitted, the `image_path` column of the metadata CSV is used |
| `--mode` | `rgb` | `rgb` \| `mask` \| `rgb_and_mask` \| `auto` |
| `--blend` | `average` | How to combine overlapping tiles: `average` or `overwrite` |

`--mode mask` reads the `mask_path` column of the metadata CSV, which stores the
absolute paths recorded at tiling time — it works as long as the tile tree has
not been moved since `tile.py` ran. `--mode rgb` with `--tile_dir` has no such
dependency.

### `uncertainty.py` — ensemble variance maps

| Flag | Default | Meaning |
|---|---|---|
| `--model` | *required* | Architecture name; used to label the output subdirectory |
| `--data` | *required* | Ensemble output root containing `model_01/`, `model_02/`, … (discovered by globbing `model_*`, so K is whatever exists) |
| `--output` | `uncertainty_output` | Output root; results land in `{output}/{model}/` |
| `--data_range` | `None` (all) | `START,END` — process only those WSI folders under each member directory. The summary is then named `summary_wsi{NNN}.json` |
| `--mask_dir` | `None` | Tissue-mask root, walked recursively and matched by stem. Falls back to the `images` → `masks` path swap |
| `--min_tissue_fraction` | `0.0` | Skip tiles below this tissue fraction. Study setting: `0.1` |
| `--lower-percentile` | `1.0` | Lower bound for the global heatmap normalisation |
| `--upper-percentile` | `99.0` | Upper bound. **Both percentile flags affect `heatmaps/` only — `raw_npy/` is never rescaled** |

### `aggregate_uncertainty.py` — per-tile σ̄

| Flag | Default | Meaning |
|---|---|---|
| `--uncertainty_dir` | *required* | Directory of `<stem>.npy` maps — use `raw_npy/` |
| `--tiles_metadata` | *required* | Dataset root with per-WSI `tiles_metadata.csv`; WSI membership comes from the `NNN/` path component so repeated tile IDs do not collide |
| `--mask_dir` | `None` | Tissue masks (`<stem>.tif` or `NNN/masks/<stem>.tif`). Without it, all pixels are used |
| `--min_tissue_fraction` | `0.0` | Minimum tissue fraction to include a tile |
| `--outdir` | *required* | One `{wsi_stem}.csv` per WSI, columns `tile_name, mean_uncertainty` |

### `uncertainty_calibration.py` — uncertainty vs. cycle error

| Flag | Default | Meaning |
|---|---|---|
| `--uncertainty_dir` | *required* | Flat directory of `<stem>.npy` from `uncertainty.py`. Use `raw_npy/`, never `heatmaps/` |
| `--error_dirs` | *required* | One or more flat directories of `<stem>.npy` from `evaluation.py --save_error_npy`. Multiple directories are averaged per-pixel first |
| `--mask_dir` | `None` | Flat directory of tissue masks `<stem>.tif` (any non-zero = tissue). Required unless `--no_mask` |
| `--no_mask` | off | Use all pixels. Background inflates ρ and ECE — for diagnostics only |
| `--tiles_metadata` | `None` | Dataset root with per-WSI `tiles_metadata.csv`; enables the `per_wsi.csv` rollup via the `source_file` column |
| `--outdir` | `calibration_out` | Output directory |
| `--n_bins` | `10` | Quantile bins for the reliability diagram and ECE |
| `--min_tissue_pixels` | `256` | Skip tiles with fewer tissue pixels than this |
| `--title` | `""` | Title prefix for the figure panels, e.g. the model name |

Outputs: `per_tile.csv`, `per_wsi.csv` (with `--tiles_metadata`), `summary.json`,
and `calibration.png` (reliability diagram, within-tile ρ histogram, across-tile
mean(U) vs mean(E) scatter).

### `aggregate_calibration.py` — pool tiles per model

| Flag | Default | Meaning |
|---|---|---|
| `--base` | *required* | Directory with one subdirectory per model. **Every `per_tile.csv` anywhere beneath a model's subdirectory is pooled**, so any per-WSI layout works |
| `--models` | the six families | Which model subdirectories to aggregate |
| `--outdir` | `calibration_combined` | Writes `{model}/summary.json`, `{model}/calibration.png`, `all_models.csv` and the combined 2 × 3 panels |
| `--n_bins` | `10` | Quantile bins for the reliability diagram and ECE |

### `plot_uncertainty_boxplot.py` — per-family σ̄ distribution

| Flag | Default | Meaning |
|---|---|---|
| `--base` | *required* | Directory with one subdirectory per model. **Every `per_wsi_csv/*.csv` beneath a model's subdirectory is loaded** |
| `--models` | the six families | Which model subdirectories to plot |
| `--outdir` | `uncertainty_boxplot` | Pooled box and violin figures, per-WSI figures under `per_wsi/`, and `uncertainty_quantiles.csv` |

### `apply_he_mask.py` — CPA post-processing, step 1

| Flag | Default | Meaning |
|---|---|---|
| `--psr_masks` | *required* | Directory of PSR mask TIFs, or a single TIF |
| `--he_masks` | *required* | H&E tissue masks (non-zero = tissue), matched by stem. A mask of different spatial size is resized nearest-neighbour |
| `--outdir` | *required* | Output directory |

Zeros every PSR prediction outside the H&E tissue boundary; labels inside are
preserved. Multi-channel TIFs use the `[..., 0]` slice. PSR files with no
matching H&E mask are warned about and skipped — so check the warning count,
since a stem mismatch silently leaves every mask unprocessed. For a PSR mask
`slide.tif`, both `slide.tif` and `slide_mask.tif` are accepted in `--he_masks`,
the latter being what `reconstruct.py --mode mask` writes.

### `fill_tissue_holes.py` — CPA post-processing, step 2

| Flag | Default | Meaning |
|---|---|---|
| `--masks` | *required* | Directory of PSR mask TIFs, or a single TIF |
| `--outdir` | *required* | Output directory |

Fills enclosed background inside the tissue footprint, treating the **union** of
labels 1 and 2 as foreground — filling only label 1 would treat every
PSR-positive pixel as a hole and relabel it. Filled pixels become label 1.

### `compare_psr.py` — CPA agreement

| Flag | Default | Meaning |
|---|---|---|
| `--masks_real` | *required* | Directory of real-SR WSI mask TIFs |
| `--masks_generated` | *required* | One or more directories of generated-SR mask TIFs; pass several to compare configurations in one run |
| `--labels` | directory names | One label per `--masks_generated` entry. For `plot_combined_metrics.py`, use `{model_size}_model/{data_size}_data` |
| `--outdir` | `psr_comparison` | Output directory |
| `--label_tissue` | `1` | nnU-Net label index for the tissue class |
| `--label_psr` | `2` | nnU-Net label index for the PSR-positive class |
| `--strip_prefix` | off | Drop the first `_`-delimited filename token before stem matching, e.g. `SR_slide.tif` ↔ `HE_slide.tif` |

Outputs `per_wsi.csv` (wsi, condition, psr_fraction), `summary.json`,
`comparison.png`, `paired_scatter.png` and `paired_metrics.png`. The paired
metrics vs. real SR are the primary comparison: `n_matched`, `pearson_r` /
`pearson_pvalue`, `spearman_rho` / `spearman_pvalue`, `mae_paired` (the headline
CPA MAE) and `mean_paired_diff_generated_minus_real` (signed bias; positive means
over-estimation). Unpaired distributional metrics were removed — the paper
reports paired CPA MAE.

### `plot_combined_metrics.py` — 2 × 2 metric overview

| Flag | Default | Meaning |
|---|---|---|
| `--eval_indir` | *required* | Root of the per-configuration metric CSVs |
| `--psr_indir` | *required* | Root of the per-model `compare_psr.py` outputs |
| `--outdir` | `combined_metrics_plot` | Writes `combined_metrics.png` and `combined_metrics.csv` |

Expected layouts:

```
{eval_indir}/{model}/data_{data_size}/model_{model_size}/{patch_ssim,lpips,fid}.csv
{eval_indir}/{model}/results/data_{data_size}/model_{model_size}/…     # also accepted
{psr_indir}/{model}/summary.json
{psr_indir}/{model}/per_wsi.csv
```

with `{model_size}` and `{data_size}` each one of `small` | `medium` | `large`,
and `compare_psr.py` labels named `{model_size}_model/{data_size}_data`.

### `plot_ranking_correlation.py` — cross-metric rank agreement

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | *required* | `combined_metrics.csv` from `plot_combined_metrics.py` |
| `--outdir` | `combined_metrics_plot` | Writes `ranking_correlation.png` (4 × 4, RdYlGn), `ranking_correlation.csv` and `ranking_correlation_pvalues.csv` |

---

## 7. Repository layout

```
train.py                     unified training entry point
inference.py                 unified inference entry point
evaluation.py                FID, patch-SSIM, LPIPS, cycle-regeneration error
tile.py  reconstruct.py      WSI ↔ tile conversion
utils.py  base_models.py     shared blocks (ResNet, PatchGAN, DDPM UNet)

models/                      the six architectures
datasets/                    unpaired / single-domain loaders, transforms
trainer/                     step-based training loop with auto-resume
tests/                       pytest suite (100 tests, CPU-only)

uncertainty.py               ensemble variance maps
aggregate_uncertainty.py     per-tile σ̄ → per-WSI CSVs
uncertainty_calibration.py   uncertainty vs. cycle error
aggregate_calibration.py     pool per-WSI results per model
plot_uncertainty_boxplot.py  per-family σ̄ distribution

compare_psr.py               CPA agreement vs. real SR
apply_he_mask.py             CPA mask cleanup
fill_tissue_holes.py         CPA hole filling

plot_combined_metrics.py     2 × 2 metric overview
plot_ranking_correlation.py  cross-metric rank agreement

MODEL_DIAGRAMS.md            architecture diagrams
CLAUDE.md                    conventions and architecture notes
```

### Architecture in brief

| Model | Key mechanism |
|---|---|
| CycleGAN | Cycle-consistency loss over paired Enc → Bottleneck → Dec generators |
| UNIT | Shared bottleneck plus a KL term on a variational latent |
| MUNIT | Content/style decomposition with AdaIN; style sampling at inference |
| DCLGAN | Dual patch-level contrastive (InfoNCE) feature matching |
| UVCGAN | UNet–ViT hybrid with cycle-consistency and a masked-image pretrain stage |
| CycleDiffusion | Two unconditional DDPMs; DDIM inversion of the source into a shared noise code, then decode |

All models implement one interface consumed by `BaseTrainer`:
`generator_parameters()`, `discriminator_parameters()`,
`compute_generator_loss(batch) → (loss, log_dict, visuals)` and
`compute_discriminator_loss(batch, visuals) → (loss, log_dict)`. Adding a family
means implementing those four methods — nothing else in the pipeline changes.

---

## 8. Notes and caveats

- **Images are normalised to [−1, 1]** for training and denormalised to [0, 1]
  for saving. Supported formats: `.png .jpg .jpeg .tif .tiff .bmp .webp`.
- **No external diffusion libraries.** DDPM/DDIM sampling is implemented from
  scratch in `base_models.py`.
- **Registration.** SSIM and LPIPS assume co-registered pairs; the test WSIs were
  aligned with VALIS beforehand. Residual misregistration inflates both metrics,
  which is one reason the paper does not select models on them alone.
- **Cycle error is a proxy.** It measures round-trip self-consistency, not
  correctness against a ground-truth SR image, which does not exist in the
  unpaired setting.
- **Uncertainty ≠ error.** Ensemble variance measures disagreement among equally
  plausible solutions. Section 5.6 exists precisely to test how far that
  disagreement tracks error, and the answer differs by family.
- **Determinism.** `--seed` fixes torch, numpy and Python RNGs. Exact
  bit-for-bit reproduction across different GPUs, CUDA versions or with `--amp`
  is not expected; the reported metrics are stable well within the differences
  the study reports.
- The scope of this branch is the published paper: every entry point maps to a
  reported result. Models, metrics and flags that no section uses were removed
  deliberately and should not be reintroduced here.
