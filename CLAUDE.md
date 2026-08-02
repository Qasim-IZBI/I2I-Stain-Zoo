# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

For the user-facing walkthrough (install → data → training → inference →
evaluation → uncertainty), see `README.md`. This file is the complete flag
reference plus architecture and conventions.

## Project Overview

I2I-Stain-Zoo is an image-to-image translation research codebase for virtual
staining of histopathology images (H&E → Sirius Red). It implements **six**
unpaired architectures behind one training/inference interface: CycleGAN, UNIT,
MUNIT, DCLGAN, UVCGAN, and CycleDiffusion (the only diffusion-based family).

The repository backs a paper with two experiments:

- **Scaling study** — 54 configurations (6 models × 3 generator sizes × 3 training
  data fractions), each evaluated on Patch-SSIM, LPIPS, FID and CPA MAE.
- **Uncertainty study** — the best configuration per family retrained as a deep
  ensemble, reduced to per-pixel variance and calibrated against cycle error.

Scope rule: the repository was pruned so that every entry point maps to a
reported result. Do not reintroduce models, metrics or flags that no section of
the paper uses.

## Commands

### Tiling

Tiles are written per-WSI into numbered subfolders (`001/`, `002/`, …) under the
output directory. Each holds `images/` (RGB) and `masks/` (tissue masks, if
`--mask` given). Filenames are `{tile_id:07d}.tif`. Re-running on the same output
directory resumes from the next free index.

Tiling is **non-overlapping** (stride = tile size); this is the study protocol and
is no longer configurable.

```bash
# Study protocol: 512×512 tiles downsampled to 256×256, tissue-filtered
python tile.py --rgb path/to/wsi --output path/to/tiles --mask path/to/masks \
    --tile_size 512 --resize_to 256 --image_type trainA --tissue_threshold 0.5

# Test set — keep all tiles (no tissue filtering)
python tile.py --rgb path/to/wsi --output path/to/tiles --image_type testA \
    --tile_size 512 --resize_to 256 --tissue_threshold 0
```

Output structure:

```
path/to/tiles/
  trainA/
    001/
      images/            ← 0000001.tif, 0000002.tif, …
      masks/             ← only if --mask provided
      tiles_metadata.csv ← stride, overlap, x/y positions for this WSI
    002/
```

`tiles_metadata.csv` retains `stride` and `overlap` columns for compatibility with
already-released tilings; `reconstruct.py` does not read them.

### Training

Step-based, not epoch-based, so runs are comparable across data fractions with
different tile counts. Logs print every `--log_steps` with wall time for the
interval; checkpoints save every `--save_steps`.

```bash
# Single-stage models (cyclegan, unit, munit, dclgan, cyclediffusion)
python train.py --model cyclegan --dataA path/to/tiles/trainA --dataB path/to/tiles/trainB \
    --steps 750000 --amp --output ./results/

# Train on a subset of WSIs (folders 001–007 = 25% data fraction)
python train.py --model cyclegan --dataA ... --dataB ... \
    --data_range 1,7 --steps 750000 --amp --output ./results/

# Custom log and checkpoint frequency
python train.py --model cyclegan --dataA ... --dataB ... --steps 750000 --amp \
    --log_steps 500 --save_steps 100000 --output ./results/

# Initialise from a pretrained checkpoint (any model)
python train.py --model cyclegan --dataA ... --dataB ... --steps 750000 \
    --init_ckpt ./prev_run/checkpoints/step_250000.pt --output ./new_run/

# Deterministic run (ensemble members differ only by seed)
python train.py --model cyclegan --dataA ... --dataB ... --steps 750000 --amp \
    --seed 1 --output ./ensemble/model_01/

# UVCGAN (2-stage): masked-image pretrain → cycle-consistent finetune
python train.py --model uvcgan --uvcgan_stage pretrain --dataA ... --dataB ... \
    --steps 250000 --amp --output ./uvcgan/stage1/
python train.py --model uvcgan --uvcgan_stage finetune \
    --uvcgan_init_ckpt ./uvcgan/stage1/checkpoints/step_250000.pt \
    --dataA ... --dataB ... --steps 500000 --amp --output ./uvcgan/stage2/

# Report A→B parameter count without training
python train.py --model cyclegan --cyclegan_ngf 128 --cyclegan_n_blocks 10 --count_params
```

**AMP:** `--amp` is silently disabled for `cyclediffusion` — its UNet runs fp32
internally and `GradScaler` overflows around 56k steps. A notice is printed.

#### UGAC — aleatoric uncertainty (CycleGAN, experimental)

`--cyclegan_ugac` enables the UGAC heads of Upadhyay et al., *Robustness via
Uncertainty-aware Cycle Consistency* (NeurIPS 2021). The decoders become
`Decoder3Head`, predicting the parameters of a zero-mean generalized Gaussian
over the per-pixel cycle residual: mu (the image), 1/alpha (scale) and beta
(shape). The L1 cycle loss is replaced by the GGD negative log-likelihood
(`ggd_nll`, paper Eq. 8), weighted by `lambda_cycle` as before.

```bash
# UGAC
python train.py --model cyclegan --cyclegan_ugac \
    --dataA ... --dataB ... --steps 750000 --amp --output ./runs/cyclegan_ugac/

# vanilla (default; omit the flag)
python train.py --model cyclegan --dataA ... --dataB ... --steps 750000 --amp \
    --output ./runs/cyclegan/

# per-pixel aleatoric SD alongside the translated tiles
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --ckpt ./runs/cyclegan_ugac/checkpoints/step_750000.pt \
    --outdir ./out/ --save_aleatoric
```

UGAC ensembles at the small generator size, K = 10 members per data block,
over five **disjoint** 7-specimen blocks (50 jobs):

```bash
sbatch scripts/train_ensemble_cyclegan_ugac.sh          # --array=0-49
sbatch --array=0-49%10 scripts/train_ensemble_cyclegan_ugac.sh   # cap concurrency
sbatch --array=10-19 scripts/train_ensemble_cyclegan_ugac.sh     # one block only
```

| tasks | folders | output |
|---|---|---|
| 0–9   | 001–007 | `ensemble_ugac/cyclegan/data_001_007/model_small/models/model_{01..10}/` |
| 10–19 | 008–014 | `…/data_008_014/…` |
| 20–29 | 015–021 | `…/data_015_021/…` |
| 30–39 | 022–028 | `…/data_022_028/…` |
| 40–49 | 029–035 | `…/data_029_035/…` |

Blocks are disjoint rather than nested, so differences across them reflect
*which* slides were seen, not how many — the opposite of the nested 25/50/100%
fractions in the scaling study. Epistemic variance is computed within a block.

The last block needs folders 031–035, which are outside the 001–030 training
set; a pre-flight check fails the job immediately with the missing paths rather
than letting it die inside the dataloader hours later.

Inference over the trained members — same 50-job decomposition, so array indices
line up one-to-one with the training jobs:

```bash
sbatch scripts/infer_ensemble_cyclegan_ugac.sh       # A→B + aleatoric maps
sbatch scripts/infer_ensemble_cyclegan_ugac_B2A.sh   # B→A, for regen error
```

| step | reads | writes |
|---|---|---|
| train | trainA/trainB block | `{block}/model_small/models/model_{NN}/` |
| A2B | that checkpoint | `{block}/model_small/inference/model_{NN}/` + `aleatoric_npy/` |
| B2A | the A2B tiles | `{block}/model_small/inference_B2A/model_{NN}/` |

`--save_aleatoric` needs no architecture flag: `ugac` is restored from the
checkpoint, and inference.py refuses a non-UGAC checkpoint rather than emitting
garbage. Epistemic uncertainty is a separate step — run `uncertainty.py` across
the ten `model_{01..10}` directories *within one block*.
Kept separate from `ensemble/` because the UGAC objective differs from the
vanilla runs stored there.

`--save_aleatoric` writes `{outdir}/aleatoric_npy/<stem>.npy` as `[H,W]` float32
**standard deviations** — the same convention as `uncertainty.py`'s `raw_npy/`,
so the maps feed `uncertainty_calibration.py` unchanged.

Notes:

- `ugac` is stored in the checkpoint config and restored automatically, so
  inference needs no flag. Loading a vanilla checkpoint with `--save_aleatoric`
  exits with an error rather than emitting garbage.
- Aleatoric variance is closed-form, `sigma^2 = alpha^2 * Gamma(3/beta) / Gamma(1/beta)`
  (`ggd_aleatoric_var`) — one forward pass, no sampling. This is complementary to
  the ensemble epistemic term: total `sigma^2 = sigma^2_ale + sigma^2_epi`.
- At `(alpha, beta) = (1, 1)` the NLL is exactly L1, so UGAC strictly generalises
  the vanilla objective. Verified in `tests/test_ugac.py`.
- Positivity uses softplus plus a floor rather than the paper's ReLU, which can
  emit exactly zero and make `log(inv_alpha)` and `lgamma(1/beta)` diverge.
  `beta` is clamped to [0.2, 4.0] and the power term is evaluated in log space.
- Head overhead is ~6.3k parameters (<0.06% of the A->B generator), so the S/M/L
  budgets are unaffected.
- **Changes what the model learns** — a UGAC run is not comparable to an existing
  vanilla checkpoint without retraining.

#### Generator size configurations

Targets are ~10 M (S), ~50 M (M), ~100 M (L) A→B parameters.

```bash
# CycleGAN / DCLGAN
--cyclegan_ngf 64  --cyclegan_n_blocks 8     # S
--cyclegan_ngf 128 --cyclegan_n_blocks 10    # M
--cyclegan_ngf 192 --cyclegan_n_blocks 9     # L

# UNIT (adds shared bottleneck blocks)
--unit_ngf 64  --unit_n_blocks 8  --unit_n_blocks_shared 2   # S
--unit_ngf 128 --unit_n_blocks 10 --unit_n_blocks_shared 2   # M
--unit_ngf 192 --unit_n_blocks 9  --unit_n_blocks_shared 3   # L

# MUNIT
--munit_ngf 64  --munit_n_content_blocks 3   # S
--munit_ngf 128 --munit_n_content_blocks 5   # M
--munit_ngf 192 --munit_n_content_blocks 4   # L

# UVCGAN
--uvcgan_ngf 48  --uvcgan_vit_features 96  --uvcgan_vit_blocks 6    # S
--uvcgan_ngf 96  --uvcgan_vit_features 384 --uvcgan_vit_blocks 6    # M
--uvcgan_ngf 128 --uvcgan_vit_features 384 --uvcgan_vit_blocks 17   # L

# CycleDiffusion (count covers both eps_A and eps_B; both needed at inference)
--cd_base_channels 48  --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 1   # S
--cd_base_channels 84  --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 2   # M
--cd_base_channels 128 --cd_channel_mult 1,2,4   --cd_num_res_blocks 2   # L
```

Shared hyperparameters: Adam, lr `2e-4`, β = (0.5, 0.999), batch size 1, linear LR
decay from the halfway point to zero. Loss weights: CycleGAN/UVCGAN λ_cyc 10,
λ_id 0.5; UNIT λ_GAN 1, λ_recon 10, λ_KL 0.01; MUNIT λ_img 10, λ_c 1, λ_s 1;
DCLGAN λ_cyc 10, λ_id 0, λ_DCL 1; CycleDiffusion ε-prediction only.

### Inference

```bash
# GAN models — single forward pass per tile
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --outdir ./output/

# Subset of WSIs (folders 001–003)
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --data_range 1,3 --ckpt model.pt --outdir ./output/

# CycleDiffusion — DDIM inversion with eps_A, decode with eps_B
python inference.py --model cyclediffusion --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --cd_steps 200 --outdir ./output/

# B→A is symmetric for CycleDiffusion (invert with eps_B, decode with eps_A)
python inference.py --model cyclediffusion --direction B2A --data path/to/tiles/testB \
    --ckpt model.pt --cd_steps 200 --outdir ./output_B2A/

# MUNIT with random style sampling
python inference.py --model munit --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --num_samples 3 --outdir ./output/

# MUNIT with style from a reference image
python inference.py --model munit --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --style_image ref.png --outdir ./output/

# Deterministic output (all models)
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --seed 42 --outdir ./output/

# Resume an interrupted job — skips written tiles, redoes the most recent one
python inference.py --model cyclegan --direction A2B --data path/to/tiles/testA \
    --ckpt model.pt --outdir ./output/ --resume
```

More `--cd_steps` improves DDIM inversion fidelity at proportional cost; 200 each
way is the study setting.

### Evaluation

```bash
# FID — InceptionV3 pool3, 2048-d (unpaired, distribution-level)
python evaluation.py --metric fid --path_real real/ --path_fake generated/ --device cuda

# Patch-based SSIM (paired, matched by filename)
python evaluation.py --metric patch_ssim --path_real real/ --path_fake generated/ \
    --patch_size 64 --patches_per_image 16

# LPIPS — VGG16 perceptual distance, lower is better (paired)
python evaluation.py --metric lpips --path_real real/ --path_fake generated/ --device cuda

# Cycle reconstruction error A→B'→A' (MAE in [0,255])
python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \
    --direction A2B --device cuda

# …with heatmaps, overlays and raw per-pixel maps for calibration
python evaluation.py --metric regen_error --path_A data/HE --model cyclegan --ckpt model.pt \
    --direction A2B --overlay_dir ./regen/ --save_error_npy --device cuda

# Regen error from precomputed A' tiles — no model inference re-run
python evaluation.py --metric regen_error \
    --path_A data/HE --path_A_regen ./inference_B2A/ \
    --overlay_dir ./regen/ --save_error_npy

# Save results to CSV (any metric)
python evaluation.py --metric patch_ssim --path_real real/ --path_fake generated/ \
    --save_csv results.csv

# Tissue-only evaluation — masks auto-detected (images/ → masks/ sibling)
python evaluation.py --metric patch_ssim --path_real testB/tiles/testB \
    --path_fake ./inference/ --min_tissue_fraction 0.1
```

`--save_error_npy` writes `<overlay_dir>/error_npy/<stem>.npy` and requires
`--overlay_dir`. `--path_A_regen` mode needs neither `--model` nor `--ckpt`, and
works for every family including CycleDiffusion.

**Tissue filtering (all metrics):** `--min_tissue_fraction FLOAT` sets the minimum
fraction of non-zero mask pixels for a tile to count (default 0 = all). `--mask_dir`
overrides mask auto-detection. Tiles with no matching mask are always included.

**Available metrics are `fid`, `patch_ssim`, `lpips`, `regen_error` only.** Full-image
SSIM, the DINOv2 FID backend and the external-judge error proxy were removed: the
paper uses patch-SSIM, InceptionV3, and each model's own inverse generator.

### Reconstruction

Reconstructed files keep the original WSI filename. Mask outputs are
`{stem}_mask.tif`. Overlapping regions are averaged by default.

```bash
# Pass the dataset directory — all per-WSI CSVs are found automatically
python reconstruct.py --metadata path/to/tiles/trainA --output ./reconstructed/

# Reconstruct from translated tiles (inference output)
python reconstruct.py --metadata path/to/tiles/testA \
    --tile_dir ./inference_output/ --output ./reconstructed/

# Or a single per-WSI CSV
python reconstruct.py --metadata path/to/tiles/trainA/001/tiles_metadata.csv \
    --output ./reconstructed/

# Both RGB and mask, averaging overlaps
python reconstruct.py --metadata path/to/tiles_metadata.csv --output ./reconstructed/ \
    --mode rgb_and_mask --blend average
```

`--mode` is `rgb` | `mask` | `rgb_and_mask` | `auto`; `--blend` is `average` | `overwrite`.

### Uncertainty Maps

```bash
python uncertainty.py --model cyclegan --data /path/to/ensemble_outputs/ \
    --output ./uncertainty_out

# Tissue-masked, custom normalisation bounds, WSI subset
python uncertainty.py --model cyclegan --data /path/to/ensemble_outputs/ \
    --output ./uncertainty_out --mask_dir path/to/tiles/testA \
    --min_tissue_fraction 0.001 --data_range 1,5 \
    --lower-percentile 1 --upper-percentile 99
```

- Expects ensemble member directories `model_01/`, `model_02/`, … under `--data`;
  members are discovered by globbing `model_*`, so K is whatever exists.
- Per-pixel value is **√(Σ per-channel sample variance)** (ddof=1) in 0–255
  intensity units — i.e. already a standard deviation, not a variance.
- Outputs: `raw_npy/` (the input to every downstream step), `heatmaps/` (magma PNGs,
  qualitative only), `mean_rgb/` (ensemble mean, used as the virtual stain),
  `summary.json`.
- The percentile flags affect `heatmaps/` only; `raw_npy/` is never rescaled.

Reduce to per-tile σ̄ and plot the per-family distribution:

```bash
python aggregate_uncertainty.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --tiles_metadata  path/to/tiles/testA \
    --mask_dir        path/to/tiles/testA \
    --min_tissue_fraction 0.001 \
    --outdir          ./uncertainty_out/cyclegan/per_wsi_csv/

python plot_uncertainty_boxplot.py --base /path/to/ensemble --outdir ./uncertainty_boxplot/
```

`aggregate_uncertainty.py` writes one CSV per WSI (`tile_name, mean_uncertainty`),
deriving WSI membership from the `NNN/` component of the npy path so tile IDs
repeating across WSIs do not collide.

### Descriptor-Space Uncertainty (φ_struct)

Implements `kidney_ood_data_plan.md` §5 and the §2.1 error decomposition. Averages in
**descriptor space**, never pixel space — this is a companion to `uncertainty.py`, not a
replacement: that one is per-tile per-pixel and produced the BMVC numbers, but
`uncertainty_strategy.md` §2.1 forbids pixel-space averaging for the bias identity.

φ_struct is six marginal statistics of a ~1–2 mm region, in two reference classes:

| Component | Reference | Pays the floor |
|---|---|---|
| `task_specific_value` (CPA), `beta0_per_mm2`, `beta1_per_mm2`, `regional_dispersion` | real PSR, level B | yes |
| `lumen_fraction`, `tissue_fraction` | H&E input, level A | **no** |

```bash
# One ensemble -> procedural uncertainty only
python compute_phi_uncertainty.py \
    --ensemble /path/ensemble/cyclegan/data_large/model_medium/wsi_masks_final \
    --tiles_metadata /path/tiles/testA --he_dir /path/reconstructed_he \
    --outdir ./phi_uncertainty/

# Fold x seed grid -> procedural AND data-exposure (one --fold per data block)
python compute_phi_uncertainty.py \
    --fold /path/ensemble_ugac/cyclegan/data_001_007/model_small/wsi_masks_final \
    --fold /path/ensemble_ugac/cyclegan/data_008_014/model_small/wsi_masks_final \
    --tiles_metadata /path/tiles/testA --outdir ./phi_uncertainty/
```

Outputs `per_region.csv` (μ per descriptor, Var, procedural, data_exposure) and
`summary.json` (aggregates, reference classes, parameter record).

Package `uncertainty_phi/`: `descriptors` (the vector), `regions` (grid from
tiles_metadata x/y), `ensemble` (per-member φ, then μ/Var), `decompose` (law of total
variance), `floor` (bracketed floor covariance), `whiten` (Ledoit–Wolf, Mahalanobis,
bias²).

Notes that will bite otherwise:

- **Reconstructions are at 0.221 µm/px**, not the 0.442 the model saw:
  `utils.reconstruct_wsi` upsamples tiles back to `tile_size`. Regions are sized in mm
  for that reason; `--mpp` defaults to the source resolution.
- **Regions, not tiles.** β₀/β₁ and dispersion do not decompose over tiles — components
  and loops cross tile boundaries — so the pipeline consumes stitched WSI masks.
- **The H&E tissue footprint is computed per WSI, then cropped.** Doing it per region
  loses every lumen touching a region border, since `binary_fill_holes` only fills
  enclosed background.
- **Σ comes from the floor, never from the observed discrepancies** — whitening by the
  covariance of what you are measuring normalises the bias away. `whiten.py` takes Σ as
  an explicit argument so there is no code path that gets this wrong.
- **Negative bias² and data components are reported, not clipped.** A negative value is
  the go/no-go signal that the discrepancy has sunk into the floor.
- Bias against a real target is **not** computed yet: it needs the floor measured (§7)
  and, for kidney, the liver-trained segmenter validated out of distribution (§6.2).

#### Per-descriptor floor — the go/no-go pilot

Run this **before** building on the bias term. If the observed discrepancy lands near
the floor there is no headroom and `bias² = observed² − d` comes out at or below zero.

```bash
python estimate_floor.py \
    --real_psr /path/psr_masks/real/psr_masks_wsi_final \
    --tiles_metadata /path/tiles/testB \
    --real_he /path/reconstructed_he \
    --outdir ./floor_pilot/
```

The readout is **per descriptor**, not pooled, because a single number hides whether
any individual component is stable enough between levels to carry a bias signal. CPA
averages over millions of pixels and concentrates fast; β₀/β₁ count discrete events and
behave Poisson-like, so a region holding ~50 loops has a relative SD near 14% between
levels before threshold sensitivity. The decisive column is
`floor_to_signal` = floor SD / between-region SD, with verdicts `usable` (<0.5),
`marginal` (0.5–0.9), `floor-limited` (≥0.9).

Four estimators, in decreasing authority — precedence is per component, so each
descriptor uses the best bound available to it:

| Source | Covers | Direction | Needs |
|---|---|---|---|
| `direct` (`--psr_level_b`) | all six | measured | a **second real PSR level** |
| `variogram` (default) | all six | conservative | nothing extra |
| `cross_stain` (`--real_he`) | lumen, tissue only | conservative | real H&E WSIs |
| `split_half` (always) | all six | **anti-conservative** | nothing |

**Why the variogram matters.** Without a second PSR level the collagen descriptors have
no cross-level upper bound at all — cross-stain cannot reach them, since collagen is not
measurable in H&E. That would leave them on the split-half *lower* bound, and too small
a floor makes `bias² = observed² − floor²` too **large**: bias would read high, the
unsafe direction.

The variogram substitutes in-plane spatial variation. Semivariance rises with separation
and flattens at a **sill**, the fully-decorrelated limit. When structures no longer align
between levels the effective through-plane separation is already large, so the relevant
lag sits at or near the sill — the exact level spacing need not be known — and
`γ(∞) ≥ γ(h)` means the sill *over*-estimates the floor, which under-states bias.

It assumes rough isotropy at region scale: that moving 200 µm sideways perturbs a
descriptor about as much as moving 200 µm deeper. Fine for liver, shakier for kidney's
cortex/medulla layering, so restrict to a cortex mask there (§8). Lags beyond half the
largest separation are discarded — only corner-to-corner pairs survive out there and edge
effects produce a spurious upturn. `sill_reached` is reported **per descriptor**; where
it is False that component's curve is still climbing and its bound is an under-estimate.

Every row carries `floor_source` and `bound_direction`, so a number resting on the
anti-conservative lower bound is visible as such: it can support an upper-bound claim
about bias, never a point estimate.

If the topological terms come back floor-limited, CPA stands alone and the §5.3
lumen-filler blind spot reopens — a real result, worth knowing early.

#### Stain-perturbation sensitivity — is the "bias" really the segmenter?

Applying one segmenter to both arms cancels *anatomy-driven* error, which is common to
real and virtual, but **not** *appearance-driven* error — appearance is precisely where
the two arms differ (§6.2). This bounds the second component with no manual annotation.

Holds anatomy fixed and moves only colour: take a real PSR slide and transform it toward
the virtual's LAB statistics, t = 0 (untouched) → t = 1 (real anatomy, virtual colour).
Segment each step. The tissue never changes, so any descriptor drift is measurement
artefact, and its size is the error bar that belongs on any bias number.

```bash
python stain_sensitivity.py make-series \
    --real_psr /path/real_psr_wsis/ --virtual_psr /path/.../reconstructed/model_01/ \
    --outdir /work2/.../perturbation/

sbatch scripts/segment_psr_perturbation.sh        # array size must match --fractions

python stain_sensitivity.py analyse \
    --masks /work2/.../perturbation/masks/ \
    --tiles_metadata /path/tiles/testB --outdir /work2/.../perturbation/
```

Reads `shift_over_region_sd` — the artefact in units of real biological spread.
Anything above ~0.25 is flagged: the segmenter reacts to colour at a scale comparable
to genuine variation, so fold that shift into the floor and treat a bias of similar
size as unproven.

Two guards worth knowing: `make-series` reports the **out-of-gamut fraction** at each t,
because clipping is non-invertible and would break the fixed-anatomy premise — narrow
`--fractions` if it climbs past a percent. And a descriptor with zero between-region
spread but a non-zero shift reports `inf` rather than `n/a`; that is the worst case, not
an unknown one.

Sequence it as: eyeball the kidney masks (catches catastrophic failure, free) → this
test (bounds the differential, hours) → manual annotation only if this shows large
sensitivity.

**Registration:** none of this needs it, and neither does the ensemble variance — all
members generate from the same H&E, so region *r* is the same tissue across members.
Bias against the real PSR *does* need region-level correspondence (a thumbnail affine,
not pixel registration; §3), and that mapping does not exist yet. Within a corresponding
region no per-structure matching is required: β₀/β₁ are densities, marginal statistics
in the same sense as CPA.

SLURM chain for the UGAC grid (all seven scripts share one decomposition, so array
indices line up end to end):

```bash
sbatch scripts/train_ensemble_cyclegan_ugac.sh       # 0-49   5 blocks x 10 seeds
sbatch scripts/infer_ensemble_cyclegan_ugac.sh       # 0-49   A→B + aleatoric
sbatch scripts/recon_ensemble_ugac.sh                # 0-249  + 5 test WSIs
sbatch scripts/segment_psr_ugac.sh                   # 0-249  Dataset314_SR_light
sbatch scripts/apply_he_mask_ugac.sh                 # 0-49
sbatch scripts/fill_tissue_holes_ugac.sh             # 0-49   -> wsi_masks_final/
```

### Uncertainty Calibration

Pairs ensemble uncertainty with cycle-reconstruction error and reduces both to
scalar scores: within-tile Spearman ρ, across-tile Pearson/Spearman, and a
tile-level reliability diagram with ECE.

**Prerequisite:** re-tile the test set with zero overlap so border pixels are not
double-counted in pooled ECE and across-tile statistics.

```bash
# (a) Train K members with different --seed into model_01/ … model_NN/
# (b) Run inference per member, mirroring that layout
# (c) Per-pixel ensemble variance
python uncertainty.py --model cyclegan --data ./ensemble_outputs/cyclegan/ \
    --output ./uncertainty_out/

# (d) Per-pixel error maps — self-cycle, per member
python evaluation.py --metric regen_error \
    --path_A path/to/tiles/testA \
    --model cyclegan --ckpt ./ensemble/cyclegan/model_01/checkpoints/step_750000.pt \
    --direction A2B --overlay_dir ./regen_cyclegan_m01/ --save_error_npy --device cuda

# (e) Calibration
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --error_dirs      ./regen_cyclegan_m01/error_npy/ \
    --mask_dir        ./tissue_masks_flat/ \
    --tiles_metadata  path/to/tiles/testA \
    --outdir          ./calibration_cyclegan/

# (e′) Ensemble-mean error — pass every member's error dir; averaged per-pixel
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --error_dirs      ./regen_cyclegan_m01/error_npy/ ./regen_cyclegan_m02/error_npy/ \
                      ./regen_cyclegan_m03/error_npy/ \
    --mask_dir ./tissue_masks_flat/ --tiles_metadata path/to/tiles/testA \
    --outdir ./calibration_cyclegan/
```

Inputs:

- `--uncertainty_dir` — flat directory of `<stem>.npy` from `uncertainty.py`. Use
  `raw_npy/`.
- `--error_dirs` — one or more flat directories of `<stem>.npy` from
  `evaluation.py --save_error_npy`; multiple are averaged per-pixel first.
- `--mask_dir` — flat directory of tissue mask `<stem>.tif` (any non-zero = tissue).
  Required unless `--no_mask`; background inflates ρ and ECE spuriously.
- `--tiles_metadata` *(optional)* — dataset root with per-WSI `tiles_metadata.csv`,
  enabling the `per_wsi.csv` rollup via the `source_file` column.

Other flags: `--n_bins 10` (quantile bins for reliability/ECE),
`--min_tissue_pixels 256` (skip near-empty tiles), `--title` (figure prefix).

Outputs in `--outdir`:

- `per_tile.csv` — tile_stem, source_wsi, n_tissue_pixels, spearman_rho,
  pearson_rho_within, mean_u, mean_e
- `per_wsi.csv` — per-WSI rollup (only with `--tiles_metadata`)
- `summary.json` — within-tile Spearman mean/std/median, across-tile
  Pearson/Spearman, ECE, reliability bins, parameter record
- `calibration.png` — 1×3 figure: reliability diagram, within-tile ρ histogram,
  across-tile mean(U) vs mean(E) scatter

Interpretation: within-tile ρ → +1 means uncertain pixels are wrong pixels, ≈0
uninformative, <0 anti-calibrated. Across-tile ρ catches the case where uncertainty
is locally calibrated but flat at tile level (useless for triage). ECE → 0 means bins
lie on y = x after p1–p99 normalisation.

Cycle error is a **proxy**, not ground truth: when forward and inverse generators
share a bias, both may ignore the same feature and the round trip still reconstructs
the source despite a poor forward translation.

### Aggregate Calibration — Per-Model Summaries

Pools the per-WSI `per_tile.csv` files from `scripts/run_calibration_all.sh` and recomputes
every metric on the full tile pool per model, so the Spearman distribution,
across-tile correlations and reliability diagram come from all tiles together rather
than an average of per-WSI summaries.

```bash
python aggregate_calibration.py --base /path/to/ensemble --outdir ./calibration_combined/
sbatch scripts/aggregate_calibration.sh
```

Outputs `{outdir}/{model}/summary.json`, `{outdir}/{model}/calibration.png`, and
`{outdir}/all_models.csv`. `--n_bins` sets the quantile bin count (default 10).
Expects `per_tile.csv` at
`ensemble/{MODEL}/data_large/{MODEL_SIZE}/calibration/{MODEL}/wsi{NNN}/`.

### PSR Positive Area Segmentation

Collagen masks come from a frozen nnU-Net v2 model, **`Dataset314_SR_light`**,
run in WSI mode on reconstructed whole slides via a direct `nnUNetv2_predict`
call (no wrapper script). The same dataset is used for the scaling study and the
ensemble study.

```bash
export nnUNet_results=/path/to/nnunet/nnUNet_results
export nnUNet_raw=/path/to/nnunet/nnUNet_raw

nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i /path/to/reconstructed_wsis/ \
    -o /path/to/wsi_masks/ \
    -f 0 -tr nnUNetTrainer -c 2d -p nnUNetPlans \
    -npp 1 -nps 1 -device cpu
```

Inference tiles must be stitched into whole slides first (`reconstruct.py`, or
`scripts/recon_all_configs.sh` for the 54-config grid).

Label convention: `0` background, `1` tissue, `2` PSR-positive. `compare_psr.py`
reads masks with `tifffile` and takes `[..., 0]` for 3-channel TIFs.

`-npp`/`-nps` are nnU-Net worker counts; keep them at 1 on constrained nodes —
the committed scripts request 256 GB and run CPU-only on the `paula` partition.

### PSR Mask Post-Processing

Both steps run between segmentation and comparison and materially affect CPA.

```bash
# Zero out PSR predictions outside the H&E tissue boundary
python apply_he_mask.py --psr_masks ./psr_masks_wsi/ --he_masks ./he_tissue_masks/ \
    --outdir ./psr_masks_wsi_cleaned/

# Fill enclosed background inside the tissue footprint (labels 1+2 as foreground)
python fill_tissue_holes.py --masks ./psr_masks_wsi_cleaned/ \
    --outdir ./psr_masks_wsi_final/
```

`apply_he_mask.py` accepts a directory (matched by stem) or a single TIF pair;
multi-channel TIFs use the `[..., 0]` slice, and an HE mask of different spatial size
is resized nearest-neighbour. PSR files with no matching HE mask are warned and
skipped.

`fill_tissue_holes.py` treats the **union** of labels 1 and 2 as foreground —
filling only label 1 would mark every PSR-positive pixel as a hole and relabel it.

### PSR Distribution Comparison

Compares collagen proportionate area between real SR and one or more generated
mask sets, matched by WSI stem.

```bash
python compare_psr.py --masks_real ./psr_masks_real/ \
    --masks_generated ./masks_cyclegan/ ./masks_unit/ \
    --labels cyclegan unit --outdir ./psr_comparison/
```

Outputs: `per_wsi.csv` (wsi, condition, psr_fraction), `summary.json` (per-condition
stats and paired metrics vs real), `comparison.png` (box + points),
`paired_scatter.png` (real vs generated per WSI, annotated with r and ρ),
`paired_metrics.png`.

Paired metrics vs real SR — the primary comparison, since testA/testB are serial
sections of the same blocks: `n_matched`, `pearson_r`/`pearson_pvalue`,
`spearman_rho`/`spearman_pvalue`, `mae_paired` (the headline CPA MAE), and
`mean_paired_diff_generated_minus_real` (signed bias; positive = over-estimates).

Flags: `--label_tissue` / `--label_psr` (default 1 / 2), `--strip_prefix` (drop the
first `_`-delimited token before stem matching, e.g. `SR_slide.tif` ↔ `HE_slide.tif`).

Unpaired distributional metrics (Wasserstein, KS, std ratio) were removed — the
paper reports paired CPA MAE.

### Combined Metric Figures

```bash
python plot_combined_metrics.py --eval_indir /path/to/Eval --psr_indir /path/to/psr_comparison \
    --outdir ./combined_metrics_plot/
sbatch scripts/plot_combined_metrics.sh
```

Writes `combined_metrics.png` (2×2: Patch-SSIM, LPIPS, FID, CPA MAE) and
`combined_metrics.csv` (outer join, one row per model × model_size × data_size).
Colour = data size, marker = model size, error bars = ±1 std across WSIs, star =
best config per model.

```bash
python plot_ranking_correlation.py --csv ./combined_metrics_plot/combined_metrics.csv \
    --outdir ./combined_metrics_plot/
sbatch scripts/plot_ranking_correlation.sh
```

Ranks all 54 configurations by each metric independently (rank 1 = best,
direction-aware) and computes the pairwise Spearman matrix. Writes
`ranking_correlation.png` (4×4, RdYlGn), `ranking_correlation.csv`, and
`ranking_correlation_pvalues.csv`. This answers whether image-quality metrics can
substitute for the CPA pipeline when selecting a model across the whole zoo.

### SLURM Pipelines

All job scripts live in `scripts/`. They resolve Python through
`PROJECT_ROOT=I2I-Stain-Zoo`, a path relative to the **submit directory**, and never
`cd` — so submit from the parent of the repository:

```bash
sbatch I2I-Stain-Zoo/scripts/infer_small_models.sh
```

Every script has a pre-flight skip guard and is safe to re-submit after an
interruption.

**Scaling study (54 configs):**

```bash
sbatch scripts/infer_small_models.sh          # A→B tiles
sbatch scripts/eval_all_configs.sh            # FID, patch-SSIM, LPIPS
sbatch scripts/recon_all_configs.sh           # stitch tiles → reconstructed WSIs
sbatch scripts/segment_psr_nn_light_all_configs.sh  # Dataset314_SR_light → wsi_masks/
sbatch scripts/apply_he_mask_all_configs.sh   # → psr_masks_wsi_cleaned/
sbatch scripts/fill_tissue_holes_all_configs.sh  # → psr_masks_wsi_final/
sbatch scripts/compare_psr_all_configs.sh     # → CPA MAE per model
```

Real SR reference: `scripts/recon_real_psr.sh` (stitch real testB tiles) →
**[no committed segmentation script]** → `scripts/apply_he_mask_real.sh` →
`scripts/fill_tissue_holes_real.sh` → consumed by `scripts/compare_psr_all_configs.sh`
as `psr_masks/real/psr_masks_wsi_final/`.

The real-SR segmentation step has no script in the repository: run
`nnUNetv2_predict` with `Dataset314_SR_light` over the output of
`scripts/recon_real_psr.sh` by hand. Note that `scripts/apply_he_mask_real.sh` currently reads
`psr_masks/real/psr_masks_wsi/`, so either write the predictions there or adjust
that path.

**Reverse inference and regen error:**

```bash
sbatch scripts/infer_B2A_all.sh        # B'→A' tiles, for regen error without re-inference
sbatch scripts/eval_regen_B2A_all.sh   # CPU-only MAE pass over tile pairs
```

**Ensemble / uncertainty:**

```bash
sbatch scripts/train_ensemble_cyclegan.sh        # one per family; array size sets K
sbatch scripts/infer_ensemble_cyclegan.sh        # A→B per member
sbatch scripts/infer_ensemble_cyclegan_B2A.sh    # B→A per member
sbatch scripts/compute_ensemble_uncertainty.sh   # per-pixel variance
sbatch scripts/compute_ensemble_regen_error.sh   # per-member error maps
sbatch scripts/run_calibration_all.sh            # calibration per model per WSI
sbatch scripts/aggregate_calibration.sh          # pool tiles per model
sbatch scripts/recon_ensemble_A2B.sh
```

Ensemble CPA mirrors the same segmentation: `scripts/recon_ensemble_A2B.sh` →
`scripts/segment_psr_nn_light_ensemble.sh` (Dataset314_SR_light) →
`scripts/apply_he_mask_ensemble.sh` → `scripts/fill_tissue_holes_ensemble.sh` →
`scripts/compare_psr_ensemble.sh`.

All six families use **K = 10** ensemble members (`--array=0-9` in the
`train_ensemble_*` and `infer_ensemble_*` scripts; the CycleDiffusion inference
scripts use `--array=0-49`, a 2D decomposition of 10 members × 5 test WSIs).

## Architecture

### Model Interface

All models implement the interface consumed by `BaseTrainer`
(`trainer/base_trainer.py`):

- `generator_parameters()` → params for the generator optimiser
- `discriminator_parameters()` → params for the discriminator optimiser (may be empty)
- `compute_generator_loss(batch)` → `(loss, log_dict, visuals_dict)`
- `compute_discriminator_loss(batch, visuals)` → `(loss, log_dict)`

### Shared Components (`base_models.py`)

GAN building blocks: `Encoder` → `ResnetBottleneck` → `Decoder` (with `ResnetBlock`),
`NLayerDiscriminator` (70×70 PatchGAN), `ImagePool`, `GANLoss`, `init_weights`,
`info_nce`, `PatchSampler`, `discriminator_loss`, `identity_loss`.

Diffusion blocks (used by CycleDiffusion): `DiffusionSchedule`, `timestep_embedding`,
`UNetConfig`, `DDPMUNet`, `ResBlock`, `AttentionBlock`, `Downsample`, `Upsample`,
`GroupNorm32`, `SiLU`, `ZeroModule`. These previously lived in `models/miudiff.py`
and were moved here when MIUDiff was removed — CycleDiffusion imports them from
`base_models`.

### Model-Specific Notes

| Model | Key mechanism |
|---|---|
| CycleGAN | Cycle-consistency loss, paired Enc→Bn→Dec generators |
| UNIT | Shared bottleneck + KL divergence on a variational latent |
| MUNIT | Content/style decomposition with AdaIN; style sampling at inference |
| DCLGAN | Dual patch-level contrastive (InfoNCE) feature matching |
| UVCGAN | UNet–ViT hybrid with cycle-consistency; masked-image pretrain stage |
| CycleDiffusion | Two unconditional DDPMs; DDIM inversion of the source into a shared noise code, then decode |

### Data Pipeline

- `datasets/unpaired_dataset.py` — training (A+B folders, pseudo-random pairing)
- `datasets/single_domain_dataset.py` — inference (single folder)
- `datasets/target_only_dataset.py` — domain-B-only loading (retained; no current
  model uses it since the pretrain-on-B stages were removed)
- `datasets/transforms.py` — resize to 256×256, normalise to [-1, 1]
- Formats: `.png .jpg .jpeg .tif .tiff .bmp .webp`
- With `data_range=(start, end)` datasets load `root/{i:03d}/images/` for `i` in
  `[start, end]`; without it they walk the whole root

### Key Conventions

- Images normalised to [-1, 1] for training, denormalised to [0, 1] for saving.
- AMP via `--amp` (disabled for CycleDiffusion, see above).
- Checkpoints: `{"model": state_dict, "config": asdict(cfg), "model_name": str, …}`
  at `output/checkpoints/step_<N>.pt`.
- Config auto-restores from the checkpoint on `--init_ckpt` (train) and `--ckpt`
  (inference); checkpoints without `"config"` fall back to CLI args and defaults.
- **Step-based training:** `--steps` total optimiser updates, `--save_steps` (default
  250,000) checkpoint frequency, `--log_steps` (default 1,000) logging frequency.
- **Auto-resume:** `BaseTrainer.train()` scans `output/checkpoints/` for `step_*.pt`
  at startup and resumes from the latest — no flag needed. If the target step count
  is already reached it exits immediately.
- **Training time tracking:** elapsed time accumulates across resumes and is stored
  in each checkpoint; `output/training_meta.json` records `accumulated_seconds`,
  `human_readable`, `last_updated_step`, `avg_seconds_per_1k_steps`.
- **Log format:** `[S00001000 |   12.3s] loss_G:0.4017 …` — step number and wall time
  since the previous log.
- No external diffusion libraries — DDPM/DDIM sampling is implemented from scratch.
- No `requirements.txt`. Dependencies: torch, torchvision, numpy, scipy, pandas,
  matplotlib, pillow, tifffile, tqdm, pytest (plus nnU-Net v2 for CPA only).
- Test suite: `pytest tests/ -q` — 100 tests, CPU-only, ~2 s.
