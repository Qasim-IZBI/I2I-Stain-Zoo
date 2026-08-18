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

#### The ensemble grid — crossed subsets × seeds

The uncertainty study trains **vanilla CycleGAN** on a crossed grid: K = 5
disjoint training subsets × S = 10 seeds = **M = 50 members**, at the small
generator size, spanning training folders 001–035 in subsets of seven.

```bash
sbatch scripts/train_ensemble_cyclegan_grid.sh          # --array=0-49
sbatch --array=0-49%10 scripts/train_ensemble_cyclegan_grid.sh  # cap concurrency
sbatch --array=10-19 scripts/train_ensemble_cyclegan_grid.sh    # one subset only
```

| tasks | folders | output |
|---|---|---|
| 0–9   | 001–007 | `ensemble_grid/cyclegan/data_001_007/model_small/models/model_{01..10}/` |
| 10–19 | 008–014 | `…/data_008_014/…` |
| 20–29 | 015–021 | `…/data_015_021/…` |
| 30–39 | 022–028 | `…/data_022_028/…` |
| 40–49 | 029–035 | `…/data_029_035/…` |

**Why the grid is crossed, not flat.** Members sharing a subset differ only by
seed, so their spread is *procedural*. Subset means differ because different
slides were seen, so their spread is *data exposure*. The law of total variance
separates the two exactly as they are indexed — but only if both factors vary,
which `train_ensemble_cyclegan.sh` (one subset, ten seeds) cannot do. That flat
script is retained for the BMVC-era vanilla ensemble and is a different
experiment.

Subsets are **disjoint rather than nested**, so a difference between them reflects
*which* slides were seen, not how many — the opposite of the nested 25/50/100%
fractions in the scaling study.

Subset 5 needs folders 031–035, which post-date the 001–030 BMVC training split.
They were tiled on 2026-08-10, so all five subsets are live; the pre-flight check
remains and fails fast with the missing paths rather than dying in the dataloader
hours later.

Inference uses the same 50-job decomposition, so array indices line up one-to-one:

```bash
sbatch scripts/infer_ensemble_cyclegan_grid.sh       # A→B
```

| step | reads | writes |
|---|---|---|
| train | trainA/trainB subset | `{subset}/model_small/models/model_{NN}/` |
| A2B | that checkpoint | `{subset}/model_small/inference/model_{NN}/` |

Kept under `ensemble_grid/` rather than `ensemble/` so the crossed grid cannot be
confused with the flat 10-member runs stored there.

#### UGAC — aleatoric heads (retired 2026-08-09)

> **Retired as the generator.** The UGAC ensemble did not produce usable virtual
> stain, and the descriptor-space decomposition contains no aleatoric term, so
> nothing downstream consumes the heads. `--cyclegan_ugac`, `--save_aleatoric` and
> `tests/test_ugac.py` all still work; the `scripts/*_ugac.sh` chain is kept for
> provenance. **Do not mix its outputs with `ensemble_grid/`.**

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

`--save_aleatoric` writes `{outdir}/aleatoric_npy/<stem>.npy` as `[H,W]` float32
**standard deviations** — the same convention as `uncertainty.py`'s `raw_npy/`,
so the maps feed `uncertainty_calibration.py` unchanged.

Notes:

- `ugac` is stored in the checkpoint config and restored automatically, so
  inference needs no flag. Loading a vanilla checkpoint with `--save_aleatoric`
  exits with an error rather than emitting garbage.
- Aleatoric variance is closed-form, `sigma^2 = alpha^2 * Gamma(3/beta) / Gamma(1/beta)`
  (`ggd_aleatoric_var`) — one forward pass, no sampling.
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

# Any layout without the scaling study's {model}/data_large/{size}/ tree —
# the UGAC and grid chains group by TRAINING SUBSET, not by model family
python plot_uncertainty_boxplot.py \
    --group data_001_007=/path/{block}/model_small/uncertainty/cyclegan/per_wsi_csv \
    --group data_008_014=... --outdir ./uncertainty_boxplot/
sbatch scripts/plot_uncertainty_boxplot.sh    # builds the five --group args
```

`--group LABEL=PATH` takes the `per_wsi_csv` directory directly and preserves the
order given, so the script needs to know no layout at all. The wrapper refuses to
plot when any block is missing: a block silently absent from the figure is worse
than a failed job, because the plot still renders and reads as a complete
comparison of whatever survived.

`aggregate_uncertainty.py` writes one CSV per WSI (`tile_name, mean_uncertainty`),
deriving WSI membership from the `NNN/` component of the npy path so tile IDs
repeating across WSIs do not collide.

### Descriptor-Space Uncertainty (φ_struct)

Implements `kidney_ood_data_plan.md` §5 and the §2.1 error decomposition. Averages in
**descriptor space**, never pixel space — this is a companion to `uncertainty.py`, not a
replacement: that one is per-tile per-pixel and produced the BMVC numbers, but
`uncertainty_strategy.md` §2.1 forbids pixel-space averaging for the bias identity.

φ_struct is six marginal statistics of a ~1–2 mm region, in two reference classes:

| Component | Read from | Reference | Pays the floor |
|---|---|---|---|
| `task_specific_value` (CPA), `beta0_per_mm2`, `beta1_per_mm2`, `regional_dispersion` | the member's collagen mask | real PSR, level B | yes |
| `lumen_fraction`, `beta0_lumen_per_mm2`, `beta1_lumen_per_mm2` | the member's **generated SR**, thresholded inside the H&E footprint | real H&E, level A — the *same physical section* | **no** |

`tissue_fraction` is **not** in the vector. It is the H&E footprint's coverage,
shared by every member and by the reference, so it has zero variance *and* zero
error — nothing to decompose and nothing to calibrate. It is still written to
`per_region.csv` as a QC column.

`beta1_lumen_per_mm2` is the direct test of the §5.3 lumen-filler failure: a model
that paints collagen over vessels keeps the whitespace area roughly and loses the
loops, which area alone cannot see.

> **The three lumen terms are UNAVAILABLE on the UC liver cohort (2026-08-16).**
> Both routes to a lumen mask are closed, for independent reasons:
>
> - **Brightness on the generated SR.** Its histogram has no bimodality and no
>   plateau: the footprint sweeps from 7% of the canvas to 100% across
>   0.50–0.725, and at 0.675 only 13.7% is bright where the slide background
>   alone is ~35–40%. The model does not reproduce whitespace, not even for
>   background. At the H&E's own 0.65 it calls **22% of the slide lumen** against
>   the H&E's 4% on the same tissue.
> - **Enclosed background in the mask.** `Dataset314_SR_light` is trained to label
>   lumen as *tissue*, so no mask contains enclosed background to find. This is a
>   property of the **segmenter, not the model** — it applies equally to the real
>   SR masks, so it is not a virtual-versus-real asymmetry.
>
> Run without `--lumen_root`: the three terms come back NaN, `decompose` handles
> NaN columns, and `calibrate_phi` reports them as having no reference rather than
> scoring them. The machinery is kept for the kidney arm or any cohort where
> whitespace survives translation.
>
> A consequence worth carrying into the methods: since the segmenter counts lumen
> as tissue, **CPA's denominator is tissue-including-lumen** on both arms. It is
> consistent, so it does not bias the comparison, but it is not the same
> denominator a histologist would assume.

Lumen densities are per mm² of the **H&E footprint**, not of the label mask's
tissue. The reference side is the real H&E and has no collagen labels at all, and a
density is only comparable if its denominator is.

```bash
# One ensemble -> procedural uncertainty only
python compute_phi_uncertainty.py \
    --ensemble /path/ensemble/cyclegan/data_large/model_medium/wsi_masks_final \
    --tiles_metadata /path/tiles/testA --he_dir /path/reconstructed_he \
    --outdir ./phi_uncertainty/

# Subset x seed grid -> procedural AND data-exposure (one --fold per subset, all 5)
# --he_masks is REQUIRED here, not optional; see below.
python compute_phi_uncertainty.py \
    --fold /path/ensemble_grid/cyclegan/data_001_007/model_small/wsi_masks_final \
    --fold /path/ensemble_grid/cyclegan/data_008_014/model_small/wsi_masks_final \
    --tiles_metadata /path/tiles/testA --he_masks /path/HE_tissue \
    --region_px 2048 --outdir ./phi_uncertainty/

# Kidney arm: cortex only
python compute_phi_uncertainty.py \
    --fold ... (all five) \
    --tiles_metadata /path/tiles/testA_kidney --he_dir /path/reconstructed_he_kidney \
    --roi_dir /path/cortex_masks/ --outdir ./phi_uncertainty_kidney/
```

**`--he_masks` is required with `--fold`.** The tissue filter needs a reference to
measure coverage against, and with `--he_masks` it uses the H&E footprint — a
property of the slide, so every fold filters to the same regions. Without it the
filter falls back to the first member's collagen mask, which is a *model output*:
a region near `--min_tissue_fraction` is then kept by one fold and dropped by
another, and the run aborts with

```
fold .../data_008_014/... produced 119 regions but the first fold produced 118
```

That message is about the tissue filter, not about `--tiles_metadata` or
`--region_mm`. A single `--ensemble` is unaffected — there is only one grid to be
consistent with — but pass `--he_masks` there too, so the two runs are comparable.

`--roi_dir` restricts the grid to an anatomical compartment, given per-WSI binary
masks named `<stem>.tif` (resized nearest-neighbour if annotated at thumbnail
magnification). The kidney arm needs it for two independent reasons: cortex and
medulla differ systematically in fibrosis distribution, so a grid sampling both
mixes two populations; and cortex/medulla layering breaks the isotropy the
variogram floor assumes. `--min_roi_fraction` (default 0.5) thresholds on
**coverage, not the centre point** — a region half in medulla is not a cortex
measurement. A WSI with no matching mask is **excluded and warned about**, never
passed through whole: a missing case is recoverable, a contaminated one is not.

Outputs `per_region.csv` and `summary.json` (aggregates, reference classes,
parameter record). Per region the CSV carries:

| column | what |
|---|---|
| `mu_<name>` | ensemble mean, the point prediction |
| `sd_total_<name>` / `sd_procedural_<name>` / `sd_data_<name>` | per-descriptor spread — what calibration pairs with an error |
| `foldN_mu_<name>` / `foldN_sd_<name>` | the subset-level prediction, one per training subset |
| `var_total_descriptor_space`, `var_total_anova`, `procedural`, `data_exposure` | the summed scalars |
| `tissue_fraction` | QC only — H&E footprint coverage |
| `y0/y1/x0/x1`, `area_mm2` | the region box, which the heatmap and the calibration reuse verbatim |
| `wsi_h`, `wsi_w` | the frame the boxes were cut from, so a reference can be checked for *matching* it rather than merely covering it |

A negative variance component is a real ANOVA outcome near zero and is reported
rather than clipped, but it has no square root, so its `sd_` column is empty
rather than NaN.

`--region_px` fixes the region side in pixels exactly, overriding `--region_mm`.
Sizes are in mm by default because reconstructions sit at source resolution and a
pixel count means a different physical scale on a different cohort; use it where
the pixel grid itself matters, i.e. a seamless heatmap.

`per_region.csv` carries **two** totals and they are not the same number.
`var_total_descriptor_space` is the pooled plug-in variance over every member of
every fold, which ignores the fold structure; `var_total_anova` is the ANOVA total
`summary.json` reports, and it is the one that equals `procedural + data_exposure`.

On SLURM, as one job or one per WSI:

```bash
sbatch scripts/compute_phi_uncertainty_grid.sh          # all folds, all WSIs, one job
sbatch scripts/compute_phi_uncertainty_grid_array.sh    # --array=0-19, one WSI per task
python aggregate_phi_uncertainty.py \
    --indir ./phi_uncertainty/per_wsi --outdir ./phi_uncertainty/ --expect 20
```

Splitting over WSIs is **exact, not an approximation**: `decompose()` works region
by region and regions never cross slide boundaries, so per-WSI runs hold final
per-region numbers. Only the three means in `summary.json` are cohort-level, and
`aggregate_phi_uncertainty.py` recovers them by pooling the rows — which is what
`var_total_anova` is written for. It refuses to pool runs disagreeing on
`region_mm`, `mpp`, the fold list or the thresholds, and `--expect` makes a short
pool an error rather than a quietly smaller cohort.

Both scripts split over **WSIs, never folds** — one fold alone gives procedural
variance and no data-exposure term at all — and take every path from the
environment, so the kidney arm needs no edit:

```bash
sbatch --export=ALL,TEST_A=...,HE_DIR=...,ROI_DIR=...,WHITE_THRESH=...,OUTDIR=... \
    scripts/compute_phi_uncertainty_grid_array.sh
```

Their pre-flights catch the two failures the pipeline otherwise swallows: a member
with no mask for a slide yields an all-NaN slab rather than an error, shrinking the
effective member count behind the variance; and `region.crop` is bare numpy slicing
with no resize and no shape assertion, so an H&E at the wrong pyramid level returns
a short crop instead of raising.

`--he_dir` accepts the **original H&E WSIs**, not only reconstructions. Tiling
starts at `(0,0)` with stride = tile size and `reconstruct_wsi` upsamples tiles back
to `tile_size`, so the reconstruction is the original truncated to a whole number of
tiles at the same origin and scale — region boxes index identical pixels in either.
The original skips the 512→256→512 round trip, so it is the sharper input for
`he_bright`; it changes only the `mu_lumen_fraction` / `mu_tissue_fraction` columns,
never the variance, since those two terms are identical across members.

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
- **`--white_thresh` is a per-cohort measurement, not a constant.** It is the whitespace
  cut behind `lumen_fraction` and `tissue_fraction` (default 0.85, `WHITE_THRESH` in
  `descriptors.py`; the SLURM scripts take it as `WHITE_THRESH=`). `he_bright` requires
  **every channel** to clear it, so the number to compare against is the per-pixel
  channel **minimum** — an 8-bit conversion in Fiji shows a channel *average*, always
  the larger of the two, so a grey level read off one is an upper bound on the
  threshold rather than the threshold. **A `mu_lumen_fraction` around 1e-5 in
  `per_region.csv` means the cut sits above the lumens** and they are being counted as
  tissue; the UC liver cohort has them at grey ~180, i.e. 0.706 before the min-channel
  correction. Changing it moves only the two H&E-referenced descriptors, which are
  identical across members and so contribute zero variance — the decomposition is
  unaffected, only the level-A columns need the re-run.
- **`--qc_dir` writes the lumen call out for inspection.** One region per WSI as a
  TIF pair — the label mask (0 outside, 1 tissue, 2 lumen) and the matching H&E
  crop — with the region box in the filename and the measurements in the TIFF
  description. Same label convention as the nnU-Net masks, so Fiji overlays them.
  Written once per WSI, not per member. `--qc_max_px` caps the crop; at full
  resolution a 1.5 mm H&E region is ~100 MB before compression.
- **`--tiles_metadata` is optional.** Without it the region grid is sized from each
  mask. The real SR arm is evaluated whole-slide and has no tiling, and a tiled
  extent is truncated to a whole number of tiles — up to one region row/column
  shorter than the image — so do not compare a run built one way against a run
  built the other.
- **Σ comes from the floor, never from the observed discrepancies** — whitening by the
  covariance of what you are measuring normalises the bias away. `whiten.py` takes Σ as
  an explicit argument so there is no code path that gets this wrong.
- **Negative bias² and data components are reported, not clipped.** A negative value is
  the go/no-go signal that the discrepancy has sunk into the floor.
- Bias against a real target is **not** computed yet: it needs the floor measured (§7)
  and, for kidney, the liver-trained segmenter validated out of distribution (§6.2).

#### Lumen masks — the stage before φ

> Not runnable on the UC liver cohort — see the note above the threshold section.
> Kept for cohorts whose generated stain reproduces whitespace.

The three H&E-referenced descriptors are read from each member's **generated SR**,
so they are member-specific. `phi_for_wsi` loads the H&E once per WSI, outside the
member loop, and doing this inline would mean several GB of RGB fifty times per
slide. So it is a pipeline stage, in the shape of segment → clean → fill:

```bash
sbatch scripts/make_lumen_masks_grid.sh        # --array=0-49, the virtual side

# the reference side is a single run: the real H&E against its own footprint
python make_lumen_masks.py --rgb_dir ${HE_RGB} --he_masks ${HE_TISSUE} \
    --white_thresh 0.65 --min_object_px 64 --outdir /path/lumen_masks_real
```

Then `compute_phi_uncertainty.py --lumen_root .../lumen_masks` consumes
`model_NN/` directories in parallel with the collagen ones. Member *m* must be the
same model in both; a count mismatch is refused rather than pairing one member's
collagen with another's lumen.

**The footprint comes from the H&E tissue mask** (`--he_masks`), the same one
`apply_he_mask.py` applies to the collagen masks — so the study carries one
definition of tissue, not two. It also keeps `--white_thresh` out of the
denominator: the footprint is what breaks across the threshold sweep, so deriving
it by thresholding would leave an unstable parameter under every lumen density.
Holes are filled, so a tissue segmentation that excluded lumens still works. Pass
the same directory to `compute_phi_uncertainty.py --he_masks`, which then needs no
H&E RGB at all.

**`--min_object_px` removes speckle once per slide, not per region**, and must be
identical on both arms (the rule §5.4.4 already imposes on collagen). Cleaning per
region would leave `lumen_fraction` measured on the raw mask while β₀/β₁ were
measured on a cleaned one — area and topology disagreeing about what a lumen is.
The default 64 px is ~3.1 µm² at 0.221 µm/px, i.e. speckle; a 10 µm capillary is
~1600 px. Each run reports what fraction of raw lumen area it removed and warns
above half.

#### Calibration — does the spread predict the error?

Three stages, because two of them are measurements and only the third is cheap:

| stage | reads | writes | cost |
|---|---|---|---|
| `compute_phi_uncertainty.py` | the 50 ensemble mask sets | `per_region.csv` (μ, σ) | hours |
| `compute_phi_reference.py` | the real masks | `reference_phi.csv` (`real_*`) | hours |
| `calibrate_phi.py` | **both CSVs** | ρ, E\|z\|, the figures | seconds |

```bash
sbatch scripts/check_frame_alignment.sh   # Step 0: does the SR share the H&E frame?

# measure the real tissue ONCE
sbatch scripts/compute_phi_reference.sh
#   or directly:
python compute_phi_reference.py \
    --phi_csv  ./phi_uncertainty/per_region.csv \
    --real_psr /path/psr_masks/real/psr_masks_wsi_final --strip_prefix \
    --he_masks /path/HE_tissue \
    --outdir   ./calibration_phi/

# then calibrate as often as you like
python calibrate_phi.py \
    --phi_csv       ./phi_uncertainty/per_region.csv \
    --reference_csv ./calibration_phi/reference_phi.csv \
    --outdir        ./calibration_phi/
python calibrate_phi.py --phi_csv ... --reference_csv ... \
    --prediction fold --outdir ./calibration_phi_fold/
```

Ensemble spread measures disagreement between members, not error. The BMVC 2026
result is that cycle error does not calibrate it; this asks the same question of an
external target. Per descriptor: Spearman ρ(σ, |error|), E|z| where z = |error|/σ,
a reliability curve, and a normalised ECE.

**Why the reference is its own stage.** It loads a full-slide mask per WSI and runs
`betti` plus a structure tensor over every region, and *nothing about it depends on
the ensemble* — only on the real masks and the region boxes. Folding it into the
calibration meant every change to `--n_bins`, `--n_boot`, `--prediction` or the
figure re-measured tissue. `reference_phi.csv` also stands on its own: it is what
the descriptors are on real liver, which the methods section needs regardless.

Two arms, independent, either omittable — chosen when the **reference** is built,
not at calibration time:

- `--real_psr` scores the four collagen terms against the real SR. Region *r* is
  only the same tissue if the SR was resampled onto the H&E grid; the geometry is
  checked against `wsi_h`/`wsi_w` and a mismatch **exits** rather than scoring
  different tissue under the same region id. Being *larger* than the regions is
  not enough — one UC slide is 34794×27942 against the H&E's 32521×23201, which
  covers every box while aligning with none of them. **On the UC liver cohort this
  is the only available arm**, which makes that frame check the gate for the whole
  calibration.

  **One excess is benign and expected:** φ is gridded on a reconstruction, which
  `utils.reconstruct_wsi` truncates to a whole number of tiles, so the original is
  larger by up to one tile at the same origin and scale — 24967×34757 against a φ
  frame of 24576×34304 (= 48×512 and 67×512) is the same frame, and the boxes
  index identical pixels. Truncation cannot lose a whole tile, so an excess below
  `--tile_size` (default 512) is accepted with a `[note]` and anything larger
  exits. **`--tile_size` is `tile.py --tile_size`, not `--resize_to`** —
  reconstructions sit at source resolution.
- `--real_lumen` scores the three H&E-referenced terms against the real H&E. Same
  physical section, so no floor and no frame question — but see why the lumen
  terms cannot be computed on this cohort at all.

**`--strip_prefix` is required with `--real_psr`.** The real collagen masks are
named after the SR slides (`SR_d31_BDL+A_M2`) while φ is gridded on the H&E
(`HE_d31_BDL+A_M2`), so without it every WSI is skipped and the run ends with `no
reference regions produced`. Same rule as `apply_he_mask.py` and `compare_psr.py`;
the error message tests whether stripping would bridge the two and says so. Both
sides are keyed the same way, so it does not break `--he_masks`, which already
matched. Two files collapsing to one key is **fatal**, not last-one-wins.

Regions come from `--phi_csv` verbatim rather than by rebuilding a grid, so the two
sides cannot drift apart through a parameter that differs by one. **Pass the same
`--he_masks` the φ run used** — a footprint built differently on the two sides means
the comparison divides by different denominators.

Two consistency checks run automatically, because a mismatch on either side is
invisible in the output:

- **Against the φ run.** `compute_phi_reference.py` reads `summary.json` beside
  `--phi_csv` and refuses if `--mpp`, `--min_object_px` or `--closing_px` differ —
  the error is a difference between two measurements, so a component size that
  differs between them puts part of that difference into the parameters rather
  than the tissue. `--allow_param_mismatch` overrides it deliberately.
- **Against the grid.** The region boxes travel inside `reference_phi.csv`, so
  `calibrate_phi.py` proves the reference belongs to the grid it is used on.
  `--region_px 1024` against a reference built at 2048 keeps every parameter
  identical while every region moves, and region 7 of slide 3 exists in both — on
  different tissue. That exits too.

`reference_phi.json` records the mask directories and thresholds behind the
reference, and `calibrate_phi.py` copies it into `summary.json`, so a result
carries a trace of how its target was measured.

**It also copies the φ run's own parameters forward**, under `phi_run` — read
from the `summary.json` beside `--phi_csv`. `--roi_dir`, `--region_px` and
`--min_tissue_fraction` belong to `compute_phi_uncertainty.py`, not here: the
grid arrives already built. Without this, a calibration result cannot answer
*"was the kidney arm restricted to cortex?"*, which is invisible in the numbers
and decides whether the grid mixed cortex and medulla. The run prints it:

```
[phi run] regions: region_px=2048  min_tissue_fraction=0.25  ROI=/path/cortex_masks (min_roi_fraction=0.5)
[phi run] regions: region_px=2048  min_tissue_fraction=0.25  ROI=none — the grid covers the whole slide
```

For runs that predate this, the φ parameters are in
`<phi_uncertainty>/agg_phi/summary.json` under `params`.

Outputs `per_region_calibration.csv`, `summary.json`, `calibration_phi.png` (the
working panel) and, as the figure-quality pair, **`reliability_phi.png` plus
`reliability_bins.csv`** — the diagram and the exact numbers behind every point,
so it can be restyled for the manuscript without re-running anything. Per bin:
`sd_lo`/`sd_hi`, `mean_sd`, `mean_error`, `expected_error` (= 0.80·σ),
`ratio_obs_over_expected`, `se_error_by_case`, `n`, `n_wsi`.

**All three variance components are scored, in one figure.** `reliability_phi.png`
draws one panel per descriptor with three curves — total σ, procedural σ (seed)
and data-exposure σ (subset). The prediction is the mean of all 50 in every case,
so the **error is identical across the three and only σ moves**; whichever gives
the higher ρ is the component the calibration rests on. That is the comparison
the crossed 5×10 grid exists to support, and a flat seed-only ensemble has no
data-exposure term to put beside the other two.

`reliability_bins.csv` carries a `component` column, so each curve's points are
recoverable separately. Where the ANOVA left a negative variance component there
is no SD, and those regions drop out of **that component only** — the count is
reported as `n_dropped` and printed on the figure, because a component estimated
as zero on half the regions is a finding about the ensemble rather than a missing
measurement.

`--prediction fold` is a different *prediction*, not a fourth component: it pairs
each subset's mean with that subset's own procedural spread. **Each subset is
scored separately** — five ρ per descriptor, plus an agreement block giving their
median, range and whether the sign is consistent.

Pooling the five would be worse than merely optimistic. Every region enters five
times against **one shared target**, so it adds no evidence; and because subsets
sit at different σ *and* different error levels, pooling induces a between-subset
trend that exists inside no subset. On the UC liver run the pooled β₀ came out at
ρ = +0.312 while the five subsets gave +0.015, −0.017, +0.109, +0.123, +0.091 —
**larger than any of them**. Read the agreement block first: a descriptor whose ρ
changes sign between subsets has not been shown to calibrate, however tight a
pooled interval looks.

Three things the reliability figure does that the compact panel does not, each
because the compact one can mislead:

- **Error bars clustered on the case**, not the region. Over ~285 regions a bin
  mean looks precise when those regions come from twenty slides.
- **The bin population underneath**, annotated with its slide count. Quantile
  bins hold equal region counts by construction but *not* equal numbers of
  slides, and a bin drawn from three cases should not read like one drawn from
  twenty.
- **Axes scaled per panel**, so the calibration line's slope differs between
  them. A shared scale is the textbook form but only works while σ and error are
  comparable — on an over-confident descriptor (β₀ at σ ~40 against error ~500)
  equal limits crush every point into a corner. The diagonal is not drawn at all:
  read the points against the dashed line.

Two conventions that must be stated wherever the numbers are:

- **σ is a predictive SD** — the spread of members, not the standard error of the
  mean. σ/√50 would be tiny and the test would collapse into a test of bias.
- **Reliability is absolute, and the line is E|e| = 0.80σ, not the diagonal.** For
  Gaussian error the mean absolute deviation is σ·√(2/π), so a diagonal would call
  a perfectly calibrated ensemble 20% over-confident. `reliability_bins` in
  `uncertainty_calibration.py` min-max normalises both axes because *pixel*
  uncertainty and *pixel* error carry different units; σ_CPA and |ΔCPA| are both in
  CPA, so normalising here would discard exactly what makes this stronger. The
  normalised ECE is reported for continuity but on synthetic data reads ~0.35
  whether the ensemble is calibrated, over-confident or useless — do not lead with
  it.

**`within_slide.csv` is the confound-controlled result, and on this cohort it is
the one to report.** ρ is computed *inside* each slide and summarised over
slides, so the **slide is the unit of replication** — pooling ~2850 regions from
20 cases and correlating once treats them as 2850 observations. `n_positive` out
of `n_slides` is a sign test anyone can read without trusting a bootstrap.

It also carries `rho_partial_mu_*`, which removes the point prediction `mu`. This
matters more than anything else in the output: **σ tracks how much structure a
region holds** — on the UC liver cohort ρ(σ, μ_CPA) = **+0.76** — and absolute
error grows with the same thing, so a raw ρ is largely the two sharing that
dependence rather than the ensemble knowing where it is wrong. The partial asks
the question that survives review: *does the spread say anything the point
prediction does not already imply?* It partials on `μ` and never on `real`,
because μ is available at inference and the reference is not.

Measured (CPA, slide as unit, n = 20):

| σ | raw ρ | partialled on μ |
|---|---|---|
| total | +0.278 [+0.200, +0.355], 19/20 | **+0.150** [+0.065, +0.241], 14/20, p = 0.006 |
| procedural | +0.246 [+0.169, +0.322], 18/20 | +0.094 [+0.012, +0.181], 12/20, p = 0.105 |
| data-exposure | +0.245 [+0.170, +0.317], 19/20 | **+0.143** [+0.060, +0.229], 16/20, p = 0.006 |

So the ranking claim survives the control, at about half the raw size — and the
component that survives is the **data-exposure** one, which only a crossed grid
can measure. Procedural alone does not.

**A raw ρ that collapses under the partial is a structure-content map wearing an
uncertainty label. Report both, always.**

**`risk_coverage.csv` / `risk_coverage.png` — read the μ baseline before quoting
these.** Ranking by the point prediction alone requires no ensemble and on this
cohort **beats σ** for triage: at 80% coverage μ gives −15.1% against σ's −7.8%
(within-slide: −15.9% vs −11.5%). Absolute error grows with the amount of
collagen, so "discard the regions with most predicted collagen" is a strong
heuristic. Do not present selective prediction as a headline without that
comparison — it is the first check a reader will run. ρ answers
whether σ ranks the error; this answers what that buys. Regions are sorted by σ,
the least certain are discarded, and the error is measured on what remains, at
each `--coverages` fraction (default 1.0 0.9 0.8 0.7 0.5).

Three reference points, and all three belong in any statement of the result:

- **Random selection is unbiased**, so its curve is *exactly* zero — no
  Monte-Carlo baseline is needed, and the curve's departure from zero is the
  effect.
- **The oracle**, ranking by true error, is the ceiling. `capture_of_oracle` is
  the fraction reached, and it is the honest measure of how far this is from
  solved. It is NaN at full coverage, where the ratio is 0/0 and floating point
  would otherwise render a confident 100%.
- **The bootstrap CI** resamples whole slides, deciding whether the reduction
  survives a cohort of twenty cases.

On the UC liver cohort, CPA at 80% coverage: MAE −8.1%, CI [−14.9%, −3.3%],
against an oracle of −41.2% — about a fifth of the achievable gain. The three
topological descriptors sit flat on the random line at every coverage.

**Quote the cluster-bootstrap CI, not the naive p.** `--n_boot` (default 2000)
resamples whole *slides*: regions inside one are spatially correlated, so a p-value
over ~2850 regions describes a cohort of twenty cases as if it held 2850. A slide
drawn twice contributes its regions twice, which is the point. `rho_shuffled` is the
negative control — mean |ρ| with the σ–error pairing broken — and must sit near zero
for ρ to mean anything.

`--prediction grand` pairs the mean of all 50 with the total spread (the deployed
prediction); `--prediction fold` pairs each subset's mean with its procedural spread
alone. Comparing them is the data-exposure claim, which a flat seed-only ensemble
cannot pose.

#### Cycle error vs ensemble spread — the head-to-head

The paper's central contrast is that the cheap self-consistency proxy fails
where a task-relevant target works. Cited on one side and measured on the other,
that is two studies; this makes it one.

```bash
sbatch --export=ALL,REGEN_ROOTS='/path/regen/model_01 /path/regen/model_02' \
    scripts/compare_uncertainty_sources.sh

python compare_uncertainty_sources.py \
    --phi_csv        ./phi_uncertainty/per_region.csv \
    --reference_csv  ./calibration_phi/reference_phi.csv \
    --regen_root     /path/regen/model_01 --regen_root /path/regen/model_02 \
    --tiles_metadata /path/tiles/testA \
    --outdir         ./compare_sources/
```

Regen error enters as another **`component`**, so every existing analysis treats
it as one more curve: the same regions, the same target, the same
slide-clustered bootstrap, the same μ partial, and both figures. A null for
either source therefore cannot be blamed on the protocol.

**Prerequisite, and it is the expensive part:** per-member regen maps at
`REGEN_ROOT/model_NN/wsi{NNN}/error_npy/<tile>.npy`, which is what
`scripts/compute_ensemble_regen_error.sh` already writes. It needs B→A inference
per member first. **Two or three members is enough** — cycle error is a property
of one model's forward/inverse pair, not of the ensemble — so do not queue fifty
before looking at three.

How the two scales are reconciled: regen error is **per tile and per pixel**,
φ is **per region and per descriptor**. Tiling is non-overlapping with stride =
tile size from origin (0,0), so tiles nest exactly — a 2048 px region holds
sixteen 512 px tiles, none straddling a boundary — and the aggregation is exact
rather than an approximation. The run prints the tiles-per-region range and warns
if it is not constant, which is the signal that the region size is not a whole
multiple of the tile size.

Per-tile means are averaged across members rather than averaging the maps first.
The two give the same number because the mean is linear, and the second would
read hundreds of GB. They are cached to `tile_errors.csv`, so re-running with
different binning or bootstrap settings is seconds.

**`tile_name` is read as a string, deliberately.** Tile names are zero-padded
numerics, which pandas parses as `int64` — the lookup then asks for `1.npy`,
every tile reports no error map, and the run fails claiming the directory layout
is wrong when only the padding was lost.

#### Uncertainty heatmaps

```bash
python plot_uncertainty_heatmap.py --phi_csv ./phi_uncertainty/per_region.csv \
    --downsample 32 --outdir ./uncertainty_heatmaps/
```

One PNG per slide (σ on the top row, σ/μ below, one column per descriptor) plus a
float32 TIF per descriptor for overlaying in Fiji or QuPath, carrying its geometry
in the TIFF description. Pair with `--region_px 2048` on the φ run so the blocks
tile without a seam.

**Read both rows.** σ for a count-based descriptor rises with how much structure a
region holds, so a raw σ map can be a collagen-density map wearing an uncertainty
label; σ/μ divides that out. Three states stay distinct: blank = the tissue filter
dropped that region (an absent measurement is not a low one), "constant" = the
descriptor has no spread at all, and a colour ramp only where there is real range.

#### Choosing `--white_thresh` — the plateau, not a guess

```bash
sbatch scripts/calibrate_white_thresh.sh                       # H&E arm
sbatch --export=ALL,HE_DIR=/path/real_sr_wsis,TILES_METADATA=none,\
OUTDIR=/path/white_thresh_sr scripts/calibrate_white_thresh.sh  # SR arm
```

Sweeps the threshold over a few slides and writes `white_thresh.png` (brightness
histogram inside tissue, the `lumen_fraction` curve, and |d ln(lumen)/dt|),
`white_thresh.csv` and `white_thresh.json`. **Run it for both stains** — they sit
at different whitespace levels, and `estimate_floor` takes a separate
`--white_thresh_psr`.

Two things it decides, and only the first is obvious:

- **The stable window.** The footprint fails at *both* ends. Too high and the
  slide background stops reading as bright, so `binary_fill_holes` absorbs it and
  `tissue_fraction` jumps — +21% at 0.725 on every UC liver slide. Too low and the
  tissue itself reads as bright and the footprint erodes — 0.59 → 0.12 by 0.500 on
  the SR. `stable_window` takes the longest run where `tissue_fraction` moves less
  than 0.5% per step; outside it the number describes a different object, not a
  smaller lumen.
- **Whether a plateau exists at all.** A threshold is only reproducible where the
  measurement stops depending on it, so the tool requires |d ln(lumen)/dt| below
  an absolute cut, not merely the flattest point available — without that a
  uniformly sloped curve reports its own middle as a plateau.

Measured on the UC liver cohort: **H&E stable 0.500–0.675, SR 0.600–0.700,
intersection 0.600–0.675, and no plateau in either** (12% and 9% per step). So
`lumen_fraction` there is a convention rather than a measurement; **0.65** is the
committed choice, clear of both cliffs. Use the same value in
`compute_phi_uncertainty.py` and `estimate_floor.py`.

Everything is computed on the per-pixel channel **minimum**, which is what
`he_bright` thresholds — a Fiji 8-bit conversion shows a channel *average* and so
bounds the threshold from above rather than being it.

#### Per-descriptor floor — the go/no-go pilot

Run this **before** building on the bias term. If the observed discrepancy lands near
the floor there is no headroom and `bias² = observed² − d` comes out at or below zero.

```bash
sbatch scripts/estimate_floor.sh              # 12 h / 96 G, single job

# or directly; --tiles_metadata omitted, so the grid is sized from each mask
python estimate_floor.py \
    --real_psr /path/psr_masks/real/psr_masks_wsi_final \
    --white_thresh 0.65 \
    --outdir ./floor_pilot/
```

Not an array job: the variogram bins region pairs across the whole cohort, so the
slides cannot be split over tasks. It loads a full-slide mask per WSI and runs
`betti` plus a structure tensor over every region — hours, not minutes.

Outputs `floor_per_descriptor.csv`, `floor.json` (covariances, the variogram
curve, provenance) and **`floor.png`**: panel A the verdict per descriptor over
the usable/marginal/floor-limited bands with the bracket drawn, panel B the
variogram curves normalised by their own plateau. Read B before believing A — a
floor from a sill that never flattened is an under-estimate, and an
under-estimated floor makes bias read high.

`--region_mm`, `--min_tissue_fraction`, `--min_object_px` and `--closing_px` must
match the `compute_phi_uncertainty` run this floor is compared against, or it
bounds a different discrepancy than the one being measured.

Sweep region size and pool the runs — the one knob that moves the verdict:

```bash
sbatch --export=ALL,REGION_MM=0.75,OUTDIR=./floor_075 scripts/estimate_floor.sh
python plot_floor_sweep.py --runs ./floor_075 ./floor_150 ./floor_250 \
    --outdir ./floor_sweep/
```

`--real_psr` supplies **masks**; the cross-stain bound additionally needs the PSR
**RGB** via `--real_psr_rgb`, because the two stain-invariant descriptors have to be
measured on both images. Passing `--real_he` alone gives no cross-stain bound and says
so — it used to build both sides from the same image, making the delta identically
zero, and a zero floor reads as maximal bias. `cross_stain_floor` now raises on an
all-zero delta rather than returning a bound that measures nothing.

**On the UC liver cohort the cross-stain arm is not computable** and the committed
`scripts/estimate_floor.sh` leaves it off. It bounds only `lumen_fraction` and
`tissue_fraction`, and the SR can measure neither — its footprint is unstable
across the whole sweep, so the bound would describe the thresholds rather than the
level offset. Nothing is lost: the variogram covers all six descriptors and
outranks cross-stain in precedence. Those two rows then read *unknown*, which is
what a level-A descriptor with no estimate should say.

Estimators skip **uncomputable columns, not every row**: a descriptor that is NaN
throughout is dropped from Σ and reported as unknown, rather than the row filter
discarding every region and taking the computable descriptors with it.

**Split-half pairs stay inside a slide.** Pairing across cases folds between-case
biology into a quantity that is supposed to be within-slide sampling noise, which
inflated the lower bound past the variogram upper bound for every descriptor —
an incoherent bracket.

The two stains sit at different whitespace levels (~180 grey for H&E against ~185 for
SR on the UC cohort), hence a separate `--white_thresh_psr`; it defaults to
`--white_thresh`. Use the same `--white_thresh` here as in
`compute_phi_uncertainty.py`, or the floor and the thing it bounds are measured
differently.

The bound assumes region *r* is the same tissue in both images. Thumbnail registration
makes that approximately true, but the grid comes from one `--tiles_metadata` while the
two slides are tiled separately, so any origin or extent mismatch is absorbed into the
floor on top of the level offset. That inflates it — the safe direction — but it is in
the number. The collagen descriptors are unaffected: the variogram is single-slide.

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
| `cross_stain` (`--real_he` **and** `--real_psr_rgb`) | lumen, tissue only | conservative | real H&E **and** real PSR RGB |
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

**Result on the UC liver cohort (2026-08-15): no headroom at any region size.**

| region | CPA | β₀ | β₁ | dispersion | regions | variogram pairs / lag span |
|---|---|---|---|---|---|---|
| 0.75 mm | 1.06 | 1.11 | 1.00 | 1.23 | 1058 | 13,690 / 4.9× |
| 1.5 mm | 0.87 | 0.97 | 0.80 | 1.19 | 279 | 1,197 / 2.2× |
| 2.5 mm | 0.71 | 0.76 | 0.67 | 0.93 | 99 | 120 / 1.4× |

The ratios improve with region size exactly as §4.2 predicts — the floor averages
out faster than the biology — but nothing reaches `usable` (<0.5), and the best
numbers rest on the weakest variogram: at 2.5 mm the sill spans 1.4× of lag over
120 pairs, where a flat curve is the absence of evidence rather than evidence of a
plateau. At 0.75 mm, where the estimate is well conditioned, every descriptor is
floor-limited. Extrapolating, CPA would need ~6 mm regions to clear 0.5, at which
point ~15 regions survive across 20 slides.

**So the bias branch is closed on this cohort.** Report the procedural vs
data-exposure decomposition as the result and this sweep as the documented reason
bias is not claimed. The only estimator that would reopen it is `--psr_level_b`, a
second real PSR level per case, which supersedes the variogram with a measured
cross-level floor.

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

SLURM chain for the ensemble grid (all six scripts share one decomposition — the
same `RANGE_STARTS`, `RANGE_ENDS` and `N_MEMBERS` — so array indices line up end
to end; change one and they all have to change):

```bash
sbatch scripts/train_ensemble_cyclegan_grid.sh       # 0-49   5 subsets x 10 seeds
sbatch scripts/infer_ensemble_cyclegan_grid.sh       # 0-49   A→B
sbatch scripts/recon_ensemble_grid.sh                # 0-249  x 5 test WSIs
sbatch scripts/segment_psr_grid.sh                   # 0-249  Dataset314_SR_light
sbatch scripts/apply_he_mask_grid.sh                 # 0-49
sbatch scripts/fill_tissue_holes_grid.sh             # 0-49   -> wsi_masks_final/
```

Lumen masks first, if the H&E-referenced descriptors are wanted (they are what
the calibration study scores):

```bash
sbatch scripts/make_lumen_masks_grid.sh              # 0-49   -> lumen_masks/
```

Then φ_struct over the finished grid, either in one job or one per WSI:

```bash
sbatch scripts/compute_phi_uncertainty_grid.sh       # all folds, all WSIs, one job
sbatch scripts/compute_phi_uncertainty_grid_array.sh # 0-19   one WSI per task
python aggregate_phi_uncertainty.py --indir .../per_wsi --outdir ... --expect 20

# then the two consumers
sbatch scripts/compute_phi_reference.sh              # real tissue, once
python calibrate_phi.py --phi_csv .../per_region.csv \
    --reference_csv .../calibration_phi/reference_phi.csv --outdir ...
python plot_uncertainty_heatmap.py --phi_csv .../per_region.csv --downsample 32
```

These two are **not** part of the shared decomposition above: they split over WSIs,
while the six chain scripts split over subsets × members. They read all five folds
in every task, which is why they cannot be indexed the same way.

The committed scripts target the 5-WSI BMVC test set. For the 20-case held-out
cohorts, repoint `TEST_A`, then `N_WSIS` (recon, segment) and `WSI_COUNT`
(apply_he_mask, fill_tissue_holes), and scale `--array` to match.

The retired UGAC chain (`scripts/*_ugac.sh`, same array layout, writing to
`ensemble_ugac/`) is kept for provenance. Do not mix its outputs with these.

**Post-inference runbook:** `PIPELINE_AFTER_INFERENCE.md` sequences the whole
thing — which steps are gated, which are not, and what is still unbuilt.

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
run in WSI mode on whole slides via a direct `nnUNetv2_predict` call. The same
dataset is used for the scaling study, the ensemble study and the real-SR
reference, which is the whole point — one segmenter, so anatomy-driven error is
common to both arms and cancels.

Committed wrappers exist for the grid (`scripts/segment_psr_grid.sh`) and for the
real SR originals (`scripts/segment_psr_real.sh`); the call below is what to run by
hand for anything else. Both wrappers stage each slide into a temp directory under
the `_0000` channel suffix nnU-Net demands, then rename the prediction back.

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

`-npp`/`-nps` are nnU-Net worker counts; keep them at 1 on constrained nodes.

**Request 256 GB.** These are whole slides at 0.221 µm/px and nnU-Net holds several
full-size arrays at once: the input as float32 RGB, one float32 logit plane per
class, and a Gaussian accumulator. On a 34677×40514 case that is ~43 GB before any
transient copy, and 64 GB OOM-kills the job **after** the sliding window has
finished — leaving a header-only TIF that reads back as shape `(0,)` and fails two
frames later in `apply_he_mask` rather than at the read. When the GPU cannot hold
the logits nnU-Net prints "Moving results arrays to CPU" and the host takes the
whole burden, so VRAM pressure arrives as a RAM request.

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
skipped — but a run where **nothing** matched raises, since an empty output directory
otherwise reads to the SLURM skip guards as a finished stage.

`--strip_prefix` drops the first `_`-delimited token from both sides before matching,
so `SR_slide.tif` pairs with `HE_slide.tif` — the same rule as `compare_psr.py`, and
required on the real-SR arm, whose masks are named after the SR slides while the
tissue masks are named after the H&E ones. Two HE masks collapsing to one key is
**fatal, not last-one-wins**: applying the wrong slide's footprint changes the CPA
denominator with nothing visible to show for it.

`fill_tissue_holes.py` treats the **union** of labels 1 and 2 as foreground —
filling only label 1 would mark every PSR-positive pixel as a hole and relabel it.

Both write through `utils.write_label_mask`, which is **zlib-compressed and atomic**.
A three-valued mask stored raw costs one byte per pixel — 603 MB for a 23552×25600
slide, and each slide passes through four full-size copies — while zlib takes one to
two orders of magnitude off; every reader here goes through `tifffile.imread` and
decompresses transparently. The write lands on `<name>.tif.partial` and is renamed
only on success, so an interrupted write (a full filesystem, an exhausted quota)
leaves nothing the `*.tif` skip guards can count as a finished slide. **Files written
by nnU-Net itself are not protected** — a truncated `wsi_masks/` entry reads back as
shape `(0,)` and fails downstream in `apply_mask`, not at the read.

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

Open with `set -eo pipefail`, **not** `-euo`. The Anaconda module runs `activate.d`
hooks that read unset variables, so `-u` there kills the job before the first echo
and the log comes back empty. Switch `-u` on after `conda activate`, as the grid
family does. Whether it bites depends on the submitting environment, since
`--export=ALL` carries whatever conda state the login shell had.

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

**Real SR reference** — the arm every generated mask set is measured against. Two
routes to it, differing only in what gets segmented:

```bash
# (a) from stitched testB tiles — the BMVC route
sbatch scripts/recon_real_psr.sh              # stitch real testB tiles
#   [no committed segmentation script for this route: run nnUNetv2_predict with
#    Dataset314_SR_light over its output by hand. apply_he_mask_real.sh reads
#    psr_masks/real/psr_masks_wsi/, so write the predictions there or adjust it.]
sbatch scripts/apply_he_mask_real.sh          # → psr_masks_wsi_cleaned/
sbatch scripts/fill_tissue_holes_real.sh      # → psr_masks_wsi_final/

# (b) from the original thumbnail-registered SR WSIs — no reconstruction at all
sbatch scripts/segment_psr_real.sh            # --array=0-19, one slide per task
sbatch scripts/apply_he_mask_real_sr.sh       # → psr_masks_wsi_cleaned/
sbatch scripts/fill_tissue_holes_real_sr.sh   # → psr_masks_wsi_final/
```

Either way the output is consumed as `psr_masks/real/psr_masks_wsi_final/` by
`scripts/compare_psr_all_configs.sh`, and as `--real_psr` by `estimate_floor.py`.
Route (b)'s three scripts take every path from the environment (`SR_WSI_DIR`,
`HE_MASKS_DIR`, `PSR_DIR`, `OUT_DIR`, `WSI_COUNT`, `STRIP_PREFIX`), so a new cohort
needs no edit.

Two asymmetries against the virtual arm that route (b) introduces, neither of which
cancels and neither of which is visible in the output:

- **Resolution parity is not checked for you.** The virtual arm is segmented on
  reconstructions at 0.221 µm/px, and `Dataset314_SR_light` is a 2d model with a
  fixed patch size, so it sees whatever scale it is handed. An original at a
  different mpp makes every CPA difference confounded with scale.
  `segment_psr_real.sh` logs each slide's shape and resolution tags for comparison
  against the reconstructed virtual WSIs.
- **The H&E footprint is exact on the virtual arm and approximate here.** There the
  mask is generated *from* the H&E; here the SR is a serial section registered only
  at thumbnail level, and the nearest-neighbour resize in `apply_he_mask.py` corrects
  scale but not translation or rotation. Slide edges and detached fragments are where
  it shows.

Applying the same footprint to both arms is nevertheless correct: CPA's denominator
is tissue area, so measuring each arm on its own footprint makes the two fractions
incomparable whatever the collagen does.

`segment_psr_real.sh` discovers slides null-delimited (`find -print0 | sort -z`),
because this cohort has slides named `'SR_w10_BDL+A_M7'.tif` with the quotes inside
the filename, and errors if `--array` overshoots the slide count rather than
silently producing nothing.

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

These are the **flat** BMVC-era ensembles: one training set, ten seeds, so they
carry procedural variance only. The uncertainty-decomposition study uses the
crossed `scripts/*_grid.sh` chain instead (5 subsets × 10 seeds, `--array=0-49`),
which is the only design that can separate data exposure from procedural spread.
Do not read `train_ensemble_cyclegan.sh` and `train_ensemble_cyclegan_grid.sh` as
variants of one experiment.

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
- Test suite: `pytest tests/ -q` — 289 tests, CPU-only, ~5 s.
