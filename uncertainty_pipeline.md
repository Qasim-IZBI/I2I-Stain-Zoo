# Uncertainty Pipeline

End-to-end documentation for the ensemble epistemic uncertainty pipeline,
covering two calibration tracks: **regen_error** (pixel-level, existing) and
**CPA-MAE** (WSI-level, new).

---

## 1. Overview

Epistemic uncertainty is estimated via deep ensembles: train the same architecture
N times with different random seeds. Per-pixel variance across ensemble members
captures where the model is uncertain.

Two calibration questions are answered:

| Track | Error proxy | Granularity | Data points |
|---|---|---|---|
| **Regen-error** | `\|A − mean(B2A)\|` per pixel | Tile-level Spearman ρ, AUSE, ECE | ~thousands of tiles |
| **CPA-MAE** | `\|PSR_fraction_member − PSR_fraction_real\|` per WSI | WSI-level scatter | 6 models × 5 WSIs = 30 |

---

## 2. Models and ensemble sizes

All ensembles use `data_large` and the model sizes below.

| Model | Model size | Members | Checkpoint step |
|---|---|---|---|
| CycleGAN | model_medium | 10 | step_750000.pt |
| UNIT | model_medium | 10 | step_750000.pt |
| MUNIT | model_medium | 10 | step_750000.pt |
| DCLGAN | model_small | 10 | step_750000.pt |
| UVCGAN | model_small | 10 | step_500000.pt (stage2) |
| CycleDiffusion | model_small | 10 | step_750000.pt |

> MIUDiff is excluded from both calibration tracks — it has no inverse generator
> for regen_error self-cycle and no PSR segmentation pipeline set up.

Ensemble roots live at:
```
/work2/bz66izin-VSproject/ensemble/{MODEL}/data_large/{MODEL_SIZE}/
```

---

## 3. Regen-error calibration track

### 3.1 Pipeline

```
train_ensemble_{model}.sh           → models/model_{01..10}/checkpoints/
        ↓
infer_ensemble_{model}.sh           → inference/model_{01..10}/          (A→B)
        ↓
infer_ensemble_{model}_B2A.sh       → inference_B2A/model_{01..10}/      (B→A self-cycle)
        ↓
compute_ensemble_regen_stats.sh     → regen_stats/{MODEL}/mean_rgb/{NNN}/images/
        ↓                             (mean B→A tile per WSI via uncertainty.py)
compute_ensemble_regen_error.sh     → regen_error/wsi{NNN}/error_npy/
        ↓                             (|A − mean_B2A| per pixel, [H,W] .npy)
compute_ensemble_uncertainty.sh     → uncertainty/{MODEL}/raw_npy/{NNN}/images/
        ↓                             (pixel variance across members, --log-compress)
run_calibration_all.sh              → calibration/{MODEL}/wsi{NNN}/
        ↓                             (per_tile.csv, summary.json, calibration.png)
aggregate_calibration.sh            → calibration_combined/{MODEL}/
                                      all_models.csv
```

> **Temp state:** `aggregate_calibration.py` currently reads `calibration_nolog/`
> instead of `calibration/` (commit cb4e33a). A second run was done without
> `--log-compress` for comparison. Revert when decided.

### 3.2 Calibration metrics (per tile, then aggregated)

- **Within-tile Spearman ρ** — do uncertain pixels = wrong pixels?
- **Across-tile Pearson/Spearman** — do uncertain tiles = wrong tiles?
- **AUSE** — error-weighted ranking quality (skipped with `--no_ause` for speed)
- **ECE** — reliability diagram shape

### 3.3 Key scripts

| Script | Jobs | Purpose |
|---|---|---|
| `compute_ensemble_regen_stats.sh` | 30 (6×5 WSIs) | Mean B→A per tile |
| `compute_ensemble_regen_error.sh` | 30 (6×5 WSIs) | Per-pixel `\|A − mean_B2A\|` |
| `compute_ensemble_uncertainty.sh` | 30 (6×5 WSIs) | Per-pixel ensemble variance |
| `run_calibration_all.sh` | 30 (6×5 WSIs) | Pair uncertainty + error |
| `aggregate_calibration.sh` | 1 | Pool WSIs per model |

### 3.4 Judge-based B2A (alternative to self-cycle)

Uses a single fixed CycleGAN checkpoint as the B→A inverter for ALL 6 models,
removing the self-cycle confound.

```
infer_judge_B2A_all.sh    → {model}/judge_B2A/model_{01..10}/
```

- **Array:** 60 jobs (6 models × 10 members)
- **Judge checkpoint:** `/work2/bz66izin-VSproject/Outputs_noamp/cyclegan/results/data_large/model_medium/checkpoints/step_750000.pt`
- Output mirrors `inference_B2A/` structure (preserves `{NNN}/images/` subfolders)

### 3.5 Regen-error comparison plot

```
plot_regen_error_boxplot.py / .sh
```

Reads `regen_error/wsi{NNN}/error_npy/` for all 6 models and produces:
- `regen_error_boxplot.png` — pooled boxplot
- `regen_error_violin.png` — pooled violin
- `regen_error_bar.png` — mean ± std bar chart with annotations
- `per_wsi/` — per-WSI versions of the above
- `regen_error_quantiles.csv`

---

## 4. CPA-MAE calibration track

Calibrates uncertainty against PSR/CPA positive area error at the WSI level.

### 4.1 Approach (Option 2)

For each model and each WSI:

```
mean_cpa_mae[wsi] = mean over 10 members of |PSR_fraction_member − PSR_fraction_real|
```

This measures the expected error of a single draw from the ensemble — what
uncertainty is designed to predict.

### 4.2 Pipeline

```
segment_psr_nn_light_ensemble.sh   → wsi_masks/model_{01..10}/          (nnUNet PSR masks)
        ↓
apply_he_mask_ensemble.sh          → wsi_masks_cleaned/model_{01..10}/  (background zeroed)
        ↓
fill_tissue_holes_ensemble.sh      → wsi_masks_final/model_{01..10}/    (holes filled)
        ↓
compare_psr_ensemble.sh            → ensemble_cpa_comparison/{MODEL}/per_wsi.csv
        ↓                             (PSR fractions: real + 10 members × 5 WSIs)
compute_mean_cpa_mae.py / .sh      → ensemble_cpa_comparison/aggregated/mean_cpa_mae.csv
                                      (30 rows = 6 models × 5 WSIs)
```

In parallel:

```
aggregate_uncertainty.sh           → uncertainty/{MODEL}/per_wsi_csv/{wsi_stem}.csv
        ↓                             (tile_name, mean_uncertainty — one CSV per WSI)
compute_wsi_uncertainty.py / .sh   → ensemble_cpa_comparison/aggregated/wsi_uncertainty.csv
                                      (30 rows = 6 models × 5 WSIs)
```

### 4.3 Key scripts

| Script | Jobs | Purpose |
|---|---|---|
| `segment_psr_nn_light_ensemble.sh` | 300 (6×10×5) | nnUNet PSR masks per member per WSI |
| `apply_he_mask_ensemble.sh` | 60 (6×10) | Zero background predictions |
| `fill_tissue_holes_ensemble.sh` | 60 (6×10) | Fill enclosed holes |
| `compare_psr_ensemble.sh` | 6 (one per model) | PSR fractions all members vs real |
| `compute_mean_cpa_mae.sh` | 1 | Mean `\|member − real\|` per WSI |
| `compute_wsi_uncertainty.sh` | 1 | Mean tile uncertainty → WSI scalar |

### 4.4 Output CSVs and join

**`mean_cpa_mae.csv`** (30 rows):

| Column | Description |
|---|---|
| `model` | Model display name |
| `wsi` | WSI stem |
| `mean_cpa_mae` | Mean of `\|member_i − real\|` across 10 members |
| `std_cpa_mae` | Std across members |
| `n_members` | Members that contributed |
| `real_psr_fraction` | Real SR PSR fraction |
| `mean_psr_fraction` | Ensemble mean PSR fraction |

**`wsi_uncertainty.csv`** (30 rows):

| Column | Description |
|---|---|
| `model` | Model display name |
| `wsi` | WSI stem |
| `mean_uncertainty` | Mean of per-tile uncertainty across all tiles in the WSI |
| `std_uncertainty` | Std across tiles |
| `n_tiles` | Tiles that contributed |

**Join for calibration scatter:**
```python
import pandas as pd
mae = pd.read_csv("mean_cpa_mae.csv")
unc = pd.read_csv("wsi_uncertainty.csv")
df  = mae.merge(unc, on=["model", "wsi"])
# df has 30 rows — plot mean_uncertainty vs mean_cpa_mae, colour = model
```

### 4.5 Real SR reference masks

PSR masks for the real SR test set (processed through the same pipeline):
```
/work2/bz66izin-VSproject/psr_masks/real/psr_masks_wsi_final/
```

HE tissue masks (used by `apply_he_mask_ensemble.sh`):
```
/work2/bz66izin-VSproject/HE_tissue/
```

---

## 5. Uncertainty visualisation

```
aggregate_uncertainty.sh           → uncertainty/{MODEL}/per_wsi_csv/   (tile-level)
        ↓
plot_uncertainty_boxplot.py / .sh  → uncertainty_boxplot/
                                      uncertainty_boxplot.png
                                      uncertainty_violin.png
                                      uncertainty_quantiles.csv
                                      per_wsi/
```

> **Temp state:** `plot_uncertainty_boxplot.py` currently reads `uncertainty_nolog/`
> (commit 0a93c5e). A second uncertainty run was done without `--log-compress`
> for comparison. Revert when decided.

---

## 6. Data paths summary

```
/work2/bz66izin-VSproject/
  ensemble/
    {MODEL}/data_large/{MODEL_SIZE}/
      models/model_{01..10}/checkpoints/         ← trained weights
      inference/model_{01..10}/                  ← A→B tiles
      inference_B2A/model_{01..10}/              ← B→A self-cycle tiles
      judge_B2A/model_{01..10}/                  ← B→A judge (CycleGAN) tiles
      regen_stats/{MODEL}/mean_rgb/{NNN}/images/ ← mean B→A per tile
      regen_error/wsi{NNN}/error_npy/            ← |A − mean_B2A| .npy
      uncertainty/{MODEL}/raw_npy/{NNN}/images/  ← pixel variance .npy
      uncertainty/{MODEL}/per_wsi_csv/           ← tile-level means per WSI
      calibration/{MODEL}/wsi{NNN}/              ← regen calibration per WSI
      wsi_masks/model_{01..10}/                  ← nnUNet PSR masks (raw)
      wsi_masks_cleaned/model_{01..10}/          ← HE-masked PSR masks
      wsi_masks_final/model_{01..10}/            ← hole-filled PSR masks

  ensemble_cpa_comparison/
    {MODEL}/per_wsi.csv                          ← PSR fractions real + members
    aggregated/
      mean_cpa_mae.csv                           ← WSI-level CPA error (30 rows)
      wsi_uncertainty.csv                        ← WSI-level uncertainty (30 rows)

  calibration_combined/{MODEL}/                  ← aggregated regen calibration
  regen_error_boxplot/                           ← regen_error comparison figures
  uncertainty_boxplot/                           ← uncertainty distribution figures
  psr_masks/real/psr_masks_wsi_final/            ← real SR reference masks
  HE_tissue/                                     ← HE tissue boundary masks
```

---

## 7. Run order

### Regen-error track
```bash
# Already done for most models — check regen_error/wsi{NNN}/error_npy/ exists
sbatch compute_ensemble_regen_error.sh   # if not done
sbatch compute_ensemble_uncertainty.sh   # if not done
sbatch run_calibration_all.sh
sbatch aggregate_calibration.sh

# Optional: judge-based B2A (alternative to self-cycle)
sbatch infer_judge_B2A_all.sh

# Comparison plot
sbatch plot_regen_error_boxplot.sh
```

### CPA-MAE track
```bash
# PSR masks (long — GPU jobs)
sbatch segment_psr_nn_light_ensemble.sh   # 300 jobs
sbatch apply_he_mask_ensemble.sh          # 60 jobs
sbatch fill_tissue_holes_ensemble.sh      # 60 jobs

# Comparison and aggregation (fast — CPU)
sbatch compare_psr_ensemble.sh            # 6 jobs
sbatch compute_mean_cpa_mae.sh            # 1 job

# Uncertainty aggregation
sbatch aggregate_uncertainty.sh           # needs compute_ensemble_uncertainty.sh done
sbatch compute_wsi_uncertainty.sh         # 1 job

# Now join mean_cpa_mae.csv + wsi_uncertainty.csv on (model, wsi)
```
