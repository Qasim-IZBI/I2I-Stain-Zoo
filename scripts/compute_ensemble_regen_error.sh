#!/bin/bash
#SBATCH --job-name=i2i_ens_regen_err
#SBATCH --output=logs_ensemble/regen_error_%A_%a.out
#SBATCH --error=logs_ensemble/regen_error_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-29   # 30 jobs = 6 models × 5 WSIs

# Computes |A - mean(B2A)| per tile using evaluation.py's precomputed-A' mode.
# Per-WSI jobs so error maps are organised by WSI to align with uncertainty.py
# output (raw_npy/001/images/*, raw_npy/002/images/*, ...) for calibration.
#
# Inputs (per WSI):
#   --path_A      : testA/{NNN}/images/   (ground truth HE for this WSI)
#   --path_A_regen: regen_stats/{model}/mean_rgb/{NNN}/images/
# Outputs (per WSI, written to {ENSEMBLE_ROOT}/regen_error/wsi{NNN}/):
#   heatmaps/   : per-tile error heatmap PNGs
#   error_npy/  : per-tile error maps as .npy (consumed by uncertainty_calibration.py)

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble

# -----------------------------
# 2D decomposition: 6 models × 5 WSIs
# Layout: tasks 0–4  → cyclegan        WSI 1–5
#         tasks 5–9  → unit             WSI 1–5
#         tasks 10–14 → munit           WSI 1–5
#         tasks 15–19 → dclgan          WSI 1–5
#         tasks 20–24 → uvcgan          WSI 1–5
#         tasks 25–29 → cyclediffusion  WSI 1–5
# -----------------------------
N_WSIS=5
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))   # 0 … 5
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))      # 0 … 4

WSI_NUM=$(( WSI_IDX + 1 ))                       # 1 … 5
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")          # 001 … 005

# Per-model config
# Index:        0          1      2       3        4        5
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL="${MODELS[$MODEL_IDX]}"
MODEL_SIZE="${MODEL_SIZES[$MODEL_IDX]}"

TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
MASK_DIR="${TEST_A}/${WSI_FOLDER}/masks"
MIN_TISSUE=0.1

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# Per-WSI input paths — both are leaf image directories so list_images finds
# tiles directly; matching is by basename (tile IDs are unique across WSIs).
PATH_A="${TEST_A}/${WSI_FOLDER}/images"
PATH_A_REGEN="${ENSEMBLE_ROOT}/regen_stats/${MODEL}/mean_rgb/${WSI_FOLDER}/images"

# Per-WSI output dir keeps error maps WSI-organised to align with uncertainty
# output structure (raw_npy/001/images/*, raw_npy/002/images/*, ...).
OVERLAY_DIR="${ENSEMBLE_ROOT}/regen_error/wsi${WSI_FOLDER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}  WSI=${WSI_FOLDER}"
echo "path_A       : ${PATH_A}"
echo "path_A_regen : ${PATH_A_REGEN}"
echo "Output       : ${OVERLAY_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${PATH_A}" ]; then
    echo "[ERROR] testA WSI directory not found: ${PATH_A}"
    exit 1
fi

if [ ! -d "${PATH_A_REGEN}" ] || [ -z "$(ls -A "${PATH_A_REGEN}" 2>/dev/null)" ]; then
    echo "[ERROR] Mean B2A WSI directory missing or empty: ${PATH_A_REGEN}"
    echo "        Run compute_ensemble_regen_stats.sh first."
    exit 1
fi

# Per-WSI skip guard
if [ -d "${OVERLAY_DIR}/error_npy" ] && [ -n "$(ls -A "${OVERLAY_DIR}/error_npy" 2>/dev/null)" ]; then
    echo "[SKIP] Already completed: ${OVERLAY_DIR}/error_npy"
    exit 0
fi

mkdir -p "${OVERLAY_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# No --device needed: compute_regen_error_precomputed is CPU-only (pure MAE).
run_cmd python I2I-Stain-Zoo/evaluation.py \
    --metric              regen_error \
    --path_A              "${PATH_A}" \
    --path_A_regen        "${PATH_A_REGEN}" \
    --overlay_dir         "${OVERLAY_DIR}" \
    --save_error_npy \
    --mask_dir            "${MASK_DIR}" \
    --min_tissue_fraction "${MIN_TISSUE}"

echo "Done. Error maps for ${MODEL} WSI ${WSI_FOLDER} saved to ${OVERLAY_DIR}"
