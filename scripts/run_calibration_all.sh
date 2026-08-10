#!/bin/bash
#SBATCH --job-name=i2i_calibration
#SBATCH --output=logs_ensemble/calibration_%A_%a.out
#SBATCH --error=logs_ensemble/calibration_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-29   # 30 jobs = 6 models × 5 WSIs

# Runs uncertainty_calibration.py per-WSI for each model.
# Inputs consumed:
#   uncertainty/{MODEL}/raw_npy/{NNN}/images/  — from compute_ensemble_uncertainty.sh
#   regen_error/wsi{NNN}/error_npy/            — from compute_ensemble_regen_error.sh
#   testA/{NNN}/masks/                         — tissue masks
# Outputs written to:
#   calibration/{MODEL}/wsi{NNN}/
#     per_tile.csv, summary.json, calibration.png
#
# Array layout
# tasks 0–4   → cyclegan        (model_medium)  WSI 1–5
# tasks 5–9   → unit            (model_medium)  WSI 1–5
# tasks 10–14 → munit           (model_medium)  WSI 1–5
# tasks 15–19 → dclgan          (model_small)   WSI 1–5
# tasks 20–24 → uvcgan          (model_small)   WSI 1–5
# tasks 25–29 → cyclediffusion  (model_small)   WSI 1–5

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
# -----------------------------
N_WSIS=5
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))   # 0 … 5
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))      # 0 … 4

WSI_NUM=$(( WSI_IDX + 1 ))                        # 1 … 5
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")           # 001 … 005

# Per-model config
# Index:        0          1      2       3        4        5
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL="${MODELS[$MODEL_IDX]}"
MODEL_SIZE="${MODEL_SIZES[$MODEL_IDX]}"

TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

UNCERTAINTY_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/raw_npy/${WSI_FOLDER}/images"
ERROR_DIR="${ENSEMBLE_ROOT}/regen_error/wsi${WSI_FOLDER}/error_npy"
MASK_DIR="${TEST_A}/${WSI_FOLDER}/masks"
OUTDIR="${ENSEMBLE_ROOT}/calibration/${MODEL}/wsi${WSI_FOLDER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}  WSI=${WSI_FOLDER}"
echo "uncertainty_dir : ${UNCERTAINTY_DIR}"
echo "error_dir       : ${ERROR_DIR}"
echo "mask_dir        : ${MASK_DIR}"
echo "outdir          : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${UNCERTAINTY_DIR}" ] || [ -z "$(ls -A "${UNCERTAINTY_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Uncertainty npy dir missing or empty: ${UNCERTAINTY_DIR}"
    echo "        Run compute_ensemble_uncertainty.sh first."
    exit 1
fi

if [ ! -d "${ERROR_DIR}" ] || [ -z "$(ls -A "${ERROR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Regen-error npy dir missing or empty: ${ERROR_DIR}"
    echo "        Run compute_ensemble_regen_error.sh first."
    exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
    echo "[ERROR] Mask directory not found: ${MASK_DIR}"
    exit 1
fi

# Skip guard
if [ -f "${OUTDIR}/summary.json" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}/summary.json"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/uncertainty_calibration.py \
    --uncertainty_dir     "${UNCERTAINTY_DIR}" \
    --error_dirs          "${ERROR_DIR}" \
    --mask_dir            "${MASK_DIR}" \
    --tiles_metadata      "${TEST_A}" \
    --outdir              "${OUTDIR}" \
    --min_tissue_pixels   256

echo "Done. Calibration for ${MODEL} WSI ${WSI_FOLDER} saved to ${OUTDIR}"
