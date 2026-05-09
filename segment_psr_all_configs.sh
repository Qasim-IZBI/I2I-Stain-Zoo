#!/bin/bash
#SBATCH --job-name=i2i_seg_psr_all
#SBATCH --output=logs_seg_psr/seg_%A_%a.out
#SBATCH --error=logs_seg_psr/seg_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u   # conda activate scripts may reference unset variables (e.g. QT_XCB_GL_INTEGRATION)
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"

mkdir -p logs_seg_psr

# -----------------------------
# Helper: echo and run a command
# -----------------------------
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches recon_all_configs.sh
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
SIZES=("small" "medium" "large")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

SIZE_ID=$(( TASK_ID / 18 ))
DATA_ID=$(( (TASK_ID % 18) / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
SIZE=${SIZES[$SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

# Reconstructed WSIs produced by recon_all_configs.sh
RECON_BASE=/work2/bz66izin-VSproject/reconstruction
RECON_DIR=${RECON_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/reconstructed

# PSR segmentation output
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks

# nnUNet model settings
NNUNET_RESULTS=/work2/bz66izin-VSproject/nnunet/results
NNUNET_DATASET=1
NNUNET_CONFIG=2d
NNUNET_FOLDS="1 2 3 4"

# WSI range used during reconstruction (must match RANGE_END in recon_all_configs.sh)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Reconstructed WSIs must exist
if [ ! -d "${RECON_DIR}" ] || [ -z "$(ls -A "${RECON_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Reconstructed WSIs not found or empty: ${RECON_DIR} — skipping."
    exit 1
fi

# 2. Skip if expected number of mask TIFs already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" ! -name "*_mask.tif" | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} PSR masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial segmentation detected (${N_DONE}/${N_EXPECTED} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Recon dir : ${RECON_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Segment
# -----------------------------
run_cmd python "${PROJECT_ROOT}/segment_psr.py" \
    --data             "${RECON_DIR}" \
    --outdir           "${OUT_DIR}" \
    --nnunet_results   "${NNUNET_RESULTS}" \
    --nnunet_dataset   "${NNUNET_DATASET}" \
    --nnunet_config    "${NNUNET_CONFIG}" \
    --nnunet_folds     "${NNUNET_FOLDS}" \
    --device           cuda

echo "Done. PSR masks saved to ${OUT_DIR}"
