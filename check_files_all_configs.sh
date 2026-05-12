#!/bin/bash
#SBATCH --job-name=i2i_check_files
#SBATCH --output=logs_check_files/check_%A_%a.out
#SBATCH --error=logs_check_files/check_%A_%a.err

#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_check_files

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
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

SEG_BASE=/work2/bz66izin-VSproject/psr_masks

# Reference: post-processed real SR masks
REAL_DIR=${SEG_BASE}/real/psr_masks_wsi_final

# Generated masks for this config
GEN_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi_final

# -----------------------------
# Pre-flight checks
# -----------------------------
if [ ! -d "${REAL_DIR}" ] || [ -z "$(ls -A "${REAL_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real PSR masks not found or empty: ${REAL_DIR}"
    exit 1
fi

if [ ! -d "${GEN_DIR}" ] || [ -z "$(ls -A "${GEN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Generated PSR masks not found or empty: ${GEN_DIR}"
    exit 1
fi

echo "Reference : ${REAL_DIR}"
echo "Generated : ${GEN_DIR}"

# check_files.py exits 1 if any files are missing — propagated via set -e
run_cmd python "${PROJECT_ROOT}/check_files.py" \
    --dirA "${REAL_DIR}" \
    --dirB "${GEN_DIR}" \
    --ext  .tif
