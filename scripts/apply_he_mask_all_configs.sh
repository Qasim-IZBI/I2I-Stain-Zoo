#!/bin/bash
#SBATCH --job-name=i2i_apply_he_mask
#SBATCH --output=logs_apply_he/apply_he_%A_%a.out
#SBATCH --error=logs_apply_he/apply_he_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u   # conda activate scripts may reference unset variables
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_apply_he

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches recon_masks_all_configs.sh
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

# Reconstructed WSI-level HE tissue masks (non-zero = tissue, shared across all configs)
HE_MASKS_DIR=/work2/bz66izin-VSproject/HE_tissue

# Reconstructed WSI-level PSR masks produced by recon_masks_all_configs.sh
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
PSR_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/wsi_masks

# Output: cleaned masks with background signal zeroed out
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi_cleaned

WSI_COUNT=5   # number of WSI TIFs expected (WSI range 1–5)

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. PSR masks must exist
if [ ! -d "${PSR_DIR}" ] || [ -z "$(ls -A "${PSR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] PSR masks not found or empty: ${PSR_DIR}"
    echo "        Run recon_masks_all_configs.sh first."
    exit 1
fi

# 2. HE tissue masks must exist
if [ ! -d "${HE_MASKS_DIR}" ] || [ -z "$(ls -A "${HE_MASKS_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] HE tissue masks not found or empty: ${HE_MASKS_DIR}"
    exit 1
fi

# 3. Skip if expected number of cleaned masks already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} cleaned masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial output detected (${N_DONE}/${WSI_COUNT} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "PSR masks : ${PSR_DIR}"
echo "HE masks  : ${HE_MASKS_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Apply HE tissue mask — zeros out background-region predictions
# Labels inside tissue boundary are preserved (0=bg, 1=tissue, 2=PSR+)
# -----------------------------
run_cmd python "${PROJECT_ROOT}/apply_he_mask.py" \
    --psr_masks "${PSR_DIR}" \
    --he_masks  "${HE_MASKS_DIR}" \
    --outdir    "${OUT_DIR}"

echo "Done. Cleaned PSR masks saved to ${OUT_DIR}"
