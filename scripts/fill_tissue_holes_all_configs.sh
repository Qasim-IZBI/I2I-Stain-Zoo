#!/bin/bash
#SBATCH --job-name=i2i_fill_holes
#SBATCH --output=logs_fill_holes/fill_holes_%A_%a.out
#SBATCH --error=logs_fill_holes/fill_holes_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u   # conda activate scripts may reference unset variables
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_fill_holes

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches apply_he_mask_all_configs.sh
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

# Input: HE-masked PSR masks produced by apply_he_mask_all_configs.sh
IN_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi_cleaned

# Output: hole-filled masks (ready for compare_psr.py / cross_stain_consistency.py)
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi_final

WSI_COUNT=5   # number of WSI TIFs expected (WSI range 1–5)

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Cleaned masks must exist
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Cleaned PSR masks not found or empty: ${IN_DIR}"
    echo "        Run apply_he_mask_all_configs.sh first."
    exit 1
fi

# 2. Skip if expected number of filled masks already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} filled masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial output detected (${N_DONE}/${WSI_COUNT} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Input dir : ${IN_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Fill enclosed background holes in the tissue+PSR footprint.
# Newly filled pixels are assigned label 1 (tissue); labels 1 and 2 are unchanged.
# -----------------------------
run_cmd python "${PROJECT_ROOT}/fill_tissue_holes.py" \
    --masks  "${IN_DIR}" \
    --outdir "${OUT_DIR}"

echo "Done. Hole-filled PSR masks saved to ${OUT_DIR}"
