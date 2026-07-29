#!/bin/bash
#SBATCH --job-name=i2i_recon_masks_all
#SBATCH --output=logs_recon/recon_masks_%A_%a.out
#SBATCH --error=logs_recon/recon_masks_%A_%a.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_recon

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches segment_psr_all_configs.sh
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
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/
PROJECT_ROOT=I2I-Stain-Zoo

# Original testA tiles — provides tiles_metadata.csv with x/y coordinates
TEST_A="${DATA_DIR}testA/tiles/testA"

# WSI range (must match DATA_RANGE in segment_psr_all_configs.sh)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# Per-tile PSR masks produced by segment_psr_all_configs.sh
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
TILE_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/tile_masks

# Reconstructed WSI-level mask output (flat directory consumed by compare_psr.py)
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Tile masks must exist
if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Tile masks not found or empty: ${TILE_DIR} — skipping."
    exit 1
fi

# 2. Skip if expected number of reconstructed WSI masks already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} reconstructed WSI masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial reconstruction detected (${N_DONE}/${N_EXPECTED} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Tile dir  : ${TILE_DIR}"
echo "Metadata  : ${TEST_A}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Reconstruct tile masks → WSI-level mask TIFs
# Mask tiles are stored in {NNN}/images/ (nnUNet label maps, values 0/1/2).
# --mode rgb treats them as image tiles; compare_psr.py reads mask[...,0]
# so 3-channel output is handled correctly.
# -----------------------------
run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${TEST_A}" \
    --tile_dir "${TILE_DIR}" \
    --output   "${OUT_DIR}" \
    --mode     rgb \
    --blend    average

echo "Done. Reconstructed WSI masks saved to ${OUT_DIR}"
