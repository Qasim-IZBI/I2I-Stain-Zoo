#!/bin/bash
#SBATCH --job-name=i2i_recon_all
#SBATCH --output=logs_recon/recon_%A_%a.out
#SBATCH --error=logs_recon/recon_%A_%a.err

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

# Let Python use all allocated CPUs for tile I/O
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_recon

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
# Decomposition matches eval_all_configs.sh
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

# Original testA tiles directory — provides tiles_metadata.csv files for each WSI
TEST_A="${DATA_DIR}testA/tiles/testA"

# WSI range used during inference (must match DATA_RANGE in the inference script)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# Inference output (flat directory of translated tiles)
INFER_BASE=/work2/bz66izin-VSproject/inference
TILE_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference

# Reconstruction output
RECON_BASE=/work2/bz66izin-VSproject/reconstruction
OUT_DIR=${RECON_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/reconstructed

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Inference output must exist and be non-empty
if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Inference output not found or empty: ${TILE_DIR} — skipping."
    exit 1
fi

# 2. Skip if expected number of reconstructed WSI TIFs already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} reconstructed WSIs already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial reconstruction detected (${N_DONE}/${N_EXPECTED} WSIs). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Tile dir  : ${TILE_DIR}"
echo "Metadata  : ${TEST_A}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Reconstruct
# -----------------------------
run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${TEST_A}" \
    --tile_dir "${TILE_DIR}" \
    --output   "${OUT_DIR}" \
    --mode     rgb \
    --blend    average

echo "Done. Reconstructed WSIs saved to ${OUT_DIR}"
