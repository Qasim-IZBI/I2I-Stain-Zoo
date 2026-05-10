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

# Inference tiles produced by infer_small_models.sh / infer_all_54.sh
INFER_BASE=/work2/bz66izin-VSproject/inference
TILE_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference

# Per-tile PSR mask output (NNN/images/ structure, ready for reconstruct.py)
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/tile_masks

# nnUNet model settings
NNUNET_RESULTS=/work2/bz66izin-VSproject/nnunet/results
NNUNET_DATASET=214
NNUNET_CONFIG=2d
NNUNET_FOLDS="1 2 3 4"

# WSI range used during inference (must match DATA_RANGE in the inference script)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))
DATA_RANGE="${RANGE_START},${RANGE_END}"

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Inference tiles must exist
if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Inference tiles not found or empty: ${TILE_DIR} — skipping."
    exit 1
fi

# 2. Skip if all expected WSI tile-mask folders are already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} tile-mask folders already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial segmentation detected (${N_DONE}/${N_EXPECTED} folders). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Tile dir  : ${TILE_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "WSI range : ${RANGE_START}–${RANGE_END}"

# -----------------------------
# Segment tiles with white border padding (tile mode).
# Each tile is padded with 256px of white before nnUNet so the model sees
# tissue against background — required for correct tissue class predictions.
# The prediction is cropped back to the original tile size automatically.
# Output: {OUT_DIR}/{NNN}/images/{tile}.tif
# Next step: reconstruct.py --tile_dir to stitch masks into WSI TIFs
# -----------------------------
run_cmd python "${PROJECT_ROOT}/segment_psr.py" \
    --data             "${TILE_DIR}" \
    --tile_mode \
    --data_range       "${DATA_RANGE}" \
    --pad_border       256 \
    --outdir           "${OUT_DIR}" \
    --nnunet_results   "${NNUNET_RESULTS}" \
    --nnunet_dataset   "${NNUNET_DATASET}" \
    --nnunet_config    "${NNUNET_CONFIG}" \
    --nnunet_folds     "${NNUNET_FOLDS}" \
    --device           cuda

echo "Done. Per-tile PSR masks saved to ${OUT_DIR}"
