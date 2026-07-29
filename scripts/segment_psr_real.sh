#!/bin/bash
#SBATCH --job-name=i2i_seg_psr_real
#SBATCH --output=logs_seg_psr/seg_real_%j.out
#SBATCH --error=logs_seg_psr/seg_real_%j.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=paula
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
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
# Fixed paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/
PROJECT_ROOT=I2I-Stain-Zoo

# Real testB tiles (NNN/images/ structure)
TEST_B="${DATA_DIR}testB/tiles/testB"

# Per-tile PSR mask output (NNN/images/ structure, ready for reconstruct.py)
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
OUT_DIR=${SEG_BASE}/real/tile_masks

# nnUNet model settings
NNUNET_RESULTS=/work2/bz66izin-VSproject/nnunet/results
NNUNET_DATASET=214
NNUNET_CONFIG=2d
NNUNET_FOLDS="1 2 3 4"

RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))
DATA_RANGE="${RANGE_START},${RANGE_END}"

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${TEST_B}" ] || [ -z "$(ls -A "${TEST_B}" 2>/dev/null)" ]; then
    echo "[ERROR] Real testB tiles not found or empty: ${TEST_B}"
    exit 1
fi

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

echo "Tile dir  : ${TEST_B}"
echo "Output dir: ${OUT_DIR}"
echo "WSI range : ${RANGE_START}–${RANGE_END}"

# -----------------------------
# Segment tiles with white border padding (tile mode).
# Each tile is padded with 256px of white before nnUNet so the model sees
# tissue against background — required for correct tissue class predictions.
# The prediction is cropped back to the original tile size automatically.
# Output: {OUT_DIR}/{NNN}/images/{tile}.tif
# Next step: recon_masks_real.sh to stitch masks into WSI TIFs
# -----------------------------
run_cmd python "${PROJECT_ROOT}/segment_psr.py" \
    --data             "${TEST_B}" \
    --tile_mode \
    --data_range       "${DATA_RANGE}" \
    --pad_border       256 \
    --outdir           "${OUT_DIR}" \
    --nnunet_results   "${NNUNET_RESULTS}" \
    --nnunet_dataset   "${NNUNET_DATASET}" \
    --nnunet_config    "${NNUNET_CONFIG}" \
    --nnunet_folds     "${NNUNET_FOLDS}" \
    --device           cuda

echo "Done. Real PSR tile masks saved to ${OUT_DIR}"
