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
PROJECT_ROOT=I2I-Stain-Zoo

# Reconstructed real testB WSIs produced by recon_real_psr.sh
RECON_DIR=/work2/bz66izin-VSproject/reconstruction/real/reconstructed

# WSI-level PSR mask output (flat directory consumed by compare_psr.py)
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
OUT_DIR=${SEG_BASE}/real/psr_masks_wsi

# nnUNet model settings
NNUNET_RESULTS=/work2/bz66izin-VSproject/nnunet/results
NNUNET_DATASET=214
NNUNET_CONFIG=2d
NNUNET_FOLDS="1 2 3 4"

RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${RECON_DIR}" ] || [ -z "$(find "${RECON_DIR}" -maxdepth 1 -name "*.tif" 2>/dev/null | head -1)" ]; then
    echo "[ERROR] Reconstructed real WSIs not found in: ${RECON_DIR} — run recon_real_psr.sh first."
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} WSI masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial segmentation detected (${N_DONE}/${N_EXPECTED} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "WSI dir   : ${RECON_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Segment reconstructed WSIs directly (WSI mode — no --tile_mode).
# nnUNet model was trained on full WSIs and needs slide-level context to
# correctly classify tissue vs. background.
# Output: flat directory of {stem}.tif mask files consumed by compare_psr.py.
# -----------------------------
run_cmd python "${PROJECT_ROOT}/segment_psr.py" \
    --data             "${RECON_DIR}" \
    --outdir           "${OUT_DIR}" \
    --nnunet_results   "${NNUNET_RESULTS}" \
    --nnunet_dataset   "${NNUNET_DATASET}" \
    --nnunet_config    "${NNUNET_CONFIG}" \
    --nnunet_folds     "${NNUNET_FOLDS}" \
    --device           cuda

echo "Done. Real WSI-level PSR masks saved to ${OUT_DIR}"
