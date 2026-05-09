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

# Reconstructed real PSR WSIs (output of recon_real_psr.sh)
RECON_BASE=/work2/bz66izin-VSproject/reconstruction
RECON_DIR=${RECON_BASE}/real/reconstructed

# PSR segmentation output
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
OUT_DIR=${SEG_BASE}/real/psr_masks

# nnUNet model settings
NNUNET_RESULTS=/work2/bz66izin-VSproject/nnunet/results
NNUNET_DATASET=1
NNUNET_CONFIG=2d
NNUNET_FOLDS="1 2 3 4"

RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${RECON_DIR}" ] || [ -z "$(ls -A "${RECON_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Reconstructed WSIs not found or empty: ${RECON_DIR}"
    echo "        Run recon_real_psr.sh first."
    exit 1
fi

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
echo "WSI range : ${RANGE_START}–${RANGE_END} (expecting ${N_EXPECTED} WSIs)"

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

echo "Done. Real PSR masks saved to ${OUT_DIR}"
