#!/bin/bash
#SBATCH --job-name=i2i_recon_masks_real
#SBATCH --output=logs_recon/recon_masks_real_%j.out
#SBATCH --error=logs_recon/recon_masks_real_%j.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
# Fixed paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/
PROJECT_ROOT=I2I-Stain-Zoo

# Real testB tiles — provides tiles_metadata.csv with x/y coordinates
TEST_B="${DATA_DIR}testB/tiles/testB"

# WSI range (must match DATA_RANGE in segment_psr_real.sh)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# Per-tile PSR masks produced by segment_psr_real.sh
SEG_BASE=/work2/bz66izin-VSproject/psr_masks
TILE_DIR=${SEG_BASE}/real/tile_masks

# Reconstructed WSI-level mask output (flat directory consumed by compare_psr.py)
OUT_DIR=${SEG_BASE}/real/psr_masks_wsi

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Tile masks must exist
if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real tile masks not found or empty: ${TILE_DIR}"
    echo "        Run segment_psr_real.sh first."
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
echo "Metadata  : ${TEST_B}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Reconstruct tile masks → WSI-level mask TIFs
# -----------------------------
run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${TEST_B}" \
    --tile_dir "${TILE_DIR}" \
    --output   "${OUT_DIR}" \
    --mode     rgb \
    --blend    average

echo "Done. Reconstructed real WSI masks saved to ${OUT_DIR}"
