#!/bin/bash
#SBATCH --job-name=i2i_recon_real_psr
#SBATCH --output=logs_recon/recon_real_%j.out
#SBATCH --error=logs_recon/recon_real_%j.err

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

# Real PSR (testB) tiles — original tiles, no --tile_dir override needed
TEST_B=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testB/tiles/testB

# Output
RECON_BASE=/work2/bz66izin-VSproject/reconstruction
OUT_DIR=${RECON_BASE}/real/reconstructed

RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${TEST_B}" ] || [ -z "$(ls -A "${TEST_B}" 2>/dev/null)" ]; then
    echo "[ERROR] testB tiles directory not found or empty: ${TEST_B}"
    exit 1
fi

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

echo "Metadata  : ${TEST_B}"
echo "Output dir: ${OUT_DIR}"
echo "WSI range : ${RANGE_START}–${RANGE_END} (expecting ${N_EXPECTED} WSIs)"

# -----------------------------
# Reconstruct
# -----------------------------
run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${TEST_B}" \
    --output   "${OUT_DIR}" \
    --mode     rgb \
    --blend    average

echo "Done. Reconstructed real PSR WSIs saved to ${OUT_DIR}"
