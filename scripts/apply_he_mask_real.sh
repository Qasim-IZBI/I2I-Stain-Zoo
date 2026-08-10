#!/bin/bash
#SBATCH --job-name=i2i_apply_he_mask_real
#SBATCH --output=logs_apply_he/apply_he_real_%j.out
#SBATCH --error=logs_apply_he/apply_he_real_%j.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
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
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

# Reconstructed WSI-level HE tissue masks (shared across all conditions)
HE_MASKS_DIR=/work2/bz66izin-VSproject/HE_tissue

SEG_BASE=/work2/bz66izin-VSproject/psr_masks

# Reconstructed WSI-level real PSR masks produced by recon_masks_real.sh
PSR_DIR=${SEG_BASE}/real/psr_masks_wsi

# Output: cleaned masks with background signal zeroed out
OUT_DIR=${SEG_BASE}/real/psr_masks_wsi_cleaned

WSI_COUNT=5   # number of WSI TIFs expected (WSI range 1–5)

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${PSR_DIR}" ] || [ -z "$(ls -A "${PSR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real PSR masks not found or empty: ${PSR_DIR}"
    echo "        Run recon_masks_real.sh first."
    exit 1
fi

if [ ! -d "${HE_MASKS_DIR}" ] || [ -z "$(ls -A "${HE_MASKS_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] HE tissue masks not found or empty: ${HE_MASKS_DIR}"
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} cleaned masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    [ "${N_DONE}" -gt 0 ] && echo "[WARN] Partial output detected (${N_DONE}/${WSI_COUNT} masks). Re-running."
fi

mkdir -p "${OUT_DIR}"

echo "PSR masks : ${PSR_DIR}"
echo "HE masks  : ${HE_MASKS_DIR}"
echo "Output dir: ${OUT_DIR}"

run_cmd python "${PROJECT_ROOT}/apply_he_mask.py" \
    --psr_masks "${PSR_DIR}" \
    --he_masks  "${HE_MASKS_DIR}" \
    --outdir    "${OUT_DIR}"

echo "Done. Cleaned real PSR masks saved to ${OUT_DIR}"
