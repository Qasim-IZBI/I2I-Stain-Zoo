#!/bin/bash
#SBATCH --job-name=i2i_fill_holes_real
#SBATCH --output=logs_fill_holes/fill_holes_real_%j.out
#SBATCH --error=logs_fill_holes/fill_holes_real_%j.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

mkdir -p logs_fill_holes

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

SEG_BASE=/work2/bz66izin-VSproject/psr_masks

# Input: HE-masked real PSR masks produced by apply_he_mask_real.sh
IN_DIR=${SEG_BASE}/real/psr_masks_wsi_cleaned

# Output: hole-filled masks (ready for compare_psr.py as the real reference)
OUT_DIR=${SEG_BASE}/real/psr_masks_wsi_final

WSI_COUNT=5   # number of WSI TIFs expected (WSI range 1–5)

# -----------------------------
# Pre-flight checks
# -----------------------------

if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Cleaned real PSR masks not found or empty: ${IN_DIR}"
    echo "        Run apply_he_mask_real.sh first."
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} filled masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    [ "${N_DONE}" -gt 0 ] && echo "[WARN] Partial output detected (${N_DONE}/${WSI_COUNT} masks). Re-running."
fi

mkdir -p "${OUT_DIR}"

echo "Input dir : ${IN_DIR}"
echo "Output dir: ${OUT_DIR}"

run_cmd python "${PROJECT_ROOT}/fill_tissue_holes.py" \
    --masks  "${IN_DIR}" \
    --outdir "${OUT_DIR}"

echo "Done. Hole-filled real PSR masks saved to ${OUT_DIR}"
