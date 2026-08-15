#!/bin/bash
#SBATCH --job-name=i2i_fillholes_real_sr
#SBATCH --output=logs_real_sr/fillholes_real_sr_%j.out
#SBATCH --error=logs_real_sr/fillholes_real_sr_%j.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Fill enclosed background inside the tissue footprint of the real SR masks.
# Labels 1 and 2 are one foreground: filling only label 1 would mark every
# PSR-positive pixel as a hole and relabel it.
#
# The real arm must get exactly the same treatment as the virtual one — this
# step moves CPA (it grows the tissue denominator), so skipping it on one arm
# and not the other produces a difference that looks like model bias.
#
# Input  : ${IN_DIR}    apply_he_mask_real_sr.sh
# Output : ${OUT_DIR}   consumed as the real reference by compare_psr and as
#                       --real_psr by estimate_floor.py

# -eo, not -euo: the Anaconda module runs activate.d hooks that read
# unset variables (qt-main_activate.sh), so -u there kills the job before
# the first echo. -u is switched on below, once conda is done.
set -eo pipefail

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

mkdir -p logs_real_sr

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — overridable via --export
# -----------------------------
IN_DIR="${IN_DIR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_cleaned}"
OUT_DIR="${OUT_DIR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_final}"
WSI_COUNT="${WSI_COUNT:-20}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Cleaned real SR masks not found or empty: ${IN_DIR}"
    echo "        Run apply_he_mask_real_sr.sh first."
    exit 1
fi

N_IN=$(find "${IN_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
if [ "${N_IN}" -lt "${WSI_COUNT}" ]; then
    echo "[ERROR] Only ${N_IN}/${WSI_COUNT} cleaned masks in ${IN_DIR}."
    echo "        A short real reference silently drops slides from the paired"
    echo "        comparison. Finish the previous step, or lower WSI_COUNT."
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} filled masks already present in ${OUT_DIR}."
        exit 0
    fi
    [ "${N_DONE}" -gt 0 ] && echo "[WARN] Partial output (${N_DONE}/${WSI_COUNT}). Re-running."
fi

mkdir -p "${OUT_DIR}"

echo "Input  : ${IN_DIR} (${N_IN} slides)"
echo "Output : ${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/fill_tissue_holes.py" \
    --masks  "${IN_DIR}" \
    --outdir "${OUT_DIR}"

echo "Done. Final real SR masks saved to ${OUT_DIR}"
