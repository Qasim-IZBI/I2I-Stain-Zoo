#!/bin/bash
#SBATCH --job-name=i2i_hemask_real_sr
#SBATCH --output=logs_real_sr/hemask_real_sr_%j.out
#SBATCH --error=logs_real_sr/hemask_real_sr_%j.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Zero out real-SR collagen predictions outside the H&E tissue footprint.
#
# The point of applying the SAME footprint to both arms is that CPA's
# denominator is tissue area: measure the real slide on its own footprint and
# the virtual one on the H&E footprint and the two fractions are not comparable,
# whatever the collagen does.
#
# But note the asymmetry, because it does not cancel. On the virtual arm the
# mask is generated FROM the H&E, so the footprint is exact by construction. On
# this arm the SR is a serial section registered only at THUMBNAIL level, so the
# footprint is approximate — apply_he_mask resizes it nearest-neighbour to the
# SR dimensions, which corrects the scale but not a translation or rotation
# offset. Slide edges and detached tissue fragments are where that shows up.
# Eyeball a few outputs against the SR before this feeds a bias number.
#
# Input  : ${PSR_DIR}   segment_psr_real.sh
# Output : ${OUT_DIR}   -> fill_tissue_holes_real_sr.sh

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

mkdir -p logs_real_sr

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — overridable via --export
# -----------------------------
PSR_DIR="${PSR_DIR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi}"
HE_MASKS_DIR="${HE_MASKS_DIR:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_tissue/testA}"
OUT_DIR="${OUT_DIR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_cleaned}"
WSI_COUNT="${WSI_COUNT:-20}"

# The SR masks are named after the SR slides and the tissue masks after the H&E
# ones, so matching needs the first '_'-delimited token dropped from both sides
# (SR_slide <-> HE_slide). Set STRIP_PREFIX=0 if both sets already share stems.
STRIP_PREFIX="${STRIP_PREFIX:-1}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${PSR_DIR}" ] || [ -z "$(ls -A "${PSR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real PSR masks not found or empty: ${PSR_DIR}"
    echo "        Run segment_psr_real.sh first."
    exit 1
fi

if [ ! -d "${HE_MASKS_DIR}" ] || [ -z "$(ls -A "${HE_MASKS_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] HE tissue masks not found or empty: ${HE_MASKS_DIR}"
    exit 1
fi

N_IN=$(find "${PSR_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
if [ "${N_IN}" -lt "${WSI_COUNT}" ]; then
    echo "[ERROR] Only ${N_IN}/${WSI_COUNT} real PSR masks in ${PSR_DIR}."
    echo "        Finish segment_psr_real.sh before cleaning, or lower WSI_COUNT."
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} cleaned masks already present in ${OUT_DIR}."
        exit 0
    fi
    [ "${N_DONE}" -gt 0 ] && echo "[WARN] Partial output (${N_DONE}/${WSI_COUNT}). Re-running."
fi

mkdir -p "${OUT_DIR}"

STRIP_ARGS=()
[ "${STRIP_PREFIX}" = "1" ] && STRIP_ARGS=(--strip_prefix)

echo "PSR masks : ${PSR_DIR} (${N_IN} slides)"
echo "HE masks  : ${HE_MASKS_DIR}"
echo "Output    : ${OUT_DIR}"
echo "Strip prefix: ${STRIP_PREFIX}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/apply_he_mask.py" \
    --psr_masks "${PSR_DIR}" \
    --he_masks  "${HE_MASKS_DIR}" \
    --outdir    "${OUT_DIR}" \
    "${STRIP_ARGS[@]}"

echo "Done. Cleaned real SR masks saved to ${OUT_DIR}"
