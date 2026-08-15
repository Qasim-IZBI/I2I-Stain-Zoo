#!/bin/bash
#SBATCH --job-name=i2i_white_thresh
#SBATCH --output=logs_real_sr/white_thresh_%j.out
#SBATCH --error=logs_real_sr/white_thresh_%j.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Pick --white_thresh from the H&E rather than guessing it.
#
# Not an array job: it sweeps a handful of slides and pools them, and the answer
# is one number for the cohort. Cost is one whole-slide binary_fill_holes per
# threshold per slide — 17 thresholds x 3 slides at ~1.4 Gpixel each — so N_WSIS
# and the threshold step are the knobs if it runs long.
#
# MEMORY: the channel-minimum array is 1 byte/pixel (1.4 GB on the largest
# slide) and each threshold allocates two bools plus the fill output on top.
# 96G leaves room; it never holds the RGB, which is memmapped in row blocks.
#
# Run this for BOTH stains. The H&E value goes to compute_phi_uncertainty
# (--white_thresh / WHITE_THRESH) and estimate_floor (--white_thresh); the SR
# value goes to estimate_floor --white_thresh_psr. The two stains sit at
# different whitespace levels, and if either threshold lands on a slope rather
# than in the valley, the cross-stain floor measures your threshold choice
# instead of the biological level offset.
#
#   sbatch --export=ALL,\
#   HE_DIR=/work2/bz66izin-UC_project/ID_SR/no_overlap/testB/export_rgb/testB,\
#   TILES_METADATA=none,\
#   OUTDIR=/work2/bz66izin-UC_project/white_thresh_sr \
#       I2I-Stain-Zoo/scripts/calibrate_white_thresh.sh

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_real_sr

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — overridable via --export. Defaults are the liver H&E arm.
# -----------------------------
HE_DIR="${HE_DIR:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_rgb/testA}"
# Optional. Set TILES_METADATA=none on the real SR arm, which is evaluated
# whole-slide and has no tiling — the grid is then sized from each image.
# Note the `-` rather than `:-`: with `:-` an explicitly EMPTY value would fall
# back to this default, so "set it to empty" could not turn the flag off.
TILES_METADATA="${TILES_METADATA-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/white_thresh_he}"

N_WSIS="${N_WSIS:-3}"
T_MIN="${T_MIN:-0.50}"
T_MAX="${T_MAX:-0.90}"
T_STEP="${T_STEP:-0.025}"
# Marked on the figure as "where you are now" — set it to whatever the last phi
# run used, so the plot answers whether that choice was on the plateau.
CURRENT="${CURRENT:-0.70}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${HE_DIR}" ]; then
    echo "[ERROR] H&E directory not found: ${HE_DIR}"
    exit 1
fi

META_ARGS=()
if [ -n "${TILES_METADATA}" ] && [ "${TILES_METADATA}" != "none" ]; then
    if [ ! -d "${TILES_METADATA}" ]; then
        echo "[ERROR] tiles_metadata root not found: ${TILES_METADATA}"
        echo "        Its source_file entries must name the images in ${HE_DIR}."
        echo "        Set TILES_METADATA=none to size the grid from the"
        echo "        images instead — the real SR arm has no tiling."
        exit 1
    fi
    META_ARGS=(--tiles_metadata "${TILES_METADATA}")
else
    echo "[INFO] No tiles_metadata: region grid sized from each image."
fi

N_IMG=$(find "${HE_DIR}" -maxdepth 1 -type f \( -name '*.tif' -o -name '*.tiff' \) | wc -l)
if [ "${N_IMG}" -eq 0 ]; then
    echo "[ERROR] No TIFs in ${HE_DIR}"
    exit 1
fi

if [ -s "${OUTDIR}/white_thresh.png" ] && [ -s "${OUTDIR}/white_thresh.csv" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}"
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "H&E dir  : ${HE_DIR} (${N_IMG} slides, sweeping ${N_WSIS})"
if [ -n "${META_ARGS[*]}" ]; then
    echo "Metadata : ${TILES_METADATA}"
else
    echo "Metadata : <none — grid sized from the images themselves>"
fi
echo "Output   : ${OUTDIR}"
echo "Sweep    : ${T_MIN} to ${T_MAX} step ${T_STEP}, current marked at ${CURRENT}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/calibrate_white_thresh.py" \
    --he_dir         "${HE_DIR}" \
    "${META_ARGS[@]}" \
    --outdir         "${OUTDIR}" \
    --n_wsis         "${N_WSIS}" \
    --t_min          "${T_MIN}" \
    --t_max          "${T_MAX}" \
    --t_step         "${T_STEP}" \
    --current        "${CURRENT}"

echo "Done. Read ${OUTDIR}/white_thresh.png — panel C is the decisive one."
