#!/bin/bash
#SBATCH --job-name=i2i_floor
#SBATCH --output=logs_real_sr/floor_%j.out
#SBATCH --error=logs_real_sr/floor_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# The section 7 go/no-go: per-descriptor biological floor.
#
# Run this BEFORE building anything on the bias term. If the observed
# virtual-vs-real discrepancy lands near the floor there is no headroom and
# bias^2 = observed^2 - d comes out at or below zero — a genuine stop condition,
# and cheaper to discover here than after a registration effort.
#
# Not an array job: the variogram bins region pairs by separation across the
# whole cohort, so the slides cannot be split over tasks.
#
# COST: one full-slide mask per WSI (1.4 Gpixel uint8), then betti() and a
# structure tensor over every ~6787^2 region. Hours, not minutes, and the peak
# is a handful of full-size intermediates per region on top of the slide.
#
# Reads : ${REAL_PSR}   fill_tissue_holes_real_sr.sh
# Writes: ${OUTDIR}/floor_per_descriptor.csv, floor.json

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables (qt-main_activate.sh), so -u there kills the job before the first
# echo. -u is switched on below, once conda is done.
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
# Paths — overridable via --export
# -----------------------------
REAL_PSR="${REAL_PSR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_final}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/floor_liver}"

# Optional. The real SR is evaluated whole-slide and has no tiling, so `none`
# here sizes the region grid from each mask instead — which is the liver arm.
TILES_METADATA="${TILES_METADATA-none}"

# These four MUST match the compute_phi_uncertainty run this floor will be
# compared against. A floor measured with different region sizing or different
# mask cleaning does not bound that discrepancy, it bounds a different one.
REGION_MM="${REGION_MM:-1.5}"
MIN_TISSUE_FRACTION="${MIN_TISSUE_FRACTION:-0.25}"
MIN_OBJECT_PX="${MIN_OBJECT_PX:-16}"
CLOSING_PX="${CLOSING_PX:-0}"

# Microns per pixel OF THE MASKS. These come from the ORIGINAL SR WSIs, not from
# reconstructions, so confirm the export magnification: at the wrong mpp every
# region is the wrong physical size and the per-mm^2 densities go with it.
MPP="${MPP:-0.221}"

# Cross-stain arm. Off by default: it bounds only lumen_fraction and
# tissue_fraction, and on this cohort the SR cannot measure either — its
# footprint is unstable across the whole sweep, so the bound would describe the
# thresholds rather than the level offset. The variogram covers all six
# descriptors and outranks cross-stain in precedence anyway, so nothing is lost.
# Set BOTH to enable it; --real_he alone produces no bound.
REAL_HE="${REAL_HE-}"
REAL_PSR_RGB="${REAL_PSR_RGB-}"
WHITE_THRESH="${WHITE_THRESH:-0.65}"
WHITE_THRESH_PSR="${WHITE_THRESH_PSR-}"

# A SECOND real PSR level, if it ever exists. It supersedes the whole bracket:
# a measured cross-level floor rather than a bounded one.
PSR_LEVEL_B="${PSR_LEVEL_B-}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${REAL_PSR}" ] || [ -z "$(ls -A "${REAL_PSR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real PSR masks not found or empty: ${REAL_PSR}"
    echo "        Run fill_tissue_holes_real_sr.sh first."
    exit 1
fi

N_MASKS=$(find "${REAL_PSR}" -maxdepth 1 -name "*.tif" | wc -l)
echo "Real PSR : ${REAL_PSR} (${N_MASKS} masks)"

ARGS=(--real_psr "${REAL_PSR}"
      --outdir "${OUTDIR}"
      --region_mm "${REGION_MM}"
      --mpp "${MPP}"
      --min_tissue_fraction "${MIN_TISSUE_FRACTION}"
      --min_object_px "${MIN_OBJECT_PX}"
      --closing_px "${CLOSING_PX}")

if [ -n "${TILES_METADATA}" ] && [ "${TILES_METADATA}" != "none" ]; then
    if [ ! -d "${TILES_METADATA}" ]; then
        echo "[ERROR] tiles_metadata root not found: ${TILES_METADATA}"
        echo "        Set TILES_METADATA=none to size the grid from each mask."
        exit 1
    fi
    ARGS+=(--tiles_metadata "${TILES_METADATA}")
    echo "Grid     : from ${TILES_METADATA}"
else
    echo "Grid     : sized from each mask (no tiling on this arm)"
fi

if [ -n "${REAL_HE}" ] && [ -n "${REAL_PSR_RGB}" ]; then
    # Both sides need their own image. The stems must match the mask stems, so
    # on a cohort whose PSR masks are named after the SR slides while the H&E
    # files are named after the H&E ones, this silently finds nothing.
    ARGS+=(--real_he "${REAL_HE}" --real_psr_rgb "${REAL_PSR_RGB}"
           --white_thresh "${WHITE_THRESH}")
    [ -n "${WHITE_THRESH_PSR}" ] && ARGS+=(--white_thresh_psr "${WHITE_THRESH_PSR}")
    echo "Cross-stain: H&E ${REAL_HE} vs PSR ${REAL_PSR_RGB}"
elif [ -n "${REAL_HE}" ] || [ -n "${REAL_PSR_RGB}" ]; then
    echo "[ERROR] The cross-stain bound needs BOTH REAL_HE and REAL_PSR_RGB."
    echo "        One image used twice gives a zero floor, which inflates bias."
    exit 1
else
    echo "Cross-stain: off — lumen_fraction and tissue_fraction will read NaN,"
    echo "             which is by design: they are level-A and pay no floor."
fi

[ -n "${PSR_LEVEL_B}" ] && ARGS+=(--psr_level_b "${PSR_LEVEL_B}")

if [ -s "${OUTDIR}/floor_per_descriptor.csv" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}"
    exit 0
fi

mkdir -p "${OUTDIR}"
echo "Output   : ${OUTDIR}"
echo "Region   : ${REGION_MM} mm at ${MPP} um/px, min_tissue ${MIN_TISSUE_FRACTION}"
echo "Cleaning : min_object_px ${MIN_OBJECT_PX}, closing_px ${CLOSING_PX}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/estimate_floor.py" "${ARGS[@]}"

echo
echo "Done. Read ${OUTDIR}/floor_per_descriptor.csv — the decisive column is"
echo "floor_to_signal: usable <0.5, marginal 0.5-0.9, floor-limited >=0.9."
echo "Every row also carries floor_source and bound_direction, so a number"
echo "resting on the anti-conservative split-half lower bound is visible as such."
