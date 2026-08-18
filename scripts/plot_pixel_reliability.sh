#!/bin/bash
#SBATCH --job-name=i2i_px_reliability
#SBATCH --output=logs_ensemble_ugac/px_reliability_%j.out
#SBATCH --error=logs_ensemble_ugac/px_reliability_%j.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# One reliability figure over EVERY slide: total, procedural and data-exposure
# sigma against the same cycle error.
#
# Consumes what calibrate_pixel_components.sh wrote per WSI:
#
#   {OUTROOT}/wsi{NNN}/{total,procedural,data_exposure}/raw_npy/...
#   {OUTROOT}/wsi{NNN}/mean_rgb/...            (enables the mu partial)
#
# ONE JOB, ALL SLIDES — this is the whole point of it being separate from the
# decomposition array. The two statistics that make the figure defensible are
# both between-slide: the bootstrap resamples SLIDES, and within_slide computes
# rho inside each and summarises over them. Both need at least three, so run
# per WSI they return NaN and nothing respectively. Twenty one-slide figures
# would each show a rho with no interval and no confound control.
#
# The decomposition stays an array because it is memory-bound per WSI (fifty
# members live at once); the figure is cheap and must see everything.
#
# > RETIRED CHAIN. Kept for provenance beside the rest of scripts/*_ugac.sh.
#
#   sbatch I2I-Stain-Zoo/scripts/plot_pixel_reliability.sh

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo.
set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p logs_ensemble_ugac

PROJECT_ROOT=I2I-Stain-Zoo

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
OUTROOT="${OUTROOT:-${UGAC_ROOT}/pixel_components}"
OUTDIR="${OUTDIR:-${OUTROOT}/reliability}"
N_WSIS="${N_WSIS:-20}"
REGEN_BLOCKS="${REGEN_BLOCKS:-}"
MIN_TISSUE_PIXELS="${MIN_TISSUE_PIXELS:-256}"
N_BINS="${N_BINS:-10}"
N_BOOT="${N_BOOT:-2000}"

if [ -z "${REGEN_BLOCKS}" ]; then
    for i in "${!RANGE_STARTS[@]}"; do
        REGEN_BLOCKS="${REGEN_BLOCKS} $(printf 'data_%03d_%03d' \
            "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")"
    done
fi

echo "Components : ${OUTROOT}"
echo "Output     : ${OUTDIR}"
echo "Blocks     : ${REGEN_BLOCKS}"

# -----------------------------
# Build one positional triple per WSI
# -----------------------------
ARGS=()
MISSING=()
FOUND=0
for w in $(seq 1 "${N_WSIS}"); do
    WF=$(printf "%03d" "${w}")
    COMP="${OUTROOT}/wsi${WF}"
    if [ ! -d "${COMP}/total/raw_npy" ]; then
        MISSING+=("${WF}")
        continue
    fi
    # every block's error for this WSI, comma-separated: the script averages
    # them, matching an all-50 sigma with an all-50 error
    ERRS=""
    SHORT=0
    for B in ${REGEN_BLOCKS}; do
        E="${UGAC_ROOT}/${B}/${MODEL_SIZE}/regen_error/wsi${WF}/error_npy"
        [ -d "${E}" ] || { SHORT=1; break; }
        ERRS="${ERRS}${ERRS:+,}${E}"
    done
    if [ "${SHORT}" -ne 0 ]; then
        MISSING+=("${WF}(regen)")
        continue
    fi
    ARGS+=(--components "${COMP}" --error_dirs "${ERRS}"
           --mask_dir "${TEST_A}/${WF}/masks")
    FOUND=$(( FOUND + 1 ))
done

echo "Slides     : ${FOUND}/${N_WSIS}"

if [ "${FOUND}" -lt 3 ]; then
    echo "[ERROR] Only ${FOUND} slide(s) available; at least three are needed."
    echo "        The bootstrap resamples slides and the partial is computed"
    echo "        within each, so below three both return nothing and the figure"
    echo "        would carry a rho with no interval and no confound control."
    echo "        Run calibrate_pixel_components.sh to completion."
    exit 1
fi

# Short is not fatal — the interval simply covers fewer cases — but a figure
# that silently rests on twelve slides while the text says twenty is worse than
# a warning nobody reads, so it is named.
if [ "${#MISSING[@]}" -ne 0 ]; then
    echo "[WARN] ${#MISSING[@]} slide(s) missing and excluded: ${MISSING[*]}"
    echo "       The clustered interval will cover ${FOUND} cases, not ${N_WSIS}."
fi

if [ -f "${OUTDIR}/summary.json" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}/summary.json"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() { echo "Running command:"; printf ' %q' "$@"; echo; "$@"; }

run_cmd python "${PROJECT_ROOT}/plot_pixel_reliability.py" \
    "${ARGS[@]}" \
    --tiles_metadata    "${TEST_A}" \
    --min_tissue_pixels "${MIN_TISSUE_PIXELS}" \
    --n_bins            "${N_BINS}" \
    --n_boot            "${N_BOOT}" \
    --outdir            "${OUTDIR}"

echo
echo "Done. ${OUTDIR}/reliability_pixel.png over ${FOUND} slide(s)."
echo "Read the partial column in within_slide.csv, not the raw rho: sigma"
echo "largely tracks how much structure a tile holds, and so does the error."
