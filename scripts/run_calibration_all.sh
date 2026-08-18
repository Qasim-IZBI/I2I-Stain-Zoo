#!/bin/bash
#SBATCH --job-name=i2i_calibration
#SBATCH --output=logs_ensemble_ugac/calibration_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/calibration_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-99   # 100 jobs = 5 data blocks x 20 test WSIs

# Does the ensemble's spread predict its cycle-reconstruction error, per tile?
#
# The last step of the UGAC chain, pairing the two quantities the previous two
# branches produced:
#
#   uncertainty/cyclegan/raw_npy/{NNN}/images/   forward-pass spread   (A->B)
#   regen_error/wsi{NNN}/error_npy/              round-trip error      (A->B->A)
#   {TEST_A}/{NNN}/masks/                        tissue
#     -> calibration/cyclegan/wsi{NNN}/  per_tile.csv summary.json calibration.png
#
# Note what is being paired and what it is worth. Cycle error is a
# self-consistency PROXY, not ground truth: when the forward and inverse
# generators share a bias, both ignore the same feature and the round trip
# reconstructs the source despite a poor forward translation. The BMVC 2026
# result is that it does not calibrate. Measuring the ensemble against an
# external target instead is compare_uncertainty_sources.py, which reuses the
# same regen_error/ directories.
#
# Decomposition matches the rest of the chain:
#   tasks  0– 19  ->  folders 001–007   WSI 1–20
#   tasks 20– 39  ->  folders 008–014   WSI 1–20
#   tasks 40– 59  ->  folders 015–021   WSI 1–20
#   tasks 60– 79  ->  folders 022–028   WSI 1–20
#   tasks 80– 99  ->  folders 029–035   WSI 1–20
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/run_calibration_all.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=20-39 I2I-Stain-Zoo/scripts/run_calibration_all.sh

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo and the log comes
# back empty. It goes on after conda activate, as the rest of this family does.
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

mkdir -p logs_ensemble_ugac

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# 2D decomposition: 5 data blocks x 20 test WSIs
#
# N_WSIS must match --array above: the two are one decomposition, and changing
# either alone silently remaps every task to the wrong block.
# -----------------------------
N_WSIS=20

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))    # 0 … 4
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))     # 0 … 19

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$RANGE_ID]}" "${RANGE_ENDS[$RANGE_ID]}")

WSI_NUM=$(( WSI_IDX + 1 ))                      # 1 … 20
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")

# -----------------------------
# Paths — overridable via --export
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
MIN_TISSUE_PIXELS="${MIN_TISSUE_PIXELS:-256}"
N_BINS="${N_BINS:-10}"

ENSEMBLE_ROOT="${UGAC_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"

UNCERTAINTY_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/raw_npy/${WSI_FOLDER}/images"
ERROR_DIR="${ENSEMBLE_ROOT}/regen_error/wsi${WSI_FOLDER}/error_npy"
MASK_DIR="${TEST_A}/${WSI_FOLDER}/masks"
OUTDIR="${ENSEMBLE_ROOT}/calibration/${MODEL}/wsi${WSI_FOLDER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  WSI=${WSI_FOLDER}"
echo "uncertainty_dir : ${UNCERTAINTY_DIR}"
echo "error_dir       : ${ERROR_DIR}"
echo "mask_dir        : ${MASK_DIR}"
echo "outdir          : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${UNCERTAINTY_DIR}" ] || [ -z "$(ls -A "${UNCERTAINTY_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Uncertainty npy dir missing or empty: ${UNCERTAINTY_DIR}"
    echo "        Run compute_ensemble_uncertainty.sh --array=${SLURM_ARRAY_TASK_ID} first."
    exit 1
fi

if [ ! -d "${ERROR_DIR}" ] || [ -z "$(ls -A "${ERROR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Regen-error npy dir missing or empty: ${ERROR_DIR}"
    echo "        Run compute_ensemble_regen_error.sh --array=${SLURM_ARRAY_TASK_ID} first."
    exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
    echo "[ERROR] Mask directory not found: ${MASK_DIR}"
    echo "        Tissue masks are required — without them background pixels"
    echo "        inflate both rho and ECE. Tile with --mask, or pass --no_mask"
    echo "        deliberately and say so wherever the numbers are reported."
    exit 1
fi

# uncertainty_calibration.py scores the INTERSECTION of the uncertainty stems,
# the error stems and the mask stems. The two npy sets should already agree —
# both were tissue-filtered at 0.1 against these same masks — so a difference
# means one of the upstream jobs did not finish this WSI, and the run would
# quietly report a correlation over the tiles that happen to be in both.
N_U=$(find "${UNCERTAINTY_DIR}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
N_E=$(find "${ERROR_DIR}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
echo "Tiles           : ${N_U} uncertainty, ${N_E} error"
if [ "${N_U}" -ne "${N_E}" ]; then
    echo "[WARN] Tile counts differ (${N_U} vs ${N_E}). Only the intersection is"
    echo "       scored, so per_tile.csv will be shorter than either. Both were"
    echo "       filtered at the same threshold against the same masks, so this"
    echo "       points at an unfinished upstream job rather than at filtering."
fi

# Skip guard
if [ -f "${OUTDIR}/summary.json" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}/summary.json"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# --tiles_metadata enables the per_wsi.csv rollup via the source_file column.
# It is the dataset ROOT here, not this WSI's folder, because the rollup maps
# every tile name back to its slide.
run_cmd python "${PROJECT_ROOT}/uncertainty_calibration.py" \
    --uncertainty_dir     "${UNCERTAINTY_DIR}" \
    --error_dirs          "${ERROR_DIR}" \
    --mask_dir            "${MASK_DIR}" \
    --tiles_metadata      "${TEST_A}" \
    --outdir              "${OUTDIR}" \
    --n_bins              "${N_BINS}" \
    --min_tissue_pixels   "${MIN_TISSUE_PIXELS}" \
    --title               "${RANGE_TAG} WSI ${WSI_FOLDER}"

echo
echo "Done. ${RANGE_TAG} WSI ${WSI_FOLDER} -> ${OUTDIR}"
echo "Pool the tiles across WSIs per block with aggregate_calibration.py; a"
echo "mean of per-WSI summaries is not the same as recomputing on the pooled"
echo "tiles, and the reliability bins in particular need the full pool."
