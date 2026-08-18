#!/bin/bash
#SBATCH --job-name=i2i_ens_regen_err
#SBATCH --output=logs_ensemble_ugac/regen_error_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/regen_error_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-99   # 100 jobs = 5 data blocks x 20 test WSIs

# Cycle-reconstruction error |A - mean(A')| per tile, for the UGAC CycleGAN
# ensemble, via evaluation.py's precomputed-A' mode.
#
# Consumes what compute_ensemble_regen_stats.sh writes:
#
#   {UGAC_ROOT}/{block}/model_small/regen_stats/cyclegan/mean_rgb/{NNN}/images/
#
# against the real H&E tiles, and writes:
#
#   {UGAC_ROOT}/{block}/model_small/regen_error/wsi{NNN}/
#       error_npy/   per-tile [H,W] float32 MAE in 0-255  <- the consumable
#       heatmaps/    per-tile PNGs, qualitative only
#
# That `regen_error/wsi{NNN}/error_npy/` layout is exactly what
# compare_uncertainty_sources.py takes as `--regen_root`, so the head-to-head
# against ensemble spread needs no reshuffling:
#
#   python compare_uncertainty_sources.py --regen_root \
#       {UGAC_ROOT}/{block}/model_small/regen_error ...
#
# Note what a --regen_root means HERE. Each block yields ONE error map set,
# computed from that block's mean A' — not one per member. So repeating
# --regen_root across the five blocks averages over blocks, giving the grand
# mean cycle error, which is the right companion to the grand ensemble spread.
# It is not the per-member average that script's help describes for the flat
# ensembles, and the two must not be mixed in one run.
#
# Decomposition matches compute_ensemble_regen_stats.sh exactly:
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
#   sbatch I2I-Stain-Zoo/scripts/compute_ensemble_regen_error.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=20-39 I2I-Stain-Zoo/scripts/compute_ensemble_regen_error.sh

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

# MUST be the tiles the A2B inference consumed, or A and A' describe different
# slides under the same folder number.
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
MIN_TISSUE="${MIN_TISSUE:-0.1}"

ENSEMBLE_ROOT="${UGAC_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"

# Both are LEAF image directories for one WSI. evaluation.py matches by
# basename, which is safe here for that reason and not because tile ids are
# globally unique — they are NOT, every slide has a 0000001. Pointing either of
# these at a dataset root instead of a single WSI would silently pair tiles
# across slides.
PATH_A="${TEST_A}/${WSI_FOLDER}/images"
PATH_A_REGEN="${ENSEMBLE_ROOT}/regen_stats/${MODEL}/mean_rgb/${WSI_FOLDER}/images"
MASK_DIR="${TEST_A}/${WSI_FOLDER}/masks"

OVERLAY_DIR="${ENSEMBLE_ROOT}/regen_error/wsi${WSI_FOLDER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  WSI=${WSI_FOLDER}"
echo "path_A       : ${PATH_A}"
echo "path_A_regen : ${PATH_A_REGEN}"
echo "Output       : ${OVERLAY_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${PATH_A}" ]; then
    echo "[ERROR] testA WSI directory not found: ${PATH_A}"
    echo "        Is the test set smaller than N_WSIS=${N_WSIS}?"
    exit 1
fi

if [ ! -d "${PATH_A_REGEN}" ] || [ -z "$(ls -A "${PATH_A_REGEN}" 2>/dev/null)" ]; then
    echo "[ERROR] Mean A' directory missing or empty: ${PATH_A_REGEN}"
    echo "        Run compute_ensemble_regen_stats.sh --array=${SLURM_ARRAY_TASK_ID} first."
    exit 1
fi

# evaluation.py scores the INTERSECTION of the two directories, so a short
# mean_rgb/ produces a smaller error set with no complaint — and the missing
# tiles are not random, they are wherever the B2A pass or the mean failed.
N_A=$(find "${PATH_A}" -maxdepth 1 -type f \
        \( -name '*.tif' -o -name '*.tiff' -o -name '*.png' \) 2>/dev/null | wc -l)
N_R=$(find "${PATH_A_REGEN}" -maxdepth 1 -type f \
        \( -name '*.tif' -o -name '*.tiff' -o -name '*.png' \) 2>/dev/null | wc -l)
echo "Tiles        : ${N_A} in A, ${N_R} in mean A'"
if [ "${N_A}" -ne "${N_R}" ]; then
    echo "[ERROR] Tile counts differ (${N_A} vs ${N_R}). evaluation.py would score"
    echo "        the intersection and report a number for fewer tiles than the"
    echo "        slide has, with nothing downstream to show which are missing."
    echo "        Re-run compute_ensemble_regen_stats.sh for this block/WSI, and"
    echo "        check it did not tissue-filter (it must not pass --mask_dir)."
    exit 1
fi

# Per-WSI skip guard
if [ -d "${OVERLAY_DIR}/error_npy" ] && \
   [ -n "$(ls -A "${OVERLAY_DIR}/error_npy" 2>/dev/null)" ]; then
    echo "[SKIP] Already completed: ${OVERLAY_DIR}/error_npy"
    exit 0
fi

mkdir -p "${OVERLAY_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# No --device: compute_regen_error_precomputed is CPU-only (pure MAE).
#
# --mask_dir IS wanted here, unlike in compute_ensemble_regen_stats.sh. There it
# would have dropped tiles from an image; here it removes background from a
# statistic, which is the point — unmasked, the error is diluted by slide
# background wherever a tile is mostly empty.
run_cmd python "${PROJECT_ROOT}/evaluation.py" \
    --metric              regen_error \
    --path_A              "${PATH_A}" \
    --path_A_regen        "${PATH_A_REGEN}" \
    --overlay_dir         "${OVERLAY_DIR}" \
    --save_error_npy \
    --mask_dir            "${MASK_DIR}" \
    --min_tissue_fraction "${MIN_TISSUE}"

echo
echo "Done. ${RANGE_TAG} WSI ${WSI_FOLDER} -> ${OVERLAY_DIR}/error_npy/"
echo "Once every block is complete, the head-to-head against ensemble spread is:"
echo "  sbatch --export=ALL,REGEN_ROOTS='$(for i in "${!RANGE_STARTS[@]}"; do \
    printf "%s/%s/%s/regen_error " "${UGAC_ROOT}" \
    "$(printf 'data_%03d_%03d' "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")" \
    "${MODEL_SIZE}"; done)' \\"
echo "      ${PROJECT_ROOT}/scripts/compare_uncertainty_sources.sh"
