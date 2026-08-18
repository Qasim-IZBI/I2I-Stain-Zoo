#!/bin/bash
#SBATCH --job-name=i2i_agg_uncertainty
#SBATCH --output=logs_ensemble_ugac/agg_uncertainty_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/agg_uncertainty_%A_%a.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-4   # 5 jobs = 1 per data block

# Per-tile sigma-bar, rolled up per WSI, for the UGAC CycleGAN ensemble.
#
# Consumes what compute_ensemble_uncertainty.sh writes:
#
#   {UGAC_ROOT}/{block}/model_small/uncertainty/cyclegan/raw_npy/{NNN}/images/*.npy
#
# and reduces each per-pixel map to one number per tile:
#
#   {UGAC_ROOT}/{block}/model_small/uncertainty/cyclegan/per_wsi_csv/*.csv
#
# ONE JOB PER BLOCK, not per WSI. aggregate_uncertainty.py derives WSI
# membership from the {NNN}/ component of each npy path — which is what keeps
# tile ids repeating across slides from colliding — so a single pass over
# raw_npy/ already writes every per-WSI CSV. Splitting it further would have the
# twenty tasks of a block writing into one directory for no gain.
#
# Decomposition matches compute_ensemble_uncertainty.sh's block axis, so a task
# here corresponds to twenty tasks there:
#   task 0  ->  folders 001–007   (uncertainty tasks  0– 19)
#   task 1  ->  folders 008–014   (uncertainty tasks 20– 39)
#   task 2  ->  folders 015–021   (uncertainty tasks 40– 59)
#   task 3  ->  folders 022–028   (uncertainty tasks 60– 79)
#   task 4  ->  folders 029–035   (uncertainty tasks 80– 99)
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/aggregate_uncertainty.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=1 I2I-Stain-Zoo/scripts/aggregate_uncertainty.sh

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
# Block axis — identical to compute_ensemble_uncertainty.sh
# -----------------------------
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_ID=${SLURM_ARRAY_TASK_ID}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$RANGE_ID]}" "${RANGE_ENDS[$RANGE_ID]}")

# Expected number of test WSIs, so a short run is visible. The tool has no
# --expect of its own, unlike aggregate_phi_uncertainty.py.
N_WSIS="${N_WSIS:-20}"

# -----------------------------
# Paths — overridable via --export
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"

ENSEMBLE_ROOT="${UGAC_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"
UNCERTAINTY_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/raw_npy"
OUT_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/per_wsi_csv"

# MUST be the same tiles compute_ensemble_uncertainty.sh used. The rollup keys
# tiles by the {NNN}/ folder and reads masks from the same tree, so a different
# test set silently maps tiles to the wrong slide or drops them for want of a
# mask. The two scripts previously disagreed here — this one pointed at
# VS_Data/temp while that one used VS_Data/eval_imgs.
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
MIN_TISSUE="${MIN_TISSUE:-0.1}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}"
echo "Uncertainty dir : ${UNCERTAINTY_DIR}"
echo "Tiles / masks   : ${TEST_A}"
echo "Output          : ${OUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${UNCERTAINTY_DIR}" ] || \
   [ -z "$(find "${UNCERTAINTY_DIR}" -name '*.npy' -maxdepth 4 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No .npy uncertainty maps under: ${UNCERTAINTY_DIR}"
    echo "        Run compute_ensemble_uncertainty.sh --array=$(( RANGE_ID * N_WSIS ))-$(( RANGE_ID * N_WSIS + N_WSIS - 1 )) first."
    exit 1
fi

if [ ! -d "${TEST_A}" ]; then
    echo "[ERROR] Test tiles not found: ${TEST_A}"
    exit 1
fi

# Each uncertainty job writes one WSI folder, so a block whose array only partly
# finished leaves a raw_npy/ that looks populated but covers fewer slides. The
# rollup would then quietly emit fewer CSVs than the cohort has.
NPY_WSIS=$(find "${UNCERTAINTY_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "${NPY_WSIS}" -ne "${N_WSIS}" ]; then
    echo "[WARN] ${UNCERTAINTY_DIR} holds ${NPY_WSIS} WSI folder(s), expected ${N_WSIS}."
    echo "       compute_ensemble_uncertainty.sh has not finished this block."
    echo "       Proceeding — the rollup covers what exists — but the CSV count"
    echo "       below will be short and must not be pooled as a full cohort."
fi

# Skip guard: at least one CSV already written for this block
if [ -d "${OUT_DIR}" ] && \
   [ -n "$(find "${OUT_DIR}" -maxdepth 1 -name '*.csv' 2>/dev/null | head -1)" ]; then
    echo "[SKIP] Already completed: ${OUT_DIR}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/aggregate_uncertainty.py" \
    --uncertainty_dir     "${UNCERTAINTY_DIR}" \
    --tiles_metadata      "${TEST_A}" \
    --mask_dir            "${TEST_A}" \
    --min_tissue_fraction "${MIN_TISSUE}" \
    --outdir              "${OUT_DIR}"

WROTE=$(find "${OUT_DIR}" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
echo
echo "Done. ${RANGE_TAG}: ${WROTE} per-WSI CSV(s) -> ${OUT_DIR}/"
if [ "${WROTE}" -ne "${N_WSIS}" ]; then
    echo "[WARN] Expected ${N_WSIS} CSVs, wrote ${WROTE}. A slide missing here is a"
    echo "       slide missing from every downstream distribution — check that the"
    echo "       uncertainty array covered all ${N_WSIS} WSIs for this block."
fi
