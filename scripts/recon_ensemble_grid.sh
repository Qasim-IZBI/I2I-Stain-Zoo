#!/bin/bash
#SBATCH --job-name=i2i_recon_grid
#SBATCH --output=logs_ensemble_grid/recon_grid_%A_%a.out
#SBATCH --error=logs_ensemble_grid/recon_grid_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-249   # 250 jobs = 5 subsets x 10 members x 5 test WSIs

# Stitch the A→B tiles of the grid ensemble into whole slides.
#
# Required by the phi_struct pipeline: beta_0, beta_1 and regional dispersion do
# not decompose over tiles, because components and loops cross tile boundaries.
# The topology of a region is not a function of its tiles' topologies, so the
# descriptors must be computed on stitched masks cropped to regions.
#
# 3D decomposition, extending train_ensemble_cyclegan_grid.sh by a WSI axis:
#   RANGE_ID  = TASK_ID / 50        (0-4  -> subset)
#   MEMBER_ID = (TASK_ID % 50) / 5  (0-9  -> member 01-10)
#   WSI_ID    = TASK_ID % 5         (0-4  -> folder 001-005)
#
# Input  : {subset}/model_small/inference/model_{NN}/     (flat B' tiles)
# Output : {subset}/model_small/reconstructed/model_{NN}/ (one TIF per WSI)
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/recon_ensemble_grid.sh
# One subset only (e.g. folders 008-014):
#   sbatch --array=50-99 I2I-Stain-Zoo/scripts/recon_ensemble_grid.sh

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

mkdir -p logs_ensemble_grid

# -----------------------------
# 3D decomposition: 5 subsets x 10 members x 5 WSIs
# -----------------------------
# NOTE: N_WSIS and TEST_A below describe the 5-WSI BMVC test set. For the
# 20-case held-out cohorts set N_WSIS to that cohort's WSI count and scale
# --array to 5 x 10 x N_WSIS - 1.
N_MEMBERS=10
N_WSIS=5

TASK_ID=${SLURM_ARRAY_TASK_ID}
RANGE_ID=$(( TASK_ID / (N_MEMBERS * N_WSIS) ))
MEMBER_ID=$(( (TASK_ID % (N_MEMBERS * N_WSIS)) / N_WSIS ))
WSI_ID=$(( TASK_ID % N_WSIS ))

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")

MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))       # 01 … 10
WSI_FOLDER=$(printf "%03d" $(( WSI_ID + 1 )))      # 001 … 005

PROJECT_ROOT=I2I-Stain-Zoo
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
WSI_METADATA="${TEST_A}/${WSI_FOLDER}/tiles_metadata.csv"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble_grid/cyclegan/${RANGE_TAG}/model_small"
TILE_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"
OUT_DIR="${ENSEMBLE_ROOT}/reconstructed/model_${MEMBER}"

# The output filename comes from the original WSI stem and is not known ahead of
# time, so completion is tracked with a per-WSI sentinel.
SENTINEL="${OUT_DIR}/.done_${WSI_FOLDER}"

echo "TASK_ID=${TASK_ID}  SUBSET=${RANGE_TAG}  MEMBER=${MEMBER}  WSI=${WSI_FOLDER}"
echo "Tiles    : ${TILE_DIR}"
echo "Metadata : ${WSI_METADATA}"
echo "Output   : ${OUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${WSI_METADATA}" ]; then
    echo "[ERROR] Metadata not found: ${WSI_METADATA}"
    exit 1
fi

if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] A→B tiles missing or empty: ${TILE_DIR}"
    echo "        Run infer_ensemble_cyclegan_grid.sh first."
    exit 1
fi

if [ -f "${SENTINEL}" ]; then
    echo "[SKIP] WSI ${WSI_FOLDER} already reconstructed for ${RANGE_TAG} member ${MEMBER}."
    exit 0
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${WSI_METADATA}" \
    --tile_dir "${TILE_DIR}" \
    --output   "${OUT_DIR}" \
    --mode     rgb

touch "${SENTINEL}"
echo "Done. ${RANGE_TAG} member ${MEMBER} WSI ${WSI_FOLDER} → ${OUT_DIR}"
