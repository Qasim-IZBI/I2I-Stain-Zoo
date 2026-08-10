#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_grid_infer
#SBATCH --output=logs_ensemble_grid/cyclegan_grid_infer_%A_%a.out
#SBATCH --error=logs_ensemble_grid/cyclegan_grid_infer_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = K=5 subsets x S=10 seeds

# A→B inference for the crossed (subset x seed) CycleGAN grid.
#
# Array layout matches train_ensemble_cyclegan_grid.sh exactly:
#   tasks  0– 9  ->  folders 001–007   members 01–10
#   tasks 10–19  ->  folders 008–014   members 01–10
#   tasks 20–29  ->  folders 015–021   members 01–10
#   tasks 30–39  ->  folders 022–028   members 01–10
#   tasks 40–49  ->  folders 029–035   members 01–10
#
# Output: {subset}/model_small/inference/model_{NN}/   translated tiles
#
# No --save_aleatoric. The decomposition identity has no aleatoric term, and the
# vanilla generator has no GGD heads to produce one; inference.py would refuse a
# non-UGAC checkpoint anyway.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/infer_ensemble_cyclegan_grid.sh
# Single subset only (e.g. folders 008–014):
#   sbatch --array=10-19 I2I-Stain-Zoo/scripts/infer_ensemble_cyclegan_grid.sh

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
# 2D decomposition: K=5 subsets x S=10 seeds
# -----------------------------
N_MEMBERS=10

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))    # 0 … 4
MEMBER_ID=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))   # 0 … 9

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")

MEMBER=$(printf "%02d" $((MEMBER_ID + 1)))   # 01 … 10

PROJECT_ROOT=I2I-Stain-Zoo

# NOTE: repoint TEST_A at the 20-case held-out cohort (liver) or the 20-case
# kidney cohort, and set DATA_RANGE to that cohort's folder count, before the
# evaluation runs. The path below is the 5-WSI BMVC test set.
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
DATA_RANGE="1,5"

STEPS=750000

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble_grid/cyclegan/${RANGE_TAG}/model_small
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_${STEPS}.pt"
OUTDIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  SUBSET=${RANGE_TAG}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclegan_grid.sh first (same array index)."
    exit 1
fi

if [ ! -d "${TEST_A}" ]; then
    echo "[ERROR] Test tiles not found: ${TEST_A}"
    exit 1
fi

if [ -d "${OUTDIR}" ] && [ -n "$(ls -A "${OUTDIR}" 2>/dev/null)" ]; then
    echo "[SKIP] Output directory already populated: ${OUTDIR}"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/inference.py" \
    --model      cyclegan \
    --direction  A2B \
    --data       "${TEST_A}" \
    --ckpt       "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir     "${OUTDIR}"

echo "Done. ${RANGE_TAG} member ${MEMBER} A2B saved to ${OUTDIR}"
