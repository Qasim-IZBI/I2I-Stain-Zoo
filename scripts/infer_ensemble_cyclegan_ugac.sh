#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_ugac_infer
#SBATCH --output=logs_ensemble_ugac/cyclegan_ugac_infer_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/cyclegan_ugac_infer_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = 5 data blocks x 10 ensemble members

# A→B inference for the UGAC CycleGAN ensemble, with per-pixel aleatoric
# uncertainty saved alongside the translated tiles.
#
# Array layout matches train_ensemble_cyclegan_ugac.sh exactly:
#   tasks  0– 9  ->  folders 001–007   members 01–10
#   tasks 10–19  ->  folders 008–014   members 01–10
#   tasks 20–29  ->  folders 015–021   members 01–10
#   tasks 30–39  ->  folders 022–028   members 01–10
#   tasks 40–49  ->  folders 029–035   members 01–10
#
# Outputs per job:
#   .../{block}/model_small/inference/model_{NN}/                 translated tiles
#   .../{block}/model_small/inference/model_{NN}/aleatoric_npy/   [H,W] float32 SD
#
# The aleatoric maps use the same convention as uncertainty.py raw_npy/, so they
# feed uncertainty_calibration.py directly. Epistemic uncertainty is a separate
# step: run uncertainty.py across the 10 model_NN dirs within one block.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/infer_ensemble_cyclegan_ugac.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=10-19 I2I-Stain-Zoo/scripts/infer_ensemble_cyclegan_ugac.sh

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

# -----------------------------
# 2D decomposition: 5 data blocks x 10 members
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
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
DATA_RANGE="1,5"   # testA has 5 folders (001–005) — use all

STEPS=750000

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/${RANGE_TAG}/model_small
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_${STEPS}.pt"
OUTDIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclegan_ugac.sh first (same array index)."
    exit 1
fi

if [ ! -d "${TEST_A}" ]; then
    echo "[ERROR] Test tiles not found: ${TEST_A}"
    exit 1
fi

if [ -d "${OUTDIR}/aleatoric_npy" ] && [ -n "$(ls -A "${OUTDIR}/aleatoric_npy" 2>/dev/null)" ]; then
    echo "[SKIP] Aleatoric maps already present: ${OUTDIR}/aleatoric_npy"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# --save_aleatoric requires a checkpoint trained with --cyclegan_ugac; inference.py
# exits with a clear error otherwise rather than emitting garbage. cfg.ugac is
# restored from the checkpoint, so no architecture flag is needed here.
run_cmd python "${PROJECT_ROOT}/inference.py" \
    --model      cyclegan \
    --direction  A2B \
    --data       "${TEST_A}" \
    --ckpt       "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir     "${OUTDIR}" \
    --save_aleatoric

echo "Done. ${RANGE_TAG} member ${MEMBER} A2B + aleatoric saved to ${OUTDIR}"
