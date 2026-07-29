#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_ugac_b2a
#SBATCH --output=logs_ensemble_ugac/cyclegan_ugac_infer_B2A_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/cyclegan_ugac_infer_B2A_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = 5 data blocks x 10 ensemble members

# B→A re-translation of the A→B tiles from infer_ensemble_cyclegan_ugac.sh,
# giving the A' needed for cycle-reconstruction error without re-running the
# forward pass at evaluation time.
#
# Array layout matches train_ensemble_cyclegan_ugac.sh exactly:
#   tasks  0– 9  ->  folders 001–007   members 01–10
#   tasks 10–19  ->  folders 008–014   members 01–10
#   tasks 20–29  ->  folders 015–021   members 01–10
#   tasks 30–39  ->  folders 022–028   members 01–10
#   tasks 40–49  ->  folders 029–035   members 01–10
#
# Input  : .../{block}/model_small/inference/model_{NN}/       (B' tiles)
# Output : .../{block}/model_small/inference_B2A/model_{NN}/   (A' tiles)
#
# No --save_aleatoric here: the aleatoric map of interest is the one on the
# forward A→B translation, already written by the A2B job. This pass exists
# only to produce A' for the regen-error error proxy.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/infer_ensemble_cyclegan_ugac_B2A.sh

set -eo pipefail

module purge
module load Anaconda3

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
DATA_RANGE="1,5"   # 5 test WSIs (001–005)

STEPS=750000

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/${RANGE_TAG}/model_small
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_${STEPS}.pt"
IN_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"
OUTDIR="${ENSEMBLE_ROOT}/inference_B2A/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Input      : ${IN_DIR}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclegan_ugac.sh first (same array index)."
    exit 1
fi

if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] A→B inference output missing or empty: ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan_ugac.sh first (same array index)."
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
    --direction  B2A \
    --data       "${IN_DIR}" \
    --ckpt       "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir     "${OUTDIR}"

echo "Done. ${RANGE_TAG} member ${MEMBER} B2A saved to ${OUTDIR}"
