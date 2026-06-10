#!/bin/bash
#SBATCH --job-name=i2i_judge_b2a
#SBATCH --output=logs_ensemble/judge_b2a_%A_%a.out
#SBATCH --error=logs_ensemble/judge_b2a_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-29   # 30 jobs = 6 models × 5 ensemble members

# Runs CycleGAN B2A (judge model) on the A→B inference output of every model.
# Produces judge-reconstructed A' tiles used for judge_regen_error calibration.
#
# A single fixed CycleGAN checkpoint is used as the judge across all 6 models,
# giving a comparable error signal regardless of which model generated B'.
#
# Inputs (per job):
#   {MODEL_ROOT}/inference/model_{MEMBER}/{NNN}/images/*.tif  (A→B output)
# Outputs (per job):
#   {MODEL_ROOT}/judge_B2A/model_{MEMBER}/{NNN}/images/*.tif  (judge A' tiles)
#
# Array layout
# tasks 0–4   → cyclegan        (model_medium)  members 01–05
# tasks 5–9   → unit            (model_medium)  members 01–05
# tasks 10–14 → munit           (model_medium)  members 01–05
# tasks 15–19 → dclgan          (model_small)   members 01–05
# tasks 20–24 → uvcgan          (model_small)   members 01–05
# tasks 25–29 → cyclediffusion  (model_small)   members 01–05

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

mkdir -p logs_ensemble

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Fixed CycleGAN judge checkpoint (same for all 6 models)
# -----------------------------
JUDGE_CKPT="/work2/bz66izin-VSproject/ensemble/cyclegan/data_large/model_medium/models/model_01/checkpoints/step_750000.pt"

DATA_RANGE="1,5"   # testA WSI folders 001–005

# -----------------------------
# 2D decomposition: 6 models × 5 members
# -----------------------------
N_MEMBERS=5
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))   # 0 … 5
MEMBER_IDX=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))  # 0 … 4

MEMBER=$(printf "%02d" $(( MEMBER_IDX + 1 )))      # 01 … 05

# Per-model config
# Index:          0          1      2       3        4        5
MODELS=(      cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=( model_medium model_medium model_medium model_small model_small model_small )

MODEL="${MODELS[$MODEL_IDX]}"
MODEL_SIZE="${MODEL_SIZES[$MODEL_IDX]}"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"
IN_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"
OUTDIR="${ENSEMBLE_ROOT}/judge_B2A/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}  MEMBER=${MEMBER}"
echo "Judge ckpt : ${JUDGE_CKPT}"
echo "Input      : ${IN_DIR}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${JUDGE_CKPT}" ]; then
    echo "[ERROR] Judge checkpoint not found: ${JUDGE_CKPT}"
    exit 1
fi

if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] A→B inference output missing or empty: ${IN_DIR}"
    echo "        Run infer_ensemble_${MODEL}.sh first."
    exit 1
fi

# Skip guard
if [ -d "${OUTDIR}" ] && [ -n "$(ls -A "${OUTDIR}" 2>/dev/null)" ]; then
    echo "[SKIP] Output already populated: ${OUTDIR}"
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
    --model     cyclegan \
    --direction B2A \
    --data      "${IN_DIR}" \
    --ckpt      "${JUDGE_CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir    "${OUTDIR}"

echo "Done. Judge B2A for ${MODEL} member ${MEMBER} saved to ${OUTDIR}"
