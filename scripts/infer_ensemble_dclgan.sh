#!/bin/bash
#SBATCH --job-name=i2i_ens_dcl_infer
#SBATCH --output=logs_ensemble/dclgan_infer_%A_%a.out
#SBATCH --error=logs_ensemble/dclgan_infer_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-9   # 10 ensemble members

set -euo pipefail

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

# -----------------------------
# Config
# -----------------------------
MEMBER=$(printf "%02d" $((SLURM_ARRAY_TASK_ID + 1)))   # 01 … 10
DATA_RANGE="1,5"   # testA has 5 folders (001–005) — use all for inference

PROJECT_ROOT=I2I-Stain-Zoo
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble/dclgan/data_large/model_small
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_750000.pt"
OUTDIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_dclgan.sh first."
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
    --model dclgan \
    --direction A2B \
    --data "${TEST_A}" \
    --ckpt "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir "${OUTDIR}"

echo "Done. Inference for member ${MEMBER} saved to ${OUTDIR}"
