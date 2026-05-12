#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_infer
#SBATCH --output=logs_ensemble/cyclegan_infer_%A_%a.out
#SBATCH --error=logs_ensemble/cyclegan_infer_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4   # 5 ensemble members

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
MEMBER=$(printf "%02d" $((SLURM_ARRAY_TASK_ID + 1)))   # 01 … 05

PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
TEST_A="${DATA_DIR}/QP_HE/tiles/testA"

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble/cyclegan/data_large/model_medium
CKPT="${ENSEMBLE_ROOT}/model_${MEMBER}/checkpoints/step_5000000.pt"
OUTDIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclegan.sh first."
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
    --model cyclegan \
    --direction A2B \
    --data "${TEST_A}" \
    --ckpt "${CKPT}" \
    --outdir "${OUTDIR}"

echo "Done. Inference for member ${MEMBER} saved to ${OUTDIR}"
