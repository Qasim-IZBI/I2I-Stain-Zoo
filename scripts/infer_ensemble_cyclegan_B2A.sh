#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_b2a
#SBATCH --output=logs_ensemble/cyclegan_infer_B2A_%A_%a.out
#SBATCH --error=logs_ensemble/cyclegan_infer_B2A_%A_%a.err

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
DATA_RANGE="1,5"   # 5 test WSIs (001–005)

PROJECT_ROOT=I2I-Stain-Zoo

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble/cyclegan/data_large/model_medium
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_750000.pt"

# Input: A→B translated tiles produced by infer_ensemble_cyclegan.sh
IN_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

# Output: B→A re-translated tiles (A' = judge(B'))
OUTDIR="${ENSEMBLE_ROOT}/inference_B2A/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}"
echo "Checkpoint : ${CKPT}"
echo "Input      : ${IN_DIR}"
echo "Output     : ${OUTDIR}"

# -----------------------------
# Pre-flight: checkpoint must exist
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclegan.sh first."
    exit 1
fi

# Pre-flight: A→B output must exist and be populated
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] A→B inference output missing or empty: ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan.sh first."
    exit 1
fi

# Skip guard
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
    --model     cyclegan \
    --direction B2A \
    --data      "${IN_DIR}" \
    --ckpt      "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --outdir    "${OUTDIR}"

echo "Done. Member ${MEMBER} B2A saved to ${OUTDIR}"
