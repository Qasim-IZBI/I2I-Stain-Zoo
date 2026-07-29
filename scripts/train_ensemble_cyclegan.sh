#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_train
#SBATCH --output=logs_ensemble/cyclegan_train_%A_%a.out
#SBATCH --error=logs_ensemble/cyclegan_train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4   # 5 ensemble members

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

# -----------------------------
# Config: medium model, large data
# -----------------------------
MEMBER=$(printf "%02d" $((SLURM_ARRAY_TASK_ID + 1)))   # 01 … 05
SEED=$((SLURM_ARRAY_TASK_ID + 1))                       # 1  … 5

PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

# large data size — folders 001–028
DATA_RANGE="1,30"

# medium CycleGAN: ngf=128, n_blocks=10 (matches train_all_54.sh)
NGF=128
NBLOCKS=10

OUTPUT_DIR=/work2/bz66izin-VSproject/ensemble/cyclegan/data_large/model_medium/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${OUTPUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${OUTPUT_DIR}/checkpoints/step_750000.pt" ]; then
    echo "[SKIP] step_750000.pt already present — member ${MEMBER} is done."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/train.py" \
    --model cyclegan \
    --steps 750000 \
    --cyclegan_ngf "${NGF}" \
    --cyclegan_n_blocks "${NBLOCKS}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --data_range "${DATA_RANGE}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}"

echo "Done. Member ${MEMBER} saved to ${OUTPUT_DIR}"
