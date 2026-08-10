#!/bin/bash
#SBATCH --job-name=i2i_ens_uvcgan_train
#SBATCH --output=logs_ensemble/uvcgan_train_%A_%a.out
#SBATCH --error=logs_ensemble/uvcgan_train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-9   # 10 ensemble members

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

mkdir -p logs_ensemble

# -----------------------------
# Config: small model, large data
# -----------------------------
MEMBER=$(printf "%02d" $((SLURM_ARRAY_TASK_ID + 1)))   # 01 … 10
SEED=$((SLURM_ARRAY_TASK_ID + 1))                       # 1  … 10

PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

# large data size — folders 001–028
DATA_RANGE="1,30"

# small UVCGAN: ngf=48, vit_blocks=6, vit_features=96 (matches train_all_54.sh)
NGF=48
VBLOCKS=6
VFEATS=96

MEMBER_DIR=/work2/bz66izin-VSproject/ensemble/uvcgan/data_large/model_small/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${MEMBER_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${MEMBER_DIR}/stage2/checkpoints/step_500000.pt" ]; then
    echo "[SKIP] step_500000.pt already present — member ${MEMBER} is fully done."
    exit 0
fi

mkdir -p "${MEMBER_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Stage 1 — masked image modelling pretrain
# -----------------------------
if [ -f "${MEMBER_DIR}/stage1/checkpoints/step_250000.pt" ]; then
    echo "[SKIP stage1] step_250000.pt already present — skipping pretrain."
else
    run_cmd python "${PROJECT_ROOT}/train.py" \
        --model uvcgan \
        --uvcgan_stage pretrain \
        --steps 250000 \
        --uvcgan_ngf "${NGF}" \
        --uvcgan_vit_blocks "${VBLOCKS}" \
        --uvcgan_vit_features "${VFEATS}" \
        --dataA "${DATA_A}" \
        --dataB "${DATA_B}" \
        --data_range "${DATA_RANGE}" \
        --seed "${SEED}" \
        --output "${MEMBER_DIR}/stage1/"
fi

# -----------------------------
# Stage 2 — cycle-consistent finetuning
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model uvcgan \
    --uvcgan_stage finetune \
    --steps 500000 \
    --uvcgan_init_ckpt "${MEMBER_DIR}/stage1/checkpoints/step_250000.pt" \
    --uvcgan_ngf "${NGF}" \
    --uvcgan_vit_blocks "${VBLOCKS}" \
    --uvcgan_vit_features "${VFEATS}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --data_range "${DATA_RANGE}" \
    --seed "${SEED}" \
    --amp \
    --output "${MEMBER_DIR}/stage2/"

echo "Done. Member ${MEMBER} saved to ${MEMBER_DIR}"
