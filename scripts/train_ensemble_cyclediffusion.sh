#!/bin/bash
#SBATCH --job-name=i2i_ens_cd_train
#SBATCH --output=logs_ensemble/cyclediffusion_train_%A_%a.out
#SBATCH --error=logs_ensemble/cyclediffusion_train_%A_%a.err

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

# large data size — folders 001–030
DATA_RANGE="1,30"

# small CycleDiffusion: base=48, channel_mult=1,2,2,4, num_res_blocks=1 (~10.7M A→B)
CD_BASE=48
CD_MULT="1,2,2,4"
CD_RESBLOCKS=1

MEMBER_DIR=/work2/bz66izin-VSproject/ensemble/cyclediffusion/data_large/model_small/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${MEMBER_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${MEMBER_DIR}/checkpoints/step_750000.pt" ]; then
    echo "[SKIP] step_750000.pt already present — member ${MEMBER} is fully done."
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
# Single-stage joint training (both eps_A and eps_B simultaneously)
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model cyclediffusion \
    --steps 750000 \
    --cd_base_channels "${CD_BASE}" \
    --cd_channel_mult  "${CD_MULT}" \
    --cd_num_res_blocks "${CD_RESBLOCKS}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --data_range "${DATA_RANGE}" \
    --seed "${SEED}" \
    --output "${MEMBER_DIR}"

echo "Done. Member ${MEMBER} saved to ${MEMBER_DIR}"
