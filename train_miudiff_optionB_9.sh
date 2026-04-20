#!/bin/bash
#SBATCH --job-name=miudiff_B9
#SBATCH --output=logs_all/miudiff_B_%A_%a.out
#SBATCH --error=logs_all/miudiff_B_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-8   # 9 jobs = 3 model-sizes x 3 data-sizes

# Option B architecture: 3-level UNet, 1 ResBlock per level (simpler/faster,
# same output normalisation fix as Option A).
# Fewer residual accumulations per level → less activation compounding.
#
# Model sizes (approx):
#   small  → base=64,  mult=1,2,4, n_res=1  (~17.6M params)
#   medium → base=96,  mult=1,2,4, n_res=1  (~38.1M params)
#   large  → base=160, mult=1,2,4, n_res=1  (~103.9M params)

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
conda activate i2istain

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs_all

# -----------------------------
# Helper: echo and run a command
# -----------------------------
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes
# MODEL_SIZE_ID varies fastest (0=small, 1=medium, 2=large).
# DATA_SIZE_ID   varies slowest (0=small, 1=medium, 2=large).
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

DATA_SIZE_ID=$(( TASK_ID / 3 ))
MODEL_SIZE_ID=$(( TASK_ID % 3 ))

MODELSIZES=("small"  "medium" "large")
DATASIZES=("small"   "medium" "large")
DATA_RANGES=("1,7"   "1,14"   "1,28")

MODELSIZE=${MODELSIZES[$MODEL_SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_SIZE_ID]}
DATA_RANGE=${DATA_RANGES[$DATA_SIZE_ID]}

# Option B architecture: 3 levels, 1 ResBlock per level
#                          small   medium  large
MIU_MCHLS=(               "64"    "96"    "160")
MIU_CHLMULT="1,2,4"       # same for all sizes
MIU_NRES=1                # 1 ResBlock per level for all sizes

MCHL=${MIU_MCHLS[$MODEL_SIZE_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=miudiff (Option B)"
echo "MODEL_SIZE=${MODELSIZE}  (base_channels=${MCHL}, channel_mult=${MIU_CHLMULT}, num_res_blocks=${MIU_NRES})"
echo "DATA_SIZE=${DATASIZE}  (data_range=${DATA_RANGE})"

# -----------------------------
# Paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/
PROJECT_ROOT=I2I-Stain-Zoo

DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

BASE=/work2/bz66izin-VSproject/Outputs/miudiff_optB
MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${MODELSIZE}

mkdir -p "${MODEL_DIR}"
echo "Output directory: ${MODEL_DIR}"

# -----------------------------
# Stage 1 — unconditional DDPM pretraining on domain B
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model miudiff \
    --miu_stage pretrain \
    --steps 3500000 \
    --data_range "${DATA_RANGE}" \
    --miu_base_channels "${MCHL}" \
    --miu_channel_mult "${MIU_CHLMULT}" \
    --miu_num_res_blocks "${MIU_NRES}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${MODEL_DIR}/stage1/" \
    --amp

# -----------------------------
# Stage 2 — conditional finetuning (eps_cond) initialised from stage 1
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model miudiff \
    --miu_stage finetune \
    --steps 750000 \
    --miu_init_ckpt "${MODEL_DIR}/stage1/checkpoints/step_3500000.pt" \
    --data_range "${DATA_RANGE}" \
    --miu_base_channels "${MCHL}" \
    --miu_channel_mult "${MIU_CHLMULT}" \
    --miu_num_res_blocks "${MIU_NRES}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${MODEL_DIR}/stage2/" \
    --amp

# -----------------------------
# Stage 3 — finetune with patch contrastive loss (PCL) from stage 2
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model miudiff \
    --miu_stage finetune \
    --miu_pcl \
    --steps 750000 \
    --miu_init_ckpt "${MODEL_DIR}/stage2/checkpoints/step_750000.pt" \
    --data_range "${DATA_RANGE}" \
    --miu_base_channels "${MCHL}" \
    --miu_channel_mult "${MIU_CHLMULT}" \
    --miu_num_res_blocks "${MIU_NRES}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${MODEL_DIR}/stage3/" \
    --amp
