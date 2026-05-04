#!/bin/bash
#SBATCH --job-name=cyclediffusion_9
#SBATCH --output=logs_cyclediff/cyclediffusion_%A_%a.out
#SBATCH --error=logs_cyclediff/cyclediffusion_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-8   # 9 jobs = 3 model-sizes x 3 data-sizes

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
#
# Job layout:
#   0: model=small  data=small    5: model=large  data=medium
#   1: model=medium data=small    6: model=small  data=large
#   2: model=large  data=small    7: model=medium data=large
#   3: model=small  data=medium   8: model=large  data=large
#   4: model=medium data=medium
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

DATA_SIZE_ID=$(( TASK_ID / 3 ))
MODEL_SIZE_ID=$(( TASK_ID % 3 ))

MODELSIZES=("small"   "medium"  "large")
DATASIZES=("small"    "medium"  "large")
DATA_RANGES=("1,7"    "1,15"    "1,30")

MODELSIZE=${MODELSIZES[$MODEL_SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_SIZE_ID]}
DATA_RANGE=${DATA_RANGES[$DATA_SIZE_ID]}

# CycleDiffusion architecture per model size
#                              small      medium     large
CD_BASE_CHANNELS=(             "48"       "84"       "128")
CD_CHANNEL_MULTS=(             "1,2,2,4"  "1,2,2,4"  "1,2,4")
CD_NUM_RES_BLOCKS=(            "1"        "2"        "2")

BASE_CH=${CD_BASE_CHANNELS[$MODEL_SIZE_ID]}
CH_MULT=${CD_CHANNEL_MULTS[$MODEL_SIZE_ID]}
N_RES=${CD_NUM_RES_BLOCKS[$MODEL_SIZE_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=cyclediffusion"
echo "MODEL_SIZE=${MODELSIZE}  (base_channels=${BASE_CH}, channel_mult=${CH_MULT}, num_res_blocks=${N_RES})"
echo "DATA_SIZE=${DATASIZE}  (data_range=${DATA_RANGE})"

# -----------------------------
# Paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/
PROJECT_ROOT=I2I-Stain-Zoo

DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

BASE=/work2/bz66izin-VSproject/Outputs_noamp/cyclediffusion
MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${MODELSIZE}

mkdir -p "${MODEL_DIR}"
echo "Output directory: ${MODEL_DIR}"

# -----------------------------
# Training — single stage
# Both eps_A (domain A) and eps_B (domain B) are trained jointly.
# No warm-start or stage split needed.
# -----------------------------
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model cyclediffusion \
    --steps 2000000 \
    --data_range "${DATA_RANGE}" \
    --cd_base_channels "${BASE_CH}" \
    --cd_channel_mult "${CH_MULT}" \
    --cd_num_res_blocks "${N_RES}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --output "${MODEL_DIR}/" \
    --amp

echo "Done: cyclediffusion ${MODELSIZE} model / ${DATASIZE} data finished successfully."
