#!/bin/bash
#SBATCH --job-name=diffusion_4
#SBATCH --output=logs_diffusion/train_%A_%a.out
#SBATCH --error=logs_diffusion/train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-3   # 4 jobs: 0=miudiff, 1=unitddpm, 2=cyclediffusion, 3=unsb

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
conda activate i2istain

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs_diffusion

# -----------------------------------------------------------------------
# Helper: echo and run a command
# -----------------------------------------------------------------------
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------------------------------------------------
# Fixed config: small model size, small dataset
# -----------------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

MODELS=("miudiff" "unitddpm" "cyclediffusion" "unsb")
MODEL=${MODELS[$TASK_ID]}

DATA_RANGE="1,7"    # small dataset
TOTAL_STEPS=2000000 # 2M steps total across all stages

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "DATA_RANGE=${DATA_RANGE} (small dataset)"
echo "TOTAL_STEPS=${TOTAL_STEPS}"

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
PROJECT_ROOT=I2I-Stain-Zoo

DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

BASE=/work2/bz66izin-VSproject/Outputs
MODEL_DIR="${BASE}/${MODEL}/results/data_small/model_small"

mkdir -p "${MODEL_DIR}"
echo "Output directory: ${MODEL_DIR}"

# -----------------------------------------------------------------------
# Per-model training
#
# 2M steps allocated proportionally across stages:
#
#   MIUDiff    (3 stages): 1,500,000 + 250,000 + 250,000 = 2,000,000
#              Small size: --miu_base_channels 32 --miu_channel_mult 1,2,4,4
#                          (~11.4M A→B params, consistent with train_all_54.sh)
#
#   UNIT-DDPM  (2 stages): 1,500,000 + 500,000 = 2,000,000
#              Small size: --unitddpm_base_channels 64 --unitddpm_channel_mult 1,2,2,4
#                          --unitddpm_num_res_blocks 1 (~9.5M A→B params)
#              Stage 2 uses --unitddpm_lambda_struct 1.0 for structural supervision.
#
#   CycleDiffusion (1 stage): 2,000,000
#              Small size: --cd_base_channels 48 --cd_channel_mult 1,2,2,4
#                          --cd_num_res_blocks 1 (~10.7M A→B params)
#
#   UNSB       (1 stage): 2,000,000
#              Small size: --unsb_base_channels 64 --unsb_channel_mult 1,2,2,4
#                          --unsb_num_res_blocks 1 (~9.5M A→B params)
# -----------------------------------------------------------------------

case "${MODEL}" in

    # ------------------------------------------------------------------
    miudiff)
        # Stage 1 — unconditional DDPM on domain B (builds target prior)
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage pretrain \
            --steps 1500000 \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels 32 \
            --miu_channel_mult "1,2,4,4" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage1/"

        # Stage 2 — conditional finetune with structural loss
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage finetune \
            --miu_lambda_struct 1.0 \
            --steps 250000 \
            --miu_init_ckpt "${MODEL_DIR}/stage1/checkpoints/step_1500000.pt" \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels 32 \
            --miu_channel_mult "1,2,4,4" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage2/"

        # Stage 3 — finetune with PCL refinement
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage finetune \
            --miu_pcl \
            --miu_lambda_struct 1.0 \
            --steps 250000 \
            --miu_init_ckpt "${MODEL_DIR}/stage2/checkpoints/step_250000.pt" \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels 32 \
            --miu_channel_mult "1,2,4,4" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage3/"
        ;;

    # ------------------------------------------------------------------
    unitddpm)
        # Stage 1 — unconditional DDPM on domain B (builds target prior)
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model unitddpm \
            --unitddpm_stage pretrain \
            --steps 1500000 \
            --data_range "${DATA_RANGE}" \
            --unitddpm_base_channels 64 \
            --unitddpm_channel_mult "1,2,2,4" \
            --unitddpm_num_res_blocks 1 \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage1/"

        # Stage 2 — conditional A→B finetune with structural supervision
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model unitddpm \
            --unitddpm_stage finetune \
            --unitddpm_lambda_struct 1.0 \
            --steps 500000 \
            --unitddpm_init_ckpt "${MODEL_DIR}/stage1/checkpoints/step_1500000.pt" \
            --data_range "${DATA_RANGE}" \
            --unitddpm_base_channels 64 \
            --unitddpm_channel_mult "1,2,2,4" \
            --unitddpm_num_res_blocks 1 \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage2/"
        ;;

    # ------------------------------------------------------------------
    cyclediffusion)
        # Single stage — joint unconditional DDPM on both domains
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model cyclediffusion \
            --steps "${TOTAL_STEPS}" \
            --data_range "${DATA_RANGE}" \
            --cd_base_channels 48 \
            --cd_channel_mult "1,2,2,4" \
            --cd_num_res_blocks 1 \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/"
        ;;

    # ------------------------------------------------------------------
    unsb)
        # Single stage — score network + adversarial discriminator
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model unsb \
            --steps "${TOTAL_STEPS}" \
            --data_range "${DATA_RANGE}" \
            --unsb_base_channels 64 \
            --unsb_channel_mult "1,2,2,4" \
            --unsb_num_res_blocks 1 \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/"
        ;;

    *)
        echo "Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Done: ${MODEL} finished successfully."
