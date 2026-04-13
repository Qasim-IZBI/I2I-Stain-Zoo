#!/bin/bash
#SBATCH --job-name=ml_training_all
#SBATCH --output=logs_all/train_%A_%a.out
#SBATCH --error=logs_all/train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

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
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "miudiff" "uvcgan")
SIZES=("small" "medium" "large")
DATASIZES=("small" "medium" "large")
DATA_RANGES=("1,7" "1,14" "1,28")

TASK_ID=${SLURM_ARRAY_TASK_ID}

# 3 model-sizes x 3 data-sizes x 6 models
SIZE_ID=$(( TASK_ID / 18 ))
DATA_ID=$(( (TASK_ID % 18) / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
SIZE=${SIZES[$SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}
DATA_RANGE=${DATA_RANGES[$DATA_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"
echo "DATA_RANGE=${DATA_RANGE}"

# -----------------------------
# Fixed data paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/
PROJECT_ROOT=I2I-Stain-Zoo

DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

# -----------------------------
# Hyperparameters by model size
# -----------------------------
# Shared by cyclegan, unit, dclgan
COMMON_NGFS=("64" "128" "192")
COMMON_NBLOCKS=("8" "10" "9")

# unit only
UNIT_NSBLOCKS=("2" "2" "3")

# munit only
MUNIT_NGFS=("64" "128" "192")
MUNIT_NCBLOCKS=("3" "5" "4")
MUNIT_NABLOCKS=("4" "4" "4")

# miudiff only
MIU_MCHLS=("32" "64" "64")
MIU_CHLMULTS=("1,2,4,4" "1,2,2,4,4" "1,2,4,8")

# uvcgan only
UVC_NGFS=("48" "96" "128")
UVC_VBLOCKS=("6" "6" "17")
UVC_VFEATS=("96" "384" "384")

# -----------------------------
# Build command by model
# -----------------------------
case "${MODEL}" in
    cyclegan)
        NGF=${COMMON_NGFS[$SIZE_ID]}
        NBLOCK=${COMMON_NBLOCKS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/cyclegan
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        CMD=(
            python "${PROJECT_ROOT}/train.py"
            --model cyclegan
            --data_range "${DATA_RANGE}"
            --cyclegan_ngf "${NGF}"
            --cyclegan_n_blocks "${NBLOCK}"
            --dataA "${DATA_A}"
            --dataB "${DATA_B}"
            --output "${MODEL_DIR}"
            --amp
        )
        ;;

    unit)
        NGF=${COMMON_NGFS[$SIZE_ID]}
        NBLOCK=${COMMON_NBLOCKS[$SIZE_ID]}
        NSBLOCK=${UNIT_NSBLOCKS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/unit
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        CMD=(
            python "${PROJECT_ROOT}/train.py"
            --model unit
            --data_range "${DATA_RANGE}"
            --unit_ngf "${NGF}"
            --unit_n_blocks "${NBLOCK}"
            --unit_n_blocks_shared "${NSBLOCK}"
            --dataA "${DATA_A}"
            --dataB "${DATA_B}"
            --output "${MODEL_DIR}"
            --amp
        )
        ;;

    munit)
        NGF=${MUNIT_NGFS[$SIZE_ID]}
        NCBLOCK=${MUNIT_NCBLOCKS[$SIZE_ID]}
        NABLOCK=${MUNIT_NABLOCKS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/munit
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        CMD=(
            python "${PROJECT_ROOT}/train.py"
            --model munit
            --data_range "${DATA_RANGE}"
            --munit_ngf "${NGF}"
            --munit_n_content_blocks "${NCBLOCK}"
            --munit_n_adain_blocks "${NABLOCK}"
            --dataA "${DATA_A}"
            --dataB "${DATA_B}"
            --output "${MODEL_DIR}"
            --amp
        )
        ;;

    dclgan)
        NGF=${COMMON_NGFS[$SIZE_ID]}
        NBLOCK=${COMMON_NBLOCKS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/dclgan
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        CMD=(
            python "${PROJECT_ROOT}/train.py"
            --model dclgan
            --data_range "${DATA_RANGE}"
            --dclgan_ngf "${NGF}"
            --dclgan_n_blocks "${NBLOCK}"
            --dataA "${DATA_A}"
            --dataB "${DATA_B}"
            --output "${MODEL_DIR}"
            --amp
        )
        ;;

    miudiff)
        # MIUDiff requires 3 sequential stages within a single job.
        # Stage 1 (pretrain):   3,500,000 steps — trains eps_uncond on domain B
        # Stage 2 (finetune):     750,000 steps — trains eps_cond conditioned on source
        # Stage 3 (finetune+PCL): 750,000 steps — adds patch contrastive refinement
        MCHL=${MIU_MCHLS[$SIZE_ID]}
        CHLMULT=${MIU_CHLMULTS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/miudiff
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        mkdir -p "${MODEL_DIR}"
        echo "Output directory: ${MODEL_DIR}"

        # Stage 1 — unconditional DDPM pretraining on domain B
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage pretrain \
            --steps 3500000 \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels "${MCHL}" \
            --miu_channel_mult "${CHLMULT}" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage1/" \
            --amp

        # Stage 2 — conditional finetuning (eps_cond) initialised from stage 1
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage finetune \
            --steps 750000 \
            --miu_init_ckpt "${MODEL_DIR}/stage1/checkpoints/step_3500000.pt" \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels "${MCHL}" \
            --miu_channel_mult "${CHLMULT}" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage2/" \
            --amp

        # Stage 3 — finetune with patch contrastive loss (PCL) from stage 2
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model miudiff \
            --miu_stage finetune \
            --miu_pcl \
            --steps 750000 \
            --miu_init_ckpt "${MODEL_DIR}/stage2/checkpoints/step_750000.pt" \
            --data_range "${DATA_RANGE}" \
            --miu_base_channels "${MCHL}" \
            --miu_channel_mult "${CHLMULT}" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage3/" \
            --amp

        exit 0
        ;;

    uvcgan)
        # UVCGAN requires 2 sequential stages within a single job.
        # Stage 1 (pretrain):  1,250,000 steps — masked image modelling
        # Stage 2 (finetune):  3,750,000 steps — cycle-consistent translation
        NGF=${UVC_NGFS[$SIZE_ID]}
        VBLOCK=${UVC_VBLOCKS[$SIZE_ID]}
        VFEAT=${UVC_VFEATS[$SIZE_ID]}
        BASE=/work2/bz66izin-VSproject/Outputs/uvcgan
        MODEL_DIR=${BASE}/results/data_${DATASIZE}/model_${SIZE}

        mkdir -p "${MODEL_DIR}"
        echo "Output directory: ${MODEL_DIR}"

        # Stage 1 — masked image modelling pretrain
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model uvcgan \
            --uvcgan_stage pretrain \
            --steps 1250000 \
            --data_range "${DATA_RANGE}" \
            --uvcgan_ngf "${NGF}" \
            --uvcgan_vit_blocks "${VBLOCK}" \
            --uvcgan_vit_features "${VFEAT}" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage1/" \
            --amp

        # Stage 2 — cycle-consistent finetuning initialised from stage 1
        run_cmd python "${PROJECT_ROOT}/train.py" \
            --model uvcgan \
            --uvcgan_stage finetune \
            --steps 3750000 \
            --uvcgan_init_ckpt "${MODEL_DIR}/stage1/checkpoints/step_1250000.pt" \
            --data_range "${DATA_RANGE}" \
            --uvcgan_ngf "${NGF}" \
            --uvcgan_vit_blocks "${VBLOCK}" \
            --uvcgan_vit_features "${VFEAT}" \
            --dataA "${DATA_A}" \
            --dataB "${DATA_B}" \
            --output "${MODEL_DIR}/stage2/" \
            --amp

        exit 0
        ;;

    *)
        echo "Unknown model: ${MODEL}"
        exit 1
        ;;
esac

# Single-stage models (cyclegan, unit, munit, dclgan) reach here
mkdir -p "${MODEL_DIR}"
echo "Output directory: ${MODEL_DIR}"
echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
