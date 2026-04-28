#!/bin/bash
#SBATCH --job-name=i2i_infer_all
#SBATCH --output=logs_infer/infer_%A_%a.out
#SBATCH --error=logs_infer/infer_%A_%a.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
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

mkdir -p logs_infer

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
# Helper: find the highest-step checkpoint in a directory
# -----------------------------
find_latest_ckpt() {
    local dir="$1"
    local latest=""
    local max_step=-1
    for f in "${dir}"/step_*.pt; do
        [ -f "$f" ] || continue
        step=$(basename "$f" .pt | sed 's/step_//')
        if [ "$step" -gt "$max_step" ]; then
            max_step="$step"
            latest="$f"
        fi
    done
    echo "$latest"
}

# -----------------------------
# Axes  (same decomposition as train_all_54.sh)
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "miudiff" "uvcgan")
SIZES=("small" "medium" "large")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

SIZE_ID=$(( TASK_ID / 18 ))
DATA_ID=$(( (TASK_ID % 18) / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
SIZE=${SIZES[$SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# -----------------------------
# Fixed paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
PROJECT_ROOT=I2I-Stain-Zoo

TEST_A="${DATA_DIR}/QP_HE/tiles/testA"

# WSI subfolders to infer on (e.g. "1,2" → 001/images/ and 002/images/)
DATA_RANGE="1,2"

# MIUDiff: number of DDIM denoising steps for inference (200-300 typical)
MIU_STEPS=200

# -----------------------------
# Resolve checkpoint and run inference
# -----------------------------
case "${MODEL}" in

    cyclegan | unit | munit | dclgan)
        MODEL_DIR=/work2/bz66izin-VSproject/Outputs/${MODEL}/results/data_${DATASIZE}/model_${SIZE}
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No checkpoint found in ${MODEL_DIR}/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      "${MODEL}" \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    miudiff)
        # Inference uses the stage-3 (finetune + PCL) checkpoint
        MODEL_DIR=/work2/bz66izin-VSproject/Outputs/miudiff/results/data_${DATASIZE}/model_${SIZE}
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/stage3/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No stage-3 checkpoint found in ${MODEL_DIR}/stage3/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      miudiff \
            --miu_stage  finetune \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --miu_steps  "${MIU_STEPS}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    uvcgan)
        # Inference uses the stage-2 (finetune) checkpoint
        MODEL_DIR=/work2/bz66izin-VSproject/Outputs/uvcgan/results/data_${DATASIZE}/model_${SIZE}
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/stage2/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No stage-2 checkpoint found in ${MODEL_DIR}/stage2/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      uvcgan \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    *)
        echo "[ERROR] Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Done. Results saved to ${MODEL_DIR}/inference"
