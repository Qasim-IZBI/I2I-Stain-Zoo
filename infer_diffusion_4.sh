#!/bin/bash
#SBATCH --job-name=diff4_infer
#SBATCH --output=logs_diffusion/infer_%A_%a.out
#SBATCH --error=logs_diffusion/infer_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
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
    local latest="" max_step=-1 step f
    for f in "${dir}"/step_*.pt; do
        [ -f "$f" ] || continue
        step=$(basename "$f" .pt | sed 's/step_//')
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if [ "$step" -gt "$max_step" ]; then
            max_step="$step"
            latest="$f"
        fi
    done
    echo "$latest"
}

# -----------------------------
# Axes — mirrors train_diffusion_4.sh
# -----------------------------
MODELS=("miudiff" "unitddpm" "cyclediffusion" "unsb")

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL=${MODELS[$TASK_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"

# -----------------------------
# Fixed paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
PROJECT_ROOT=I2I-Stain-Zoo

TEST_A="${DATA_DIR}/QP_HE/tiles/testA"

# WSI subfolders to infer on (mirrors infer_all_54.sh)
DATA_RANGE="1,2"

# DDIM denoising steps (200–300 typical)
DDIM_STEPS=200

BASE=/work2/bz66izin-VSproject/Outputs
MODEL_DIR="${BASE}/${MODEL}/results/data_small/model_small"

echo "MODEL_DIR=${MODEL_DIR}"

# -----------------------------
# Per-model inference
# Config (arch flags) is auto-restored from the checkpoint — no need to repeat them.
# -----------------------------
case "${MODEL}" in

    miudiff)
        # Inference uses the stage-3 (finetune + PCL) checkpoint
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
            --miu_steps  "${DDIM_STEPS}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    unitddpm)
        # Inference uses the stage-2 (conditional finetune) checkpoint
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/stage2/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No stage-2 checkpoint found in ${MODEL_DIR}/stage2/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model          unitddpm \
            --unitddpm_stage finetune \
            --direction      A2B \
            --data           "${TEST_A}" \
            --data_range     "${DATA_RANGE}" \
            --ckpt           "${CKPT}" \
            --unitddpm_steps "${DDIM_STEPS}" \
            --outdir         "${MODEL_DIR}/inference"
        ;;

    cyclediffusion)
        # Single-stage model — checkpoint is directly under MODEL_DIR/checkpoints
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No checkpoint found in ${MODEL_DIR}/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      cyclediffusion \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --cd_steps   "${DDIM_STEPS}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    unsb)
        # Single-stage model — checkpoint is directly under MODEL_DIR/checkpoints
        CKPT=$(find_latest_ckpt "${MODEL_DIR}/checkpoints")

        if [ -z "${CKPT}" ]; then
            echo "[ERROR] No checkpoint found in ${MODEL_DIR}/checkpoints — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      unsb \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --unsb_steps "${DDIM_STEPS}" \
            --outdir     "${MODEL_DIR}/inference"
        ;;

    *)
        echo "[ERROR] Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Done. Results saved to ${MODEL_DIR}/inference"
