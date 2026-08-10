#!/bin/bash
#SBATCH --job-name=i2i_infer_B2A
#SBATCH --output=logs_infer_B2A/infer_%A_%a.out
#SBATCH --error=logs_infer_B2A/infer_%A_%a.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-17   # 18 jobs = 6 models x 3 data-sizes  (model_small only)

# Runs B→A reverse inference on the A→B translated tiles produced by
# infer_small_models.sh, giving A' tiles needed for regen-error evaluation.
# Input:  ${INFER_BASE}/${MODEL}/.../inference/          (flat B' .tif tiles)
# Output: ${INFER_BASE}/${MODEL}/.../inference_B2A/      (flat A' .tif tiles)
# Filenames are preserved so evaluation.py can pair A vs A' by stem.

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs_infer_B2A

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

count_output_tiles() {
    local outdir="$1"
    if [ -d "$outdir" ]; then
        find "$outdir" -maxdepth 1 -name "*.tif" | wc -l
    else
        echo 0
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Axes  (must match infer_small_models.sh)
# ─────────────────────────────────────────────────────────────────────────────
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

DATA_ID=$(( TASK_ID / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}
SIZE=small   # fixed: model_small only

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed paths  (must match infer_small_models.sh)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT=I2I-Stain-Zoo
TRAIN_BASE=/work2/bz66izin-VSproject/Outputs_noamp
INFER_BASE=/work2/bz66izin-VSproject/inference

TARGET_STEP=1000000
CD_STEPS=200

# ─────────────────────────────────────────────────────────────────────────────
# Resolve checkpoint and run B→A inference
# ─────────────────────────────────────────────────────────────────────────────
case "${MODEL}" in

    cyclegan | unit | munit | dclgan)
        MODEL_DIR=${TRAIN_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/checkpoints/step_${TARGET_STEP}.pt"

        IN_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference
        OUT_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference_B2A

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] Checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        if [ ! -d "${IN_DIR}" ]; then
            echo "[ERROR] A→B inference output not found: ${IN_DIR}"
            echo "        Run infer_small_models.sh first."
            exit 1
        fi

        n_in=$(count_output_tiles "${IN_DIR}")
        n_out=$(count_output_tiles "${OUT_DIR}")
        echo "[CHECK] B' tiles in ${IN_DIR}: ${n_in}"
        echo "[CHECK] A' tiles already in ${OUT_DIR}: ${n_out}"
        if [ "${n_in}" -gt 0 ] && [ "${n_in}" -eq "${n_out}" ]; then
            echo "[SKIP]  Output complete (${n_out}/${n_in}). Exiting."
            exit 0
        fi

        echo "Checkpoint: ${CKPT}"
        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model     "${MODEL}" \
            --direction B2A \
            --data      "${IN_DIR}" \
            --ckpt      "${CKPT}" \
            --outdir    "${OUT_DIR}"
        ;;

    uvcgan)
        TARGET_STEP_STAGE2=750000
        MODEL_DIR=${TRAIN_BASE}/uvcgan/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/stage2/checkpoints/step_${TARGET_STEP_STAGE2}.pt"

        IN_DIR=${INFER_BASE}/uvcgan/results/data_${DATASIZE}/model_${SIZE}/inference
        OUT_DIR=${INFER_BASE}/uvcgan/results/data_${DATASIZE}/model_${SIZE}/inference_B2A

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] stage-2 checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        if [ ! -d "${IN_DIR}" ]; then
            echo "[ERROR] A→B inference output not found: ${IN_DIR}"
            echo "        Run infer_small_models.sh first."
            exit 1
        fi

        n_in=$(count_output_tiles "${IN_DIR}")
        n_out=$(count_output_tiles "${OUT_DIR}")
        echo "[CHECK] B' tiles in ${IN_DIR}: ${n_in}"
        echo "[CHECK] A' tiles already in ${OUT_DIR}: ${n_out}"
        if [ "${n_in}" -gt 0 ] && [ "${n_in}" -eq "${n_out}" ]; then
            echo "[SKIP]  Output complete (${n_out}/${n_in}). Exiting."
            exit 0
        fi

        echo "Checkpoint: ${CKPT}"
        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model     uvcgan \
            --direction B2A \
            --data      "${IN_DIR}" \
            --ckpt      "${CKPT}" \
            --outdir    "${OUT_DIR}"
        ;;

    cyclediffusion)
        MODEL_DIR=${TRAIN_BASE}/cyclediffusion/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/checkpoints/step_${TARGET_STEP}.pt"

        IN_DIR=${INFER_BASE}/cyclediffusion/results/data_${DATASIZE}/model_${SIZE}/inference
        OUT_DIR=${INFER_BASE}/cyclediffusion/results/data_${DATASIZE}/model_${SIZE}/inference_B2A

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] Checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        if [ ! -d "${IN_DIR}" ]; then
            echo "[ERROR] A→B inference output not found: ${IN_DIR}"
            echo "        Run infer_small_models.sh first."
            exit 1
        fi

        n_in=$(count_output_tiles "${IN_DIR}")
        n_out=$(count_output_tiles "${OUT_DIR}")
        echo "[CHECK] B' tiles in ${IN_DIR}: ${n_in}"
        echo "[CHECK] A' tiles already in ${OUT_DIR}: ${n_out}"
        if [ "${n_in}" -gt 0 ] && [ "${n_in}" -eq "${n_out}" ]; then
            echo "[SKIP]  Output complete (${n_out}/${n_in}). Exiting."
            exit 0
        fi

        echo "Checkpoint: ${CKPT}"
        # B→A: DDIM-invert with eps_B, decode with eps_A
        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model     cyclediffusion \
            --direction B2A \
            --data      "${IN_DIR}" \
            --ckpt      "${CKPT}" \
            --cd_steps  "${CD_STEPS}" \
            --outdir    "${OUT_DIR}"
        ;;

    *)
        echo "[ERROR] Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Done. A' tiles saved to ${OUT_DIR}"
echo "Next: compute regen error with evaluation.py --metric ssim/lpips/patch_ssim"
echo "      using --path_real <original testA tiles> and --path_fake ${OUT_DIR}"
