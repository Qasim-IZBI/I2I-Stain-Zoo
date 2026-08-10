#!/bin/bash
#SBATCH --job-name=i2i_infer_small
#SBATCH --output=logs_infer_small/infer_%A_%a.out
#SBATCH --error=logs_infer_small/infer_%A_%a.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-17   # 18 jobs = 6 models x 3 data-sizes  (model_small only, no miudiff)

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u   # conda activate scripts may reference unset variables (e.g. QT_XCB_GL_INTEGRATION)
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p logs_infer_small

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
# Helper: count .tif tiles produced by inference (flat output dir)
# -----------------------------
count_output_tiles() {
    local outdir="$1"
    if [ -d "$outdir" ]; then
        find "$outdir" -maxdepth 1 -name "*.tif" | wc -l
    else
        echo 0
    fi
}

# -----------------------------
# Helper: count source .tif tiles for a given data_range
# -----------------------------
count_source_tiles() {
    local datadir="$1"
    local start="$2"
    local end="$3"
    local total=0
    local folder n
    for i in $(seq "$start" "$end"); do
        folder=$(printf "%03d" "$i")
        if [ -d "${datadir}/${folder}/images" ]; then
            n=$(find "${datadir}/${folder}/images" -name "*.tif" | wc -l)
            total=$(( total + n ))
        fi
    done
    echo "$total"
}

# -----------------------------
# Axes  (model_small only; 6 models without miudiff; 3 data-sizes)
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Layout: task 0-5 → data_small, 6-11 → data_medium, 12-17 → data_large
DATA_ID=$(( TASK_ID / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}
SIZE=small   # fixed: this script targets model_small only

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# -----------------------------
# Fixed paths
# -----------------------------
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/
PROJECT_ROOT=I2I-Stain-Zoo

TEST_A="${DATA_DIR}testA/tiles/testA"

# WSI subfolders to infer on (e.g. "1,2" → 001/images/ and 002/images/)
DATA_RANGE="1,5"
RANGE_START=1
RANGE_END=5

# Checkpoint step to load (1M)
TARGET_STEP=1000000

# CycleDiffusion DDIM steps
CD_STEPS=200

# Source of trained weights (unchanged output tree from training)
TRAIN_BASE=/work2/bz66izin-VSproject/Outputs_noamp

# Separate output root for this inference run
INFER_BASE=/work2/bz66izin-VSproject/inference

# -----------------------------
# Pre-flight: check if inference is already complete
# -----------------------------
check_already_done() {
    local outdir="$1"
    local n_src n_out
    n_src=$(count_source_tiles "${TEST_A}" "${RANGE_START}" "${RANGE_END}")
    n_out=$(count_output_tiles "${outdir}")
    echo "[CHECK] Source tiles in testA (range ${RANGE_START}-${RANGE_END}): ${n_src}"
    echo "[CHECK] Output tiles already in ${outdir}: ${n_out}"
    if [ "${n_src}" -gt 0 ] && [ "${n_src}" -eq "${n_out}" ]; then
        echo "[SKIP]  Output directory is complete (${n_out}/${n_src} tiles present). Exiting."
        exit 0
    fi
    if [ "${n_out}" -gt 0 ]; then
        echo "[WARN]  Partial output detected (${n_out}/${n_src} tiles). Re-running inference."
    fi
}

# -----------------------------
# Resolve checkpoint and run inference
# -----------------------------
case "${MODEL}" in

    cyclegan | unit | munit | dclgan)
        MODEL_DIR=${TRAIN_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/checkpoints/step_${TARGET_STEP}.pt"
        OUT_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] Checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        check_already_done "${OUT_DIR}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      "${MODEL}" \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --outdir     "${OUT_DIR}"
        ;;

    uvcgan)
        # Inference uses the stage-2 (finetune) checkpoint
        TARGET_STEP_STAGE2=750000
        MODEL_DIR=${TRAIN_BASE}/uvcgan/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/stage2/checkpoints/step_${TARGET_STEP_STAGE2}.pt"
        OUT_DIR=${INFER_BASE}/uvcgan/results/data_${DATASIZE}/model_${SIZE}/inference

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] stage-2 checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        check_already_done "${OUT_DIR}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      uvcgan \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --outdir     "${OUT_DIR}"
        ;;

    cyclediffusion)
        MODEL_DIR=${TRAIN_BASE}/cyclediffusion/results/data_${DATASIZE}/model_${SIZE}
        CKPT="${MODEL_DIR}/checkpoints/step_${TARGET_STEP}.pt"
        OUT_DIR=${INFER_BASE}/cyclediffusion/results/data_${DATASIZE}/model_${SIZE}/inference

        if [ ! -f "${CKPT}" ]; then
            echo "[ERROR] Checkpoint not found: ${CKPT} — skipping."
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        check_already_done "${OUT_DIR}"

        run_cmd python "${PROJECT_ROOT}/inference.py" \
            --model      cyclediffusion \
            --direction  A2B \
            --data       "${TEST_A}" \
            --data_range "${DATA_RANGE}" \
            --ckpt       "${CKPT}" \
            --cd_steps   "${CD_STEPS}" \
            --outdir     "${OUT_DIR}"
        ;;

    *)
        echo "[ERROR] Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Done. Results saved to ${OUT_DIR}"
