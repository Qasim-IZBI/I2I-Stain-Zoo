#!/bin/bash
#SBATCH --job-name=i2i_eval_all
#SBATCH --output=logs_eval/eval_%A_%a.out
#SBATCH --error=logs_eval/eval_%A_%a.err

#SBATCH --time=4:00:00
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
set +u   # conda activate scripts may reference unset variables (e.g. QT_XCB_GL_INTEGRATION)
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

mkdir -p logs_eval

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
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches infer_all_54.sh
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
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
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/
PROJECT_ROOT=I2I-Stain-Zoo

# Real target-domain (B) images — used as path_real for all metrics.
# For SSIM/LPIPS (paired), filenames here must match inference output filenames.
TEST_B="${DATA_DIR}testB/tiles/testB"

# Minimum tissue fraction to include a tile (0 = all tiles, 0.1 = at least 10% tissue).
# Masks are auto-detected from the tile structure (images/ → masks/ sibling).
# Set to 0 to disable tissue filtering.
MIN_TISSUE_FRACTION=0.1

# Inference output produced by the corresponding inference run
INFER_BASE=/work2/bz66izin-VSproject/inference
FAKE_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/inference

# Evaluation results saved here
EVAL_BASE=/work2/bz66izin-VSproject/evaluation
OUT_DIR=${EVAL_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Inference output must exist
if [ ! -d "${FAKE_DIR}" ] || [ -z "$(ls -A "${FAKE_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Inference output not found or empty: ${FAKE_DIR} — skipping."
    exit 1
fi

# 2. Real images must exist
if [ ! -d "${TEST_B}" ]; then
    echo "[ERROR] Real image directory not found: ${TEST_B} — skipping."
    exit 1
fi

# 3. Skip if all metric CSVs already exist
if [ -f "${OUT_DIR}/fid.csv" ] && \
   [ -f "${OUT_DIR}/patch_ssim.csv" ] && \
   [ -f "${OUT_DIR}/lpips.csv" ]; then
    echo "[SKIP] All metric CSVs already present in ${OUT_DIR}. Exiting."
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "Inference dir    : ${FAKE_DIR}"
echo "Real image dir   : ${TEST_B}"
echo "Output dir       : ${OUT_DIR}"
echo "Min tissue frac  : ${MIN_TISSUE_FRACTION}"

# -----------------------------
# FID  (distribution-level, unpaired — Inception backend)
# -----------------------------
if [ ! -f "${OUT_DIR}/fid.csv" ]; then
    echo "--- Running FID ---"
    run_cmd python "${PROJECT_ROOT}/evaluation.py" \
        --metric               fid \
        --path_real            "${TEST_B}" \
        --path_fake            "${FAKE_DIR}" \
        --device               cuda \
        --min_tissue_fraction  "${MIN_TISSUE_FRACTION}" \
        --save_csv             "${OUT_DIR}/fid.csv"
else
    echo "[SKIP] fid.csv already exists."
fi

# -----------------------------
# Patch-SSIM  (paired by filename, patch-based)
# -----------------------------
if [ ! -f "${OUT_DIR}/patch_ssim.csv" ]; then
    echo "--- Running Patch-SSIM ---"
    run_cmd python "${PROJECT_ROOT}/evaluation.py" \
        --metric               patch_ssim \
        --path_real            "${TEST_B}" \
        --path_fake            "${FAKE_DIR}" \
        --patch_size           64 \
        --patches_per_image    16 \
        --device               cuda \
        --min_tissue_fraction  "${MIN_TISSUE_FRACTION}" \
        --save_csv             "${OUT_DIR}/patch_ssim.csv"
else
    echo "[SKIP] patch_ssim.csv already exists."
fi

# -----------------------------
# LPIPS  (paired by filename, VGG16)
# -----------------------------
if [ ! -f "${OUT_DIR}/lpips.csv" ]; then
    echo "--- Running LPIPS ---"
    run_cmd python "${PROJECT_ROOT}/evaluation.py" \
        --metric               lpips \
        --path_real            "${TEST_B}" \
        --path_fake            "${FAKE_DIR}" \
        --device               cuda \
        --min_tissue_fraction  "${MIN_TISSUE_FRACTION}" \
        --save_csv             "${OUT_DIR}/lpips.csv"
else
    echo "[SKIP] lpips.csv already exists."
fi

echo "Done. Results saved to ${OUT_DIR}"
