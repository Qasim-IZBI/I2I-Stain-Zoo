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
conda activate i2istain

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

mkdir -p logs_eval

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these before submitting
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
PROJECT_ROOT=I2I-Stain-Zoo

# Real target-domain B images (used for FID reference distribution and
# as paired ground-truth for SSIM / patch-SSIM / LPIPS).
# Recursive walk, so subdirectory layout (001/images/, 002/images/, …) is fine.
# Paired metrics match tiles by basename, so only tiles present in both the
# real directory AND the inference output will be scored.
TEST_B_REAL="${DATA_DIR}/QP_SR/tiles/testB"

# Root where all inference outputs live (written by infer_all_54.sh)
OUTPUT_BASE=/work2/bz66izin-VSproject/Outputs

# Root where per-metric CSVs for each configuration will be saved.
# Layout: ${EVAL_DIR}/${model}/data_${datasize}/model_${size}/{metric}.csv
EVAL_DIR=/work2/bz66izin-VSproject/Eval

# patch-SSIM settings
PATCH_SIZE=64
PATCHES_PER_IMAGE=16

# ─────────────────────────────────────────────────────────────────────────────
# Job decomposition  (same ordering as infer_all_54.sh)
# ─────────────────────────────────────────────────────────────────────────────
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
echo "MODEL=${MODEL}  MODEL_SIZE=${SIZE}  DATA_SIZE=${DATASIZE}"

# ─────────────────────────────────────────────────────────────────────────────
# Resolve inference output directory
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR="${OUTPUT_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}"
INFER_DIR="${MODEL_DIR}/inference"

if [ ! -d "${INFER_DIR}" ]; then
    echo "[ERROR] Inference directory not found: ${INFER_DIR}"
    echo "        Run infer_all_54.sh first for this configuration."
    exit 1
fi

# Quick sanity-check: at least one image present
n_images=$(find "${INFER_DIR}" -maxdepth 1 \
    \( -name '*.tif' -o -name '*.tiff' -o -name '*.png' -o -name '*.jpg' \) \
    | wc -l)
if [ "${n_images}" -eq 0 ]; then
    echo "[ERROR] No images found in ${INFER_DIR} — inference output missing."
    exit 1
fi
echo "Found ${n_images} generated images in ${INFER_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Per-configuration output directory
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR="${EVAL_DIR}/${MODEL}/data_${DATASIZE}/model_${SIZE}"
mkdir -p "${OUT_DIR}"

PYTHON="python ${PROJECT_ROOT}/evaluation.py"

# ─────────────────────────────────────────────────────────────────────────────
# 1. FID — Inception (unpaired, distribution-level)
# ─────────────────────────────────────────────────────────────────────────────
echo "===== FID (Inception) ====="
run_cmd ${PYTHON} \
    --metric     fid \
    --path_real  "${TEST_B_REAL}" \
    --path_fake  "${INFER_DIR}" \
    --backend    inception \
    --device     cuda \
    --save_csv   "${OUT_DIR}/fid_inception.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 2. FID — DINO (unpaired, DINOv2 feature space)
# ─────────────────────────────────────────────────────────────────────────────
echo "===== FID (DINO) ====="
run_cmd ${PYTHON} \
    --metric     fid \
    --path_real  "${TEST_B_REAL}" \
    --path_fake  "${INFER_DIR}" \
    --backend    dino \
    --device     cuda \
    --save_csv   "${OUT_DIR}/fid_dino.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 3. SSIM (paired by filename)
# ─────────────────────────────────────────────────────────────────────────────
echo "===== SSIM ====="
run_cmd ${PYTHON} \
    --metric     ssim \
    --path_real  "${TEST_B_REAL}" \
    --path_fake  "${INFER_DIR}" \
    --device     cuda \
    --save_csv   "${OUT_DIR}/ssim.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Patch-SSIM (paired by filename)
# ─────────────────────────────────────────────────────────────────────────────
echo "===== Patch-SSIM ====="
run_cmd ${PYTHON} \
    --metric             patch_ssim \
    --path_real          "${TEST_B_REAL}" \
    --path_fake          "${INFER_DIR}" \
    --patch_size         "${PATCH_SIZE}" \
    --patches_per_image  "${PATCHES_PER_IMAGE}" \
    --device             cuda \
    --save_csv           "${OUT_DIR}/patch_ssim.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 5. LPIPS (paired by filename, VGG16 perceptual distance — lower is better)
# ─────────────────────────────────────────────────────────────────────────────
echo "===== LPIPS ====="
run_cmd ${PYTHON} \
    --metric     lpips \
    --path_real  "${TEST_B_REAL}" \
    --path_fake  "${INFER_DIR}" \
    --device     cuda \
    --save_csv   "${OUT_DIR}/lpips.csv"

echo ""
echo "All metrics saved to ${OUT_DIR}/"
echo "Run aggregate_eval.py after all 54 jobs finish to produce eval_summary.csv."
