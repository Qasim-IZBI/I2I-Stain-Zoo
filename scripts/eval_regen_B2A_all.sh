#!/bin/bash
#SBATCH --job-name=i2i_regen_eval
#SBATCH --output=logs_regen_eval/regen_%A_%a.out
#SBATCH --error=logs_regen_eval/regen_%A_%a.err

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0   # no GPU needed: precomputed-A' mode is CPU-only
#SBATCH --array=0-17   # 18 jobs = 6 models x 3 data-sizes  (model_small only)

# Computes cycle regen error MAE for all 6 models using precomputed A' tiles from
# infer_B2A_all.sh.  Calls evaluation.py --metric regen_error --path_A_regen so
# no model inference is re-run (works for all models including cyclediffusion).
#
# Prerequisites: infer_small_models.sh and infer_B2A_all.sh must have run first.
#
# Outputs per config:
#   ${REGEN_EVAL_DIR}/${MODEL}/data_${DATASIZE}/model_small/
#     regen_mae.csv          ← per-tile MAE + MEAN row
#     overlays/heatmaps/     ← hot-colormap error heatmaps
#     overlays/overlays/     ← 50/50 blend of heatmap and original A tile
#     overlays/error_npy/    ← raw [H,W] float32 error maps for calibration

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_regen_eval

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# Axes  (must match infer_small_models.sh and infer_B2A_all.sh)
# ─────────────────────────────────────────────────────────────────────────────
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

DATA_ID=$(( TASK_ID / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}
SIZE=small

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed paths  (must match infer_small_models.sh / infer_B2A_all.sh)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap
INFER_BASE=/work2/bz66izin-VSproject/inference

# Original testA tiles (WSI subfolder structure); matched against A' tiles by stem
TEST_A="${DATA_DIR}/testA/tiles/testA"

# Per-config inference directories
MODEL_INFER_DIR=${INFER_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}
IN_A_REGEN="${MODEL_INFER_DIR}/inference_B2A"   # precomputed A' tiles

# Eval output
REGEN_EVAL_DIR=/work2/bz66izin-VSproject/Eval_Regen
OUT_DIR="${REGEN_EVAL_DIR}/${MODEL}/data_${DATASIZE}/model_${SIZE}"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d "${IN_A_REGEN}" ]; then
    echo "[ERROR] B→A inference output not found: ${IN_A_REGEN}"
    echo "        Run infer_B2A_all.sh first."
    exit 1
fi

n_regen=$(find "${IN_A_REGEN}" -maxdepth 1 -name "*.tif" | wc -l)
if [ "${n_regen}" -eq 0 ]; then
    echo "[ERROR] No A' tiles found in ${IN_A_REGEN}"
    exit 1
fi
echo "[CHECK] A' tiles in ${IN_A_REGEN}: ${n_regen}"

# Skip if already done
if [ -f "${OUT_DIR}/regen_mae.csv" ]; then
    n_rows=$(wc -l < "${OUT_DIR}/regen_mae.csv")
    echo "[CHECK] regen_mae.csv already has ${n_rows} rows in ${OUT_DIR}"
    # n_rows includes header + MEAN row, so n_data_rows = n_rows - 2
    n_data=$(( n_rows - 2 ))
    if [ "${n_data}" -ge "${n_regen}" ]; then
        echo "[SKIP]  regen_mae.csv is complete (${n_data} tiles). Exiting."
        exit 0
    fi
    echo "[WARN]  Partial CSV detected (${n_data}/${n_regen} tiles). Re-running."
fi

mkdir -p "${OUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Run regen error evaluation (precomputed-A' mode — no GPU / model needed)
# ─────────────────────────────────────────────────────────────────────────────
run_cmd python "${PROJECT_ROOT}/evaluation.py" \
    --metric        regen_error \
    --path_A        "${TEST_A}" \
    --path_A_regen  "${IN_A_REGEN}" \
    --overlay_dir   "${OUT_DIR}/overlays" \
    --save_error_npy \
    --save_csv      "${OUT_DIR}/regen_mae.csv"

echo ""
echo "Done. Results saved to ${OUT_DIR}/"
echo "  regen_mae.csv        — per-tile MAE + dataset mean"
echo "  overlays/heatmaps/   — error heatmaps"
echo "  overlays/overlays/   — blended overlays"
echo "  overlays/error_npy/  — raw float32 error maps"
