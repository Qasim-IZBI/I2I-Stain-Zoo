#!/bin/bash
#SBATCH --job-name=i2i_ens_regen_err
#SBATCH --output=logs_ensemble/regen_error_%A_%a.out
#SBATCH --error=logs_ensemble/regen_error_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 models

# Computes |A - mean(B2A)| per tile using evaluation.py's precomputed-A' mode.
# Inputs:
#   --path_A      : original testA tiles (ground truth HE)
#   --path_A_regen: mean B2A prediction from compute_ensemble_regen_stats.sh
#                   (regen_stats/{model}/mean_rgb/)
# Outputs per model (written to {ENSEMBLE_ROOT}/regen_error/):
#   heatmaps/   : per-tile error heatmap PNGs
#   error_npy/  : per-tile error maps as .npy (consumed by uncertainty_calibration.py)

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble

# -----------------------------
# Per-model config
# Index:        0          1      2       3        4        5
# -----------------------------
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SIZE="${MODEL_SIZES[$SLURM_ARRAY_TASK_ID]}"

TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"
MEAN_B2A="${ENSEMBLE_ROOT}/regen_stats/${MODEL}/mean_rgb"
OVERLAY_DIR="${ENSEMBLE_ROOT}/regen_error"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}"
echo "path_A       : ${TEST_A}"
echo "path_A_regen : ${MEAN_B2A}"
echo "Output       : ${OVERLAY_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${MEAN_B2A}" ] || [ -z "$(ls -A "${MEAN_B2A}" 2>/dev/null)" ]; then
    echo "[ERROR] Mean B2A directory missing or empty: ${MEAN_B2A}"
    echo "        Run compute_ensemble_regen_stats.sh first."
    exit 1
fi

# Skip guard
if [ -d "${OVERLAY_DIR}/error_npy" ] && [ -n "$(ls -A "${OVERLAY_DIR}/error_npy" 2>/dev/null)" ]; then
    echo "[SKIP] error_npy already populated: ${OVERLAY_DIR}/error_npy"
    exit 0
fi

mkdir -p "${OVERLAY_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# No --device flag needed: compute_regen_error_precomputed runs on CPU only
# (no model inference — pure pixel-wise MAE between A and mean B2A).
run_cmd python I2I-Stain-Zoo/evaluation.py \
    --metric      regen_error \
    --path_A      "${TEST_A}" \
    --path_A_regen "${MEAN_B2A}" \
    --overlay_dir  "${OVERLAY_DIR}" \
    --save_error_npy

echo "Done. Error maps for ${MODEL} saved to ${OVERLAY_DIR}"
