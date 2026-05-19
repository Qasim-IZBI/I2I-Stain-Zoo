#!/bin/bash
#SBATCH --job-name=i2i_ens_regen_stats
#SBATCH --output=logs_ensemble/regen_stats_%A_%a.out
#SBATCH --error=logs_ensemble/regen_stats_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-29   # 30 jobs = 6 models × 5 WSIs

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
# 2D decomposition: 6 models × 5 WSIs
# Layout: tasks 0–4  → cyclegan        WSI 1–5
#         tasks 5–9  → unit             WSI 1–5
#         tasks 10–14 → munit           WSI 1–5
#         tasks 15–19 → dclgan          WSI 1–5
#         tasks 20–24 → uvcgan          WSI 1–5
#         tasks 25–29 → cyclediffusion  WSI 1–5
# -----------------------------
N_WSIS=5
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))   # 0 … 5
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))      # 0 … 4

WSI_NUM=$(( WSI_IDX + 1 ))                       # 1 … 5
DATA_RANGE="${WSI_NUM},${WSI_NUM}"

# Per-model config
# Index:        0          1      2       3        4        5
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL="${MODELS[$MODEL_IDX]}"
MODEL_SIZE="${MODEL_SIZES[$MODEL_IDX]}"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"
IN_DIR="${ENSEMBLE_ROOT}/inference_B2A"
OUT_DIR="${ENSEMBLE_ROOT}/regen_stats"

WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}  WSI=${WSI_FOLDER}"
echo "Input  : ${IN_DIR}"
echo "Output : ${OUT_DIR}"
echo "Range  : ${DATA_RANGE}"

# -----------------------------
# Pre-flight: B2A inference output must exist
# -----------------------------
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -d "${IN_DIR}"/model_* 2>/dev/null)" ]; then
    echo "[ERROR] B2A inference output missing or contains no model_* dirs: ${IN_DIR}"
    echo "        Run infer_ensemble_${MODEL}_B2A.sh first."
    exit 1
fi

# Per-WSI skip guard
SUMMARY_FILE="${OUT_DIR}/${MODEL}/summary_wsi${WSI_FOLDER}.json"
if [ -f "${SUMMARY_FILE}" ]; then
    echo "[SKIP] Already completed: ${SUMMARY_FILE}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# Reuses uncertainty.py — applied to inference_B2A/ it computes mean prediction
# (mean_rgb/) and variance maps (raw_npy/, norm_npy/, heatmaps/) across all
# B→A ensemble members for the given WSI.
run_cmd python I2I-Stain-Zoo/uncertainty.py \
    --model      "${MODEL}" \
    --data       "${IN_DIR}" \
    --output     "${OUT_DIR}" \
    --data_range "${DATA_RANGE}" \
    --log-compress \
    --lower-percentile 1 \
    --upper-percentile 99

echo "Done. Regen stats for ${MODEL} WSI ${WSI_FOLDER} saved to ${OUT_DIR}"
