#!/bin/bash
#SBATCH --job-name=i2i_recon_ens_A2B
#SBATCH --output=logs_recon_ensemble/recon_ens_%A_%a.out
#SBATCH --error=logs_recon_ensemble/recon_ens_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-299   # 300 jobs = 6 models × 10 ensemble members × 5 test WSIs

set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u   # conda activate scripts may reference unset variables
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_recon_ensemble

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models × 10 members × 5 WSIs = 300 jobs)
# MODEL_ID  = TASK_ID / 50        (0–5)
# MEMBER_ID = (TASK_ID % 50) / 5  (0–9 → member 01–10)
# WSI_ID    = TASK_ID % 5         (0–4 → folder 001–005)
# -----------------------------
MODELS=(      "cyclegan"      "unit"         "munit"        "dclgan"      "uvcgan"      "cyclediffusion")
MODEL_SIZES=( "model_medium"  "model_medium" "model_medium" "model_small" "model_small" "model_small")

TASK_ID=${SLURM_ARRAY_TASK_ID}

MODEL_ID=$(( TASK_ID / 50 ))
MEMBER_ID=$(( (TASK_ID % 50) / 5 ))
WSI_ID=$(( TASK_ID % 5 ))

MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))       # 01 … 10
WSI_FOLDER=$(printf "%03d" $(( WSI_ID + 1 )))      # 001 … 005

MODEL=${MODELS[$MODEL_ID]}
MODEL_SIZE=${MODEL_SIZES[$MODEL_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "MEMBER=${MEMBER}"
echo "WSI_FOLDER=${WSI_FOLDER}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

# Single-WSI metadata CSV — gives reconstruct.py the x/y tile coordinates
TEST_A="/work2/bz66izin-VSproject/VS_Data/temp/tiles/testA"
WSI_METADATA="${TEST_A}/${WSI_FOLDER}/tiles_metadata.csv"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# Flat inference tile directory written by infer_ensemble_*.sh
TILE_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

# Reconstructed WSI TIF output (shared directory; each job writes one TIF)
OUT_DIR="${ENSEMBLE_ROOT}/reconstructed/model_${MEMBER}"

# Per-WSI sentinel — avoids re-running a completed WSI even when the output
# filename is not known ahead of time (it comes from the original WSI stem)
SENTINEL="${OUT_DIR}/.done_${WSI_FOLDER}"

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Skip gracefully if this member's inference output does not exist yet
#    (handles models where only N < 10 members have been trained/inferred so far)
if [ ! -d "${TILE_DIR}" ] || [ -z "$(ls -A "${TILE_DIR}" 2>/dev/null)" ]; then
    echo "[SKIP] Inference output not found or empty: ${TILE_DIR}"
    echo "       Either this member has not been inferred yet, or the array index is out of range."
    exit 0
fi

# 2. Skip if this specific WSI has already been reconstructed
if [ -f "${SENTINEL}" ]; then
    echo "[SKIP] WSI ${WSI_FOLDER} already reconstructed (sentinel: ${SENTINEL}). Exiting."
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "Tile dir  : ${TILE_DIR}"
echo "Metadata  : ${WSI_METADATA}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Reconstruct one WSI: pass the single per-WSI metadata CSV so only
# tiles belonging to this WSI are stitched. --mode rgb handles RGB
# inference output; overlapping tiles are averaged.
# -----------------------------
run_cmd python "${PROJECT_ROOT}/reconstruct.py" \
    --metadata "${WSI_METADATA}" \
    --tile_dir "${TILE_DIR}" \
    --output   "${OUT_DIR}" \
    --mode     rgb \
    --blend    average

touch "${SENTINEL}"
echo "Done. Reconstructed WSI ${WSI_FOLDER} for ${MODEL} member ${MEMBER} → ${OUT_DIR}"
