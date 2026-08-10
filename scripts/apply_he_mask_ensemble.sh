#!/bin/bash
#SBATCH --job-name=i2i_apply_he_ens
#SBATCH --output=logs_apply_he/apply_he_ens_%A_%a.out
#SBATCH --error=logs_apply_he/apply_he_ens_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-59   # 60 jobs = 6 models × 10 ensemble members

# Applies HE tissue mask to WSI-level PSR masks produced by
# segment_psr_nn_light_ensemble.sh, zeroing out predictions in background
# regions of the translated SR images.
#
# Inputs (per job):
#   {ENSEMBLE_ROOT}/wsi_masks/model_{MEMBER}/  — WSI PSR masks from nnUNet
#   HE_MASKS_DIR                               — reconstructed HE tissue masks
# Outputs (per job):
#   {ENSEMBLE_ROOT}/wsi_masks_cleaned/model_{MEMBER}/  — cleaned PSR masks
#
# Array layout
# tasks  0–9  → cyclegan        (model_medium)  members 01–10
# tasks 10–19 → unit            (model_medium)  members 01–10
# tasks 20–29 → munit           (model_medium)  members 01–10
# tasks 30–39 → dclgan          (model_small)   members 01–10
# tasks 40–49 → uvcgan          (model_small)   members 01–10
# tasks 50–59 → cyclediffusion  (model_small)   members 01–10

set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_apply_he

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models × 10 members = 60 jobs)
# -----------------------------
MODELS=(      "cyclegan"      "unit"         "munit"        "dclgan"      "uvcgan"      "cyclediffusion")
MODEL_SIZES=( "model_medium"  "model_medium" "model_medium" "model_small" "model_small" "model_small")

N_MEMBERS=10
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))   # 0 … 5
MEMBER_IDX=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))  # 0 … 9

MEMBER=$(printf "%02d" $(( MEMBER_IDX + 1 )))      # 01 … 10

MODEL=${MODELS[$MODEL_IDX]}
MODEL_SIZE=${MODEL_SIZES[$MODEL_IDX]}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "MEMBER=${MEMBER}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

# Reconstructed WSI-level HE tissue masks (shared across all configs and members)
HE_MASKS_DIR=/work2/bz66izin-VSproject/HE_tissue

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# WSI-level PSR masks produced by segment_psr_nn_light_ensemble.sh
PSR_DIR="${ENSEMBLE_ROOT}/wsi_masks/model_${MEMBER}"

# Output: cleaned masks with background signal zeroed out
OUT_DIR="${ENSEMBLE_ROOT}/wsi_masks_cleaned/model_${MEMBER}"

WSI_COUNT=5   # number of WSI TIFs expected per member

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. PSR masks must exist
if [ ! -d "${PSR_DIR}" ] || [ -z "$(ls -A "${PSR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] PSR masks not found or empty: ${PSR_DIR}"
    echo "        Run segment_psr_nn_light_ensemble.sh first."
    exit 1
fi

# 2. HE tissue masks must exist
if [ ! -d "${HE_MASKS_DIR}" ] || [ -z "$(ls -A "${HE_MASKS_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] HE tissue masks not found or empty: ${HE_MASKS_DIR}"
    exit 1
fi

# 3. Skip if expected number of cleaned masks already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} cleaned masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial output detected (${N_DONE}/${WSI_COUNT} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "PSR masks : ${PSR_DIR}"
echo "HE masks  : ${HE_MASKS_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Apply HE tissue mask
# -----------------------------
run_cmd python "${PROJECT_ROOT}/apply_he_mask.py" \
    --psr_masks "${PSR_DIR}" \
    --he_masks  "${HE_MASKS_DIR}" \
    --outdir    "${OUT_DIR}"

echo "Done. Cleaned masks for ${MODEL} member ${MEMBER} saved to ${OUT_DIR}"
