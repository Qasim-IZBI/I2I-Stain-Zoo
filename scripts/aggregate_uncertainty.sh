#!/bin/bash
#SBATCH --job-name=i2i_agg_uncertainty
#SBATCH --output=logs_ensemble/agg_uncertainty_%A_%a.out
#SBATCH --error=logs_ensemble/agg_uncertainty_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 jobs = 1 per model

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble

# -----------------------------
# Model config — must match compute_ensemble_uncertainty.sh
# Index:        0          1      2       3        4        5
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SIZE="${MODEL_SIZES[$SLURM_ARRAY_TASK_ID]}"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# uncertainty.py writes: {ENSEMBLE_ROOT}/uncertainty/{MODEL}/raw_npy/NNN/images/tile.npy
UNCERTAINTY_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/raw_npy"
OUT_DIR="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/per_wsi_csv"

TEST_A="/work2/bz66izin-VSproject/VS_Data/temp/tiles/testA"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}"
echo "Uncertainty dir : ${UNCERTAINTY_DIR}"
echo "Output          : ${OUT_DIR}"

# -----------------------------
# Pre-flight: uncertainty maps must exist
# -----------------------------
if [ ! -d "${UNCERTAINTY_DIR}" ] || [ -z "$(find "${UNCERTAINTY_DIR}" -name "*.npy" -maxdepth 4 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No .npy uncertainty maps found under: ${UNCERTAINTY_DIR}"
    echo "        Run compute_ensemble_uncertainty.sh first."
    exit 1
fi

# Skip guard: directory exists and contains at least one CSV
if [ -d "${OUT_DIR}" ] && [ -n "$(find "${OUT_DIR}" -name "*.csv" -maxdepth 1 2>/dev/null | head -1)" ]; then
    echo "[SKIP] Already completed: ${OUT_DIR}"
    exit 0
fi

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/aggregate_uncertainty.py \
    --uncertainty_dir "${UNCERTAINTY_DIR}" \
    --tiles_metadata  "${TEST_A}" \
    --mask_dir        "${TEST_A}" \
    --min_tissue_fraction 0.1 \
    --outdir          "${OUT_DIR}"

echo "Done. Per-WSI uncertainty CSVs saved to ${OUT_DIR}/"
