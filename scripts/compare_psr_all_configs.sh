#!/bin/bash
#SBATCH --job-name=i2i_compare_psr
#SBATCH --output=logs_compare_psr/compare_psr_%A_%a.out
#SBATCH --error=logs_compare_psr/compare_psr_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 jobs — one per model type

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"
echo "Running on CPU (no GPU requested)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_compare_psr

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
SIZES=("small" "medium" "large")
DATASIZES=("small" "medium" "large")

MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

SEG_BASE=/work2/bz66izin-VSproject/psr_masks

# Real SR reference — post-processed to match generated pipeline
REAL_DIR=${SEG_BASE}/real/psr_masks_wsi_final

# Output directory for this model's comparison
OUT_DIR=/work2/bz66izin-VSproject/psr_comparison/${MODEL}

# -----------------------------
# Pre-flight: real masks must exist
# -----------------------------
if [ ! -d "${REAL_DIR}" ] || [ -z "$(ls -A "${REAL_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Real PSR masks not found or empty: ${REAL_DIR}"
    echo "        Run fill_tissue_holes_real.sh first."
    exit 1
fi

# -----------------------------
# Build --masks_generated and --labels lists dynamically.
# All 9 configs (3 model-sizes x 3 data-sizes) that exist are included;
# missing configs are skipped with a warning rather than failing the job.
# -----------------------------
MASKS_ARGS=()
LABEL_ARGS=()

for SIZE in "${SIZES[@]}"; do
    for DATASIZE in "${DATASIZES[@]}"; do
        MASK_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/psr_masks_wsi_final
        if [ -d "${MASK_DIR}" ] && [ -n "$(ls -A "${MASK_DIR}" 2>/dev/null)" ]; then
            MASKS_ARGS+=("${MASK_DIR}")
            LABEL_ARGS+=("${SIZE}_model/${DATASIZE}_data")
        else
            echo "[WARN] Missing or empty — skipping: ${MASK_DIR}"
        fi
    done
done

if [ ${#MASKS_ARGS[@]} -eq 0 ]; then
    echo "[ERROR] No generated PSR masks found for model '${MODEL}'. Exiting."
    exit 1
fi

# Skip if output already complete (summary.json present)
if [ -f "${OUT_DIR}/summary.json" ]; then
    echo "[SKIP] summary.json already present in ${OUT_DIR}. Exiting."
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "Real masks : ${REAL_DIR}"
echo "Configs    : ${#MASKS_ARGS[@]}/9"
echo "Output dir : ${OUT_DIR}"

run_cmd python "${PROJECT_ROOT}/compare_psr.py" \
    --masks_real       "${REAL_DIR}" \
    --masks_generated  "${MASKS_ARGS[@]}" \
    --labels           "${LABEL_ARGS[@]}" \
    --outdir           "${OUT_DIR}" \
    --strip_prefix

echo "Done. PSR comparison for '${MODEL}' saved to ${OUT_DIR}"
