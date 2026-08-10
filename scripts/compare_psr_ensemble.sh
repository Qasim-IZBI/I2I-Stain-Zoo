#!/bin/bash
#SBATCH --job-name=i2i_compare_psr_ens
#SBATCH --output=logs_compare_psr/compare_psr_ens_%A_%a.out
#SBATCH --error=logs_compare_psr/compare_psr_ens_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 jobs — one per model type

# Runs compare_psr.py for each model, passing all 10 ensemble member masks as
# --masks_generated. The resulting per_wsi.csv contains PSR fractions for the
# real SR and every member, which can be pivoted to compute mean CPA-MAE per
# WSI (Option 2 aggregation step).
#
# Pipeline position:
#   fill_tissue_holes_ensemble.sh  →  wsi_masks_final/model_{MEMBER}/
#           ↓
#   compare_psr_ensemble.sh        →  ensemble_cpa_comparison/{MODEL}/per_wsi.csv
#           ↓
#   (downstream) mean |member_i − real| per WSI → mean_cpa_mae per WSI
#
# Inputs (per job):
#   {ENSEMBLE_ROOT}/wsi_masks_final/model_{01..10}/   — hole-filled ensemble masks
#   REAL_DIR                                          — real SR reference masks
# Outputs (per job):
#   {OUT_BASE}/{MODEL}/per_wsi.csv     — PSR fractions: real + 10 members × 5 WSIs
#   {OUT_BASE}/{MODEL}/summary.json    — mae_paired per member (vs real)
#   {OUT_BASE}/{MODEL}/comparison.png  — boxplot real vs all members
#   {OUT_BASE}/{MODEL}/paired_scatter.png
#   {OUT_BASE}/{MODEL}/paired_metrics.png

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

mkdir -p logs_compare_psr

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (one job per model)
# -----------------------------
MODELS=(      "cyclegan"      "unit"         "munit"        "dclgan"      "uvcgan"      "cyclediffusion")
MODEL_SIZES=( "model_medium"  "model_medium" "model_medium" "model_small" "model_small" "model_small")

MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
MODEL_SIZE=${MODEL_SIZES[${SLURM_ARRAY_TASK_ID}]}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${MODEL_SIZE}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# Real SR reference masks (post-processed, same as compare_psr_all_configs.sh)
REAL_DIR=/work2/bz66izin-VSproject/psr_masks/real/psr_masks_wsi_final

# Output directory for this model
OUT_BASE=/work2/bz66izin-VSproject/ensemble_cpa_comparison
OUT_DIR="${OUT_BASE}/${MODEL}"

N_MEMBERS=10

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
# Members whose wsi_masks_final directory is missing or empty are skipped
# with a warning rather than failing the job.
# -----------------------------
MASKS_ARGS=()
LABEL_ARGS=()

for MEMBER_IDX in $(seq 1 ${N_MEMBERS}); do
    MEMBER=$(printf "%02d" "${MEMBER_IDX}")
    MASK_DIR="${ENSEMBLE_ROOT}/wsi_masks_final/model_${MEMBER}"
    if [ -d "${MASK_DIR}" ] && [ -n "$(ls -A "${MASK_DIR}" 2>/dev/null)" ]; then
        MASKS_ARGS+=("${MASK_DIR}")
        LABEL_ARGS+=("member_${MEMBER}")
    else
        echo "[WARN] Missing or empty — skipping: ${MASK_DIR}"
    fi
done

if [ ${#MASKS_ARGS[@]} -eq 0 ]; then
    echo "[ERROR] No ensemble PSR masks found for model '${MODEL}'. Exiting."
    echo "        Run fill_tissue_holes_ensemble.sh first."
    exit 1
fi

# Skip if output already complete
if [ -f "${OUT_DIR}/per_wsi.csv" ]; then
    echo "[SKIP] per_wsi.csv already present in ${OUT_DIR}. Exiting."
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "Real masks : ${REAL_DIR}"
echo "Members    : ${#MASKS_ARGS[@]}/${N_MEMBERS}"
echo "Output dir : ${OUT_DIR}"

run_cmd python "${PROJECT_ROOT}/compare_psr.py" \
    --masks_real      "${REAL_DIR}" \
    --masks_generated "${MASKS_ARGS[@]}" \
    --labels          "${LABEL_ARGS[@]}" \
    --outdir          "${OUT_DIR}" \
    --strip_prefix

echo "Done. CPA comparison for '${MODEL}' saved to ${OUT_DIR}"
echo ""
echo "Next step: compute mean CPA-MAE per WSI from ${OUT_DIR}/per_wsi.csv"
echo "  python -c \""
echo "  import pandas as pd, numpy as np"
echo "  df = pd.read_csv('${OUT_DIR}/per_wsi.csv')"
echo "  real = df[df.condition=='real'].set_index('wsi')['psr_fraction']"
echo "  mem  = df[df.condition!='real'].copy()"
echo "  mem['abs_diff'] = mem.apply(lambda r: abs(r.psr_fraction - real[r.wsi]), axis=1)"
echo "  print(mem.groupby('wsi')['abs_diff'].mean())"
echo "  \""
