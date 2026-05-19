#!/bin/bash
#SBATCH --job-name=i2i_recon_heatmaps
#SBATCH --output=logs_ensemble/recon_heatmaps_%A_%a.out
#SBATCH --error=logs_ensemble/recon_heatmaps_%A_%a.err
#
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 jobs — one per model

# Reconstructs WSI-level heatmaps for both uncertainty and regen-error maps.
# One SLURM array job per model; each job processes all 5 test WSIs.
#
# Prerequisite outputs consumed (per model):
#   uncertainty/{MODEL}/raw_npy/  — from compute_ensemble_uncertainty.sh
#   regen_error/wsi{NNN}/error_npy/ — from compute_ensemble_regen_error.sh
#
# Outputs written to:
#   wsi_heatmaps/uncertainty/   — {wsi_stem}_heatmap.tif  + colorbar_legend.png
#   wsi_heatmaps/regen_error/   — {wsi_stem}_heatmap.tif  (per-WSI scale)
#   (both also write {wsi_stem}.npy when --save_npy is active)
#
# Array layout
# task 0 → cyclegan        (model_medium)
# task 1 → unit            (model_medium)
# task 2 → munit           (model_medium)
# task 3 → dclgan          (model_small)
# task 4 → uvcgan          (model_small)
# task 5 → cyclediffusion  (model_small)

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

# ---------------------------------------------------------------------------
# Model arrays
# Index:          0          1      2       3        4        5
MODELS=(    cyclegan   unit   munit   dclgan   uvcgan   cyclediffusion )
MODEL_SIZES=(model_medium model_medium model_medium model_small model_small model_small)

MODEL_IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$MODEL_IDX]}"
MODEL_SIZE="${MODEL_SIZES[$MODEL_IDX]}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MODEL=${MODEL}  SIZE=${MODEL_SIZE}"

# ---------------------------------------------------------------------------
# Fixed paths
N_WSIS=5
PROJECT_ROOT=I2I-Stain-Zoo
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"
OUT_BASE="${ENSEMBLE_ROOT}/wsi_heatmaps"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# ===========================================================================
# Part 1 — Uncertainty WSI heatmaps
#   npy_dir layout: uncertainty/{MODEL}/raw_npy/NNN/images/*.npy  (nested)
#   All 5 WSIs reconstructed in a single call.
#   --global_norm ensures a shared colour scale across all WSIs and writes
#   a colorbar_legend.png so the heatmaps are directly comparable.
# ===========================================================================
UNCERTAINTY_NPY="${ENSEMBLE_ROOT}/uncertainty/${MODEL}/raw_npy"
OUT_UNCERTAINTY="${OUT_BASE}/uncertainty"

if [ ! -d "${UNCERTAINTY_NPY}" ] || [ -z "$(ls -A "${UNCERTAINTY_NPY}" 2>/dev/null)" ]; then
    echo "[SKIP-uncertainty] raw_npy dir missing or empty: ${UNCERTAINTY_NPY}"
    echo "                   Run compute_ensemble_uncertainty.sh first."
else
    N_DONE_U=$(find "${OUT_UNCERTAINTY}" -maxdepth 1 -name "*_heatmap.tif" 2>/dev/null | wc -l)
    if [ "${N_DONE_U}" -ge "${N_WSIS}" ]; then
        echo "[SKIP-uncertainty] ${N_DONE_U}/${N_WSIS} heatmaps already present: ${OUT_UNCERTAINTY}"
    else
        mkdir -p "${OUT_UNCERTAINTY}"
        echo "--- Uncertainty: reconstructing ${N_WSIS} WSI heatmaps ---"
        run_cmd python "${PROJECT_ROOT}/reconstruct_heatmaps.py" \
            --metadata   "${TEST_A}" \
            --npy_dir    "${UNCERTAINTY_NPY}" \
            --output     "${OUT_UNCERTAINTY}" \
            --colormap   magma \
            --global_norm \
            --save_npy
        echo "Uncertainty heatmaps → ${OUT_UNCERTAINTY}"
    fi
fi

# ===========================================================================
# Part 2 — Regen-error WSI heatmaps
#   npy_dir layout: regen_error/wsi{NNN}/error_npy/*.npy  (flat, per-WSI)
#   Tile names are not unique across WSIs (all start from 0000001), so each
#   WSI is processed separately with --data_range N,N pointing to the
#   WSI-specific npy_dir.
#   --global_norm within each call computes percentile bounds from the full
#   WSI canvas (consistent with how evaluation.py normalizes per-batch).
# ===========================================================================
OUT_REGEN="${OUT_BASE}/regen_error"
mkdir -p "${OUT_REGEN}"

for WSI_IDX in $(seq 0 $(( N_WSIS - 1 ))); do
    WSI_NUM=$(( WSI_IDX + 1 ))
    WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")

    REGEN_NPY="${ENSEMBLE_ROOT}/regen_error/wsi${WSI_FOLDER}/error_npy"

    if [ ! -d "${REGEN_NPY}" ] || [ -z "$(ls -A "${REGEN_NPY}" 2>/dev/null)" ]; then
        echo "[SKIP-regen WSI ${WSI_FOLDER}] npy dir missing or empty: ${REGEN_NPY}"
        continue
    fi

    # Skip guard: look for the heatmap file that reconstruct_heatmaps.py would write.
    # The stem comes from the source_file column in tiles_metadata.csv; we can't know
    # the exact filename in advance, so we check if any new heatmap was added after the
    # previous WSI was processed — instead, use a sentinel file.
    SENTINEL="${OUT_REGEN}/.done_wsi${WSI_FOLDER}"
    if [ -f "${SENTINEL}" ]; then
        echo "[SKIP-regen WSI ${WSI_FOLDER}] already done (sentinel present)"
        continue
    fi

    echo "--- Regen error WSI ${WSI_FOLDER}: reconstructing heatmap ---"
    run_cmd python "${PROJECT_ROOT}/reconstruct_heatmaps.py" \
        --metadata   "${TEST_A}" \
        --npy_dir    "${REGEN_NPY}" \
        --output     "${OUT_REGEN}" \
        --colormap   hot \
        --data_range "${WSI_NUM},${WSI_NUM}" \
        --save_npy

    touch "${SENTINEL}"
    echo "Regen error WSI ${WSI_FOLDER} heatmap → ${OUT_REGEN}"
done

echo ""
echo "=== Done: ${MODEL} (${MODEL_SIZE}) ==="
echo "    Uncertainty   → ${OUT_UNCERTAINTY}"
echo "    Regen error   → ${OUT_REGEN}"
