#!/bin/bash
#SBATCH --job-name=i2i_ens_uncertainty
#SBATCH --output=logs_ensemble_ugac/uncertainty_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/uncertainty_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-24   # 25 jobs = 5 data blocks x 5 test WSIs

# Per-pixel EPISTEMIC uncertainty over the UGAC CycleGAN ensemble.
#
# Consumes what infer_ensemble_cyclegan_ugac.sh writes:
#
#   {UGAC_ROOT}/cyclegan/{block}/model_small/inference/model_{01..10}/
#                                                     .../aleatoric_npy/
#
# and reduces the ten members of ONE block to the spread between them:
#
#   {UGAC_ROOT}/cyclegan/{block}/model_small/uncertainty/cyclegan/
#       raw_npy/  heatmaps/  mean_rgb/  summary_wsi{NNN}.json
#
# Two uncertainties live side by side here and this script produces only one of
# them. `aleatoric_npy/` is per member and comes from the UGAC heads — the
# model's own estimate of irreducible noise, written at inference. What
# uncertainty.py adds is the disagreement BETWEEN members, which the heads
# cannot see. Both use the same convention (float32 [H,W] standard deviations in
# 0–255 intensity units), so uncertainty_calibration.py takes either unchanged.
#
# Decomposition — one job per (data block, test WSI), so it is the inference
# array with the member axis collapsed, since every member of a block is read
# together:
#   tasks  0– 4  ->  folders 001–007   WSI 1–5
#   tasks  5– 9  ->  folders 008–014   WSI 1–5
#   tasks 10–14  ->  folders 015–021   WSI 1–5
#   tasks 15–19  ->  folders 022–028   WSI 1–5
#   tasks 20–24  ->  folders 029–035   WSI 1–5
#
# Per WSI rather than all five at once because uncertainty.py holds every
# member's tile for a filename in memory at once, and because the summary is
# tagged per WSI so the jobs cannot race each other writing it.
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/compute_ensemble_uncertainty.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=5-9 I2I-Stain-Zoo/scripts/compute_ensemble_uncertainty.sh

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo and the log comes
# back empty. It goes on after conda activate, as the rest of this family does.
set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble_ugac

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# 2D decomposition: 5 data blocks x 5 test WSIs
# -----------------------------
N_WSIS=5
N_MEMBERS=10          # members per block, as written by the inference array

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))    # 0 … 4
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))     # 0 … 4

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")

WSI_NUM=$(( WSI_IDX + 1 ))                      # 1 … 5
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")
DATA_RANGE="${WSI_NUM},${WSI_NUM}"

# -----------------------------
# Paths — overridable via --export, defaults match the UGAC inference script
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"

ENSEMBLE_ROOT="${UGAC_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"
IN_DIR="${ENSEMBLE_ROOT}/inference"
OUT_DIR="${ENSEMBLE_ROOT}/uncertainty"

# Tissue masks resolved as TEST_A/NNN/masks/<tile>.tif
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
MIN_TISSUE="${MIN_TISSUE:-0.1}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  WSI=${WSI_FOLDER}"
echo "Input  : ${IN_DIR}"
echo "Output : ${OUT_DIR}"
echo "Range  : ${DATA_RANGE}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${IN_DIR}" ]; then
    echo "[ERROR] Inference output missing: ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan_ugac.sh --array=$(( RANGE_ID * N_MEMBERS ))-$(( RANGE_ID * N_MEMBERS + N_MEMBERS - 1 )) first."
    exit 1
fi

FOUND=$(ls -d "${IN_DIR}"/model_* 2>/dev/null | wc -l)
if [ "${FOUND}" -eq 0 ]; then
    echo "[ERROR] No model_* directories under ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan_ugac.sh first."
    exit 1
fi

# uncertainty.py globs model_* and computes the variance over whatever it finds,
# so a half-finished block silently produces a spread over fewer members — a
# smaller number that looks like a result. Refuse instead.
if [ "${FOUND}" -ne "${N_MEMBERS}" ]; then
    echo "[ERROR] ${IN_DIR} holds ${FOUND} model_* dirs, expected ${N_MEMBERS}."
    echo "        The variance would be computed over a shrunken ensemble and"
    echo "        nothing downstream would show it. Finish the inference array"
    echo "        for this block, or set N_MEMBERS if the design really changed."
    ls -d "${IN_DIR}"/model_* 2>/dev/null | sed 's/^/          /'
    exit 1
fi

if [ ! -d "${TEST_A}/${WSI_FOLDER}" ]; then
    echo "[ERROR] Test WSI folder not found: ${TEST_A}/${WSI_FOLDER}"
    echo "        Is the test set smaller than N_WSIS=${N_WSIS}?"
    exit 1
fi

# Per-WSI skip guard: the summary JSON this specific job writes. uncertainty.py
# tags it with the WSI when --data_range names a single folder, which is what
# keeps the five jobs of a block from racing on one filename.
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

run_cmd python "${PROJECT_ROOT}/uncertainty.py" \
    --model               "${MODEL}" \
    --data                "${IN_DIR}" \
    --output              "${OUT_DIR}" \
    --data_range          "${DATA_RANGE}" \
    --lower-percentile    1 \
    --upper-percentile    99 \
    --mask_dir            "${TEST_A}" \
    --min_tissue_fraction "${MIN_TISSUE}"

echo
echo "Done. Epistemic uncertainty for ${RANGE_TAG} WSI ${WSI_FOLDER} -> ${OUT_DIR}/${MODEL}/"
echo "raw_npy/ holds per-pixel SDs in 0-255 units, the same convention as each"
echo "member's inference/model_NN/aleatoric_npy/, so both feed"
echo "uncertainty_calibration.py unchanged."
