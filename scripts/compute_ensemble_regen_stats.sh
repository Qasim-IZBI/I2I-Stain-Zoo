#!/bin/bash
#SBATCH --job-name=i2i_ens_regen_stats
#SBATCH --output=logs_ensemble_ugac/regen_stats_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/regen_stats_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-99   # 100 jobs = 5 data blocks x 20 test WSIs

# Ensemble-mean A' tiles, from the B->A pass of the UGAC CycleGAN ensemble.
#
# Consumes what infer_ensemble_cyclegan_ugac_B2A.sh writes:
#
#   {UGAC_ROOT}/{block}/model_small/inference_B2A/model_{01..10}/
#
# and reduces the ten members to:
#
#   {UGAC_ROOT}/{block}/model_small/regen_stats/cyclegan/
#       mean_rgb/   the ensemble-mean A' — THE POINT OF THIS STEP
#       raw_npy/    per-pixel spread of A' across members
#       heatmaps/   magma PNGs of the same, qualitative only
#       summary_wsi{NNN}.json
#
# `mean_rgb/` is what compute_ensemble_regen_error.sh consumes as
# `--path_A_regen`, giving cycle-reconstruction error against the real H&E
# without re-running the forward pass at evaluation time. The variance maps come
# along because uncertainty.py computes them anyway; they are the disagreement
# between members about the ROUND TRIP, which is not the same quantity as the
# forward-pass uncertainty in .../uncertainty/ and should not be pooled with it.
#
# Decomposition matches compute_ensemble_uncertainty.sh — one job per (block,
# test WSI), the inference array with the member axis collapsed:
#   tasks  0– 19  ->  folders 001–007   WSI 1–20
#   tasks 20– 39  ->  folders 008–014   WSI 1–20
#   tasks 40– 59  ->  folders 015–021   WSI 1–20
#   tasks 60– 79  ->  folders 022–028   WSI 1–20
#   tasks 80– 99  ->  folders 029–035   WSI 1–20
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/compute_ensemble_regen_stats.sh
# Single block only (e.g. folders 008–014):
#   sbatch --array=20-39 I2I-Stain-Zoo/scripts/compute_ensemble_regen_stats.sh
# Cap concurrency (100 jobs each reading ten members is heavy on the filesystem):
#   sbatch --array=0-99%20 I2I-Stain-Zoo/scripts/compute_ensemble_regen_stats.sh

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
# 2D decomposition: 5 data blocks x 20 test WSIs
#
# N_WSIS must match --array above: the two are one decomposition, and changing
# either alone silently remaps every task to the wrong block.
# -----------------------------
N_WSIS=20
N_MEMBERS=10          # members per block, as written by the B2A inference array

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))    # 0 … 4
WSI_IDX=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))     # 0 … 19

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$RANGE_ID]}" "${RANGE_ENDS[$RANGE_ID]}")

WSI_NUM=$(( WSI_IDX + 1 ))                      # 1 … 20
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")
DATA_RANGE="${WSI_NUM},${WSI_NUM}"

# -----------------------------
# Paths — overridable via --export, defaults match the B2A inference script
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"

ENSEMBLE_ROOT="${UGAC_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"
IN_DIR="${ENSEMBLE_ROOT}/inference_B2A"
OUT_DIR="${ENSEMBLE_ROOT}/regen_stats"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  WSI=${WSI_FOLDER}"
echo "Input  : ${IN_DIR}"
echo "Output : ${OUT_DIR}"
echo "Range  : ${DATA_RANGE}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${IN_DIR}" ]; then
    echo "[ERROR] B2A inference output missing: ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan_ugac_B2A.sh --array=$(( RANGE_ID * N_MEMBERS ))-$(( RANGE_ID * N_MEMBERS + N_MEMBERS - 1 )) first."
    exit 1
fi

FOUND=$(ls -d "${IN_DIR}"/model_* 2>/dev/null | wc -l)
if [ "${FOUND}" -eq 0 ]; then
    echo "[ERROR] No model_* directories under ${IN_DIR}"
    echo "        Run infer_ensemble_cyclegan_ugac_B2A.sh first."
    exit 1
fi

# uncertainty.py globs model_* and averages over whatever it finds, so a
# half-finished block silently produces a mean A' over fewer members. That is
# not merely a smaller number — it is a DIFFERENT reconstruction, and the regen
# error computed from it would be attributed to the full ensemble.
if [ "${FOUND}" -ne "${N_MEMBERS}" ]; then
    echo "[ERROR] ${IN_DIR} holds ${FOUND} model_* dirs, expected ${N_MEMBERS}."
    echo "        mean_rgb/ would be an average over a shrunken ensemble and"
    echo "        nothing downstream would show it. Finish the B2A array for this"
    echo "        block, or set N_MEMBERS if the design really changed."
    ls -d "${IN_DIR}"/model_* 2>/dev/null | sed 's/^/          /'
    exit 1
fi

# The member dirs must actually contain this WSI. discover_common_filenames
# INTERSECTS filenames across members, so a folder the inference never produced
# yields an empty intersection rather than an error, and the job would write a
# summary describing nothing. infer_ensemble_cyclegan_ugac_B2A.sh pins
# DATA_RANGE="1,5", so this is the expected failure for WSIs 6-20.
if [ ! -d "${IN_DIR}/model_01/${WSI_FOLDER}/images" ]; then
    echo "[ERROR] ${IN_DIR}/model_01/${WSI_FOLDER}/images does not exist."
    echo "        The B2A inference for this block did not cover WSI ${WSI_FOLDER}."
    echo "        Widen DATA_RANGE in infer_ensemble_cyclegan_ugac_B2A.sh to the"
    echo "        full test set and re-run that array for this block."
    echo "        Present in model_01:"
    ls -d "${IN_DIR}/model_01"/[0-9][0-9][0-9] 2>/dev/null | sed 's/^/          /' | head
    exit 1
fi

# Per-WSI skip guard: the summary JSON this specific job writes. uncertainty.py
# tags it with the WSI when --data_range names a single folder, which is what
# keeps the twenty jobs of a block from racing on one filename.
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

# NO --mask_dir / --min_tissue_fraction here, unlike compute_ensemble_uncertainty.sh.
# Those drop tiles below the tissue threshold, and a dropped tile has no mean_rgb
# entry — so the regen error would be missing exactly where the reconstruction is
# thinnest, and evaluation.py would silently score fewer tiles than it was given.
# Tissue filtering belongs downstream, on the error maps, where it removes
# background from a statistic rather than removing tiles from an image.
run_cmd python "${PROJECT_ROOT}/uncertainty.py" \
    --model            "${MODEL}" \
    --data             "${IN_DIR}" \
    --output           "${OUT_DIR}" \
    --data_range       "${DATA_RANGE}" \
    --lower-percentile 1 \
    --upper-percentile 99

echo
echo "Done. Regen stats for ${RANGE_TAG} WSI ${WSI_FOLDER} -> ${OUT_DIR}/${MODEL}/"
echo "Feed mean_rgb/ to the regen error as --path_A_regen:"
echo "  python ${PROJECT_ROOT}/evaluation.py --metric regen_error \\"
echo "      --path_A <testA>/${WSI_FOLDER}/images \\"
echo "      --path_A_regen ${OUT_DIR}/${MODEL}/mean_rgb/${WSI_FOLDER}/images \\"
echo "      --overlay_dir <REGEN_ROOT>/wsi${WSI_FOLDER}/ --save_error_npy"
