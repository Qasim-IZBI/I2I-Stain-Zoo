#!/bin/bash
#SBATCH --job-name=i2i_px_components
#SBATCH --output=logs_ensemble_ugac/px_components_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/px_components_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-19   # 20 jobs = 1 per test WSI

# Does the ensemble's spread predict its cycle error — and WHICH spread?
#
# Two steps in one job, per WSI:
#
#   1. decompose_pixel_uncertainty.py reads all 5 subsets x 10 members at once
#      and splits each pixel into procedural (between seeds, within a subset)
#      and data_exposure (between subsets), plus their total.
#   2. uncertainty_calibration.py scores each of the three against the SAME
#      regen error, so only the uncertainty source differs.
#
# ONE JOB PER WSI, not per (block, WSI): the decomposition needs every subset
# together, so the block axis is consumed rather than parallelised over. That is
# also why this asks for more memory than its neighbours — fifty 256x256x3 tiles
# are live at once, plus the fold stacks.
#
# The regen error is per block (each from that block's mean A'), so ONE is
# chosen as the target. Cycle error is a property of a model's forward/inverse
# pair; it is not decomposable into these components, and averaging it across
# blocks would pair a grand-mean error with a within-subset sigma. Set
# REGEN_BLOCK to compare against a different subset's error.
#
# > RETIRED CHAIN. Kept for provenance beside the rest of scripts/*_ugac.sh.
# > Do NOT mix its outputs with ensemble_grid/.
#
#   sbatch I2I-Stain-Zoo/scripts/calibrate_pixel_components.sh
#   sbatch --array=0 I2I-Stain-Zoo/scripts/calibrate_pixel_components.sh   # one WSI

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo.
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

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

WSI_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")

UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
# Which block's regen error to score against — see the note above.
REGEN_BLOCK="${REGEN_BLOCK:-data_001_007}"
OUTROOT="${OUTROOT:-${UGAC_ROOT}/pixel_components}"
MIN_TISSUE_PIXELS="${MIN_TISSUE_PIXELS:-256}"

OUTDIR="${OUTROOT}/wsi${WSI_FOLDER}"
ERROR_DIR="${UGAC_ROOT}/${REGEN_BLOCK}/${MODEL_SIZE}/regen_error/wsi${WSI_FOLDER}/error_npy"
MASK_DIR="${TEST_A}/${WSI_FOLDER}/masks"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  WSI=${WSI_FOLDER}"
echo "Output      : ${OUTDIR}"
echo "Regen error : ${ERROR_DIR}   (block ${REGEN_BLOCK})"

# -----------------------------
# Pre-flight
# -----------------------------
FOLD_ARGS=()
for i in "${!RANGE_STARTS[@]}"; do
    TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")
    D="${UGAC_ROOT}/${TAG}/${MODEL_SIZE}/inference"
    if [ ! -d "${D}/model_01/${WSI_FOLDER}/images" ]; then
        echo "[ERROR] ${D}/model_01/${WSI_FOLDER}/images missing."
        echo "        Every subset must cover this WSI — the decomposition is"
        echo "        over all five, and a missing one is not a smaller sample"
        echo "        but a different quantity. Widen DATA_RANGE in"
        echo "        infer_ensemble_cyclegan_ugac.sh and re-run its array."
        exit 1
    fi
    N=$(ls -d "${D}"/model_* 2>/dev/null | wc -l)
    echo "  ${TAG}: ${N} members"
    FOLD_ARGS+=(--fold "${D}")
done

if [ ! -d "${ERROR_DIR}" ] || [ -z "$(ls -A "${ERROR_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Regen-error dir missing or empty: ${ERROR_DIR}"
    echo "        Run compute_ensemble_regen_error.sh for block ${REGEN_BLOCK}."
    exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
    echo "[ERROR] Mask directory not found: ${MASK_DIR}"
    exit 1
fi

if [ -f "${OUTDIR}/calibration_total/summary.json" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() { echo "Running command:"; printf ' %q' "$@"; echo; "$@"; }

# -----------------------------
# 1. decompose
# -----------------------------
run_cmd python "${PROJECT_ROOT}/decompose_pixel_uncertainty.py" \
    "${FOLD_ARGS[@]}" \
    --data_range "${WSI_NUM},${WSI_NUM}" \
    --output     "${OUTDIR}"

# -----------------------------
# 2. calibrate each component against the same error
# -----------------------------
for COMP in total procedural data_exposure; do
    U_DIR="${OUTDIR}/${COMP}/raw_npy/${WSI_FOLDER}/images"
    C_OUT="${OUTDIR}/calibration_${COMP}"
    if [ ! -d "${U_DIR}" ] || [ -z "$(ls -A "${U_DIR}" 2>/dev/null)" ]; then
        echo "[ERROR] Decomposition produced nothing for ${COMP}: ${U_DIR}"
        exit 1
    fi
    echo
    echo "--- calibrating ${COMP} ---"
    mkdir -p "${C_OUT}"
    run_cmd python "${PROJECT_ROOT}/uncertainty_calibration.py" \
        --uncertainty_dir   "${U_DIR}" \
        --error_dirs        "${ERROR_DIR}" \
        --mask_dir          "${MASK_DIR}" \
        --tiles_metadata    "${TEST_A}" \
        --outdir            "${C_OUT}" \
        --min_tissue_pixels "${MIN_TISSUE_PIXELS}" \
        --title             "${COMP} WSI ${WSI_FOLDER}"
done

echo
echo "Done. WSI ${WSI_FOLDER} -> ${OUTDIR}/calibration_{total,procedural,data_exposure}/"
echo
echo "Compare the three summary.json files. They share one error target, so a"
echo "difference in rho is a difference between the SPREADS, not between the"
echo "errors — which is the question the crossed grid exists to pose."
echo "Check negative_data_fraction in the decomposition summary first: where"
echo "data_exposure came out negative it is NaN and those pixels drop out, so a"
echo "large fraction means its rho rests on fewer pixels than the other two."
