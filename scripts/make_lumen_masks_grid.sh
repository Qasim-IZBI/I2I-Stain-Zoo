#!/bin/bash
#SBATCH --job-name=i2i_lumen_grid
#SBATCH --output=logs_ensemble_grid/lumen_grid_%A_%a.out
#SBATCH --error=logs_ensemble_grid/lumen_grid_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = 5 subsets x 10 members

# Lumen masks from each member's reconstructed SR, inside the H&E tissue
# footprint. The stage that lets compute_phi_uncertainty read masks instead of
# thresholding several GB of RGB fifty times per slide.
#
# Decomposition matches train_ensemble_cyclegan_grid.sh:
#   RANGE_ID  = TASK_ID / 10   (0-4 -> subset)
#   MEMBER_ID = TASK_ID % 10   (0-9 -> member 01-10)
#
# Reads : {subset}/model_small/reconstructed/model_{NN}/  +  the real H&E
# Writes: {subset}/model_small/lumen_masks/model_{NN}/
#
# The REFERENCE side is a separate single run, not this array — the real H&E
# thresholded against its own footprint:
#
#   python I2I-Stain-Zoo/make_lumen_masks.py \
#       --rgb_dir ${HE_DIR} --he_dir ${HE_DIR} \
#       --white_thresh ${WHITE_THRESH_HE} --min_object_px ${MIN_OBJECT_PX} \
#       --outdir /work2/.../lumen_masks_real
#
# MIN_OBJECT_PX must be identical on both arms. A component size that differs
# between virtual and reference makes the discrepancy partly a parameter choice,
# the same rule section 5.4.4 imposes on the collagen mask.

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

mkdir -p logs_ensemble_grid

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — overridable via --export
# -----------------------------
GRID_ROOT="${GRID_ROOT:-/work2/bz66izin-UC_project/ensemble/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
HE_DIR="${HE_DIR:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_rgb/testA}"

# Thresholds: the SR one reads the generated stain, the H&E one builds the
# footprint. Both from scripts/calibrate_white_thresh.sh — run it on a member's
# reconstructions, not on the real SR, since that is what enters phi.
WHITE_THRESH_SR="${WHITE_THRESH_SR:-0.65}"
WHITE_THRESH_HE="${WHITE_THRESH_HE:-0.65}"
MIN_OBJECT_PX="${MIN_OBJECT_PX:-64}"

N_MEMBERS=10
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))
MEMBER_ID=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$RANGE_ID]}" "${RANGE_ENDS[$RANGE_ID]}")
MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))

ENSEMBLE_ROOT="${GRID_ROOT}/${RANGE_TAG}/${MODEL_SIZE}"
IN_DIR="${ENSEMBLE_ROOT}/reconstructed/model_${MEMBER}"
OUT_DIR="${ENSEMBLE_ROOT}/lumen_masks/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  SUBSET=${RANGE_TAG}  MEMBER=${MEMBER}"
echo "Input   : ${IN_DIR}"
echo "H&E     : ${HE_DIR}"
echo "Output  : ${OUT_DIR}"
echo "Thresh  : SR ${WHITE_THRESH_SR}, H&E footprint ${WHITE_THRESH_HE}"
echo "Speckle : min_object_px ${MIN_OBJECT_PX}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Reconstructions missing or empty: ${IN_DIR}"
    echo "        Run recon_ensemble_grid.sh first."
    exit 1
fi

if [ ! -d "${HE_DIR}" ]; then
    echo "[ERROR] H&E directory not found: ${HE_DIR}"
    echo "        The footprint comes from the H&E, never from thresholding the"
    echo "        SR: the SR footprint is unstable at both ends of the sweep."
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# make_lumen_masks skips slides that already have a mask, so a re-submit after
# an interruption resumes rather than repeating.
run_cmd python "${PROJECT_ROOT}/make_lumen_masks.py" \
    --rgb_dir           "${IN_DIR}" \
    --he_dir            "${HE_DIR}" \
    --white_thresh      "${WHITE_THRESH_SR}" \
    --footprint_thresh  "${WHITE_THRESH_HE}" \
    --min_object_px     "${MIN_OBJECT_PX}" \
    --outdir            "${OUT_DIR}"

echo "Done. ${RANGE_TAG} member ${MEMBER} → ${OUT_DIR}"
echo "Check lumen_masks.json: if much of the raw lumen area was removed as"
echo "speckle, the threshold is catching noise or min_object_px is eating"
echo "real structure."
