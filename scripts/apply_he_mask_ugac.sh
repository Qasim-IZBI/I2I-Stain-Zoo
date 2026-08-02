#!/bin/bash
#SBATCH --job-name=i2i_hemask_ugac
#SBATCH --output=logs_ensemble_ugac/i2i_hemask_ugac_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/i2i_hemask_ugac_%A_%a.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = 5 data blocks x 10 ensemble members

# Zero out PSR predictions falling outside the H&E tissue boundary. The segmenter
# was trained on full WSIs against a white glass background; translated tissue
# lacks that context, so it over-calls signal in background regions.
#
# Decomposition matches train/infer_ensemble_cyclegan_ugac.sh:
#   RANGE_ID  = TASK_ID / 10   (0-4 -> data block)
#   MEMBER_ID = TASK_ID % 10   (0-9 -> member 01-10)

set -eo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble_ugac

N_MEMBERS=10
RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))
MEMBER_ID=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")
MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))

PROJECT_ROOT=I2I-Stain-Zoo
HE_MASKS_DIR=/work2/bz66izin-VSproject/HE_tissue
WSI_COUNT=5

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/${RANGE_TAG}/model_small"
IN_DIR="${ENSEMBLE_ROOT}/wsi_masks/model_${MEMBER}"
OUT_DIR="${ENSEMBLE_ROOT}/wsi_masks_cleaned/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  BLOCK=${RANGE_TAG}  MEMBER=${MEMBER}"
echo "Input  : ${IN_DIR}"
echo "Output : ${OUT_DIR}"

if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Input missing or empty: ${IN_DIR}"
    exit 1
fi

if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${WSI_COUNT}" ]; then
        echo "[SKIP] ${N_DONE}/${WSI_COUNT} masks already present in ${OUT_DIR}."
        exit 0
    fi
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/apply_he_mask.py" \
    --psr_masks "${IN_DIR}" \
    --he_masks  "${HE_MASKS_DIR}" \
    --outdir    "${OUT_DIR}"

echo "Done. ${RANGE_TAG} member ${MEMBER} → ${OUT_DIR}"
