#!/bin/bash
#SBATCH --job-name=i2i_seg_ugac
#SBATCH --output=logs_ensemble_ugac/seg_ugac_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/seg_ugac_%A_%a.err

#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-249   # 250 jobs = 5 data blocks x 10 members x 5 test WSIs

# Collagen segmentation of the reconstructed UGAC ensemble WSIs, using
# Dataset314_SR_light in WSI mode — the same segmenter as the scaling study and
# the vanilla ensemble, so the numbers stay on one footing.
#
# Decomposition matches recon_ensemble_ugac.sh exactly:
#   RANGE_ID  = TASK_ID / 50        (0-4  -> data block)
#   MEMBER_ID = (TASK_ID % 50) / 5  (0-9  -> member 01-10)
#   WSI_ID    = TASK_ID % 5         (0-4  -> folder 001-005)
#
# Input  : {block}/model_small/reconstructed/model_{NN}/
# Output : {block}/model_small/wsi_masks/model_{NN}/
#
# CAVEAT (kidney_ood_data_plan.md section 6.2): Dataset314_SR_light is trained on
# LIVER SR. Applying it to any other organ is an out-of-distribution use and its
# failure would be indistinguishable from the model bias under measurement.
# Validate against manual annotation before trusting non-liver numbers.

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

# -----------------------------
# 3D decomposition: 5 blocks x 10 members x 5 WSIs
# -----------------------------
N_MEMBERS=10
N_WSIS=5

TASK_ID=${SLURM_ARRAY_TASK_ID}
RANGE_ID=$(( TASK_ID / (N_MEMBERS * N_WSIS) ))
MEMBER_ID=$(( (TASK_ID % (N_MEMBERS * N_WSIS)) / N_WSIS ))
WSI_ID=$(( TASK_ID % N_WSIS ))

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")

MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))
WSI_FOLDER=$(printf "%03d" $(( WSI_ID + 1 )))

TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
WSI_METADATA="${TEST_A}/${WSI_FOLDER}/tiles_metadata.csv"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/${RANGE_TAG}/model_small"
RECON_DIR="${ENSEMBLE_ROOT}/reconstructed/model_${MEMBER}"
OUT_DIR="${ENSEMBLE_ROOT}/wsi_masks/model_${MEMBER}"

export nnUNet_results="/work2/bz66izin-VSproject/nnunet/nnUNet_results"
export nnUNet_raw="/work2/bz66izin-VSproject/nnunet/nnUNet_raw"

echo "TASK_ID=${TASK_ID}  BLOCK=${RANGE_TAG}  MEMBER=${MEMBER}  WSI=${WSI_FOLDER}"

# -----------------------------
# Resolve the WSI filename from the metadata
# -----------------------------
if [ ! -f "${WSI_METADATA}" ]; then
    echo "[ERROR] Metadata CSV not found: ${WSI_METADATA}"
    exit 1
fi

WSI_TIF=$(python3 -c "
import pandas as pd
print(pd.read_csv('${WSI_METADATA}')['source_file'].iloc[0])
")
WSI_STEM="${WSI_TIF%.tif}"
OUT_MASK="${WSI_STEM}.tif"

# nnUNet expects a _0000 channel suffix. Accept either naming so this works
# whether or not reconstruct.py already applied it.
RECON_TIF="${RECON_DIR}/${WSI_STEM}_0000.tif"
if [ ! -f "${RECON_TIF}" ]; then
    RECON_TIF="${RECON_DIR}/${WSI_STEM}.tif"
fi

echo "WSI stem  : ${WSI_STEM}"
echo "Recon TIF : ${RECON_TIF}"
echo "Output    : ${OUT_DIR}/${OUT_MASK}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${RECON_TIF}" ]; then
    echo "[ERROR] Reconstructed WSI not found: ${RECON_DIR}/${WSI_STEM}{_0000,}.tif"
    echo "        Run recon_ensemble_ugac.sh first (same array index)."
    exit 1
fi

if [ -f "${OUT_DIR}/${OUT_MASK}" ]; then
    echo "[SKIP] Mask already present: ${OUT_DIR}/${OUT_MASK}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

# nnUNet predicts over a directory, so stage this single WSI in a temp dir
TMP_BASE=$(mktemp -d)
TMP_IN="${TMP_BASE}/in"
TMP_OUT="${TMP_BASE}/out"
mkdir -p "${TMP_IN}" "${TMP_OUT}"
cleanup() { rm -rf "${TMP_BASE}"; }
trap cleanup EXIT

cp "${RECON_TIF}" "${TMP_IN}/${WSI_STEM}_0000.tif"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i "${TMP_IN}" \
    -o "${TMP_OUT}" \
    -f 0 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    -npp 1 \
    -nps 1 \
    -device cuda

mv "${TMP_OUT}/${OUT_MASK}" "${OUT_DIR}/${OUT_MASK}"

echo "Done. ${RANGE_TAG} member ${MEMBER} WSI ${WSI_FOLDER} → ${OUT_DIR}/${OUT_MASK}"
