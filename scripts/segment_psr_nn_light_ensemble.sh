#!/bin/bash
#SBATCH --job-name=i2i_seg_ens_nn_light
#SBATCH --output=logs_seg_ensemble/seg_%A_%a.out
#SBATCH --error=logs_seg_ensemble/seg_%A_%a.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-299   # 300 jobs = 6 models × 10 ensemble members × 5 test WSIs

set -eo pipefail
set -x

mkdir -p logs_seg_ensemble

echo "Host: $(hostname)"

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

# -----------------------------
# Helper: echo and run a command
# -----------------------------
run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# -----------------------------
# Axes  (6 models × 10 members × 5 WSIs = 300 jobs)
# MODEL_ID  = TASK_ID / 50        (0–5)
# MEMBER_ID = (TASK_ID % 50) / 5  (0–9 → member 01–10)
# WSI_ID    = TASK_ID % 5         (0–4 → folder 001–005)
# Decomposition matches recon_ensemble_A2B.sh
# -----------------------------
MODELS=(      "cyclegan"      "unit"         "munit"        "dclgan"      "uvcgan"      "cyclediffusion")
MODEL_SIZES=( "model_medium"  "model_medium" "model_medium" "model_small" "model_small" "model_small")

TASK_ID=${SLURM_ARRAY_TASK_ID}

MODEL_ID=$(( TASK_ID / 50 ))
MEMBER_ID=$(( (TASK_ID % 50) / 5 ))
WSI_ID=$(( TASK_ID % 5 ))

MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))       # 01 … 10
WSI_FOLDER=$(printf "%03d" $(( WSI_ID + 1 )))      # 001 … 005

MODEL=${MODELS[$MODEL_ID]}
MODEL_SIZE=${MODEL_SIZES[$MODEL_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "MEMBER=${MEMBER}"
echo "WSI_FOLDER=${WSI_FOLDER}"

# -----------------------------
# Fixed paths
# -----------------------------

# Original testA tiles — tiles_metadata.csv holds source_file (the WSI TIF name)
TEST_A="/work2/bz66izin-VSproject/VS_Data/temp/tiles/testA"
WSI_METADATA="${TEST_A}/${WSI_FOLDER}/tiles_metadata.csv"

ENSEMBLE_ROOT="/work2/bz66izin-VSproject/ensemble/${MODEL}/data_large/${MODEL_SIZE}"

# Reconstructed WSI TIFs from recon_ensemble_A2B.sh
RECON_DIR="${ENSEMBLE_ROOT}/reconstructed/model_${MEMBER}"

# WSI-level mask output (one TIF per WSI, flat directory)
OUT_DIR="${ENSEMBLE_ROOT}/wsi_masks/model_${MEMBER}"

# nnUNet model settings (matches segment_psr_nn_light_all_configs.sh)
export nnUNet_results="/work2/bz66izin-VSproject/nnunet/nnUNet_results"
export nnUNet_raw="/work2/bz66izin-VSproject/nnunet/nnUNet_raw"

# -----------------------------
# Derive WSI TIF filename from tiles_metadata.csv
# source_file is the original WSI filename saved during tiling; reconstruct.py
# names its output identically, so RECON_DIR/{source_file} is the input TIF.
# -----------------------------
if [ ! -f "${WSI_METADATA}" ]; then
    echo "[ERROR] Metadata CSV not found: ${WSI_METADATA}"
    exit 1
fi

WSI_TIF=$(python3 -c "
import pandas as pd
df = pd.read_csv('${WSI_METADATA}')
print(df['source_file'].iloc[0])
")

# Strip .tif extension to get the stem
WSI_STEM="${WSI_TIF%.tif}"

# Reconstructed file uses _0000.tif suffix (nnUNet single-channel input convention)
RECON_TIF="${RECON_DIR}/${WSI_STEM}_0000.tif"

# nnUNet output mask: strips the _0000 channel suffix → plain stem.tif
OUT_MASK="${WSI_STEM}.tif"

echo "WSI stem  : ${WSI_STEM}"
echo "Recon TIF : ${RECON_TIF}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Skip gracefully if this member's reconstructed WSI does not exist yet
if [ ! -f "${RECON_TIF}" ]; then
    echo "[SKIP] Reconstructed WSI not found: ${RECON_TIF}"
    echo "       Run recon_ensemble_A2B.sh first (or member/WSI not yet processed)."
    exit 0
fi

# 2. Skip if this WSI's mask already exists in the output directory
if [ -f "${OUT_DIR}/${OUT_MASK}" ]; then
    echo "[SKIP] Mask already present: ${OUT_DIR}/${OUT_MASK}. Exiting."
    exit 0
fi

mkdir -p "${OUT_DIR}"

# -----------------------------
# Create a per-job temp directory for nnUNet (requires a directory as input,
# not a single file). Symlink the single WSI TIF into it, run prediction,
# then move the result to the shared output directory.
# Use SLURM_TMPDIR if available (local SSD), otherwise fall back to mktemp.
# -----------------------------
if [ -n "${SLURM_TMPDIR:-}" ]; then
    TMP_BASE="${SLURM_TMPDIR}/seg_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
    TMP_BASE=$(mktemp -d)
fi

TMP_IN="${TMP_BASE}/input"
TMP_OUT="${TMP_BASE}/output"
mkdir -p "${TMP_IN}" "${TMP_OUT}"

# Symlink the WSI TIF into the temp input directory with _0000 suffix
ln -s "${RECON_TIF}" "${TMP_IN}/${WSI_STEM}_0000.tif"

echo "Temp input : ${TMP_IN}"
echo "Temp output: ${TMP_OUT}"

# Ensure cleanup on exit (success or failure)
cleanup() {
    rm -rf "${TMP_BASE}"
}
trap cleanup EXIT

# -----------------------------
# Run nnUNet WSI-level prediction on the single-WSI temp directory
# -----------------------------
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

# Move the predicted mask to the shared output directory
mv "${TMP_OUT}/${OUT_MASK}" "${OUT_DIR}/${OUT_MASK}"

echo "Done. PSR mask for ${MODEL} member ${MEMBER} WSI ${WSI_FOLDER} → ${OUT_DIR}/${OUT_MASK}"
