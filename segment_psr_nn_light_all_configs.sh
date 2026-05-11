#!/bin/bash
#SBATCH --job-name=i2i_seg_nn_light_all
#SBATCH --output=logs_seg_nn_light/seg_%A_%a.out
#SBATCH --error=logs_seg_nn_light/seg_%A_%a.err

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --array=0-53   # 54 jobs = 6 models x 3 model-sizes x 3 data-sizes

set -euo pipefail
set -x

mkdir -p logs_seg_nn_light

echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"

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
# Axes  (6 models x 3 model-sizes x 3 data-sizes = 54 jobs)
# Decomposition matches recon_all_configs.sh
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")
SIZES=("small" "medium" "large")
DATASIZES=("small" "medium" "large")

TASK_ID=${SLURM_ARRAY_TASK_ID}

SIZE_ID=$(( TASK_ID / 18 ))
DATA_ID=$(( (TASK_ID % 18) / 6 ))
MODEL_ID=$(( TASK_ID % 6 ))

MODEL=${MODELS[$MODEL_ID]}
SIZE=${SIZES[$SIZE_ID]}
DATASIZE=${DATASIZES[$DATA_ID]}

echo "TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "MODEL_SIZE=${SIZE}"
echo "DATA_SIZE=${DATASIZE}"

# -----------------------------
# Fixed paths
# -----------------------------
RECON_BASE=/work2/bz66izin-VSproject/reconstruction_750k
SEG_BASE=/work2/bz66izin-VSproject/psr_masks

# Reconstructed WSI TIFs from recon_all_configs.sh
IN_DIR=${RECON_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/reconstructed

# WSI-level mask output (flat directory of nnUNet predictions)
OUT_DIR=${SEG_BASE}/${MODEL}/results/data_${DATASIZE}/model_${SIZE}/wsi_masks

# WSI range used during reconstruction (must match recon_all_configs.sh)
RANGE_START=1
RANGE_END=5
N_EXPECTED=$(( RANGE_END - RANGE_START + 1 ))

# nnUNet model settings
export nnUNet_results="/work2/bz66izin-VSproject/nnunet/nnUNet_results"
export nnUNet_raw="/work2/bz66izin-VSproject/nnunet/nnUNet_raw"

# -----------------------------
# Pre-flight checks
# -----------------------------

# 1. Reconstructed WSIs must exist and be non-empty
if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Reconstructed WSIs not found or empty: ${IN_DIR} — skipping."
    exit 1
fi

# 2. Skip if expected number of predicted WSI masks already present
if [ -d "${OUT_DIR}" ]; then
    N_DONE=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_DONE}" -ge "${N_EXPECTED}" ]; then
        echo "[SKIP] ${N_DONE}/${N_EXPECTED} WSI masks already present in ${OUT_DIR}. Exiting."
        exit 0
    fi
    if [ "${N_DONE}" -gt 0 ]; then
        echo "[WARN] Partial segmentation detected (${N_DONE}/${N_EXPECTED} masks). Re-running."
    fi
fi

mkdir -p "${OUT_DIR}"

echo "Input dir : ${IN_DIR}"
echo "Output dir: ${OUT_DIR}"

# -----------------------------
# Run nnUNet WSI-level prediction
# -----------------------------
run_cmd nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i "${IN_DIR}" \
    -o "${OUT_DIR}" \
    -f 0 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    -npp 1 \
    -nps 1

echo "Done. WSI PSR masks (nn_light) saved to ${OUT_DIR}"
