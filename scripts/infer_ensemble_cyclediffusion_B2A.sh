#!/bin/bash
#SBATCH --job-name=i2i_ens_cd_b2a
#SBATCH --output=logs_ensemble/cyclediffusion_infer_B2A_%A_%a.out
#SBATCH --error=logs_ensemble/cyclediffusion_infer_B2A_%A_%a.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = 10 ensemble members × 5 test WSIs

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

# -----------------------------
# 2D decomposition: 10 members × 5 test WSIs
# Layout: tasks 0–4  → member_01 WSI 1–5
#         tasks 5–9  → member_02 WSI 1–5  …
#         tasks 45–49 → member_10 WSI 1–5
# -----------------------------
N_WSIS=5
MEMBER_ID=$(( SLURM_ARRAY_TASK_ID / N_WSIS ))   # 0 … 9
WSI_ID=$(( SLURM_ARRAY_TASK_ID % N_WSIS ))       # 0 … 4

MEMBER=$(printf "%02d" $(( MEMBER_ID + 1 )))     # 01 … 10
WSI_NUM=$(( WSI_ID + 1 ))                        # 1 … 5
WSI_FOLDER=$(printf "%03d" "${WSI_NUM}")         # 001 … 005
DATA_RANGE="${WSI_NUM},${WSI_NUM}"               # "1,1" … "5,5"

# CycleDiffusion DDIM denoising steps (must match A2B run)
CD_STEPS=200

PROJECT_ROOT=I2I-Stain-Zoo

ENSEMBLE_ROOT=/work2/bz66izin-VSproject/ensemble/cyclediffusion/data_large/model_small
CKPT="${ENSEMBLE_ROOT}/models/model_${MEMBER}/checkpoints/step_750000.pt"

# Input: A→B translated tiles produced by infer_ensemble_cyclediffusion.sh
IN_DIR="${ENSEMBLE_ROOT}/inference/model_${MEMBER}"

# Output: B→A re-translated tiles (A' = judge(B'))
OUTDIR="${ENSEMBLE_ROOT}/inference_B2A/model_${MEMBER}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}  WSI=${WSI_FOLDER}"
echo "Checkpoint : ${CKPT}"
echo "Input      : ${IN_DIR}/${WSI_FOLDER}/images/"
echo "Output     : ${OUTDIR}/${WSI_FOLDER}/images/"

# -----------------------------
# Pre-flight: checkpoint must exist
# -----------------------------
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run train_ensemble_cyclediffusion.sh first."
    exit 1
fi

# Pre-flight: A→B output must exist for this WSI
if [ ! -d "${IN_DIR}/${WSI_FOLDER}/images" ] || \
   [ -z "$(ls -A "${IN_DIR}/${WSI_FOLDER}/images" 2>/dev/null)" ]; then
    echo "[ERROR] A→B inference output missing or empty: ${IN_DIR}/${WSI_FOLDER}/images/"
    echo "        Run infer_ensemble_cyclediffusion.sh first."
    exit 1
fi

# Per-WSI skip guard: safe for parallel jobs because each WSI writes to its own subdir
if [ -d "${OUTDIR}/${WSI_FOLDER}/images" ] && \
   [ -n "$(ls -A "${OUTDIR}/${WSI_FOLDER}/images" 2>/dev/null)" ]; then
    echo "[SKIP] WSI ${WSI_FOLDER} B2A output already populated — skipping."
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# B→A: DDIM-invert B' tiles with eps_B, decode with eps_A
run_cmd python "${PROJECT_ROOT}/inference.py" \
    --model      cyclediffusion \
    --direction  B2A \
    --data       "${IN_DIR}" \
    --ckpt       "${CKPT}" \
    --data_range "${DATA_RANGE}" \
    --cd_steps   "${CD_STEPS}" \
    --outdir     "${OUTDIR}"

echo "Done. Member ${MEMBER} WSI ${WSI_FOLDER} B2A saved to ${OUTDIR}/${WSI_FOLDER}/"
