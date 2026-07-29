#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_ugac
#SBATCH --output=logs_ensemble_ugac/cyclegan_ugac_train_%A_%a.out
#SBATCH --error=logs_ensemble_ugac/cyclegan_ugac_train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4   # 5 ensemble members

# Trains a 5-member CycleGAN ensemble with UGAC aleatoric uncertainty heads
# (Upadhyay et al., NeurIPS 2021) at the SMALL generator size on the full
# training set. Members differ only by --seed.
#
# Each member yields both uncertainty components:
#   aleatoric  per-member, closed form from the GGD heads
#              (inference.py --save_aleatoric)
#   epistemic  variance across the 5 members (uncertainty.py)
#
# Note: --cyclegan_ugac replaces the L1 cycle loss with the GGD NLL, so these
# checkpoints are NOT comparable to the vanilla ensemble in
# train_ensemble_cyclegan.sh without retraining that one too.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_ugac.sh

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

# -----------------------------
# Config: small model, large data, UGAC
# -----------------------------
MEMBER=$(printf "%02d" $((SLURM_ARRAY_TASK_ID + 1)))   # 01 … 05
SEED=$((SLURM_ARRAY_TASK_ID + 1))                       # 1  … 5

PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

# large data size — folders 001–030 (100% training fraction)
DATA_RANGE="1,30"

# small CycleGAN: ngf=64, n_blocks=8  (~10M A→B params)
# UGAC heads add ~6.3k params, so the size budget is unchanged.
NGF=64
NBLOCKS=8

STEPS=750000

OUTPUT_DIR=/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/data_large/model_small/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${OUTPUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${OUTPUT_DIR}/checkpoints/step_${STEPS}.pt" ]; then
    echo "[SKIP] step_${STEPS}.pt already present — member ${MEMBER} is done."
    exit 0
fi

for D in "${DATA_A}" "${DATA_B}"; do
    if [ ! -d "${D}" ]; then
        echo "[ERROR] Data directory not found: ${D}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

# train.py auto-resumes from the latest checkpoint under ${OUTPUT_DIR}/checkpoints/,
# so a requeued or timed-out job continues rather than restarting.
run_cmd python "${PROJECT_ROOT}/train.py" \
    --model cyclegan \
    --cyclegan_ugac \
    --steps "${STEPS}" \
    --cyclegan_ngf "${NGF}" \
    --cyclegan_n_blocks "${NBLOCKS}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --data_range "${DATA_RANGE}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}"

echo "Done. UGAC member ${MEMBER} saved to ${OUTPUT_DIR}"
