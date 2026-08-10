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
#SBATCH --array=0-49   # 50 jobs = 5 data ranges x 10 ensemble members

# Trains CycleGAN ensembles with UGAC aleatoric uncertainty heads
# (Upadhyay et al., NeurIPS 2021) at the SMALL generator size.
#
# Five DISJOINT data blocks of 7 specimens each, 10 seeds per block:
#
#   tasks  0– 9  ->  folders 001–007   members 01–10
#   tasks 10–19  ->  folders 008–014   members 01–10
#   tasks 20–29  ->  folders 015–021   members 01–10
#   tasks 30–39  ->  folders 022–028   members 01–10
#   tasks 40–49  ->  folders 029–035   members 01–10
#
# Because the blocks are disjoint rather than nested, differences across blocks
# reflect WHICH slides were seen, not how many — the opposite of the nested
# 25/50/100% fractions used in the scaling study.
#
# Each member yields both uncertainty components:
#   aleatoric  per-member, closed form from the GGD heads
#              (inference.py --save_aleatoric)
#   epistemic  variance across the 10 members within a block (uncertainty.py)
#
# NOTE ON THE LAST BLOCK: the training set is folders 001–030, so 029–035
# requires folders 031–035 to exist under trainA/ and trainB/. The pre-flight
# check below fails fast with an explicit message if they do not, rather than
# letting a 48h job die inside the dataloader.
#
# Note: --cyclegan_ugac replaces the L1 cycle loss with the GGD NLL, so these
# checkpoints are NOT comparable to the vanilla ensemble in
# train_ensemble_cyclegan.sh without retraining that one too.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_ugac.sh
#
# To cap how many run at once (50 x 48h GPU is a large allocation):
#   sbatch --array=0-49%10 I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_ugac.sh
# To run a single block, e.g. folders 008–014:
#   sbatch --array=10-19  I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_ugac.sh

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
# 2D decomposition: 5 data blocks x 10 members
# -----------------------------
N_MEMBERS=10

RANGE_ID=$(( SLURM_ARRAY_TASK_ID / N_MEMBERS ))    # 0 … 4
MEMBER_ID=$(( SLURM_ARRAY_TASK_ID % N_MEMBERS ))   # 0 … 9

RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

RANGE_START=${RANGE_STARTS[$RANGE_ID]}
RANGE_END=${RANGE_ENDS[$RANGE_ID]}
DATA_RANGE="${RANGE_START},${RANGE_END}"
RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_START}" "${RANGE_END}")

MEMBER=$(printf "%02d" $((MEMBER_ID + 1)))   # 01 … 10
SEED=$((MEMBER_ID + 1))                       # 1  … 10

PROJECT_ROOT=I2I-Stain-Zoo
DATA_DIR=/work2/bz66izin-VSproject/VS_Data
DATA_A="${DATA_DIR}/QP_HE/tiles/trainA/"
DATA_B="${DATA_DIR}/QP_SR/tiles/trainB/"

# small CycleGAN: ngf=64, n_blocks=8  (~10.2M A→B params)
# UGAC heads add ~6.3k params, so the size budget is unchanged.
NGF=64
NBLOCKS=8

STEPS=750000

OUTPUT_DIR=/work2/bz66izin-VSproject/ensemble_ugac/cyclegan/${RANGE_TAG}/model_small/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  RANGE=${DATA_RANGE} (${RANGE_TAG})  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${OUTPUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${OUTPUT_DIR}/checkpoints/step_${STEPS}.pt" ]; then
    echo "[SKIP] step_${STEPS}.pt already present — ${RANGE_TAG} member ${MEMBER} is done."
    exit 0
fi

# Every folder in the range must exist: datasets/common.py raises
# FileNotFoundError on the first missing one, so catch it here instead.
MISSING=""
for D in "${DATA_A}" "${DATA_B}"; do
    if [ ! -d "${D}" ]; then
        echo "[ERROR] Data directory not found: ${D}"
        exit 1
    fi
    for i in $(seq "${RANGE_START}" "${RANGE_END}"); do
        SUB=$(printf "%s/%03d/images" "${D%/}" "${i}")
        [ -d "${SUB}" ] || MISSING="${MISSING}\n    ${SUB}"
    done
done
if [ -n "${MISSING}" ]; then
    echo "[ERROR] --data_range ${DATA_RANGE} names folders that do not exist:"
    printf "%b\n" "${MISSING}"
    echo "        The training set is folders 001-030; a range reaching past 030"
    echo "        needs those specimens tiled into trainA/ and trainB/ first."
    exit 1
fi

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

echo "Done. UGAC ${RANGE_TAG} member ${MEMBER} saved to ${OUTPUT_DIR}"
