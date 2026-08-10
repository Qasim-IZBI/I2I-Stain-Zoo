#!/bin/bash
#SBATCH --job-name=i2i_ens_cg_grid
#SBATCH --output=logs_ensemble_grid/cyclegan_grid_train_%A_%a.out
#SBATCH --error=logs_ensemble_grid/cyclegan_grid_train_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-49   # 50 jobs = K=5 subsets x S=10 seeds

# The crossed (subset x seed) grid behind the uncertainty decomposition:
# K = 5 disjoint training subsets, S = 10 seeds, M = KS = 50 members of
# VANILLA CycleGAN at the small generator size.
#
#   tasks  0– 9  ->  folders 001–007   seeds 1–10
#   tasks 10–19  ->  folders 008–014   seeds 1–10
#   tasks 20–29  ->  folders 015–021   seeds 1–10
#   tasks 30–39  ->  folders 022–028   seeds 1–10
#   tasks 40–49  ->  folders 029–035   seeds 1–10
#
# WHY THE GRID IS CROSSED. Members sharing a subset differ only in seed, so
# their spread is procedural. Subset means differ because different slides were
# seen, so their spread is data exposure. The law of total variance separates
# the two exactly as they are indexed — but only if both factors vary, which is
# what train_ensemble_cyclegan.sh (one subset, 10 seeds) cannot do.
#
# The subsets are DISJOINT rather than nested, so a difference between subsets
# reflects WHICH slides were seen, not how many — the opposite of the nested
# 25/50/100% fractions in the scaling study.
#
# VANILLA, NOT UGAC. This supersedes train_ensemble_cyclegan_ugac.sh, whose
# GGD-NLL cycle loss did not produce usable virtual stain (2026-08-09). The
# decomposition identity carries no aleatoric term, so nothing downstream needs
# the UGAC heads. Those scripts are kept for provenance; do not mix outputs.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_grid.sh
#
# To cap how many run at once (50 x 48h GPU is a large allocation):
#   sbatch --array=0-49%10 I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_grid.sh
# To run a single subset, e.g. folders 008–014:
#   sbatch --array=10-19  I2I-Stain-Zoo/scripts/train_ensemble_cyclegan_grid.sh

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

# -----------------------------
# 2D decomposition: K=5 subsets x S=10 seeds
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
# Chosen so 50 members stay tractable, and so the data budget still binds:
# 7 of 35 cases sits near the smallest fraction of the scaling grid, where
# translation quality still responds to data. Data-exposure variance is only
# observable where that is true.
NGF=64
NBLOCKS=8

STEPS=750000

OUTPUT_DIR=/work2/bz66izin-VSproject/ensemble_grid/cyclegan/${RANGE_TAG}/model_small/models/model_${MEMBER}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  SUBSET=${DATA_RANGE} (${RANGE_TAG})  MEMBER=${MEMBER}  SEED=${SEED}"
echo "Output : ${OUTPUT_DIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ -f "${OUTPUT_DIR}/checkpoints/step_${STEPS}.pt" ]; then
    echo "[SKIP] step_${STEPS}.pt already present — ${RANGE_TAG} member ${MEMBER} is done."
    exit 0
fi

# Every folder in the subset must exist: datasets/common.py raises
# FileNotFoundError on the first missing one, so catch it here instead of
# letting a 48h job die inside the dataloader.
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
    echo "        The grid spans folders 001-035 across five subsets of seven."
    echo "        Subset 5 (029-035) needs 031-035, which post-date the 001-030"
    echo "        BMVC training split and must be tiled into trainA/ and trainB/."
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
    --steps "${STEPS}" \
    --cyclegan_ngf "${NGF}" \
    --cyclegan_n_blocks "${NBLOCKS}" \
    --dataA "${DATA_A}" \
    --dataB "${DATA_B}" \
    --data_range "${DATA_RANGE}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}"

echo "Done. ${RANGE_TAG} member ${MEMBER} saved to ${OUTPUT_DIR}"
