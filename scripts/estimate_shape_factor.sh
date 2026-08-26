#!/bin/bash
#SBATCH --job-name=i2i_w2_shape
#SBATCH --output=logs_cali/w2_shape_%j.out
#SBATCH --error=logs_cali/w2_shape_%j.err

#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# W-2: is the 0.80 sigma reference line justified, or is it Gaussian assumed?
#
# Reads ONE flat CSV. The signed residual mu - real is recoverable from it even
# though only |mu - real| is stored as `error`, which is the single fact that
# makes W-2 a re-read rather than a re-run.
#
# RUN IT TWICE, once per cohort — the kidney arm has its own published block and
# comparing it against the liver numbers is meaningless:
#
#   sbatch scripts/estimate_shape_factor.sh
#   sbatch --export=ALL,TABLE=/path/kidney/per_region_calibration.csv,\
#          OUTDIR=/path/w2_kidney,COHORT=kidney scripts/estimate_shape_factor.sh
#
# The PIXEL protocol cannot be run at all: evaluation.py:633 computes
# torch.abs(x - x_prime).mean(dim=1) before saving, so the sign is destroyed at
# source. The script exits with that explanation rather than scoring something
# else. Answering W-2 there needs that line changed and the regen-error stage
# re-run, which is a separate job.

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

mkdir -p logs_cali

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — all overridable via --export
# -----------------------------
TABLE="${TABLE:-/work2/bz66izin-UC_project/ID_HE/calibration_phi/per_region_calibration.csv}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/w2_shape_factor}"

DESCRIPTOR="${DESCRIPTOR:-task_specific_value}"
# Which published block the acceptance checks compare against: liver, kidney, or
# none. Getting this wrong produces a page of spurious NOs.
COHORT="${COHORT:-liver}"
MIN_REGIONS="${MIN_REGIONS:-10}"
N_BOOT="${N_BOOT:-2000}"

echo "table  : ${TABLE}"
echo "outdir : ${OUTDIR}"
echo "cohort : ${COHORT}"

if [[ ! -f "${TABLE}" ]]; then
    echo "ERROR: ${TABLE} does not exist."
    echo "  Written by calibrate_phi.py (per_region_calibration.csv) or by"
    echo "  compare_uncertainty_sources.py (per_region_sources.csv). It must"
    echo "  carry both mu AND real."
    exit 1
fi

mkdir -p "${OUTDIR}"

python "${PROJECT_ROOT}/estimate_shape_factor.py" \
    --table "${TABLE}" \
    --outdir "${OUTDIR}" \
    --descriptor "${DESCRIPTOR}" \
    --cohort "${COHORT}" \
    --min_regions_per_slide "${MIN_REGIONS}" \
    --n_boot "${N_BOOT}" \
    --seed 0

echo "Done. Read the confound block before quoting a distribution name: a low"
echo "kappa is heavy tails OR a sigma that mis-tracks the local error scale,"
echo "and only the kappa(u)/kappa(r) gap separates them."
