#!/bin/bash
#SBATCH --job-name=i2i_w30_rank
#SBATCH --output=logs_cali/w30_rank_%j.out
#SBATCH --error=logs_cali/w30_rank_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# W-30: does the spread add anything to the point prediction?
#
# Reads ONE flat CSV and nothing else — no masks, no reconstructions, no model.
# Half an hour and 16G is generous; the whole thing is a few thousand rows and
# a bootstrap. It is a SLURM script only because that is where the CSVs live.
#
# The input is the table the paper's own risk-coverage figure is built from:
#
#   compare_uncertainty_sources.py -> per_region_sources.csv   <- USE THIS ONE
#   calibrate_phi.py               -> per_region_calibration.csv
#
# Both have the columns this needs. The first is what make_paper_figures.py
# consumes, so it is the one whose numbers --check_published expects; running
# against the second and then finding the acceptance checks fail would send
# someone hunting for a bug that is only a choice of table.

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
TABLE="${TABLE:-/work2/bz66izin-UC_project/ID_HE/compare_sources/per_region_sources.csv}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/w30_ranking_rules}"

DESCRIPTOR="${DESCRIPTOR:-task_specific_value}"
# The deployment baseline W-30 asks about. Change it only to answer a different
# question than the one the supervisor raised.
REFERENCE_RULE="${REFERENCE_RULE:-mu}"
COVERAGES="${COVERAGES:-1.0 0.9 0.8 0.7 0.6 0.5}"
N_BOOT="${N_BOOT:-2000}"
MIN_REGIONS="${MIN_REGIONS:-10}"
# The eight published numbers are the LIVER arm's. Set to 0 for kidney, where
# comparing against them is meaningless and the warnings would be noise.
CHECK_PUBLISHED="${CHECK_PUBLISHED:-1}"

echo "table  : ${TABLE}"
echo "outdir : ${OUTDIR}"
echo "descr  : ${DESCRIPTOR}"
echo "ref    : ${REFERENCE_RULE}"

if [[ ! -f "${TABLE}" ]]; then
    echo "ERROR: ${TABLE} does not exist."
    echo "  It is written by compare_uncertainty_sources.py (per_region_sources.csv)"
    echo "  or calibrate_phi.py (per_region_calibration.csv). Point TABLE= at one."
    exit 1
fi

mkdir -p "${OUTDIR}"

# Guarded expansion below: `set -u` is on and bash 3.2 treats an empty
# array as unbound, which would kill the job for the kidney arm only.
EXTRA=()
if [[ "${CHECK_PUBLISHED}" == "1" ]]; then
    EXTRA+=(--check_published)
fi

python "${PROJECT_ROOT}/compare_ranking_rules.py" \
    --table "${TABLE}" \
    --outdir "${OUTDIR}" \
    --descriptor "${DESCRIPTOR}" \
    --reference_rule "${REFERENCE_RULE}" \
    --coverages ${COVERAGES} \
    --min_regions_per_slide "${MIN_REGIONS}" \
    --n_boot "${N_BOOT}" \
    --seed 0 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done. The claim is in ${OUTDIR}/rank_rule_deltas.csv — the paired"
echo "difference against '${REFERENCE_RULE}', not the two marginal curves."
