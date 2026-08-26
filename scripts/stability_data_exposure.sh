#!/bin/bash
#SBATCH --job-name=i2i_w29_stab
#SBATCH --output=logs_cali/w29_stab_%j.out
#SBATCH --error=logs_cali/w29_stab_%j.err

#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# W-29: how stable is the data-exposure component with K = 5 subsets?
#
# Reads per_region.csv from the CROSSED grid run and its summary.json beside it
# (for n_seeds_per_fold — S is not recoverable from the CSV). Nothing else.
#
# The main deliverable is DISCLOSURE, not a number: procedural carries K(S-1) =
# 45 df and data exposure carries K-1 = 4, so the two components of one
# decomposition are estimated to precisions differing by about 3.4x. That is a
# property of the design and no resampling of the existing grid can change it.
# The run confirms both figures against the actual grid rather than assuming
# them.
#
# The seed-dimension contrast is PARAMETRIC by default, because per-member phi
# is not on disk: compute_phi_uncertainty.py holds the member block in memory
# and writes only the fold summaries. Set MEMBER_NPZ to an exact dump if one is
# ever produced — the format is one [n_seeds, n_regions] array per fold under
# keys fold1..foldK, which is a one-line addition there.

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
PER_REGION="${PER_REGION:-/work2/bz66izin-UC_project/ID_HE/phi_uncertainty/agg_phi/per_region.csv}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/w29_stability}"

DESCRIPTOR="${DESCRIPTOR:-task_specific_value}"
# Only needed if summary.json is missing beside PER_REGION.
SEEDS_PER_FOLD="${SEEDS_PER_FOLD-}"
# Exact seed contrast, if a member-level dump exists. Empty = parametric.
MEMBER_NPZ="${MEMBER_NPZ-}"
N_DRAWS="${N_DRAWS:-200}"
N_BOOT="${N_BOOT:-2000}"

echo "per_region : ${PER_REGION}"
echo "outdir     : ${OUTDIR}"
echo "member npz : ${MEMBER_NPZ:-<none, seed contrast is parametric>}"

if [[ ! -f "${PER_REGION}" ]]; then
    echo "ERROR: ${PER_REGION} does not exist."
    echo "  It is written by compute_phi_uncertainty.py run with one --fold per"
    echo "  training subset, or pooled by aggregate_phi_uncertainty.py. A single"
    echo "  --ensemble run has no data-exposure term at all."
    exit 1
fi
if [[ ! -f "$(dirname "${PER_REGION}")/summary.json" && -z "${SEEDS_PER_FOLD}" ]]; then
    echo "ERROR: no summary.json beside ${PER_REGION} and SEEDS_PER_FOLD unset."
    echo "  S is not recorded in per_region.csv. Set e.g."
    echo "  SEEDS_PER_FOLD='10 10 10 10 10'"
    exit 1
fi

mkdir -p "${OUTDIR}"

EXTRA=()
if [[ -n "${SEEDS_PER_FOLD}" ]]; then
    EXTRA+=(--seeds_per_fold ${SEEDS_PER_FOLD})
fi
if [[ -n "${MEMBER_NPZ}" ]]; then
    EXTRA+=(--member_npz "${MEMBER_NPZ}")
fi

# Guarded expansion: `set -u` is on and bash 3.2 treats an empty array as unbound.
python "${PROJECT_ROOT}/stability_data_exposure.py" \
    --per_region "${PER_REGION}" \
    --outdir "${OUTDIR}" \
    --descriptor "${DESCRIPTOR}" \
    --n_draws "${N_DRAWS}" \
    --n_boot "${N_BOOT}" \
    --seed 0 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done. Read the reconstruction gate at the top of the log first: it is what"
echo "proves the recomputation runs the paper's estimator. The spec's bracketing"
echo "check is reported but is NOT a diagnostic — see the note beside it."
