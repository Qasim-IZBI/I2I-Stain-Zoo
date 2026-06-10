#!/bin/bash
#SBATCH --job-name=i2i_mean_cpa_mae
#SBATCH --output=logs_ensemble/mean_cpa_mae_%j.out
#SBATCH --error=logs_ensemble/mean_cpa_mae_%j.err

#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Computes mean CPA-MAE per WSI per model from compare_psr_ensemble.sh output.
# Reads per_wsi.csv for each model and pivots to:
#   mean_cpa_mae[wsi] = mean over members of |PSR_fraction_member - PSR_fraction_real|
#
# Prerequisite: compare_psr_ensemble.sh must have completed for all 6 models.
#
# Output: {OUTDIR}/mean_cpa_mae.csv  (30 rows = 6 models x 5 WSIs)

set -eo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_ensemble

CPA_BASE=/work2/bz66izin-VSproject/ensemble_cpa_comparison
OUTDIR=/work2/bz66izin-VSproject/ensemble_cpa_comparison/aggregated

# Pre-flight: at least one per_wsi.csv must exist
if [ -z "$(find "${CPA_BASE}" -name "per_wsi.csv" 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No per_wsi.csv files found under ${CPA_BASE}"
    echo "        Run compare_psr_ensemble.sh first."
    exit 1
fi

# Skip guard
if [ -f "${OUTDIR}/mean_cpa_mae.csv" ]; then
    echo "[SKIP] mean_cpa_mae.csv already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/compute_mean_cpa_mae.py \
    --cpa_base "${CPA_BASE}" \
    --outdir   "${OUTDIR}"

echo "Done. mean_cpa_mae.csv saved to ${OUTDIR}"
