#!/bin/bash
#SBATCH --job-name=i2i_wsi_uncertainty
#SBATCH --output=logs_ensemble/wsi_uncertainty_%j.out
#SBATCH --error=logs_ensemble/wsi_uncertainty_%j.err

#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Aggregates tile-level uncertainty CSVs (from aggregate_uncertainty.sh) to
# one mean per WSI per model.
#
# Prerequisite: aggregate_uncertainty.sh must have completed for all 6 models.
#
# Output: {OUTDIR}/wsi_uncertainty.csv  (30 rows = 6 models x 5 WSIs)
# Joins with mean_cpa_mae.csv on (model, wsi) for calibration analysis.

set -eo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_ensemble

BASE=/work2/bz66izin-VSproject/ensemble
OUTDIR=/work2/bz66izin-VSproject/ensemble_cpa_comparison/aggregated

# Pre-flight: at least one per_wsi_csv directory must exist
if [ -z "$(find "${BASE}" -type d -name "per_wsi_csv" 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No per_wsi_csv directories found under ${BASE}"
    echo "        Run aggregate_uncertainty.sh first."
    exit 1
fi

# Skip guard
if [ -f "${OUTDIR}/wsi_uncertainty.csv" ]; then
    echo "[SKIP] wsi_uncertainty.csv already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/compute_wsi_uncertainty.py \
    --base   "${BASE}" \
    --outdir "${OUTDIR}"

echo "Done. wsi_uncertainty.csv saved to ${OUTDIR}"
