#!/bin/bash
#SBATCH --job-name=i2i_aggregate_calib
#SBATCH --output=logs_ensemble/aggregate_calib_%j.out
#SBATCH --error=logs_ensemble/aggregate_calib_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Aggregates per-WSI calibration CSVs (written by run_calibration_all.sh) into
# per-model summaries and a single all_models.csv comparison table.
# Inputs consumed:
#   ensemble/{MODEL}/data_large/{MODEL_SIZE}/calibration/{MODEL}/wsi{NNN}/per_tile.csv
# Outputs written to:
#   {OUTDIR}/{MODEL}/summary.json
#   {OUTDIR}/{MODEL}/calibration.png
#   {OUTDIR}/all_models.csv

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble

BASE="/work2/bz66izin-VSproject/ensemble"
OUTDIR="/work2/bz66izin-VSproject/ensemble/calibration_combined"

# Pre-flight: check at least one per_tile.csv exists
FOUND=$(find "${BASE}" -path "*/calibration/*/wsi*/per_tile.csv" 2>/dev/null | wc -l)
if [ "${FOUND}" -eq 0 ]; then
    echo "[ERROR] No per_tile.csv files found under ${BASE}"
    echo "        Run run_calibration_all.sh first."
    exit 1
fi
echo "Found ${FOUND} per_tile.csv file(s) to aggregate."

# Skip guard
if [ -f "${OUTDIR}/all_models.csv" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}/all_models.csv"
    exit 0
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/aggregate_calibration.py \
    --base   "${BASE}" \
    --outdir "${OUTDIR}"

echo "Done. Aggregated calibration results saved to ${OUTDIR}"
