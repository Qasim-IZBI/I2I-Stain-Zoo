#!/bin/bash
#SBATCH --job-name=i2i_unc_boxplot
#SBATCH --output=logs_ensemble/unc_boxplot_%j.out
#SBATCH --error=logs_ensemble/unc_boxplot_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_ensemble

BASE="/work2/bz66izin-VSproject/ensemble"
OUTDIR="/work2/bz66izin-VSproject/uncertainty_boxplot"

# Pre-flight: at least one per_wsi_csv directory must exist
if [ -z "$(find "${BASE}" -type d -name "per_wsi_csv" 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No per_wsi_csv directories found under ${BASE}"
    echo "        Run aggregate_uncertainty.sh first."
    exit 1
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/plot_uncertainty_boxplot.py \
    --base   "${BASE}" \
    --outdir "${OUTDIR}"

echo "Done. Figure saved to ${OUTDIR}/uncertainty_boxplot.png"
