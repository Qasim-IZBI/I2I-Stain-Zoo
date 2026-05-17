#!/bin/bash
#SBATCH --job-name=i2i_psr_spearman
#SBATCH --output=logs_psr_spearman/psr_spearman_%j.out
#SBATCH --error=logs_psr_spearman/psr_spearman_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_psr_spearman

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these before submitting
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT=I2I-Stain-Zoo

# combined_metrics.csv produced by plot_combined_metrics.py
CSV=/work2/bz66izin-VSproject/combined_metrics_plot/combined_metrics.csv

# Output directory for psr_spearman.png and psr_spearman.csv
OUTDIR=/work2/bz66izin-VSproject/combined_metrics_plot

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${CSV}" ]; then
    echo "[ERROR] combined_metrics.csv not found: ${CSV}"
    echo "        Run plot_combined_metrics.sh first."
    exit 1
fi

# Skip if already done
if [ -f "${OUTDIR}/psr_spearman.png" ]; then
    echo "[SKIP] psr_spearman.png already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "Input  : ${CSV}"
echo "Output : ${OUTDIR}"

python "${PROJECT_ROOT}/plot_psr_spearman.py" \
    --csv    "${CSV}" \
    --outdir "${OUTDIR}"

echo "Done. Outputs saved to ${OUTDIR}"
