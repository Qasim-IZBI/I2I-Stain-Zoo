#!/bin/bash
#SBATCH --job-name=i2i_plot_combined
#SBATCH --output=logs_plot_combined/plot_combined_%j.out
#SBATCH --error=logs_plot_combined/plot_combined_%j.err

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_plot_combined

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these before submitting
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT=I2I-Stain-Zoo

# Root evaluation directory with per-model metric CSVs
# (produced by eval_all_54.sh or eval_all_configs.sh)
EVAL_INDIR=/work2/bz66izin-VSproject/Eval

# PSR comparison directory with per-model summary.json files
# (produced by compare_psr_all_configs.sh)
PSR_INDIR=/work2/bz66izin-VSproject/psr_comparison

# Output directory for the combined figure
OUTDIR=/work2/bz66izin-VSproject/combined_metrics_plot

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d "${EVAL_INDIR}" ]; then
    echo "[ERROR] Eval directory not found: ${EVAL_INDIR}"
    echo "        Run eval_all_54.sh or eval_all_configs.sh first."
    exit 1
fi

if [ ! -d "${PSR_INDIR}" ]; then
    echo "[ERROR] PSR comparison directory not found: ${PSR_INDIR}"
    echo "        Run compare_psr_all_configs.sh first."
    exit 1
fi

# Skip if already done
if [ -f "${OUTDIR}/combined_metrics.png" ]; then
    echo "[SKIP] combined_metrics.png already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "Eval input : ${EVAL_INDIR}"
echo "PSR input  : ${PSR_INDIR}"
echo "Output     : ${OUTDIR}"

python "${PROJECT_ROOT}/plot_combined_metrics.py" \
    --eval_indir "${EVAL_INDIR}" \
    --psr_indir  "${PSR_INDIR}" \
    --outdir     "${OUTDIR}"

echo "Done. Figure saved to ${OUTDIR}/combined_metrics.png"
