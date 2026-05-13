#!/bin/bash
#SBATCH --job-name=i2i_plot_eval
#SBATCH --output=logs_plot_eval/plot_eval_%j.out
#SBATCH --error=logs_plot_eval/plot_eval_%j.err

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

mkdir -p logs_plot_eval

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these before submitting
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT=I2I-Stain-Zoo

# Root evaluation directory: must contain one subdirectory per model, each with
# the per-metric CSVs written by eval_all_54.sh or eval_all_configs.sh.
# Both path layouts are tried automatically:
#   ${INDIR}/${model}/data_${d}/model_${s}/{metric}.csv          (eval_all_54.sh)
#   ${INDIR}/${model}/results/data_${d}/model_${s}/{metric}.csv  (eval_all_configs.sh)
INDIR=/work2/bz66izin-VSproject/Eval

# Output directory for all_configs.csv, best_per_model.csv, and metrics.png
OUTDIR=/work2/bz66izin-VSproject/eval_plots

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d "${INDIR}" ]; then
    echo "[ERROR] Evaluation directory not found: ${INDIR}"
    echo "        Run eval_all_54.sh first."
    exit 1
fi

# Skip if already done
if [ -f "${OUTDIR}/metrics.png" ]; then
    echo "[SKIP] metrics.png already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "Input  : ${INDIR}"
echo "Output : ${OUTDIR}"

python "${PROJECT_ROOT}/plot_eval_metrics.py" \
    --indir  "${INDIR}" \
    --outdir "${OUTDIR}"

echo "Done. Outputs saved to ${OUTDIR}"
