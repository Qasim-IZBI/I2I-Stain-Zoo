#!/bin/bash
#SBATCH --job-name=i2i_rank_psr
#SBATCH --output=logs_rank_psr/rank_psr_%A_%a.out
#SBATCH --error=logs_rank_psr/rank_psr_%A_%a.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-5   # 6 jobs — one per model type

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

mkdir -p logs_rank_psr

# -----------------------------
# Axes
# -----------------------------
MODELS=("cyclegan" "unit" "munit" "dclgan" "uvcgan" "cyclediffusion")

MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL}"

# -----------------------------
# Fixed paths
# -----------------------------
PROJECT_ROOT=I2I-Stain-Zoo

# Input: root containing one subdirectory per model, each with summary.json
# (produced by compare_psr_all_configs.sh)
INDIR=/work2/bz66izin-VSproject/psr_comparison

# Output: one subdirectory per model
OUTDIR=/work2/bz66izin-VSproject/psr_best_config/${MODEL}

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${INDIR}/${MODEL}/summary.json" ]; then
    echo "[ERROR] Missing: ${INDIR}/${MODEL}/summary.json"
    echo "        Run compare_psr_all_configs.sh first."
    exit 1
fi

# Skip if already done
if [ -f "${OUTDIR}/best_per_model.csv" ]; then
    echo "[SKIP] best_per_model.csv already present in ${OUTDIR}. Exiting."
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "Input  : ${INDIR}"
echo "Output : ${OUTDIR}"

python "${PROJECT_ROOT}/rank_psr_configs.py" \
    --indir  "${INDIR}" \
    --outdir "${OUTDIR}" \
    --models "${MODEL}"

echo "Done. Results for '${MODEL}' saved to ${OUTDIR}"
