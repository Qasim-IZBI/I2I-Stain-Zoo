#!/bin/bash
#SBATCH --job-name=i2i_rank_psr
#SBATCH --output=logs_rank_psr/rank_psr_%j.out
#SBATCH --error=logs_rank_psr/rank_psr_%j.err

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

mkdir -p logs_rank_psr

PROJECT_ROOT=I2I-Stain-Zoo

# Input: root containing one subdirectory per model, each with summary.json
# (produced by compare_psr_all_configs.sh)
INDIR=/work2/bz66izin-VSproject/psr_comparison

# Output
OUTDIR=/work2/bz66izin-VSproject/psr_best_config

# Pre-flight: at least one model's summary.json must exist
if ! ls "${INDIR}"/*/summary.json &>/dev/null; then
    echo "[ERROR] No summary.json files found under ${INDIR}"
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
    --outdir "${OUTDIR}"

echo "Done. Results saved to ${OUTDIR}"
