#!/bin/bash
#SBATCH --job-name=i2i_regen_boxplot
#SBATCH --output=logs_ensemble/regen_boxplot_%j.out
#SBATCH --error=logs_ensemble/regen_boxplot_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Plots per-tile regen_error distributions across all 6 model families.
# Reads precomputed error_npy/ directories produced by compute_ensemble_regen_error.sh.
# Outputs boxplot, violin, and bar chart to OUTDIR, plus per-WSI figures.
#
# Prerequisites: compute_ensemble_regen_error.sh must have completed first.

set -eo pipefail

module purge
module load Anaconda3

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_ensemble

BASE="/work2/bz66izin-VSproject/ensemble"
TEST_A="/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA"
OUTDIR="/work2/bz66izin-VSproject/regen_error_boxplot"

# Pre-flight: at least one error_npy directory must exist
if [ -z "$(find "${BASE}" -type d -name "error_npy" 2>/dev/null | head -1)" ]; then
    echo "[ERROR] No error_npy directories found under ${BASE}"
    echo "        Run compute_ensemble_regen_error.sh first."
    exit 1
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python I2I-Stain-Zoo/plot_regen_error_boxplot.py \
    --base    "${BASE}" \
    --test_a  "${TEST_A}" \
    --outdir  "${OUTDIR}"

echo "Done. Figures saved to ${OUTDIR}/"
