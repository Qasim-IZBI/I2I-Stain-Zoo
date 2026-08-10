#!/bin/bash
#SBATCH --job-name=i2i_seg_perturb
#SBATCH --output=logs_perturbation/seg_perturb_%A_%a.out
#SBATCH --error=logs_perturbation/seg_perturb_%A_%a.err

#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-4   # one job per perturbation step t

# Segment the stain-perturbation series written by
#   python stain_sensitivity.py make-series
#
# Every step holds anatomy fixed and moves only colour, from the real PSR's own
# statistics (t=0) toward the virtual PSR's (t=1). Descriptor drift across the
# series is therefore pure measurement artefact - the error bar that belongs on
# any bias number (kidney_ood_data_plan.md section 6.2).
#
# Uses the SAME segmenter as everything else, Dataset314_SR_light, because the
# question is how THAT model reacts to colour.
#
#   Input  : {PERTURB_ROOT}/t{X}/         (RGB, _0000.tif)
#   Output : {PERTURB_ROOT}/masks/t{X}/   (label masks)
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/segment_psr_perturbation.sh
# Adjust --array if make-series was given a different --fractions list.

set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_perturbation

# Must match the --fractions given to make-series
FRACTIONS=(0.00 0.25 0.50 0.75 1.00)
T=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

PERTURB_ROOT=/work2/bz66izin-VSproject/perturbation
IN_DIR="${PERTURB_ROOT}/t${T}"
OUT_DIR="${PERTURB_ROOT}/masks/t${T}"

export nnUNet_results="/work2/bz66izin-VSproject/nnunet/nnUNet_results"
export nnUNet_raw="/work2/bz66izin-VSproject/nnunet/nnUNet_raw"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  t=${T}"
echo "Input  : ${IN_DIR}"
echo "Output : ${OUT_DIR}"

if [ ! -d "${IN_DIR}" ] || [ -z "$(ls -A "${IN_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] Perturbed images missing or empty: ${IN_DIR}"
    echo "        Run: python stain_sensitivity.py make-series ..."
    exit 1
fi

N_IN=$(find "${IN_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
if [ -d "${OUT_DIR}" ]; then
    N_OUT=$(find "${OUT_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    if [ "${N_OUT}" -ge "${N_IN}" ] && [ "${N_IN}" -gt 0 ]; then
        echo "[SKIP] ${N_OUT}/${N_IN} masks already present in ${OUT_DIR}."
        exit 0
    fi
fi

mkdir -p "${OUT_DIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i "${IN_DIR}" \
    -o "${OUT_DIR}" \
    -f 0 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    -npp 1 \
    -nps 1 \
    -device cuda

echo "Done. t=${T} -> ${OUT_DIR}"
echo "next: python stain_sensitivity.py analyse --masks ${PERTURB_ROOT}/masks/ ..."
