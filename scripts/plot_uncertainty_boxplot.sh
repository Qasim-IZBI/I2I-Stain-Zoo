#!/bin/bash
#SBATCH --job-name=i2i_unc_boxplot
#SBATCH --output=logs_ensemble_ugac/unc_boxplot_%j.out
#SBATCH --error=logs_ensemble_ugac/unc_boxplot_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Per-tile uncertainty distributions for the UGAC CycleGAN ensemble, one box
# per training subset.
#
# Consumes what aggregate_uncertainty.sh writes:
#
#   {UGAC_ROOT}/{block}/model_small/uncertainty/cyclegan/per_wsi_csv/*.csv
#
# and writes to OUTDIR: uncertainty_boxplot.png, uncertainty_violin.png,
# uncertainty_quantiles.csv, and a per_wsi/ figure pair per slide.
#
# ONE JOB, not an array. The whole point is a single figure comparing the five
# subsets on shared axes, so they have to be read together.
#
# **The groups are the five DATA BLOCKS, not the six model families.**
# plot_uncertainty_boxplot.py defaults to the scaling study's
# {model}/data_large/{size}/ tree, which this chain does not have — one family
# across five training subsets instead. --group names each explicitly, so the
# script needs to know no layout at all. What the figure then shows is how much
# the spread between members depends on WHICH seven slides they were trained on.
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/plot_uncertainty_boxplot.sh
# The BMVC per-family figure is still the script's default, with no --group:
#   python I2I-Stain-Zoo/plot_uncertainty_boxplot.py \
#       --base /work2/bz66izin-VSproject/ensemble --outdir <dir>

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo and the log comes
# back empty. It goes on after conda activate, as the rest of this family does.
set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"

mkdir -p logs_ensemble_ugac

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Block axis — identical to aggregate_uncertainty.sh
# -----------------------------
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

# -----------------------------
# Paths — overridable via --export
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"
OUTDIR="${OUTDIR:-/work2/bz66izin-VSproject/ensemble_ugac/uncertainty_boxplot}"

echo "UGAC root : ${UGAC_ROOT}"
echo "Output    : ${OUTDIR}"

# -----------------------------
# Build one --group per block, skipping any that has not been aggregated
# -----------------------------
GROUP_ARGS=()
MISSING=()
for i in "${!RANGE_STARTS[@]}"; do
    TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")
    CSV_DIR="${UGAC_ROOT}/${TAG}/${MODEL_SIZE}/uncertainty/${MODEL}/per_wsi_csv"
    if [ -d "${CSV_DIR}" ] && \
       [ -n "$(find "${CSV_DIR}" -maxdepth 1 -name '*.csv' 2>/dev/null | head -1)" ]; then
        N=$(find "${CSV_DIR}" -maxdepth 1 -name '*.csv' | wc -l)
        echo "  ${TAG}: ${N} per-WSI CSV(s)"
        GROUP_ARGS+=(--group "${TAG}=${CSV_DIR}")
    else
        echo "  ${TAG}: [missing] ${CSV_DIR}"
        MISSING+=("${TAG}")
    fi
done

if [ "${#GROUP_ARGS[@]}" -eq 0 ]; then
    echo "[ERROR] No per_wsi_csv directories found under ${UGAC_ROOT}"
    echo "        Run aggregate_uncertainty.sh first."
    exit 1
fi

# A block silently absent from the figure is worse than a failed job: the plot
# still renders and reads as a complete comparison of whatever survived.
if [ "${#MISSING[@]}" -ne 0 ]; then
    echo
    echo "[ERROR] ${#MISSING[@]} of ${#RANGE_STARTS[@]} blocks have no aggregated"
    echo "        CSVs: ${MISSING[*]}"
    echo "        A missing block would drop out of the figure without trace, so"
    echo "        this refuses rather than plotting a partial comparison."
    echo "        Finish aggregate_uncertainty.sh, or pass --group by hand to"
    echo "        compare a deliberate subset."
    exit 1
fi

mkdir -p "${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/plot_uncertainty_boxplot.py" \
    "${GROUP_ARGS[@]}" \
    --outdir "${OUTDIR}"

echo
echo "Done. Figures in ${OUTDIR}/"
echo "One box per training subset: the spread BETWEEN boxes is how much the"
echo "ensemble's uncertainty depends on which seven slides it saw, which a"
echo "single-subset ensemble cannot show at all."
