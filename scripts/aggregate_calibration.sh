#!/bin/bash
#SBATCH --job-name=i2i_aggregate_calib
#SBATCH --output=logs_ensemble_ugac/aggregate_calib_%j.out
#SBATCH --error=logs_ensemble_ugac/aggregate_calib_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Pools the per-WSI calibration CSVs of each training subset and recomputes
# every metric on the full tile pool.
#
# Consumes what run_calibration_all.sh writes:
#
#   {UGAC_ROOT}/{block}/model_small/calibration/cyclegan/wsi{NNN}/per_tile.csv
#
# and writes to OUTDIR: {block}/summary.json, {block}/*.png, all_models.csv,
# plus the three combined_*.png panels across blocks.
#
# **Pooling, not averaging.** The tiles of all twenty WSIs are concatenated and
# the metrics recomputed on them, which is not the mean of twenty per-WSI
# summaries — the reliability bins in particular are quantiles of the pooled
# uncertainty, and a per-WSI mean of them describes no distribution at all.
#
# ONE JOB, not an array: the combined panels compare the blocks on shared axes,
# so they have to be read together.
#
# **The groups are the five DATA BLOCKS, not the six model families.**
# aggregate_calibration.py defaults to the scaling study's
# {model}/data_large/{size}/ tree, which this chain does not have. --group names
# each block's calibration directory explicitly, so the script needs to know no
# layout at all.
#
# > RETIRED CHAIN. The UGAC ensemble did not produce usable virtual stain and
# > nothing downstream consumes the heads; this is kept for provenance beside
# > the rest of scripts/*_ugac.sh. Do NOT mix its outputs with ensemble_grid/.
#
# Submit from the parent directory of the repository:
#   sbatch I2I-Stain-Zoo/scripts/aggregate_calibration.sh
# The BMVC per-family table is still the default, with no --group:
#   python I2I-Stain-Zoo/aggregate_calibration.py \
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_ensemble_ugac

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Block axis — identical to run_calibration_all.sh
# -----------------------------
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

# -----------------------------
# Paths — overridable via --export
# -----------------------------
UGAC_ROOT="${UGAC_ROOT:-/work2/bz66izin-VSproject/ensemble_ugac/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
MODEL="${MODEL:-cyclegan}"
N_WSIS="${N_WSIS:-20}"
N_BINS="${N_BINS:-10}"
OUTDIR="${OUTDIR:-/work2/bz66izin-VSproject/ensemble_ugac/calibration_combined}"

echo "UGAC root : ${UGAC_ROOT}"
echo "Output    : ${OUTDIR}"
echo "Expecting : ${N_WSIS} WSIs per block"

# -----------------------------
# Build one --group per block
# -----------------------------
GROUP_ARGS=()
INCOMPLETE=()
MISSING=()
for i in "${!RANGE_STARTS[@]}"; do
    TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")
    CALIB_DIR="${UGAC_ROOT}/${TAG}/${MODEL_SIZE}/calibration/${MODEL}"
    N=$(find "${CALIB_DIR}" -mindepth 2 -maxdepth 2 -name per_tile.csv 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "  ${TAG}: [missing] ${CALIB_DIR}"
        MISSING+=("${TAG}")
        continue
    fi
    echo "  ${TAG}: ${N}/${N_WSIS} per_tile.csv"
    [ "${N}" -ne "${N_WSIS}" ] && INCOMPLETE+=("${TAG}(${N})")
    GROUP_ARGS+=(--group "${TAG}=${CALIB_DIR}")
done

if [ "${#MISSING[@]}" -ne 0 ]; then
    echo
    echo "[ERROR] ${#MISSING[@]} block(s) have no per_tile.csv: ${MISSING[*]}"
    echo "        A block silently absent from all_models.csv and the combined"
    echo "        panels would read as a complete comparison of whatever"
    echo "        survived, so this refuses rather than pooling a subset."
    echo "        Run run_calibration_all.sh to completion."
    exit 1
fi

# Short is not the same as absent, and it is the more dangerous case: the block
# still appears in every figure, computed over fewer slides than its neighbours.
if [ "${#INCOMPLETE[@]}" -ne 0 ]; then
    echo
    echo "[ERROR] ${#INCOMPLETE[@]} block(s) have fewer than ${N_WSIS} WSIs:"
    echo "        ${INCOMPLETE[*]}"
    echo "        They would still be plotted, pooled over fewer slides than the"
    echo "        others — a difference between blocks that is not a property of"
    echo "        the models. Finish run_calibration_all.sh, or set N_WSIS if the"
    echo "        cohort really is this size."
    exit 1
fi

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

run_cmd python "${PROJECT_ROOT}/aggregate_calibration.py" \
    "${GROUP_ARGS[@]}" \
    --n_wsis "${N_WSIS}" \
    --n_bins "${N_BINS}" \
    --outdir "${OUTDIR}"

echo
echo "Done. ${OUTDIR}/all_models.csv has one row per training subset."
echo "Read the spread ACROSS rows: it is how much the calibration depends on"
echo "which seven slides the members saw, which a single-subset ensemble"
echo "cannot show at all."
