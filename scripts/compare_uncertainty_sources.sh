#!/bin/bash
#SBATCH --job-name=i2i_cmp_src
#SBATCH --output=logs_cali/cmp_src_%j.out
#SBATCH --error=logs_cali/cmp_src_%j.err

#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Cycle-reconstruction error against ensemble spread, on the SAME regions.
#
# The paper's central contrast — the cheap self-consistency proxy fails where a
# task-relevant target works — is otherwise a citation beside a measurement.
# This scores both with the same regions, the same target and the same
# slide-clustered statistics, so only the uncertainty source differs.
#
# PREREQUISITE, and it is the expensive one: per-member regen error maps.
#   sbatch scripts/infer_ensemble_cyclegan_B2A.sh      # B->A per member
#   evaluation.py --metric regen_error --path_A <testA/NNN/images> \
#       --path_A_regen <that member's B2A tiles> \
#       --overlay_dir <REGEN_ROOT>/model_NN/wsi<NNN>/ --save_error_npy
# giving REGEN_ROOT/model_NN/wsi{NNN}/error_npy/<tile>.npy.
#
# Members are averaged. Two or three is enough to characterise cycle error —
# it is a property of one model's forward/inverse pair, not of the ensemble —
# and each member costs a full B->A inference pass, so do not queue fifty
# before looking at three.
#
# The tile pass caches per-tile means to tile_errors.csv, so a re-run with
# different binning or bootstrap settings is seconds.

# -eo, not -euo: the Anaconda module runs activate.d hooks that read unset
# variables, so -u there kills the job before the first echo.
set -eo pipefail

module purge
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
set +u
conda activate i2istain
set -u

echo "Host: $(hostname)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p logs_cali

PROJECT_ROOT=I2I-Stain-Zoo

PHI_CSV="${PHI_CSV:-/work2/bz66izin-UC_project/ID_HE/phi_uncertainty/agg_phi/per_region.csv}"
REFERENCE_CSV="${REFERENCE_CSV:-/work2/bz66izin-UC_project/ID_HE/phi_reference/reference_phi.csv}"
TILES_METADATA="${TILES_METADATA:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA}"
# Space-separated list of member roots, each holding wsi{NNN}/error_npy/
REGEN_ROOTS="${REGEN_ROOTS:-}"
# Optional flat per-tile tissue masks; without them background dilutes the
# regen error of edge regions, which flatters neither source in particular but
# adds noise to both.
MASK_DIR="${MASK_DIR-}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/compare_sources}"

if [ -z "${REGEN_ROOTS}" ]; then
    echo "[ERROR] REGEN_ROOTS is empty. Pass one or more member roots:"
    echo "  sbatch --export=ALL,REGEN_ROOTS='/path/regen/model_01 /path/regen/model_02' \\"
    echo "      ${PROJECT_ROOT}/scripts/compare_uncertainty_sources.sh"
    exit 1
fi

for f in "${PHI_CSV}" "${REFERENCE_CSV}"; do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] Missing: ${f}"
        echo "        Run compute_phi_uncertainty + compute_phi_reference first."
        exit 1
    fi
done

ROOT_ARGS=()
for r in ${REGEN_ROOTS}; do
    if [ ! -d "${r}" ]; then
        echo "[ERROR] Regen root not found: ${r}"
        exit 1
    fi
    # Fail here rather than after the tile pass reports every tile missing.
    if ! ls "${r}"/wsi*/error_npy/*.npy >/dev/null 2>&1; then
        echo "[ERROR] No wsi*/error_npy/*.npy under ${r}"
        echo "        evaluation.py --save_error_npy writes error_npy/ inside"
        echo "        whatever --overlay_dir it is given; it must be per WSI."
        exit 1
    fi
    ROOT_ARGS+=(--regen_root "${r}")
done

MASK_ARGS=()
[ -n "${MASK_DIR}" ] && MASK_ARGS=(--mask_dir "${MASK_DIR}")

mkdir -p "${OUTDIR}"
echo "phi      : ${PHI_CSV}"
echo "reference: ${REFERENCE_CSV}"
echo "regen    : ${REGEN_ROOTS}"
echo "outdir   : ${OUTDIR}"

run_cmd() { echo "Running command:"; printf ' %q' "$@"; echo; "$@"; }

run_cmd python "${PROJECT_ROOT}/compare_uncertainty_sources.py" \
    --phi_csv        "${PHI_CSV}" \
    --reference_csv  "${REFERENCE_CSV}" \
    --tiles_metadata "${TILES_METADATA}" \
    "${ROOT_ARGS[@]}" "${MASK_ARGS[@]}" \
    --outdir         "${OUTDIR}"

echo
echo "Done. Read within_slide.csv: the partial column is the one that matters,"
echo "since both sources otherwise largely report how much structure a region holds."
