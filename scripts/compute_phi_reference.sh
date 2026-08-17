#!/bin/bash
#SBATCH --job-name=i2i_phi_ref
#SBATCH --output=logs_cali/phi_ref_%j.out
#SBATCH --error=logs_cali/phi_ref_%j.err

#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# phi_struct of the REAL tissue, on the grid the phi run already defined.
#
# The expensive half of the calibration and the one that never changes: it
# loads a full-slide mask per WSI and runs betti plus a structure tensor over
# every region. Nothing about it depends on the ensemble, so it runs ONCE and
# every later calibration reuses reference_phi.csv — changing --prediction,
# --n_bins, --n_boot or anything about the figure costs seconds after this.
#
#   compute_phi_uncertainty.py   ensemble masks -> per_region.csv
#   THIS                         real masks     -> reference_phi.csv
#   calibrate_phi.py             both CSVs      -> the calibration
#
# Not an array job: it is one pass over twenty slides, and splitting it would
# produce partial references that then have to be pooled in the right order.
#
# 128G because a UC liver slide is ~35k x 40k and several full-size arrays are
# live at once. Same reasoning as the segmentation step, one size down: no
# float32 logit planes here, only label masks.

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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs_cali

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — all overridable via --export
# -----------------------------
PHI_CSV="${PHI_CSV:-/work2/bz66izin-UC_project/ID_HE/phi_uncertainty/agg_phi/per_region.csv}"
# The real collagen masks. Named after the SR slides while phi is gridded on the
# H&E, which is what STRIP_PREFIX is for.
REAL_PSR="${REAL_PSR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi_final}"
# The same tissue masks apply_he_mask.py applies — one definition of tissue.
HE_MASKS="${HE_MASKS:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_tissue/testA}"
# Lumen masks of the real H&E. Empty on any cohort whose generated stain does
# not reproduce whitespace — see the note in CLAUDE.md; the UC liver arm is one.
REAL_LUMEN="${REAL_LUMEN-}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/calibration_phi}"

STRIP_PREFIX="${STRIP_PREFIX:-1}"
# Must match the phi run. Checked automatically against its summary.json where
# one exists, and the job fails rather than measuring the two sides differently.
MPP="${MPP:-0.221}"
MIN_OBJECT_PX="${MIN_OBJECT_PX:-16}"
CLOSING_PX="${CLOSING_PX:-0}"
WHITE_THRESH="${WHITE_THRESH:-0.65}"
TILE_SIZE="${TILE_SIZE:-512}"

echo "phi_csv : ${PHI_CSV}"
echo "real PSR: ${REAL_PSR:-<none>}"
echo "real lum: ${REAL_LUMEN:-<none>}"
echo "tissue  : ${HE_MASKS}"
echo "outdir  : ${OUTDIR}"

# -----------------------------
# Pre-flight
# -----------------------------
if [ ! -f "${PHI_CSV}" ]; then
    echo "[ERROR] No phi CSV: ${PHI_CSV}"
    echo "        Run compute_phi_uncertainty_grid_array.sh and pool it with"
    echo "        aggregate_phi_uncertainty.py first."
    exit 1
fi

if [ -z "${REAL_PSR}" ] && [ -z "${REAL_LUMEN}" ]; then
    echo "[ERROR] Neither REAL_PSR nor REAL_LUMEN is set — nothing to measure."
    exit 1
fi

for d in "${REAL_PSR}" "${REAL_LUMEN}" "${HE_MASKS}"; do
    if [ -n "${d}" ] && [ ! -d "${d}" ]; then
        echo "[ERROR] Not a directory: ${d}"
        exit 1
    fi
done

# The frame guard compares each reference against the phi run's recorded
# wsi_h/wsi_w and exits on a mismatch, so a misaligned SR fails here rather than
# scoring different tissue under the same region id. Confirm alignment first if
# this is a new cohort: sbatch scripts/check_frame_alignment.sh
if [ -f "${OUTDIR}/reference_phi.csv" ]; then
    echo "[skip] ${OUTDIR}/reference_phi.csv already exists."
    echo "       Delete it to recompute — the calibration reads it as-is."
    exit 0
fi

mkdir -p "${OUTDIR}"

ARGS=(--phi_csv "${PHI_CSV}" --outdir "${OUTDIR}"
      --mpp "${MPP}" --min_object_px "${MIN_OBJECT_PX}"
      --closing_px "${CLOSING_PX}" --white_thresh "${WHITE_THRESH}"
      --tile_size "${TILE_SIZE}")
[ -n "${REAL_PSR}" ]   && ARGS+=(--real_psr "${REAL_PSR}")
[ -n "${REAL_LUMEN}" ] && ARGS+=(--real_lumen "${REAL_LUMEN}")
[ -n "${HE_MASKS}" ]   && ARGS+=(--he_masks "${HE_MASKS}")
[ "${STRIP_PREFIX}" = "1" ] && ARGS+=(--strip_prefix)

echo "Running command:"
printf ' %q' python "${PROJECT_ROOT}/compute_phi_reference.py" "${ARGS[@]}"
echo
python "${PROJECT_ROOT}/compute_phi_reference.py" "${ARGS[@]}"

echo
echo "Done. Calibrate with (seconds, no masks read):"
echo "  python ${PROJECT_ROOT}/calibrate_phi.py \\"
echo "      --phi_csv ${PHI_CSV} \\"
echo "      --reference_csv ${OUTDIR}/reference_phi.csv \\"
echo "      --outdir ${OUTDIR}"
echo "  ... and again with --prediction fold for the data-exposure claim."
