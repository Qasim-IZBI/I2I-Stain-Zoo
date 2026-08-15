#!/bin/bash
#SBATCH --job-name=i2i_phi_uncertainty
#SBATCH --output=logs_ensemble_grid/i2i_phi_uncertainty_%j.out
#SBATCH --error=logs_ensemble_grid/i2i_phi_uncertainty_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Descriptor-space (phi_struct) uncertainty decomposition over the crossed
# ensemble grid: 5 disjoint training subsets x 10 seeds.
#
# NOT an array job. The law of total variance needs every fold in one process —
# splitting it per subset would give five procedural numbers and no data-exposure
# term at all. One job, five --fold arguments, single pass.
#
# Reads : {subset}/model_small/wsi_masks_final/model_{NN}/   (fill_tissue_holes_grid.sh)
# Writes: ${OUTDIR}/per_region.csv, ${OUTDIR}/summary.json
#
# Kidney arm (cortex only) — override at submit time:
#   sbatch --export=ALL,\
#   TEST_A=/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA_kidney/tiles/testA,\
#   HE_DIR=/work2/bz66izin-VSproject/reconstruction/real_he_kidney/reconstructed,\
#   ROI_DIR=/work2/bz66izin-VSproject/cortex_masks,\
#   OUTDIR=/work2/bz66izin-VSproject/phi_uncertainty/kidney \
#       I2I-Stain-Zoo/scripts/compute_phi_uncertainty_grid.sh

# -eo, not -euo: the Anaconda module runs activate.d hooks that read
# unset variables (qt-main_activate.sh), so -u there kills the job before
# the first echo. -u is switched on below, once conda is done.
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

mkdir -p logs_ensemble_grid

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — every one overridable via --export so the kidney arm needs no edit
# -----------------------------
GRID_ROOT="${GRID_ROOT:-/work2/bz66izin-VSproject/ensemble_grid/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
TEST_A="${TEST_A:-/work2/bz66izin-VSproject/VS_Data/eval_imgs/no_overlap/testA/tiles/testA}"
# Reconstructed H&E RGB WSIs (NOT the HE_tissue binary masks apply_he_mask.py takes).
# Without these, lumen_fraction and tissue_fraction come out NaN — the two
# descriptors that pay no floor. Set HE_DIR=none to accept that deliberately.
HE_DIR="${HE_DIR:-/work2/bz66izin-VSproject/reconstruction/real_he/reconstructed}"
# Anatomical compartment mask, kidney arm only. Empty = whole slide (liver).
ROI_DIR="${ROI_DIR:-}"
OUTDIR="${OUTDIR:-/work2/bz66izin-VSproject/phi_uncertainty/liver}"

REGION_MM="${REGION_MM:-1.5}"
MIN_TISSUE_FRACTION="${MIN_TISSUE_FRACTION:-0.25}"
MIN_ROI_FRACTION="${MIN_ROI_FRACTION:-0.5}"
# Whitespace cut for lumen_fraction / tissue_fraction. EVERY channel must
# clear it, so set it from the per-pixel channel MINIMUM, not from the grey
# level an 8-bit conversion reports. A lumen_fraction near 1e-5 in
# per_region.csv means this sits above the lumens.
WHITE_THRESH="${WHITE_THRESH:-0.85}"
# Optional: one region per WSI written as a TIF pair (label mask + H&E
# crop) for inspecting in Fiji, to see whether the threshold found lumens
# or pale tissue. Empty = off. QC_MAX_PX caps the crop; 0 = full res, at
# which a 1.5 mm H&E region is ~100 MB before compression.
QC_DIR="${QC_DIR-}"
QC_MAX_PX="${QC_MAX_PX:-0}"

# Decomposition — identical to train_ensemble_cyclegan_grid.sh and the rest of
# the grid chain. Change it here and it has to change in all six.
N_MEMBERS=10
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

# -----------------------------
# Pre-flight: every fold must be complete before the variance means anything
# -----------------------------
FOLD_ARGS=()
MISSING=0

for i in "${!RANGE_STARTS[@]}"; do
    RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")
    FOLD_DIR="${GRID_ROOT}/${RANGE_TAG}/${MODEL_SIZE}/wsi_masks_final"

    if [ ! -d "${FOLD_DIR}" ]; then
        echo "[ERROR] Fold directory missing: ${FOLD_DIR}"
        MISSING=1
        continue
    fi

    N_FOUND=$(find "${FOLD_DIR}" -maxdepth 1 -type d -name "model_*" | wc -l)
    if [ "${N_FOUND}" -lt "${N_MEMBERS}" ]; then
        echo "[ERROR] ${RANGE_TAG}: ${N_FOUND}/${N_MEMBERS} members under ${FOLD_DIR}"
        MISSING=1
        continue
    fi

    echo "[OK]    ${RANGE_TAG}: ${N_FOUND} members"
    FOLD_ARGS+=(--fold "${FOLD_DIR}")
done

if [ "${MISSING}" -ne 0 ]; then
    echo "[ERROR] Incomplete grid. Run fill_tissue_holes_grid.sh to completion first."
    echo "        A partial grid would silently under-state data-exposure variance."
    exit 1
fi

if [ ! -d "${TEST_A}" ]; then
    echo "[ERROR] tiles_metadata root not found: ${TEST_A}"
    exit 1
fi

HE_ARGS=()
if [ "${HE_DIR}" = "none" ]; then
    echo "[WARN] HE_DIR=none — lumen_fraction and tissue_fraction will be NaN."
elif [ -d "${HE_DIR}" ]; then
    HE_ARGS=(--he_dir "${HE_DIR}")
else
    echo "[ERROR] H&E directory not found: ${HE_DIR}"
    echo "        Either the RGB reconstruction of testA or the ORIGINAL H&E WSIs;"
    echo "        both sit in the same pixel frame. NOT the HE_tissue binary masks."
    echo "        Set HE_DIR=none to run without the two H&E-referenced descriptors."
    exit 1
fi

# H&E geometry check. region.crop() is bare numpy slicing with no resize and no
# shape assertion (ensemble.py:120), unlike load_roi_mask — an H&E at the wrong
# scale returns a short crop instead of raising, and the run completes with
# plausible-looking wrong lumen/tissue fractions. Header read only; the slide is
# never loaded. Originals may be LARGER than the extent (tiling truncates to a
# whole number of tiles at a top-left origin) but must never be smaller.
if [ "${#HE_ARGS[@]}" -gt 0 ]; then
    python - "${TEST_A}" "${HE_DIR}" <<'PY' || exit 1
import sys
from pathlib import Path

import pandas as pd
import tifffile

meta_root, he_dir = Path(sys.argv[1]), Path(sys.argv[2])
he_index = {p.stem: p for p in sorted(he_dir.iterdir())
            if p.suffix.lower() in (".tif", ".tiff", ".png")}

bad = 0
for csv_path in sorted(meta_root.rglob("tiles_metadata.csv")):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    stem = Path(str(df["source_file"].unique()[0])).stem
    h = int((df["y"] + df["tile_size"]).max())
    w = int((df["x"] + df["tile_size"]).max())

    path = he_index.get(stem)
    if path is None:
        print(f"[WARN] no H&E for {stem} — its lumen/tissue terms will be NaN")
        continue

    shape = tifffile.TiffFile(path).series[0].shape
    hh, ww = shape[0], shape[1]
    if hh < h or ww < w:
        print(f"[ERROR] {path.name}: {hh}x{ww} is smaller than the region extent "
              f"{h}x{w}. Wrong pyramid level or a downsampled export?")
        bad += 1
    else:
        print(f"[OK]    {stem}: H&E {hh}x{ww} covers extent {h}x{w}")

sys.exit(1 if bad else 0)
PY
fi

QC_ARGS=()
if [ -n "${QC_DIR}" ]; then
    mkdir -p "${QC_DIR}"
    QC_ARGS=(--qc_dir "${QC_DIR}" --qc_max_px "${QC_MAX_PX}")
    echo "QC images: ${QC_DIR}"
fi

ROI_ARGS=()
if [ -n "${ROI_DIR}" ]; then
    if [ ! -d "${ROI_DIR}" ]; then
        echo "[ERROR] --roi_dir given but not found: ${ROI_DIR}"
        exit 1
    fi
    ROI_ARGS=(--roi_dir "${ROI_DIR}" --min_roi_fraction "${MIN_ROI_FRACTION}")
fi

# Skip guard
if [ -s "${OUTDIR}/per_region.csv" ] && [ -s "${OUTDIR}/summary.json" ]; then
    echo "[SKIP] Already completed: ${OUTDIR}"
    exit 0
fi

mkdir -p "${OUTDIR}"

echo "Grid root : ${GRID_ROOT} (${MODEL_SIZE})"
echo "Metadata  : ${TEST_A}"
echo "H&E       : ${HE_DIR}"
echo "ROI       : ${ROI_DIR:-<none, whole slide>}"
echo "Output    : ${OUTDIR}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/compute_phi_uncertainty.py" \
    "${FOLD_ARGS[@]}" \
    --tiles_metadata "${TEST_A}" \
    "${HE_ARGS[@]}" \
    "${ROI_ARGS[@]}" \
    --region_mm "${REGION_MM}" \
    --min_tissue_fraction "${MIN_TISSUE_FRACTION}" \
    --white_thresh "${WHITE_THRESH}" \
    "${QC_ARGS[@]}" \
    --outdir "${OUTDIR}"

echo "Done. phi_struct decomposition written to ${OUTDIR}/"
