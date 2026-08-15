#!/bin/bash
#SBATCH --job-name=i2i_phi_array
#SBATCH --output=logs_ensemble_grid/i2i_phi_array_%A_%a.out
#SBATCH --error=logs_ensemble_grid/i2i_phi_array_%A_%a.err

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --ntasks=1
#SBATCH --array=0-19   # 20 jobs = 1 per test WSI

# Descriptor-space (phi_struct) uncertainty, one WSI per task.
#
# Per-WSI decomposition is EXACT, not an approximation. `decompose()` works
# region by region — procedural and data-exposure at region r depend only on the
# 50 members' descriptor vectors at r — and regions never cross slide
# boundaries. The only cohort-level quantities are the three means in
# summary.json, which `aggregate_phi_uncertainty.py` recovers by pooling the
# per-region rows afterwards.
#
# Every task still reads ALL FIVE folds. The split is over WSIs, never over
# folds: one fold alone gives procedural variance and no data-exposure term at
# all, so a per-fold array would silently compute the wrong thing.
#
# Reads : {subset}/model_small/wsi_masks_final/model_{NN}/<wsi>.tif  (x5 folds)
# Writes: ${OUTDIR}/per_wsi/wsi{NNN}/per_region.csv + summary.json
#
# Then pool:
#   python I2I-Stain-Zoo/aggregate_phi_uncertainty.py \
#       --indir ${OUTDIR}/per_wsi --outdir ${OUTDIR}
#
# Kidney arm — override at submit time, same as the single-job script:
#   sbatch --export=ALL,TEST_A=...,HE_DIR=...,ROI_DIR=...,OUTDIR=... \
#       I2I-Stain-Zoo/scripts/compute_phi_uncertainty_grid_array.sh

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
# Paths — all overridable via --export
# -----------------------------
GRID_ROOT="${GRID_ROOT:-/work2/bz66izin-VSproject/ensemble_grid/cyclegan}"
MODEL_SIZE="${MODEL_SIZE:-model_small}"
TEST_A="${TEST_A:-/work2/bz66izin-UC_project/ID/no_overlap/testA/tiles/testA}"
# Reconstructed H&E, or the ORIGINAL H&E WSIs — both sit in the same pixel
# frame. NOT the export_tissue binary masks. `none` accepts NaN for the two
# H&E-referenced descriptors.
HE_DIR="${HE_DIR:-/work2/bz66izin-UC_project/ID/no_overlap/testA/export_rgb/testA}"
ROI_DIR="${ROI_DIR:-}"
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID/phi_uncertainty}"

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

# Grid decomposition — identical to train_ensemble_cyclegan_grid.sh and the rest
# of the chain. Change it here and it has to change in all of them.
N_MEMBERS=10
RANGE_STARTS=(1  8  15 22 29)
RANGE_ENDS=(  7  14 21 28 35)

# This task's WSI
WSI=$(printf "%03d" $(( SLURM_ARRAY_TASK_ID + 1 )))
META_CSV="${TEST_A}/${WSI}/tiles_metadata.csv"
TASK_OUT="${OUTDIR}/per_wsi/wsi${WSI}"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}  WSI=${WSI}"
echo "Metadata : ${META_CSV}"
echo "Output   : ${TASK_OUT}"

# -----------------------------
# Pre-flight
# -----------------------------
# iter_metadata_csvs globs <root>/*/tiles_metadata.csv one level deep, so the
# per-WSI selection has to be the CSV file itself — it accepts a file path and
# yields it directly (regions.py:162). Handing it the NNN/ directory would match
# nothing and produce zero regions.
if [ ! -f "${META_CSV}" ]; then
    echo "[ERROR] No metadata for WSI ${WSI}: ${META_CSV}"
    echo "        Is the cohort smaller than the --array range?"
    exit 1
fi

FOLD_ARGS=()
FOLD_DIRS=()
MISSING=0
for i in "${!RANGE_STARTS[@]}"; do
    RANGE_TAG=$(printf "data_%03d_%03d" "${RANGE_STARTS[$i]}" "${RANGE_ENDS[$i]}")
    FOLD_DIR="${GRID_ROOT}/${RANGE_TAG}/${MODEL_SIZE}/wsi_masks_final"
    if [ ! -d "${FOLD_DIR}" ]; then
        echo "[ERROR] Fold directory missing: ${FOLD_DIR}"
        MISSING=1
        continue
    fi
    FOLD_ARGS+=(--fold "${FOLD_DIR}")
    FOLD_DIRS+=("${FOLD_DIR}")
done
if [ "${MISSING}" -ne 0 ]; then
    echo "[ERROR] Incomplete grid. Run fill_tissue_holes_grid.sh to completion first."
    exit 1
fi

HE_ARGS=()
if [ "${HE_DIR}" = "none" ]; then
    echo "[WARN] HE_DIR=none — lumen_fraction and tissue_fraction will be NaN."
elif [ -d "${HE_DIR}" ]; then
    HE_ARGS=(--he_dir "${HE_DIR}")
else
    echo "[ERROR] H&E directory not found: ${HE_DIR}"
    exit 1
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

# Coverage + geometry check for THIS WSI. A member with no mask for this slide
# yields an all-NaN slab rather than an error (ensemble.py:115), which quietly
# shrinks the member count behind the variance — so fail here instead. The H&E
# check catches a wrong pyramid level, which region.crop() would silently accept
# as a short crop.
python - "${META_CSV}" "${HE_DIR}" "${N_MEMBERS}" "${FOLD_DIRS[@]}" <<'PY' || exit 1
import sys
from pathlib import Path

import pandas as pd
import tifffile

meta_csv, he_dir, n_members = sys.argv[1:4]
fold_dirs = [Path(p) for p in sys.argv[4:]]
n_members = int(n_members)

df = pd.read_csv(meta_csv)
if df.empty:
    print(f"[ERROR] empty metadata: {meta_csv}")
    sys.exit(1)

stem = Path(str(df["source_file"].unique()[0])).stem
h = int((df["y"] + df["tile_size"]).max())
w = int((df["x"] + df["tile_size"]).max())
print(f"[INFO]  WSI {stem}: region extent {h}x{w}")

rc = 0

# every member of every fold must hold this WSI
for root in fold_dirs:
    tag = root.parents[1].name
    found = sum(
        1 for m in sorted(root.glob("model_*"))
        if any((m / f"{stem}{ext}").exists() for ext in (".tif", ".tiff", ".png"))
    )
    if found < n_members:
        print(f"[ERROR] {tag}: {found}/{n_members} members have a mask for {stem}")
        rc = 1
    else:
        print(f"[OK]    {tag}: {found}/{n_members} members")

if he_dir != "none":
    idx = {p.stem: p for p in sorted(Path(he_dir).iterdir())
           if p.suffix.lower() in (".tif", ".tiff", ".png")}
    path = idx.get(stem)
    if path is None:
        print(f"[WARN]  no H&E for {stem} — its lumen/tissue terms will be NaN")
    else:
        shape = tifffile.TiffFile(path).series[0].shape
        hh, ww = shape[0], shape[1]
        if hh < h or ww < w:
            print(f"[ERROR] {path.name}: {hh}x{ww} smaller than extent {h}x{w}. "
                  f"Wrong pyramid level or a downsampled export?")
            rc = 1
        else:
            print(f"[OK]    H&E {hh}x{ww} covers extent {h}x{w}")

sys.exit(rc)
PY

# Skip guard
if [ -s "${TASK_OUT}/per_region.csv" ] && [ -s "${TASK_OUT}/summary.json" ]; then
    echo "[SKIP] Already completed: ${TASK_OUT}"
    exit 0
fi

mkdir -p "${TASK_OUT}"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd python "${PROJECT_ROOT}/compute_phi_uncertainty.py" \
    "${FOLD_ARGS[@]}" \
    --tiles_metadata "${META_CSV}" \
    "${HE_ARGS[@]}" \
    "${ROI_ARGS[@]}" \
    --region_mm "${REGION_MM}" \
    --min_tissue_fraction "${MIN_TISSUE_FRACTION}" \
    --white_thresh "${WHITE_THRESH}" \
    "${QC_ARGS[@]}" \
    --outdir "${TASK_OUT}"

echo "Done. WSI ${WSI} → ${TASK_OUT}/"
echo "Pool all 20 with: python ${PROJECT_ROOT}/aggregate_phi_uncertainty.py \\"
echo "    --indir ${OUTDIR}/per_wsi --outdir ${OUTDIR}"
