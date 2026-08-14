#!/bin/bash
#SBATCH --job-name=i2i_seg_real
#SBATCH --output=logs_real_sr/seg_real_%A_%a.out
#SBATCH --error=logs_real_sr/seg_real_%A_%a.err

#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --array=0-19   # 20 jobs = 1 per real SR WSI

# Collagen segmentation of the REAL SR whole slides — the reference arm the
# virtual stain is measured against. Fills the gap CLAUDE.md records as "no
# committed segmentation script": this step was previously run by hand.
#
# Input is the ORIGINAL thumbnail-registered SR WSIs, not a reconstruction from
# testB tiles. Two consequences, both real:
#
#   * RESOLUTION PARITY IS ON YOU. The virtual arm is segmented on
#     reconstructions at the source 0.221 um/px, because reconstruct_wsi
#     upsamples each tile back to tile_size. Dataset314_SR_light is a 2d model
#     with a fixed patch size, so it sees structures at whatever scale the input
#     happens to be. If these originals are at a different mpp, the two arms are
#     segmented at different scales and every CPA difference is confounded with
#     that. The pre-flight prints each slide's dimensions — compare them against
#     the reconstructed virtual WSIs before trusting a single number.
#   * Registration is thumbnail-level, so the H&E footprint applied downstream
#     is approximate on this arm and exact on the virtual one (see
#     apply_he_mask_real_sr.sh).
#
# Output : ${OUT_DIR}/<slide>.tif   labels 0 background / 1 tissue / 2 PSR+
#
# CAVEAT (kidney_ood_data_plan.md section 6.2): Dataset314_SR_light is trained on
# LIVER SR. Applying it to another organ is out-of-distribution use and its
# failure would be indistinguishable from the model bias under measurement.

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

mkdir -p logs_real_sr

# -----------------------------
# Paths — overridable via --export
# -----------------------------
# Directory of original registered SR WSIs, one TIF per slide. VERIFY THIS.
SR_WSI_DIR="${SR_WSI_DIR:-/work2/bz66izin-UC_project/ID_SR/no_overlap/testB/export_rgb/testB}"
OUT_DIR="${OUT_DIR:-/work2/bz66izin-UC_project/psr_masks/real/psr_masks_wsi}"

export nnUNet_results="${nnUNet_results:-/work2/bz66izin-VSproject/nnunet/nnUNet_results}"
export nnUNet_raw="${nnUNet_raw:-/work2/bz66izin-VSproject/nnunet/nnUNet_raw}"

# -----------------------------
# Pick this task's slide
# -----------------------------
if [ ! -d "${SR_WSI_DIR}" ]; then
    echo "[ERROR] SR WSI directory not found: ${SR_WSI_DIR}"
    echo "        Pass the right one: sbatch --export=ALL,SR_WSI_DIR=... $0"
    exit 1
fi

# Null-delimited so filenames with spaces, quotes or '+' survive — this cohort
# has slides literally named 'HE_w10_BDL+A_M7'.tif, quotes included. A read loop
# rather than `mapfile -d ''`, which needs bash 4.4.
SLIDES=()
while IFS= read -r -d '' f; do
    SLIDES+=("$f")
done < <(find "${SR_WSI_DIR}" -maxdepth 1 -type f \
    \( -name '*.tif' -o -name '*.tiff' \) -print0 | LC_ALL=C sort -z)

N_SLIDES=${#SLIDES[@]}
if [ "${N_SLIDES}" -eq 0 ]; then
    echo "[ERROR] No TIFs in ${SR_WSI_DIR}"
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_SLIDES}" ]; then
    echo "[ERROR] Task ${SLURM_ARRAY_TASK_ID} exceeds the ${N_SLIDES} slides found."
    echo "        Set --array=0-$(( N_SLIDES - 1 ))."
    exit 1
fi

SR_TIF="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
SR_NAME=$(basename "${SR_TIF}")
SR_STEM="${SR_NAME%.*}"
OUT_MASK="${OUT_DIR}/${SR_STEM}.tif"

echo "TASK_ID=${SLURM_ARRAY_TASK_ID}/${N_SLIDES}"
echo "Slide  : ${SR_TIF}"
echo "Output : ${OUT_MASK}"

if [ -f "${OUT_MASK}" ]; then
    echo "[SKIP] Mask already present: ${OUT_MASK}"
    exit 0
fi

# Report the geometry so resolution parity against the virtual arm is checkable
# from the logs rather than assumed.
python - "${SR_TIF}" <<'PY'
import sys
import tifffile
with tifffile.TiffFile(sys.argv[1]) as tf:
    s = tf.series[0]
    print(f"[INFO] shape={s.shape} dtype={s.dtype} pages={len(tf.pages)}")
    tags = tf.pages[0].tags
    res = tags.get("XResolution")
    unit = tags.get("ResolutionUnit")
    if res is not None:
        print(f"[INFO] XResolution={res.value} unit={getattr(unit, 'value', None)}")
    else:
        print("[INFO] no resolution tag — confirm the mpp against the reconstructions by hand")
PY

mkdir -p "${OUT_DIR}"

# nnUNet predicts over a directory and demands a _0000 channel suffix, so stage
# this one slide in a temp dir and rename the prediction back afterwards.
TMP_BASE=$(mktemp -d)
TMP_IN="${TMP_BASE}/in"
TMP_OUT="${TMP_BASE}/out"
mkdir -p "${TMP_IN}" "${TMP_OUT}"
cleanup() { rm -rf "${TMP_BASE}"; }
trap cleanup EXIT

cp "${SR_TIF}" "${TMP_IN}/${SR_STEM}_0000.tif"

run_cmd() {
    echo "Running command:"
    printf ' %q' "$@"
    echo
    "$@"
}

run_cmd nnUNetv2_predict \
    -d Dataset314_SR_light \
    -i "${TMP_IN}" \
    -o "${TMP_OUT}" \
    -f 0 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    -npp 1 \
    -nps 1 \
    -device cuda

if [ ! -f "${TMP_OUT}/${SR_STEM}.tif" ]; then
    echo "[ERROR] nnUNet produced no mask for ${SR_STEM}. Contents of ${TMP_OUT}:"
    ls -la "${TMP_OUT}"
    exit 1
fi

# The move out of /tmp crosses filesystems, so it is a copy and can be
# interrupted. Land it under a .partial name and rename within OUT_DIR, which is
# atomic: a copy that runs out of space leaves <name>.tif.partial and never a
# truncated .tif for the skip guard to count as a finished slide.
mv "${TMP_OUT}/${SR_STEM}.tif" "${OUT_MASK}.partial"
mv "${OUT_MASK}.partial" "${OUT_MASK}"

echo "Done. ${SR_STEM} → ${OUT_MASK}"
