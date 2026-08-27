#!/bin/bash
#SBATCH --job-name=i2i_supp_figs
#SBATCH --output=logs_cali/supp_figs_%j.out
#SBATCH --error=logs_cali/supp_figs_%j.err

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# The two supplement figures of FIGURE_REQUESTS.md.
#
#   F-1  stability_data_exposure.pdf   re-render of the W-29 plot, one column
#   F-2  region_mapping.pdf            the analysis grid on both stains
#
# Neither changes a result. Each also writes a DRAFT CAPTION beside it
# (*.caption.tex) — the manuscript will adapt it, but the numbers in it come
# from the run rather than being retyped.
#
#   sbatch scripts/make_supp_figures.sh                     # both
#   sbatch --export=ALL,FIGURE=f1 scripts/make_supp_figures.sh
#   sbatch --export=ALL,FIGURE=f2,WSI=HE_d31_BDL+A_M2 scripts/make_supp_figures.sh
#
# 96G and an hour are for F-2, which reads two whole slides: a UC case is ~35k x
# 40k, so each is ~4 GB as RGB before any copy. F-1 alone needs neither — run it
# with FIGURE=f1 and it finishes in about a minute.
#
# ---------------------------------------------------------------------------
# F-1 REFUSES TO WRITE THE FIGURE if it cannot reproduce the numbers section 8
# already quotes: median share 0.508, leave-one-subset-out range 0.282 to 0.562,
# five replicates, seed spread under 0.005. That is deliberate — the paper
# commits to those values, so a figure disagreeing with them means either the
# input moved or the text needs updating, and neither is fixed by shipping the
# plot. FORCE=1 renders anyway, for inspection only.
#
# F-2 NEEDS A CHOICE FROM YOU. Run it once with LIST_CASES=1, pick a case near
# the MEDIAN region count, and pass it as WSI=. FIGURE_REQUESTS is explicit that
# it must be an ordinary case rather than the best-registered one: a reader who
# downloads the release and finds the figure unrepresentative is worse off than
# one who sees a plain figure.
#
#   sbatch --export=ALL,FIGURE=f2,LIST_CASES=1 scripts/make_supp_figures.sh
# ---------------------------------------------------------------------------

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
# What to build
# -----------------------------
FIGURE="${FIGURE:-both}"          # f1 | f2 | both
LIST_CASES="${LIST_CASES:-0}"
FORCE="${FORCE:-0}"

# -----------------------------
# Paths — all overridable via --export
# -----------------------------
OUTDIR="${OUTDIR:-/work2/bz66izin-UC_project/ID_HE/supp_figures}"

# F-1: the crossed-grid phi run. summary.json must sit beside it (for S).
PER_REGION="${PER_REGION:-/work2/bz66izin-UC_project/ID_HE/phi_uncertainty/agg_phi/per_region.csv}"
DESCRIPTOR="${DESCRIPTOR:-task_specific_value}"
PANELS="${PANELS:-left}"          # left | both
N_BOOT="${N_BOOT:-2000}"
N_DRAWS="${N_DRAWS:-200}"

# F-2: the two stains, and the case to show.
HE_DIR="${HE_DIR:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_rgb/testA}"
SR_DIR="${SR_DIR:-/work2/bz66izin-UC_project/ID_SR/no_overlap/testB/export_rgb/testB}"
WSI="${WSI-}"
REGION_INDEX="${REGION_INDEX-}"   # empty = the middle of the kept list
MPP="${MPP:-0.221}"
MIN_TISSUE_FRACTION="${MIN_TISSUE_FRACTION:-0.25}"
TILE_SIZE="${TILE_SIZE:-512}"     # tile.py --tile_size, NOT --resize_to
THUMB_PX="${THUMB_PX:-1600}"
DPI="${DPI:-300}"
LEVEL="${LEVEL-}"                 # pyramid level, if the TIFs have one

echo "figure     : ${FIGURE}"
echo "outdir     : ${OUTDIR}"
echo "per_region : ${PER_REGION}"
[[ "${FIGURE}" != "f1" ]] && echo "he / sr    : ${HE_DIR} | ${SR_DIR}"

mkdir -p "${OUTDIR}"

if [[ ! -f "${PER_REGION}" ]]; then
    echo "ERROR: ${PER_REGION} does not exist."
    echo "  Written by compute_phi_uncertainty.py over the crossed grid, or"
    echo "  pooled by aggregate_phi_uncertainty.py."
    exit 1
fi

# ---------------------------------------------------------------------------
# F-1
# ---------------------------------------------------------------------------
if [[ "${FIGURE}" == "f1" || "${FIGURE}" == "both" ]]; then
    echo
    echo "=== F-1: stability_data_exposure.pdf ==="
    F1_EXTRA=()
    [[ "${FORCE}" == "1" ]] && F1_EXTRA+=(--force)

    # Guarded expansion: `set -u` is on and bash 3.2 treats an empty array as
    # unbound.
    python "${PROJECT_ROOT}/make_supp_figures.py" stability \
        --per_region "${PER_REGION}" \
        --outdir "${OUTDIR}" \
        --descriptor "${DESCRIPTOR}" \
        --panels "${PANELS}" \
        --n_boot "${N_BOOT}" \
        --n_draws "${N_DRAWS}" \
        --seed 0 \
        ${F1_EXTRA[@]+"${F1_EXTRA[@]}"}
fi

# ---------------------------------------------------------------------------
# F-2
# ---------------------------------------------------------------------------
if [[ "${FIGURE}" == "f2" || "${FIGURE}" == "both" ]]; then
    echo
    echo "=== F-2: region_mapping.pdf ==="

    if [[ "${LIST_CASES}" == "1" ]]; then
        python "${PROJECT_ROOT}/make_supp_figures.py" region-mapping \
            --per_region "${PER_REGION}" --outdir "${OUTDIR}" --list_cases
        echo
        echo "Pick a case near the MEDIAN and re-run with WSI=<that case>."
        exit 0
    fi

    if [[ -z "${WSI}" ]]; then
        echo "WSI is unset, so there is no case to draw."
        echo "  Run once with LIST_CASES=1, pick one near the MEDIAN region"
        echo "  count, then re-run with WSI=<case>. FIGURE_REQUESTS asks for an"
        echo "  ordinary case, not the best-registered one."
        if [[ "${FIGURE}" == "both" ]]; then
            echo "  (F-1 above is finished and written.)"
            exit 0
        fi
        exit 1
    fi

    for d in "${HE_DIR}" "${SR_DIR}"; do
        if [[ ! -d "${d}" ]]; then
            echo "ERROR: ${d} is not a directory."
            exit 1
        fi
    done

    F2_EXTRA=()
    [[ -n "${REGION_INDEX}" ]] && F2_EXTRA+=(--region_index "${REGION_INDEX}")
    [[ -n "${LEVEL}" ]] && F2_EXTRA+=(--level "${LEVEL}")

    python "${PROJECT_ROOT}/make_supp_figures.py" region-mapping \
        --per_region "${PER_REGION}" \
        --he "${HE_DIR}" \
        --sr "${SR_DIR}" \
        --outdir "${OUTDIR}" \
        --wsi "${WSI}" \
        --mpp "${MPP}" \
        --min_tissue_fraction "${MIN_TISSUE_FRACTION}" \
        --tile_size "${TILE_SIZE}" \
        --thumb_px "${THUMB_PX}" \
        --dpi "${DPI}" \
        ${F2_EXTRA[@]+"${F2_EXTRA[@]}"}
fi

echo
echo "Done. In ${OUTDIR}:"
ls -lh "${OUTDIR}" || true
echo
echo "Before shipping F-2, open it and confirm no slide label, barcode, scanner"
echo "overlay or handwriting survived into either thumbnail — the review copy is"
echo "double-blind and whole-slide formats embed a label image."
