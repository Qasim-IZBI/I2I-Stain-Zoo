#!/bin/bash
#SBATCH --job-name=i2i_frame_check
#SBATCH --output=logs_real_sr/frame_check_%j.out
#SBATCH --error=logs_real_sr/frame_check_%j.err

#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=paula
#SBATCH --ntasks=1

# Step 0: do the real SR and the H&E share a coordinate frame?
#
# This is the gate for the whole calibration study. phi is gridded on the H&E
# frame; scoring it against the real SR needs region r to be the same tissue in
# both, which holds only if the SR was RESAMPLED onto the H&E grid rather than
# merely registered to it at thumbnail level.
#
#   aligned     -> region-level pairing, ~6000 paired regions at 2048 px
#   not aligned -> calibrate_phi --real_psr exits on its geometry check; fall
#                  back to WSI-level pairing at n = 20, or resample first
#
# Reads TIFF headers only, so it is seconds across twenty multi-GB slides. It is
# small enough for the login node; it is a batch script so the result lands in a
# log beside everything else.
#
# Exits 0 either way — "not aligned" is a finding, not a failure.

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

mkdir -p logs_real_sr

PROJECT_ROOT=I2I-Stain-Zoo

# -----------------------------
# Paths — overridable via --export
# -----------------------------
HE_DIR="${HE_DIR:-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/export_rgb/testA}"
SR_DIR="${SR_DIR:-/work2/bz66izin-UC_project/ID_SR/no_overlap/testB}"
# Optional. The region boxes run to max(y+tile_size), which is what
# calibrate_phi actually compares against — a slide can be big enough overall
# and still be short of the last region.
TILES_METADATA="${TILES_METADATA-/work2/bz66izin-UC_project/ID_HE/no_overlap/testA/tiles/testA}"

for d in "${HE_DIR}" "${SR_DIR}"; do
    if [ ! -d "${d}" ]; then
        echo "[ERROR] not a directory: ${d}"
        exit 1
    fi
done

echo "H&E : ${HE_DIR}"
echo "SR  : ${SR_DIR}"
echo "Grid: ${TILES_METADATA:-<none>}"
echo

python - "${HE_DIR}" "${SR_DIR}" "${TILES_METADATA}" "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

import tifffile

he_dir, sr_dir, tiles_meta, project_root = (Path(sys.argv[1]), Path(sys.argv[2]),
                                            sys.argv[3], sys.argv[4])
sys.path.insert(0, project_root)
from apply_he_mask import normalize_stem          # the SR_/HE_ prefix rule


def index(d):
    """stripped stem -> path, so SR_x pairs with HE_x."""
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".tif", ".tiff"):
            out.setdefault(normalize_stem(p.stem, True), p)
    return out


def geom(p):
    """(h, w, resolution tag) from the header — no pixel data is read."""
    with tifffile.TiffFile(p) as tf:
        s = tf.series[0]
        res = tf.pages[0].tags.get("XResolution")
        return s.shape[0], s.shape[1], (res.value if res is not None else None)


# The extent the region grid actually runs to, per WSI. A slide can be large
# enough overall and still fall short of the last region box.
extent = {}
if tiles_meta and tiles_meta != "none" and Path(tiles_meta).is_dir():
    import pandas as pd
    for csv in sorted(Path(tiles_meta).glob("*/tiles_metadata.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        stem = Path(str(df["source_file"].unique()[0])).stem
        extent[normalize_stem(stem, True)] = (
            int((df["y"] + df["tile_size"]).max()),
            int((df["x"] + df["tile_size"]).max()),
        )

he, sr = index(he_dir), index(sr_dir)
common = sorted(set(he) & set(sr))
print(f"H&E {len(he)}   SR {len(sr)}   paired {len(common)}")

unpaired = sorted(set(he) ^ set(sr))
if unpaired:
    # a naming mismatch would otherwise look like a dimension mismatch
    print(f"\n[WARN] {len(unpaired)} unpaired stem(s) — check the naming before "
          f"reading anything below:")
    for k in unpaired[:10]:
        side = "H&E only" if k in he else "SR only"
        print(f"   {k}   ({side})")

if not common:
    print("\nNothing to compare. If the two sets use different prefixes, the "
          "SR_/HE_ rule in normalize_stem did not bridge them.")
    raise SystemExit(0)

print()
print(f"{'case':30s} {'H&E h x w':>19s} {'SR h x w':>19s}  verdict")
print("-" * 92)

aligned = short = 0
for k in common:
    hy, hx, hres = geom(he[k])
    sy, sx, sres = geom(sr[k])

    if (hy, hx) == (sy, sx):
        verdict, ok = "IDENTICAL", True
    elif abs(hy - sy) <= 2 and abs(hx - sx) <= 2:
        verdict, ok = "within 2 px", True
    else:
        verdict, ok = f"DIFFER  x{sy / hy:.3f} x{sx / hx:.3f}", False
    aligned += ok

    print(f"{k[:30]:30s} {f'{hy} x {hx}':>19s} {f'{sy} x {sx}':>19s}  {verdict}")

    if k in extent:
        ey, ex = extent[k]
        if sy < ey or sx < ex:
            short += 1
            print(f"{'':30s} {'':19s} {'':19s}  [!] short of the region extent "
                  f"{ey} x {ex} — calibrate_phi would exit here")
    if hres != sres:
        # same frame but different scale still confounds every CPA difference
        print(f"{'':30s} {'':19s} {'':19s}  [!] XResolution {hres} vs {sres}")

print("-" * 92)
print(f"{aligned}/{len(common)} share the H&E frame"
      + (f", {short} short of the region extent" if short else ""))

if aligned == len(common) and not short:
    print("\n=> ALIGNED. Region-level pairing is available:")
    print("   calibrate_phi.py --real_psr <real PSR masks> ...   (~6000 regions)")
    print("\n   Same dimensions is necessary, not sufficient: two slides can match")
    print("   in size and still be offset or rotated. Overlay one case visually")
    print("   before trusting it.")
else:
    print("\n=> NOT ALIGNED. The SR was registered but not resampled onto the H&E")
    print("   grid, so region r is different tissue on each side. calibrate_phi")
    print("   --real_psr will exit on its geometry check rather than score it.")
    print("   Options: resample the SR onto the H&E grid, or fall back to")
    print("   WSI-level pairing at n = 20.")
PY

echo
echo "Done."
