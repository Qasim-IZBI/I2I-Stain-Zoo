"""Aggregate tile-level uncertainty CSVs to one mean per WSI per model.

Reads the per-WSI tile CSVs produced by aggregate_uncertainty.py
(columns: tile_name, mean_uncertainty) and reduces each WSI to a single
scalar: mean of mean_uncertainty across all tiles.

Output joins directly with mean_cpa_mae.csv from compute_mean_cpa_mae.py
on (model, wsi) for calibration analysis.

Output
------
{outdir}/wsi_uncertainty.csv  — columns: model, wsi, mean_uncertainty,
                                 std_uncertainty, n_tiles

Example
-------
python compute_wsi_uncertainty.py \
    --base   /work2/bz66izin-VSproject/ensemble \
    --outdir ./cpa_mae_per_wsi/

python compute_wsi_uncertainty.py   # uses default paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]

MODEL_DISPLAY_NAMES = {
    "cyclegan":       "CycleGAN",
    "unit":           "UNIT",
    "munit":          "MUNIT",
    "dclgan":         "DCLGAN",
    "uvcgan":         "UVCGAN",
    "cyclediffusion": "CycleDiffusion",
}

MODEL_SIZES = {
    "cyclegan":       "model_medium",
    "unit":           "model_medium",
    "munit":          "model_medium",
    "dclgan":         "model_small",
    "uvcgan":         "model_small",
    "cyclediffusion": "model_small",
}

DEFAULT_BASE = Path("/work2/bz66izin-VSproject/ensemble")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate tile-level uncertainty to WSI-level mean per model."
    )
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"Ensemble root directory (default: {DEFAULT_BASE})")
    ap.add_argument("--outdir", type=Path, default=Path("cpa_mae_per_wsi"),
                    help="Output directory (default: ./cpa_mae_per_wsi/)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for model in MODELS:
        model_size   = MODEL_SIZES[model]
        per_wsi_dir  = (args.base / model / "data_large" / model_size
                        / "uncertainty" / model / "per_wsi_csv")

        if not per_wsi_dir.exists():
            print(f"  [WARN] Not found — skipping {model}: {per_wsi_dir}")
            continue

        csv_files = sorted(per_wsi_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [WARN] Empty directory — skipping {model}: {per_wsi_dir}")
            continue

        model_rows = []
        for csv_path in csv_files:
            wsi_stem = csv_path.stem
            tiles    = pd.read_csv(csv_path, usecols=["mean_uncertainty"])["mean_uncertainty"].dropna()
            if tiles.empty:
                print(f"    [WARN] No tile data in {csv_path.name} — skipping.")
                continue
            model_rows.append({
                "model":            MODEL_DISPLAY_NAMES[model],
                "wsi":              wsi_stem,
                "mean_uncertainty": float(tiles.mean()),
                "std_uncertainty":  float(tiles.std(ddof=1)) if len(tiles) > 1 else 0.0,
                "n_tiles":          int(len(tiles)),
            })

        all_rows.extend(model_rows)
        print(f"  {MODEL_DISPLAY_NAMES[model]:>15}: {len(model_rows)} WSI(s)  "
              f"mean_uncertainty = "
              f"{np.mean([r['mean_uncertainty'] for r in model_rows]):.4f}")

    if not all_rows:
        raise RuntimeError(
            f"No per_wsi_csv data found under {args.base}. "
            "Run aggregate_uncertainty.sh first."
        )

    out_df   = pd.DataFrame(all_rows)
    out_path = args.outdir / "wsi_uncertainty.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")

    print(f"\nSaved → {out_path}")
    print(f"Shape : {out_df.shape}  ({len(out_df)} rows = "
          f"{out_df['model'].nunique()} models × "
          f"{out_df['wsi'].nunique()} WSIs)")
    print("\nPreview:")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
