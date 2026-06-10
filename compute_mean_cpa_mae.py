"""Compute mean CPA-MAE per WSI per model from compare_psr_ensemble.sh output.

For each model, reads the per_wsi.csv produced by compare_psr_ensemble.sh
(which contains PSR fractions for the real SR and all 10 ensemble members)
and computes:

    mean_cpa_mae[wsi] = mean over members of |PSR_fraction_member − PSR_fraction_real|

This gives 5 values per model (one per test WSI), ready to be paired with
WSI-level mean uncertainty from aggregate_uncertainty.py for calibration.

Outputs
-------
{outdir}/mean_cpa_mae.csv  — columns: model, wsi, mean_cpa_mae, std_cpa_mae,
                              n_members, real_psr_fraction, mean_psr_fraction

Example
-------
python compute_mean_cpa_mae.py \
    --cpa_base  /work2/bz66izin-VSproject/ensemble_cpa_comparison \
    --outdir    ./cpa_mae_per_wsi/

python compute_mean_cpa_mae.py   # uses default paths
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

DEFAULT_CPA_BASE = Path("/work2/bz66izin-VSproject/ensemble_cpa_comparison")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_model_cpa_mae(per_wsi_csv: Path) -> pd.DataFrame:
    """Pivot per_wsi.csv to get mean |member_i - real| per WSI.

    Returns a DataFrame with columns:
        wsi, mean_cpa_mae, std_cpa_mae, n_members,
        real_psr_fraction, mean_psr_fraction
    """
    df = pd.read_csv(per_wsi_csv)

    real_rows = df[df["condition"] == "real"].set_index("wsi")["psr_fraction"]
    member_rows = df[df["condition"] != "real"].copy()

    if real_rows.empty:
        raise ValueError(f"No 'real' rows found in {per_wsi_csv}")
    if member_rows.empty:
        raise ValueError(f"No member rows found in {per_wsi_csv}")

    # Per-member absolute difference vs real for each WSI
    member_rows = member_rows[member_rows["wsi"].isin(real_rows.index)].copy()
    member_rows["abs_diff"] = member_rows.apply(
        lambda r: abs(r["psr_fraction"] - real_rows[r["wsi"]]), axis=1
    )

    # Aggregate across members per WSI
    agg = (
        member_rows.groupby("wsi")["abs_diff"]
        .agg(
            mean_cpa_mae="mean",
            std_cpa_mae=lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            n_members="count",
        )
        .reset_index()
    )

    # Attach real PSR fraction and ensemble mean PSR fraction
    agg["real_psr_fraction"] = agg["wsi"].map(real_rows)
    mean_psr = member_rows.groupby("wsi")["psr_fraction"].mean()
    agg["mean_psr_fraction"] = agg["wsi"].map(mean_psr)

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute mean CPA-MAE per WSI per model from compare_psr_ensemble output."
    )
    ap.add_argument("--cpa_base", type=Path, default=DEFAULT_CPA_BASE,
                    help=f"Root of ensemble_cpa_comparison outputs (default: {DEFAULT_CPA_BASE})")
    ap.add_argument("--outdir", type=Path, default=Path("cpa_mae_per_wsi"),
                    help="Output directory (default: ./cpa_mae_per_wsi/)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []

    for model in MODELS:
        csv_path = args.cpa_base / model / "per_wsi.csv"
        if not csv_path.exists():
            print(f"  [WARN] Not found — skipping {model}: {csv_path}")
            continue

        df = compute_model_cpa_mae(csv_path)
        df.insert(0, "model", MODEL_DISPLAY_NAMES[model])
        all_rows.append(df)

        print(f"  {MODEL_DISPLAY_NAMES[model]:>15}: {len(df)} WSI(s)  "
              f"mean_cpa_mae = {df['mean_cpa_mae'].mean():.4f} "
              f"± {df['mean_cpa_mae'].std(ddof=1) if len(df) > 1 else 0:.4f}")

    if not all_rows:
        raise RuntimeError(
            f"No per_wsi.csv files found under {args.cpa_base}. "
            "Run compare_psr_ensemble.sh first."
        )

    out_df = pd.concat(all_rows, ignore_index=True)
    out_path = args.outdir / "mean_cpa_mae.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved → {out_path}")
    print(f"Shape : {out_df.shape}  ({len(out_df)} rows = "
          f"{out_df['model'].nunique()} models × "
          f"{out_df['wsi'].nunique()} WSIs)")
    print("\nColumns: " + ", ".join(out_df.columns.tolist()))
    print("\nPreview:")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
