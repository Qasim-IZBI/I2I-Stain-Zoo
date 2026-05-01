"""
aggregate_eval.py — Collect per-configuration metric CSVs written by eval_all_54.sh
and produce a single summary CSV.

Expected directory layout (written by eval_all_54.sh):
    {EVAL_DIR}/{model}/data_{datasize}/model_{size}/fid_inception.csv
    {EVAL_DIR}/{model}/data_{datasize}/model_{size}/fid_dino.csv
    {EVAL_DIR}/{model}/data_{datasize}/model_{size}/ssim.csv
    {EVAL_DIR}/{model}/data_{datasize}/model_{size}/patch_ssim.csv
    {EVAL_DIR}/{model}/data_{datasize}/model_{size}/lpips.csv

Output:
    {EVAL_DIR}/eval_summary.csv

Usage:
    python aggregate_eval.py --eval_dir /work2/bz66izin-VSproject/Eval
    python aggregate_eval.py --eval_dir /work2/bz66izin-VSproject/Eval --out results/eval_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional


MODELS     = ["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"]
SIZES      = ["small", "medium", "large"]
DATASIZES  = ["small", "medium", "large"]

# (csv_filename, column_with_value)  for each metric
METRIC_FILES = {
    "fid_inception": ("fid_inception.csv", "value"),
    "fid_dino":      ("fid_dino.csv",      "value"),
    "ssim":          ("ssim.csv",          "ssim"),
    "patch_ssim":    ("patch_ssim.csv",    "patch_ssim"),
    "lpips":         ("lpips.csv",         "lpips"),
}

SUMMARY_FIELDS = [
    "model", "model_size", "data_size",
    "fid_inception", "fid_dino",
    "ssim", "patch_ssim", "lpips",
    "n_pairs_ssim", "n_pairs_lpips",
    "n_real_fid", "n_fake_fid",
    "status",
]


def _read_fid_csv(path: str) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """Return (value, n_real, n_fake) from a single-row FID CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return float(row["value"]), int(row.get("n_real", 0)), int(row.get("n_fake", 0))
    return None, None, None


def _read_paired_csv(path: str, value_col: str) -> tuple[Optional[float], int]:
    """
    Return (mean_value, n_pairs) from a per-image CSV that has a final MEAN row
    (filename == 'MEAN').  Falls back to computing the mean manually if the MEAN
    row is absent.
    """
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, 0

    mean_row = next((r for r in rows if r.get("filename", "").upper() == "MEAN"), None)
    data_rows = [r for r in rows if r.get("filename", "").upper() != "MEAN"]

    if mean_row is not None:
        mean_val = float(mean_row[value_col])
    elif data_rows:
        vals = [float(r[value_col]) for r in data_rows if value_col in r]
        mean_val = sum(vals) / len(vals) if vals else None
    else:
        mean_val = None

    return mean_val, len(data_rows)


def aggregate(eval_dir: str) -> list[dict]:
    rows: list[dict] = []
    missing_configs: list[str] = []

    for model in MODELS:
        for datasize in DATASIZES:
            for size in SIZES:
                cfg_dir = Path(eval_dir) / model / f"data_{datasize}" / f"model_{size}"
                row: dict = {
                    "model": model,
                    "model_size": size,
                    "data_size": datasize,
                }

                issues: list[str] = []

                # FID (inception)
                fid_inc_path = cfg_dir / "fid_inception.csv"
                if fid_inc_path.exists():
                    val, n_real, n_fake = _read_fid_csv(str(fid_inc_path))
                    row["fid_inception"] = f"{val:.4f}" if val is not None else ""
                    row["n_real_fid"] = n_real or ""
                    row["n_fake_fid"] = n_fake or ""
                else:
                    row["fid_inception"] = ""
                    row["n_real_fid"] = ""
                    row["n_fake_fid"] = ""
                    issues.append("fid_inception missing")

                # FID (dino)
                fid_dino_path = cfg_dir / "fid_dino.csv"
                if fid_dino_path.exists():
                    val, _, _ = _read_fid_csv(str(fid_dino_path))
                    row["fid_dino"] = f"{val:.4f}" if val is not None else ""
                else:
                    row["fid_dino"] = ""
                    issues.append("fid_dino missing")

                # SSIM
                ssim_path = cfg_dir / "ssim.csv"
                if ssim_path.exists():
                    val, n_pairs = _read_paired_csv(str(ssim_path), "ssim")
                    row["ssim"] = f"{val:.4f}" if val is not None else ""
                    row["n_pairs_ssim"] = n_pairs
                else:
                    row["ssim"] = ""
                    row["n_pairs_ssim"] = ""
                    issues.append("ssim missing")

                # patch-SSIM
                pssim_path = cfg_dir / "patch_ssim.csv"
                if pssim_path.exists():
                    val, _ = _read_paired_csv(str(pssim_path), "patch_ssim")
                    row["patch_ssim"] = f"{val:.4f}" if val is not None else ""
                else:
                    row["patch_ssim"] = ""
                    issues.append("patch_ssim missing")

                # LPIPS
                lpips_path = cfg_dir / "lpips.csv"
                if lpips_path.exists():
                    val, n_pairs = _read_paired_csv(str(lpips_path), "lpips")
                    row["lpips"] = f"{val:.4f}" if val is not None else ""
                    row["n_pairs_lpips"] = n_pairs
                else:
                    row["lpips"] = ""
                    row["n_pairs_lpips"] = ""
                    issues.append("lpips missing")

                row["status"] = "complete" if not issues else "partial: " + ", ".join(issues)
                rows.append(row)

                if issues:
                    missing_configs.append(f"{model}/data_{datasize}/model_{size}: {', '.join(issues)}")

    if missing_configs:
        print(f"\n[WARNING] {len(missing_configs)} configurations have missing metric files:")
        for c in missing_configs:
            print(f"  {c}")
        print()

    return rows


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-config eval CSVs into one summary CSV.")
    ap.add_argument("--eval_dir", required=True,
                    help="Root evaluation directory written by eval_all_54.sh")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: {eval_dir}/eval_summary.csv)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.eval_dir, "eval_summary.csv")

    print(f"Scanning: {args.eval_dir}")
    rows = aggregate(args.eval_dir)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(1 for r in rows if r["status"] == "complete")
    print(f"Wrote {len(rows)} rows ({complete}/{len(rows)} complete) → {out_path}")


if __name__ == "__main__":
    main()
