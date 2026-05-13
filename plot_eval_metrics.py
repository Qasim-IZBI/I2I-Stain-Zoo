"""
plot_eval_metrics.py — visualise FID, patch-SSIM and LPIPS across all 54 configs.

Reads per-metric CSV files produced by eval_all_configs.sh or eval_all_54.sh.
Both path layouts are tried automatically:
  {indir}/{model}/results/data_{d}/model_{s}/{metric}.csv   (eval_all_configs.sh)
  {indir}/{model}/data_{d}/model_{s}/{metric}.csv           (eval_all_54.sh)

Produces a 1×3 grouped-scatter figure using the same visual encoding as
rank_psr_configs.py:

  x = model family   colour = data size   marker = model size
  9 configs per model family

Outputs
-------
all_configs.csv — all (model, config) × metric values for inspection
metrics.png     — 1×3 grouped-scatter figure
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODELS            = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]
DATASIZES         = ["small", "medium", "large"]
MODEL_SIZES       = ["small", "medium", "large"]
DATA_SIZE_COLORS  = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
MODEL_SIZE_MARKERS = {"small": "o", "medium": "s", "large": "^"}

# Per-metric display config. filenames are tried in order; first found wins.
METRICS = {
    "fid": {
        "filenames":     ["fid.csv", "fid_inception.csv"],
        "value_col":     "value",
        "higher_better": False,
        "ylabel":        "FID ↓",
        "title":         "FID (Inception)",
        "has_std":       False,   # distribution-level scalar — no per-image variance
    },
    "patch_ssim": {
        "filenames":     ["patch_ssim.csv"],
        "value_col":     "patch_ssim",
        "higher_better": True,
        "ylabel":        "Patch-SSIM ↑",
        "title":         "Patch-SSIM",
        "has_std":       True,
    },
    "lpips": {
        "filenames":     ["lpips.csv"],
        "value_col":     "lpips",
        "higher_better": False,
        "ylabel":        "LPIPS ↓",
        "title":         "LPIPS",
        "has_std":       True,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_csv(indir: Path, model: str, model_size: str, data_size: str,
              filenames: list) -> Path | None:
    """Try both directory layouts and all candidate filenames; return first hit."""
    prefixes = [
        indir / model / "results" / f"data_{data_size}" / f"model_{model_size}",
        indir / model / f"data_{data_size}" / f"model_{model_size}",
    ]
    for prefix in prefixes:
        for fname in filenames:
            p = prefix / fname
            if p.exists():
                return p
    return None


def _read_csv(csv_path: Path, value_col: str, has_std: bool) -> tuple[float, float]:
    """Return (mean, std). MEAN summary row excluded for per-image CSVs; std=0 for FID."""
    df = pd.read_csv(csv_path, dtype=str)
    if "filename" in df.columns:
        data = df[df["filename"] != "MEAN"][value_col].astype(float)
        mean = float(data.mean())
        std  = float(data.std(ddof=1)) if (has_std and len(data) > 1) else 0.0
    else:
        mean = float(df[value_col].iloc[0])
        std  = 0.0
    return mean, std


# ── data loading ──────────────────────────────────────────────────────────────

def load_all(indir: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for model_size in MODEL_SIZES:
            for data_size in DATASIZES:
                config = f"{model_size}_model/{data_size}_data"
                row = {"model": model, "config": config,
                       "model_size": model_size, "data_size": data_size}
                found_any = False
                for metric, mcfg in METRICS.items():
                    p = _find_csv(indir, model, model_size, data_size, mcfg["filenames"])
                    if p is None:
                        row[metric] = None
                        row[f"{metric}_std"] = None
                        continue
                    try:
                        mean, std = _read_csv(p, mcfg["value_col"], mcfg["has_std"])
                        row[metric] = mean
                        row[f"{metric}_std"] = std
                        found_any = True
                    except Exception as e:
                        print(f"[WARN] {p}: {e}")
                        row[metric] = None
                        row[f"{metric}_std"] = None
                if found_any:
                    rows.append(row)
                else:
                    print(f"[WARN] No CSVs found: {model}/{config}")
    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def _metric_panel(ax, df: pd.DataFrame, metric: str, mcfg: dict) -> None:
    models  = [m for m in MODELS if m in df["model"].values]
    x_pos   = {m: i for i, m in enumerate(models)}
    combos  = [(ms, ds) for ds in DATASIZES for ms in MODEL_SIZES]
    spacing = 0.08
    offsets = {c: (i - (len(combos) - 1) / 2) * spacing for i, c in enumerate(combos)}

    for model_size in MODEL_SIZES:
        for data_size in DATASIZES:
            off = offsets[(model_size, data_size)]
            xs, ys, lowers, uppers = [], [], [], []
            for model in models:
                row = df[
                    (df["model"] == model) &
                    (df["model_size"] == model_size) &
                    (df["data_size"] == data_size)
                ].dropna(subset=[metric])
                if row.empty:
                    continue
                val     = float(row[metric].iloc[0])
                std_raw = row[f"{metric}_std"].iloc[0]
                std     = float(std_raw) if (std_raw is not None and not np.isnan(float(std_raw))) else 0.0
                xs.append(x_pos[model] + off)
                ys.append(val)
                lowers.append(min(std, val))   # clip so bar never crosses 0
                uppers.append(std)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=[lowers, uppers],
                color=DATA_SIZE_COLORS[data_size],
                linestyle="none",
                marker=MODEL_SIZE_MARKERS[model_size], markersize=6,
                markeredgecolor="black", markeredgewidth=0.5,
                ecolor="gray", elinewidth=1, capsize=3, alpha=0.85,
            )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(list(x_pos.keys()), fontsize=8, rotation=12, ha="right")
    ax.set_ylabel(mcfg["ylabel"], fontsize=9)
    ax.set_title(mcfg["title"], fontsize=9)
    ax.set_ylim(bottom=0)


def plot_all(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, mcfg) in zip(axes, METRICS.items()):
        _metric_panel(ax, df, metric, mcfg)

    # shared legend (right side)
    color_handles = [
        Patch(facecolor=DATA_SIZE_COLORS[s], edgecolor="black", linewidth=0.7,
              label=f"{s} data")
        for s in DATASIZES
    ]
    marker_handles = [
        Line2D([0], [0], color="black", marker=MODEL_SIZE_MARKERS[s],
               linestyle="none", markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"{s} model")
        for s in MODEL_SIZES
    ]
    fig.legend(
        handles=color_handles + marker_handles,
        fontsize=8, loc="center right", bbox_to_anchor=(1.02, 0.5),
        title="Colour = data size  |  Marker = model size",
        title_fontsize=7.5, frameon=True,
    )
    fig.suptitle(
        "Evaluation metrics — 9 configs per model family   "
        "colour = data size   marker = model size",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot FID, patch-SSIM and LPIPS across all 54 model configurations."
    )
    parser.add_argument(
        "--indir", type=Path, required=True,
        help="Root evaluation directory containing per-model subdirectories with metric "
             "CSVs (the --outdir / EVAL_DIR / EVAL_BASE used in the eval SLURM script).",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("eval_plots"),
        help="Output directory for CSVs and figure [%(default)s]",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_all(args.indir)
    if df.empty:
        raise RuntimeError(f"No metric CSVs found under {args.indir}")

    df.to_csv(args.outdir / "all_configs.csv", index=False)

    plot_all(df, args.outdir / "metrics.png")


if __name__ == "__main__":
    main()
