"""
plot_combined_metrics.py — single figure with four subplots:
  Patch-SSIM | LPIPS | FID | PSR-MAE (paired)

Sources
-------
  Patch-SSIM / LPIPS / FID : metric CSVs produced by eval_all_configs.sh /
                              eval_all_54.sh (same layout as plot_eval_metrics.py)
  PSR-MAE                  : summary.json files produced by compare_psr.py /
                              compare_psr_all_configs.sh (same layout as
                              rank_psr_configs.py)

Usage
-----
  python plot_combined_metrics.py \
      --eval_indir  /work2/bz66izin-VSproject/Eval \
      --psr_indir   /work2/bz66izin-VSproject/psr_comparison \
      --outdir      ./combined_metrics_plot/

The legend is placed as a standalone panel to the left of all four subplots.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── shared constants ──────────────────────────────────────────────────────────

MODELS             = ["cyclegan", "unit", "munit", "dclgan", "uvcgan", "cyclediffusion"]
DATASIZES          = ["small", "medium", "large"]
MODEL_SIZES        = ["small", "medium", "large"]
DATA_SIZE_COLORS   = {"small": "#1f77b4", "medium": "#ff7f0e", "large": "#2ca02c"}
MODEL_SIZE_MARKERS = {"small": "o", "medium": "s", "large": "^"}

EVAL_METRICS = {
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
    "fid": {
        "filenames":     ["fid.csv", "fid_inception.csv"],
        "value_col":     "value",
        "higher_better": False,
        "ylabel":        "FID ↓",
        "title":         "FID (Inception)",
        "has_std":       False,
    },
}


# ── eval metric helpers (from plot_eval_metrics.py) ───────────────────────────

def _find_eval_csv(indir: Path, model: str, model_size: str, data_size: str,
                   filenames: list) -> Path | None:
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


def _read_eval_csv(csv_path: Path, value_col: str, has_std: bool) -> tuple[float, float, list]:
    df = pd.read_csv(csv_path, dtype=str)
    if "filename" in df.columns:
        data = df[df["filename"] != "MEAN"].copy()
        data[value_col] = data[value_col].astype(float)
        if has_std:
            has_relpath = data["filename"].str.contains("/", regex=False).any()
            if has_relpath:
                data["wsi"] = data["filename"].apply(lambda f: str(f).split("/")[0])
                wsi_means = data.groupby("wsi")[value_col].mean()
                mean = float(wsi_means.mean())
                std  = float(wsi_means.std(ddof=1)) if len(wsi_means) > 1 else 0.0
                return mean, std, wsi_means.tolist()
        vals  = data[value_col]
        mean  = float(vals.mean())
        std   = float(vals.std(ddof=1)) if (has_std and len(vals) > 1) else 0.0
        return mean, std, []
    mean = float(df[value_col].iloc[0])
    return mean, 0.0, []


def load_eval_data(indir: Path) -> tuple[pd.DataFrame, dict]:
    rows     = []
    wsi_data = {}
    for model in MODELS:
        for model_size in MODEL_SIZES:
            for data_size in DATASIZES:
                row = {"model": model, "model_size": model_size, "data_size": data_size}
                found_any = False
                for metric, mcfg in EVAL_METRICS.items():
                    p = _find_eval_csv(indir, model, model_size, data_size, mcfg["filenames"])
                    if p is None:
                        row[metric] = None
                        row[f"{metric}_std"] = None
                        continue
                    try:
                        mean, std, wsi_vals = _read_eval_csv(p, mcfg["value_col"], mcfg["has_std"])
                        row[metric] = mean
                        row[f"{metric}_std"] = std
                        if wsi_vals:
                            wsi_data[(model, model_size, data_size, metric)] = wsi_vals
                        found_any = True
                    except Exception as e:
                        print(f"[WARN] {p}: {e}")
                        row[metric] = None
                        row[f"{metric}_std"] = None
                if found_any:
                    rows.append(row)
    return pd.DataFrame(rows), wsi_data


# ── PSR helpers (from rank_psr_configs.py) ────────────────────────────────────

def _parse_config(config: str):
    left, right = config.split("/")
    return left.replace("_model", ""), right.replace("_data", "")


def load_psr_data(indir: Path) -> tuple[pd.DataFrame, dict]:
    rows     = []
    wsi_data = {}
    for model in MODELS:
        json_path = indir / model / "summary.json"
        if not json_path.exists():
            print(f"[WARN] Missing PSR summary: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        for config, m in data.get("pairwise_vs_real", {}).items():
            model_size, data_size = _parse_config(config)
            rows.append({
                "model":          model,
                "model_size":     model_size,
                "data_size":      data_size,
                "mae_paired":     m.get("mae_paired"),
                "mae_paired_std": m.get("mae_paired_std"),
                "pearson_r":      m.get("pearson_r"),
                "n_matched":      m.get("n_matched"),
            })
        per_wsi_path = indir / model / "per_wsi.csv"
        if per_wsi_path.exists():
            pw = pd.read_csv(per_wsi_path)
            real_df = (pw[pw["condition"] == "real"][["wsi", "psr_fraction"]]
                       .rename(columns={"psr_fraction": "real_frac"}))
            for config in pw["condition"].unique():
                if config == "real":
                    continue
                gen_df = (pw[pw["condition"] == config][["wsi", "psr_fraction"]]
                          .rename(columns={"psr_fraction": "gen_frac"}))
                paired = real_df.merge(gen_df, on="wsi")
                if not paired.empty:
                    ms, ds = _parse_config(config)
                    wsi_data[(model, ms, ds)] = (
                        (paired["gen_frac"] - paired["real_frac"]).abs().tolist()
                    )
    return pd.DataFrame(rows), wsi_data


def pick_best_psr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df[df["data_size"] == "large"]
          .dropna(subset=["mae_paired"])
          .sort_values(["mae_paired", "pearson_r"], ascending=[True, False])
          .groupby("model", sort=False)
          .first()
          .reset_index()
    )


# ── shared panel drawing ──────────────────────────────────────────────────────

def _offsets():
    combos  = [(ms, ds) for ds in DATASIZES for ms in MODEL_SIZES]
    spacing = 0.08
    return {c: (i - (len(combos) - 1) / 2) * spacing for i, c in enumerate(combos)}


def _draw_separators(ax, n_models: int) -> None:
    """Draw light vertical lines between each model group."""
    for i in range(n_models - 1):
        ax.axvline(i + 0.5, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)


def _draw_eval_panel(ax, df: pd.DataFrame, metric: str, mcfg: dict,
                     wsi_data: dict) -> None:
    models = [m for m in MODELS if m in df["model"].values]
    x_pos  = {m: i for i, m in enumerate(models)}
    offsets = _offsets()

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
                std     = float(std_raw) if (std_raw is not None and
                                             not np.isnan(float(std_raw))) else 0.0
                x = x_pos[model] + off
                if mcfg["has_std"]:
                    wsi_vals = wsi_data.get((model, model_size, data_size, metric), [])
                    if wsi_vals:
                        ax.scatter(
                            [x] * len(wsi_vals), wsi_vals,
                            color=DATA_SIZE_COLORS[data_size],
                            marker=MODEL_SIZE_MARKERS[model_size],
                            s=18, alpha=0.35, linewidths=0, zorder=2,
                        )
                xs.append(x)
                ys.append(val)
                lowers.append(min(std, val))
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
                zorder=3,
            )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(list(x_pos.keys()), fontsize=8, rotation=0, ha="center")
    ax.set_ylabel(mcfg["ylabel"], fontsize=9)
    ax.set_title(mcfg["title"], fontsize=9)
    ax.set_ylim(bottom=0)
    _draw_separators(ax, len(models))


def _draw_psr_panel(ax, df: pd.DataFrame, best: pd.DataFrame,
                    wsi_data: dict) -> None:
    models  = [m for m in MODELS if m in df["model"].values]
    x_pos   = {m: i for i, m in enumerate(models)}
    offsets = _offsets()

    for model_size in MODEL_SIZES:
        for data_size in DATASIZES:
            off = offsets[(model_size, data_size)]
            xs, ys, lowers, uppers = [], [], [], []
            for model in models:
                row = df[
                    (df["model"] == model) &
                    (df["model_size"] == model_size) &
                    (df["data_size"] == data_size)
                ].dropna(subset=["mae_paired"])
                if row.empty:
                    continue
                mae = float(row["mae_paired"].iloc[0])
                std_raw = row["mae_paired_std"].iloc[0] if "mae_paired_std" in row.columns else None
                std = float(std_raw) if (std_raw is not None and
                                         not (isinstance(std_raw, float) and np.isnan(std_raw))) else 0.0
                x = x_pos[model] + off
                wsi_vals = wsi_data.get((model, model_size, data_size), [])
                if wsi_vals:
                    ax.scatter(
                        [x] * len(wsi_vals), wsi_vals,
                        color=DATA_SIZE_COLORS[data_size],
                        marker=MODEL_SIZE_MARKERS[model_size],
                        s=18, alpha=0.35, linewidths=0, zorder=2,
                    )
                xs.append(x)
                ys.append(mae)
                lowers.append(min(std, mae))
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
                zorder=3,
            )

    # star = best config per model (large data only)
    if not best.empty:
        for _, brow in best.iterrows():
            model      = brow["model"]
            mae        = brow.get("mae_paired")
            model_size = brow.get("model_size")
            data_size  = brow.get("data_size")
            if model not in x_pos or mae is None or pd.isna(mae):
                continue
            off = offsets.get((model_size, data_size), 0)
            ax.scatter(
                [x_pos[model] + off], [mae],
                color=DATA_SIZE_COLORS.get(data_size, "gray"),
                marker="*", s=260, edgecolors="black", linewidths=0.8, zorder=5,
            )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(list(x_pos.keys()), fontsize=8, rotation=0, ha="center")
    ax.set_ylabel("PSR-MAE (paired) ↓", fontsize=9)
    ax.set_title("PSR-MAE", fontsize=9)
    ax.set_ylim(bottom=0)
    _draw_separators(ax, len(models))


# ── legend handles ────────────────────────────────────────────────────────────

def _legend_handles(include_star: bool = True) -> list:
    handles = [
        Patch(facecolor=DATA_SIZE_COLORS[s], edgecolor="black", linewidth=0.7,
              label=f"{s} data")
        for s in DATASIZES
    ]
    handles += [
        Line2D([0], [0], color="black", marker=MODEL_SIZE_MARKERS[s],
               linestyle="none", markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"{s} model")
        for s in MODEL_SIZES
    ]
    if include_star:
        handles.append(
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="gray", markeredgecolor="black",
                   markersize=12, label="best PSR config")
        )
    return handles


# ── main figure ───────────────────────────────────────────────────────────────

def plot_combined(eval_df: pd.DataFrame, eval_wsi: dict,
                  psr_df: pd.DataFrame,  psr_wsi: dict,
                  best_psr: pd.DataFrame, outpath: Path) -> None:

    # Layout: 2×2 metric panels + narrow legend column on the right
    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.25],
        wspace=0.20,
        hspace=0.25,
        left=0.06, right=0.98,
        top=0.92, bottom=0.10,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    metric_order = ["patch_ssim", "lpips", "fid"]
    for ax, metric in zip(axes[:3], metric_order):
        _draw_eval_panel(ax, eval_df, metric, EVAL_METRICS[metric], eval_wsi)

    _draw_psr_panel(axes[3], psr_df, best_psr, psr_wsi)

    # Single shared legend inside the right panel
    handles = _legend_handles(include_star=not best_psr.empty)
    ax_leg.legend(
        handles=handles,
        loc="center",
        fontsize=7.5,
        frameon=True,
        title="Colour = data size\nMarker = model size",
        title_fontsize=7,
        handlelength=1.2,
        handleheight=0.8,
        borderpad=0.5,
        labelspacing=0.35,
        handletextpad=0.4,
    )

    fig.suptitle(
        "Evaluation metrics — 9 configs per model family",
        fontsize=10, y=0.97,
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined 1×4 metric figure: Patch-SSIM, LPIPS, FID, PSR-MAE."
    )
    parser.add_argument(
        "--eval_indir", type=Path, required=True,
        help="Root evaluation directory with metric CSVs "
             "(same --indir as plot_eval_metrics.py).",
    )
    parser.add_argument(
        "--psr_indir", type=Path, required=True,
        help="PSR comparison directory with per-model summary.json files "
             "(same --indir as rank_psr_configs.py).",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("combined_metrics_plot"),
        help="Output directory [%(default)s]",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading eval metrics …")
    eval_df, eval_wsi = load_eval_data(args.eval_indir)
    if eval_df.empty:
        print("[WARN] No eval metric CSVs found — eval panels will be empty.")

    print("Loading PSR data …")
    psr_df, psr_wsi = load_psr_data(args.psr_indir)
    best_psr = pick_best_psr(psr_df)
    if psr_df.empty:
        print("[WARN] No PSR summary.json files found — PSR panel will be empty.")

    # ── save combined CSV ──────────────────────────────────────────────────────
    keys = ["model", "model_size", "data_size"]
    combined = eval_df.merge(
        psr_df[keys + ["mae_paired", "mae_paired_std", "pearson_r", "n_matched"]],
        on=keys, how="outer",
    ).sort_values(keys).reset_index(drop=True)
    csv_path = args.outdir / "combined_metrics.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    outpath = args.outdir / "combined_metrics.png"
    plot_combined(eval_df, eval_wsi, psr_df, psr_wsi, best_psr, outpath)


if __name__ == "__main__":
    main()
