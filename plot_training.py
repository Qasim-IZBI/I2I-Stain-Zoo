# plot_training.py
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch


def find_latest_checkpoint(ckpt_dir):
    """Find the checkpoint with the highest step number."""
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r"step_(\d+)\.pt$")
    best, best_step = None, -1
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best = os.path.join(ckpt_dir, f)
    return best


def main():
    parser = argparse.ArgumentParser("Plot Training Losses & Save Summary")
    parser.add_argument("--run", type=str, required=True,
                        help="Training output directory (contains loss_log.csv and checkpoints/)")
    args = parser.parse_args()

    run_dir = args.run
    log_path = os.path.join(run_dir, "loss_log.csv")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # ---- Load loss log ----
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No loss_log.csv found in {run_dir}")

    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} log entries from {log_path}")

    # Identify loss columns (everything except step/elapsed columns)
    loss_cols = [c for c in df.columns if c not in ("global_step", "elapsed_s")]

    x = df["global_step"]
    plot_dir = os.path.join(run_dir, "loss_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Individual plots ----
    for col in loss_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, df[col], linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved {col} plot to {plot_dir}/{col}.png")

    # ---- Combined plot ----
    n = len(loss_cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for idx, col in enumerate(loss_cols):
        ax = axes[idx, 0]
        ax.plot(x, df[col], linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    combined_path = os.path.join(plot_dir, "all_losses.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined loss plot to {combined_path}")

    # ---- Load training metadata (saved by BaseTrainer at start of training) ----
    meta_path = os.path.join(run_dir, "training_meta.json")
    training_meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            training_meta = json.load(f)
        print(f"Loaded training metadata from {meta_path}")
    else:
        print("No training_meta.json found — falling back to checkpoint only")

    # ---- Load checkpoint for config (fallback if no training_meta) ----
    model_name = None
    config = None
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_name = ckpt.get("model_name")
        config = ckpt.get("config")
        print(f"Loaded config from {ckpt_path}")

    # ---- Build summary ----
    last_row = df.iloc[-1]
    final_losses = {col: float(last_row[col]) for col in loss_cols}

    # Use training_meta as base if available, otherwise build from checkpoint
    if training_meta:
        summary = training_meta
        summary["training_info"] = {
            "total_steps": int(df["global_step"].max()),
            "final_losses": final_losses,
        }
    else:
        summary = {
            "model_name": model_name,
            "config": _make_serializable(config) if config else None,
            "training_info": {
                "total_steps": int(df["global_step"].max()),
                "final_losses": final_losses,
            },
        }

    summary["loss_history"] = df.to_dict(orient="records")

    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")


def _make_serializable(obj):
    """Convert tuples and other non-JSON types for json.dump."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
