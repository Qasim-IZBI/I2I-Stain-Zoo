"""
vis_inference.py — Visual inference grid for all 54 training runs.

For each run (6 models × 3 model sizes × 3 data sizes) the script finds the
latest checkpoint and runs A→B inference on a handful of testA images.

Output: 3 PNG figures, each a 6-row (model type) × 3-col grid.

  --group_by data_size   (default)
      3 figures — one per model size (small / medium / large)
      columns   — data sizes (small / medium / large)

  --group_by model_size
      3 figures — one per data size (small / medium / large)
      columns   — model sizes (small / medium / large)

Each cell shows a horizontal strip of translated images.
Add --show_source to prepend the corresponding source image above each strip.
"""

import argparse
import glob
import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ensure repo-root imports work when the script is called from another directory
REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from datasets.transforms import default_train_transform
from models.cyclegan import CycleGAN, CycleGANConfig
from models.dclgan   import DCLGAN,  DCLGANConfig
from models.miudiff  import MIUDiff, MIUDiffConfig
from models.munit    import MUNIT,   MUNITConfig
from models.unit     import UNIT,    UNITConfig
from models.uvcgan   import UVCGAN,  UVCGANConfig

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (mirrors the sbatch scripts)
# ──────────────────────────────────────────────────────────────────────────────
BASE_OUTPUTS = "/work2/bz66izin-VSproject/Outputs"
TEST_A       = "/work2/bz66izin-VSproject/VS_Data/QP_HE/tiles/testA"

MODELS    = ["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"]
SIZES     = ["small", "medium", "large"]
DATASIZES = ["small", "medium", "large"]

# For multi-stage models, inference uses the final stage's checkpoint
LAST_STAGE = {"miudiff": "stage3", "uvcgan": "stage2"}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ckpt_root(model: str, datasize: str, size: str, base: str) -> str:
    run_dir = os.path.join(base, model, "results", f"data_{datasize}", f"model_{size}")
    stage = LAST_STAGE.get(model)
    if stage:
        run_dir = os.path.join(run_dir, stage)
    return os.path.join(run_dir, "checkpoints")


def latest_ckpt(model: str, datasize: str, size: str, base: str):
    """Return path to the highest-step checkpoint file, or None."""
    root = _ckpt_root(model, datasize, size, base)
    pts  = glob.glob(os.path.join(root, "step_*.pt"))
    if not pts:
        return None
    return max(
        pts,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str, ckpt_path: str, device, miu_steps: int = 50):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config")

    if model_name == "cyclegan":
        net = CycleGAN(CycleGANConfig(**cfg) if cfg else CycleGANConfig())

    elif model_name == "unit":
        net = UNIT(UNITConfig(**cfg) if cfg else UNITConfig())

    elif model_name == "munit":
        net = MUNIT(MUNITConfig(**cfg) if cfg else MUNITConfig())

    elif model_name == "dclgan":
        net = DCLGAN(DCLGANConfig(**cfg) if cfg else DCLGANConfig())

    elif model_name == "miudiff":
        if cfg:
            cfg = dict(cfg)
            cfg["stage"]        = "finetune"
            cfg["sample_steps"] = miu_steps
        else:
            cfg = {"stage": "finetune", "sample_steps": miu_steps}
        net = MIUDiff(MIUDiffConfig(**cfg))

    elif model_name == "uvcgan":
        net = UVCGAN(UVCGANConfig(**cfg) if cfg else UVCGANConfig())

    else:
        raise ValueError(f"Unknown model: {model_name}")

    sd = ckpt.get("model", ckpt)
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()
    return net


# ──────────────────────────────────────────────────────────────────────────────
# Per-model A→B inference
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def translate(net, model_name: str, x: torch.Tensor, device) -> torch.Tensor:
    """Return A→B translated tensor (1, 3, H, W) in [-1, 1]."""
    x = x.to(device)

    if model_name in ("cyclegan", "dclgan", "uvcgan"):
        return net.forward_A2B(x)

    if model_name == "unit":
        y, _ = net.forward_A2B(x)
        return y

    if model_name == "munit":
        c, _ = net.encode_A(x)
        s    = torch.randn(1, net.cfg.style_dim, device=device)
        return net.decode_B(c, s)

    if model_name == "miudiff":
        return net.sample_A2B(x)

    raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────
def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) in [-1, 1] → (H, W, 3) uint8."""
    arr = t.squeeze(0).cpu().float().clamp(-1, 1)
    return ((arr + 1) / 2 * 255).byte().numpy().transpose(1, 2, 0)


def collect_images(data_dir: str, n: int, seed: int = 42) -> list:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    paths = [
        os.path.join(root, fname)
        for root, _, files in os.walk(data_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() in exts
    ]
    if not paths:
        return []
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


def make_cell_image(src_row, tgt_row, show_source: bool) -> np.ndarray:
    """
    Horizontal strip of translated tiles.
    When show_source=True, source row is stacked above with a thin separator.
    Returns (H, W, 3) uint8.
    """
    strip_tgt = np.concatenate(tgt_row, axis=1)
    if show_source and src_row:
        strip_src = np.concatenate(src_row[: len(tgt_row)], axis=1)
        sep = np.full((4, strip_tgt.shape[1], 3), 180, dtype=np.uint8)
        return np.concatenate([strip_src, sep, strip_tgt], axis=0)
    return strip_tgt


# ──────────────────────────────────────────────────────────────────────────────
# Figure builder
# ──────────────────────────────────────────────────────────────────────────────
def build_figure(cells, row_labels, col_labels, title, num_images, show_source):
    """
    cells: dict (row_idx, col_idx) → ndarray | None
    Returns a matplotlib Figure with a 6×3 subplot grid.
    """
    nrows  = len(row_labels)
    ncols  = len(col_labels)
    cell_w = 1.3 * num_images
    cell_h = 1.3 * (2 if show_source else 1)
    fig_w  = 1.4 + ncols * cell_w
    fig_h  = 0.7 + nrows * cell_h

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.04, "hspace": 0.10},
        squeeze=False,
    )

    for r, row_lbl in enumerate(row_labels):
        for c, col_lbl in enumerate(col_labels):
            ax   = axes[r][c]
            cell = cells.get((r, c))
            if cell is not None:
                ax.imshow(cell, interpolation="bilinear")
            else:
                ax.set_facecolor("#d8d8d8")
                ax.text(0.5, 0.5, "no checkpoint",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=7, color="#555")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_lbl, fontsize=9, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(row_lbl, fontsize=9, fontweight="bold", labelpad=4)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Run A→B inference on a sample of testA tiles for all 54 runs "
                    "and save one 6×3 figure per size level."
    )
    ap.add_argument("--base",        default=BASE_OUTPUTS,
                    help="Root outputs directory  [%(default)s]")
    ap.add_argument("--data",        default=TEST_A,
                    help="testA tile directory     [%(default)s]")
    ap.add_argument("--outdir",      default=None,
                    help="Where to save figures    [BASE/vis_inference/]")
    ap.add_argument("--group_by",    choices=["model_size", "data_size"],
                    default="data_size",
                    help="What the 3 columns represent in each figure  [%(default)s]")
    ap.add_argument("--num_images",  type=int, default=3,
                    help="Tiles shown per cell                          [%(default)s]")
    ap.add_argument("--show_source", action="store_true",
                    help="Add source-image row above translated row")
    ap.add_argument("--miu_steps",   type=int, default=50,
                    help="DDIM steps for MIUDiff (fewer = faster)       [%(default)s]")
    ap.add_argument("--seed",        type=int, default=42,
                    help="RNG seed for image sampling and style noise   [%(default)s]")
    ap.add_argument("--device",      default=None,
                    help="torch device  (auto-detected if omitted)")
    ap.add_argument("--sizes",       nargs="+", choices=["small", "medium", "large"],
                    default=None,
                    help="Which size levels to generate figures for (default: all three). "
                         "With --group_by data_size these are model sizes; "
                         "with --group_by model_size these are data sizes. "
                         "Example: --sizes small large")
    ap.add_argument("--dry_run",     action="store_true",
                    help="Print checkpoint paths only; skip model loading and inference")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.base, "vis_inference")
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Outdir : {outdir}")
    print(f"GroupBy: {args.group_by}  ({args.num_images} images/cell"
          + (", show_source" if args.show_source else "") + ")")

    # ── collect test images ──────────────────────────────────────────────────
    img_paths = collect_images(args.data, args.num_images, seed=args.seed)
    if not img_paths and not args.dry_run:
        sys.exit(f"[ERROR] No images found in: {args.data}")
    print(f"\nTest images ({len(img_paths)}):")
    for p in img_paths:
        print(f"  {p}")

    transform = default_train_transform(image_size=256)
    src_tensors, src_arrays = [], []
    for p in img_paths:
        t = transform(Image.open(p).convert("RGB")).unsqueeze(0)
        src_tensors.append(t)
        src_arrays.append(_to_uint8(t))

    # ── axis assignments based on group_by ─────────────────────────────────
    if args.group_by == "data_size":
        # one figure per model size; columns = data sizes
        outer_axis    = SIZES
        inner_axis    = DATASIZES
        col_header    = "Data"
        fig_header    = "Model size"

        def to_run(model, outer, inner):
            return model, inner, outer   # (model, datasize, modelsize)

    else:  # model_size
        # one figure per data size; columns = model sizes
        outer_axis    = DATASIZES
        inner_axis    = SIZES
        col_header    = "Model"
        fig_header    = "Data size"

        def to_run(model, outer, inner):
            return model, outer, inner   # (model, datasize, modelsize)

    if args.sizes:
        unknown = set(args.sizes) - set(outer_axis)
        if unknown:
            sys.exit(f"[ERROR] --sizes {unknown} not valid for --group_by {args.group_by}; "
                     f"choices: {outer_axis}")
        outer_axis = [v for v in outer_axis if v in args.sizes]

    torch.manual_seed(args.seed)

    # ── one figure per outer_val ─────────────────────────────────────────────
    for outer_val in outer_axis:
        col_labels = [f"{col_header}: {v}" for v in inner_axis]
        row_labels = [m.upper() for m in MODELS]
        cells      = {}

        print(f"\n{'─'*64}")
        print(f"{fig_header}: {outer_val}")
        print(f"{'─'*64}")

        for r, model in enumerate(MODELS):
            for c, inner_val in enumerate(inner_axis):
                m_name, datasize, modelsize = to_run(model, outer_val, inner_val)
                ckpt = latest_ckpt(m_name, datasize, modelsize, args.base)

                tag = f"  {m_name:10s}  data={datasize:7s}  model={modelsize:7s}"
                if ckpt is None:
                    warnings.warn(
                        f"No checkpoint: {m_name} data={datasize} model={modelsize}"
                    )
                    print(f"{tag}  → [MISSING]")
                    cells[(r, c)] = None
                    continue

                step_name = os.path.splitext(os.path.basename(ckpt))[0]
                print(f"{tag}  → {step_name}", flush=True)

                if args.dry_run:
                    cells[(r, c)] = None
                    continue

                # ── load ──
                try:
                    net = load_model(m_name, ckpt, device, miu_steps=args.miu_steps)
                except Exception as exc:
                    warnings.warn(f"Load failed ({m_name} {step_name}): {exc}")
                    cells[(r, c)] = None
                    continue

                # ── inference ──
                tgt_imgs = []
                for t in src_tensors:
                    try:
                        y = translate(net, m_name, t, device)
                        tgt_imgs.append(_to_uint8(y))
                    except Exception as exc:
                        warnings.warn(f"Inference failed ({m_name}): {exc}")
                        tgt_imgs.append(np.zeros((256, 256, 3), dtype=np.uint8))

                cells[(r, c)] = make_cell_image(src_arrays, tgt_imgs, args.show_source)

                del net
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if args.dry_run:
            continue

        title = (
            f"{fig_header}: {outer_val}  —  A→B inference  "
            f"({args.num_images} tiles/cell"
            + (", source shown" if args.show_source else "")
            + ")"
        )
        fig = build_figure(
            cells, row_labels, col_labels, title, args.num_images, args.show_source
        )
        out_path = os.path.join(outdir, f"vis_{args.group_by}_{outer_val}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved → {out_path}")

    if not args.dry_run:
        print(f"\nAll figures saved to: {outdir}")


if __name__ == "__main__":
    main()
