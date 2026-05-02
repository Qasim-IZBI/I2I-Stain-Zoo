# infer.py
import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import get_device

from datasets.single_domain_dataset import SingleDomainDataset
from datasets.target_only_dataset import TargetOnlyDataset
from datasets.transforms import default_train_transform

from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig
from models.miudiff import MIUDiff, MIUDiffConfig
from models.uvcgan import UVCGAN, UVCGANConfig


# =============================================================================
# Reinhard LAB colour transfer helpers
# =============================================================================

def _rgb_to_lab(img):
    """[H,W,3] float32 [0,1] -> CIE LAB"""
    img = img.clip(0, 1)
    lin = np.where(img > 0.04045,
                   ((img + 0.055) / 1.055) ** 2.4,
                   img / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = (lin.reshape(-1, 3) @ M.T).reshape(img.shape)
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(xyz.clip(0)), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab):
    """CIE LAB [H,W,3] -> [H,W,3] float32 [0,1]"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps, kappa = 0.008856, 903.3
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    y = np.where(L > eps * kappa, ((L + 16.0) / 116.0) ** 3, L / kappa)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([x, y, z], axis=-1) * np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    lin = (xyz.reshape(-1, 3) @ M_inv.T).reshape(xyz.shape).clip(0, None)
    rgb = np.where(lin > 0.0031308,
                   1.055 * lin ** (1.0 / 2.4) - 0.055,
                   12.92 * lin)
    return rgb.clip(0, 1)


def _load_color_ref_stats(path):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    lab = _rgb_to_lab(img)
    flat = lab.reshape(-1, 3)
    return {"mean": flat.mean(axis=0), "std": flat.std(axis=0) + 1e-6}


def save_tile(y, path, color_ref_stats=None):
    """Save a [-1,1] NCHW tensor tile with optional Reinhard colour normalisation."""
    img = (y.squeeze(0).permute(1, 2, 0).cpu().float().numpy().clip(-1, 1) + 1.0) / 2.0
    if color_ref_stats is not None:
        lab = _rgb_to_lab(img)
        for c in range(3):
            m = lab[..., c].mean()
            s = lab[..., c].std() + 1e-6
            lab[..., c] = (lab[..., c] - m) / s * color_ref_stats["std"][c] + color_ref_stats["mean"][c]
        img = _lab_to_rgb(lab)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    save_image(t, path)


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_cfg = ckpt.get("config")

    if args.model == "cyclegan":
        cfg = CycleGANConfig(**saved_cfg) if saved_cfg else CycleGANConfig()
        model = CycleGAN(cfg)

    elif args.model == "unit":
        cfg = UNITConfig(**saved_cfg) if saved_cfg else UNITConfig()
        model = UNIT(cfg)

    elif args.model == "munit":
        cfg = MUNITConfig(**saved_cfg) if saved_cfg else MUNITConfig(style_dim=args.style_dim)
        model = MUNIT(cfg)

    elif args.model == "dclgan":
        cfg = DCLGANConfig(**saved_cfg) if saved_cfg else DCLGANConfig()
        model = DCLGAN(cfg)

    elif args.model == "miudiff":
        miu_stage = args.miu_stage
        if saved_cfg:
            ckpt_stage = saved_cfg.get("stage", "finetune")
            if miu_stage == "finetune" and ckpt_stage == "pretrain":
                print("[WARN] Checkpoint was saved at stage 'pretrain' but --miu_stage finetune "
                      "was requested. eps_cond has random weights — output will be garbage. "
                      "Did you mean --miu_stage pretrain?")
            # Override runtime-only params from CLI args
            saved_cfg["stage"] = miu_stage
            saved_cfg["sample_steps"] = args.miu_steps
            saved_cfg["guidance_scale"] = args.miu_guidance
            saved_cfg["miu_pcl"] = args.miu_pcl
            saved_cfg["pcl_refine_steps"] = args.pcl_refine_steps
            saved_cfg["pcl_refine_lr"] = args.pcl_refine_lr
            if args.miu_cond_type is not None:
                saved_cfg["cond_type"] = args.miu_cond_type
            cfg = MIUDiffConfig(**saved_cfg)
        else:
            cfg = MIUDiffConfig(
                stage=miu_stage,
                sample_steps=args.miu_steps,
                guidance_scale=args.miu_guidance,
                miu_pcl=args.miu_pcl,
                pcl_refine_steps=args.pcl_refine_steps,
                pcl_refine_lr=args.pcl_refine_lr,
            )
        model = MIUDiff(cfg)

    elif args.model == "uvcgan":
        cfg = UVCGANConfig(**saved_cfg) if saved_cfg else UVCGANConfig()
        model = UVCGAN(cfg)

    else:
        raise ValueError(args.model)

    if saved_cfg:
        print(f"Restored {args.model} config from checkpoint")

    sd = ckpt["model"] if "model" in ckpt else ckpt
    # MIUDiff: strict=False for both stages —
    #   pretrain ckpt may lack eps_cond keys;
    #   stage-3 ckpt carries PCL-only networks (feat_x, feat_y, proj) not used at inference.
    strict = args.model != "miudiff"
    model.load_state_dict(sd, strict=strict)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser("Unified I2I Inference")

    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"], required=True)
    parser.add_argument("--direction", choices=["A2B", "B2A"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data_range", type=str, default=None,
                        help="Load tiles from a range of numbered folders, e.g. '1,6' loads "
                             "001/images/ through 006/images/ under --data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")

    # Colour normalisation (all models)
    parser.add_argument("--color_ref", type=str, default=None,
                        help="Path to a representative target-domain tile. When provided, "
                             "Reinhard LAB colour transfer is applied to every output tile "
                             "so its colour statistics match the reference.")

    # MUNIT
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--style_image", type=str, default=None,
                        help="Reference image to extract style from (MUNIT only)")

    # MIU-Diff
    parser.add_argument("--miu_cond_type", type=str, default=None,
                        help="Override cond_type from checkpoint (gray|sobel). "
                             "Usually not needed — restored automatically from the checkpoint.")
    parser.add_argument("--miu_stage", choices=["pretrain", "finetune"], default="finetune",
                        help="pretrain: unconditional sampling with eps_uncond only; "
                             "finetune: conditional A→B with MI guidance (default)")
    parser.add_argument("--miu_steps", type=int, default=300)
    parser.add_argument("--miu_guidance", type=float, default=1.0)
    parser.add_argument("--miu_pcl", action="store_true")
    parser.add_argument("--pcl_refine_steps", type=int, default=0)
    parser.add_argument("--pcl_refine_lr", type=float, default=0.05)
    parser.add_argument("--num_uncond_samples", type=int, default=None,
                        help="MIUDiff pretrain only: generate this many unconditional samples "
                             "from pure noise instead of reading from --data")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the global RNG seed for deterministic MIUDiff sampling.")
    parser.add_argument("--miu_noise_level", type=float, default=1.0,
                        help="SDEdit-style init for MIUDiff finetune (0–1). "
                             "1.0 = pure noise (default). Note: values < 1.0 cause HE colour "
                             "bleed for cross-domain (H&E→SR) translation; use --color_ref "
                             "instead for colour consistency.")

    args = parser.parse_args()

    device = get_device()
    model = load_model(args, device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    color_ref_stats = _load_color_ref_stats(args.color_ref) if args.color_ref else None

    transform = default_train_transform(image_size=256)

    data_range = None
    if args.data_range:
        parts = args.data_range.split(",")
        data_range = (int(parts[0]), int(parts[1]))

    # MIUDiff pretrain with --num_uncond_samples: skip dataset entirely
    uncond_count_mode = (
        args.model == "miudiff"
        and args.miu_stage == "pretrain"
        and args.num_uncond_samples is not None
    )

    if uncond_count_mode:
        loader = None
    elif args.model == "miudiff" and args.miu_stage == "pretrain":
        # Use TargetOnlyDataset so output filenames are anchored to real B tiles
        dataset = TargetOnlyDataset(args.data, transform=transform, data_range=data_range)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataset = SingleDomainDataset(args.data, transform=transform, data_range=data_range)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- MIUDiff pretrain, count-based (no dataset) ----
    if uncond_count_mode:
        if args.miu_guidance != 1.0:
            print("[WARN] --miu_guidance is ignored for --miu_stage pretrain.")
        for i in range(args.num_uncond_samples):
            y = model.sample_uncond(batch_size=1)
            save_tile(y, f"{args.outdir}/uncond_{i:04d}.tif", color_ref_stats)
        return

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Unpack: SingleDomainDataset yields (tensor, path);
            #         TargetOnlyDataset yields {"B": tensor, "path_B": str}
            if isinstance(batch, dict):
                x    = batch["B"].to(device)
                path = batch["path_B"]
            else:
                x, path = batch
                x = x.to(device)

            rel  = os.path.relpath(path[0], args.data)
            stem = os.path.splitext(rel)[0]   # e.g. "001/images/0000001"
            os.makedirs(os.path.join(args.outdir, os.path.dirname(stem)), exist_ok=True)

            if args.model == "cyclegan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)

            elif args.model == "unit":
                if args.direction == "A2B":
                    y, _ = model.forward_A2B(x)
                else:
                    y, _ = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)

            elif args.model == "munit":
                if args.direction == "A2B":
                    c, _ = model.encode_A(x)
                    if args.style_image:
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_B(ref)
                        y = model.decode_B(c, s)
                        save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_B(c, s)
                            save_tile(y, f"{args.outdir}/{stem}_{k}.tif", color_ref_stats)
                else:
                    c, _ = model.encode_B(x)
                    if args.style_image:
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_A(ref)
                        y = model.decode_A(c, s)
                        save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_A(c, s)
                            save_tile(y, f"{args.outdir}/{stem}_{k}.tif", color_ref_stats)

            elif args.model == "dclgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)

            elif args.model == "miudiff":
                if args.miu_stage == "pretrain":
                    if args.miu_guidance != 1.0:
                        print("[WARN] --miu_guidance is ignored for --miu_stage pretrain.")
                    y = model.sample_uncond(batch_size=1)
                    save_tile(y, f"{args.outdir}/{stem}_uncond.tif", color_ref_stats)
                else:
                    y = model.sample_A2B(x, noise_level=args.miu_noise_level)
                    save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)

            elif args.model == "uvcgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif", color_ref_stats)




if __name__ == "__main__":
    main()
