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
from datasets.transforms import default_train_transform

from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig
from models.uvcgan import UVCGAN, UVCGANConfig
from models.cyclediffusion import CycleDiffusion, CycleDiffusionConfig


def save_tile(y, path):
    """Save a [-1,1] NCHW tensor tile as an image in [0,1]."""
    img = (y.squeeze(0).permute(1, 2, 0).cpu().float().numpy().clip(-1, 1) + 1.0) / 2.0
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

    elif args.model == "uvcgan":
        cfg = UVCGANConfig(**saved_cfg) if saved_cfg else UVCGANConfig()
        model = UVCGAN(cfg)

    elif args.model == "cyclediffusion":
        if saved_cfg:
            saved_cfg["sample_steps"] = args.cd_steps
            cfg = CycleDiffusionConfig(**saved_cfg)
        else:
            cfg = CycleDiffusionConfig(sample_steps=args.cd_steps)
        model = CycleDiffusion(cfg)

    else:
        raise ValueError(args.model)

    if saved_cfg:
        print(f"Restored {args.model} config from checkpoint")

    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser("Unified I2I Inference")

    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "uvcgan",
                                            "cyclediffusion"], required=True)
    parser.add_argument("--direction", choices=["A2B", "B2A"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data_range", type=str, default=None,
                        help="Load tiles from a range of numbered folders, e.g. '1,6' loads "
                             "001/images/ through 006/images/ under --data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tiles whose output already exists; re-process the "
                             "most recently written tile (guards against partial writes "
                             "on interruption).")

    # MUNIT
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--style_image", type=str, default=None,
                        help="Reference image to extract style from (MUNIT only)")

    # ---- CycleDiffusion inference ----
    parser.add_argument("--cd_steps", type=int, default=200,
                        help="DDIM inversion + decode steps for CycleDiffusion")

    parser.add_argument("--save_aleatoric", action="store_true",
                        help="Save per-pixel aleatoric SD as .npy under {outdir}/aleatoric_npy/ "
                             "(CycleGAN trained with --cyclegan_ugac only). Same [H,W] float32 "
                             "convention as uncertainty.py raw_npy/, so it feeds "
                             "uncertainty_calibration.py directly.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the global RNG seed for deterministic sampling.")

    args = parser.parse_args()

    device = get_device()
    model = load_model(args, device)

    save_aleatoric = args.save_aleatoric
    if save_aleatoric:
        if args.model != "cyclegan" or not getattr(model.cfg, "ugac", False):
            raise SystemExit("--save_aleatoric requires a cyclegan checkpoint trained with "
                             "--cyclegan_ugac (UGAC heads are not present in this model).")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    transform = default_train_transform(image_size=256)

    data_range = None
    if args.data_range:
        parts = args.data_range.split(",")
        data_range = (int(parts[0]), int(parts[1]))

    dataset = SingleDomainDataset(args.data, transform=transform, data_range=data_range)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Resume: collect already-completed output tiles ----
    resume_skip = set()   # absolute paths to skip
    resume_last = None    # absolute path of last tile (re-processed even if it exists)
    if args.resume:
        all_tifs = []
        for root, _, files in os.walk(args.outdir):
            for f in files:
                if f.lower().endswith('.tif'):
                    all_tifs.append(os.path.abspath(os.path.join(root, f)))
        if all_tifs:
            resume_last = max(all_tifs, key=os.path.getmtime)
            resume_skip = set(all_tifs) - {resume_last}
            print(f"[resume] {len(all_tifs)} existing tiles found. "
                  f"Skipping {len(resume_skip)}, re-processing last: "
                  f"{os.path.relpath(resume_last, args.outdir)}")
        else:
            print("[resume] No existing output tiles found; starting from the beginning.")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # SingleDomainDataset yields (tensor, path)
            x, path = batch
            x = x.to(device)

            rel  = os.path.relpath(path[0], args.data)
            stem = os.path.splitext(rel)[0]   # e.g. "001/images/0000001"
            os.makedirs(os.path.join(args.outdir, os.path.dirname(stem)), exist_ok=True)

            if args.model == "cyclegan":
                if save_aleatoric:
                    fwd = (model.forward_A2B_uncertainty if args.direction == "A2B"
                           else model.forward_B2A_uncertainty)
                    y, a_var = fwd(x)
                    npy_path = os.path.join(args.outdir, "aleatoric_npy", stem + ".npy")
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    # store per-pixel aleatoric SD, matching uncertainty.py's raw_npy convention
                    np.save(npy_path, a_var.sqrt().squeeze().cpu().numpy().astype("float32"))
                else:
                    y = (model.forward_A2B(x) if args.direction == "A2B"
                         else model.forward_B2A(x))
                save_tile(y, f"{args.outdir}/{stem}.tif")

            elif args.model == "unit":
                if args.direction == "A2B":
                    y, _ = model.forward_A2B(x)
                else:
                    y, _ = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif")

            elif args.model == "munit":
                if args.direction == "A2B":
                    c, _ = model.encode_A(x)
                    if args.style_image:
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_B(ref)
                        y = model.decode_B(c, s)
                        save_tile(y, f"{args.outdir}/{stem}.tif")
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_B(c, s)
                            suffix = f"_{k}" if args.num_samples > 1 else ""
                            save_tile(y, f"{args.outdir}/{stem}{suffix}.tif")
                else:
                    c, _ = model.encode_B(x)
                    if args.style_image:
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_A(ref)
                        y = model.decode_A(c, s)
                        save_tile(y, f"{args.outdir}/{stem}.tif")
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_A(c, s)
                            suffix = f"_{k}" if args.num_samples > 1 else ""
                            save_tile(y, f"{args.outdir}/{stem}{suffix}.tif")

            elif args.model == "dclgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif")

            elif args.model == "uvcgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif")

            elif args.model == "cyclediffusion":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_tile(y, f"{args.outdir}/{stem}.tif")




if __name__ == "__main__":
    main()
