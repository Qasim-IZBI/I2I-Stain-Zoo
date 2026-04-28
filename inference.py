# infer.py
import os
import argparse
import torch
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
    # pretrain checkpoints may have only eps_uncond keys (trimmed); strict=False tolerates that
    strict = not (args.model == "miudiff" and args.miu_stage == "pretrain")
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

    # MUNIT
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--style_image", type=str, default=None,
                        help="Reference image to extract style from (MUNIT only)")

    # MIU-Diff
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

    args = parser.parse_args()

    device = get_device()
    model = load_model(args, device)

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
            save_image((y + 1) / 2, f"{args.outdir}/uncond_{i:04d}.tif")
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
                save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")

            elif args.model == "unit":
                if args.direction == "A2B":
                    y, _ = model.forward_A2B(x)
                else:
                    y, _ = model.forward_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")

            elif args.model == "munit":
                if args.direction == "A2B":
                    c, _ = model.encode_A(x)
                    if args.style_image:
                        from PIL import Image
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_B(ref)
                        y = model.decode_B(c, s)
                        save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_B(c, s)
                            save_image((y + 1) / 2, f"{args.outdir}/{stem}_{k}.tif")
                else:
                    c, _ = model.encode_B(x)
                    if args.style_image:
                        from PIL import Image
                        ref = transform(Image.open(args.style_image).convert("RGB"))
                        ref = ref.unsqueeze(0).to(device)
                        _, s = model.encode_A(ref)
                        y = model.decode_A(c, s)
                        save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")
                    else:
                        for k in range(args.num_samples):
                            s = torch.randn(1, model.cfg.style_dim, device=device)
                            y = model.decode_A(c, s)
                            save_image((y + 1) / 2, f"{args.outdir}/{stem}_{k}.tif")

            elif args.model == "dclgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")

            elif args.model == "miudiff":
                if args.miu_stage == "pretrain":
                    if args.miu_guidance != 1.0:
                        print("[WARN] --miu_guidance is ignored for --miu_stage pretrain.")
                    y = model.sample_uncond(batch_size=1)
                    save_image((y + 1) / 2, f"{args.outdir}/{stem}_uncond.tif")
                else:
                    y = model.sample_A2B(x)
                    save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")

            elif args.model == "uvcgan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")




if __name__ == "__main__":
    main()
