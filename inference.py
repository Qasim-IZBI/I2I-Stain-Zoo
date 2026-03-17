# infer.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.single_domain_dataset import SingleDomainDataset
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
        if saved_cfg:
            # Override runtime-only params from CLI args
            saved_cfg["stage"] = "finetune"
            saved_cfg["sample_steps"] = args.miu_steps
            saved_cfg["guidance_scale"] = args.miu_guidance
            saved_cfg["miu_pcl"] = args.miu_pcl
            saved_cfg["pcl_refine_steps"] = args.pcl_refine_steps
            saved_cfg["pcl_refine_lr"] = args.pcl_refine_lr
            cfg = MIUDiffConfig(**saved_cfg)
        else:
            cfg = MIUDiffConfig(
                stage="finetune",
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
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser("Unified I2I Inference")

    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"], required=True)
    parser.add_argument("--direction", choices=["A2B", "B2A"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")

    # MUNIT
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--style_image", type=str, default=None,
                        help="Reference image to extract style from (MUNIT only)")

    # MIU-Diff
    parser.add_argument("--miu_steps", type=int, default=300)
    parser.add_argument("--miu_guidance", type=float, default=1.0)
    parser.add_argument("--miu_pcl", action="store_true")
    parser.add_argument("--pcl_refine_steps", type=int, default=0)
    parser.add_argument("--pcl_refine_lr", type=float, default=0.05)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    transform = default_train_transform(image_size=256)

    dataset = SingleDomainDataset(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(args.outdir, exist_ok=True)

    with torch.no_grad():
        for i, (x, path) in enumerate(loader):
            x = x.to(device)
            stem = os.path.splitext(os.path.basename(path[0]))[0]

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
                    y, _ = model.G_A2B(x)
                else:
                    y, _ = model.G_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{stem}.tif")

            elif args.model == "miudiff":
                # only meaningful direction is A2B (H&E -> IHC)
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
