# train.py
import argparse
import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from datasets.unpaired_dataset import UnpairedDataset
from datasets.target_only_dataset import TargetOnlyDataset
from datasets.transforms import default_train_transform


# Models
from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig
from models.miudiff import MIUDiff, MIUDiffConfig, init_miudiff_from_stage1
from models.uvcgan import UVCGAN, UVCGANConfig


_CONFIG_CLS = {
    "cyclegan": CycleGANConfig,
    "unit": UNITConfig,
    "munit": MUNITConfig,
    "dclgan": DCLGANConfig,
}
_MODEL_CLS = {
    "cyclegan": CycleGAN,
    "unit": UNIT,
    "munit": MUNIT,
    "dclgan": DCLGAN,
}


def build_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- GAN models with --init_ckpt: restore config from checkpoint ---
    if args.init_ckpt and args.model in _CONFIG_CLS:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        saved_cfg = ckpt.get("config")
        if saved_cfg:
            cfg = _CONFIG_CLS[args.model](**saved_cfg)
            print(f"Restored {args.model} config from checkpoint")
        else:
            # Old checkpoint without config — fall back to CLI args / defaults
            cfg = _build_default_gan_config(args)
        model = _MODEL_CLS[args.model](cfg)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained checkpoint from {args.init_ckpt}")
        if missing:
            print(f"  {len(missing)} missing key(s)")
        if unexpected:
            print(f"  {len(unexpected)} unexpected key(s)")
        return model

    # --- GAN models without --init_ckpt ---
    if args.model in _CONFIG_CLS:
        cfg = _build_default_gan_config(args)
        return _MODEL_CLS[args.model](cfg)

    # --- MIUDiff ---
    if args.model == "miudiff":
        cfg = MIUDiffConfig(
            stage=args.miu_stage,
            sample_steps=args.miu_steps,
            guidance_scale=args.miu_guidance,
            base_channels=args.miu_base_channels,
            channel_mult=tuple(int(x) for x in args.miu_channel_mult.split(",")),
            miu_pcl=args.miu_pcl,
            lambda_pcl=args.lambda_pcl,
            pcl_n_patches=args.pcl_n_patches,
            pcl_proj_dim=args.pcl_proj_dim,
            pcl_temp=args.pcl_temp,
            lambda_mi=args.lambda_mi,
        )
        model = MIUDiff(cfg)
        init_ckpt = args.miu_init_ckpt or args.init_ckpt
        if args.miu_stage == "finetune" and init_ckpt:
            if args.miu_pcl:
                # Stage 2→3: load checkpoint directly (eps_cond already trained).
                # init_miudiff_from_stage1 would overwrite trained eps_cond with eps_uncond.
                ckpt = torch.load(init_ckpt, map_location=device)
                sd = ckpt["model"] if "model" in ckpt else ckpt
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(f"Loaded stage-2 checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
            else:
                # Stage 1→2: copy eps_uncond weights into eps_cond
                init_miudiff_from_stage1(model, init_ckpt, device)
        return model

    # --- UVCGAN ---
    if args.model == "uvcgan":
        cfg = UVCGANConfig(
            pretrain=(args.uvcgan_stage == "pretrain"),
            vit_n_blocks=args.uvcgan_vit_blocks,
            vit_features=args.uvcgan_vit_features,
        )
        model = UVCGAN(cfg)
        uvcgan_ckpt = args.uvcgan_init_ckpt or args.init_ckpt
        if args.uvcgan_stage == "finetune" and uvcgan_ckpt:
            ckpt = torch.load(uvcgan_ckpt, map_location=device)
            sd = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(sd, strict=True)
            print(f"Loaded UVCGAN pretrain checkpoint from {uvcgan_ckpt}")
        return model

    raise ValueError(f"Unknown model: {args.model}")


def _build_default_gan_config(args):
    """Build config from CLI args for the 4 GAN models (no checkpoint)."""
    if args.model == "cyclegan":
        return CycleGANConfig()
    elif args.model == "unit":
        return UNITConfig()
    elif args.model == "munit":
        return MUNITConfig(style_dim=args.style_dim)
    elif args.model == "dclgan":
        return DCLGANConfig(
            lambda_dcl=args.lambda_dcl,
            n_patches=args.n_patches,
            proj_dim=args.proj_dim,
        )
    raise ValueError(f"No default config for {args.model}")



def main():
    parser = argparse.ArgumentParser("Unified I2I Training")

    # ---- model ----
    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"], required=True)

    # ---- data ----
    parser.add_argument("--dataA", type=str, required=True)
    parser.add_argument("--dataB", type=str, required=True)

    # ---- training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_epochs", type=int, default=5)
    parser.add_argument("--init_ckpt", type=str, default=None,
                    help="Pretrained checkpoint to initialise model weights")

    # ---- MUNIT specific ----
    parser.add_argument("--style_dim", type=int, default=8)


    # ---- DCLGAN specific ----
    parser.add_argument("--lambda_dcl", type=float, default=1.0)
    parser.add_argument("--n_patches", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)


    # ---- MIU-Diff specific ----
    parser.add_argument("--miu_stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--miu_steps", type=int, default=300)
    parser.add_argument("--miu_guidance", type=float, default=1.0)
    parser.add_argument("--miu_base_channels", type=int, default=64,
                    help="Base channel width for MIUDiff UNet")
    parser.add_argument("--miu_channel_mult", type=str, default="1,2,2,4",
                    help="Comma-separated channel multipliers for MIUDiff UNet")
    parser.add_argument("--miu_pcl", action="store_true")
    parser.add_argument("--lambda_mi", type=float, default=1.0)
    parser.add_argument("--lambda_pcl", type=float, default=0.1)
    parser.add_argument("--pcl_n_patches", type=int, default=256)
    parser.add_argument("--pcl_proj_dim", type=int, default=128)
    parser.add_argument("--pcl_temp", type=float, default=0.07)
    parser.add_argument("--miu_init_ckpt", type=str, default=None,
                    help="Checkpoint from MIU-Diff stage-1 to initialize stage-2")

    # ---- UVCGAN specific ----
    parser.add_argument("--uvcgan_stage", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--uvcgan_init_ckpt", type=str, default=None,
                    help="Checkpoint from UVCGAN pretrain stage")
    parser.add_argument("--uvcgan_vit_blocks", type=int, default=6)
    parser.add_argument("--uvcgan_vit_features", type=int, default=192,
                    help="ViT hidden dimension for UVCGAN bottleneck")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset ----
    transform = default_train_transform(image_size=256)

    if args.model == "miudiff" and args.miu_stage == "pretrain":
        dataset = TargetOnlyDataset(root_B=args.dataB, transform=transform)
    else:
        dataset = UnpairedDataset(root_A=args.dataA, root_B=args.dataB, transform=transform)



    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- model ----
    model = build_model(args).to(device)

    # ---- trainer ----
    trainer = BaseTrainer(
        model=model,
        dataloader=loader,
        device=device,
        model_name=args.model,
        lr=args.lr,
        betas=(0.5, 0.999),
        use_amp=args.amp,
        save_dir=args.output + '/checkpoints',
        sample_dir=args.output + '/samples',
        save_epochs=args.save_epochs
    )

    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
