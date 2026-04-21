# train.py
import argparse
import torch
from torch.utils.data import DataLoader

from utils import get_device

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
    device = get_device()

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
            num_res_blocks=args.miu_num_res_blocks,
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
            ngf=args.uvcgan_ngf,
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
        return CycleGANConfig(
            ngf=args.cyclegan_ngf,
            n_blocks=args.cyclegan_n_blocks,
        )
    elif args.model == "unit":
        n_shared = args.unit_n_blocks_shared
        n_side = (args.unit_n_blocks - n_shared) // 2
        if n_side * 2 + n_shared != args.unit_n_blocks:
            raise ValueError(
                f"--unit_n_blocks ({args.unit_n_blocks}) - --unit_n_blocks_shared "
                f"({n_shared}) must be even so pre/post halves are equal."
            )
        n_down = 2  # fixed for UNIT
        z_dim = args.unit_ngf * (2 ** n_down)
        return UNITConfig(
            ngf=args.unit_ngf,
            z_dim=z_dim,
            n_blocks_total=args.unit_n_blocks,
            n_blocks_shared=n_shared,
            n_blocks_private_pre=n_side,
            n_blocks_private_post=n_side,
        )
    elif args.model == "munit":
        return MUNITConfig(
            ngf=args.munit_ngf,
            style_dim=args.style_dim,
            n_content_blocks=args.munit_n_content_blocks,
            n_adain_blocks=args.munit_n_adain_blocks,
        )
    elif args.model == "dclgan":
        return DCLGANConfig(
            ngf=args.dclgan_ngf,
            n_blocks=args.dclgan_n_blocks,
            lambda_dcl=args.lambda_dcl,
            n_patches=args.n_patches,
            proj_dim=args.proj_dim,
        )
    raise ValueError(f"No default config for {args.model}")



def _count_a2b_params(model, model_name, miu_stage="finetune"):
    """Return (total_params, a2b_params) for the generator of the given model."""
    def n(m):
        return sum(p.numel() for p in m.parameters())

    total = n(model)

    if model_name == "cyclegan":
        a2b = n(model.Enc_A) + n(model.Bn_A) + n(model.Dec_B)
    elif model_name == "unit":
        a2b = (n(model.E_A) + n(model.latent_A) + n(model.bn_pre_A)
               + n(model.bn_shared) + n(model.bn_post_B) + n(model.Dec_B))
    elif model_name == "munit":
        # random-style A→B: content encoder A + content bottleneck A + AdaIN decoder B
        a2b = n(model.Ec_A) + n(model.Bn_A) + n(model.AdaIN_B) + n(model.Dec_B)
    elif model_name == "dclgan":
        a2b = n(model.Enc_A) + n(model.Bn_A) + n(model.Dec_B)
    elif model_name == "uvcgan":
        a2b = n(model.G_A2B)
    elif model_name == "miudiff":
        if miu_stage == "pretrain":
            a2b = n(model.eps_uncond)
        else:
            # Both UNets active at inference (classifier-free guidance)
            a2b = n(model.eps_uncond) + n(model.eps_cond)
    else:
        a2b = total

    return total, a2b


def _print_param_counts(args, device):
    model = build_model(args).to(device)
    miu_stage = getattr(args, "miu_stage", "finetune")
    total, a2b = _count_a2b_params(model, args.model, miu_stage)

    print(f"\n{'=' * 52}")
    print(f"  Model        : {args.model}")
    print(f"  Total params : {total:>12,}")
    print(f"  A→B params   : {a2b:>12,}  ({a2b/1e6:.2f}M)")
    print(f"{'=' * 52}\n")


def main():
    parser = argparse.ArgumentParser("Unified I2I Training")

    # ---- model ----
    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "miudiff", "uvcgan"], required=True)
    parser.add_argument("--count_params", action="store_true",
                        help="Print A→B generator parameter count and exit (no training)")

    # ---- data ----
    parser.add_argument("--dataA", type=str, default=None,
                        help="Path to domain A tiles (required unless --count_params)")
    parser.add_argument("--dataB", type=str, default=None,
                        help="Path to domain B tiles (required unless --count_params)")
    parser.add_argument("--data_range", type=str, default=None,
                        help="Load tiles from a range of numbered folders, e.g. '1,6' loads "
                             "001/images/ through 006/images/ under dataA and dataB")

    # ---- training ----
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Total number of optimiser steps to train for")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for checkpoints (required unless --count_params)")
    parser.add_argument("--save_steps", type=int, default=250_000,
                        help="Save a checkpoint every this many steps")
    parser.add_argument("--log_steps", type=int, default=1_000,
                        help="Log losses every this many steps")
    parser.add_argument("--init_ckpt", type=str, default=None,
                    help="Pretrained checkpoint to initialise model weights")

    # ---- CycleGAN architecture ----
    parser.add_argument("--cyclegan_ngf", type=int, default=64,
                        help="Generator base channels for CycleGAN")
    parser.add_argument("--cyclegan_n_blocks", type=int, default=9,
                        help="ResNet blocks in CycleGAN bottleneck")

    # ---- UNIT architecture ----
    parser.add_argument("--unit_ngf", type=int, default=64,
                        help="Generator base channels for UNIT")
    parser.add_argument("--unit_n_blocks", type=int, default=9,
                        help="Total ResNet blocks in UNIT bottleneck (pre + shared + post)")
    parser.add_argument("--unit_n_blocks_shared", type=int, default=3,
                        help="Shared blocks in UNIT bottleneck; pre=post=(total-shared)//2")

    # ---- MUNIT architecture ----
    parser.add_argument("--munit_ngf", type=int, default=64,
                        help="Generator base channels for MUNIT")
    parser.add_argument("--munit_n_content_blocks", type=int, default=4,
                        help="Content ResNet blocks for MUNIT")
    parser.add_argument("--munit_n_adain_blocks", type=int, default=4,
                        help="AdaIN ResNet blocks in MUNIT decoder")

    # ---- MUNIT style ----
    parser.add_argument("--style_dim", type=int, default=8)

    # ---- DCLGAN architecture ----
    parser.add_argument("--dclgan_ngf", type=int, default=64,
                        help="Generator base channels for DCLGAN")
    parser.add_argument("--dclgan_n_blocks", type=int, default=9,
                        help="ResNet blocks in DCLGAN bottleneck")

    # ---- DCLGAN contrastive ----
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
    parser.add_argument("--miu_num_res_blocks", type=int, default=2,
                    help="ResBlocks per level in MIUDiff UNet (default 2; use 1 for simpler/faster variant)")
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
    parser.add_argument("--uvcgan_ngf", type=int, default=64,
                    help="Generator base channels for UVCGAN UNet encoder/decoder")
    parser.add_argument("--uvcgan_vit_blocks", type=int, default=6)
    parser.add_argument("--uvcgan_vit_features", type=int, default=192,
                    help="ViT hidden dimension for UVCGAN bottleneck")


    args = parser.parse_args()

    device = get_device()

    # ---- parameter count dry-run ----
    if args.count_params:
        _print_param_counts(args, device)
        return

    # ---- dataset ----
    transform = default_train_transform(image_size=256)

    data_range = None
    if args.data_range:
        parts = args.data_range.split(",")
        data_range = (int(parts[0]), int(parts[1]))

    if args.model == "miudiff" and args.miu_stage == "pretrain":
        dataset = TargetOnlyDataset(root_B=args.dataB, transform=transform, data_range=data_range)
    else:
        dataset = UnpairedDataset(root_A=args.dataA, root_B=args.dataB, transform=transform, data_range=data_range)



    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- model ----
    model = build_model(args).to(device)

    # MIUDiff runs its UNet inside autocast(enabled=False) — entirely fp32.
    # Enabling GradScaler on fp32 computations causes the scale to double every
    # growth_interval steps until it overflows fp32 (~2^128 at step ~55k with the
    # default growth_interval=500), permanently locking the scale at +inf.
    use_amp = args.amp and args.model != "miudiff"
    if args.amp and args.model == "miudiff":
        print("[MIUDiff] AMP disabled: DDPMUNet runs in fp32 internally; GradScaler would overflow at ~56k steps.")

    # ---- trainer ----
    trainer = BaseTrainer(
        model=model,
        dataloader=loader,
        device=device,
        model_name=args.model,
        lr=args.lr,
        betas=(0.5, 0.999),
        use_amp=use_amp,
        save_dir=args.output + '/checkpoints',
        sample_dir=args.output + '/samples',
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )

    trainer.train(total_steps=args.steps)


if __name__ == "__main__":
    main()
