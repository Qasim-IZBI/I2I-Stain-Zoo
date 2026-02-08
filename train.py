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
from models.miudiff import MIUDiff, MIUDiffConfig


def build_model(args):
    if args.model == "cyclegan":
        cfg = CycleGANConfig()
        model = CycleGAN(cfg)

    elif args.model == "unit":
        cfg = UNITConfig()
        model = UNIT(cfg)

    elif args.model == "munit":
        cfg = MUNITConfig(style_dim=args.style_dim)
        model = MUNIT(cfg)

    elif args.model == "dclgan":
        cfg = DCLGANConfig(
            lambda_dcl=args.lambda_dcl,
            n_patches=args.n_patches,
            proj_dim=args.proj_dim,
        )
        model = DCLGAN(cfg)

    elif args.model == "miudiff":
        cfg = MIUDiffConfig(
            stage=args.miu_stage,
            sample_steps=args.miu_steps,
            guidance_scale=args.miu_guidance,
            miu_pcl=args.miu_pcl,
            lambda_pcl=args.lambda_pcl,
            pcl_n_patches=args.pcl_n_patches,
            pcl_proj_dim=args.pcl_proj_dim,
            pcl_temp=args.pcl_temp,
        )
        model = MIUDiff(cfg)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model



def main():
    parser = argparse.ArgumentParser("Unified I2I Training")

    # ---- model ----
    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan", "miudiff"], required=True)

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

    # ---- MUNIT specific ----
    parser.add_argument("--style_dim", type=int, default=8)


    # ---- DCLGAN specific ----
    parser.add_argument("--lambda_dcl", type=float, default=1.0)
    parser.add_argument("--n_patches", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)


    # ---- MIU-Diff PCL  specific ----
    parser.add_argument("--miu_stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--miu_steps", type=int, default=300)
    parser.add_argument("--miu_guidance", type=float, default=1.0)
    parser.add_argument("--miu_pcl", action="store_true")
    parser.add_argument("--lambda_pcl", type=float, default=0.1)
    parser.add_argument("--pcl_n_patches", type=int, default=256)
    parser.add_argument("--pcl_proj_dim", type=int, default=128)
    parser.add_argument("--pcl_temp", type=float, default=0.07)


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
        lr=args.lr,
        betas=(0.5, 0.999),
        use_amp=args.amp,
        sample_every=500,
        save_dir=args.output + '/checkpoints',
        sample_dir=args.output + '/samples'
    )

    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
