# train.py
import argparse
import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from datasets.unpaired_dataset import UnpairedDataset

# Models
from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig


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

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model



def main():
    parser = argparse.ArgumentParser("Unified I2I Training")

    # ---- model ----
    parser.add_argument("--model", choices=["cyclegan", "unit", "munit", "dclgan"], required=True)

    # ---- data ----
    parser.add_argument("--dataA", type=str, required=True)
    parser.add_argument("--dataB", type=str, required=True)

    # ---- training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")

    # ---- MUNIT specific ----
    parser.add_argument("--style_dim", type=int, default=8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- DCLGAN specific ----
    parser.add_argument("--lambda_dcl", type=float, default=1.0)
    parser.add_argument("--n_patches", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=256)

    # ---- dataset ----
    dataset = UnpairedDataset(
        root_A=args.dataA,
        root_B=args.dataB,
        transform=None,  # plug your transforms here
    )

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
    )

    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
