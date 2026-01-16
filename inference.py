# infer.py
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.single_domain_dataset import SingleDomainDataset

from models.cyclegan import CycleGAN, CycleGANConfig
from models.unit import UNIT, UNITConfig
from models.munit import MUNIT, MUNITConfig
from models.dclgan import DCLGAN, DCLGANConfig


def load_model(args, device):
    if args.model == "cyclegan":
        model = CycleGAN(CycleGANConfig())

    elif args.model == "unit":
        model = UNIT(UNITConfig())

    elif args.model == "munit":
        model = MUNIT(MUNITConfig(style_dim=args.style_dim))

    elif args.model == "dclgan":
        model = DCLGAN(DCLGANConfig())

    else:
        raise ValueError(args.model)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser("Unified I2I Inference")

    parser.add_argument("--model", choices=["cyclegan", "unit", "munit"], required=True)
    parser.add_argument("--direction", choices=["A2B", "B2A"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")

    # MUNIT
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    dataset = SingleDomainDataset(args.data, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (x,path) in enumerate(loader):
            x = x.to(device)

            if args.model == "cyclegan":
                if args.direction == "A2B":
                    y = model.forward_A2B(x)
                else:
                    y = model.forward_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{i}.png")

            elif args.model == "unit":
                if args.direction == "A2B":
                    y, _ = model.forward_A2B(x)
                else:
                    y, _ = model.forward_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{i}.png")

            elif args.model == "munit":
                if args.direction == "A2B":
                    c, _ = model.encode_A(x)
                    for k in range(args.num_samples):
                        s = torch.randn(1, args.style_dim, device=device)
                        y = model.decode_B(c, s)
                        save_image((y + 1) / 2, f"{args.outdir}/{i}_{k}.png")
                else:
                    c, _ = model.encode_B(x)
                    for k in range(args.num_samples):
                        s = torch.randn(1, args.style_dim, device=device)
                        y = model.decode_A(c, s)
                        save_image((y + 1) / 2, f"{args.outdir}/{i}_{k}.png")
            elif args.model == "dclgan":
                if args.direction == "A2B":
                    y, _ = model.G_A2B(x)
                else:
                    y, _ = model.G_B2A(x)
                save_image((y + 1) / 2, f"{args.outdir}/{i}.png")



if __name__ == "__main__":
    main()
