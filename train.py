# train.py

import os
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import ResnetGenerator, NLayerDiscriminator, GANLoss, init_net
from datasets import UnpairedImageDataset
from utils import ImagePool


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN training (PyTorch)")
    parser.add_argument("--dataroot", type=str, required=True, help="path to dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_identity", type=float, default=5.0)
    parser.add_argument("--load_size", type=int, default=286)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Using Device {device}')
    # Dataset
    dataset = UnpairedImageDataset(
        root=args.dataroot, phase="train", load_size=args.load_size, crop_size=args.crop_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Networks
    netG_A2B = init_net(ResnetGenerator(3, 3, n_blocks=9), device)
    netG_B2A = init_net(ResnetGenerator(3, 3, n_blocks=9), device)

    netD_A = init_net(NLayerDiscriminator(3, n_layers=3), device)
    netD_B = init_net(NLayerDiscriminator(3, n_layers=3), device)

    # Losses
    criterionGAN = GANLoss().to(device)
    criterionCycle = nn.L1Loss().to(device)
    criterionIdt = nn.L1Loss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(
        list(netG_A2B.parameters()) + list(netG_B2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_D = optim.Adam(
        list(netD_A.parameters()) + list(netD_B.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    # Image pools
    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    step = 0
    for epoch in range(1, args.epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{args.epochs}]", ncols=100)
        for batch in loop:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            ########################
            # 1. Update Generators #
            ########################
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) ≈ B, G_B2A(A) ≈ A
            idt_B = netG_A2B(real_B)
            loss_idt_B = criterionIdt(idt_B, real_B) * args.lambda_cycle * args.lambda_identity

            idt_A = netG_B2A(real_A)
            loss_idt_A = criterionIdt(idt_A, real_A) * args.lambda_cycle * args.lambda_identity

            loss_idt = (loss_idt_A + loss_idt_B) * 0.5

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_G_A2B = criterionGAN(pred_fake_B, True)

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_G_B2A = criterionGAN(pred_fake_A, True)

            # Cycle loss
            rec_A = netG_B2A(fake_B)
            loss_cycle_A = criterionCycle(rec_A, real_A) * args.lambda_cycle

            rec_B = netG_A2B(fake_A)
            loss_cycle_B = criterionCycle(rec_B, real_B) * args.lambda_cycle

            loss_cycle = (loss_cycle_A + loss_cycle_B) * 0.5

            # Total G loss
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle + loss_idt
            loss_G.backward()
            optimizer_G.step()

            ########################
            # 2. Update Discriminators #
            ########################
            optimizer_D.zero_grad()

            # D_A
            # Real
            pred_real_A = netD_A(real_A)
            loss_D_A_real = criterionGAN(pred_real_A, True)
            # Fake
            fake_A_detached = fake_A_pool.query(fake_A)
            pred_fake_A = netD_A(fake_A_detached.detach())
            loss_D_A_fake = criterionGAN(pred_fake_A, False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()

            # D_B
            pred_real_B = netD_B(real_B)
            loss_D_B_real = criterionGAN(pred_real_B, True)
            fake_B_detached = fake_B_pool.query(fake_B)
            pred_fake_B = netD_B(fake_B_detached.detach())
            loss_D_B_fake = criterionGAN(pred_fake_B, False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()

            optimizer_D.step()

            step += 1

            loop.set_postfix(
                G=loss_G.item(),
                D_A=loss_D_A.item(),
                D_B=loss_D_B.item(),
            )

            # Save some samples occasionally
            if step % 500 == 0:
                with torch.no_grad():
                    # denormalize from [-1,1] to [0,1]
                    def denorm(x):
                        return (x + 1.0) / 2.0

                    sample_A = real_A[:4]
                    sample_B = real_B[:4]
                    fake_B_vis = netG_A2B(sample_A)
                    fake_A_vis = netG_B2A(sample_B)

                    grid = torch.cat(
                        [denorm(sample_A), denorm(fake_B_vis), denorm(sample_B), denorm(fake_A_vis)],
                        dim=0,
                    )
                    save_path = os.path.join(args.sample_dir, f"epoch{epoch}_step{step}.png")
                    save_image(grid, save_path, nrow=4)

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "netG_A2B": netG_A2B.state_dict(),
                "netG_B2A": netG_B2A.state_dict(),
                "netD_A": netD_A.state_dict(),
                "netD_B": netD_B.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
