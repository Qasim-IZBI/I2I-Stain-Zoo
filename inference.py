# inference.py
import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from models import ResnetGenerator  # from your previous code


def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


def load_image(path, tfm):
    img = Image.open(path).convert("RGB")
    return tfm(img)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="folder of images to translate")
    ap.add_argument("--output_dir", required=True, help="where to save translated images")
    ap.add_argument("--checkpoint", required=True, help="path to epoch_*.pt")
    ap.add_argument("--direction", choices=["A2B", "B2A"], default="A2B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--crop_size", type=int, default=256, help="must match training crop_size")
    ap.add_argument("--resize_to", type=int, default=None, help="optional: resize shortest side to this")
    ap.add_argument("--keep_size", action="store_true", help="skip center-crop; keep original size (slower)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalization must match training: [-1, 1]
    base = []
    if args.resize_to is not None:
        base.append(transforms.Resize((args.resize_to, args.resize_to), Image.BICUBIC))

    if args.keep_size:
        # keep original size (no crop) – good for WSI-ish tiles if already correct size
        base += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    else:
        base += [
            transforms.Resize((args.crop_size, args.crop_size), Image.BICUBIC),
            transforms.CenterCrop((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    tfm = transforms.Compose(base)

    # Build generators and load checkpoint
    netG_A2B = ResnetGenerator(3, 3, n_blocks=9).to(device).eval()
    netG_B2A = ResnetGenerator(3, 3, n_blocks=9).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    netG_A2B.load_state_dict(ckpt["netG_A2B"])
    netG_B2A.load_state_dict(ckpt["netG_B2A"])

    netG = netG_A2B if args.direction == "A2B" else netG_B2A

    paths = list_images(args.input_dir)
    if not paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    def denorm(x):  # [-1,1] -> [0,1]
        return (x + 1.0) / 2.0

    for p in tqdm(paths):
        x = load_image(p, tfm).unsqueeze(0).to(device)
        y = netG(x)
        y_vis = denorm(y).clamp(0, 1)

        out_name = os.path.splitext(os.path.basename(p))[0] + f"_{args.direction}.png"
        out_path = os.path.join(args.output_dir, out_name)
        save_image(y_vis, out_path)

    print(f"Done. Saved {len(paths)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
