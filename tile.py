# tile.py
import argparse
from utils import create_tiles


def main():
    parser = argparse.ArgumentParser("WSI Tiling")

    parser.add_argument("--rgb", type=str, required=True,
                        help="Root folder containing image_type subfolder with RGB .tif files")
    parser.add_argument("--output", type=str, required=True,
                        help="Root output folder for tiles")
    parser.add_argument("--mask", type=str, default=None,
                        help="Root folder containing image_type subfolder with mask .tif files")
    parser.add_argument("--tile_size", type=int, default=256,
                        help="Size of square tiles extracted from the WSI")
    parser.add_argument("--resize_to", type=int, default=None,
                        help="Resize extracted tiles to this size before saving")
    parser.add_argument("--overlap", type=float, default=0,
                        help="Fractional overlap between tiles (0 to <1)")
    parser.add_argument("--tissue_threshold", type=float, default=0.5,
                        help="Minimum tissue fraction to keep a tile (for train sets)")
    parser.add_argument("--image_type", type=str, default="trainA",
                        help="Subfolder name, e.g. trainA, trainB, testA, testB")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: all CPUs)")

    args = parser.parse_args()

    create_tiles(
        rgb_folder_path=args.rgb,
        output_folder_dir=args.output,
        mask_folder_path=args.mask,
        tile_size=args.tile_size,
        resize_to=args.resize_to,
        overlap=args.overlap,
        tissue_threshold=args.tissue_threshold,
        image_type=args.image_type,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
