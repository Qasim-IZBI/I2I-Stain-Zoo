# reconstruct.py
import argparse
from utils import reconstruct_wsi


def main():
    parser = argparse.ArgumentParser("WSI Reconstruction from Tiles")

    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to a dataset directory containing per-WSI tiles_metadata.csv "
                             "files (e.g. path/to/tiles/trainA), or a single tiles_metadata.csv")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for reconstructed WSIs")
    parser.add_argument("--tile_dir", type=str, default=None,
                        help="Directory containing tile images (e.g. inference output). "
                             "If omitted, uses image_path from metadata CSV")
    parser.add_argument("--mode", choices=["rgb", "mask", "rgb_and_mask", "auto"],
                        default="rgb",
                        help="What to reconstruct (default: rgb)")
    parser.add_argument("--blend", choices=["average", "overwrite"],
                        default="average",
                        help="How to handle overlapping tiles (default: average)")

    args = parser.parse_args()

    reconstruct_wsi(
        metadata_csv=args.metadata,
        output_dir=args.output,
        tile_dir=args.tile_dir,
        mode=args.mode,
        blend=args.blend,
    )


if __name__ == "__main__":
    main()
