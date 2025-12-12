import os
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from tqdm.auto import tqdm


def _process_single_image(
    file_name,
    rgb_folder,
    output_images_dir,
    output_masks_dir,
    mask_folder,
    tile_size,
    stride,
    tissue_threshold,
    image_type,
):
    """
    Process a single RGB (and optional mask) image:
    - Extract tiles
    - Optionally filter by tissue threshold (for train-like sets)
    - Save RGB tiles and mask tiles (if mask available)
    - Collect metadata for each tile
    """
    rgb_path = rgb_folder / file_name
    rgb_img = tifffile.imread(rgb_path)

    mask_img = None
    has_mask = mask_folder is not None
    if has_mask:
        mask_path = mask_folder / file_name
        if mask_path.exists():
            mask_img = tifffile.imread(mask_path)
            assert rgb_img.shape[:2] == mask_img.shape[:2], \
                f"Image and mask dimensions must match for {file_name}"
        else:
            # Mask folder provided but mask file missing for this image
            has_mask = False

    height, width = rgb_img.shape[:2]
    tile_id = 0
    metadata_rows = []

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            rgb_tile = rgb_img[y:y + tile_size, x:x + tile_size]

            tissue_fraction = None
            save_tile = False

            if image_type in {"testA", "testB"}:
                # Test images: always save tiles (if you want filtering here, change this logic)
                save_tile = True
            else:
                # Train images: apply tissue mask filtering if mask is available
                if has_mask:
                    mask_tile = mask_img[y:y + tile_size, x:x + tile_size]
                    tissue_fraction = float(np.mean(mask_tile > 0))
                    if tissue_fraction >= tissue_threshold:
                        save_tile = True
                else:
                    # Original behavior: if no mask for train images, nothing is saved.
                    save_tile = False

            if not save_tile:
                continue

            tile_name_base = f"{file_name}_tile_{tile_id:07d}"
            img_tile_path = output_images_dir / f"{tile_name_base}.tif"
            Image.fromarray(rgb_tile).save(img_tile_path)

            mask_tile_path = None
            if has_mask:
                mask_tile = mask_img[y:y + tile_size, x:x + tile_size]
                mask_tile_path = output_masks_dir / f"{tile_name_base}.tif"
                Image.fromarray(mask_tile).save(mask_tile_path)

            metadata_rows.append(
                {
                    "source_file": file_name,
                    "tile_id": tile_id,
                    "tile_name": tile_name_base,
                    "image_path": str(img_tile_path),
                    "mask_path": str(mask_tile_path) if mask_tile_path else "",
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "tissue_fraction": tissue_fraction
                    if tissue_fraction is not None
                    else "",
                    "image_type": image_type,
                }
            )

            tile_id += 1

    return tile_id, metadata_rows


def create_tiles(
    rgb_folder_path,
    output_folder_dir,
    mask_folder_path=None,
    tile_size=256,
    overlap=0,
    tissue_threshold=0.5,
    image_type="trainA",
    num_workers=None,
    metadata_csv_name="tiles_metadata.csv",
):
    """
    Create tiles from RGB images (and optional masks).

    Parameters
    ----------
    rgb_folder_path : str or Path
        Root folder containing image_type subfolder with RGB .tif files.
    output_folder_dir : str or Path
        Root output folder. Tiles will be saved in:
            <output_folder_dir>/<image_type>/images/
            <output_folder_dir>/<image_type>/masks/
    mask_folder_path : str or Path, optional
        Root folder containing image_type subfolder with mask .tif files.
        If provided, paired mask tiles will be saved.
    tile_size : int, default 256
        Size of square tiles.
    overlap : float, default 0
        Fractional overlap between tiles (0 to <1). E.g., 0.5 gives 50% overlap.
    tissue_threshold : float, default 0.5
        Minimum fraction of tissue (mask > 0) required to keep a tile
        for non-test image types.
    image_type : str, default "trainA"
        Subfolder name under both rgb_folder_path and mask_folder_path.
        Example: "trainA", "trainB", "testA", "testB".
    num_workers : int, optional
        Number of worker processes. If None, use os.cpu_count().
    metadata_csv_name : str, default "tiles_metadata.csv"
        Name of the CSV metadata file saved in the image_type output folder.

    Returns
    -------
    None
    """
    rgb_folder = Path(rgb_folder_path) / image_type
    output_root = Path(output_folder_dir) / image_type
    output_images_dir = output_root / "images"
    output_masks_dir = output_root / "masks"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    mask_folder = Path(mask_folder_path) / image_type if mask_folder_path else None

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError(
            f"Invalid overlap {overlap}. It must be < 1. "
            f"Got stride={stride} for tile_size={tile_size}."
        )

    # Collect list of .tif files
    file_names = [
        f for f in os.listdir(rgb_folder)
        if f.lower().endswith(".tif")
    ]

    if not file_names:
        print(f"No .tif files found in {rgb_folder}")
        return

    # Prepare multiprocessing
    total_tiles = 0
    all_metadata = []

    num_workers = num_workers or os.cpu_count() or 1

    print(
        f"Processing {len(file_names)} images from {rgb_folder} "
        f"with {num_workers} worker(s)..."
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_image,
                file_name,
                rgb_folder,
                output_images_dir,
                output_masks_dir,
                mask_folder,
                tile_size,
                stride,
                tissue_threshold,
                image_type,
            ): file_name
            for file_name in file_names
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tiling"):
            file_name = futures[future]
            try:
                tile_count, metadata_rows = future.result()
                total_tiles += tile_count
                all_metadata.extend(metadata_rows)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Save metadata CSV
    if all_metadata:
        metadata_path = output_root / metadata_csv_name
        fieldnames = [
            "source_file",
            "tile_id",
            "tile_name",
            "image_path",
            "mask_path",
            "x",
            "y",
            "tile_size",
            "tissue_fraction",
            "image_type",
        ]
        with metadata_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_metadata:
                writer.writerow(row)

        print(f"Saved metadata for {len(all_metadata)} tiles to {metadata_path}")

    print(f"Total tiles saved: {total_tiles} (images in {output_images_dir}, masks in {output_masks_dir})")
    return



def reconstruct_wsi(
    metadata_csv,
    output_dir,
    mode="rgb_and_mask",
    blend="average",   # "average" or "overwrite"
):
    """
    Reconstruct full WSI(s) from tiles using the metadata CSV.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the tiles_metadata.csv generated during tiling.
    output_dir : str or Path
        Where reconstructed WSIs will be saved.
    mode : str
        "rgb" -> reconstruct only RGB images
        "mask" -> reconstruct only mask images
        "rgb_and_mask" -> reconstruct both
        "auto" -> reconstruct mask only if mask paths exist
    blend : str
        "average" -> average overlapping regions (recommended)
        "overwrite" -> last tile wins

    Returns
    -------
    dict:
        Dictionary mapping source_file -> dict with reconstructed paths.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)

    # Detect whether masks exist
    has_masks = df["mask_path"].notna().any() and (df["mask_path"] != "").any()
    if mode == "auto":
        mode = "rgb_and_mask" if has_masks else "rgb"

    results = {}

    # Group tiles by WSI source file
    for source_file, group in df.groupby("source_file"):

        print(f"\nReconstructing WSI: {source_file}")
        wsi_name = Path(source_file).stem  # removes .tif extension


        # Determine output WSI size
        max_x = (group["x"] + group["tile_size"]).max()
        max_y = (group["y"] + group["tile_size"]).max()

        # Prepare output arrays
        rgb_canvas = np.zeros((max_y, max_x, 3), dtype=np.float32)
        rgb_weight = np.zeros((max_y, max_x, 1), dtype=np.float32)

        mask_canvas = None
        mask_weight = None

        if mode in {"mask", "rgb_and_mask"} and has_masks:
            mask_canvas = np.zeros((max_y, max_x), dtype=np.float32)
            mask_weight = np.zeros((max_y, max_x), dtype=np.float32)

        # ---- Insert tiles ----
        for _, row in tqdm(group.iterrows(), total=len(group), desc="Placing tiles"):
            x, y = int(row["x"]), int(row["y"])
            tile_size = int(row["tile_size"])

            # Load RGB tile
            if mode in {"rgb", "rgb_and_mask"}:
                rgb_tile = np.array(Image.open(row["image_path"]))
                rgb_canvas[y:y+tile_size, x:x+tile_size] += rgb_tile
                rgb_weight[y:y+tile_size, x:x+tile_size] += 1

            # Load mask tile if available
            if (
                mode in {"mask", "rgb_and_mask"} 
                and has_masks 
                and isinstance(row["mask_path"], str)
                and row["mask_path"] != ""
            ):
                mask_tile = np.array(Image.open(row["mask_path"]))
                mask_canvas[y:y+tile_size, x:x+tile_size] += mask_tile
                mask_weight[y:y+tile_size, x:x+tile_size] += 1

        # ---- Blend overlapping areas ----
        def finalize(canvas, weight):
            if blend == "average":
                canvas = np.divide(
                    canvas, weight, 
                    out=np.zeros_like(canvas),
                    where=weight > 0
                )
            elif blend == "overwrite":
                pass
            return canvas

        rgb_output = None
        mask_output = None

        if mode in {"rgb", "rgb_and_mask"}:
            rgb_reconstructed = finalize(rgb_canvas, rgb_weight)
            rgb_output = output_dir / f"{wsi_name}_reconstructed_rgb.tif"
            Image.fromarray(rgb_reconstructed.astype(np.uint8)).save(rgb_output)

        if mode in {"mask", "rgb_and_mask"} and has_masks:
            mask_reconstructed = finalize(mask_canvas, mask_weight)
            mask_output = output_dir / f"{wsi_name}_reconstructed_mask.tif"
            Image.fromarray(mask_reconstructed.astype(np.uint8)).save(mask_output)

        results[source_file] = {
            "rgb": str(rgb_output) if rgb_output else None,
            "mask": str(mask_output) if mask_output else None,
        }

    print("\nReconstruction complete!")
    return results
