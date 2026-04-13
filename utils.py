import os
import re
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from tqdm.auto import tqdm


def get_device():
    """Return CUDA device if available, otherwise CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_serializable(obj):
    """Recursively convert tuples/non-JSON types so json.dump can handle them."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def _process_single_image(
    file_name,
    img_idx,
    rgb_folder,
    img_folder,
    mask_folder,
    tile_size,
    stride,
    overlap,
    tissue_threshold,
    image_type,
    resize_to=None,
):
    """
    Process a single RGB (and optional mask) image:
    - Extract tiles
    - Optionally filter by tissue threshold (for train-like sets)
    - Save RGB tiles under img_folder/images/ and mask tiles under img_folder/masks/
    - Collect metadata for each tile
    """
    images_dir = img_folder / "images"
    masks_dir = img_folder / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)

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
            masks_dir.mkdir(parents=True, exist_ok=True)
        else:
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
                save_tile = True
            else:
                if has_mask:
                    mask_tile = mask_img[y:y + tile_size, x:x + tile_size]
                    tissue_fraction = float(np.mean(mask_tile > 0))
                    if tissue_fraction >= tissue_threshold:
                        save_tile = True
                else:
                    save_tile = False

            if not save_tile:
                continue

            tile_name = f"{tile_id:07d}"
            img_tile_path = images_dir / f"{tile_name}.tif"
            rgb_pil = Image.fromarray(rgb_tile)
            if resize_to is not None:
                rgb_pil = rgb_pil.resize((resize_to, resize_to), Image.LANCZOS)
            rgb_pil.save(img_tile_path)

            mask_tile_path = None
            if has_mask:
                mask_tile = mask_img[y:y + tile_size, x:x + tile_size]
                mask_tile_path = masks_dir / f"{tile_name}.tif"
                mask_pil = Image.fromarray(mask_tile)
                if resize_to is not None:
                    mask_pil = mask_pil.resize((resize_to, resize_to), Image.NEAREST)
                mask_pil.save(mask_tile_path)

            metadata_rows.append(
                {
                    "source_file": file_name,
                    "img_idx": f"{img_idx:03d}",
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "image_path": str(img_tile_path),
                    "mask_path": str(mask_tile_path) if mask_tile_path else "",
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "saved_size": resize_to if resize_to is not None else tile_size,
                    "stride": stride,
                    "overlap": overlap,
                    "tissue_fraction": tissue_fraction if tissue_fraction is not None else "",
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
    resize_to=None,
    overlap=0,
    tissue_threshold=0.5,
    image_type="trainA",
    num_workers=None,
    metadata_csv_name="tiles_metadata.csv",
):
    """
    Create tiles from RGB images (and optional masks).

    Each WSI gets its own numbered subfolder (001/, 002/, ...) under
    <output_folder_dir>/<image_type>/. RGB tiles are saved in <idx>/images/
    and mask tiles in <idx>/masks/. If the output directory already contains
    numbered folders, new images continue from the next available index.

    Parameters
    ----------
    rgb_folder_path : str or Path
        Root folder containing image_type subfolder with RGB .tif files.
    output_folder_dir : str or Path
        Root output folder. Tiles will be saved in:
            <output_folder_dir>/<image_type>/<img_idx:03d>/images/
            <output_folder_dir>/<image_type>/<img_idx:03d>/masks/
    mask_folder_path : str or Path, optional
        Root folder containing image_type subfolder with mask .tif files.
    tile_size : int, default 256
        Size of square tiles extracted from the WSI.
    resize_to : int, optional
        Resize each extracted tile to this size before saving.
    overlap : float, default 0
        Fractional overlap between tiles (0 to <1).
    tissue_threshold : float, default 0.5
        Minimum tissue fraction to keep a tile for non-test image types.
    image_type : str, default "trainA"
        Subfolder name under both rgb_folder_path and mask_folder_path.
    num_workers : int, optional
        Number of worker processes. If None, use os.cpu_count().
    metadata_csv_name : str, default "tiles_metadata.csv"
        Name of the CSV metadata file saved in the image_type output folder.
    """
    rgb_folder = Path(rgb_folder_path) / image_type
    output_root = Path(output_folder_dir) / image_type
    output_root.mkdir(parents=True, exist_ok=True)

    mask_folder = Path(mask_folder_path) / image_type if mask_folder_path else None

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError(
            f"Invalid overlap {overlap}. It must be < 1. "
            f"Got stride={stride} for tile_size={tile_size}."
        )

    # --- Resume: find the next available img_idx ---
    existing = [
        int(d.name) for d in output_root.iterdir()
        if d.is_dir() and re.fullmatch(r"\d{3}", d.name)
    ]
    start_idx = max(existing) + 1 if existing else 1

    # --- Collect input files ---
    file_names = sorted(
        f for f in os.listdir(rgb_folder)
        if f.lower().endswith(".tif")
    )
    if not file_names:
        print(f"No .tif files found in {rgb_folder}")
        return

    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    print(
        f"Processing {len(file_names)} images from {rgb_folder} "
        f"(starting at img_idx={start_idx:03d}) with {num_workers} worker(s)..."
    )

    total_tiles = 0
    new_metadata = []

    fieldnames = [
        "source_file",
        "img_idx",
        "tile_id",
        "tile_name",
        "image_path",
        "mask_path",
        "x",
        "y",
        "tile_size",
        "saved_size",
        "stride",
        "overlap",
        "tissue_fraction",
        "image_type",
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_image,
                file_name,
                img_idx,
                rgb_folder,
                output_root / f"{img_idx:03d}",
                mask_folder,
                tile_size,
                stride,
                overlap,
                tissue_threshold,
                image_type,
                resize_to,
            ): (img_idx, file_name)
            for img_idx, file_name in enumerate(file_names, start=start_idx)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tiling"):
            img_idx, file_name = futures[future]
            try:
                tile_count, metadata_rows = future.result()
                total_tiles += tile_count
                new_metadata.extend(metadata_rows)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

            # Write per-WSI metadata CSV into the WSI subfolder alongside images/ and masks/
            if metadata_rows:
                wsi_dir = output_root / f"{img_idx:03d}"
                metadata_path = wsi_dir / metadata_csv_name
                with metadata_path.open("w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metadata_rows)
                print(f"  Saved metadata ({tile_count} tiles) → {metadata_path}")

    print(f"Total tiles saved: {total_tiles}")


def reconstruct_wsi(
    metadata_csv,
    output_dir,
    tile_dir=None,
    mode="rgb",
    blend="average",
):
    """
    Reconstruct full WSI(s) from tiles using the metadata CSV.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the tiles_metadata.csv generated during tiling.
    output_dir : str or Path
        Where reconstructed WSIs will be saved.
    tile_dir : str or Path, optional
        Directory containing translated tile images (e.g. inference output).
        Tiles are matched by tile_name from the CSV. If None, falls back to
        the image_path column in the CSV.
    mode : str
        "rgb" -> reconstruct only RGB images
        "mask" -> reconstruct only mask images
        "rgb_and_mask" -> reconstruct both
        "auto" -> reconstruct mask only if mask paths exist
    blend : str
        "average" -> average overlapping regions (recommended for overlapping tiles)
        "overwrite" -> last tile wins

    Returns
    -------
    dict:
        Dictionary mapping source_file -> dict with reconstructed paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = Path(tile_dir) if tile_dir else None

    metadata_csv = Path(metadata_csv)
    if metadata_csv.is_dir():
        # Accept a dataset root directory — collect all per-WSI CSVs inside it
        csv_files = sorted(metadata_csv.glob("*/tiles_metadata.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No per-WSI tiles_metadata.csv files found under: {metadata_csv}\n"
                f"Expected structure: {metadata_csv}/<NNN>/tiles_metadata.csv"
            )
        df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(metadata_csv)

    has_masks = df["mask_path"].notna().any() and (df["mask_path"] != "").any()
    if mode == "auto":
        mode = "rgb_and_mask" if has_masks else "rgb"

    results = {}

    for source_file, group in df.groupby("source_file"):
        print(f"\nReconstructing WSI: {source_file}")

        max_x = (group["x"] + group["tile_size"]).max()
        max_y = (group["y"] + group["tile_size"]).max()

        rgb_canvas = np.zeros((max_y, max_x, 3), dtype=np.float32)
        rgb_weight = np.zeros((max_y, max_x, 1), dtype=np.float32)

        mask_canvas = None
        mask_weight = None
        if mode in {"mask", "rgb_and_mask"} and has_masks:
            mask_canvas = np.zeros((max_y, max_x), dtype=np.float32)
            mask_weight = np.zeros((max_y, max_x), dtype=np.float32)

        for _, row in tqdm(group.iterrows(), total=len(group), desc="Placing tiles"):
            x, y = int(row["x"]), int(row["y"])
            tile_size = int(row["tile_size"])
            saved_size = int(row["saved_size"]) if "saved_size" in row and pd.notna(row.get("saved_size")) else tile_size
            needs_resize = saved_size != tile_size

            tile_path = _resolve_tile_path(row, tile_dir)

            if mode in {"rgb", "rgb_and_mask"} and tile_path is not None:
                rgb_pil = Image.open(tile_path)
                if needs_resize:
                    rgb_pil = rgb_pil.resize((tile_size, tile_size), Image.LANCZOS)
                rgb_tile = np.array(rgb_pil)
                rgb_canvas[y:y+tile_size, x:x+tile_size] += rgb_tile
                rgb_weight[y:y+tile_size, x:x+tile_size] += 1

            if (
                mode in {"mask", "rgb_and_mask"}
                and has_masks
                and isinstance(row["mask_path"], str)
                and row["mask_path"] != ""
            ):
                mask_pil = Image.open(row["mask_path"])
                if needs_resize:
                    mask_pil = mask_pil.resize((tile_size, tile_size), Image.NEAREST)
                mask_tile = np.array(mask_pil)
                mask_canvas[y:y+tile_size, x:x+tile_size] += mask_tile
                mask_weight[y:y+tile_size, x:x+tile_size] += 1

        def finalize(canvas, weight):
            if blend == "average":
                canvas = np.divide(
                    canvas, weight,
                    out=np.zeros_like(canvas),
                    where=weight > 0
                )
            return canvas

        rgb_output = None
        mask_output = None

        if mode in {"rgb", "rgb_and_mask"}:
            rgb_reconstructed = finalize(rgb_canvas, rgb_weight)
            rgb_output = output_dir / source_file
            Image.fromarray(rgb_reconstructed.astype(np.uint8)).save(rgb_output)

        if mode in {"mask", "rgb_and_mask"} and has_masks:
            mask_reconstructed = finalize(mask_canvas, mask_weight)
            mask_stem = Path(source_file).stem
            mask_output = output_dir / f"{mask_stem}_mask.tif"
            Image.fromarray(mask_reconstructed.astype(np.uint8)).save(mask_output)

        results[source_file] = {
            "rgb": str(rgb_output) if rgb_output else None,
            "mask": str(mask_output) if mask_output else None,
        }

    print("\nReconstruction complete!")
    return results


def _resolve_tile_path(row, tile_dir):
    """Return the path to a tile image, checking tile_dir first then image_path."""
    if tile_dir is not None:
        tile_name = row["tile_name"]
        for ext in (".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            candidate = tile_dir / f"{tile_name}{ext}"
            if candidate.exists():
                return candidate
        return None
    return row["image_path"]
