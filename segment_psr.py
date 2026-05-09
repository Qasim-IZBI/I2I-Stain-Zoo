import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def discover_wsi_tifs(data_dir: Path) -> list:
    tifs = sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))
    if not tifs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {data_dir}")
    return tifs


def discover_tiles(data_dir: Path, data_range=None) -> list:
    """
    Walk {data_dir}/{NNN}/images/*.tif structure.
    Returns list of (tile_path, idx_str, tile_stem) tuples.
    """
    tiles = []
    if data_range is not None:
        start, end = data_range
        idx_dirs = [data_dir / f"{i:03d}" for i in range(start, end + 1)
                    if (data_dir / f"{i:03d}").is_dir()]
    else:
        idx_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit())

    for idx_dir in idx_dirs:
        images_dir = idx_dir / "images"
        if not images_dir.is_dir():
            continue
        for tile in sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff")):
            tiles.append((tile, idx_dir.name, tile.stem))

    return tiles


def build_nnunet_input(wsi_paths: list, tmp_dir: Path) -> dict:
    """Stage WSIs into nnUNet input format: {stem}_0000{suffix}. Returns stem → original path."""
    mapping = {}
    for wsi in wsi_paths:
        stem = wsi.stem
        link = tmp_dir / f"{stem}_0000{wsi.suffix}"
        try:
            link.symlink_to(wsi.resolve())
        except OSError:
            shutil.copy2(wsi, link)
        mapping[stem] = wsi
    return mapping


def build_nnunet_input_tiles(tiles: list, tmp_dir: Path) -> dict:
    """
    Stage tiles into nnUNet input format with unique names {idx}_{tile_stem}_0000.tif.
    Returns mapping: nnunet_stem → (idx_str, tile_stem, suffix).
    """
    mapping = {}
    for tile_path, idx_str, tile_stem in tiles:
        nnunet_stem = f"{idx_str}_{tile_stem}"
        link_name = f"{nnunet_stem}_0000{tile_path.suffix}"
        link = tmp_dir / link_name
        try:
            link.symlink_to(tile_path.resolve())
        except OSError:
            shutil.copy2(tile_path, link)
        mapping[nnunet_stem] = (idx_str, tile_stem, tile_path.suffix)
    return mapping


def run_nnunet_predict(tmp_input: Path, tmp_output: Path, args) -> None:
    env = os.environ.copy()
    if args.nnunet_results:
        env["nnUNet_results"] = str(args.nnunet_results)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(tmp_input),
        "-o", str(tmp_output),
        "-d", str(args.nnunet_dataset),
        "-c", args.nnunet_config,
        "-f", *args.nnunet_folds.split(),
        "-npp", str(args.npp),
        "-nps", str(args.nps),
        "-device", args.device,
    ]
    if args.nnunet_trainer:
        cmd += ["-tr", args.nnunet_trainer]

    print("Running:", " ".join(str(c) for c in cmd))

    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=None if args.verbose else subprocess.PIPE,
            stderr=None if args.verbose else subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nnUNetv2_predict not found in PATH. "
            "Install nnU-Net v2 and ensure the correct conda environment is active."
        )

    if result.returncode != 0:
        if not args.verbose and result.stderr:
            sys.stderr.write(result.stderr.decode(errors="replace"))
        raise RuntimeError(f"nnUNetv2_predict exited with code {result.returncode}")


def collect_outputs(tmp_output: Path, outdir: Path, stem_map: dict) -> None:
    """Copy predicted masks to outdir, restoring original filenames (without _0000)."""
    outdir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for mask in sorted(tmp_output.iterdir()):
        if not mask.is_file():
            continue
        stem = mask.stem
        # nnUNet v2 strips _0000 from the output name, but handle it defensively
        if stem.endswith("_0000"):
            stem = stem[:-5]
        original = stem_map.get(stem)
        suffix = original.suffix if original else mask.suffix
        shutil.copy2(mask, outdir / f"{stem}{suffix}")
        copied += 1
    print(f"Saved {copied} mask(s) to {outdir}")


def _nnunet_stem(filename: str) -> str:
    """Strip extension(s) and defensive _0000 suffix from an nnUNet output filename."""
    stem = filename
    for suf in (".tif", ".tiff", ".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def collect_tile_outputs(raw_dir: Path, outdir: Path, mapping: dict) -> None:
    """
    Copy nnUNet tile predictions from raw_dir into {outdir}/{idx}/images/{tile}.tif.
    Skips files already present at the destination (idempotent on re-run).
    """
    copied = 0
    for mask in sorted(raw_dir.iterdir()):
        if not mask.is_file():
            continue
        stem = _nnunet_stem(mask.name)
        if stem not in mapping:
            print(f"[WARN] No mapping found for nnUNet output: {mask.name} (stem={stem!r})")
            continue
        idx_str, tile_stem, orig_suffix = mapping[stem]
        out_dir = outdir / idx_str / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{tile_stem}{orig_suffix}"
        if not dest.exists():
            shutil.copy2(mask, dest)
            copied += 1
    print(f"Saved {copied} tile mask(s) to {outdir}")


def _stream_tile_outputs(raw_dir: Path, outdir: Path, mapping: dict,
                         stop: threading.Event) -> None:
    """
    Background thread: poll raw_dir every second and copy completed nnUNet
    predictions into {outdir}/{NNN}/images/ as they appear.
    Runs until stop is set, then does one final sweep.
    """
    moved: set = set()
    while True:
        for f in sorted(raw_dir.glob("*")):
            if f.name in moved or not f.is_file():
                continue
            stem = _nnunet_stem(f.name)
            if stem not in mapping:
                continue
            idx_str, tile_stem, orig_suffix = mapping[stem]
            dest_dir = outdir / idx_str / "images"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{tile_stem}{orig_suffix}"
            try:
                shutil.copy2(f, dest)
                moved.add(f.name)
            except OSError:
                pass  # file still being written by nnUNet; retry next cycle
        if stop.is_set():
            break
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="PSR positive area segmentation via nnUNet v2.\n"
                    "WSI mode (default): takes pre-reconstructed WSI TIFs.\n"
                    "Tile mode (--tile_mode): segments inference tiles directly and saves\n"
                    "per-tile masks as {outdir}/{NNN}/images/{tile}.tif, ready for\n"
                    "reconstruct.py --tile_dir to stitch into WSI-level masks."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="WSI TIF directory (default) or tile directory root (--tile_mode)")
    parser.add_argument("--tile_mode", action="store_true",
                        help="Segment individual tiles instead of reconstructed WSIs. "
                             "--data must have {NNN}/images/*.tif structure (inference output). "
                             "Outputs saved as {outdir}/{NNN}/images/{tile}.tif.")
    parser.add_argument("--data_range", type=str, default=None,
                        help="(tile_mode) Limit to WSI folders start,end inclusive (e.g. '1,5')")
    parser.add_argument("--outdir", type=Path, default=Path("psr_masks"),
                        help="Output directory for predicted mask TIFs [%(default)s]")
    parser.add_argument("--nnunet_results", type=Path, default=None,
                        help="Path to nnUNet results folder (sets nnUNet_results env var). "
                             "If omitted, the existing NNUNET_RESULTS env var is used.")
    parser.add_argument("--nnunet_dataset", type=int, required=True,
                        help="nnUNet dataset ID (e.g. 1 for Dataset001_PSR)")
    parser.add_argument("--nnunet_config", type=str, default="2d",
                        help="nnUNet configuration, e.g. 2d or 3d_fullres [%(default)s]")
    parser.add_argument("--nnunet_folds", type=str, default="all",
                        help="Folds to use, space-separated or 'all' [%(default)s]")
    parser.add_argument("--nnunet_trainer", type=str, default=None,
                        help="nnUNet trainer class override (uses nnUNet default if omitted)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Compute device [%(default)s]")
    parser.add_argument("--npp", type=int, default=1,
                        help="nnUNet preprocessing workers (-npp) [%(default)s]")
    parser.add_argument("--nps", type=int, default=1,
                        help="nnUNet segmentation export workers (-nps) [%(default)s]")
    parser.add_argument("--verbose", action="store_true",
                        help="Stream nnUNetv2_predict stdout/stderr live")
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"--data directory not found: {args.data}")
    if args.nnunet_results is None and "NNUNET_RESULTS" not in os.environ:
        parser.error(
            "--nnunet_results is required when NNUNET_RESULTS is not set in the environment"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.tile_mode:
        data_range = None
        if args.data_range:
            start, end = args.data_range.split(",")
            data_range = (int(start.strip()), int(end.strip()))

        tiles = discover_tiles(args.data, data_range)
        if not tiles:
            raise FileNotFoundError(
                f"No tiles found in {args.data} "
                f"(expected {{NNN}}/images/*.tif structure)"
            )
        n_wsis = len(set(t[1] for t in tiles))
        print(f"Found {len(tiles)} tile(s) across {n_wsis} WSI folder(s) in {args.data}")

        # Persistent staging dir: survives crashes so nnUNet can resume (it skips
        # cases that already have output here) and partial results are never lost.
        nnunet_raw_dir = args.outdir / "_nnunet_raw"
        nnunet_raw_dir.mkdir(parents=True, exist_ok=True)

        tmp_input = Path(tempfile.mkdtemp(prefix="nnunet_tiles_in_"))
        try:
            tile_mapping = build_nnunet_input_tiles(tiles, tmp_input)

            # Background thread copies completed predictions into {NNN}/images/ as
            # nnUNet writes them — results appear incrementally during the run.
            stop_event = threading.Event()
            collector = threading.Thread(
                target=_stream_tile_outputs,
                args=(nnunet_raw_dir, args.outdir, tile_mapping, stop_event),
                daemon=True,
            )
            collector.start()
            try:
                run_nnunet_predict(tmp_input, nnunet_raw_dir, args)
            finally:
                stop_event.set()
                collector.join()
        finally:
            shutil.rmtree(tmp_input, ignore_errors=True)

        # Final sweep: catch any tiles the background thread may have missed
        collect_tile_outputs(nnunet_raw_dir, args.outdir, tile_mapping)
        # Staging dir is no longer needed after all results are in place
        shutil.rmtree(nnunet_raw_dir, ignore_errors=True)

    else:
        wsi_paths = discover_wsi_tifs(args.data)
        print(f"Found {len(wsi_paths)} WSI(s) in {args.data}")

        tmp_input = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
        try:
            build_nnunet_input(wsi_paths, tmp_input)
            run_nnunet_predict(tmp_input, args.outdir, args)
            print(f"Masks saved to {args.outdir}")
        finally:
            shutil.rmtree(tmp_input, ignore_errors=True)


if __name__ == "__main__":
    main()
