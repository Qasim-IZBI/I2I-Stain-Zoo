import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def discover_wsi_tifs(data_dir: Path) -> list:
    tifs = sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))
    if not tifs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {data_dir}")
    return tifs


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


def run_nnunet_predict(tmp_input: Path, tmp_output: Path, args) -> None:
    env = os.environ.copy()
    if args.nnunet_results:
        env["NNUNET_RESULTS"] = str(args.nnunet_results)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(tmp_input),
        "-o", str(tmp_output),
        "-d", str(args.nnunet_dataset),
        "-c", args.nnunet_config,
        "-f", *args.nnunet_folds.split(),
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


def main():
    parser = argparse.ArgumentParser(
        description="PSR positive area segmentation via nnUNet v2. "
                    "Prerequisite: reconstruct WSI TIFs from tiles first using reconstruct.py."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory of pre-reconstructed WSI TIFs")
    parser.add_argument("--outdir", type=Path, default=Path("psr_masks"),
                        help="Output directory for predicted mask TIFs [%(default)s]")
    parser.add_argument("--nnunet_results", type=Path, default=None,
                        help="Path to nnUNet results folder (sets NNUNET_RESULTS). "
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
    parser.add_argument("--verbose", action="store_true",
                        help="Stream nnUNetv2_predict stdout/stderr live")
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"--data directory not found: {args.data}")
    if args.nnunet_results is None and "NNUNET_RESULTS" not in os.environ:
        parser.error(
            "--nnunet_results is required when NNUNET_RESULTS is not set in the environment"
        )

    wsi_paths = discover_wsi_tifs(args.data)
    print(f"Found {len(wsi_paths)} WSI(s) in {args.data}")

    tmp_input  = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
    tmp_output = Path(tempfile.mkdtemp(prefix="nnunet_out_"))
    try:
        stem_map = build_nnunet_input(wsi_paths, tmp_input)
        run_nnunet_predict(tmp_input, tmp_output, args)
        collect_outputs(tmp_output, args.outdir, stem_map)
    finally:
        shutil.rmtree(tmp_input,  ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)


if __name__ == "__main__":
    main()
