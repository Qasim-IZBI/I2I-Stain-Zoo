"""Check that two directories contain the same set of filenames.

Reports files present in one directory but missing from the other.
Exits with code 0 if the sets match, 1 if there are any discrepancies.

Usage
-----
python check_files.py --dirA ./psr_masks/real/psr_masks_wsi_final/ \
                      --dirB ./psr_masks/cyclegan/.../psr_masks_wsi_final/

# Restrict to a specific extension (default: all files)
python check_files.py --dirA ./real/ --dirB ./generated/ --ext .tif
"""

import argparse
import sys
from pathlib import Path


def collect_names(directory: Path, ext: str | None) -> set[str]:
    return {
        p.name for p in directory.iterdir()
        if p.is_file() and (ext is None or p.suffix.lower() == ext.lower())
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check that two directories contain the same filenames. "
                    "Exits 0 if they match, 1 if any files are missing."
    )
    parser.add_argument("--dirA", type=Path, required=True, help="First directory (reference).")
    parser.add_argument("--dirB", type=Path, required=True, help="Second directory (to check).")
    parser.add_argument("--ext", type=str, default=None,
                        help="Only consider files with this extension, e.g. .tif (default: all files).")
    args = parser.parse_args()

    for d in (args.dirA, args.dirB):
        if not d.is_dir():
            print(f"[ERROR] Directory not found: {d}")
            sys.exit(1)

    names_a = collect_names(args.dirA, args.ext)
    names_b = collect_names(args.dirB, args.ext)

    only_in_a = sorted(names_a - names_b)
    only_in_b = sorted(names_b - names_a)
    common    = names_a & names_b

    print(f"dirA : {args.dirA}  ({len(names_a)} file(s))")
    print(f"dirB : {args.dirB}  ({len(names_b)} file(s))")
    print(f"Match: {len(common)} file(s) in common")

    if only_in_a:
        print(f"\n[MISSING from dirB] {len(only_in_a)} file(s):")
        for name in only_in_a:
            print(f"  {name}")

    if only_in_b:
        print(f"\n[MISSING from dirA] {len(only_in_b)} file(s):")
        for name in only_in_b:
            print(f"  {name}")

    if not only_in_a and not only_in_b:
        print("OK — both directories contain identical filenames.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
