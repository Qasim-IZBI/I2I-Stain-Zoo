# check_dataset.py
"""
Scan one or more dataset directories for unreadable or corrupt tiles.

Usage:
    # Check a single directory
    python check_dataset.py --data path/to/tiles/trainA

    # Check multiple directories in one pass
    python check_dataset.py --data path/to/tiles/trainA path/to/tiles/trainB

    # Limit to a WSI range
    python check_dataset.py --data path/to/tiles/trainA --data_range 1,6

    # Use multiple workers for faster scanning
    python check_dataset.py --data path/to/tiles/trainA --workers 8

    # Save the list of bad tiles to a file
    python check_dataset.py --data path/to/tiles/trainA --save_bad bad_tiles.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from PIL import Image

from datasets.common import list_images, list_images_from_range


def check_tile(path: str) -> Optional[str]:
    """
    Try to fully decode the tile.  Returns an error string on failure, None on success.
    Image.open() is lazy — calling .convert("RGB") forces the full decode and surfaces
    any truncation or I/O errors.
    """
    try:
        with Image.open(path) as img:
            img.convert("RGB")
        return None
    except Exception as e:
        return str(e)


def collect_paths(roots: List[str], data_range: Optional[Tuple[int, int]]) -> List[Tuple[str, str]]:
    """Return [(root_label, path), ...] for all images across all roots."""
    all_paths: List[Tuple[str, str]] = []
    for root in roots:
        try:
            if data_range is not None:
                paths = list_images_from_range(root, *data_range)
            else:
                paths = list_images(root)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        label = os.path.basename(root.rstrip("/"))
        for p in paths:
            all_paths.append((label, p))
    return all_paths


def main():
    parser = argparse.ArgumentParser(description="Check dataset tiles for I/O or decode errors.")
    parser.add_argument("--data", nargs="+", required=True, metavar="DIR",
                        help="One or more dataset root directories to scan.")
    parser.add_argument("--data_range", type=str, default=None, metavar="START,END",
                        help="Only scan WSI subfolders in this range, e.g. 1,6")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel reader threads (default: 4).")
    parser.add_argument("--save_bad", type=str, default=None, metavar="FILE",
                        help="Write paths of bad tiles to this file (one per line).")
    args = parser.parse_args()

    data_range = None
    if args.data_range:
        parts = args.data_range.split(",")
        if len(parts) != 2:
            print("--data_range must be START,END  e.g.  1,6", file=sys.stderr)
            sys.exit(1)
        data_range = (int(parts[0]), int(parts[1]))

    print(f"Collecting tile paths from: {args.data}")
    entries = collect_paths(args.data, data_range)
    total = len(entries)
    print(f"Found {total:,} tiles — scanning with {args.workers} workers...\n")

    bad: List[Tuple[str, str, str]] = []  # (root_label, path, error)
    checked = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_tile, path): (label, path) for label, path in entries}
        for future in as_completed(futures):
            label, path = futures[future]
            error = future.result()
            checked += 1
            if error:
                bad.append((label, path, error))
                print(f"  [BAD]  {path}\n         {error}")
            if checked % max(1, total // 20) == 0:
                pct = checked / total * 100
                print(f"  ... {checked:,}/{total:,}  ({pct:.0f}%)  bad so far: {len(bad)}")

    print()
    print("=" * 60)
    print(f"Scan complete: {total:,} tiles checked")
    print(f"  OK : {total - len(bad):,}")
    print(f"  BAD: {len(bad):,}")

    if bad:
        print()
        print("Bad tiles:")
        for label, path, error in bad:
            print(f"  [{label}] {path}")
            print(f"           {error}")

        if args.save_bad:
            with open(args.save_bad, "w") as f:
                for _, path, _ in bad:
                    f.write(path + "\n")
            print(f"\nBad tile paths written to: {args.save_bad}")

        sys.exit(1)
    else:
        print("\nAll tiles are readable.")
        sys.exit(0)


if __name__ == "__main__":
    main()
