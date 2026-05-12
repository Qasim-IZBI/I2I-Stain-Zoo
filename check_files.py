"""Check that two directories contain the same set of filenames.

Reports files present in one directory but missing from the other.
Exits with code 0 if the sets match, 1 if there are any discrepancies.

Use --strip_prefix when the two directories use different filename prefixes
that should be ignored during comparison, e.g.:
  dirA: SR_d31_BDL+v_M2.tif
  dirB: HE_d31_BDL+v_M2.tif
Both reduce to 'd31_BDL+v_M2.tif' after stripping the first '_'-delimited token.

Usage
-----
python check_files.py --dirA ./psr_masks/real/psr_masks_wsi_final/ \
                      --dirB ./psr_masks/cyclegan/.../psr_masks_wsi_final/

# Restrict to a specific extension (default: all files)
python check_files.py --dirA ./real/ --dirB ./generated/ --ext .tif

# Ignore differing prefixes (SR_ vs HE_)
python check_files.py --dirA ./real/ --dirB ./generated/ --ext .tif --strip_prefix
"""

import argparse
import sys
from pathlib import Path


def normalize(name: str, strip_prefix: bool) -> str:
    """Return the comparison key for a filename."""
    if not strip_prefix:
        return name
    # Drop everything up to and including the first underscore, keep extension
    stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
    parts = stem.split("_", 1)
    base = parts[1] if len(parts) > 1 else stem
    return f"{base}.{ext}" if ext else base


def collect_names(directory: Path, ext: str | None, strip_prefix: bool) -> dict[str, str]:
    """Return {normalized_name: original_name} for files in directory."""
    result = {}
    for p in directory.iterdir():
        if p.is_file() and (ext is None or p.suffix.lower() == ext.lower()):
            key = normalize(p.name, strip_prefix)
            result[key] = p.name
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check that two directories contain the same filenames. "
                    "Exits 0 if they match, 1 if any files are missing."
    )
    parser.add_argument("--dirA", type=Path, required=True, help="First directory (reference).")
    parser.add_argument("--dirB", type=Path, required=True, help="Second directory (to check).")
    parser.add_argument("--ext", type=str, default=None,
                        help="Only consider files with this extension, e.g. .tif (default: all files).")
    parser.add_argument("--strip_prefix", action="store_true",
                        help="Ignore the first underscore-delimited token when comparing names "
                             "(e.g. SR_slide.tif and HE_slide.tif are treated as matching).")
    args = parser.parse_args()

    for d in (args.dirA, args.dirB):
        if not d.is_dir():
            print(f"[ERROR] Directory not found: {d}")
            sys.exit(1)

    map_a = collect_names(args.dirA, args.ext, args.strip_prefix)
    map_b = collect_names(args.dirB, args.ext, args.strip_prefix)

    keys_a, keys_b = set(map_a), set(map_b)
    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common    = keys_a & keys_b

    print(f"dirA : {args.dirA}  ({len(map_a)} file(s))")
    print(f"dirB : {args.dirB}  ({len(map_b)} file(s))")
    if args.strip_prefix:
        print("Note : prefix-stripped comparison active")
    print(f"Match: {len(common)} file(s) in common")

    if only_in_a:
        print(f"\n[MISSING from dirB] {len(only_in_a)} file(s):")
        for key in only_in_a:
            print(f"  {map_a[key]}")

    if only_in_b:
        print(f"\n[MISSING from dirA] {len(only_in_b)} file(s):")
        for key in only_in_b:
            print(f"  {map_b[key]}")

    if not only_in_a and not only_in_b:
        print("OK — both directories contain matching filenames.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
