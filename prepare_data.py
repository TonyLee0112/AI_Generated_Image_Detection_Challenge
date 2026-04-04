"""
One-time dataset preparation script for NTIRE 2026 DeepFake Detection.

Downloads are obtained manually from CodaBench:
  https://www.codabench.org/competitions/12761/

Expected input layout (--zip_dir):
  <zip_dir>/
    shard_0.zip  (or shard_0/)
    shard_1.zip
    ...
    shard_5.zip
    val_official.zip  (optional, no labels)

Output layout (data/ntire2026/):
  train/
    shard_0/
      labels.csv
      images/
    shard_1/ ... shard_5/
  val_official/
    clear/
      images/
    distorted/
      images/
  manifests/
    train_shards.txt
    val_info.txt

Usage:
  python prepare_data.py --zip_dir /path/to/downloads
  python prepare_data.py --zip_dir /path/to/downloads --dry_run
"""

import argparse
import zipfile
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "ntire2026"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare NTIRE 2026 dataset")
    p.add_argument(
        "--zip_dir",
        required=True,
        type=Path,
        help="Directory containing downloaded shard zip files or folders",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without copying files",
    )
    p.add_argument(
        "--shards",
        nargs="+",
        type=int,
        default=list(range(6)),
        help="Which shards to prepare (default: 0-5)",
    )
    return p.parse_args()


def prepare_shard(src: Path, shard_num: int, dry_run: bool) -> bool:
    dest = DATA_ROOT / "train" / f"shard_{shard_num}"

    if dest.is_dir() and (dest / "labels.csv").is_file():
        print(f"  [SKIP] shard_{shard_num} already exists at {dest}")
        return True

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    # Try zip first
    zip_path = src / f"shard_{shard_num}.zip"
    folder_path = src / f"shard_{shard_num}"

    if zip_path.is_file():
        print(f"  Extracting {zip_path} -> {dest}")
        if not dry_run:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        return True

    elif folder_path.is_dir():
        print(f"  Copying {folder_path} -> {dest}")
        if not dry_run:
            shutil.copytree(str(folder_path), str(dest), dirs_exist_ok=True)
        return True

    else:
        print(f"  [WARN] shard_{shard_num} not found in {src} (tried .zip and folder)")
        return False


def prepare_val(src: Path, dry_run: bool):
    dest = DATA_ROOT / "val_official"
    zip_path = src / "val_official.zip"
    folder_path = src / "val_official"

    if dest.is_dir():
        print(f"  [SKIP] val_official already exists at {dest}")
        return

    if zip_path.is_file():
        print(f"  Extracting {zip_path} -> {dest}")
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
    elif folder_path.is_dir():
        print(f"  Copying {folder_path} -> {dest}")
        if not dry_run:
            shutil.copytree(str(folder_path), str(dest), dirs_exist_ok=True)
    else:
        print(f"  [SKIP] No val_official source found (optional)")


def write_manifests(shards: list, dry_run: bool):
    manifests_dir = DATA_ROOT / "manifests"
    if not dry_run:
        manifests_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifests_dir / "train_shards.txt"
    print(f"  Writing manifest: {train_manifest}")
    if not dry_run:
        with open(train_manifest, "w") as f:
            for s in shards:
                shard_path = DATA_ROOT / "train" / f"shard_{s}"
                f.write(f"shard_{s}\t{shard_path}\n")

    val_info = manifests_dir / "val_info.txt"
    if not dry_run:
        with open(val_info, "w") as f:
            f.write(
                "Official validation set has no labels.\n"
                "For local supervised validation, use shard_5 from training data.\n"
                "Config default: train=shard_0~4, val=shard_5\n"
            )


def verify_shard(shard_num: int) -> bool:
    d = DATA_ROOT / "train" / f"shard_{shard_num}"
    ok = d.is_dir() and (d / "labels.csv").is_file() and (d / "images").is_dir()
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] shard_{shard_num}: {d}")
    return ok


def main():
    args = parse_args()

    if not args.zip_dir.is_dir():
        print(f"[ERROR] --zip_dir does not exist: {args.zip_dir}")
        sys.exit(1)

    print(f"\n=== NTIRE 2026 Dataset Preparation ===")
    print(f"Source: {args.zip_dir}")
    print(f"Target: {DATA_ROOT}")
    print(f"Shards: {args.shards}")
    if args.dry_run:
        print("[DRY RUN] No files will be moved.")
    print()

    print("--- Training shards ---")
    results = []
    for s in args.shards:
        results.append(prepare_shard(args.zip_dir, s, args.dry_run))

    print("\n--- Validation set (optional) ---")
    prepare_val(args.zip_dir, args.dry_run)

    print("\n--- Manifests ---")
    write_manifests(args.shards, args.dry_run)

    if not args.dry_run:
        print("\n--- Verification ---")
        all_ok = all(verify_shard(s) for s in args.shards)
        if all_ok:
            print("\n[SUCCESS] All shards verified. Run: python train.py")
        else:
            print("\n[PARTIAL] Some shards missing. Check the warnings above.")
    else:
        print("\n[DRY RUN complete] Re-run without --dry_run to apply.")


if __name__ == "__main__":
    main()
