#!/usr/bin/env python3
"""
=============================================================================
Phase 1 - Task 1.2: Retrieve Source Materials (Copy to Workspace)
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Copy the raw Label Studio JSON file and the image folder into the
         standardised workspace created by Task 1.1.

How it works:
  1. Copies `Dataset/label_studio_texture_labels.json`  → `data/raw_json/`
  2. Copies every file in `Dataset/TEXTURE DI INTRECCI/` → `data/images/`

The script skips files that already exist at the destination (based on name
and size) so it can be re-run safely without duplicating work.

Assumptions:
  - The `Dataset/` folder sits directly under the project root and contains:
      • label_studio_texture_labels.json
      • TEXTURE DI INTRECCI/   (folder with ~183 image files)
  - The workspace directories (`data/raw_json/`, `data/images/`) already exist
    (created by setup_directories.py).

Usage (from the project root):
    python phase1/copy_source_materials.py
=============================================================================
"""

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source locations (under Dataset/)
SOURCE_JSON = PROJECT_ROOT / "Dataset" / "label_studio_texture_labels.json"
SOURCE_IMAGES_DIR = PROJECT_ROOT / "Dataset" / "TEXTURE DI INTRECCI"

# Destination locations (under data/)
DEST_JSON_DIR = PROJECT_ROOT / "data" / "raw_json"
DEST_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
# ---------------------------------------------------------------------------


def copy_file_if_needed(src: Path, dst_dir: Path) -> str:
    """
    Copy a single file from `src` into `dst_dir` unless an identical-size
    file already exists at the destination.

    Returns:
        A status string: "[COPIED]" or "[SKIPPED]".
    """
    dst = dst_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "[SKIPPED]"  # Already present, same size → skip
    shutil.copy2(src, dst)  # copy2 preserves metadata (timestamps)
    return "[COPIED] "


def main():
    """Copy JSON and images into the standardised data/ workspace."""
    print("=" * 65)
    print("Phase 1 – Task 1.2: Retrieve Source Materials (Copy)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Copy the Label Studio JSON export
    # ------------------------------------------------------------------
    print("\n--- Copying Label Studio JSON export ---")
    if not SOURCE_JSON.exists():
        print(f"  ERROR: Source JSON not found at {SOURCE_JSON}")
        print("  Make sure the Dataset/ folder is in the project root.")
        return
    if not DEST_JSON_DIR.exists():
        print(f"  ERROR: Destination directory does not exist: {DEST_JSON_DIR}")
        print("  Run setup_directories.py first.")
        return

    status = copy_file_if_needed(SOURCE_JSON, DEST_JSON_DIR)
    print(f"  {status} {SOURCE_JSON.name} → {DEST_JSON_DIR}")

    # ------------------------------------------------------------------
    # 2. Copy all image files from TEXTURE DI INTRECCI
    # ------------------------------------------------------------------
    print("\n--- Copying image files ---")
    if not SOURCE_IMAGES_DIR.exists():
        print(f"  ERROR: Source image folder not found at {SOURCE_IMAGES_DIR}")
        return
    if not DEST_IMAGES_DIR.exists():
        print(f"  ERROR: Destination directory does not exist: {DEST_IMAGES_DIR}")
        print("  Run setup_directories.py first.")
        return

    # Gather all files (non-recursive; TEXTURE DI INTRECCI is flat)
    image_files = sorted([
        f for f in SOURCE_IMAGES_DIR.iterdir()
        if f.is_file()
    ])
    print(f"  Found {len(image_files)} files in source folder.\n")

    copied_count = 0
    skipped_count = 0
    for img in image_files:
        stat = copy_file_if_needed(img, DEST_IMAGES_DIR)
        if "COPIED" in stat:
            copied_count += 1
        else:
            skipped_count += 1
        # Only print per-file detail in verbose mode to keep output clean
        # print(f"  {stat} {img.name}")

    print(f"  Copied:  {copied_count} files")
    print(f"  Skipped: {skipped_count} files (already present)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("✓ Source materials are now in the standardised workspace.")
    print(f"  JSON:   {DEST_JSON_DIR / SOURCE_JSON.name}")
    print(f"  Images: {DEST_IMAGES_DIR}  ({len(image_files)} files)")
    print("=" * 65)


if __name__ == "__main__":
    main()
