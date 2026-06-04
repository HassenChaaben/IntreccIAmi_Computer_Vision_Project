#!/usr/bin/env python3
"""
=============================================================================
Phase 1 - Task 1.1: Directory Setup
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Create the standardised workspace directory structure required by the
         project instructions (see Instructions §6, Step 1).

This script creates the following directory tree under the project root:
  data/
  ├── raw_json/     ← will hold the Label Studio JSON export
  └── images/       ← will hold the original texture images

The script is idempotent: re-running it will not fail or overwrite
existing data.

Usage (from the project root):
    python phase1/setup_directories.py
=============================================================================
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# The project root is one level above this script's folder.
# Adjust PROJECT_ROOT if your layout differs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ---------------------------------------------------------------------------

# Directories to create, relative to PROJECT_ROOT
DIRECTORIES_TO_CREATE = [
    "data",               # Top-level data folder
    "data/raw_json",      # For the Label Studio JSON export file
    "data/images",        # For the TEXTURE DI INTRECCI images
]


def main():
    """Create all required directories under the project root."""
    print("=" * 60)
    print("Phase 1 – Task 1.1: Directory Setup")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}\n")

    for rel_dir in DIRECTORIES_TO_CREATE:
        dir_path = PROJECT_ROOT / rel_dir
        if dir_path.exists():
            # Directory already exists – nothing to do
            print(f"  [EXISTS]  {dir_path}")
        else:
            # Create the directory (and any missing parents)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] {dir_path}")

    print("\n✓ Directory setup complete.")
    print("Next step: copy source materials into these directories (Task 1.2).")


if __name__ == "__main__":
    main()
