#!/usr/bin/env python3
"""
=============================================================================
Phase 2 – Master Runner: Metadata Extraction & Normalization
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Execute all Phase 2 tasks in sequence:
         2.1 Raw JSON Parsing
         2.2 Integrity Verification (First Technical Check)
         2.3 Field Normalization & Parsing Script
         
         All these tasks are encapsulated within the `normalize_dataset.py` 
         script. This master runner ensures it's run with the correct path context.

Usage (from the project root):
    python phase2/run_phase2.py
=============================================================================
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# This file lives in phase2/, so the phase2 directory is its parent.
PHASE2_DIR = Path(__file__).resolve().parent
# ---------------------------------------------------------------------------


def run_script(script_name: str) -> bool:
    """
    Run a Python script located in the phase2/ directory.

    Args:
        script_name: The filename of the script (e.g. "normalize_dataset.py").

    Returns:
        True if the script exited successfully, False otherwise.
    """
    script_path = PHASE2_DIR / script_name
    print(f"\n{'-' * 65}")
    print(f"  Running: {script_name}")
    print(f"{'-' * 65}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PHASE2_DIR.parent),  # Run from project root
    )
    return result.returncode == 0


def main():
    """Execute all Phase 2 tasks."""
    print("+" + "=" * 63 + "+")
    print("|  Phase 2 - Metadata Extraction & Normalization (Master Runner) |")
    print("+" + "=" * 63 + "+")

    # -- Tasks 2.1, 2.2, 2.3 ------------------------------------------
    ok = run_script("normalize_dataset.py")
    if not ok:
        print("\n[ERROR] Tasks 2.1-2.3 failed. Aborting.")
        sys.exit(1)

    # -- Done ----------------------------------------------------------
    print("\n+" + "=" * 63 + "+")
    print("|         [OK] Phase 2 complete (Tasks 2.1, 2.2, 2.3)        |")
    print("+" + "=" * 63 + "+")


if __name__ == "__main__":
    main()
