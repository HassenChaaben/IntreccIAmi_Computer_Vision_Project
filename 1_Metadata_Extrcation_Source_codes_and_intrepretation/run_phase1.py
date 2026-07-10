#!/usr/bin/env python3
"""
=============================================================================
Phase 1 – Master Runner: Environment & Data Preparation
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Execute all Phase 1 tasks in sequence:
         1.1 Directory Setup
         1.2 Retrieve Source Materials (copy JSON + images)
         1.3 Environment Installation  ← SKIPPED (run on GPU server)

This script simply calls setup_directories.py and copy_source_materials.py
in order. It provides a single entry-point for the entire Phase 1 workflow.

Usage (from the project root):
    python phase1/run_phase1.py
=============================================================================
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# This file lives in phase1/, so the phase1 directory is its parent.
PHASE1_DIR = Path(__file__).resolve().parent
# ---------------------------------------------------------------------------


def run_script(script_name: str) -> bool:
    """
    Run a Python script located in the phase1/ directory.

    Args:
        script_name: The filename of the script (e.g. "setup_directories.py").

    Returns:
        True if the script exited successfully, False otherwise.
    """
    script_path = PHASE1_DIR / script_name
    print(f"\n{'─' * 65}")
    print(f"  Running: {script_name}")
    print(f"{'─' * 65}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PHASE1_DIR.parent),  # Run from project root
    )
    return result.returncode == 0


def main():
    """Execute all Phase 1 tasks sequentially."""
    print("╔" + "═" * 63 + "╗")
    print("║  Phase 1 – Environment & Data Preparation (Master Runner)   ║")
    print("╚" + "═" * 63 + "╝")

    # ── Task 1.1 ──────────────────────────────────────────────────────
    ok = run_script("setup_directories.py")
    if not ok:
        print("\n✗ Task 1.1 failed. Aborting.")
        sys.exit(1)

    # ── Task 1.2 ──────────────────────────────────────────────────────
    ok = run_script("copy_source_materials.py")
    if not ok:
        print("\n✗ Task 1.2 failed. Aborting.")
        sys.exit(1)

    # ── Task 1.3 (SKIPPED) ────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Task 1.3: Environment Installation  →  SKIPPED")
    print("  (To be executed on the university GPU server)")
    print(f"{'─' * 65}")

    # ── Done ──────────────────────────────────────────────────────────
    print("\n╔" + "═" * 63 + "╗")
    print("║         ✓  Phase 1 complete (Tasks 1.1 + 1.2)              ║")
    print("╚" + "═" * 63 + "╝")


if __name__ == "__main__":
    main()
