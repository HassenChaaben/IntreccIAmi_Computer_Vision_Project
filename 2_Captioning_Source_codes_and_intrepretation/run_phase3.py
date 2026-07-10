#!/usr/bin/env python3
"""
=============================================================================
Phase 3 – Master Runner: VLM Captioning (Ollama Engine)
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Execute all Phase 3 captioning engines (Z-Image, Flux, SDXL) in sequence.
         Supports testing locally using the --use_mock flag.
=============================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve directory path
PHASE3_DIR = Path(__file__).resolve().parent

def run_engine(script_name: str, use_mock: bool) -> bool:
    """Run a specific captioning engine script."""
    script_path = PHASE3_DIR / script_name
    print(f"\n{'-' * 65}")
    print(f"  Running Caption Engine: {script_name}")
    print(f"  Mode: {'MOCK (local test)' if use_mock else 'OLLAMA qwen32b-caption (GPU)'}")
    print(f"{'-' * 65}\n")

    cmd = [sys.executable, str(script_path)]
    if use_mock:
        cmd.append("--use_mock")

    # Execute and wait for completion
    result = subprocess.run(
        cmd,
        cwd=str(PHASE3_DIR.parent)  # Run from project root
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Master Runner")
    parser.add_argument("--use_mock", action="store_true", help="Force mock captioning for all engines")
    args = parser.parse_args()

    print("+" + "=" * 63 + "+")
    print("|          Phase 3 - VLM Captioning (Master Runner)          |")
    print("+" + "=" * 63 + "+")

    # 1. Run Z-Image
    ok = run_engine("caption_zimage.py", args.use_mock)
    if not ok:
        print("\n[ERROR] Z-Image caption engine failed. Aborting.")
        sys.exit(1)

    # 2. Run Flux
    ok = run_engine("caption_flux.py", args.use_mock)
    if not ok:
        print("\n[ERROR] Flux caption engine failed. Aborting.")
        sys.exit(1)

    # 3. Run SDXL
    ok = run_engine("caption_sdxl.py", args.use_mock)
    if not ok:
        print("\n[ERROR] SDXL caption engine failed. Aborting.")
        sys.exit(1)

    print("\n+" + "=" * 63 + "+")
    print("|              [OK] Phase 3 Captioning Complete              |")
    print("+" + "=" * 63 + "+")


if __name__ == "__main__":
    main()
