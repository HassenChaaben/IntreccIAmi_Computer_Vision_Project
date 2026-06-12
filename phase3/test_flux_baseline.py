#!/usr/bin/env python3
"""
=============================================================================
Phase 3 - Task 3.5: Flux Baseline Model Verification (Pre-Training Inference)
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Run baseline image generation (before LoRA training) using DiffSynth-Studio
         pipelines specifically for the FLUX model.
=============================================================================
"""

import os
import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pre-flight: check that we have torch + diffsynth available
# ---------------------------------------------------------------------------
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("[FATAL] PyTorch is not installed. Run this script inside your GPU conda env.")
    print("        Example:  conda activate diffsynth")
    sys.exit(1)

try:
    import diffsynth
    print(f"[OK] diffsynth package found at: {diffsynth.__file__}")
except ImportError:
    print("[FATAL] diffsynth is not installed. Make sure you are inside the DiffSynth-Studio directory")
    print("        and have run:  pip install -e .")
    sys.exit(1)

from diffsynth.core import ModelConfig
from diffsynth.pipelines.flux_image import FluxImagePipeline

print("[OK] FluxImagePipeline imported successfully.\n")

# ---------------------------------------------------------------------------
# Resolve project root (handles running from 'phase3', project root, or DiffSynth-Studio)
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
if (script_dir / "data").exists():
    PROJECT_ROOT = script_dir
elif (script_dir.parent / "data").exists():
    PROJECT_ROOT = script_dir.parent
else:
    # Fallback: look relative to CWD
    cwd = Path.cwd()
    if (cwd / "data").exists():
        PROJECT_ROOT = cwd
    else:
        PROJECT_ROOT = script_dir

print(f"[INFO] PROJECT_ROOT resolved to: {PROJECT_ROOT}")


def load_caption(jsonl_path: Path, index: int):
    """Load a caption and task info from metadata_flux.jsonl."""
    if not jsonl_path.exists():
        print(f"[ERROR] Caption file not found at: {jsonl_path}")
        sys.exit(1)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if index < 0 or index >= len(records):
        print(f"[ERROR] Caption index {index} out of bounds (found {len(records)} records).")
        sys.exit(1)

    return records[index]


def run_flux_inference(caption: str, output_path: str, device: str = "cuda"):
    """Execute FLUX baseline image generation via DiffSynth-Studio."""
    print(f"\n--- Loading FLUX Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")

    print(f"Initializing FluxImagePipeline (bfloat16, {device})...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ]
    )

    print("Executing image generation pipeline (30 steps)...")
    image = pipe(
        prompt=caption,
        num_inference_steps=30,
        width=1024,
        height=1024
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"[OK] Image saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DiffSynth FLUX Baseline Inference Tester")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the caption in the JSONL dataset to test (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (defaults to data/baseline_outputs/baseline_flux_task_<id>.png)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (default: cuda)")
    args = parser.parse_args()

    # Locate the metadata_flux.jsonl path
    # On the server, it is at /home/project_id_10/DiffSynth-Studio/data/id10/flux/metadata_flux.jsonl
    jsonl_path = PROJECT_ROOT / "data" / "id10" / "flux" / "metadata_flux.jsonl"

    print(f"Loading caption from: {jsonl_path.relative_to(PROJECT_ROOT) if jsonl_path.is_relative_to(PROJECT_ROOT) else jsonl_path}")
    record = load_caption(jsonl_path, args.index)
    
    # Check for 'caption', fallback to 'prompt' or 'text'
    caption = record.get("caption", record.get("prompt", record.get("text")))
    task_id = record.get("metadata", {}).get("task_id", "unknown")

    if not caption:
        print(f"[ERROR] Caption field is empty for task index {args.index}.")
        sys.exit(1)

    print(f"Loaded Task ID: {task_id} at index {args.index}")

    # Set default output image path if not provided
    if args.output is None:
        output_dir = PROJECT_ROOT / "data" / "baseline_outputs"
        output_path = str(output_dir / f"baseline_flux_task_{task_id}.png")
    else:
        output_path = args.output

    # Run inference
    run_flux_inference(caption, output_path, device=args.device)


if __name__ == "__main__":
    main()
