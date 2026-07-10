#!/usr/bin/env python3
"""
=============================================================================
Phase 3 - Task 3.5: Baseline Model Verification (Pre-Training Inference)
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Run baseline image generation (before LoRA training) using DiffSynth-Studio
         pipelines on base models (Flux, SDXL, Z-Image) to establish pre-training visual baseline.
=============================================================================
"""

import argparse
import json
import sys
import traceback
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

# Now import the actual pipeline classes — any failure here gives a REAL traceback
from diffsynth.core import ModelConfig
from diffsynth.pipelines.z_image import ZImagePipeline
from diffsynth.pipelines.flux_image import FluxImagePipeline
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

print("[OK] All pipeline classes imported successfully.\n")

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
    """Load a caption and task info from a captioned JSONL file."""
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


# ===== FLUX =====
def run_flux_inference(caption: str, output_path: str):
    """Execute FLUX baseline image generation via DiffSynth-Studio."""
    print(f"\n--- Loading FLUX Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")

    print("Initializing FluxImagePipeline (bfloat16, CUDA)...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
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

    image.save(output_path)
    print(f"[OK] Image saved successfully to: {output_path}")


# ===== SDXL =====
def run_sdxl_inference(caption: str, output_path: str):
    """Execute SDXL baseline image generation via DiffSynth-Studio."""
    print(f"\n--- Loading SDXL Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")

    print("Initializing StableDiffusionXLPipeline (float16, CUDA)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        torch_dtype=torch.float16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors"),
            ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
        tokenizer_2_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
    )

    print("Executing image generation pipeline (30 steps)...")
    image = pipe(
        prompt=caption,
        num_inference_steps=30,
        width=1024,
        height=1024
    )

    image.save(output_path)
    print(f"[OK] Image saved successfully to: {output_path}")


# ===== Z-Image =====
def run_zimage_inference(caption: str, output_path: str):
    """Execute Z-Image baseline image generation via DiffSynth-Studio."""
    print(f"\n--- Loading Z-Image Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")

    print("Initializing ZImagePipeline (bfloat16, CUDA)...")
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    )

    print("Executing image generation pipeline (50 steps, cfg 4.0)...")
    image = pipe(
        prompt=caption,
        seed=42,
        rand_device="cuda",
        num_inference_steps=50,
        cfg_scale=4.0,
        width=1024,
        height=1024
    )

    image.save(output_path)
    print(f"[OK] Image saved successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DiffSynth Baseline Inference Tester")
    parser.add_argument("--model", type=str, choices=["zimage", "flux", "sdxl"], required=True,
                        help="The base model to test ('zimage', 'flux', or 'sdxl')")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the caption in the JSONL dataset to test (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (defaults to data/baseline_<model>_task_<id>.png)")
    args = parser.parse_args()

    # Determine default paths based on the selected model
    if args.model == "zimage":
        jsonl_path = PROJECT_ROOT / "data" / "id10" / "zimage" / "metadata_zimage.jsonl"
    elif args.model == "flux":
        jsonl_path = PROJECT_ROOT / "data" / "id10" / "flux" / "metadata_flux.jsonl"
    else:
        jsonl_path = PROJECT_ROOT / "data" / "id10" / "sdxl" / "metadata_sdxl.jsonl"

    print(f"Loading caption from: {jsonl_path.relative_to(PROJECT_ROOT)}")
    record = load_caption(jsonl_path, args.index)
    caption = record.get("caption")
    task_id = record.get("metadata", {}).get("task_id", "unknown")

    if not caption:
        print(f"[ERROR] Caption field is empty for task index {args.index}.")
        sys.exit(1)

    print(f"Loaded Task ID: {task_id} at index {args.index}")

    # Set default output image path if not provided
    if args.output is None:
        output_dir = PROJECT_ROOT / "data" / "baseline_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"baseline_{args.model}_task_{task_id}.png")
    else:
        output_path = args.output

    # Run inference
    if args.model == "flux":
        run_flux_inference(caption, output_path)
    elif args.model == "sdxl":
        run_sdxl_inference(caption, output_path)
    elif args.model == "zimage":
        run_zimage_inference(caption, output_path)


if __name__ == "__main__":
    main()
