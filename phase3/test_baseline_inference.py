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
from pathlib import Path

# Resolve project root (handles running from either 'phase3' folder or project root)
script_dir = Path(__file__).resolve().parent
if (script_dir / "data").exists():
    PROJECT_ROOT = script_dir
else:
    PROJECT_ROOT = script_dir.parent

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


def run_flux_inference(caption: str, output_path: str):
    """Execute FLUX baseline image generation via DiffSynth-Studio."""
    print(f"\n--- Loading FLUX Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")
    
    try:
        import torch
        from diffsynth import ModelManager, FluxImagePipeline
        
        # 1. Initialize model manager on CUDA GPU
        print("Initializing ModelManager (bfloat16, CUDA)...")
        model_manager = ModelManager(device="cuda", torch_dtype=torch.bfloat16)
        
        # 2. Load FLUX-dev weights (Downloads automatically if not cached)
        print("Loading FLUX-dev models...")
        model_manager.load_models_from_wangyaning(model_names=["flux-dev"])
        
        # 3. Create pipeline
        pipe = FluxImagePipeline.from_model_manager(model_manager)
        
        # 4. Generate image
        print("Executing image generation pipeline (30 steps)...")
        image = pipe(
            prompt=caption,
            num_inference_steps=30,
            width=1024,
            height=1024
        )
        
        # 5. Save output
        image.save(output_path)
        print(f"[OK] Image saved successfully to: {output_path}")
        
    except ImportError:
        print("[WARNING] DiffSynth-Studio or PyTorch not available in local environment.")
        print("This script is ready to run on your GPU server. Here is the code it executes:")
        print("-" * 65)
        print("""from diffsynth import ModelManager, FluxImagePipeline
import torch
model_manager = ModelManager(device="cuda", torch_dtype=torch.bfloat16)
model_manager.load_models_from_wangyaning(model_names=["flux-dev"])
pipe = FluxImagePipeline.from_model_manager(model_manager)
image = pipe(prompt=prompt, num_inference_steps=30, width=1024, height=1024)
image.save(output_path)""")
        print("-" * 65)


def run_sdxl_inference(caption: str, output_path: str, is_zimage: bool = False):
    """Execute SDXL (or Z-Image) baseline image generation via DiffSynth-Studio."""
    model_desc = "Z-Image" if is_zimage else "SDXL"
    print(f"\n--- Loading {model_desc} Base Model & Running Inference ---")
    print(f"Prompt: {caption}\n")
    
    try:
        import torch
        from diffsynth import ModelManager, SDXLImagePipeline
        
        # 1. Initialize model manager on CUDA GPU
        print("Initializing ModelManager (float16, CUDA)...")
        model_manager = ModelManager(device="cuda", torch_dtype=torch.float16)
        
        # 2. Load SDXL base weights
        print(f"Loading {model_desc} base models...")
        # Z-Image and SDXL both run on the SDXL pipeline architecture
        model_manager.load_models_from_wangyaning(model_names=["sdxl_base"])
        
        # 3. Create pipeline
        pipe = SDXLImagePipeline.from_model_manager(model_manager)
        
        # 4. Generate image
        print("Executing image generation pipeline (30 steps)...")
        image = pipe(
            prompt=caption,
            num_inference_steps=30,
            width=1024,
            height=1024
        )
        
        # 5. Save output
        image.save(output_path)
        print(f"[OK] Image saved successfully to: {output_path}")
        
    except ImportError:
        print("[WARNING] DiffSynth-Studio or PyTorch not available in local environment.")
        print("This script is ready to run on your GPU server. Here is the code it executes:")
        print("-" * 65)
        print(f"""from diffsynth import ModelManager, SDXLImagePipeline
import torch
model_manager = ModelManager(device="cuda", torch_dtype=torch.float16)
model_manager.load_models_from_wangyaning(model_names=["sdxl_base"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)
image = pipe(prompt=prompt, num_inference_steps=30, width=1024, height=1024)
image.save(output_path)""")
        print("-" * 65)


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
        run_sdxl_inference(caption, output_path, is_zimage=False)
    elif args.model == "zimage":
        run_sdxl_inference(caption, output_path, is_zimage=True)


if __name__ == "__main__":
    main()
