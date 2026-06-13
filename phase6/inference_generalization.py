#!/usr/bin/env python3
"""
=============================================================================
Phase 6 - Generalization Inference Demo Script
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Generate images using the trained LoRA weights for Flux, SDXL, and
         Z-Image on 10 newly-constructed, model-tailored test prompts to
         assess the LoRA generalization capacity on unseen objects and styles.
=============================================================================
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add DiffSynth-Studio to python path from various possible positions
script_dir = Path(__file__).resolve().parent
possible_paths = [
    script_dir.parent / "DiffSynth-Studio",
    script_dir / "DiffSynth-Studio",
    script_dir.parent,
    script_dir
]
for p in possible_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

from diffsynth import ModelConfig
from diffsynth.pipelines.flux_image import FluxImagePipeline
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffsynth.pipelines.z_image import ZImagePipeline

# 10 Newly-constructed generalization test prompts (unseen objects and styles)
GENERALIZATION_PROMPTS = [
    # 1. Lamp shade (rattan lattice texture)
    "intrecciami-style: A minimalist woven rattan lamp shade featuring a simple repeatable lattice texture, casting soft geometric shadows on a warm background. Studio lighting, premium texture, high resolution, macro photography.",
    # 2. Handbag (navy blue leather spina pattern)
    "intrecciami-style: A modern designer handbag crafted from woven leather, showing a tight and intricate spina weave pattern in deep navy blue. The leather strands have a smooth finish with visible natural grain. Close-up studio photograph, high resolution.",
    # 3. Decorative basket (natural rush bark semplici weave)
    "intrecciami-style: An elegant decorative basket crafted with natural rush bark, displaying a tight and structured semplici weave. The natural fibers have an unvarnished Grezzo finish. Studio background, high resolution product shot.",
    # 4. Headboard (white leather checkerboard pattern)
    "intrecciami-style: A luxurious headboard detail featuring woven white leather strips in a dense scacco checkerboard pattern. Clean studio lighting highlights the texture and fine stitching. Premium quality macro photography.",
    # 5. Wall hanging (green jute vario knots)
    "intrecciami-style: A contemporary woven wall hanging made of natural fibers and dyed green jute strands, showing a varying Vario knot technique. High contrast studio setting, close-up photograph.",
    # 6. Stool seat (natural cord spina salto pattern)
    "intrecciami-style: A handcrafted wooden stool with a woven seat made from natural paper cord in a tight, diagonal spina salto pattern. Macro photography, showing the texture of the twisted cord.",
    # 7. Tray (orange and brown leather piattina strands)
    "intrecciami-style: A designer home tray featuring a woven base with orange and brown leather piattina strands. The pattern is a repeating Intreccio semplice with a matte finish. Professional studio lighting, close-up.",
    # 8. Vase cover (split bamboo semplici structure)
    "intrecciami-style: A decorative cylindrical vase cover made of split bamboo strips, showcasing a simple over-under weave structure with a raw, natural finish. Clean studio background.",
    # 9. Coaster (circular hemp rope pattern)
    "intrecciami-style: A handcrafted decorative coaster featuring a circular pattern made of natural hemp rope, showing a tight and symmetrical spiral weave. High detail, macro photography.",
    # 10. Office chair backrest (breathable mesh of white and grey leather laces)
    "intrecciami-style: A modern ergonomic chair backrest featuring a breathable woven mesh of white and grey leather laces in an intricate cross-weaving pattern. High resolution studio photography."
]

def resolve_project_root():
    """Resolve project root directory robustly."""
    curr = Path(__file__).resolve().parent
    if curr.name == "phase6" and (curr.parent / "data").exists():
        return curr.parent
    elif (curr / "data").exists():
        return curr
    else:
        cwd = Path.cwd()
        if (cwd / "data").exists():
            return cwd
        return curr.parent

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Generalization Inference Demo")
    parser.add_argument("--model", type=str, default="flux", choices=["flux", "sdxl", "zimage"],
                        help="Model architecture to run (default: flux)")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="LoRA epoch to load (default: 1 for flux, 4 for sdxl/zimage)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps (default: 30)")
    args = parser.parse_args()

    # Determine default epoch if not specified
    if args.epoch == -1:
        if args.model == "flux":
            args.epoch = 1
        else:
            args.epoch = 4

    project_root = resolve_project_root()
    print(f"[INFO] PROJECT_ROOT resolved to: {project_root}")

    # Set up output directory
    output_dir = project_root / "Results_before_after_training" / "phase6_generations" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving generalization images to: {output_dir}")

    # Find LoRA weights
    lora_dir_name = "Flux_lora" if args.model == "flux" else "SDXL_lora" if args.model == "sdxl" else "Z-Image_lora"
    model_dir_name = "flux" if args.model == "flux" else "sdxl" if args.model == "sdxl" else "zimage"
    
    possible_lora_paths = [
        project_root / "data" / "id10" / model_dir_name / "models" / lora_dir_name / f"epoch-{args.epoch}.safetensors",
        Path(f"/home/project_id_10/DiffSynth-Studio/data/id10/{model_dir_name}/models/{lora_dir_name}/epoch-{args.epoch}.safetensors"),
        Path(f"./data/id10/{model_dir_name}/models/{lora_dir_name}/epoch-{args.epoch}.safetensors")
    ]

    lora_path = None
    for path in possible_lora_paths:
        if path.exists():
            lora_path = path
            break

    if lora_path is None:
        print(f"[ERROR] Could not find LoRA weights file for {args.model} epoch {args.epoch} in any expected locations.")
        sys.exit(1)

    print(f"[INFO] Loaded LoRA weights: {lora_path.resolve()}")

    # Initialize selected pipeline
    print(f"\n=== Initializing {args.model.upper()} Pipeline ===")
    
    if args.model == "flux":
        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=args.device,
            model_configs=[
                ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
                ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
                ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
                ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
            ]
        )
        pipe.load_lora(pipe.dit, str(lora_path), alpha=1)

    elif args.model == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            torch_dtype=torch.float16,
            device=args.device,
            model_configs=[
                ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="sd_xl_base_1.0.safetensors"),
                ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="sd_xl_base_1.0_vae.safetensors")
            ]
        )
        pipe.load_lora(pipe.unet, str(lora_path), alpha=1)

    elif args.model == "zimage":
        # Resolve tokenizer path
        local_tok_paths = [
            project_root / "models" / "Tongyi-MAI" / "Z-Image-Turbo" / "tokenizer",
            Path("/home/project_id_10/DiffSynth-Studio/models/Tongyi-MAI/Z-Image-Turbo/tokenizer"),
            Path("./models/Tongyi-MAI/Z-Image-Turbo/tokenizer")
        ]
        tokenizer_config = None
        for tok_path in local_tok_paths:
            if tok_path.exists():
                tokenizer_config = ModelConfig(str(tok_path))
                break
        if tokenizer_config is None:
            tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/")

        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=args.device,
            model_configs=[
                ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors"),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=tokenizer_config,
        )
        pipe.load_lora(module=pipe.dit, lora_config=str(lora_path))

    # Run inference for each test prompt
    print("\n=== Generating Generalization Demo Images ===")
    for idx, prompt in enumerate(GENERALIZATION_PROMPTS):
        img_id = idx + 1
        img_filename = f"gen_test_{img_id}.png"
        txt_filename = f"gen_test_{img_id}.txt"
        
        img_filepath = output_dir / img_filename
        txt_filepath = output_dir / txt_filename
        
        print(f"[{img_id}/10] Generating: {img_filename}")
        
        # Write prompt to .txt sidecar file
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(prompt + "\n")
            
        # Run model inference
        if args.model == "flux":
            image = pipe(prompt=prompt, num_inference_steps=args.steps, width=1024, height=1024)
        elif args.model == "sdxl":
            image = pipe(prompt=prompt, num_inference_steps=args.steps, width=1024, height=1024)
        elif args.model == "zimage":
            image = pipe(prompt=prompt, num_inference_steps=args.steps, width=1024, height=1024)
            
        image.save(str(img_filepath))
        
    print(f"\n[SUCCESS] Completed Phase 6 inference for {args.model.upper()}. Outputs saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
