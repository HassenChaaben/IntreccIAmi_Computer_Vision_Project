#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from pathlib import Path

# Add DiffSynth-Studio to python path so we can import from it
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir / "DiffSynth-Studio"))
from diffsynth import ModelConfig
from diffsynth.pipelines.flux_image import FluxImagePipeline

UNSEEN_PROMPTS = [
    # Intreccio semplice
    "intrecciami-style: A meticulously crafted woven square featuring an Intreccio semplice pattern with natural rattan posts and dark blue leather wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven texture showcasing the Intreccio semplice technique with white leather mignon strands. Verniciatura gloss finish. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Intreccio spina / spina salto 2
    "intrecciami-style: A meticulously crafted woven square featuring an intricate Intreccio spina salto 2 pattern. Natural rattan strands, dyed in a deep black hue, highlight the Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven diagonal weave pattern in Intreccio spina style, using natural split rattan posts and red leather wefts. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Pattern scacco / Jacquard
    "intrecciami-style: A meticulously crafted woven texture showcasing a Pattern scacco checkerboard pattern. White and dark blue leather strands are woven in a tight grid. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven scacco pattern featuring natural rattan and black leather, showing a raw Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Macramè / Vario
    "intrecciami-style: An intricate Macramè piece showcasing a Vario knot pattern, formed by thick black leather strands. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Macramè weave featuring square knots in white leather mignon. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Rinfilo / Uncinetto
    "intrecciami-style: An intricate Rinfilo weave pattern on Paglia di Vienna, featuring thick beige leather fettuccia and thin mignon. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Uncinetto texture, hand-woven from thick black leather strands in a repeating grid. Close-up studio photograph, premium texture, high resolution, macro photography."
]

def main():
    parser = argparse.ArgumentParser(description="FLUX Unseen Prompts LoRA Inference (Epoch 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--start_index", type=int, default=1, help="Prompt start index (1 to 10, default: 1)")
    parser.add_argument("--end_index", type=int, default=10, help="Prompt end index (1 to 10, default: 10)")
    args = parser.parse_args()

    # Output directory
    target_path = Path("/home/project_id_10/DiffSynth-Studio/data/id10/flux/models/Flux_lora/before-after_results")
    if not Path("/home/project_id_10").exists():
        output_dir = Path("./DiffSynth-Studio/data/id10/flux/models/Flux_lora/before-after_results")
    else:
        output_dir = target_path
    output_dir.mkdir(parents=True, exist_ok=True)

    possible_lora_paths = [
        Path("/home/project_id_10/DiffSynth-Studio/data/id10/flux/models/Flux_lora/epoch-1.safetensors"),
        Path("./data/id10/flux/models/Flux_lora/epoch-1.safetensors"),
        Path("data/id10/flux/models/Flux_lora/epoch-1.safetensors"),
    ]
    lora_path = None
    for path in possible_lora_paths:
        if path.exists():
            lora_path = path
            break

    if lora_path is None:
        print("[ERROR] Could not find FLUX LoRA weights file in any of the following locations:")
        for path in possible_lora_paths:
            print(f"  - {path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Using FLUX LoRA path: {lora_path.resolve()}")

    print("\n=== Initializing FLUX Pipeline ===")
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

    # Load LoRA weights immediately (epoch 1)
    print(f"\n=== Loading FLUX LoRA weights: {lora_path} ===")
    pipe.load_lora(pipe.dit, str(lora_path), alpha=1)

    start_idx = max(0, args.start_index - 1)
    end_idx = min(len(UNSEEN_PROMPTS), args.end_index)
    prompts_to_run = UNSEEN_PROMPTS[start_idx:end_idx]
    total_prompts = len(prompts_to_run)

    print(f"[INFO] Running FLUX inference on {total_prompts} unseen prompts (Index {start_idx+1} to {end_idx})")

    # Run Fine-tuned Model (after_unseen_{index})
    print("\n=== Generating Fine-tuned (LoRA) FLUX images ===")
    for offset, prompt in enumerate(prompts_to_run):
        actual_index = start_idx + offset + 1
        filename = f"after_unseen_{actual_index}.png"
        filepath = output_dir / filename
        if filepath.exists():
            print(f"[{offset+1}/{total_prompts}] Skipping fine-tuned (exists): {filename}")
            continue
        print(f"[{offset+1}/{total_prompts}] Generating fine-tuned: {filename}")
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            width=1024,
            height=1024
        )
        image.save(str(filepath))

    print(f"\n[SUCCESS] Completed FLUX generation. Outputs saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
