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
from diffsynth.pipelines.z_image import ZImagePipeline

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
    parser = argparse.ArgumentParser(description="Z-Image Unseen Prompts Comparison Script")
    parser.add_argument("--epoch", type=int, default=1, help="LoRA epoch to validate (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--start_index", type=int, default=1, help="Prompt start index (1 to 10, default: 1)")
    parser.add_argument("--end_index", type=int, default=10, help="Prompt end index (1 to 10, default: 10)")
    args = parser.parse_args()

    # Save exactly to the specified folder in container, with local fallback
    target_path = Path("/home/project_id_10/DiffSynth-Studio/data/id10/zimage/models/Z-Image_lora/before-after_results")
    if not Path("/home/project_id_10").exists():
        # Running locally on Windows
        output_dir = Path("./DiffSynth-Studio/data/id10/zimage/models/Z-Image_lora/before-after_results")
    else:
        output_dir = target_path
        
    output_dir.mkdir(parents=True, exist_ok=True)

    possible_lora_paths = [
        Path(f"/home/project_id_10/DiffSynth-Studio/data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors"),
        Path(f"./data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors"),
        Path(f"data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors"),
    ]
    
    lora_path = None
    for path in possible_lora_paths:
        if path.exists():
            lora_path = path
            break

    if lora_path is None:
        print("[ERROR] Could not find LoRA weights file in any of the following locations:")
        for path in possible_lora_paths:
            print(f"  - {path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Using LoRA path: {lora_path.resolve()}")

    local_tok_paths = [
        "/home/project_id_10/DiffSynth-Studio/models/Tongyi-MAI/Z-Image-Turbo/tokenizer",
        "./models/Tongyi-MAI/Z-Image-Turbo/tokenizer",
        "models/Tongyi-MAI/Z-Image-Turbo/tokenizer"
    ]
    tokenizer_config = None
    for tok_path in local_tok_paths:
        if os.path.exists(tok_path):
            tokenizer_config = ModelConfig(tok_path)
            print(f"[INFO] Using local tokenizer directory: {tok_path}")
            break
            
    if tokenizer_config is None:
        tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/")
        print("[INFO] Local tokenizer not found. Downloading via HF Hub/ModelScope...")

    print("\n=== Initializing Z-Image Pipeline ===")
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

    start_idx = max(0, args.start_index - 1)
    end_idx = min(len(UNSEEN_PROMPTS), args.end_index)

    prompts_to_run = UNSEEN_PROMPTS[start_idx:end_idx]
    total_prompts = len(prompts_to_run)
    print(f"[INFO] Running comparisons on {total_prompts} unseen prompts (Index {start_idx+1} to {end_idx})")

    print(f"\n=== Loading LoRA weights onto transformer: {lora_path} ===")
    pipe.load_lora(module=pipe.dit, lora_config=str(lora_path))

    print("\n=== Generating Fine-tuned model (LoRA) images ===")
    for offset, prompt in enumerate(prompts_to_run):
        actual_index = start_idx + offset + 1
        filename = f"after_unseen_{actual_index}.png"
        filepath = output_dir / filename
        
        if filepath.exists():
            print(f"[{offset+1}/{total_prompts}] Skipping (exists): {filename}")
            continue
            
        print(f"[{offset+1}/{total_prompts}] Generating: {filename}")
        image = pipe(
            prompt=prompt,
            seed=42,
            rand_device=args.device,
            num_inference_steps=50,
            cfg_scale=4.0,
            width=1024,
            height=1024
        )
        image.save(str(filepath))

    print(f"\n[SUCCESS] Completed generation for unseen prompts. Outputs saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
