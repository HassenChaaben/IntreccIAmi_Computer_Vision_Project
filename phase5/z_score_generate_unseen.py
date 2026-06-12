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
    # Intreccio semplice (1-10)
    "intrecciami-style: A meticulously crafted woven square featuring an Intreccio semplice pattern with natural rattan posts and dark blue leather wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven texture showcasing the Intreccio semplice technique with white leather mignon strands. Verniciatura gloss finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A symmetrical woven panel employing the Intreccio semplice technique, with red split rattan posts and natural rush bark wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A tight basketweave pattern using the Intreccio semplice method, composed of thick black leather fettuccia. Verniciatura opaca finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Intreccio semplice weave featuring light brown leather mignon posts and white wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A modern woven pattern featuring fuchsia synthetic plastic posts and white leather wefts in an Intreccio semplice weave. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: An elegant Intreccio semplice texture made of natural midollino round filaments, dyed in a warm marrone chiaro shade. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A dense woven grid utilizing the Intreccio semplice technique, composed of gray leather strobel strands. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven checkerboard texture in Intreccio semplice style, using yellow rattan and black leather wefts. Verniciatura gloss finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A minimalist woven design showing the Intreccio semplice technique with natural rush bark posts and orange leather wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Intreccio spina / spina salto 2 (11-20)
    "intrecciami-style: A meticulously crafted woven square featuring an intricate Intreccio spina salto 2 pattern. Natural rattan strands, dyed in a deep black hue, highlight the Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven diagonal weave pattern in Intreccio spina style, using natural split rattan posts and red leather wefts. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: An intricate woven panel in Intreccio spina salto 2 pattern, showcasing double posts of blue leather and natural wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A diagonal herringbone-like weave employing the Intreccio spina technique, with white leather mignon and natural rush bark. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Intreccio spina salto 2 texture, with green midollino round filaments and dark brown wefts. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven panel featuring an Intreccio spina pattern with natural bamboo bark, highlighted by a gloss verniciatura. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A complex diagonal weave using the Intreccio spina salto 2 method, composed of gray synthetic plastic and black leather. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A beautiful Intreccio spina texture featuring natural rattan posts and red split rattan wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven square panel showing the Intreccio spina salto 2 technique with white leather fettuccia. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A dense diagonal weave in Intreccio spina style with orange leather and natural midollino strands. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Pattern scacco / Jacquard (21-30)
    "intrecciami-style: A meticulously crafted woven texture showcasing a Pattern scacco checkerboard pattern. White and dark blue leather strands are woven in a tight grid. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven scacco pattern featuring natural rattan and black leather, showing a raw Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A complex Jacquard weave pattern using fuchsia and white leather strands in a repeating geometric grid. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A checkerboard Pattern scacco weave consisting of natural rush bark and yellow leather. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A beautifully aligned Jacquard woven panel with red split rattan and natural midollino. Verniciatura gloss finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A minimalist Pattern scacco checkerboard weave made of gray leather mignon and white leather fettuccia. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A dense Jacquard woven grid featuring green midollino round filaments and natural rattan wefts. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven checkerboard Pattern scacco with black and white leather strobel strands. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: An intricate Jacquard weave showing alternating bands of orange leather and natural bamboo bark. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A modern Pattern scacco weave composed of blue synthetic plastic and white leather. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Macramè / Vario (31-40)
    "intrecciami-style: An intricate Macramè piece showcasing a Vario knot pattern, formed by thick black leather strands. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Macramè weave featuring square knots in white leather mignon. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-knotted Vario weave pattern employing natural rush bark cords and brown leather. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A complex Macramè panel showing vertical knots in red and black leather fettuccia. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A beautiful Vario knot pattern featuring natural rattan fibers and blue leather mignon. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-knotted Macramè grid using green leather strobel and natural midollino. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: An intricate Vario weave with square knots in yellow leather and white leather fettuccia. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A minimalist Macramè panel with natural bamboo bark strands in a Vario pattern. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-knotted Vario texture featuring orange leather and gray synthetic plastic. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A dense Macramè weave of white leather mignon and natural split rattan. Close-up studio photograph, premium texture, high resolution, macro photography.",

    # Rinfilo / Uncinetto (41-50)
    "intrecciami-style: An intricate Rinfilo weave pattern on Paglia di Vienna, featuring thick beige leather fettuccia and thin mignon. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Uncinetto texture, hand-woven from thick black leather strands in a repeating grid. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A complex Rinfilo design showing white leather mignon woven through Paglia di Vienna. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven Uncinetto panel featuring natural rattan fibers in a tight crochet pattern. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A beautiful Rinfilo weave pattern with red and black leather fettuccia on Paglia di Vienna. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: An Uncinetto crochet pattern made of green midollino round filaments. Verniciatura gloss finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A dense Rinfilo weave featuring yellow leather mignon and natural rush bark on Paglia di Vienna. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A hand-woven Uncinetto crochet texture in blue leather strobel. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A complex Rinfilo pattern showing orange leather fettuccia woven through Paglia di Vienna. Grezzo finish. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted Uncinetto crochet design featuring white leather mignon. Close-up studio photograph, premium texture, high resolution, macro photography."
]

def main():
    parser = argparse.ArgumentParser(description="Z-Image Unseen Prompts Comparison Script")
    parser.add_argument("--epoch", type=int, default=1, help="LoRA epoch to validate (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--start_index", type=int, default=1, help="Prompt start index (1 to 50, default: 1)")
    parser.add_argument("--end_index", type=int, default=50, help="Prompt end index (1 to 50, default: 50)")
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
