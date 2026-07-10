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

SEEN_PROMPTS = [
    "intrecciami-style: A meticulously crafted woven square featuring an intricate Intreccio spina salto 2 pattern. The weave showcases a tight, symmetrical structure with double posts and wefts interlacing in a precise over-under sequence, creating a visually appealing grid-like texture. The natural rattan strands, flattened and evenly spaced, are uniformly dyed in a vibrant Verde (green) hue, highlighting the raw, unvarnished (Grezzo) finish. Encased in a light beige leather border with clean, stitched edges, the piece exemplifies artisanal craftsmanship. The weave's density and uniformity suggest a high level of skill and attention to detail, making it a striking example of traditional weaving techniques. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted woven panel showcasing the Intreccio semplice technique, featuring natural, split rattan posts and wefts. The posts are vertically aligned with a consistent 2.5 mm spacing, while the wefts are woven horizontally in an over-under pattern, creating a dense and uniform texture. The strands are uniformly 3 mm in size, maintaining a consistent thickness throughout. The material exhibits a rich, deca color green hue, enhanced by a glossy finish that highlights the natural grain and texture of the rattan. The weave pattern is repetitive and symmetrical, demonstrating skilled craftsmanship. The panel is presented in a close-up studio photograph, capturing the premium texture and high resolution of the macro photography.",
    "intrecciami-style: A meticulously crafted macramé piece showcasing a Vario weave pattern, characterized by a series of square knots (Nodo quadro) formed by black leather strands. The weave features a consistent over-under crossing technique, creating a grid-like structure with uniform spacing and density. The leather strands, approximately 9 mm in size, exhibit a natural, mignon finish, adding texture and depth to the design. The weave is tight and precise, highlighting the artisanal quality and attention to detail. The overall composition is visually striking, with the interplay of light and shadow accentuating the three-dimensional nature of the knots. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A handcrafted woven texture showcasing the Intreccio technique, featuring a harmonious blend of natural materials. The weave pattern alternates between simple Intreccio semplice and Intreccio semplice con trama doppia, creating a visually striking contrast. The posts are made of natural rush bark (Corteccia di Giunco), measuring 2.5 mm in size, spaced 13 mm apart, and maintaining a natural color. The wefts consist of two types: one set of natural rush bark matching the posts, and another set of orange-colored leather (Pelle Mignon), both woven in a consistent 2.5 mm size. The weave exhibits a balanced density with a clear over-under crossing pattern, resulting in a textured surface with a natural, unvarnished finish. The alternating bands of natural and orange wefts create a rhythmic, repeating pattern that adds depth and visual interest. The photograph captures the intricate details of the weave, highlighting the craftsmanship and texture, making it a premium, high-resolution macro image.",
    "intrecciami-style: A meticulously crafted Intreccio semplice weave showcases a harmonious blend of natural leather strands in white and light brown. The posts and wefts, both measuring 4 mm in size, are woven in a simple over-under pattern, creating a consistent and elegant texture. The white posts are spaced 9 mm apart, while the wefts alternate seamlessly between white and light brown, forming a visually appealing checkerboard-like design. The natural finish highlights the genuine leather texture, emphasizing the artisanal quality. The weave is dense yet flexible, with uniform spacing and a balanced repeat pattern, demonstrating skilled craftsmanship. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted woven texture showcasing the Intreccio semplice technique, featuring a clean and symmetrical pattern. The weave is composed of natural, mignon leather strands in a uniform white color, each measuring 4 mm in size. The posts are evenly spaced at 9 mm intervals, creating a consistent and structured grid. The wefts interlace over and under the posts in a simple yet elegant pattern, resulting in a dense and uniform texture. The material exhibits a smooth, slightly embossed finish, highlighting the natural grain of the leather. The overall construction is precise, with no visible gaps or irregularities, demonstrating exceptional craftsmanship. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted woven texture showcasing the Intreccio semplice technique. The weave features natural posts made from 6 mm thick juniper bark, spaced 13 mm apart, providing a sturdy foundation. The wefts, composed of natural grain material in light yellow and natural tones, are 9 mm wide and intricately interlaced over and under the posts in a repeating pattern. The weave exhibits a dense, uniform structure with consistent spacing, highlighting the raw, unvarnished finish that emphasizes the natural textures and colors of the materials. The craftsmanship is evident in the precise over-under crossings, creating a visually appealing and tactile surface. Close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted woven texture showcasing the Intreccio spina salto 2 technique, characterized by a double-thread interlacing pattern. The weave features natural bamboo strips with a rustic, unvarnished finish, maintaining their authentic earthy tones. The posts and wefts, both measuring 3 mm in thickness, are woven in a tight, consistent pattern with a 2-step skip, creating a visually intricate and durable structure. The natural bamboo material retains its organic texture and color, emphasizing a raw, artisanal aesthetic. The weave demonstrates a balanced density with uniform spacing, highlighting the skilled craftsmanship involved in its creation. , close-up studio photograph, premium texture, high resolution, macro photography.",
    "intrecciami-style: A meticulously crafted woven texture showcasing the Intreccio semplice con trama tripla technique. The weave features a harmonious blend of natural rattan strands, with posts consisting of dark blue and natural-colored round filaments (2 mm in size) spaced 17 mm apart. The wefts include a combination of 4 mm wide natural-colored strands interlaced with thinner 2 mm round filaments, creating a visually intricate and balanced pattern. The natural, unvarnished finish highlights the raw texture and earthy tones of the materials, resulting in a rustic yet sophisticated aesthetic. The weave demonstrates a consistent over-under crossing pattern, with a moderate density that allows for both structural integrity and visual appeal. The close-up studio photograph captures the premium texture in high resolution, emphasizing the macro details of the weave construction.",
    "intrecciami-style: A meticulously crafted Intreccio semplice weave showcases a harmonious blend of natural rattan materials. The posts are composed of round filaments, measuring 2 mm in diameter, evenly spaced and creating a sturdy foundation. The wefts are flat piattina strands, 4 mm wide, interlaced over and under the posts in a consistent, repeating pattern. The weave exhibits a dense, uniform structure with a natural, unvarnished finish, highlighting the raw texture and earthy tones of the rattan. The craftsmanship is evident in the precise over-under crossings, resulting in a visually appealing and tactile surface. Close-up studio photograph, premium texture, high resolution, macro photography."
]

def main():
    parser = argparse.ArgumentParser(description="Z-Image Seen Prompts Comparison Script")
    parser.add_argument("--epoch", type=int, default=0, help="LoRA epoch to validate (default: 0)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--start_index", type=int, default=1, help="Prompt start index (1 to 10, default: 1)")
    parser.add_argument("--end_index", type=int, default=10, help="Prompt end index (1 to 10, default: 10)")
    args = parser.parse_args()

    # Save exactly to the specified folder in container, with local fallback
    target_path = Path("/home/project_id_10/DiffSynth-Studio/data/id10/zimage/models/Z-Image_lora/before-after_results")
    if not Path("/home/project_id_10").exists():
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
    end_idx = min(len(SEEN_PROMPTS), args.end_index)

    prompts_to_run = SEEN_PROMPTS[start_idx:end_idx]
    total_prompts = len(prompts_to_run)
    print(f"[INFO] Running comparisons on {total_prompts} seen prompts (Index {start_idx+1} to {end_idx})")

    print("\n=== Phase 1: Generating BEFORE images (Base model) ===")
    for offset, prompt in enumerate(prompts_to_run):
        actual_index = start_idx + offset + 1
        filename = f"before_seen_{actual_index}.png"
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

    print(f"\n=== Loading LoRA weights onto transformer: {lora_path} ===")
    pipe.load_lora(module=pipe.dit, lora_config=str(lora_path))

    print("\n=== Phase 2: Generating AFTER images (Fine-tuned model) ===")
    for offset, prompt in enumerate(prompts_to_run):
        actual_index = start_idx + offset + 1
        filename = f"after_seen_{actual_index}.png"
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

    print(f"\n[SUCCESS] Completed generation for seen prompts. Outputs saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
