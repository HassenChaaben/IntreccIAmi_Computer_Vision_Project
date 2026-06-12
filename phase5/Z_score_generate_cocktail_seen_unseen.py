#!/usr/bin/env python3
"""
=============================================================================
Phase 5: Combined Z-Image Seen and Unseen Prompts Generation Script
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Generate before/after comparison images for both seen (10) and
         unseen (50) prompts using the trained Z-Image LoRA models.
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
from diffsynth.pipelines.z_image import ZImagePipeline

# 10 Seen Prompts (from training dataset)
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

# 50 Unseen Prompts (generalization test)
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


def resolve_project_root():
    """Resolve project root directory robustly."""
    curr = Path(__file__).resolve().parent
    # Check if we are inside phase5/
    if curr.name == "phase5" and (curr.parent / "data").exists():
        return curr.parent
    elif (curr / "data").exists():
        return curr
    else:
        # Fallback to CWD if it contains data
        cwd = Path.cwd()
        if (cwd / "data").exists():
            return cwd
        return curr.parent


def main():
    parser = argparse.ArgumentParser(description="Z-Image Combined Seen and Unseen Image Generation Script")
    parser.add_argument("--epoch", type=int, default=4, help="LoRA epoch to validate (default: 4)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--skip_before", action="store_true", help="Skip base model (before) generations")
    args = parser.parse_args()

    project_root = resolve_project_root()
    print(f"[INFO] PROJECT_ROOT resolved to: {project_root}")

    # output directory path
    output_dir = project_root / "Results_before_after_training" / f"Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_{args.epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving results to: {output_dir}")

    # Search for LoRA file
    possible_lora_paths = [
        project_root / "data" / "id10" / "zimage" / "models" / "Z-Image_lora" / f"epoch-{args.epoch}.safetensors",
        Path(f"/home/project_id_10/DiffSynth-Studio/data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors"),
        Path(f"./data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors"),
        Path(f"data/id10/zimage/models/Z-Image_lora/epoch-{args.epoch}.safetensors")
    ]
    
    lora_path = None
    for path in possible_lora_paths:
        if path.exists():
            lora_path = path
            break

    if lora_path is None:
        print(f"[ERROR] Could not find LoRA weights file for epoch {args.epoch} in any of the expected locations.")
        print("Expected paths checked:")
        for path in possible_lora_paths:
            print(f"  - {path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Using Z-Image LoRA weights: {lora_path.resolve()}")

    # Search for local tokenizer path
    local_tok_paths = [
        project_root / "models" / "Tongyi-MAI" / "Z-Image-Turbo" / "tokenizer",
        Path("/home/project_id_10/DiffSynth-Studio/models/Tongyi-MAI/Z-Image-Turbo/tokenizer"),
        Path("./models/Tongyi-MAI/Z-Image-Turbo/tokenizer"),
        Path("models/Tongyi-MAI/Z-Image-Turbo/tokenizer")
    ]
    
    tokenizer_config = None
    for tok_path in local_tok_paths:
        if tok_path.exists():
            tokenizer_config = ModelConfig(str(tok_path))
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

    # ==========================================
    # Phase 1: Generating BEFORE images (Base model)
    # ==========================================
    if not args.skip_before:
        print("\n=== Generating BEFORE images (Base model) ===")
        
        # 1. Seen Prompts (Before)
        print(f"\n--- Generating {len(SEEN_PROMPTS)} BEFORE images for SEEN prompts ---")
        for idx, prompt in enumerate(SEEN_PROMPTS, start=1):
            filename = f"before_seen_{idx}.png"
            filepath = output_dir / filename
            if filepath.exists():
                print(f"[Seen {idx}/{len(SEEN_PROMPTS)}] Skipping (exists): {filename}")
                continue
            print(f"[Seen {idx}/{len(SEEN_PROMPTS)}] Generating: {filename}")
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

        # 2. Unseen Prompts (Before)
        print(f"\n--- Generating {len(UNSEEN_PROMPTS)} BEFORE images for UNSEEN prompts ---")
        for idx, prompt in enumerate(UNSEEN_PROMPTS, start=1):
            filename = f"before_unseen_{idx}.png"
            filepath = output_dir / filename
            if filepath.exists():
                print(f"[Unseen {idx}/{len(UNSEEN_PROMPTS)}] Skipping (exists): {filename}")
                continue
            print(f"[Unseen {idx}/{len(UNSEEN_PROMPTS)}] Generating: {filename}")
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

    # ==========================================
    # Load LoRA Weights
    # ==========================================
    print(f"\n=== Loading LoRA weights onto transformer: {lora_path} ===")
    pipe.load_lora(module=pipe.dit, lora_config=str(lora_path))

    # ==========================================
    # Phase 2: Generating AFTER images (LoRA model)
    # ==========================================
    print("\n=== Generating AFTER images (Fine-tuned model) ===")

    # 1. Seen Prompts (After)
    print(f"\n--- Generating {len(SEEN_PROMPTS)} AFTER images for SEEN prompts ---")
    for idx, prompt in enumerate(SEEN_PROMPTS, start=1):
        filename = f"after_seen_{idx}.png"
        filepath = output_dir / filename
        if filepath.exists():
            print(f"[Seen {idx}/{len(SEEN_PROMPTS)}] Skipping (exists): {filename}")
            continue
        print(f"[Seen {idx}/{len(SEEN_PROMPTS)}] Generating: {filename}")
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

    # 2. Unseen Prompts (After)
    print(f"\n--- Generating {len(UNSEEN_PROMPTS)} AFTER images for UNSEEN prompts ---")
    for idx, prompt in enumerate(UNSEEN_PROMPTS, start=1):
        filename = f"after_unseen_{idx}.png"
        filepath = output_dir / filename
        if filepath.exists():
            print(f"[Unseen {idx}/{len(UNSEEN_PROMPTS)}] Skipping (exists): {filename}")
            continue
        print(f"[Unseen {idx}/{len(UNSEEN_PROMPTS)}] Generating: {filename}")
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

    print(f"\n[SUCCESS] Completed generation cocktail for seen and unseen prompts.")
    print(f"All outputs saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
