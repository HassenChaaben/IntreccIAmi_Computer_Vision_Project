#!/usr/bin/env python3
"""
=============================================================================
Phase 3 - Task 3.3.3: SDXL Caption Engine
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Generate captions for SDXL fine-tuning using Qwen3.5:27b via Ollama,
         adhering strictly to length limits (30-50 words) and tag-based styling.
=============================================================================
"""

import os
import json
import argparse
import shutil
import csv
from pathlib import Path

# Common prompt building logic matching reference instructions
def build_prompt_metadata(item):
    technique = item.get("technique") or "N/A"
    weave_types = ", ".join(item.get("weave_types") or []) or "N/A"
    
    finish_obj = item.get("finish") or {}
    finish_type = finish_obj.get("type") or "N/A"
    paint_color = ", ".join(finish_obj.get("paint_color") or []) or "None"
    ral_pantone = finish_obj.get("ral_pantone") or "N/A"
    finish_str = f"Type: {finish_type}, Paint Color: {paint_color}, RAL/Pantone: {ral_pantone}"
    
    # Format posts (vertical)
    posts_list = []
    for p in item.get("posts") or []:
        size = p.get("size") or {}
        dist = p.get("distance") or {}
        p_str = (
            f"Index {p.get('index')}: Material: {p.get('material_leaf')}, "
            f"Size: {size.get('value')} {size.get('unit')}, "
            f"Quantity: {p.get('quantity')}, "
            f"Distance: {dist.get('value')} {dist.get('unit')}, "
            f"Colors: {', '.join(p.get('colors') or [])}"
        )
        posts_list.append(p_str)
    posts_str = "\n  ".join(posts_list) if posts_list else "None"
    
    # Format wefts (horizontal)
    wefts_list = []
    for w in item.get("wefts") or []:
        size = w.get("size") or {}
        dist = w.get("distance") or {}
        w_str = (
            f"Index {w.get('index')}: Material: {w.get('material_leaf')}, "
            f"Size: {size.get('value')} {size.get('unit')}, "
            f"Distance: {dist.get('value')} {dist.get('unit')}, "
            f"Colors: {', '.join(w.get('colors') or [])}"
        )
        wefts_list.append(w_str)
    wefts_str = "\n  ".join(wefts_list) if wefts_list else "None"
    
    special_desc = item.get("special_description") or "None"
    
    return technique, weave_types, finish_str, posts_str, wefts_str, special_desc


def build_sdxl_prompt(item):
    technique, weave_types, finish_str, posts_str, wefts_str, special_desc = build_prompt_metadata(item)
    
    prompt = f"""You are an expert captioner for handcrafted and industrial woven texture datasets.
You will receive one reference image and structured metadata extracted from Label Studio.
Your task is to generate one high-quality English caption for diffusion-model LoRA training.

Rules:
1. Describe only what is visible or strongly supported by the metadata.
2. Do not invent materials, colors, objects, decorations, or product categories.
3. Emphasize weave construction: posts, wefts, over-under crossings, density, spacing, repeat pattern, and material thickness.
4. Keep the structure physically manufacturable: simple to medium complexity, repeatable module, plausible strand widths, no impossible micro-interlocks.
5. Avoid internal codes unless visually meaningful. Do not mention Label Studio, JSON, bbox, or file names.
6. Write a single caption, 30-50 words, in natural English.

Format Constraint:
- Start the caption with the trigger word 'intrecciami-style'.
- Follow with a series of comma-separated tags describing technique, weave types, finish, and materials/colors.
- Conclude with a single short, descriptive sentence of the overall pattern.

Metadata:
- technique: {technique}
- weave_types: {weave_types}
- finish: {finish_str}
- posts: {posts_str}
- wefts: {wefts_str}
- special_description: {special_desc}

Caption:"""
    return prompt


def generate_mock_sdxl_caption(item):
    # Rule-based fallback to guarantee word count limit (30-50 words) and exact structure
    technique = item.get("technique") or "Intreccio"
    
    # Extract colors and materials
    post_materials = []
    post_colors = []
    for p in item.get("posts") or []:
        if p.get("material_leaf"):
            post_materials.append(str(p.get("material_leaf")))
        post_colors.extend(p.get("colors") or [])
        
    weft_materials = []
    weft_colors = []
    for w in item.get("wefts") or []:
        if w.get("material_leaf"):
            weft_materials.append(str(w.get("material_leaf")))
        weft_colors.extend(w.get("colors") or [])
        
    mats = list(set(post_materials + weft_materials))
    material = mats[0] if mats else "rattan"
    if isinstance(material, list):
        material = material[-1] if material else "rattan"
    
    colors = list(set(post_colors + weft_colors))
    color_name = colors[0].lower() if colors else "natural"
    
    desc = item.get("special_description") or ""
    desc_clean = f", {desc.lower()}" if desc else ""

    caption = (
        f"intrecciami-style, traditional {technique.lower()} technique, {material} material, {color_name} color, "
        f"even spacing{desc_clean}. The sample shows vertical posts crossed by horizontal weft strands in a simple "
        f"repeatable over-under layout with slight natural variations."
    )
    
    # Adjust length if needed to keep within 30-50 words
    words = caption.split()
    if len(words) < 30:
        words.append("The resulting modular lattice grid represents high quality craftsmanship.")
    elif len(words) > 50:
        words = words[:45]
        words.append("in a clean studio layout.")
        
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="SDXL Caption Engine using Qwen3.5:27b via Ollama")
    parser.add_argument("--metadata", type=str, default="data/normalized_metadata.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="data/id10/sdxl")
    parser.add_argument("--use_mock", action="store_true", help="Force mock captioning without running Ollama")
    args = parser.parse_args()

    # Load normalized tasks
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} not found!")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks for SDXL captioning.")

    # Setup directories
    output_dir = Path(args.output_dir)
    images_dest_dir = output_dir / "images"
    captions_dest_dir = output_dir / "captions"
    images_dest_dir.mkdir(parents=True, exist_ok=True)
    captions_dest_dir.mkdir(parents=True, exist_ok=True)

    image_dir_path = Path(args.image_dir)
    captioned_results = []
    qa_report_rows = []

    for task in tasks:
        task_id = task.get("task_id")
        img_info = task.get("image") or {}
        filename = img_info.get("filename")
        src_image_path = image_dir_path / filename

        if not src_image_path.exists():
            print(f"Warning: Source image {src_image_path} does not exist. Skipping task {task_id}.")
            continue

        # Target paths for resume capability
        stem = Path(filename).stem
        sidecar_caption_path = captions_dest_dir / f"{stem}.txt"
        sidecar_image_caption_path = images_dest_dir / f"{stem}.txt"
        dest_image_path = images_dest_dir / filename

        # 1. Resume Checkpoint Capability
        caption = None
        if sidecar_caption_path.exists():
            with open(sidecar_caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        elif sidecar_image_caption_path.exists():
            with open(sidecar_image_caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

        # 2. Run Inference if checkpoint not found
        if not caption:
            prompt = build_sdxl_prompt(task)
            if not args.use_mock:
                try:
                    import ollama
                    # Call local Ollama Qwen3.5:27b model
                    response = ollama.chat(
                        model="qwen3.5:27b",
                        messages=[{
                            "role": "user",
                            "content": prompt,
                            "images": [str(src_image_path)]
                        }]
                    )
                    caption = response["message"]["content"].strip()
                except Exception as e:
                    print(f"Ollama inference failed for task {task_id}: {e}. Falling back to mock caption.")
                    caption = generate_mock_sdxl_caption(task)
            else:
                caption = generate_mock_sdxl_caption(task)

            # Ensure image is copied to target folder
            if not dest_image_path.exists():
                shutil.copy2(src_image_path, dest_image_path)

            # Save checkpoints in both formats
            with open(sidecar_caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
            with open(sidecar_image_caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
        else:
            # If resume is triggered, ensure image is copied if missing
            if not dest_image_path.exists():
                shutil.copy2(src_image_path, dest_image_path)

        # Word count constraint QA check (30-50 words)
        word_count = len(caption.split())
        warnings = []
        if word_count < 30 or word_count > 50:
            warnings.append(f"Word count ({word_count}) is outside the 30-50 limit")
        
        # Metadata checks
        if not task.get("technique"):
            warnings.append("Missing 'technique' metadata")
        if not task.get("posts") and not task.get("wefts"):
            warnings.append("Both 'posts' and 'wefts' arrays are empty")
            
        warning_str = "; ".join(warnings)

        qa_report_rows.append({
            "task_id": task_id,
            "filename": filename,
            "word_count": word_count,
            "warning": warning_str
        })

        # Structured schema representation
        captioned_record = {
            "image": f"images/{filename}",
            "caption": caption,
            "metadata": {
                "task_id": task_id,
                "technique": task.get("technique"),
                "weave_types": task.get("weave_types"),
                "finish": task.get("finish"),
                "posts": task.get("posts"),
                "wefts": task.get("wefts"),
                "special_description": task.get("special_description")
            }
        }
        captioned_results.append(captioned_record)

    # Save outputs
    output_jsonl_path = output_dir / "metadata_sdxl.jsonl"
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for record in captioned_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_csv_path = output_dir / "qa_report_sdxl.csv"
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "filename", "word_count", "warning"])
        writer.writeheader()
        writer.writerows(qa_report_rows)

    print(f"[OK] SDXL engine completed. Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
