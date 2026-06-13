#!/usr/bin/env python3
"""
=============================================================================
Phase 6 - MLLM-as-a-Judge Evaluation Pipeline
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Automates qualitative scoring using an MLLM-as-a-judge approach
         by querying a Qwen-Vision model via Ollama for each generated image.
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="MLLM-as-a-Judge Evaluation Pipeline")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing generated images and matching .txt prompts")
    parser.add_argument("--model", type=str, default="qwen2.5-vision",
                        help="Ollama vision model name (default: qwen2.5-vision)")
    parser.add_argument("--host", type=str, default=None,
                        help="Ollama host address (optional)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        sys.exit(1)

    print("[INFO] Attempting to import ollama library...")
    try:
        import ollama
        # Initialize client
        client = ollama.Client(host=args.host) if args.host else ollama
    except ImportError:
        print("[ERROR] ollama python package not installed. Run: pip install ollama")
        sys.exit(1)

    image_paths = sorted(list(image_dir.glob("*.png")))
    if not image_paths:
        print(f"[ERROR] No PNG files found in {image_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images to judge using MLLM model: {args.model}")

    results = []
    adherence_list = []
    identity_list = []
    manuf_list = []
    quality_list = []
    originality_list = []

    system_instruction = (
        "You are a computer vision engineering quality judge evaluating image generation models. "
        "Your job is to rate the image on a scale of 0.0 to 5.0 (decimals allowed, e.g., 4.5) for each of these criteria:\n"
        "1. Prompt Adherence: Does the image contain all elements in the prompt?\n"
        "2. Intreccio Identity: Does it look like authentic woven leather/rattan or is it warped/melted?\n"
        "3. Manufacturability: Are the crossings physically possible? Would a craftsman be able to replicate it?\n"
        "4. Visual Quality: Sharpness, realistic lighting, aesthetic appeal.\n"
        "5. Controlled Originality: Style applied cleanly to the target object.\n"
        "Respond ONLY with a raw JSON block like this:\n"
        '{"prompt_adherence": 4.5, "intreccio_identity": 4.0, "manufacturability": 4.2, "visual_quality": 4.5, "controlled_originality": 4.0, "reasoning": "Brief explanation"}'
    )

    for img_path in image_paths:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"[WARNING] Missing prompt sidecar file: {txt_path}. Skipping.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"-> Judging: {img_path.name}")
        
        try:
            response = client.generate(
                model=args.model,
                prompt=f"Prompt: {prompt}\nEvaluate this image based on the system instructions.",
                images=[str(img_path)],
                system=system_instruction,
                options={"temperature": 0.0} # Low temperature for deterministic grading
            )
            
            resp_text = response['response'].strip()
            # Clean up response to get raw JSON if model included markdown blocks
            if resp_text.startswith("```json"):
                resp_text = resp_text[7:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            resp_text = resp_text.strip()
            
            data = json.loads(resp_text)
            
            adherence = float(data.get("prompt_adherence", 0.0))
            identity = float(data.get("intreccio_identity", 0.0))
            manuf = float(data.get("manufacturability", 0.0))
            quality = float(data.get("visual_quality", 0.0))
            orig = float(data.get("controlled_originality", 0.0))
            reasoning = data.get("reasoning", "")
            
            adherence_list.append(adherence)
            identity_list.append(identity)
            manuf_list.append(manuf)
            quality_list.append(quality)
            originality_list.append(orig)
            
            mean_score = (adherence + identity + manuf + quality + orig) / 5.0
            
            results.append({
                "image": img_path.name,
                "prompt": prompt[:50] + "...",
                "adherence": adherence,
                "identity": identity,
                "manufacturability": manuf,
                "quality": quality,
                "originality": orig,
                "mean_score": mean_score,
                "reasoning": reasoning
            })
            print(f"   [SCORES] Adherence: {adherence} | Identity: {identity} | Manuf: {manuf} | Quality: {quality} | Originality: {orig} | Mean: {mean_score:.2f}")

        except Exception as e:
            print(f"   [ERROR] Failed to judge image: {e}")

    # Compute averages
    if not results:
        print("[ERROR] No images successfully graded.")
        sys.exit(1)

    mean_adh = np.mean(adherence_list)
    mean_ident = np.mean(identity_list)
    mean_manuf = np.mean(manuf_list)
    mean_qual = np.mean(quality_list)
    mean_orig = np.mean(originality_list)
    total_mean = np.mean([res["mean_score"] for res in results])

    print("\n" + "="*80)
    print("                      MLLM-AS-A-JUDGE GRADING COMPLETE")
    print("="*80)
    print(f"Prompt Adherence:    {mean_adh:.2f} / 5.0")
    print(f"Intreccio Identity:  {mean_ident:.2f} / 5.0")
    print(f"Manufacturability:   {mean_manuf:.2f} / 5.0")
    print(f"Visual Quality:      {mean_qual:.2f} / 5.0")
    print(f"Controlled Orig:     {mean_orig:.2f} / 5.0")
    print(f"OVERALL MEAN SCORE:  {total_mean:.2f} / 5.0")
    print("="*80)

    # Save CSV Report
    csv_path = image_dir / "qualitative_scoring_report.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,prompt_adherence,intreccio_identity,manufacturability,visual_quality,controlled_originality,mean_score,reasoning\n")
        for res in results:
            clean_reasoning = res['reasoning'].replace('"', '""')
            f.write(f"{res['image']},{res['adherence']},{res['identity']},{res['manufacturability']},{res['quality']},{res['originality']},{res['mean_score']:.2f},\"{clean_reasoning}\"\n")
    print(f"[INFO] Detailed CSV scores saved to: {csv_path}")

    # Save Markdown Summary
    md_path = image_dir / "mllm_judge_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# MLLM-as-a-Judge Qualitative Evaluation Report\n\n")
        f.write(f"* **Source Image Directory**: `{image_dir.name}/`\n")
        f.write(f"* **MLLM Model**: `{args.model}`\n")
        f.write(f"* **Total Judged Images**: {len(results)}\n\n")
        f.write(f"## Summary Ratings (Average / 5.0)\n\n")
        f.write(f"| Evaluation Criteria | Average Rating / 5.0 | Standard Deviation |\n")
        f.write(f"| :--- | :---: | :---: |\n")
        f.write(f"| **Prompt Adherence** | {mean_adh:.2f} | {np.std(adherence_list):.2f} |\n")
        f.write(f"| **Intreccio Identity** | {mean_ident:.2f} | {np.std(identity_list):.2f} |\n")
        f.write(f"| **Manufacturability** | {mean_manuf:.2f} | {np.std(manuf_list):.2f} |\n")
        f.write(f"| **Visual Quality** | {mean_qual:.2f} | {np.std(quality_list):.2f} |\n")
        f.write(f"| **Controlled Originality** | {mean_orig:.2f} | {np.std(originality_list):.2f} |\n")
        f.write(f"| **OVERALL MEAN SCORE** | **{total_mean:.2f}** | **{np.std([res['mean_score'] for res in results]):.2f}** |\n")
        f.write("\n\n## Per-Image Detailed Ratings\n\n")
        f.write("| Image | Adherence | Identity | Manufacturability | Quality | Originality | Mean | Reasoning |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for res in results:
            f.write(f"| {res['image']} | {res['adherence']:.1f} | {res['identity']:.1f} | {res['manufacturability']:.1f} | {res['quality']:.1f} | {res['originality']:.1f} | {res['mean_score']:.2f} | {res['reasoning']} |\n")
            
    print(f"[INFO] Markdown judge report saved to: {md_path}")

if __name__ == "__main__":
    main()
