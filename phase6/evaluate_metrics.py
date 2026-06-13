#!/usr/bin/env python3
"""
=============================================================================
Phase 6 - Automated Evaluation Metrics (CLIPScore, LPIPS, CLIP-IQA)
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Computes real-time quantitative metrics for generated LoRA outputs.
         Runs CLIPScore, LPIPS, and CLIP-IQA using PyTorch and Hugging Face.
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Compute Automated Evaluation Metrics (CLIPScore, LPIPS, CLIP-IQA)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing generated images and matching .txt prompts")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="Optional directory containing ground truth reference textures for LPIPS distance")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        sys.exit(1)

    print(f"[INFO] Initializing PyTorch device: {args.device}")
    device = torch_device = args.device if args.device == "cuda" else "cpu"

    # Stage 1: Load CLIP Model
    print("[INFO] Loading CLIP model (ViT-B/32)...")
    try:
        import torch
        import clip
    except ImportError:
        print("[ERROR] PyTorch and openai-clip are required. Run: pip install ftfy regex tqdm && pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    device = torch.device(torch_device)
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load CLIP model: {e}")
        sys.exit(1)

    # Stage 2: Load LPIPS Model if ref_dir is provided
    lpips_model = None
    ref_images = []
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
        if ref_dir.exists():
            print(f"[INFO] Reference directory found: {ref_dir}. Loading LPIPS model...")
            try:
                import lpips
                lpips_model = lpips.LPIPS(net='alex').to(device)
                # Load up to 20 reference images for comparison speed
                ref_paths = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg"))
                for rp in ref_paths[:20]:
                    ref_images.append(lpips.im2tensor(lpips.load_image(str(rp))).to(device))
                print(f"[INFO] Loaded {len(ref_images)} reference images for style similarity checks.")
            except ImportError:
                print("[WARNING] lpips library not installed. LPIPS calculation will be skipped. Run: pip install lpips")
        else:
            print(f"[WARNING] Reference directory not found: {ref_dir}. LPIPS will be skipped.")

    # Find generated images and their sidecar prompts
    image_paths = sorted(list(image_dir.glob("*.png")))
    if not image_paths:
        print(f"[ERROR] No PNG files found in {image_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images to evaluate.")

    clips_scores = []
    lpips_scores = []
    iqas_scores = []
    results = []

    for img_path in image_paths:
        # Resolve prompt
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"[WARNING] Missing prompt sidecar file: {txt_path}. Skipping image.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"-> Processing: {img_path.name}")
        
        # 1. Load image and compute features
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_input = preprocess(pil_img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"   [ERROR] Failed to load image: {e}")
            continue

        # 2. Compute CLIPScore
        try:
            text_tokens = clip.tokenize([prompt], truncate=True).to(device)
            with torch.no_grad():
                img_features = model.encode_image(img_input)
                text_features = model.encode_text(text_tokens)
                # Normalize
                img_features /= img_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                clip_score = (img_features * text_features).sum(dim=-1).item()
            clips_scores.append(clip_score)
        except Exception as e:
            print(f"   [ERROR] CLIPScore calculation failed: {e}")
            clip_score = 0.0

        # 3. Compute CLIP-IQA (Good vs Bad photo probabilities)
        try:
            quality_prompts = ["a good quality high resolution sharp photo", "a bad quality blurry low resolution noisy photo"]
            quality_tokens = clip.tokenize(quality_prompts).to(device)
            with torch.no_grad():
                logits_per_image, _ = model(img_input, quality_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            iqa_score = float(probs[0][0])
            iqas_scores.append(iqa_score)
        except Exception as e:
            print(f"   [ERROR] CLIP-IQA calculation failed: {e}")
            iqa_score = 0.0

        # 4. Compute LPIPS style distance to nearest reference
        lpips_score = 0.0
        if lpips_model and ref_images:
            try:
                import torchvision.transforms as transforms
                # Resize generated to match lpips expectations
                gen_tensor = lpips.im2tensor(lpips.load_image(str(img_path))).to(device)
                dists = []
                with torch.no_grad():
                    for ref_tensor in ref_images:
                        dist = lpips_model(gen_tensor, ref_tensor)
                        dists.append(dist.item())
                # Minimum distance representing style proximity
                lpips_score = float(np.min(dists))
                lpips_scores.append(lpips_score)
            except Exception as e:
                print(f"   [ERROR] LPIPS calculation failed: {e}")

        results.append({
            "image": img_path.name,
            "prompt": prompt[:50] + "...",
            "clip_score": clip_score,
            "lpips_score": lpips_score if lpips_model else "N/A",
            "clip_iqa": iqa_score
        })

    # Print summary & write reports
    mean_clip = np.mean(clips_scores) if clips_scores else 0.0
    mean_iqa = np.mean(iqas_scores) if iqas_scores else 0.0
    mean_lpips = np.mean(lpips_scores) if lpips_scores else 0.0

    print("\n" + "="*80)
    print("                      EVALUATION METRICS CALCULATION COMPLETE")
    print("="*80)
    print(f"Image Directory: {image_dir}")
    print(f"CLIPScore (Mean): {mean_clip:.4f} (std: {np.std(clips_scores):.4f} if clips_scores else 0.0)")
    print(f"CLIP-IQA  (Mean): {mean_iqa:.4f}")
    if lpips_model:
        print(f"LPIPS     (Mean): {mean_lpips:.4f}")
    print("="*80)

    # Save CSV Report
    csv_path = image_dir / "quantitative_scoring_report.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,clip_score,lpips_score,clip_iqa\n")
        for res in results:
            f.write(f"{res['image']},{res['clip_score']},{res['lpips_score']},{res['clip_iqa']}\n")
    print(f"[INFO] Detailed CSV results saved to: {csv_path}")

    # Save Markdown Summary
    md_path = image_dir / "automated_metrics_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Automated Metrics Evaluation Report\n\n")
        f.write(f"* **Source Image Directory**: `{image_dir.name}/`\n")
        f.write(f"* **Total Evaluated Images**: {len(results)}\n\n")
        f.write(f"## Summary Statistics\n\n")
        f.write(f"| Metric | Mean Score | Standard Deviation |\n")
        f.write(f"| :--- | :---: | :---: |\n")
        f.write(f"| **CLIPScore** (Text Alignment) | {mean_clip:.4f} | {np.std(clips_scores):.4f} |\n")
        f.write(f"| **CLIP-IQA** (Aesthetic Quality) | {mean_iqa:.4f} | {np.std(iqas_scores):.4f} |\n")
        if lpips_model:
            f.write(f"| **LPIPS** (Style Distance to Real) | {mean_lpips:.4f} | {np.std(lpips_scores):.4f} |\n")
        f.write("\n\n## Per-Image Results\n\n")
        f.write("| Image | CLIPScore | LPIPS | CLIP-IQA |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for res in results:
            f.write(f"| {res['image']} | {res['clip_score']:.4f} | {res['lpips_score']} | {res['clip_iqa']:.4f} |\n")
            
    print(f"[INFO] Markdown summary report saved to: {md_path}")

if __name__ == "__main__":
    main()
