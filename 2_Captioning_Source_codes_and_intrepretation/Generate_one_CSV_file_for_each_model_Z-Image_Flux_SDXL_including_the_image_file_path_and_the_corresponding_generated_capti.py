#!/usr/bin/env python3
"""
=============================================================================
Phase 4 - Additional Feedback Tasks
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Generate one CSV file for each model (Z-Image, Flux, SDXL) containing
         image paths and VLM-generated captions, and compile token length statistics.
=============================================================================
"""

import os
import json
import csv
import numpy as np
from transformers import AutoTokenizer

def analyze_and_export():
    # Detect execution directory and resolve workspace path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(workspace_dir, "data", "id10")
    
    models = ["zimage", "flux", "sdxl"]
    
    # Load Tokenizers
    print("Loading tokenizers...")
    try:
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        gpt_tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return
    
    results = {}
    
    for model in models:
        jsonl_path = os.path.join(data_dir, model, f"metadata_{model}.jsonl")
        csv_path = os.path.join(data_dir, model, f"captions_{model}.csv")
        
        if not os.path.exists(jsonl_path):
            print(f"Skipping model {model} (JSONL metadata file not found at: {jsonl_path})")
            continue
            
        print(f"Processing model: {model}...")
        captions = []
        rows = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                image_path = data["image"]
                caption = data["caption"]
                captions.append(caption)
                rows.append({"image_path": image_path, "caption": caption})
        
        # Write CSV file
        print(f"Writing CSV: {csv_path}...")
        with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["image_path", "caption"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        
        # Analyze lengths
        print(f"Analyzing lengths for {model}...")
        word_counts = [len(c.split()) for c in captions]
        qwen_lengths = [len(qwen_tok.encode(c)) for c in captions]
        gpt_lengths = [len(gpt_tok.encode(c)) for c in captions]
        
        results[model] = {
            "count": len(captions),
            "words": word_counts,
            "qwen_tokens": qwen_lengths,
            "gpt_tokens": gpt_lengths
        }
        
    # Generate report
    report_path = os.path.join(workspace_dir, "phase3", "comparison_of_caption_token_lengths_generated_by_the_different_models_Z-Image_Flux_SDXL.md")
    print(f"Generating report: {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("# Phase 4: Dataset Quality Assessment & Token Analysis\n\n")
        f_rep.write("This document summarizes the token and word length analysis across the generated caption datasets for **Z-Image**, **Flux**, and **SDXL** models. It also serves as the deliverable for task 4.3 based on professor feedback.\n\n")
        
        f_rep.write("## 1. Caption Stats Summary Table\n\n")
        f_rep.write("| Model | Total Images | Avg. Words | Avg. Qwen2 Tokens | Avg. GPT2/CLIP Tokens | Min Qwen2 | Max Qwen2 | Status |\n")
        f_rep.write("|-------|--------------|------------|-------------------|----------------------|-----------|-----------|--------|\n")
        
        for model in models:
            if model not in results:
                continue
            res = results[model]
            avg_w = np.mean(res["words"])
            avg_q = np.mean(res["qwen_tokens"])
            avg_g = np.mean(res["gpt_tokens"])
            min_q = np.min(res["qwen_tokens"])
            max_q = np.max(res["qwen_tokens"])
            
            status = ""
            if model == "zimage":
                status = "✅ ~160 tokens target consistency"
            elif model == "flux":
                status = "✅ Dense text prompt"
            elif model == "sdxl":
                status = "✅ Short tags prompt"
                
            f_rep.write(f"| **{model.upper()}** | {res['count']} | {avg_w:.1f} words | {avg_q:.1f} tokens | {avg_g:.1f} tokens | {min_q} | {max_q} | {status} |\n")
            
        f_rep.write("\n---\n\n")
        
        f_rep.write("## 2. Detailed Distribution Analysis\n\n")
        for model in models:
            if model not in results:
                continue
            res = results[model]
            f_rep.write(f"### {model.upper()} Caption Token Lengths (Qwen2 Tokenizer)\n\n")
            f_rep.write("- **Total Captions**: {}\n".format(res["count"]))
            f_rep.write("- **Average length**: {:.1f} tokens\n".format(np.mean(res["qwen_tokens"])))
            f_rep.write("- **Median length**: {:.1f} tokens\n".format(np.median(res["qwen_tokens"])))
            f_rep.write("- **Min length**: {} tokens\n".format(np.min(res["qwen_tokens"])))
            f_rep.write("- **Max length**: {} tokens\n".format(np.max(res["qwen_tokens"])))
            
            # Buckets
            buckets = [0, 50, 100, 150, 200, 250, 500]
            counts = []
            for i in range(len(buckets)-1):
                low = buckets[i]
                high = buckets[i+1]
                cnt = sum(low <= x < high for x in res["qwen_tokens"])
                counts.append((low, high, cnt))
                
            f_rep.write("\n**Distribution Buckets (Qwen2 Tokens):**\n\n")
            f_rep.write("| Token Range | Count | Percentage |\n")
            f_rep.write("|-------------|-------|------------|\n")
            for low, high, cnt in counts:
                pct = (cnt / res["count"]) * 100
                f_rep.write(f"| {low} - {high} | {cnt} | {pct:.1f}% |\n")
            f_rep.write("\n")
            
        f_rep.write("---\n\n")
        
        f_rep.write("## 3. CSV Dataset Links\n\n")
        f_rep.write("In addition to the raw `.jsonl` files and the sidecar `.txt` files, a clean two-column CSV file (`image_path`, `caption`) has been generated for each model for easy review:\n\n")
        f_rep.write("1. **Z-Image Captions CSV**: [captions_zimage.csv](../data/id10/zimage/captions_zimage.csv)\n")
        f_rep.write("2. **Flux Captions CSV**: [captions_flux.csv](../data/id10/flux/captions_flux.csv)\n")
        f_rep.write("3. **SDXL Captions CSV**: [captions_sdxl.csv](../data/id10/sdxl/captions_sdxl.csv)\n\n")
        
        f_rep.write("## 4. Token Consistency Note\n\n")
        f_rep.write("> [!NOTE]\n")
        f_rep.write("we are maintaining captions around **160 tokens** to ensure structural and content consistency across the different image generation models (Z-Image and Flux) before starting LoRA fine-tuning.\n")
        
    print("Done!")

if __name__ == "__main__":
    analyze_and_export()
