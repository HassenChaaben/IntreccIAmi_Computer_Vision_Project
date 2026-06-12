#!/usr/bin/env python3
import os
import json
import csv

def main():
    base_dir = "/home/project_id_10/DiffSynth-Studio/data/id10"
    models = ["zimage", "flux", "sdxl"]
    
    for model in models:
        jsonl_path = os.path.join(base_dir, model, f"metadata_{model}.jsonl")
        csv_path = os.path.join(base_dir, model, f"captions_{model}.csv")
        
        if not os.path.exists(jsonl_path):
            print(f"Skipping model {model} (JSONL metadata file not found at: {jsonl_path})")
            continue
            
        print(f"Processing model: {model}...")
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Error: JSON decode error at line {idx} in {model}: {e}")
                    continue
                
                image_path = data.get("image", "")
                # Gracefully check for 'caption', fallback to 'prompt' or 'text'
                caption = data.get("caption", data.get("prompt", data.get("text", "")))
                
                if not caption:
                    print(f"Warning: Line {idx} in metadata_{model}.jsonl has no caption/prompt/text field. Keys: {list(data.keys())}")
                    continue
                
                rows.append({"image_path": image_path, "caption": caption})
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        print(f"Writing CSV: {csv_path}...")
        with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["image_path", "caption"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
                
    print("CSV generation completed!")

if __name__ == "__main__":
    main()
