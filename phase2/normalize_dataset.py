#!/usr/bin/env python3
"""
=============================================================================
Phase 2 - Task 2.3: Field Normalization & Parsing Script
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Read the Label Studio JSON export, extract normalized metadata for
         each task, perform integrity checks against the image directory,
         and save the output to data/normalized_metadata.jsonl and 
         data/qa_report.csv.

This script implements the exact logic specified in the project instructions
(Section 7), adapting it to properly handle filename edge cases (e.g.,
Label Studio replacing spaces with underscores).

Usage (from the project root):
    python phase2/normalize_dataset.py
=============================================================================
"""

import json
import csv
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = PROJECT_ROOT / "data" / "raw_json" / "label_studio_texture_labels.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "normalized_metadata.jsonl"
QA_REPORT = PROJECT_ROOT / "data" / "qa_report.csv"
# ---------------------------------------------------------------------------


def extract_filename(path_or_url: str) -> str:
    """Extract the filename from a Label Studio path/url."""
    raw_name = Path(path_or_url).name
    # Keep the hex strip so the pipeline doesn't break, but otherwise match instructions
    match = re.match(r'^[a-f0-9]{8}-(.+)$', raw_name)
    if match:
        return match.group(1)
    return raw_name


def find_actual_image_path(filename: str, img_dir: Path) -> Path:
    """
    Find the actual image file on disk. Handles cases where Label Studio
    replaced spaces with underscores in the filename.
    """
    direct_path = img_dir / filename
    if direct_path.exists():
        return direct_path
        
    # Robust fallback: iterate through directory and match ignoring spaces/underscores
    filename_normalized = filename.replace('_', '').replace(' ', '').lower()
    for f in img_dir.iterdir():
        if f.is_file() and f.name.replace('_', '').replace(' ', '').lower() == filename_normalized:
            return f
            
    return None


def get_first_annotation(task):
    """Retrieve the first annotation from a task."""
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    # Assuming first annotation is the completed one
    return annotations[0]


def normalize_task(task, img_dir: Path):
    """Parse a Label Studio task into the normalized JSON structure."""
    data = task.get("data", {})
    image_path = data.get("image") or data.get("texture_img") or ""
    filename = extract_filename(image_path)

    # Use the helper to find actual file to prevent breakage, but output what instructions expect
    actual_path = find_actual_image_path(filename, img_dir)
    if not actual_path:
        return None, f"Missing image: {filename}"

    ann = get_first_annotation(task)
    if ann is None:
        return None, "No annotations"

    flat = {}
    bbox = None
    for item in ann.get("result", []):
        field = item.get("from_name")
        typ = item.get("type")
        val = item.get("value", {})

        if typ == "choices":
            flat[field] = val.get("choices", [])
        elif typ == "taxonomy":
            flat[field] = val.get("taxonomy", [])
        elif typ == "number":
            flat[field] = val.get("number")
        elif typ == "textarea":
            texts = val.get("text", [])
            flat[field] = texts[0] if texts else None
        elif typ == "rectanglelabels":
            bbox = {
                "x": val.get("x"), "y": val.get("y"),
                "width": val.get("width"), "height": val.get("height"),
                "rotation": val.get("rotation", 0),
                "label": (val.get("rectanglelabels") or [None])[0],
                "original_width": item.get("original_width"),
                "original_height": item.get("original_height"),
            }

    # Build repeated components: poste_1_*, poste_2_*, trama_1_*, ...
    posts = []
    wefts = []
    for i in range(1, 10):
        mat = flat.get(f"poste_{i}_materiale")
        if mat:
            posts.append({
                "index": i,
                "material_path": mat,
                "material_leaf": mat[-1] if isinstance(mat, list) and mat else mat,
                "size": {
                    "value": flat.get(f"poste_{i}_dimensione_valore"),
                    "unit": (flat.get(f"poste_{i}_dimensione_unita") or [None])[0]
                },
                "quantity": flat.get(f"poste_{i}_quantita"),
                "distance": {
                    "value": flat.get(f"poste_{i}_distanza_valore"),
                    "unit": (flat.get(f"poste_{i}_distanza_unita") or [None])[0]
                },
                "colors": flat.get(f"poste_{i}_colore", [])
            })

        mat = flat.get(f"trama_{i}_materiale")
        if mat:
            wefts.append({
                "index": i,
                "material_path": mat,
                "material_leaf": mat[-1] if isinstance(mat, list) and mat else mat,
                "size": {
                    "value": flat.get(f"trama_{i}_dimensione_valore"),
                    "unit": (flat.get(f"trama_{i}_dimensione_unita") or [None])[0]
                },
                "distance": {
                    "value": flat.get(f"trama_{i}_distanza_valore"),
                    "unit": (flat.get(f"trama_{i}_distanza_unita") or [None])[0]
                },
                "colors": flat.get(f"trama_{i}_colore", [])
            })

    normalized_data = {
        "task_id": task.get("id"),
        "image": {"filename": filename, "path": str(img_dir / filename).replace("\\", "/")},
        "code": flat.get("codice_bottega"),
        "technique": (flat.get("tecnica_usata") or [None])[0],
        "weave_types": flat.get("tipologia_intreccio", []),
        "finish": {
            "type": (flat.get("finitura_type") or [None])[0],
            "paint_color": flat.get("verniciatura_colore", []),
            "ral_pantone": flat.get("ral_pantone"),
            "special": flat.get("catalogo_bottega", [])
        },
        "posts": posts,
        "wefts": wefts,
        "special_description": flat.get("oggetto_note") or flat.get("descrizioni_speciali"),
        "bbox": bbox,
    }
    
    return normalized_data, "OK"


def main():
    print("=" * 65)
    print("Phase 2 – Task 2.1-2.3: Metadata Extraction & Normalization")
    print("=" * 65)

    if not JSON_PATH.exists():
        print(f"ERROR: Raw JSON not found at {JSON_PATH}")
        return
        
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            tasks = json.load(f)
        except json.JSONDecodeError:
            print("ERROR: Failed to parse JSON file.")
            return

    if not isinstance(tasks, list):
        print("ERROR: Root of JSON file is not a list as expected.")
        return
        
    print(f"Successfully loaded JSON. Found {len(tasks)} tasks.")

    normalized_records = []
    qa_issues = []

    for task in tasks:
        task_id = task.get("id", "Unknown")
        item, status = normalize_task(task, IMAGE_DIR)
        
        if item is not None:
            normalized_records.append(item)
            # Check for anomalies to log
            if not item.get("technique"):
                qa_issues.append({"task_id": task_id, "issue": "Missing technique field"})
            if not item.get("weave_types"):
                qa_issues.append({"task_id": task_id, "issue": "Missing weave_types field"})
        else:
            qa_issues.append({"task_id": task_id, "issue": status})
            print(f"Warning: Task {task_id} failed: {status}")

    # Write normalized JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in normalized_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Write QA report CSV
    with open(QA_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "issue"])
        writer.writeheader()
        writer.writerows(qa_issues)

    print("\n" + "=" * 65)
    print(f"Processed {len(tasks)} tasks.")
    print(f"Saved {len(normalized_records)} valid records to {OUTPUT_JSONL}")
    print(f"Logged {len(qa_issues)} QA issues to {QA_REPORT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
