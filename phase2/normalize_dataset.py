#!/usr/bin/env python3
"""
=============================================================================
Phase 2 - Task 2.3: Field Normalization & Parsing Script
=============================================================================
Project: IntreccIAmi (ID 10)
Purpose: Normalize Label Studio annotations according to the exact structure
         provided in Section 7 of the project instructions.
=============================================================================
"""

import json
from pathlib import Path
import re

# Resolve paths relative to the project root for safety
PROJECT_ROOT = Path(__file__).resolve().parent.parent
json_path = PROJECT_ROOT / "data" / "raw_json" / "label_studio_texture_labels.json"
image_dir = PROJECT_ROOT / "data" / "images"
output_path = PROJECT_ROOT / "data" / "normalized_metadata.jsonl"

with open(json_path, "r", encoding="utf-8") as f:
    tasks = json.load(f)


def extract_filename(path_or_url: str) -> str:
    # Extract the base name from the path/url as expected by instructions
    raw_name = Path(path_or_url).name
    
    # [ADJUSTMENT] Label Studio prefixes uploaded files with an 8-character hex ID (e.g. 'b22f6e86-').
    # We strip this hex prefix so it matches the actual filenames in the images folder.
    match = re.match(r'^[a-f0-9]{8}-(.+)$', raw_name)
    filename = match.group(1) if match else raw_name
    
    # [ADJUSTMENT] Handle potential filename matching mismatches (spaces vs underscores)
    if not (image_dir / filename).exists():
        space_name = filename.replace('_', ' ')
        if (image_dir / space_name).exists():
            return space_name
            
        # Fallback to case-insensitive match ignoring spaces/underscores
        norm_name = filename.replace('_', '').replace(' ', '').lower()
        if image_dir.exists():
            for f in image_dir.iterdir():
                if f.is_file() and f.name.replace('_', '').replace(' ', '').lower() == norm_name:
                    return f.name
                    
    return filename


def get_first_annotation(task):
    annotations = task.get("annotations", [])
    if not annotations:
        return None
    # Use the first completed annotation, or adapt this logic if multiple annotators exist.
    return annotations[0]


def read_choice(value):
    return value.get("choices", []) if isinstance(value, dict) else []


def read_text(value):
    texts = value.get("text", []) if isinstance(value, dict) else []
    return texts[0] if texts else None


def normalize_task(task):
    data = task.get("data", {})
    image_path = data.get("image") or data.get("texture_img") or ""
    filename = extract_filename(image_path)

    ann = get_first_annotation(task)
    if ann is None:
        return None

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

    return {
        "task_id": task.get("id"),
        # Use relative paths for portability across different environments (local vs. GPU server)
        "image": {"filename": filename, "path": f"data/images/{filename}"},
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
        # [ADJUSTMENT] The instructions check 'oggetto_note', but the actual raw JSON export
        # stores annotated notes under 'descrizioni_speciali'. We check both to avoid discarding data.
        "special_description": flat.get("oggetto_note") or flat.get("descrizioni_speciali"),
        "bbox": bbox,
    }

normalized = []
for task in tasks:
    item = normalize_task(task)
    if item is None:
        continue
    # Verify image existence using the resolved absolute image_dir
    if not (image_dir / item["image"]["filename"]).exists():
        print("Missing image:", item["image"]["filename"])
    normalized.append(item)

with open(output_path, "w", encoding="utf-8") as f:
    for item in normalized:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
