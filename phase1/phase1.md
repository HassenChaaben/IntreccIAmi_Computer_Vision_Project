# Phase 1: Environment & Data Preparation – Complete Documentation

> **Project:** IntreccIAmi (ID 10)  
> **Date:** 2026-06-04  
> **Phase Status:** ✅ Scripts created, ready to execute on GPU server  
> **Tasks Covered:** 1.1 (Directory Setup), 1.2 (Retrieve Source Materials)  
> **Task Skipped:** 1.3 (Environment Installation – to be run on the university GPU server)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Task 1.1 – Directory Setup](#2-task-11--directory-setup)
3. [Task 1.2 – Retrieve Source Materials](#3-task-12--retrieve-source-materials)
4. [Task 1.3 – Environment Installation (SKIPPED)](#4-task-13--environment-installation-skipped)
5. [Source Data Analysis](#5-source-data-analysis)
6. [File Inventory](#6-file-inventory)
7. [How to Run](#7-how-to-run)
8. [Known Issues & Notes](#8-known-issues--notes)

---

## 1. Overview

Phase 1 prepares the workspace and data for the IntreccIAmi pipeline. The goal is to:

- Create a standardised directory structure (`data/raw_json/`, `data/images/`)
- Copy the Label Studio JSON export and all texture images into that structure
- (On GPU server) Install Python dependencies, clone DiffSynth-Studio, etc.

### Files Created in This Phase

| File | Purpose |
|------|---------|
| `phase1/setup_directories.py` | Task 1.1 – Creates `data/`, `data/raw_json/`, `data/images/` |
| `phase1/copy_source_materials.py` | Task 1.2 – Copies JSON + images into workspace |
| `phase1/run_phase1.py` | Master runner – executes 1.1 + 1.2 in sequence |
| `phase1/phase1.md` | This documentation file |

---

## 2. Task 1.1 – Directory Setup

### What It Does

The script `setup_directories.py` creates the following directory tree under the project root:

```
project_root/
└── data/
    ├── raw_json/     ← for the Label Studio JSON export
    └── images/       ← for the original texture images
```

### Design Decisions

- **Idempotent:** The script checks if each directory already exists before creating it. Re-running it is safe.
- **Uses `pathlib`:** All path handling uses Python's `pathlib.Path` for cross-platform compatibility.
- **PROJECT_ROOT:** Automatically determined relative to the script location (`phase1/` → project root is one level up).

### Script: `setup_directories.py`

```python
# Key logic:
DIRECTORIES_TO_CREATE = [
    "data",               # Top-level data folder
    "data/raw_json",      # For the Label Studio JSON export file
    "data/images",        # For the TEXTURE DI INTRECCI images
]

for rel_dir in DIRECTORIES_TO_CREATE:
    dir_path = PROJECT_ROOT / rel_dir
    dir_path.mkdir(parents=True, exist_ok=True)
```

### Expected Output

```
==============================================================
Phase 1 – Task 1.1: Directory Setup
==============================================================
Project root: C:\...\mouhaymin

  [CREATED] C:\...\mouhaymin\data
  [CREATED] C:\...\mouhaymin\data\raw_json
  [CREATED] C:\...\mouhaymin\data\images

✓ Directory setup complete.
```

---

## 3. Task 1.2 – Retrieve Source Materials

### What It Does

The script `copy_source_materials.py` copies:

1. **JSON file:** `Dataset/label_studio_texture_labels.json` → `data/raw_json/`
2. **Image files:** All 183 files from `Dataset/TEXTURE DI INTRECCI/` → `data/images/`

### Design Decisions

- **Skip-if-exists:** Files already present at the destination (matching by name and size) are skipped. This makes the script re-runnable.
- **Uses `shutil.copy2`:** Preserves file metadata (timestamps, permissions).
- **Flat copy:** The source image directory is flat (no subdirectories), so we copy all files directly.

### Source Material Locations

| Source | Destination |
|--------|-------------|
| `Dataset/label_studio_texture_labels.json` (1.7 MB, 44,682 lines) | `data/raw_json/label_studio_texture_labels.json` |
| `Dataset/TEXTURE DI INTRECCI/` (183 image files, ~227 MB total) | `data/images/` |

### Script: `copy_source_materials.py`

```python
# Key logic:
def copy_file_if_needed(src: Path, dst_dir: Path) -> str:
    dst = dst_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "[SKIPPED]"   # Already present
    shutil.copy2(src, dst)   # Copy preserving metadata
    return "[COPIED]"
```

### Expected Output

```
=================================================================
Phase 1 – Task 1.2: Retrieve Source Materials (Copy)
=================================================================

--- Copying Label Studio JSON export ---
  [COPIED]  label_studio_texture_labels.json → data/raw_json

--- Copying image files ---
  Found 183 files in source folder.

  Copied:  183 files
  Skipped: 0 files (already present)

=================================================================
✓ Source materials are now in the standardised workspace.
  JSON:   data/raw_json/label_studio_texture_labels.json
  Images: data/images  (183 files)
=================================================================
```

---

## 4. Task 1.3 – Environment Installation (SKIPPED)

> **⚠ This task is intentionally skipped.** It will be performed manually on the university GPU server.

For reference, Task 1.3 requires:

- Setting up a Conda/Python virtual environment (Python 3.10 recommended)
- Installing data-processing dependencies (`python-docx`, `pandas`, etc.)
- Cloning the official DiffSynth-Studio repository
- Installing DiffSynth-Studio in editable mode (`pip install -e .`)

### Commands to Run on the GPU Server

```bash
# 1. Create and activate a Conda environment
conda create -n intrecciami python=3.10 -y
conda activate intrecciami

# 2. Install data processing dependencies
pip install python-docx pandas pillow tqdm

# 3. Clone DiffSynth-Studio
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio

# 4. Install in editable mode
pip install -e .

# 5. (Optional) Additional dependencies for evaluation
pip install clip-score lpips open-clip-torch
```

---

## 5. Source Data Analysis

This section documents what was discovered about the source data during Phase 1 research.

### 5.1 JSON File Structure

The file `label_studio_texture_labels.json` is a **Label Studio export** — a JSON array of 177 annotated tasks. Each task represents one image with its annotations.

#### Top-Level Task Structure

```json
{
    "id": 1,                          // Task ID (integer)
    "annotations": [ ... ],           // Array of annotations (usually 1)
    "file_upload": "b22f6e86-IMG_6865.jpg",  // Upload filename (prefixed with hash)
    "data": {
        "image": "/data/upload/1/b22f6e86-IMG_6865.jpg"  // Image path in Label Studio
    },
    "created_at": "2026-03-11T...",   // Timestamp
    "total_annotations": 1,           // Count of annotations
    ...
}
```

#### Annotation Structure

Each annotation contains a `result` array of labeled fields:

```json
{
    "id": 162,
    "result": [
        {
            "from_name": "tecnica_usata",    // Field name (Label Studio XML name)
            "type": "choices",               // Data type
            "value": { "choices": ["Intreccio"] }  // Actual value
        },
        ...
    ],
    "was_cancelled": false,
    ...
}
```

### 5.2 Complete Field Inventory

| Field Name (`from_name`) | Type | Occurrences (out of 177) | Description |
|--------------------------|------|--------------------------|-------------|
| `tecnica_usata` | choices | 177 (100%) | Technique used: Intreccio, Macramè, Uncinetto, Rinfilo, Cucitura |
| `tipologia_intreccio` | choices | 160 (90%) | Weave pattern type (14 distinct values) |
| `trama_1_materiale` | taxonomy | 170 (96%) | Weft #1 material (hierarchical) |
| `trama_1_dimensione_valore` | number | 169 (95%) | Weft #1 dimension (numeric value) |
| `trama_1_dimensione_unita` | choices | 168 (95%) | Weft #1 dimension unit (always "mm") |
| `poste_1_materiale` | taxonomy | 164 (93%) | Post #1 material (hierarchical) |
| `poste_1_dimensione_valore` | number | 164 (93%) | Post #1 dimension (numeric) |
| `poste_1_dimensione_unita` | choices | 161 (91%) | Post #1 dimension unit |
| `trama_1_colore` | choices | 153 (86%) | Weft #1 color |
| `poste_1_colore` | choices | 143 (81%) | Post #1 color |
| `poste_1_quantita` | choices | 142 (80%) | Post #1 quantity |
| `finitura_type` | choices | 123 (69%) | Finish type: "Grezzo (non verniciato)" or "Verniciatura" |
| `descrizioni_speciali` | textarea | 98 (55%) | Free-text special descriptions (Italian) |
| `poste_1_distanza_valore` | number | 96 (54%) | Post #1 spacing value |
| `poste_1_distanza_unita` | choices | 94 (53%) | Post #1 spacing unit |
| `oggetto_bbox` | rectanglelabels | 80 (45%) | Bounding box around texture region |
| `codice_bottega` | textarea | 51 (29%) | Workshop code (e.g., "BI-CAPL0019") |
| `trama_2_materiale` | taxonomy | 49 (28%) | Weft #2 material |
| `trama_2_dimensione_valore` | number | 49 | Weft #2 dimension |
| `trama_2_dimensione_unita` | choices | 49 | Weft #2 dimension unit |
| `trama_2_colore` | choices | 48 (27%) | Weft #2 color |
| `verniciatura_colore` | choices | 23 (13%) | Paint color (when finish = Verniciatura) |
| `ral_pantone_speciale` | choices | 16 (9%) | Special RAL/Pantone (always "Trasparente") |
| `trasparenza_percentuale` | number | 16 (9%) | Transparency percentage |
| `poste_2_materiale` | taxonomy | 12 (7%) | Post #2 material |
| `poste_2_dimensione_valore` | number | 12 | Post #2 dimension |
| `poste_2_dimensione_unita` | choices | 12 | Post #2 dimension unit |
| `trama_3_materiale` | taxonomy | 11 (6%) | Weft #3 material |
| `trama_3_dimensione_valore` | number | 11 | Weft #3 dimension |
| `trama_3_dimensione_unita` | choices | 11 | Weft #3 dimension unit |
| `trama_3_colore` | choices | 11 | Weft #3 color |
| `rinfilo_materiale_base` | taxonomy | 10 (6%) | Base material for Rinfilo technique |
| `poste_2_quantita` | choices | 10 | Post #2 quantity |
| `poste_2_colore` | choices | 9 (5%) | Post #2 color |
| `ral_pantone` | textarea | 4 (2%) | RAL/Pantone code (free text) |
| `poste_2_distanza_unita` | choices | 1 (<1%) | Post #2 spacing unit |
| `catalogo_bottega` | choices | 1 (<1%) | Workshop catalogue entry |

### 5.3 Technique Distribution

All 177 tasks have `tecnica_usata` populated. The techniques are:

| Technique | Meaning |
|-----------|---------|
| **Intreccio** | Weaving – the primary technique |
| **Macramè** | Knotting technique |
| **Uncinetto** | Crochet |
| **Rinfilo** | Re-threading / repair technique |
| **Cucitura** | Sewing / stitching |

### 5.4 Weave Types (tipologia_intreccio)

14 distinct weave types found across 160 tasks:

| Weave Type | Translation/Description |
|------------|------------------------|
| Intreccio semplice | Simple weave |
| Intreccio semplice con trama doppia | Simple weave with double weft |
| Intreccio semplice con trama tripla | Simple weave with triple weft |
| Intreccio spina salto 2 | Herringbone weave skip-2 |
| Intreccio spina salto 3 | Herringbone weave skip-3 |
| Cannage | Cane webbing |
| Giro a 2 | Round weave with 2 strands |
| Giro a 2 scontrato | Offset round weave with 2 strands |
| Giro a 3 | Round weave with 3 strands |
| Giro a 3 scontrato | Offset round weave with 3 strands |
| Jacquard | Jacquard pattern weave |
| Pattern a X | X-pattern |
| Pattern scacco | Checkerboard pattern |
| Vario | Various / mixed |

### 5.5 Material Taxonomy

Materials are stored hierarchically using Label Studio's taxonomy type. The hierarchy is:

```
├── Naturale (Natural)
│   ├── Canapa (Hemp)
│   ├── Corteccia di Giunco (Rush bark)
│   │   └── Paglia di Vienna (Vienna straw)
│   ├── Cotone (Cotton)
│   │   ├── Cordino cerato (Waxed cord)
│   │   └── Cordino ritorto (Twisted cord)
│   ├── Grano (Grain/Wheat)
│   ├── Juta (Jute) [wefts only]
│   ├── Pelle (Leather)
│   │   ├── Fettuccia (Ribbon/strip)
│   │   ├── Mignon (Small leather)
│   │   └── Strobel (Strobel leather)
│   ├── Raffia [wefts only]
│   │   └── Viscosa (Viscose raffia)
│   ├── Rattan
│   │   ├── Filo tondo (Round wire)
│   │   ├── Piattina (Flat strip)
│   │   ├── Spaccato (Split)
│   │   └── Spianato (Flattened)
│   └── Vimini (Wicker)
│       └── Spaccato (Split)
└── Sintetico (Synthetic)
    ├── Cordino Strass (Rhinestone cord)
    ├── Plastica (Plastic)
    └── Rafia Viscosa (Viscose raffia) [wefts only]
```

### 5.6 Finish Types

| Finish | Occurrences | Details |
|--------|-------------|---------|
| Grezzo (non verniciato) | Majority | Raw/unfinished |
| Verniciatura | Present | Painted – comes with `verniciatura_colore` |

When finish is "Verniciatura", the following additional fields may appear:
- `verniciatura_colore`: 13 possible colors (Arancio, Azzurro, Bianco, Fucsia, Giallo chiaro, Grigio, Marrone, Marrone chiaro, Nero, Rosa, Verde chiaro, Verde scuro, Viola)
- `ral_pantone_speciale`: Always "Trasparente" when present
- `trasparenza_percentuale`: Transparency percentage (numeric)
- `ral_pantone`: Free-text RAL/Pantone code

### 5.7 Bounding Boxes

80 out of 177 tasks include an `oggetto_bbox` field with:
- `x`, `y`: top-left corner (as percentage of image dimensions)
- `width`, `height`: box size (as percentage)
- `rotation`: rotation angle (usually 0)
- `rectanglelabels`: label (always "Texture")
- `original_width`, `original_height`: pixel dimensions of the source image

---

## 6. File Inventory

### 6.1 Images on Disk

**Total image files:** 183  
**Image formats:** `.jpg` (180 files), `.JPG` (1 file), `.jpeg` (2 files), `.png` (1 file — `1706781842036.png`)  
**Total size:** ~227 MB (from the zip file size)

### 6.2 JSON ↔ Image Matching

| Metric | Count |
|--------|-------|
| Tasks in JSON | 177 |
| Images on disk | 183 |
| Tasks matched to disk images | 174 |
| Tasks with filename mismatch (spaces → underscores) | 3 |
| Images on disk with no JSON task | 6 |

**3 mismatched filenames** (Label Studio replaced spaces with underscores in `file_upload`):

| In JSON (`file_upload`) | On Disk |
|-------------------------|---------|
| `...-IMG_0198_DA_SISTEMARE.jpg` | `IMG_0198 DA SISTEMARE.jpg` |
| `...-WhatsApp_Image_2026-04-01_at_10.35.26.jpeg` | `WhatsApp Image 2026-04-01 at 10.35.26.jpeg` |
| `...-WhatsApp_Image_2026-04-01_at_10.38.52.jpeg` | `WhatsApp Image 2026-04-01 at 10.38.52.jpeg` |

> **Resolution:** The parsing script in Phase 2 must handle this by also trying filenames with underscores replaced by spaces. All 177 images do exist on disk — there are **zero truly missing images**.

**6 images on disk with no corresponding JSON task:** These are images in the `TEXTURE DI INTRECCI` folder that were not annotated in Label Studio. They can be safely ignored for the pipeline.

---

## 7. How to Run

### Option A: Run the Master Script

```bash
cd <project_root>
python phase1/run_phase1.py
```

This will execute Task 1.1 and 1.2 in order.

### Option B: Run Tasks Individually

```bash
# Task 1.1 – Create directories
python phase1/setup_directories.py

# Task 1.2 – Copy source materials
python phase1/copy_source_materials.py
```

### Task 1.3 – On the GPU Server

```bash
# See Section 4 above for the full commands
conda create -n intrecciami python=3.10 -y
conda activate intrecciami
pip install python-docx pandas pillow tqdm
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

---

## 8. Known Issues & Notes

### 8.1 Filename Encoding Issue

Label Studio replaces spaces with underscores in the `file_upload` field. Three images are affected:
- `IMG_0198 DA SISTEMARE.jpg`
- `WhatsApp Image 2026-04-01 at 10.35.26.jpeg`
- `WhatsApp Image 2026-04-01 at 10.38.52.jpeg`

**Impact:** The Phase 2 parsing script must implement fallback matching (try the original name, then try replacing underscores with spaces).

### 8.2 Extracting Image Filename from JSON

The filename can be extracted from two places in each task:
1. `file_upload` field: Format is `<8-char-hex>-<original_filename>` (e.g., `b22f6e86-IMG_6865.jpg`)
2. `data.image` field: Format is `/data/upload/1/<8-char-hex>-<original_filename>`

The recommended approach (per Instructions §6, Step 3) is to extract just the filename part after the hex prefix.

### 8.3 Taxonomy Field Format

Material taxonomy values are stored as nested arrays, where each array represents a path in the hierarchy:

```json
"taxonomy": [
    ["Naturale"],                          // Level 1
    ["Naturale", "Rattan"],                // Level 2
    ["Naturale", "Rattan", "Spianato"]     // Level 3 (leaf)
]
```

The **leaf value** (last element of the deepest array) is the most specific material designation and should be used as the primary material name. The full path provides category context.

### 8.4 Multiple Annotations

All 177 tasks have exactly 1 annotation (`total_annotations: 1`), and none are cancelled. This simplifies parsing — we always take `annotations[0].result`.

### 8.5 Data Types Summary

| Label Studio Type | Python Value Access | Example |
|-------------------|--------------------| --------|
| `choices` | `value.choices` → `list[str]` | `["Intreccio"]` |
| `taxonomy` | `value.taxonomy` → `list[list[str]]` | `[["Naturale"], ["Naturale", "Rattan"]]` |
| `number` | `value.number` → `float` or `int` | `2.5` |
| `textarea` | `value.text` → `list[str]` | `["Nodo quadro"]` |
| `rectanglelabels` | `value.x, .y, .width, .height, .rotation, .rectanglelabels` | bbox coordinates |

### 8.6 Field Mapping for Phase 2

The Instructions document (§4.1) defines the following normalized field mapping:

| Label Studio Name | Normalized Name | Notes |
|-------------------|----------------|-------|
| `codice_bottega` | `code` | Workshop identification code |
| `tecnica_usata` | `technique` | Single choice from 5 techniques |
| `tipologia_intreccio` | `weave_types` | Can be multi-choice |
| `finitura_type` | `finish.type` | "Grezzo" or "Verniciatura" |
| `verniciatura_colore` | `finish.paint_color` | Only when finish=Verniciatura |
| `ral_pantone` | `finish.ral_pantone` | Free-text RAL/Pantone code |
| `catalogo_bottega` | `finish.special` | Workshop catalogue |
| `descrizioni_speciali` | `special_description` | Free-text notes (Italian) |
| `oggetto_bbox` | `bbox` | Bounding box coordinates |
| `poste_i_*` fields | `posts[]` array | Grouped by index (1, 2) |
| `trama_i_*` fields | `wefts[]` array | Grouped by index (1, 2, 3) |

---

*End of Phase 1 Documentation*
