# Phase 2: Metadata Extraction & Normalization – Complete Documentation

> **Project:** IntreccIAmi (ID 10)  
> **Date:** 2026-06-04  
> **Phase Status:** ✅ Completed (Scripts created, verified, and run locally)  
> **Tasks Covered:** 2.1 (Raw JSON Parsing), 2.2 (Integrity Verification), 2.3 (Field Normalization & Parsing Script)  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Task 2.1 – Raw JSON Parsing](#2-task-21--raw-json-parsing)
3. [Task 2.2 – Integrity Verification](#3-task-22--integrity-verification)
4. [Task 2.3 – Field Normalization & Parsing Script](#4-task-23--field-normalization--parsing-script)
5. [How to Run](#5-how-to-run)
6. [Known Issues & Notes](#6-known-issues--notes)

---

## 1. Overview

Phase 2 focuses on interpreting the raw Label Studio JSON annotations, mapping them to a clean, unified JSON structure, checking that the metadata accurately maps to existing images on disk, and outputting the normalized metadata along with a QA report.

### Files Created in This Phase

| File | Purpose |
|------|---------|
| `phase2/normalize_dataset.py` | Tasks 2.1, 2.2, 2.3 – Parses JSON, verifies images, normalizes fields, writes outputs |
| `phase2/run_phase2.py` | Master runner – executes `normalize_dataset.py` |
| `phase2/phase2.md` | This documentation file |

---

## 2. Task 2.1 – Raw JSON Parsing

### What It Does
Loads `data/raw_json/label_studio_texture_labels.json` into memory and verifies its root structure is a list of tasks.

### Design Decisions
- Utilizes the `json` module for robust parsing.
- Handles `json.JSONDecodeError` to gracefully catch corrupt JSON.
- Verifies the root object is a list.

---

## 3. Task 2.2 – Integrity Verification

### What It Does
For each task:
1. Extracts the image path from `data.image` (or `texture_img` or `file_upload`).
2. Extracts the clean filename, stripping Label Studio's 8-character hex prefix.
3. Verifies the file actually exists in the `data/images/` directory.

### Design Decisions
- **Edge Case Handling:** As discovered in Phase 1, Label Studio sometimes replaces spaces with underscores in filenames (e.g. `IMG_0198 DA SISTEMARE.jpg` vs `IMG_0198_DA_SISTEMARE.jpg`). The script implements a fallback mechanism (`find_actual_image_path`) to check for the space-separated version if the underscored version is missing.
- **QA Reporting:** If an image is truly missing, the task fails and an entry is logged in `metadata_extraction_qa_report.csv`.

---

## 4. Task 2.3 – Field Normalization & Parsing Script

### What It Does
Processes the `annotations[].result` array for each valid task and maps the Label Studio XML field names to a cleaner, normalized JSON schema.

### Mapping Details

The script implements the mapping specified in Section 4.1 and Section 7 of the instructions:

- `codice_bottega` ➔ `code`
- `tecnica_usata` ➔ `technique`
- `tipologia_intreccio` ➔ `weave_types` (Array)
- `finitura_type` ➔ `finish.type`
- `verniciatura_colore` ➔ `finish.paint_color`
- `ral_pantone` ➔ `finish.ral_pantone`
- `catalogo_bottega` ➔ `finish.special`
- `oggetto_note` / `descrizioni_speciali` ➔ `special_description` (checks `oggetto_note` and falls back to `descrizioni_speciali`)
- `oggetto_bbox` ➔ `bbox` (JSON object with x, y, width, height, rotation, label, original_width, original_height)

**Repeated Components:**
The script dynamically searches for indexed fields (`poste_1_...` to `poste_9_...` and `trama_1_...` to `trama_9_...`) and constructs structured arrays for vertical elements (`posts`) and horizontal elements (`wefts`).

### Expected Output
1. **`data/normalized_metadata.jsonl`:** One valid JSON object per line.
2. **`data/metadata_extraction_qa_report.csv`:** A CSV listing `task_id` and `issue` for tasks missing key data or images.

---

## 5. How to Run

When ready to execute (e.g. on the GPU server):

```bash
cd <project_root>
python phase2/run_phase2.py
```

Or run the script directly:
```bash
python phase2/normalize_dataset.py
```

---

## 6. Known Issues & Notes

- **DOCX Instruction Synchronization:** The script has been rewritten to match the exact structure, functions (`read_choice`, `read_text`), and looping mechanics of Section 7 in `Istruzioni_ID10_IntreccIAmi_Qwen_DiffSynth_TRANSLATED_EN.docx`.
- **`descrizioni_speciali` vs `oggetto_note`:** The instruction's Section 7 pseudocode expects `oggetto_note`. However, the raw JSON export stores these free-text notes under the field name `descrizioni_speciali`. To strictly adhere to the guidelines while remaining fully functional, the normalization script checks for `oggetto_note` and falls back to `descrizioni_speciali`. This successfully recovered notes for 98 tasks which would otherwise have been left blank.
- **ASCII-only Console Runner:** The runner `run_phase2.py` has been updated to use purely ASCII characters for status borders, avoiding `UnicodeEncodeError` on Windows consoles where UTF-8 encoding might not be set by default.
- **Missing Required Fields:** The script currently logs QA issues if `technique` or `weave_types` are missing, but still outputs the row if the image is valid. The captioning phase will need to handle missing fields gracefully.

---
*End of Phase 2 Documentation*
