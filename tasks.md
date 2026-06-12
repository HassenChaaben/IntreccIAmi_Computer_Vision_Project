# Project ID 10 - IntreccIAmi Tasks Tracker

This task list serves as the operational checklist for executing the **IntreccIAmi** image-generation and captioning pipeline. It outlines every step required to build the dataset, generate captions with Qwen via Ollama, train a LoRA model using DiffSynth-Studio, and evaluate the results.

---

## Phase 1: Environment & Data Preparation

- [x] **1.1 Directory Setup** ✅ `phase1/setup_directories.py`
  - [x] Create workspace root directory
  - [x] Create `data/raw_json/` directory
  - [x] Create `data/images/` directory
- [x] **1.2 Retrieve Source Materials** ✅ `phase1/copy_source_materials.py`
  - [x] Download `label_studio_texture_labels.json` and copy it into `data/raw_json/`
  - [x] Download `TEXTURE DI INTRECCI` image folder and copy all image files into `data/images/`
- [x] **1.3 Environment Installation** ✅ *Environment set up on GPU server*
  - [x] Setup Conda/Python virtual environment (Python 3.10 recommended)
  - [x] Install dependencies for data processing (`python-docx`, `pandas`, etc.)
  - [x] Clone the official `DiffSynth-Studio` repository
  - [x] Install `DiffSynth-Studio` in editable mode (`pip install -e .`)

---

## Phase 2: Metadata Extraction & Normalization

- [x] **2.1 Raw JSON Parsing** ✅ `phase2/normalize_dataset.py`
  - [x] Load the Label Studio JSON export from `data/raw_json/label_studio_texture_labels.json`
  - [x] Verify that the root of the JSON file is a list of tasks
- [x] **2.2 Integrity Verification (First Technical Check)** ✅ `phase2/normalize_dataset.py`
  - [x] For each task, extract the image identifier from `data.image`, `image.original_path`, `image.filename`, or `file_upload`
  - [x] Extract the clean filename (e.g., `IMG_0259.jpg`)
  - [x] Verify that the file exists in the `data/images/` directory
  - [x] Log any missing images or parsing anomalies in a report (`data/qa_report.csv`)
- [x] **2.3 Field Normalization & Parsing Script** ✅ `phase2/normalize_dataset.py`
  - [x] Build a Python parser that processes annotations (`annotations[].result`) for each valid task
  - [x] Map Label Studio XML names to normalized fields:
    - [x] `codice_bottega` &rarr; `code`
    - [x] `tecnica_usata` &rarr; `technique` (expected: *Intreccio, Macramè, Uncinetto, Rinfilo*)
    - [x] `tipologia_intreccio` &rarr; `weave_types` (e.g., *Vario*)
    - [x] `finitura_type` &rarr; `finish.type`
    - [x] `verniciatura_colore` &rarr; `finish.paint_color`
    - [x] `ral_pantone` &rarr; `finish.ral_pantone`
    - [x] `catalogo_bottega` &rarr; `finish.paint_mode` or `finish.special`
    - [x] `oggetto_note` &rarr; `special_description`
    - [x] `oggetto_bbox` &rarr; `bbox` (respecting `x`, `y`, `width`, `height`, `rotation`, `label`, `original_width`, `original_height`)
  - [x] Build lists for repeated structural components:
    - [x] **Poste (Vertical / supporting elements):** Group fields `poste_i_*` (materiale, dimensione, quantita, distanza, colore) into objects in the `posts` array
    - [x] **Trama (Horizontal / passing elements):** Group fields `trama_i_*` (materiale, dimensione, distanza, colore) into objects in the `wefts` array
  - [x] Save the normalized dataset as `data/normalized_metadata.jsonl`

---

## Phase 3: VLM Captioning (Ollama Engine)

- [x] **3.1 Ollama Integration & Environment** ✅ `phase3/run_phase3.py`
  - [x] Verify Ollama status and pull local Qwen vision model (recommended: `qwen3.5:27b` — latest Qwen model)
  - [x] Install local `ollama` Python client library
- [x] **3.2 Prompt Template Engineering** ✅ `phase3/run_phase3.py`
  - [x] Define the captioning expert system instructions (see instructions doc §8)
  - [x] Include captioning objective, manufacturability constraints, and JSON metadata in the prompt
  - [x] Enforce the rule: do not invent materials/colors/objects not present in metadata
  - [x] Enforce English output ready for diffusion-model training
  - [x] Save prompt template with input/output examples as a deliverable
- [x] **3.3 Model-Specific Caption Engines Implementation** ✅ `phase3/run_phase3.py`
  - [x] **3.3.1 Z-Image Caption Engine (`caption_zimage.py`)**
    - [x] Design custom aesthetic prompt engineering for Z-Image
    - [x] Format: Prefix with `intrecciami-style`, describe weave pattern/technique/finish/materials, suffix with premium texture/studio lighting descriptors
    - [x] Target caption length: 60–100 words
  - [x] **3.3.2 Flux Caption Engine (`caption_flux.py`)**
    - [x] Design custom dense prompt engineering for Flux T5 text encoder
    - [x] Format: Prefix with `intrecciami-style`, long conversational paragraph (80–120 words) detailing structural and material properties
  - [x] **3.3.3 SDXL Caption Engine (`caption_sdxl.py`)**
    - [x] Design custom tag-based prompt engineering for dual SDXL CLIP encoders
    - [x] Format: Trigger word, comma-separated descriptors (technique, weave types, finish, post/weft materials), and a short sentence (30–50 words)
- [x] **3.4 Dataset Generation & Batch Resuming** ✅ `phase3/run_phase3.py`
  - [x] Implement checkpoint-based local inference with sidecar progress trackers
  - [x] Export captioned datasets to:
    - [x] `data/id10/zimage/metadata_zimage.jsonl` (Z-Image specific captions)
    - [x] `data/id10/flux/metadata_flux.jsonl` (Flux specific captions)
    - [x] `data/id10/sdxl/metadata_sdxl.jsonl` (SDXL specific captions)
- [x] **3.5 Baseline Model Verification (Pre-Training)** ✅ *Completed baseline inference tests*
  - [x] Test the base models (Z-Image, Flux, SDXL) by running inference using a sample of the generated captions
  - [x] Inspect and save the generated images to establish a **before-LoRA** baseline for visual comparison

---

## Phase 4: Dataset Quality Assurance & Packaging

- [x] **4.1 Caption Validation & Filtering** ✅ `phase3/run_phase3.py`
  - [x] Verify that captions across all 3 models strictly adhere to their respective length constraints
  - [x] Ensure that captions have 0% hallucinated attributes (no invented materials, colors, objects, decorations, or product categories)
  - [x] Ensure that captions do not describe overly complex or physically implausible structures (manufacturability check)
  - [x] Ensure that captions do not leak internal metadata (coordinates, file names, Label Studio IDs, JSON field names, bbox values, internal codes like BTG-XXXX)
  - [x] Validate that every valid entry has a non-empty caption
  - [x] Validate sidecar `.txt` files containing clean model-specific prompt lines next to each image
  - [x] Generate `qa_report.csv` with: missing images count, empty captions, missing fields, overly long captions, manually checked examples
- [x] **4.2 Packaging & Final Directory Layout** ✅ `phase3/run_phase3.py`
  - [x] Setup the final directory layout for the 3 target models as specified in the instructions (§9):
    - [x] **Z-Image** (`data/id10/zimage/`):
      - [x] `images/` — all matched raw training images
      - [x] Sidecar `.txt` caption files alongside images (if recipe uses sidecar format)
      - [x] `metadata_zimage.jsonl` — one row per image: path + caption + normalized metadata
      - [x] `qa_report_zimage.csv` — caption validation audit report
    - [x] **Flux** (`data/id10/flux/`):
      - [x] `images/` — all matched raw training images
      - [x] Sidecar `.txt` caption files alongside images (if recipe uses sidecar format)
      - [x] `metadata_flux.jsonl` — one row per image: path + caption + normalized metadata
      - [x] `qa_report_flux.csv` — caption validation audit report
    - [x] **SDXL** (`data/id10/sdxl/`):
      - [x] `images/` — all matched raw training images
      - [x] Sidecar `.txt` caption files alongside images (if recipe uses sidecar format)
      - [x] `metadata_sdxl.jsonl` — one row per image: path + caption + normalized metadata
      - [x] `qa_report_sdxl.csv` — caption validation audit report
- [x] **4.3 Professor Feedback Tasks for captioning**
  - [x] Compile a comparison of caption token lengths generated by the different models (Z-Image, Flux, SDXL) and provide the precise link
  - [x] Generate one CSV file for each model (Z-Image, Flux, SDXL) including the image file path and the corresponding generated caption (e.g. `captions_zimage.csv`, `captions_flux.csv`, `captions_sdxl.csv`)

---

## Phase 5: LoRA Training (DiffSynth-Studio)

- [x] **5.1 Setup Base Models** ✅
  - [x] Configure DiffSynth-Studio recipes for:
    - [x] **Z-Image** — DiT architecture, `Tongyi-MAI/Z-Image` + `Z-Image-Turbo` tokenizer
    - [x] **FLUX.1-dev** — Double-stream DiT, `black-forest-labs/FLUX.1-dev`
    - [ ] **SDXL** base model — *Not trained (skipped due to GPU time constraints)*
- [x] **5.2 Custom Dataset Paths** ✅ `phase5/5.2.md`
  - [x] Map model-specific datasets (`data/id10/zimage/`, `data/id10/flux/`) to their training scripts
  - [x] Convert metadata key `"caption"` → `"prompt"` for DiffSynth-Studio compatibility
- [x] **5.3 Execution & Epoch Tuning** ✅ `phase5/5.3.md`
  - [x] **Z-Image**: 1 epoch trained (epoch-0), single GPU, rank 16, `dataset_repeat 50`
  - [x] **FLUX**: 2 epochs trained (epoch-0, epoch-1), multi-GPU (`CUDA_VISIBLE_DEVICES=1,2`), rank 32, `dataset_repeat 20`
    - [x] Epoch-0: Fresh training → `epoch-0.safetensors` (see `phase5/code_used_to_fine_tune_flux_epoch_0`)
    - [x] Epoch-1: Resumed from epoch-0 with `--lora_checkpoint` → `epoch-1.safetensors` (see `phase5/code_used_to_fine_tune_flux_epoch_1`)
  - [ ] SDXL: *Not trained*
- [x] **5.4 Post-Training Inference Validation** ✅ `phase5/5.4.md`
  - [x] **Z-Image before/after** on 50 unseen prompts (epoch-0) → `z_score_generate_unseen.py`
  - [x] **Z-Image before/after** on 10 seen prompts (epoch-0, epoch-1) → `z_score_generate_seen.py`
  - [x] **FLUX LoRA** on 50 unseen prompts (epoch-0) → `flux_generate_unseen.py`
  - [x] **FLUX LoRA** on 10 seen prompts (epoch-0) → `flux_generate_seen.py`
  - [x] **FLUX LoRA** on 50 unseen prompts (epoch-1) → `flux_generate_unseen_epoch_1.py`
  - [x] Results saved in `Results_before_after_training/` with subfolders per experiment

---

## Phase 6: Evaluation & Deliverables

- [ ] **6.1 Inference Demo**
  - [ ] Construct at least 10 model-tailored **new** test prompts (not identical to training captions) to assess LoRA generalization capacity
  - [ ] Example: "a minimalist woven rattan lamp shade with simple repeatable lattice texture"
  - [ ] Generate and save all corresponding output images as a deliverable set
- [ ] **6.2 Automated and Qualitative Scoring**
  - [ ] Compute **CLIPScore** for prompt alignment
  - [ ] Compute **LPIPS** for style/perceptual similarity
  - [ ] Compute **CLIP-IQA** for perceptual image quality estimation
  - [ ] Grade generations across the 5-point qualitative rubric using an **MLLM-as-a-judge** pipeline:
    - [ ] Prompt adherence (0–5)
    - [ ] Consistency with intreccio identity (0–5)
    - [ ] Manufacturability (0–5)
    - [ ] Visual quality (0–5)
    - [ ] Controlled originality (0–5)
- [ ] **6.3 Final Packaging for Professor Submission**
  - [ ] Consolidate scripts (`normalize_dataset.py`, `caption_zimage.py`, `caption_flux.py`, `caption_sdxl.py`) inside `Project_ID10_IntreccIAmi/`
  - [ ] Deliver the three distinct trained LoRA weight files and configurations
  - [ ] Deliver the prompt templates with input/output examples
  - [ ] Provide before/after LoRA comparison reports across Z-Image, Flux, and SDXL
  - [ ] Include `qa_report.csv` and all per-model QA reports

---

## Pre-Submission Checklist (from Instructions §14)

- [ ] All images referenced in the JSON exist in the TEXTURE DI INTRECCI folder
- [ ] Each valid entry has a non-empty caption
- [ ] Captions use metadata but do not mention JSON, Label Studio, bbox, or filename
- [ ] Materials, colors, technique, and finish are not invented
- [ ] The final dataset is reproducible: relative paths, scripts, configuration, and seed are saved
- [ ] Examples of generation before/after LoRA are included
- [ ] The evaluation includes automatic metrics and a structured qualitative judgment
