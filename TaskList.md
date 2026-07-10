# Project ID 10 - IntreccIAmi Tasks Tracker

This task list serves as the operational checklist for executing the **IntreccIAmi** image-generation, captioning, and style-learning pipeline. It outlines every step required to build the dataset, generate captions with Qwen via Ollama, train LoRA models using DiffSynth-Studio, and evaluate the results.

---

## 1_Metadata_Extrcation_Source_codes_and_intrepretation

### Phase 1: Environment & Data Preparation

- [x] **1.1 Directory Setup** ✅ [setup_directories.py](1_Metadata_Extrcation_Source_codes_and_intrepretation/setup_directories.py)
  - [x] Create workspace root directory
  - [x] Create `data/raw_json/` directory
  - [x] Create `data/images/` directory
- [x] **1.2 Retrieve Source Materials** ✅ [copy_source_materials.py](1_Metadata_Extrcation_Source_codes_and_intrepretation/copy_source_materials.py)
  - [x] Download `label_studio_texture_labels.json` and copy it into `data/raw_json/`
  - [x] Download `TEXTURE DI INTRECCI` image folder and copy all image files into `data/images/`
- [x] **1.3 Environment Installation** ✅ *Environment set up on GPU server*
  - [x] Setup Conda/Python virtual environment (Python 3.10)
  - [x] Install dependencies for data processing (`python-docx`, `pandas`, etc.)
  - [x] Clone the official `DiffSynth-Studio` repository
  - [x] Install `DiffSynth-Studio` in editable mode (`pip install -e .`)

---

### Phase 2: Metadata Extraction & Normalization

- [x] **2.1 Raw JSON Parsing** ✅ [normalize_dataset.py](1_Metadata_Extrcation_Source_codes_and_intrepretation/normalize_dataset.py)
  - [x] Load the Label Studio JSON export from `data/raw_json/label_studio_texture_labels.json`
  - [x] Verify that the root of the JSON file is a list of tasks
- [x] **2.2 Integrity Verification (First Technical Check)** ✅ [normalize_dataset.py](1_Metadata_Extrcation_Source_codes_and_intrepretation/normalize_dataset.py)
  - [x] For each task, extract the image identifier from `data.image`, `image.original_path`, `image.filename`, or `file_upload`
  - [x] Extract the clean filename (e.g., `IMG_0259.jpg`)
  - [x] Verify that the file exists in the `data/images/` directory
  - [x] Log any missing images or parsing anomalies in a report (`data/metadata_extraction_qa_report.csv`)
- [x] **2.3 Field Normalization & Parsing Script** ✅ [normalize_dataset.py](1_Metadata_Extrcation_Source_codes_and_intrepretation/normalize_dataset.py)
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

## 2_Captioning_Source_codes_and_intrepretation

### Phase 3: VLM Captioning (Ollama Engine)

- [x] **3.1 Ollama Integration & Environment** ✅ [run_phase3.py](2_Captioning_Source_codes_and_intrepretation/run_phase3.py)
  - [x] Verify Ollama status and pull local Qwen vision model (`qwen3.5:27b`)
  - [x] Install local `ollama` Python client library
- [x] **3.2 Prompt Template Engineering** ✅ [run_phase3.py](2_Captioning_Source_codes_and_intrepretation/run_phase3.py)
  - [x] Define the captioning expert system instructions
  - [x] Include captioning objective, manufacturability constraints, and JSON metadata in the prompt
  - [x] Enforce the rule: do not invent materials/colors/objects not present in metadata
  - [x] Enforce English output ready for diffusion-model training
  - [x] Save prompt template with input/output examples as a deliverable in [phase3.md](2_Captioning_Source_codes_and_intrepretation/phase3.md)
- [x] **3.3 Model-Specific Caption Engines Implementation** ✅
  - [x] **3.3.1 Z-Image Caption Engine** ([caption_zimage.py](2_Captioning_Source_codes_and_intrepretation/caption_zimage.py))
    - [x] Prefix with `intrecciami-style`, describe weave pattern/technique/finish/materials, suffix with premium texture descriptors
    - [x] Target caption length: 60–100 words
  - [x] **3.3.2 Flux Caption Engine** ([caption_flux.py](2_Captioning_Source_codes_and_intrepretation/caption_flux.py))
    - [x] Long conversational paragraph (80–120 words) detailing structural and material properties
  - [x] **3.3.3 SDXL Caption Engine** ([caption_sdxl.py](2_Captioning_Source_codes_and_intrepretation/caption_sdxl.py))
    - [x] Trigger word, comma-separated descriptors (technique, weave types, finish, post/weft materials), and a short sentence (30–50 words)
- [x] **3.4 Dataset Generation & Batch Resuming** ✅ [run_phase3.py](2_Captioning_Source_codes_and_intrepretation/run_phase3.py)
  - [x] Export captioned datasets to:
    - [x] `data/id10/zimage/metadata_zimage.jsonl`
    - [x] `data/id10/flux/metadata_flux.jsonl`
    - [x] `data/id10/sdxl/metadata_sdxl.jsonl`
- [x] **3.5 Baseline Model Verification (Pre-Training)** ✅ [test_baseline_inference.py](2_Captioning_Source_codes_and_intrepretation/test_baseline_inference.py)
  - [x] Test baseline models and save baseline generated images in results folder to establish a before-LoRA benchmark

---

### Phase 4: Dataset Quality Assurance & Packaging

- [x] **4.1 Caption Validation & Filtering** ✅ [run_phase3.py](2_Captioning_Source_codes_and_intrepretation/run_phase3.py)
  - [x] Verify length constraints, filter hallucinated attributes, and ensure no metadata leaks (coordinates, filenames, Label Studio IDs, JSON fields, etc.)
  - [x] Validate that every valid entry has a non-empty caption
  - [x] Validate sidecar `.txt` files containing clean prompt lines next to each image in the dataset directories
- [x] **4.2 Packaging & Final Directory Layout** ✅ [run_phase3.py](2_Captioning_Source_codes_and_intrepretation/run_phase3.py)
  - [x] Set up final matched training images, metadata `.jsonl` files, and QA reports in `data/id10/zimage/`, `data/id10/flux/`, and `data/id10/sdxl/`
- [x] **4.3 Professor Feedback Tasks for Captioning** ✅
  - [x] Compile a token length comparison report: [comparison_of_caption_token_lengths_generated_by_the_different_models_Z-Image_Flux_SDXL.md](2_Captioning_Source_codes_and_intrepretation/comparison_of_caption_token_lengths_generated_by_the_different_models_Z-Image_Flux_SDXL.md) (built via [compare_tokens.py](2_Captioning_Source_codes_and_intrepretation/compare_tokens.py))
  - [x] Generate per-model path-to-caption CSV mapping files: [generate_csvs.py](2_Captioning_Source_codes_and_intrepretation/generate_csvs.py)

---

## 3_Lora_tuning_Source_codes_and_intrepretation

### Phase 5: LoRA Training (DiffSynth-Studio)

- [x] **5.1 Setup Base Models & Environment** ✅ [phase4.md](3_Lora_tuning_Source_codes_and_intrepretation/phase4.md)
  - [x] Configure DiffSynth-Studio recipes for Z-Image (`Tongyi-MAI/Z-Image`), FLUX.1-dev (`black-forest-labs/FLUX.1-dev`), and SDXL base models
- [x] **5.2 Custom Dataset Paths** ✅ [phase4.md](3_Lora_tuning_Source_codes_and_intrepretation/phase4.md)
  - [x] Map datasets and convert metadata key `"caption"` &rarr; `"prompt"` for training scripts
- [x] **5.3 Execution & Epoch Tuning** ✅ [training_scripts/](3_Lora_tuning_Source_codes_and_intrepretation/training_scripts/)
  - [x] **Z-Image**: 4 epochs trained (epoch-0 to 3), single GPU, rank 32, dataset repeat 20. Total cumulative steps: 14,160 steps.
  - [x] **FLUX**: 2 epochs trained (epoch-0 to 1), multi-GPU, rank 32, dataset repeat 20. Total cumulative steps: 3,540 steps.
  - [x] **SDXL**: 4 epochs trained (epoch-0 to 3), multi-GPU, rank 32, dataset repeat 20. Total cumulative steps: 14,160 steps.
- [x] **5.4 Hyperparameter Documentation** ✅
  - [x] Deliver detailed parameter summary files: [summary_zimage.md](3_Lora_tuning_Source_codes_and_intrepretation/summary_zimage.md), [summary_flux.md](3_Lora_tuning_Source_codes_and_intrepretation/summary_flux.md), and [summary_sdxl.md](3_Lora_tuning_Source_codes_and_intrepretation/summary_sdxl.md).

---

## 4_Evaluation_Source_codes_and_intrepretation

### Phase 6: Evaluation & Deliverables

- [x] **6.1 Quantitative & Qualitative Automated Scoring** ✅ [run_eval_pipeline.py](4_Evaluation_Source_codes_and_intrepretation/code_to_evaluate/run_eval_pipeline.py)
  - [x] Compute **CLIPScore** for prompt alignment (using OpenAI CLIP ViT-B/32 text/image encoders)
  - [x] Compute **LPIPS** style distance to real artisan textures (using AlexNet backpropagation feature difference on raw pixel tensors)
  - [x] Compute **CLIP-IQA** for aesthetic quality estimation
  - [x] Grade generations using an automated **VLM-as-a-judge** simulator scoring 5 key criteria on a 5-point scale (Prompt Adherence, Intreccio Identity, Manufacturability, Visual Quality, Controlled Originality)
- [x] **6.2 Consolidated Reports Generation** ✅ [update_markdowns.py](4_Evaluation_Source_codes_and_intrepretation/code_to_evaluate/update_markdowns.py)
  - [x] Save individual reports and metrics spreadsheets under `4_Evaluation_Source_codes_and_intrepretation/` and `all_adopted_Results/Org_baseline_LoraGeneratedResults/`
  - [x] Deliver unified comparison index [comparision.md](4_Evaluation_Source_codes_and_intrepretation/comparision.md)
  - [x] Deliver comprehensive detailed evaluation report [phase6.md](4_Evaluation_Source_codes_and_intrepretation/phase6.md)

---

## Pre-Submission Checklist

- [x] All images referenced in the JSON exist in the TEXTURE DI INTRECCI folder
- [x] Each valid entry has a non-empty caption
- [x] Captions use metadata but do not mention JSON, Label Studio, bbox, or filename
- [x] Materials, colors, technique, and finish are not invented
- [x] The final dataset is reproducible: relative paths, scripts, configuration, and seed are saved
- [x] Examples of generation before/after LoRA are included
- [x] The evaluation includes automatic metrics and a structured qualitative judgment
