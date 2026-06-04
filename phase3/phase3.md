# Phase 3: VLM Captioning (Ollama Engine) – Complete Documentation

> **Project:** IntreccIAmi (ID 10)  
> **Date:** 2026-06-04  
> **Phase Status:** ✅ Scripts implemented & verified locally in mock mode.  
> **Tasks Covered:** 3.1 (Ollama Setup), 3.2 (Prompt Templates), 3.3 (Z-Image, Flux, SDXL Engines), 3.4 (Dataset Generation & Checkpoint Resuming)

---

## Table of Contents

1. [VLM Captioning Overview](#1-vlm-captioning-overview)
2. [Task 3.1 – Ollama Integration & Model Pull](#2-task-31--ollama-integration--model-pull)
3. [Task 3.2 & 3.3 – Model-Specific Caption Engines](#3-task-32--33--model-specific-caption-engines)
4. [Task 3.4 – Checkpoint-Based Resuming & Directory Layout](#4-task-34--checkpoint-based-resuming--directory-layout)
5. [Execution Guide: Running on the University GPU Server](#5-execution-guide-running-on-the-university-gpu-server)
6. [Task 3.5 – Baseline Model Verification (Pre-Training Inference)](#6-task-35--baseline-model-verification-pre-training-inference)
7. [Transferring Outputs & Quality Control Guide](#7-transferring-outputs--quality-control-guide)

---

## 1. VLM Captioning Overview

Phase 3 implements the **Vision-Language Model (VLM)** captioning pipeline. Diffusion models (such as those trained via DiffSynth-Studio) require natural language captions rather than raw JSON metadata. 

We use **Qwen3.5:27b** (via Ollama) to translate our normalized metadata records and images into high-quality descriptive captions. To support different text encoders and model behaviors, we build three specialized captioning engines:
1. **Z-Image**: Aesthetic description with high-resolution studio photography suffixes (60-100 words).
2. **Flux**: Dense conversational paragraphs detailing physical construction (80-120 words).
3. **SDXL**: Trigger word followed by comma-separated tags and a short summary sentence (30-50 words).

---

## 2. Task 3.1 – Ollama Integration & Model Pull

We leverage **Ollama** as a lightweight, fast local serving engine for the Qwen vision-language model, rather than loading the raw weights directly via HuggingFace transformers (which is slow and memory-intensive).

### Verification & Model Installation Commands:
```bash
# 1. Start or check the Ollama service on the GPU server container
# (Ollama is usually started in the background as a system service or docker command)
ollama serve &

# 2. Pull the recommended Qwen3.5:27b model (multimodal vision model)
ollama pull qwen3.5:27b

# 3. Verify that the model is loaded in memory
ollama list
```

### Python Library Dependencies:
To connect to the local Ollama API, the python client must be installed:
```bash
pip install ollama
```

---

## 3. Task 3.2 & 3.3 – Model-Specific Caption Engines

Each caption engine parses metadata variables and feeds a highly customized prompt system to Ollama. The system enforces strict rules against hallucinations (not inventing colors/materials) and pipeline leaks (never printing file names, coordinates, or Label Studio IDs).

### 3.3.1 Z-Image Engine (`caption_zimage.py`)
* **Objective**: Generate captions for Z-Image fine-tuning.
* **Constraints**: Strictly **60–100 words**.
* **Prompt Format**:
  - Starts with prefix: `intrecciami-style`
  - Body: Detailed physical weave pattern, technique, finish, and materials.
  - Suffix: `, close-up studio photograph, premium texture, high resolution, macro photography`

### 3.3.2 Flux Engine (`caption_flux.py`)
* **Objective**: Dense text captions for the Flux T5 text encoder.
* **Constraints**: Strictly **80–120 words**.
* **Prompt Format**:
  - Starts with prefix: `intrecciami-style`
  - Body: A long, descriptive, conversational paragraph detailing the mechanical structure, strand crossings, finish, repeat pattern consistency, alignment, and tactile texture.

### 3.3.3 SDXL Engine (`caption_sdxl.py`)
* **Objective**: Tag-based prompt + short description for SDXL CLIP dual encoders.
* **Constraints**: Strictly **30–50 words**.
* **Prompt Format**:
  - Starts with trigger: `intrecciami-style`
  - Body: Comma-separated tags of raw variables (e.g. `rattan material, green color, varnished finish`).
  - Concludes: A single short descriptive sentence explaining the overall pattern block.

---

## 4. Task 3.4 – Checkpoint-Based Resuming & Directory Layout

Processing 177 high-resolution images via VLM on a shared server is prone to network drops or job preemptions. Each engine implements a **resilient resume checkpoint capability**:
* When processing task `IMG_XXXX.jpg`, the script checks if `captions/IMG_XXXX.txt` or `images/IMG_XXXX.txt` already exists.
* If found, it skips the Ollama model inference and reads the existing caption in milliseconds.
* This allows the pipeline to instantly resume after a crash without wasting GPU resources.

### Output Packaging & Directory Layout
Each engine automatically copies the matched image files to its target folder and saves sidecar captions in both required structures:

```text
data/id10/
├── zimage/
│   ├── images/
│   │   ├── IMG_6865.jpg
│   │   └── IMG_6865.txt            # Sidecar caption format next to image
│   ├── captions/
│   │   └── IMG_6865.txt            # Captions directory format
│   ├── metadata_zimage.jsonl       # Unified captioned JSONL
│   └── qa_report_zimage.csv        # Length and metadata QA audit report
├── flux/
│   ├── images/
│   │   ├── IMG_6865.jpg
│   │   └── IMG_6865.txt
│   ├── captions/
│   │   └── IMG_6865.txt
│   ├── metadata_flux.jsonl
│   └── qa_report_flux.csv
└── sdxl/
    ├── images/
    │   ├── IMG_6865.jpg
    │   └── IMG_6865.txt
    ├── captions/
    │   └── IMG_6865.txt
    ├── metadata_sdxl.jsonl
    └── qa_report_sdxl.csv
```

---

## 5. Execution Guide: Running on the University GPU Server

Follow these step-by-step commands to deploy and run Phase 3 on the university GPU server:

### Step A: Connect to VPN
Load your Wireguard profile (`gruppo_18.txt`) in your Wireguard application and activate the connection.

### Step B: Connect to the Server
SSH into the GPU master host:
```bash
ssh cvdl2026-gr18@192.168.80.138
```
*(Enter password: `_lOYw$%&xG-uiop`)*

### Step C: Enter the GPU Docker Container
Start and enter your group's Docker container:
```bash
docker start cvdl2026-gr18-GPU1
docker exec -it cvdl2026-gr18-GPU1 bash
```

### Step D: Prepare local files & code
From your **local computer terminal**, copy the normalized dataset and our Phase 3 scripts into the container:
```bash
scp -r phase3/ cvdl2026-gr18@192.168.80.138:~/DiffSynth-Studio/
scp -r data/ cvdl2026-gr18@192.168.80.138:~/DiffSynth-Studio/
```

### Step E: Execute the Runner
Ensure Ollama is running and has the model pulled:
```bash
ollama pull qwen3.5:27b
pip install ollama
```

Now, execute the master runner on the server:
```bash
python phase3/run_phase3.py
```
*(To validate the pipeline locally without running Ollama/GPU, run with `python phase3/run_phase3.py --use_mock`)*

---

## 6. Task 3.5 – Baseline Model Verification (Pre-Training Inference)

Before fine-tuning the LoRAs, the professor requested establishing a **pre-training baseline** to evaluate what the base models generate using our custom prompts *prior* to any weight adjustments. 

We implemented a baseline tester script `phase3/test_baseline_inference.py` that connects directly to the DiffSynth-Studio pipelines.

### How to Run Baseline Inference on the Server:

1. **Enter the Docker container** and activate the conda environment:
   ```bash
   conda activate diffsynth
   ```

2. **Run the baseline script** specifying the model and target caption index:
   - For **Flux**:
     ```bash
     python phase3/test_baseline_inference.py --model flux --index 0
     ```
   - For **SDXL**:
     ```bash
     python phase3/test_baseline_inference.py --model sdxl --index 0
     ```
   - For **Z-Image**:
     ```bash
     python phase3/test_baseline_inference.py --model zimage --index 0
     ```

3. **Check the outputs**:
   The generated baseline images will be saved inside `data/baseline_outputs/` as `baseline_<model>_task_<task_id>.png`.
   Inspect these images to confirm that the base model (pre-LoRA) produces generic textures that lack the specific structural properties and identity of the artisan hand-woven patterns.

---

## 7. Transferring Outputs & Quality Control Guide

Once the captioning engines finish executing on the university GPU server, you need to download the generated files to your Windows local machine and inspect the captions for quality.

### 7.1 How to Send the Output back to your Windows Machine
Since you have command-line (CMD/PowerShell) access to the server, you can use the secure copy protocol (`scp`) from your **local Windows machine** (not inside the SSH session) to copy the generated data.

Open a **new CMD or PowerShell window on your local Windows computer** and run:

```cmd
# Copy the entire generated id10 dataset from the server to your local data folder
scp -r cvdl2026-gr18@192.168.80.138:~/DiffSynth-Studio/data/id10/ C:\Users\user\Downloads\Dataset_preparation\mouhaymin\data\
```
*(Enter your server password when prompted: `_lOYw$%&xG-uiop`)*

Alternatively, you can use a graphical client like **FileZilla** or **WinSCP** on Windows:
- **Host**: `192.168.80.138` (make sure your Wireguard VPN is connected!)
- **Username**: `cvdl2026-gr18`
- **Password**: `_lOYw$%&xG-uiop`
- **Port**: `22` (standard SFTP)
- Navigate to `~/DiffSynth-Studio/data/id10/` and download the folder to your local machine.

---

### 7.2 How to Verify if the Captioning is Good or Not

To verify if the captioning pipeline succeeded and matches the requirements for a high-grade project, perform these four checks:

#### Check 1: Output File Verification
Verify that the following structure has been fully generated under your local `data/id10/` folder:
- **`metadata_zimage.jsonl`**, **`metadata_flux.jsonl`**, and **`metadata_sdxl.jsonl`** exist and are non-empty.
- Each model folder has an `images/` directory containing the copied raw images and sidecar `.txt` prompt files next to them.
- Check that there are **177 tasks** processed in each JSONL file.

#### Check 2: QA Report Audits (Word Count & Errors)
Open the generated CSV reports:
- `data/id10/zimage/qa_report_zimage.csv`
- `data/id10/flux/qa_report_flux.csv`
- `data/id10/sdxl/qa_report_sdxl.csv`
Check the `warning` column:
- If there are no warning messages (except for the expected Task 129 `Both 'posts' and 'wefts' arrays are empty` anomaly), the word limits are perfectly respected!
- Ensure that the `word_count` column falls within the expected ranges:
  - Z-Image: **60–100 words**
  - Flux: **80–120 words**
  - SDXL: **30–50 words**

#### Check 3: Content Integrity Check (Visual Inspection)
Open the JSONL file or sidecar text prompts and manually review a few lines (e.g. Task 1 or 2). Ensure that:
- **0% Hallucinations**: It does not list materials or colors that are not present in the metadata (e.g. if the image contains only green rattan, the text should not mention plastic or wood).
- **Physical Plausibility**: The text correctly describes weave elements (posts, wefts, over-under crossings, spacing) that are physically constructible.
- **Trigger Word Presence**: Z-Image and Flux prompts start with `intrecciami-style`, and SDXL prompts contain `intrecciami-style` followed by comma-separated tags.
- **No Metadata Leaks**: The caption must *never* contain internal codes (like `BTG-XXXX`), coordinates, filenames (like `IMG_6865.jpg`), Label Studio keywords, or bbox percentages.

#### Check 4: Pre-Training vs. Post-Training Comparison
When you execute LoRA training later (Phase 5), compare your baseline generated images (under `data/baseline_outputs/` from Task 3.5) with the trained LoRA outputs using the same captions. 
- **Good results**: The baseline base model outputs generic textures, whereas the trained LoRA outputs distinct woven textures matching your dataset's structural identity.

---
*End of Phase 3 Documentation*
