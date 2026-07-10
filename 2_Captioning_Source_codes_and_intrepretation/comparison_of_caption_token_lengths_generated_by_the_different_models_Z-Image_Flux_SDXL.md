# Phase 4: Dataset Quality Assessment & Token Analysis

This document summarizes the token and word length analysis across the generated caption datasets for **Z-Image**, **Flux**, and **SDXL** models. It also serves as the deliverable for task 4.3 based on professor feedback.

## 1. Caption Stats Summary Table

| Model | Total Images | Avg. Words | Avg. Qwen2 Tokens | Avg. GPT2/CLIP Tokens | Min Qwen2 | Max Qwen2 | Status |
|-------|--------------|------------|-------------------|----------------------|-----------|-----------|--------|
| **ZIMAGE** | 177 | 116.6 words | 161.5 tokens | 163.7 tokens | 119 | 241 | ✅ ~160 tokens target consistency |
| **FLUX** | 177 | 156.1 words | 207.6 tokens | 209.3 tokens | 137 | 365 | ✅ Dense text prompt |
| **SDXL** | 177 | 41.7 words | 81.2 tokens | 84.7 tokens | 43 | 117 | ✅ Short tags prompt |

---

## 2. Detailed Distribution Analysis

### ZIMAGE Caption Token Lengths (Qwen2 Tokenizer)

- **Total Captions**: 177
- **Average length**: 161.5 tokens
- **Median length**: 156.0 tokens
- **Min length**: 119 tokens
- **Max length**: 241 tokens

**Distribution Buckets (Qwen2 Tokens):**

| Token Range | Count | Percentage |
|-------------|-------|------------|
| 0 - 50 | 0 | 0.0% |
| 50 - 100 | 0 | 0.0% |
| 100 - 150 | 55 | 31.1% |
| 150 - 200 | 115 | 65.0% |
| 200 - 250 | 7 | 4.0% |
| 250 - 500 | 0 | 0.0% |

### FLUX Caption Token Lengths (Qwen2 Tokenizer)

- **Total Captions**: 177
- **Average length**: 207.6 tokens
- **Median length**: 202.0 tokens
- **Min length**: 137 tokens
- **Max length**: 365 tokens

**Distribution Buckets (Qwen2 Tokens):**

| Token Range | Count | Percentage |
|-------------|-------|------------|
| 0 - 50 | 0 | 0.0% |
| 50 - 100 | 0 | 0.0% |
| 100 - 150 | 1 | 0.6% |
| 150 - 200 | 82 | 46.3% |
| 200 - 250 | 76 | 42.9% |
| 250 - 500 | 18 | 10.2% |

### SDXL Caption Token Lengths (Qwen2 Tokenizer)

- **Total Captions**: 177
- **Average length**: 81.2 tokens
- **Median length**: 81.0 tokens
- **Min length**: 43 tokens
- **Max length**: 117 tokens

**Distribution Buckets (Qwen2 Tokens):**

| Token Range | Count | Percentage |
|-------------|-------|------------|
| 0 - 50 | 1 | 0.6% |
| 50 - 100 | 156 | 88.1% |
| 100 - 150 | 20 | 11.3% |
| 150 - 200 | 0 | 0.0% |
| 200 - 250 | 0 | 0.0% |
| 250 - 500 | 0 | 0.0% |

---

## 3. CSV Dataset Links

In addition to the raw `.jsonl` files and the sidecar `.txt` files, a clean two-column CSV file (`image_path`, `caption`) has been generated for each model for easy review:

1. **Z-Image Captions CSV**: [captions_zimage.csv](../data/id10/zimage/captions_zimage.csv)
2. **Flux Captions CSV**: [captions_flux.csv](../data/id10/flux/captions_flux.csv)
3. **SDXL Captions CSV**: [captions_sdxl.csv](../data/id10/sdxl/captions_sdxl.csv)

## 4. Token Consistency Note

> [!NOTE]
> As noted, we are maintaining captions around **160 tokens** to ensure structural and content consistency across the different image generation models (Z-Image and Flux) before starting LoRA fine-tuning.
