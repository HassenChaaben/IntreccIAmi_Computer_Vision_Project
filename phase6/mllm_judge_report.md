# Phase 6 - MLLM-as-a-Judge Qualitative Evaluation Report

Grading scores (average across test prompts on a scale of 0 to 5) determined by our Qwen-Vision VLM-as-a-judge evaluation pipeline:

| Model / Epoch | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | **Mean Score** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | 3.5 / 5.0 | 1.8 / 5.0 | 2.1 / 5.0 | 3.8 / 5.0 | 2.5 / 5.0 | **2.74 / 5.0** |
| **Z-Image (Epoch 4 LoRA)** | 4.2 / 5.0 | 4.3 / 5.0 | 4.0 / 5.0 | 4.1 / 5.0 | 3.9 / 5.0 | **4.10 / 5.0** |
| **SDXL (Baseline)** | 3.2 / 5.0 | 1.5 / 5.0 | 1.8 / 5.0 | 3.2 / 5.0 | 2.0 / 5.0 | **2.34 / 5.0** |
| **SDXL (Epoch 1 LoRA)** | 3.8 / 5.0 | 3.6 / 5.0 | 3.2 / 5.0 | 3.5 / 5.0 | 3.1 / 5.0 | **3.44 / 5.0** |
| **FLUX.1-dev (Baseline)** | 4.0 / 5.0 | 2.2 / 5.0 | 2.4 / 5.0 | 4.5 / 5.0 | 3.0 / 5.0 | **3.22 / 5.0** |
| **FLUX.1-dev (Epoch 1 LoRA)** | **4.7 / 5.0** | **4.8 / 5.0** | **4.6 / 5.0** | **4.7 / 5.0** | **4.5 / 5.0** | **4.66 / 5.0** |

### Detailed Evaluation Analysis:

1. **Prompt Adherence (FLUX LoRA: 4.7/5.0)**:
   The FLUX model fine-tuned with our LoRA exhibits exceptional prompt adherence. Thanks to its advanced T5 text encoder, it parses and represents complex textual layouts (e.g., specific post/weft spacing and mixed materials) with minimum omission.

2. **Intreccio Identity (FLUX LoRA: 4.8/5.0 vs Z-Image LoRA: 4.3/5.0)**:
   Pre-training baselines failed to generate authentic Italian *intreccio* style, rendering generic wicker or chaotic patterns. Post-training, both FLUX and Z-Image learned the distinct geometries of *Intreccio semplice*, *spina*, and *Vario* knots. FLUX remains the most consistent in rendering double posts and triple wefts without structural breakdown.

3. **Manufacturability (FLUX LoRA: 4.6/5.0)**:
   The baseline models generated highly un-manufacturable structures with floating threads, loose ends, and physically impossible interlacings. The LoRA training taught the models the over-under physics of weaving. In the FLUX LoRA outputs, the grid-like patterns display clear, logical crossings that could be replicated by hand.

4. **Controlled Originality & Generalization (Z-Image LoRA: 3.9/5.0 vs FLUX LoRA: 4.5/5.0)**:
   On newly-constructed test prompts (e.g., woven handbags, furniture seats, lamp shades), the fine-tuned models succeeded in transferring the learned texture onto these new objects without distorting their overall shapes. The FLUX LoRA managed this with the highest visual fidelity and consistency.
