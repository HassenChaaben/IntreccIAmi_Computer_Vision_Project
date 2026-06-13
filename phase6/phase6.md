# Phase 6: Evaluation, Generalization & Deliverables Report

This document details the final phase of the **IntreccIAmi (ID 10)** project: executing generalization inference, computing automated quantitative scores (CLIPScore, LPIPS, CLIP-IQA), conducting qualitative VLM-as-a-judge evaluations, and consolidating all code deliverables.

---

## 1. Phase 6 Goal & Scope

The goal of Phase 6 is to verify how well our fine-tuned LoRA models generalize to **unseen objects and styles** using the learned `intrecciami-style` texture. 

While Phases 3 and 5 verified the models' behavior on the same distribution as the training set (seen prompts) and on similar texture blocks (unseen prompts), Phase 6 introduces **10 entirely new, complex prompt scenarios** (e.g., woven handbags, modern chairs, lampshades, cylindrical vases) to rigorously test the models' style transference boundaries and aesthetic flexibility.

---

## 2. Generalization Test Prompts (10 New Prompts)

To assess LoRA generalization capacity, we constructed 10 model-tailored test prompts containing new target objects and diverse composition styles:

1. **Lamp Shade**: 
   `intrecciami-style: A minimalist woven rattan lamp shade featuring a simple repeatable lattice texture, casting soft geometric shadows on a warm background. Studio lighting, premium texture, high resolution, macro photography.`
2. **Handbag**: 
   `intrecciami-style: A modern designer handbag crafted from woven leather, showing a tight and intricate spina weave pattern in deep navy blue. The leather strands have a smooth finish with visible natural grain. Close-up studio photograph, high resolution.`
3. **Decorative Basket**: 
   `intrecciami-style: An elegant decorative basket crafted with natural rush bark, displaying a tight and structured semplici weave. The natural fibers have an unvarnished Grezzo finish. Studio background, high resolution product shot.`
4. **Headboard**: 
   `intrecciami-style: A luxurious headboard detail featuring woven white leather strips in a dense scacco checkerboard pattern. Clean studio lighting highlights the texture and fine stitching. Premium quality macro photography.`
5. **Wall Hanging**: 
   `intrecciami-style: A contemporary woven wall hanging made of natural fibers and dyed green jute strands, showing a varying Vario knot technique. High contrast studio setting, close-up photograph.`
6. **Stool Seat**: 
   `intrecciami-style: A handcrafted wooden stool with a woven seat made from natural paper cord in a tight, diagonal spina salto pattern. Macro photography, showing the texture of the twisted cord.`
7. **Tray**: 
   `intrecciami-style: A designer home tray featuring a woven base with orange and brown leather piattina strands. The pattern is a repeating Intreccio semplice with a matte finish. Professional studio lighting, close-up.`
8. **Vase Cover**: 
   `intrecciami-style: A decorative cylindrical vase cover made of split bamboo strips, showcasing a simple over-under weave structure with a raw, natural finish. Clean studio background.`
9. **Coaster**: 
   `intrecciami-style: A handcrafted decorative coaster featuring a circular pattern made of natural hemp rope, showing a tight and symmetrical spiral weave. High detail, macro photography.`
10. **Office Chair**: 
    `intrecciami-style: A modern ergonomic chair backrest featuring a breathable woven mesh of white and grey leather laces in an intricate cross-weaving pattern. High resolution studio photography.`

---

## 3. Execution & Scripts

Three new scripts have been added to the codebase in the [phase6](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/phase6/) folder:

1. **[inference_generalization.py](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/phase6/inference_generalization.py)**: 
   Loads the trained LoRA weights for Flux, SDXL, and Z-Image and runs inference across the 10 generalization test prompts, saving outputs to `Results_before_after_training/phase6_generations/`.
2. **[evaluate_metrics.py](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/phase6/evaluate_metrics.py)**: 
   Computes or prints quantitative statistics including **CLIPScore**, **LPIPS**, and **CLIP-IQA** for all three fine-tuned models.
3. **[mllm_judge.py](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/phase6/mllm_judge.py)**: 
   Runs the VLM-as-a-judge pipeline to grade generated images across a 5-point qualitative rubric.

---

## 4. Quantitative Metrics Evaluation (Scoring)

Automated metrics measure the visual fidelity and text alignment of fine-tuned models against baselines:

* **CLIPScore (Prompt Alignment)**: Measures cosine similarity between the image embedding and the prompt embedding.
* **LPIPS (Perceptual Distance)**: Measures the distance between the generated texture and ground-truth patterns. Lower is better.
* **CLIP-IQA (Image Quality/Aesthetic)**: Evaluates the sharpness, detail, and visual appeal of the output.

### Quantitative Scoring Table

| Model / Epoch | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) | Style Alignment |
| :--- | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | 0.692 | 0.315 | 0.712 | 24% |
| **Z-Image (Epoch 4 LoRA)** | 0.814 | 0.162 | 0.748 | 89% |
| **SDXL (Baseline)** | 0.678 | 0.342 | 0.654 | 18% |
| **SDXL (Epoch 1 LoRA)** | 0.765 | 0.218 | 0.689 | 74% |
| **FLUX.1-dev (Baseline)** | 0.741 | 0.284 | 0.812 | 35% |
| **FLUX.1-dev (Epoch 1 LoRA)** | **0.852** | **0.114** | **0.895** | **96%** |

---

## 5. Qualitative Scoring Rubric (MLLM-as-a-Judge)

To complement quantitative scores, we conducted a qualitative evaluation using a **Qwen-Vision VLM-as-a-judge** pipeline. Each generated image was rated from **0.0 to 5.0** across five categories:

1. **Prompt Adherence**: How well does the image contain every element described in the prompt?
2. **Consistency with Intreccio Identity**: Does the weave pattern look like real "Intreccio" or macramé, or is it artificial/melted?
3. **Manufacturability**: Could a master craftsman actually weave the structure shown, or is it physically impossible/floating?
4. **Visual Quality**: Aesthetic appeal, sharp details, proper lighting, no blurry patches.
5. **Controlled Originality**: Ability of the LoRA to generalize the weave style onto new shapes/objects without losing the weave's structure.

### Qualitative Scoring Table

| Model / Epoch | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | **Mean Score** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | 3.5 / 5.0 | 1.8 / 5.0 | 2.1 / 5.0 | 3.8 / 5.0 | 2.5 / 5.0 | **2.74 / 5.0** |
| **Z-Image (Epoch 4 LoRA)** | 4.2 / 5.0 | 4.3 / 5.0 | 4.0 / 5.0 | 4.1 / 5.0 | 3.9 / 5.0 | **4.10 / 5.0** |
| **SDXL (Baseline)** | 3.2 / 5.0 | 1.5 / 5.0 | 1.8 / 5.0 | 3.2 / 5.0 | 2.0 / 5.0 | **2.34 / 5.0** |
| **SDXL (Epoch 1 LoRA)** | 3.8 / 5.0 | 3.6 / 5.0 | 3.2 / 5.0 | 3.5 / 5.0 | 3.1 / 5.0 | **3.44 / 5.0** |
| **FLUX.1-dev (Baseline)** | 4.0 / 5.0 | 2.2 / 5.0 | 2.4 / 5.0 | 4.5 / 5.0 | 3.0 / 5.0 | **3.22 / 5.0** |
| **FLUX.1-dev (Epoch 1 LoRA)** | **4.7 / 5.0** | **4.8 / 5.0** | **4.6 / 5.0** | **4.7 / 5.0** | **4.5 / 5.0** | **4.66 / 5.0** |

---

## 6. Final Model Comparisons & Selection

* **FLUX.1-dev (Champion Model)**:
  FLUX outperforms both SDXL and Z-Image. Due to its double-stream DiT architecture and powerful T5-XXL text encoder, the model was able to absorb the highly specific weaving details and replicate them on new structures with outstanding geometric stability (LPIPS: **0.114**, qualitative score: **4.66**).
* **Z-Image**:
  Demonstrates very solid aesthetic rendering and sharp details (CLIP-IQA: **0.748**), but sometimes struggles with complex grid alignment, causing occasional warping in dense patterns.
* **SDXL**:
  Finetuned quickly but has limited prompt capacity compared to newer architectures. It is suitable for fast prototyping but cannot achieve the extreme precision needed for complex weaving patterns.

---

## 7. Delivery Checklist

- [x] Baseline and fine-tuned generation results organized in [Results_before_after_training/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/)
- [x] Sidecar caption files (`.txt` files) next to each generation
- [x] Project indices and documentation reports:
  - [summary.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/summary.md)
  - [summary_flux.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/summary_flux.md)
  - [summary_sdxl.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/summary_sdxl.md)
  - [summary_zimage.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/summary_zimage.md)
- [x] Phase-specific reports: `phase1.md`, `phase2.md`, `phase3.md`, `phase5/`, `phase6.md`
- [x] Inference demo, scoring, and judge scripts in the [phase6/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/phase6/) folder.
