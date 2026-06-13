# Phase 6: Evaluation, Generalization & Deliverables Report Template

This document details the setup and execution instructions for the final phase of the **IntreccIAmi (ID 10)** project: executing generalization inference, computing automated quantitative scores (CLIPScore, LPIPS, CLIP-IQA), and conducting qualitative VLM-as-a-judge evaluations on the GPU server.

---

## 1. Phase 6 Goal & Scope

The goal of Phase 6 is to verify how well our fine-tuned LoRA models generalize to **unseen objects and styles** using the learned `intrecciami-style` texture. 

While Phases 3 and 5 verified the models' behavior on the same distribution as the training set (seen prompts) and on similar texture blocks (unseen prompts), Phase 6 introduces **10 entirely new, complex prompt scenarios** (e.g., woven handbags, modern chairs, lampshades, cylindrical vases) to rigorously test the models' style transference boundaries and aesthetic flexibility.

---

## 2. Generalization Test Prompts (10 New Prompts)

To assess LoRA generalization capacity, run inference with the following 10 model-tailored test prompts containing new target objects and diverse composition styles:

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

## 3. Execution Instructions (GPU Server)

Run the scripts below sequentially on your GPU-enabled server to generate the comparison set and calculate the evaluation metrics.

### Step 3.1: Generate Generalization Images
Execute the inference script for each model:

```bash
# 1. FLUX.1-dev LoRA (Epoch 1)
python phase6/inference_generalization.py --model flux --epoch 1 --device cuda

# 2. SDXL LoRA (Epoch 4)
python phase6/inference_generalization.py --model sdxl --epoch 4 --device cuda

# 3. Z-Image LoRA (Epoch 4)
python phase6/inference_generalization.py --model zimage --epoch 4 --device cuda
```
*Output images will be saved under `Results_before_after_training/phase6_generations/{model_name}/` alongside sidecar `.txt` prompts.*

### Step 3.2: Run Automated Evaluation Metrics
Calculate the CLIPScore, LPIPS, and CLIP-IQA scores for each set of outputs:

```bash
# Evaluate FLUX
python phase6/evaluate_metrics.py --image_dir Results_before_after_training/phase6_generations/flux/ --ref_dir data/images/ --device cuda

# Evaluate SDXL
python phase6/evaluate_metrics.py --image_dir Results_before_after_training/phase6_generations/sdxl/ --ref_dir data/images/ --device cuda

# Evaluate Z-Image
python phase6/evaluate_metrics.py --image_dir Results_before_after_training/phase6_generations/zimage/ --ref_dir data/images/ --device cuda
```
*Reports will be saved to `quantitative_scoring_report.csv` and `automated_metrics_report.md` inside each model's output directory.*

### Step 3.3: Run MLLM Qualitative Judge
Grade the outputs using the local Qwen-Vision VLM:

```bash
# Grade FLUX
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/flux/ --model qwen2.5-vision

# Grade SDXL
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/sdxl/ --model qwen2.5-vision

# Grade Z-Image
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/zimage/ --model qwen2.5-vision
```
*Grades will be saved to `qualitative_scoring_report.csv` and `mllm_judge_report.md` inside each model's output directory.*

---

## 4. Quantitative Metrics Table

*Once the scripts in **Step 3.2** have run on the GPU server, copy the average scores from the generated reports and populate this table:*

| Model / Epoch | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) | Style Alignment |
| :--- | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | [TBD] | [TBD] | [TBD] | [TBD] |
| **Z-Image (Epoch 4 LoRA)** | [TBD] | [TBD] | [TBD] | [TBD] |
| **SDXL (Baseline)** | [TBD] | [TBD] | [TBD] | [TBD] |
| **SDXL (Epoch 1 LoRA)** | [TBD] | [TBD] | [TBD] | [TBD] |
| **FLUX.1-dev (Baseline)** | [TBD] | [TBD] | [TBD] | [TBD] |
| **FLUX.1-dev (Epoch 1 LoRA)** | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 5. Qualitative Scoring Rubric Table (VLM Judge)

*Once the scripts in **Step 3.3** have run, populate this table with the average 5-point ratings:*

| Model / Epoch | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | **Mean Score** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |
| **Z-Image (Epoch 4 LoRA)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |
| **SDXL (Baseline)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |
| **SDXL (Epoch 1 LoRA)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |
| **FLUX.1-dev (Baseline)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |
| **FLUX.1-dev (Epoch 1 LoRA)** | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | [TBD] / 5.0 | **[TBD] / 5.0** |

---

## 6. How to Compare and Select the Champion Model

When interpreting the final populated tables:
1. **Prompt Adherence & CLIPScore**: Look for the model that achieves the highest text alignment. T5-based text encoders (like the one in FLUX) generally yield significantly higher prompt adherence scores when dealing with descriptive, detail-heavy prompts.
2. **Intreccio Geometry & LPIPS**: Models with lower LPIPS scores show a closer stylistic alignment to the ground-truth textures. Ensure that the fine-tuned model has successfully learned structured grid patterns rather than visual "noise".
3. **Manufacturability & Visual Realism**: Compare MLLM ratings for manufacturability. High scores indicate that the model respects physical boundaries, creating interlocking crossings rather than impossible overlapping or loose thread endpoints.
4. **Generalization capacity**: Look at "Controlled Originality" to see which model successfully maps the `intrecciami-style` texture onto complex, unseen shapes (such as handbags or furniture seats) without melting the weave's structure.
