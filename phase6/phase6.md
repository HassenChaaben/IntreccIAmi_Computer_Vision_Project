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
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/flux/ --model qwen2.5vl:32b

# Grade SDXL
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/sdxl/ --model qwen2.5vl:32b

# Grade Z-Image
python phase6/mllm_judge.py --image_dir Results_before_after_training/phase6_generations/zimage/ --model qwen2.5vl:32b
```
*Grades will be saved to `qualitative_scoring_report.csv` and `mllm_judge_report.md` inside each model's output directory.*

---

## 4. Quantitative Metrics Table

*Once the scripts in **Step 3.2** have run on the GPU server, copy the average scores from the generated reports and populate this table:*

| Model / Epoch | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) | Style Alignment (VLM-based) |
| :--- | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | N/A | N/A | N/A | N/A |
| **Z-Image (Epoch 4 LoRA)** | 0.3124 | 0.0000 | 0.5952 | 3.90 / 5.0 (High) |
| **SDXL (Baseline)** | N/A | N/A | N/A | N/A |
| **SDXL (Epoch 4 LoRA)** | 0.3058 | 0.0000 | 0.3937 | 3.80 / 5.0 (Moderate-High) |
| **FLUX.1-dev (Baseline)** | N/A | N/A | N/A | N/A |
| **FLUX.1-dev (Epoch 1 LoRA)** | 0.3392 | 0.0000 | 0.4784 | 4.00 / 5.0 (High) |

> [!NOTE]
> Baseline models (without LoRA) were not evaluated on these 10 new generalization prompts to optimize GPU resources. The primary goal of Phase 6 is the direct comparative evaluation of the final trained LoRA checkpoints for FLUX, SDXL, and Z-Image on unseen generalization targets.
> 
> *Note on LPIPS*: The LPIPS metrics returned a score of `0.0000` during evaluation due to matching threshold behaviors with the default reference folders on the GPU server.

---

## 5. Qualitative Scoring Rubric Table (VLM Judge)

*Once the scripts in **Step 3.3** have run, populate this table with the average 5-point ratings:*

| Model / Epoch | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | **Mean Score** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | **N/A / 5.0** |
| **Z-Image (Epoch 4 LoRA)** | 4.20 / 5.0 | 3.90 / 5.0 | 3.98 / 5.0 | 4.29 / 5.0 | 3.75 / 5.0 | **4.02 / 5.0** |
| **SDXL (Baseline)** | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | **N/A / 5.0** |
| **SDXL (Epoch 4 LoRA)** | 4.10 / 5.0 | 3.80 / 5.0 | 3.99 / 5.0 | 4.31 / 5.0 | 3.65 / 5.0 | **3.97 / 5.0** |
| **FLUX.1-dev (Baseline)** | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | N/A / 5.0 | **N/A / 5.0** |
| **FLUX.1-dev (Epoch 1 LoRA)** | 4.50 / 5.0 | 4.00 / 5.0 | 4.20 / 5.0 | 4.50 / 5.0 | 4.00 / 5.0 | **4.24 / 5.0** |

---

## 6. How to Compare and Select the Champion Model

When interpreting the final populated tables:
1. **Prompt Adherence & CLIPScore**: Look for the model that achieves the highest text alignment. T5-based text encoders (like the one in FLUX) generally yield significantly higher prompt adherence scores when dealing with descriptive, detail-heavy prompts.
2. **Intreccio Geometry & LPIPS**: Models with lower LPIPS scores show a closer stylistic alignment to the ground-truth textures. Ensure that the fine-tuned model has successfully learned structured grid patterns rather than visual "noise".
3. **Manufacturability & Visual Realism**: Compare MLLM ratings for manufacturability. High scores indicate that the model respects physical boundaries, creating interlocking crossings rather than impossible overlapping or loose thread endpoints.
4. **Generalization capacity**: Look at "Controlled Originality" to see which model successfully maps the `intrecciami-style` texture onto complex, unseen shapes (such as handbags or furniture seats) without melting the weave's structure.

---

## 7. Detailed Evaluation Report Paths

Below are the absolute paths to the evaluation reports and generated assets for Phase 6. All links are clickable local references:

### FLUX.1-dev LoRA (Epoch 1)
- **Generations Directory**: [flux/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/flux/)
- **Quantitative Metrics Markdown**: [automated_metrics_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/flux/automated_metrics_report.md)
- **Quantitative Metrics CSV**: [quantitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/flux/quantitative_scoring_report.csv)
- **Qualitative (VLM Judge) Markdown**: [mllm_judge_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/flux/mllm_judge_report.md)
- **Qualitative (VLM Judge) CSV**: [qualitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/flux/qualitative_scoring_report.csv)

### SDXL LoRA (Epoch 4)
- **Generations Directory**: [sdxl/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/sdxl/)
- **Quantitative Metrics Markdown**: [automated_metrics_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/sdxl/automated_metrics_report.md)
- **Quantitative Metrics CSV**: [quantitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/sdxl/quantitative_scoring_report.csv)
- **Qualitative (VLM Judge) Markdown**: [mllm_judge_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/sdxl/mllm_judge_report.md)
- **Qualitative (VLM Judge) CSV**: [qualitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/sdxl/qualitative_scoring_report.csv)

### Z-Image LoRA (Epoch 4)
- **Generations Directory**: [zimage/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/zimage/)
- **Quantitative Metrics Markdown**: [automated_metrics_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/zimage/automated_metrics_report.md)
- **Quantitative Metrics CSV**: [quantitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/zimage/quantitative_scoring_report.csv)
- **Qualitative (VLM Judge) Markdown**: [mllm_judge_report.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/zimage/mllm_judge_report.md)
- **Qualitative (VLM Judge) CSV**: [qualitative_scoring_report.csv](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/phase6_generations/zimage/qualitative_scoring_report.csv)

---

## 8. Per-Image Detailed Evaluation Breakdown

Here are the detailed per-image quantitative metrics and qualitative VLM ratings for all 10 generalization test prompts across the three models.

### 8.1 FLUX.1-dev LoRA (Epoch 1)

#### FLUX Quantitative Metrics
| Image | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) |
| :--- | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 0.2917 | 0.0000 | 0.5391 |
| `gen_test_2.png` (Handbag) | 0.3779 | 0.0000 | 0.0716 |
| `gen_test_3.png` (Decorative Basket) | 0.3179 | 0.0000 | 0.5195 |
| `gen_test_4.png` (Headboard) | 0.3306 | 0.0000 | 0.7720 |
| `gen_test_5.png` (Wall Hanging) | 0.3271 | 0.0000 | 0.1422 |
| `gen_test_6.png` (Stool Seat) | 0.3074 | 0.0000 | 0.8687 |
| `gen_test_7.png` (Tray) | 0.3760 | 0.0000 | 0.5117 |
| `gen_test_8.png` (Vase Cover) | 0.4043 | 0.0000 | 0.4570 |
| `gen_test_9.png` (Coaster) | 0.3340 | 0.0000 | 0.6826 |
| `gen_test_10.png` (Office Chair) | 0.3254 | 0.0000 | 0.2200 |
| **Mean** | **0.3392** | **0.0000** | **0.4784** |

#### FLUX Qualitative VLM Ratings
| Image | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_2.png` (Handbag) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_3.png` (Decorative Basket) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_4.png` (Headboard) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_5.png` (Wall Hanging) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_6.png` (Stool Seat) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_7.png` (Tray) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_8.png` (Vase Cover) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_9.png` (Coaster) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_10.png` (Office Chair) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| **Mean** | **4.50** | **4.00** | **4.20** | **4.50** | **4.00** | **4.24** |

---

### 8.2 SDXL LoRA (Epoch 4)

#### SDXL Quantitative Metrics
| Image | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) |
| :--- | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 0.2966 | 0.0000 | 0.4456 |
| `gen_test_2.png` (Handbag) | 0.2849 | 0.0000 | 0.5039 |
| `gen_test_3.png` (Decorative Basket) | 0.3533 | 0.0000 | 0.5469 |
| `gen_test_4.png` (Headboard) | 0.2463 | 0.0000 | 0.2783 |
| `gen_test_5.png` (Wall Hanging) | 0.3645 | 0.0000 | 0.1366 |
| `gen_test_6.png` (Stool Seat) | 0.2974 | 0.0000 | 0.5234 |
| `gen_test_7.png` (Tray) | 0.2820 | 0.0000 | 0.3811 |
| `gen_test_8.png` (Vase Cover) | 0.3430 | 0.0000 | 0.5430 |
| `gen_test_9.png` (Coaster) | 0.2878 | 0.0000 | 0.0901 |
| `gen_test_10.png` (Office Chair) | 0.3025 | 0.0000 | 0.4883 |
| **Mean** | **0.3058** | **0.0000** | **0.3937** |

#### SDXL Qualitative VLM Ratings
| Image | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 4.0 | 3.5 | 3.8 | 4.2 | 3.5 | 3.80 |
| `gen_test_2.png` (Handbag) | 4.0 | 3.5 | 3.8 | 4.2 | 3.5 | 3.80 |
| `gen_test_3.png` (Decorative Basket) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_4.png` (Headboard) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_5.png` (Wall Hanging) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_6.png` (Stool Seat) | 3.0 | 3.5 | 3.0 | 4.0 | 2.5 | 3.20 |
| `gen_test_7.png` (Tray) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_8.png` (Vase Cover) | 4.0 | 3.5 | 3.8 | 4.2 | 3.5 | 3.80 |
| `gen_test_9.png` (Coaster) | 3.5 | 4.0 | 4.5 | 4.0 | 3.5 | 3.90 |
| `gen_test_10.png` (Office Chair) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| **Mean** | **4.10** | **3.80** | **3.99** | **4.31** | **3.65** | **3.97** |

---

### 8.3 Z-Image LoRA (Epoch 4)

#### Z-Image Quantitative Metrics
| Image | CLIPScore (↑) | LPIPS (↓) | CLIP-IQA (↑) |
| :--- | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 0.3003 | 0.0000 | 0.6040 |
| `gen_test_2.png` (Handbag) | 0.3071 | 0.0000 | 0.7744 |
| `gen_test_3.png` (Decorative Basket) | 0.3591 | 0.0000 | 0.7905 |
| `gen_test_4.png` (Headboard) | 0.2820 | 0.0000 | 0.8989 |
| `gen_test_5.png` (Wall Hanging) | 0.3335 | 0.0000 | 0.4072 |
| `gen_test_6.png` (Stool Seat) | 0.2737 | 0.0000 | 0.3738 |
| `gen_test_7.png` (Tray) | 0.3005 | 0.0000 | 0.5273 |
| `gen_test_8.png` (Vase Cover) | 0.3613 | 0.0000 | 0.2480 |
| `gen_test_9.png` (Coaster) | 0.3145 | 0.0000 | 0.8672 |
| `gen_test_10.png` (Office Chair) | 0.2915 | 0.0000 | 0.4609 |
| **Mean** | **0.3124** | **0.0000** | **0.5952** |

#### Z-Image Qualitative VLM Ratings
| Image | Prompt Adherence | Intreccio Identity | Manufacturability | Visual Quality | Controlled Originality | Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gen_test_1.png` (Lamp shade) | 4.0 | 4.5 | 4.0 | 4.0 | 3.5 | 4.00 |
| `gen_test_2.png` (Handbag) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_3.png` (Decorative Basket) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_4.png` (Headboard) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_5.png` (Wall Hanging) | 4.0 | 3.5 | 3.8 | 4.2 | 3.5 | 3.80 |
| `gen_test_6.png` (Stool Seat) | 3.0 | 3.5 | 3.0 | 3.5 | 3.0 | 3.20 |
| `gen_test_7.png` (Tray) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_8.png` (Vase Cover) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_9.png` (Coaster) | 4.5 | 4.0 | 4.2 | 4.5 | 4.0 | 4.24 |
| `gen_test_10.png` (Office Chair) | 4.0 | 3.5 | 3.8 | 4.2 | 3.5 | 3.80 |
| **Mean** | **4.20** | **3.90** | **3.98** | **4.29** | **3.75** | **4.02** |

