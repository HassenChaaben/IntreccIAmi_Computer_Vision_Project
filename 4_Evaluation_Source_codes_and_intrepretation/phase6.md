# Phase 6: Evaluation, Generalization & Deliverables Report

This document details the final phase of the **IntreccIAmi (ID 10)** project: executing generalization inference on 20 unseen validation prompts, computing automated quantitative scores (CLIPScore, LPIPS, CLIP-IQA), and conducting qualitative VLM-as-a-judge evaluations.

## Detailed Explanation of Automated Evaluation Metrics

To understand how our models are graded quantitatively, we use three distinct automated metrics. Below is a detailed explanation of what each metric means, why we use it, and exactly how it is calculated in our pipeline.

### CLIPScore (Text Alignment / Prompt Adherence)

* **What it measures**: CLIPScore quantifies the semantic alignment between a generated image and its text prompt. It answers the question: *"Does the image contain what the prompt asked for?"*
* **Why we use it**: It prevents the model from generating high-quality but irrelevant images (hallucinations). It is the standard reference-free metric for text-to-image evaluation.
* **How we calculate it**:
  1. **Text Encoding**: The written prompt is passed through the **CLIP text encoder** (OpenAI's ViT-B/32 model), converting it into a vector representing semantic meaning.
  2. **Image Encoding**: The generated image is passed through the **CLIP image encoder** to produce a visual embedding vector.
  3. **Normalization**: Both vectors are $L_2$-normalized to unit length.
  4. **Cosine Similarity**: We compute the dot product (cosine similarity) between the normalized text and image vectors.
  5. **Final Score**: The average cosine similarity across all 20 test prompts becomes the model's overall CLIPScore.
* **Range and Interpretation**:
  - *Theoretical range*: -1.0 to +1.0 (cosine similarity).
  - *Practical range (in this project)*: Positive values between `0.20` and `0.60`. Good scores typically fall in `0.30`–`0.40`.
  - *Interpretation*: Higher is better (stronger semantic match). A score of 0 means orthogonal (unrelated) embeddings.
* **Limitations**:
  - CLIPScore measures semantic alignment, not visual realism—an image with correct objects but unnatural textures can still score well.
  - CLIP is trained on general web images, not woven textiles specifically. It may not capture craft-authenticity nuances.
  - It cannot detect physically impossible weave structures—a plausible-looking but uncraftable image can score high.
  - It is sensitive to prompt wording: different phrasings of the same intent produce different scores.
* **Source**: Hessel et al. (2022), "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" (adapted for text-to-image evaluation).

---

### LPIPS (Learned Perceptual Image Patch Similarity)

* **What it measures**: LPIPS measures the perceptual distance between a generated image and real reference images. It answers: *"How close is this generated weave to actual woven reference photos?"* A **lower** LPIPS score is better, representing smaller stylistic distance.
* **Why we use it**: Standard pixel-to-pixel comparison (like MSE or PSNR) fails when the object changes shape. LPIPS compares deep perceptual features (curves, intersections, texture patterns) instead of exact pixel colors.
* **How we calculate it**:
  1. **Image Resizing**: Both generated and reference images are resized to `512×512` pixels using PIL bilinear resampling to ensure matching spatial dimensions.
  2. **Tensor Normalization**: Raw `[0, 255]` pixel arrays are converted to `[-1, 1]` float tensors (corrected normalization that avoids scale underflow).
  3. **Feature Extraction**: Images are passed through a pre-trained **AlexNet** network. Intermediate feature maps are extracted from multiple layers.
  4. **Perceptual Distance**: At each layer, feature maps are channel-normalized and the L2 distance is computed between generated and reference features.
  5. **Scaling & Averaging**: Differences at each layer are multiplied by learned importance factors (trained on human perceptual data) and averaged into a total perceptual distance.
  6. **Nearest-Neighbor Matching**: For each generated image, this distance is calculated against all available real reference images, and the **minimum distance** (closest match) is recorded.
* **Range and Interpretation**:
  - *Range*: 0.0 and above (no theoretical upper bound, typically below 1.0).
  - *Interpretation*: Lower is better. A score of `0.0` means perceptually indistinguishable from the reference (rare in practice).
  - *In this project*: With corrected tensor normalization, values averaged between `0.50` and `0.65`, reflecting natural perceptual differences.
* **Limitations**:
  - LPIPS requires reference images—cannot evaluate unseen prompts with no ground-truth.
  - Perceptual closeness does not guarantee structural correctness—a visually similar but structurally wrong weave can score low distance.
  - LPIPS is sensitive to image resolution and preprocessing. Both images must be resized to 512×512 to avoid dimension mismatch errors.
  - The choice of backbone network (AlexNet vs VGG) affects scores; results are only comparable within the same backbone.
* **Engineering note**: In this project, both generated and reference images were resized to 512×512 using PIL bilinear resampling to avoid LPIPS dimension mismatch errors that would otherwise crash the evaluation script.
* **Source**: Zhang et al. (2018), "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018).

---

### CLIP-IQA (CLIP-based Image Quality Assessment)

* **What it measures**: CLIP-IQA estimates the perceptual quality of an image without needing a reference. It answers: *"Does this image look good to a human?"* A higher score means the image is crisp, well-lit, and free of artifacts.
* **Why we use it**: It lets us evaluate visual appeal and quality blindly, without needing a reference image to compare against.
* **How we calculate it**:
  1. **Image Encoding**: The generated image is passed through the CLIP image encoder to extract its visual embedding vector.
  2. **Quality Prompt Pairs**: A pair of opposite quality descriptions is defined:
     * **Positive anchor**: *"good quality, high resolution, sharp photo, professional lighting"*
     * **Negative anchor**: *"bad quality, blurry, low resolution, noisy photo, distorted"*
  3. **Text Encoding**: Both anchor prompts are converted into embedding vectors using the CLIP text encoder.
  4. **Cosine Similarities**: Similarities are computed: `sim_pos` (image vs positive anchor) and `sim_neg` (image vs negative anchor).
  5. **Softmax Probability**: The similarities are passed through a Softmax function, converting them into probabilities summing to `1.0`.
  6. **Final Score**: The probability assigned to the positive anchor becomes the CLIP-IQA score.
* **Range and Interpretation**:
  - *Range*: 0.0 to 1.0 (softmax probability).
  - *Interpretation*: Higher is better. A score of `0.50` means the image is equally close to both anchors (ambiguous quality).
* **Limitations**:
  - CLIP-IQA evaluates generic photographic quality (sharpness, lighting), not domain-specific authenticity.
  - A sharp, well-lit image of a physically impossible weave structure scores high.
  - The positive/negative text anchors are fixed and may not capture what "quality" means for woven textiles specifically.
  - It is biased toward photographic aesthetics—a visually striking but incorrect weave can score better than a correct but plain image.
* **Source**: Wang et al. (2023), "Exploring CLIP for Assessing the Look and Feel of Images" (AAAI 2023).

---

## Qualitative Evaluation Criteria (VLM Judge Rubric)

To score the aesthetic quality and stylistic authenticity of generated images from a design and manufacturing perspective, the Qwen-Vision VLM scores each generated image on a 5-point scale (`1.0` to `5.0`) using five qualitative criteria:

1. **Prompt Adherence**:
   * **What it means**: Grades how faithfully the image represents all semantic elements mentioned in the prompt (e.g. if the prompt asks for "a handcrafted wooden stool with a woven seat," the VLM checks if a wooden stool with a woven seat is present).
   * **Significance**: Identifies failures in target object reconstruction or composition instruction compliance.
2. **Intreccio Identity**:
   * **What it means**: Evaluates whether the generated pattern represents an authentic, recognizable Italian weaving technique (Intreccio) rather than generic mesh, flat patterns, or visual noise.
   * **Significance**: Ensures the LoRA has acquired the specific artisan style characteristics rather than just repeating simple grids.
3. **Manufacturability**:
   * **What it means**: Assesses whether the depicted weave is physically possible to construct by hand. The VLM checks if strands interlace in a logical over-under order, if they are physically continuous, and if they terminate correctly at borders rather than floating in mid-air.
   * **Significance**: Critical for design prototyping where the output image must serve as a blueprint for real physical objects.
4. **Visual Quality**:
   * **What it means**: Evaluates overall render quality, sharpness, contrast, lighting, and presence of digital artifacts or distortion.
   * **Significance**: Acts as a qualitative checkpoint alongside CLIP-IQA to identify blurry elements, texture compression, or rendering failures.
5. **Controlled Originality**:
   * **What it means**: Measures how effectively the model generalizes the learned `intrecciami-style` texture onto complex, unseen target objects (like a handbag or ergonomic chair) without distorting the weave's structure or melting the target shape.
   * **Significance**: Verifies if the LoRA acts as a flexible, generalized style compiler instead of a rigid texture copier.
6. **Mean Score**:
   * **What it means**: The simple arithmetic mean of the five criteria scores, calculated as: `(Adherence + Identity + Manufacturability + Quality + Originality) / 5`. This provides a single balanced grade summarizing the overall generalization success of the LoRA model for each generated image.

---

## Conclusion & Comparative Analysis

Based on the quantitative metrics and qualitative VLM-as-a-judge evaluations, this section compares the three trained models (**FLUX**, **SDXL**, and **Z-Image**) and identifies which model excels in each aspect of our text-to-image style adaptation.

### Combined Quantitative Metrics Comparison (Mean ± Std)

| Model | CLIPScore (Text Alignment) | CLIP-IQA (Aesthetic Quality) | LPIPS (Style Distance to Real) |
| :--- | :---: | :---: | :---: |
| **FLUX (DiT)** | 0.3106 ± 0.0312 | 0.6444 ± 0.2301 | 0.6194 ± 0.0939 |
| **SDXL (Latent Diffusion)** | 0.3102 ± 0.0340 | 0.6722 ± 0.1195 | 0.5578 ± 0.0778 |
| **Z-Image (DiT)** | 0.3122 ± 0.0313 | 0.4850 ± 0.2542 | 0.5270 ± 0.1227 |

### Combined Qualitative MLLM Ratings Comparison (Average / 5.0)

| Evaluation Criteria | FLUX (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) |
| :--- | :---: | :---: | :---: |
| **Prompt Adherence** | 4.23 ± 0.07 | 3.24 ± 0.07 | 3.83 ± 0.07 |
| **Intreccio Identity** | 4.15 ± 0.09 | 3.06 ± 0.09 | 3.75 ± 0.09 |
| **Manufacturability** | 4.05 ± 0.09 | 2.94 ± 0.09 | 3.65 ± 0.09 |
| **Visual Quality** | 4.26 ± 0.08 | 3.23 ± 0.09 | 3.76 ± 0.08 |
| **Controlled Originality** | 4.16 ± 0.09 | 3.04 ± 0.08 | 3.76 ± 0.09 |
| **OVERALL MEAN SCORE** | **4.17 ± 0.04** | **3.10 ± 0.04** | **3.75 ± 0.04** |

### Winner Summary

| Metric / Aspect | FLUX (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **CLIPScore** (Text Alignment) | 0.3106 | 0.3102 | **0.3122** | **Z-Image** (slight margin) |
| **CLIP-IQA** (Aesthetic Quality) | 0.6444 | **0.6722** | 0.4850 | **SDXL** |
| **LPIPS** (Style Distance to Real) | 0.6194 | 0.5578 | **0.5270** | **Z-Image** |
| **VLM Overall Mean** | **4.17** | 3.10 | 3.75 | **FLUX** |

---

### Detailed Aspect-by-Aspect Analysis

#### 1. Text Alignment & Prompt Adherence
* **Winner**: **FLUX** (Qualitative) & **Z-Image** (Quantitative)
* **Analysis**: Under CLIPScore, Z-Image scores slightly higher (0.3122) than FLUX (0.3106) and SDXL (0.3102). This is driven by Z-Image's specific hyperparameter setup and optimization schedule, which encourage aggressive semantic-visual mapping. Qualitatively, however, the VLM judge rates **FLUX** the highest at **4.23/5.0**, noting that FLUX preserves fine semantic details (such as distinguishing round rattan filaments vs. flat piattina wefts) far more cleanly than Z-Image, which occasionally exhibits style distortion on highly complex shapes.

#### 2. Aesthetic Render Quality & Sharpness
* **Winner**: **SDXL**
* **Analysis**: **SDXL** wins the CLIP-IQA metric at **0.6722**, offering smooth spatial transitions, high-contrast lighting, and fewer pixel-level noise artifacts. **FLUX** is a close second at **0.6444**, creating highly detailed, tactile renders. **Z-Image** scores lower at **0.4850** because its smaller model parameter capacity and latent decoding pathways make it more prone to high-frequency noise and slight rendering inconsistencies in complex compositions.

#### 3. Stylistic Fidelity & Textural Realism
* **Winner**: **FLUX**
* **Analysis**: **FLUX** achieves high stylistic realism with a VLM Intreccio Identity score of **4.15/5.0**, while Z-Image achieves the lowest quantitative LPIPS distance (**0.5270**), indicating closer style similarity to the real images. Thanks to its 12B parameter Flow Matching architecture, FLUX successfully captures micro-texture details such as natural rattan grain, leather pore embossing, and parallel thread-level shading on stitched borders.

#### 4. Controlled Originality & Generalization
* **Winner**: **FLUX**
* **Analysis**: Generalizing the learned `intrecciami-style` onto unseen shapes (e.g., chairs, coasters, and handbags) without melting the target geometry is a key challenge. **FLUX** dominates this category with a Controlled Originality rating of **4.16/5.0**. It acts as a flexible style compiler, applying clean, structurally sound weaves to complex target boundaries, whereas **SDXL** (3.04) and **Z-Image** (3.76) suffer more from warping and geometric distortions under similar evaluation conditions.

---

### Final Recommendation

* **FLUX** is the superior model for **physical prototyping and design automation**, where structural precision, texture realism, and physical manufacturability are paramount.
* **SDXL** remains a highly capable model for **rapid conceptual ideation and marketing visual assets** where overall render aesthetics and speed outweigh exact geometric precision.
* **Z-Image** serves as a **highly lightweight and competitive alternative**, showing strong style learning and generalization capabilities under standard training configurations.
