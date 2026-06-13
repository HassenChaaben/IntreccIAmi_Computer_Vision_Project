# Z-Image LoRA Seen & Unseen Generalization Critique (Epoch 4)

## Executive Summary
This report evaluates the visual and structural quality of the images generated at **Epoch 4**, the final checkpoint of the Z-Image (DiT) LoRA training. The model was trained using a structured curriculum: **Epoch 0** initiated learning with a high density of `dataset_repeat = 50` (500 steps) to force style acquisition, followed by **Epochs 1 to 4** at a stabilized `dataset_repeat = 20` (200 steps/epoch). 

By Epoch 4, the model has completed a total of **1,300 steps** on the 10-image dataset. Visual inspection of both the 10 Seen and 10 Unseen prompts reveals a highly mature style representation. The model consistently injects the `intrecciami-style` aesthetic—characterized by high-contrast studio lighting, rich leather borders, clean stitching, and high-frequency tactile strand details. However, a clear divergence remains between memorized seen geometries and generalized unseen configurations.

---

## Technical Evaluation: Seen Prompts (1–10)
The Seen Prompts test the model's ability to reconstruct the training set. At Epoch 4, reconstruction fidelity is exceptionally high, showing the model has successfully memorized the target distributions:

### 1. Structural & Geometric Alignment
- **Successes:** 
  - Orthogonal grids (such as the simple over-under in Prompt 2 and Prompt 10) are extremely straight. Strand widths are uniform, and the spacing matches the prompt specification (e.g., 2mm round filaments with flat 4mm piattina wefts).
  - The stitched leather borders (beige/tan) are beautifully aligned, showing straight stitching lines and realistic leather texture embossment.
- **Failures / Residual Artifacts:**
  - In Prompt 3 (Macramé Vario with black leather mignon), while the knot structures are far crisper than at Epoch 1, there is still minor structural merging. The square knots (Nodo quadro) tend to blend slightly at their overlapping edges, creating continuous organic shapes rather than physically distinct knots.

### 2. Material & Color Saturation
- **Successes:**
  - Color rendering is precise and vibrant. The "Verde" (green) in Prompt 1 and Prompt 2 is saturated correctly with natural gradient variations that mimic real-world studio lighting.
  - The contrast between natural rush bark and orange leather in Prompt 4 is clean, with minimal cross-material color bleeding.

---

## Technical Evaluation: Unseen Prompts (1–10)
The Unseen Prompts evaluate the model's generalization limits when combining known style features with new prompt instructions:

### 1. Weave Technique Generalization
- **Intreccio Semplice (Prompts 1 & 2):**
  - *Success:* Excellent translation of the style. Prompt 1 (natural rattan posts with dark blue leather wefts) successfully renders the distinct material textures and maintains a clean, flat profile.
  - *Success:* Prompt 2 (white leather mignon with gloss finish) shows highly convincing specular highlights (Verniciatura gloss) on the strand surfaces.
- **Intreccio Spina & Diagonal Weaves (Prompts 3 & 4):**
  - *Success:* Prompt 3 (deep black rattan in diagonal herringbone layout) exhibits excellent matte (Grezzo) finish control.
  - *Failure:* The diagonal alignment in Prompt 4 (red leather wefts on split rattan) shows spatial instability. Instead of a rigid, parallel diagonal grid, the strands warp slightly towards the margins, suggesting that the model struggles to maintain perspective alignment on diagonal lines when introducing new color contrasts.
- **Pattern Scacco Checkerboards (Prompts 5 & 6):**
  - *Success:* The checkerboard patterns (white/blue leather in Prompt 5 and rattan/black leather in Prompt 6) are visually striking. The alternating colors are well-delineated.
  - *Failure:* Minor grid deformation occurs at the intersections. The square tiles are not perfectly rectangular; some corners exhibit rounding or small structural overlaps.
- **Macramé & Vario (Prompts 7 & 8):**
  - *Success:* Renders thick black leather strands (Prompt 7) and white leather mignon (Prompt 8) with deep, realistic drop shadows that create strong 3D depth.
  - *Failure:* Similar to the training set, the knot topologies are simplified. The model fails to generate true interlocking physical knots, representing them instead as overlapping curved bands.
- **Rinfilo & Uncinetto (Prompts 9 & 10):**
  - *Success:* In Prompt 9, the underlying "Paglia di Vienna" (Vienna straw) pattern is recognizable, showing that the base model's pre-trained textures have not been entirely erased by the LoRA.
  - *Failure:* The crochet pattern in Prompt 10 (thick black leather grid) collapses into a generic weave grid rather than showing authentic looped crochet geometry, pointing to a resolution bottleneck in the model's texture decoder.

---

## Epoch Progression Analysis (Epoch 1 vs. Epoch 4)
- **Grid Stability:** The waviness and chaotic grid distortions that plagued Epoch 1 have stabilized. The additional 600 steps of training at lower repeat rates (`dataset_repeat = 20`) allowed the model to refine its spatial weights, leading to straighter lines and cleaner intersections.
- **VAE Reconstruction:** High-frequency details (such as rattan grain and leather stitches) are noticeably sharper. The blurry patches at diagonal crossings observed in Epoch 1 are mostly resolved.
- **Style Consistency:** The presence of the beige leather border is more consistent across all prompts, demonstrating that the `intrecciami-style` visual anchor is firmly embedded.

---

## Verdict on Goal Achievement
**Did the fine-tuning achieve the goal?**
- **Yes, with high performance.**
- The model successfully learned the custom branding, textures (rattan, leather, stitching), and lighting. It demonstrates a high-level ability to generalize these styles to unseen prompt combinations (such as black, blue, and red leather wefts) while maintaining structural coherence.
- **Limitations:** The physical manufacturability constraint is only partially satisfied. While orthogonal grids are physically plausible, complex diagonal (spina) and knotted (macramè, uncinetto) structures are generated as visual approximations rather than mechanically correct weaves.
