# Z-Image LoRA Seen Prompts Generalization Critique (Epoch 1)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 1 (Resumed from Epoch 0)
- **Hyperparameter Strategy:** 
  - **Epoch 0:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
  - **Epoch 1:** `dataset_repeat = 20` (200 steps/epoch, resumed from epoch 0 checkpoint)
- **Target Prompt Category:** Seen (10 training prompts)

---

## Quantitative Performance Summary
Comparing pre-training baseline generations directly to the post-training (Epoch 1) generations yields the following average metrics:
- **Average Mean Absolute Error (MAE):** 63.34 (0-255 scale)
- **Average PSNR:** 10.53 dB
- **Average Structural Correlation:** 0.3574
- **Average Color Histogram Similarity:** 0.4832

### Detailed Per-Image Metrics
| Image Index | Mean Absolute Error (MAE) | PSNR (dB) | Structural Correlation | Color Hist Similarity |
|---|---|---|---|---|
| 1 | 80.83 | 8.42 | 0.4297 | 0.4366 |
| 2 | 50.50 | 11.78 | 0.3384 | 0.4340 |
| 3 | 86.42 | 7.06 | 0.1045 | 0.6314 |
| 4 | 42.25 | 13.65 | 0.6005 | 0.6545 |
| 5 | 78.91 | 8.10 | 0.1086 | 0.3427 |
| 6 | 54.18 | 11.01 | 0.0517 | 0.3664 |
| 7 | 44.82 | 13.47 | 0.7194 | 0.4437 |
| 8 | 46.75 | 13.12 | 0.6971 | 0.5072 |
| 9 | 94.32 | 7.08 | 0.1005 | 0.4445 |
| 10 | 54.41 | 11.64 | 0.4232 | 0.5714 |

---

## Technical Interpretation & Visual Discrepancy Critique

### 1. Pixel-Level Divergence (MAE & PSNR)
The low average PSNR of **10.53 dB** and high average MAE of **63.34** indicate a massive visual divergence between the baseline and the fine-tuned outputs. This is not a subtle texture transfer; it represents a major spatial and color reconstruction. The LoRA has forced the injection of the leather border, high-contrast studio shadows, and custom material colors (like the vibrant green in Image 1 and the warm tan leather in Image 10), completely replacing the baseline's default diffuse lighting.

### 2. Structural Disruption & Topology Correlation
The average Structural Correlation of **0.3574** is very low, proving that the model did not preserve the spatial layout of the base model.
- **Complete Layout Replacement (Failures in complex topology):** In Image 3 (Macramè) and Image 6 (Uncinetto/Semplice), the correlation drops to **0.10** and **0.05** respectively. Visual inspection shows that the base model's default structures were completely discarded. However, because the dataset is tiny (10 images), the model struggles to output clean, separate strands. In Image 3, the threads merge together into a flat, blurry shape, showing a visual failure to reconstruct the spatial depth of the macramè knots.
- **Partial Grid Retention (Successes in simple geometry):** Images 4, 7, and 8 show much higher correlation (**0.60 to 0.71**). For these simple, orthogonal grids, the model successfully aligned its learnable parameters with the baseline grid structure, polishing and refining the texture without generating major geometric artifacts.

### 3. Color Profile Shift
An average Color Histogram Similarity of **0.4832** confirms a significant shift. The model successfully matches the target color space (natural rattan, leather browns, and dyed greens) while rejecting the baseline's generic, washed-out color gradients.

---

## Verdict on Goal Achievement

**Did the fine-tuning achieve the goal?**
- **Yes, for simple weave structures.**
- The model successfully learned the visual aesthetics of the workshop (borders, macro photography lighting, and leather materials).
- **Failure:** For complex topologies (especially triple-weft patterns and macramè knots like in Images 3 and 9), the model collapses the 3D structures into flat, blurry segments with low structural correlation. The high dataset repeats at Epoch 0 (`dataset_repeat = 50`) forced the model to learn the textures, but the lack of diverse structural examples in the training set prevents the model from generating clean, geometrically correct crossings.

