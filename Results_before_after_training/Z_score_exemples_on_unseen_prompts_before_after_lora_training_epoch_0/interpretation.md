# Z-Image LoRA Unseen Generalization Critique (Epoch 0)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 0 (Fresh training)
- **Hyperparameter Strategy:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
- **Target Prompt Category:** Unseen (Generalization test prompts)

---

## Quantitative Performance Summary
Comparing pre-training baseline generations directly to the post-training (Epoch 0) generations yields the following average metrics:
- **Average Mean Absolute Error (MAE):** 53.51 (0-255 scale)
- **Average PSNR:** 11.76 dB
- **Average Structural Correlation:** 0.4730
- **Average Color Histogram Similarity:** 0.5756

### Detailed Sample Metrics (First 10 Prompts)
| Image Index | Mean Absolute Error (MAE) | PSNR (dB) | Structural Correlation | Color Hist Similarity |
|---|---|---|---|---|
| 1 | 93.67 | 6.80 | 0.1513 | 0.4726 |
| 2 | 48.02 | 12.70 | 0.3511 | 0.2612 |
| 3 | 103.41 | 6.61 | 0.1784 | 0.2261 |
| 4 | 37.28 | 13.90 | 0.4219 | 0.7049 |
| 5 | 57.16 | 11.35 | 0.4908 | 0.3955 |
| 6 | 82.53 | 7.73 | 0.1054 | 0.3673 |
| 7 | 28.65 | 16.96 | 0.7282 | 0.8243 |
| 8 | 38.69 | 14.60 | 0.3223 | 0.6033 |
| 9 | 43.88 | 12.69 | 0.7126 | 0.6037 |
| 10 | 94.96 | 6.87 | 0.2684 | 0.3704 |

---

## Technical Interpretation & Visual Discrepancy Critique

### 1. High Style Shift & Spatial Memorization (MAE & PSNR)
An average MAE of **53.51** and PSNR of **11.76 dB** indicate a significant pixel-level change. The high repeat count of `50` at Epoch 0 immediately forces the model to inject the `intrecciami-style` visual components (the leather frames and high-contrast studio shadows) onto the outputs. However, the pixel shift is highly uneven:
- **Reconstruction Failure (Images 3 and 10):** Images 3 and 10 exhibit extremely low PSNR (**6.61 dB** and **6.87 dB**) and massive MAE values (**103.41** and **94.96**). This represents a complete breakdown of the baseline image layout. Instead of generating a clean checkerboard or diagonal pattern, the model forced a highly distorted, high-frequency noise resembling disorganized wicker-work, showing a failure to properly map the unseen geometry in these specific samples.

### 2. Structural Overfitting vs. Rejection (Correlation)
The average Structural Correlation of **0.4730** is relatively high compared to Epoch 1 (which drops to **0.3574**). This higher correlation indicates that the model is still caught in a state of conflict:
- **Style Rejection (Image 7):** Image 7 exhibits a very high correlation of **0.7282**, low MAE (**28.65**), and high PSNR (**16.96 dB**). Here, the LoRA failed to modify the baseline's structural layout. The model simply applied a faint color overlay, showing a complete failure to learn the new weave geometry for this prompt.
- **Overfitting & Distortion (Images 1 and 6):** For Images 1 and 6, the correlation is very low (**0.15** and **0.10**). The model successfully overrode the baseline structure, but because the repeat count is so high (50) for a tiny dataset, the generated strands are wavy, lack uniform spacing, and merge into one another, indicating early overfitting.

---

## Verdict on Goal Achievement

**Did the fine-tuning achieve the goal?**
- **No, not at this epoch.**
- While Epoch 0 successfully injects style textures, the high repeat count of 50 causes the model to either completely reject the style (remaining too close to the baseline, as in Image 7) or overfit and generate highly distorted, non-manufacturable geometries (as in Images 3 and 10).
- **Recommendation:** Epoch 0 is too rigid. Reducing the repeat count to 20 in Epoch 1 is necessary to regularize the model and smooth out the structural distortions.
