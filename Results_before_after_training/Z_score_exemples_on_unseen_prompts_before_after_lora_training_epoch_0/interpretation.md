# Z-Image LoRA Quantitative Evaluation (Unseen Prompts - Epoch 0)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 0 (Fresh training)
- **Hyperparameter Strategy:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
- **Target Prompt Category:** Unseen (Generalization test prompts)

## Quantitative Summary
- **Average Mean Absolute Error (MAE):** 53.51
- **Average PSNR:** 11.76 dB
- **Average Structural Correlation:** 0.4730
- **Average Color Histogram Similarity:** 0.5756

## Detailed Sample Metrics (First 10 Prompts)
| Image Index | Mean Absolute Error (MAE) | PSNR (dB) | Structural Correlation | Color Hist Similarity |
|---|---|---|---|---|
| 1 | 93.67 | 6.8 | 0.1513 | 0.4726 |
| 2 | 48.02 | 12.7 | 0.3511 | 0.2612 |
| 3 | 103.41 | 6.61 | 0.1784 | 0.2261 |
| 4 | 37.28 | 13.9 | 0.4219 | 0.7049 |
| 5 | 57.16 | 11.35 | 0.4908 | 0.3955 |
| 6 | 82.53 | 7.73 | 0.1054 | 0.3673 |
| 7 | 28.65 | 16.96 | 0.7282 | 0.8243 |
| 8 | 38.69 | 14.6 | 0.3223 | 0.6033 |
| 9 | 43.88 | 12.69 | 0.7126 | 0.6037 |
| 10 | 94.96 | 6.87 | 0.2684 | 0.3704 |


## Technical Interpretation

1. **Dramatically High Style Shift (MAE and PSNR):**
   - The average MAE is **53.51** and PSNR is **11.76 dB**. This large shift indicates that the high repeat count (`dataset_repeat = 50`) forces a strong style injection (leather borders, macro photography background contrast) even on unseen prompts.
2. **Lower Generalization Correlation:**
   - The structural correlation of **0.4730** indicates significant layout changes. However, visual inspection reveals some structural stiffness, where the model forces the specific seen training pattern layouts onto the unseen prompts, leading to early over-fitting.
3. **Color Uniformity:**
   - The color histogram similarity is **0.5756**, showing that the model is heavily shifting colors to match the training set distributions.

**Conclusion:** The model trained well for style injection, but the high repeat count (50) in a single epoch restricted its generalization flexibility, showing early over-fitting on unseen prompts.
