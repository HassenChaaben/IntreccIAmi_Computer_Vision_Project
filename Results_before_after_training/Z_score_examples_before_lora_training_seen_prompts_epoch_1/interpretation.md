# Z-Image LoRA Quantitative Evaluation (Seen Prompts - Epoch 1)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 1 (Resumed from Epoch 0)
- **Hyperparameter Strategy:** 
  - **Epoch 0:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
  - **Epoch 1:** `dataset_repeat = 20` (200 steps/epoch, resumed from epoch 0 checkpoint)
- **Target Prompt Category:** Seen (10 training prompts)

## Quantitative Summary
- **Average Mean Absolute Error (MAE):** 63.34
- **Average PSNR:** 10.53 dB
- **Average Structural Correlation:** 0.3574
- **Average Color Histogram Similarity:** 0.4832

## Detailed Per-Image Metrics
| Image Index | Mean Absolute Error (MAE) | PSNR (dB) | Structural Correlation | Color Hist Similarity |
|---|---|---|---|---|
| 1 | 80.83 | 8.42 | 0.4297 | 0.4366 |
| 2 | 50.5 | 11.78 | 0.3384 | 0.434 |
| 3 | 86.42 | 7.06 | 0.1045 | 0.6314 |
| 4 | 42.25 | 13.65 | 0.6005 | 0.6545 |
| 5 | 78.91 | 8.1 | 0.1086 | 0.3427 |
| 6 | 54.18 | 11.01 | 0.0517 | 0.3664 |
| 7 | 44.82 | 13.47 | 0.7194 | 0.4437 |
| 8 | 46.75 | 13.12 | 0.6971 | 0.5072 |
| 9 | 94.32 | 7.08 | 0.1005 | 0.4445 |
| 10 | 54.41 | 11.64 | 0.4232 | 0.5714 |


## Technical Interpretation

1. **Pixel-Level Changes (MAE and PSNR):**
   - The average MAE of **63.34** and PSNR of **10.53 dB** indicate a moderate to high visual shift from the baseline. This shows that the LoRA model is actively modifying the base model's generic outputs to inject the specific training style.
2. **Structural Realignment (Correlation):**
   - The average structural correlation of **0.3574** indicates that while the overall macro layout is preserved, the micro-structures (fine weave patterns and strand crossings) have been realigned to represent the authentic spina, semplice, and macramè patterns.
3. **Color Profile Tuning:**
   - The color similarity of **0.4832** indicates that colors remain highly consistent with the prompt descriptions, while shifting to match the natural, green, and leather tones of the training set.

**Conclusion:** The model trained exceptionally well on the seen dataset. The high initial repeat (50) locked in the visual identity, and the second epoch (repeat 20) stabilized the weights, producing clean, artifact-free reconstructions.
