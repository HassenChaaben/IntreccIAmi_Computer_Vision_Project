# Z-Image LoRA Training Summary

## 1. Hyperparameters & Setup

| Parameter | Value |
| :--- | :--- |
| **Base Model** | Z-Image (DiT architecture) |
| **LoRA Rank / Alpha** | 32 / 32 |
| **Learning Rate** | `1e-4` |
| **Training Dataset** | 177 training images |
| **Epochs 0–3 Strategy** | `dataset_repeat = 20` (→ 3,540 steps/epoch ) |
| **Epochs Completed** | Epoch 0, 1, 2, 3 (4 epochs total) |
| **Total Cumulative Steps** | 3,540 × 4 = **14,160** steps |
