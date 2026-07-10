# FLUX LoRA Training Summary

## 1. Hyperparameters & Setup

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `black-forest-labs/FLUX.1-dev` |
| **Text Encoders** | T5-XXL + CLIP ViT-L |
| **Hardware** | Multi-GPU (`CUDA_VISIBLE_DEVICES=1,2`), distributed via `accelerate` |
| **LoRA Rank / Alpha** | 32 / 32 |
| **Target Modules** | Cross-attention & Feed-Forward projections (double-stream blocks) |
| **Learning Rate** | `1e-4` |
| **Precision** | `bfloat16` |
| **dataset_repeat** | 20 (→ 3,540 instances/epoch) |
| **Dataset Size** | 177 training images |
| **Epochs Completed** | Epoch 0 (1,770 steps) → Epoch 1 (3,540 cumulative steps) |
