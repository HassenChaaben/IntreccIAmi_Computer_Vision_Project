# SDXL LoRA Training Summary

## 1. Hyperparameters & Setup

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `stabilityai/stable-diffusion-xl-base-1.0` |
| **LoRA Rank / Alpha** | 32 / 32 |
| **Target Modules** | UNet cross-attention & projection weights |
| **Hardware** | Multi-GPU (`CUDA_VISIBLE_DEVICES=1,2`), distributed via `accelerate` |
| **Learning Rate** | `1e-4` |
| **Precision** | `float32` for VAE decoding (required — see note below) |
| **dataset_repeat** | 20 (→ 177 × 20 = 3,540 instances/epoch) |
| **Dataset Size** | 177 training images |
| **Epochs Completed** | 0, 1, 2, 3 (3,540 steps/epoch → **14,160** cumulative steps at Epoch 3) |

> [!WARNING]
> **VAE float16 Instability:** The SDXL base VAE generates solid black images when run in float16 due to numerical underflow in the decoder. All inference requires `torch.float32` decoding to avoid this.
