# Phase 4: LoRA Tuning Hyperparameters & Model Configuration Summary

This document compiles and highlights the LoRA training configurations, dataset mappings, hardware setups, and training strategies implemented for the three backbones: **FLUX.1-dev**, **SDXL**, and **Z-Image**.

---

## 1. Hyperparameter Comparison Table

Below is a consolidated summary of the LoRA training parameters across all three fine-tuned backbones:

| Hyperparameter / Config | FLUX.1-dev (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) |
| :--- | :--- | :--- | :--- |
| **Model Size (Base)** | 12 Billion Parameters | 2.6 Billion Parameters | ~3.0 Billion Parameters |
| **Model ID / Source** | `black-forest-labs/FLUX.1-dev` | `stabilityai/stable-diffusion-xl-base-1.0` | `Tongyi-MAI/Z-Image` |
| **LoRA Rank / Alpha** | `32` / `32` | `32` / `32` | `32` / `32` |
| **Target Modules** | Double-stream blocks (QKV, projections, MLP) | UNet cross-attention & projections | Transformer blocks (`to_q, to_k, to_v, w1..w3`) |
| **Dataset Size** | 177 images | 177 images | 177 images |
| **Learning Rate** | `1e-4` | `1e-4` | `1e-4` |
| **Precision** | `bfloat16` | `float16` (except VAE in `float32`) | `float16` |
| **Hardware Setup** | Multi-GPU (`CUDA_VISIBLE_DEVICES=1,2`) | Multi-GPU (`CUDA_VISIBLE_DEVICES=1,2`) | Single-GPU (`CUDA_VISIBLE_DEVICES=0`) |
| **Training Steps** | 3,540 steps (Epochs 0–1) | 14,160 steps (Epochs 0–3) | 14,160 steps (Epochs 0–3) |

---

## 2. Model-Specific Summaries

### 2.1 FLUX.1-dev LoRA
* **Detailed Config**: [summary_flux.md](summary_flux.md)
* **Strategy & Steps**: `dataset_repeat = 20` (→ 1,770 steps per epoch across 2 GPUs). Completed Epochs 0 and 1 (3,540 cumulative steps).
* **Key Characteristic**: Utilizes Flow Matching (DiT architecture) and learns intricate artisan weave details (e.g. stitch margins, rattan filament curves) using `bfloat16` precision.

### 2.2 SDXL LoRA
* **Detailed Config**: [summary_sdxl.md](summary_sdxl.md)
* **Strategy & Steps**: `dataset_repeat = 20` (→ 3,540 steps per epoch). Completed Epochs 0, 1, 2, 3 (14,160 cumulative steps).
* **Key Characteristic**: Latent Diffusion UNet model. Requires VAE decoding in `float32` during evaluation/inference to prevent numerical underflow and solid black images.

### 2.3 Z-Image LoRA
* **Detailed Config**: [summary_zimage.md](summary_zimage.md)
* **Strategy & Steps**: `dataset_repeat = 20` (→ 3,540 steps per epoch). Completed Epochs 0, 1, 2, 3 (14,160 cumulative steps).
* **Key Characteristic**: Lightweight DiT backbone, rank 32, single-GPU training.
