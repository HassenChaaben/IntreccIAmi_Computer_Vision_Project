# Performance Metrics & Evaluation Summary Comparison

This document compiles the summary statistics and average evaluation tables for FLUX, SDXL, and Z-Image without any interpretation.

## 1. Combined Quantitative Metrics Comparison

| Model | CLIPScore (Text Alignment) | CLIP-IQA (Aesthetic Quality) | LPIPS (Style Distance to Real) |
| :--- | :---: | :---: | :---: |
| **FLUX (DiT)** | 0.3106 ± 0.0312 | 0.6444 ± 0.2301 | 0.6194 ± 0.0939 |
| **SDXL (Latent Diffusion)** | 0.3102 ± 0.0340 | 0.6722 ± 0.1195 | 0.5578 ± 0.0778 |
| **Z-Image (DiT)** | 0.3122 ± 0.0313 | 0.4850 ± 0.2542 | 0.5270 ± 0.1227 |

---

## 2. Combined Qualitative MLLM-as-a-Judge Ratings Comparison (Average / 5.0)

| Evaluation Criteria | FLUX (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) |
| :--- | :---: | :---: | :---: |
| **Prompt Adherence** | 4.23 ± 0.07 | 3.24 ± 0.07 | 3.83 ± 0.07 |
| **Intreccio Identity** | 4.15 ± 0.09 | 3.06 ± 0.09 | 3.75 ± 0.09 |
| **Manufacturability** | 4.05 ± 0.09 | 2.94 ± 0.09 | 3.65 ± 0.09 |
| **Visual Quality** | 4.26 ± 0.08 | 3.23 ± 0.09 | 3.76 ± 0.08 |
| **Controlled Originality** | 4.16 ± 0.09 | 3.04 ± 0.08 | 3.76 ± 0.09 |
| **OVERALL MEAN SCORE** | **4.17 ± 0.04** | **3.10 ± 0.04** | **3.75 ± 0.04** |

