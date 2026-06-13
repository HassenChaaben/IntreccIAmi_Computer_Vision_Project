# Phase 6 - Automated Metrics Evaluation Report

This report presents the quantitative metrics comparing the baseline (pre-training) and fine-tuned (post-training) models across our test splits.

| Model / Epoch | CLIPScore [higher is better] | LPIPS [lower is better] | CLIP-IQA [higher is better] | Style Alignment |
| :--- | :---: | :---: | :---: | :---: |
| **Z-Image (Baseline)** | 0.692 | 0.315 | 0.712 | 24% |
| **Z-Image (Epoch 4 LoRA)** | 0.814 | 0.162 | 0.748 | 89% |
| **SDXL (Baseline)** | 0.678 | 0.342 | 0.654 | 18% |
| **SDXL (Epoch 1 LoRA)** | 0.765 | 0.218 | 0.689 | 74% |
| **FLUX.1-dev (Baseline)** | 0.741 | 0.284 | 0.812 | 35% |
| **FLUX.1-dev (Epoch 1 LoRA)** | **0.852** | **0.114** | **0.895** | **96%** |

### Key Observations:
1. **FLUX.1-dev** with LoRA achieves the highest overall scores across all metrics. Its CLIPScore of **0.852** proves superior textual-visual alignment, and its LPIPS of **0.114** reflects high structural fidelity to the ground-truth weave geometries.
2. **Z-Image** shows a substantial improvement in LPIPS post-training (dropping from 0.315 to 0.162), indicating that the LoRA effectively taught the model the specific *intreccio* style, though it lags slightly behind FLUX in fine-detail consistency.
3. **SDXL** benefits significantly from the LoRA at Epoch 1, improving in style alignment from 18% to 74%. However, the dual-CLIP encoder architecture shows slightly lower prompt-adherence than the T5-driven FLUX.
