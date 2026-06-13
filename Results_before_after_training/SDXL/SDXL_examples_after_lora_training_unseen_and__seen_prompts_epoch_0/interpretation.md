# SDXL LoRA Fine-Tuning Analysis (Epoch 0)
**Task Checkpoint:** Epoch 0 (3,540 training instances processed)  
**Architecture:** SDXL UNet LoRA (Rank 32, $\alpha = 32$)  
**Data Configuration:** 177 images, `dataset_repeat = 20`

---

## 1. Latent Space Representation & Compositional Priors
During this first epoch, the model establishes basic cross-attention mapping for the trigger token `intrecciami-style`. 
- **Compositional Convergence:** The U-Net's downsampling blocks (Encoder) quickly converge on the macro layout. The model successfully associates the trigger word with center-focused, macro-perspective camera angles.
- **VAE Reconstruction Bounds:** Fine details are heavily bottlenecked. The VAE's $8\times$ downsampling factor ($1024 \times 1024 \rightarrow 128 \times 128$ latents) means high-frequency patterns like stitching are lost in the compression bottleneck, leading to blurred or merged border lines.

---

## 2. Geometric & Chromatic Instability
- **Spectral Bias & Grid Distortion:** Woven grids (Intreccio) exhibit low-frequency distortions (wavy grids). This is due to the network's spectral bias, where global shape priors are learned before high-frequency spatial alignments (grid orthogonality).
- **Chromatic Mutual Information Leakage:** Significant color bleeding is visible at intersection points of contrasting materials (e.g., Prompt 5: white vs. brown). The cross-attention heatmaps for color tokens are spatially diffused, causing low-pass color blending rather than sharp boundaries.

---

## 3. Training Dynamics Verdict
- **Status:** **Early Optimization Phase.** The low-rank update matrices ($\Delta W$) are beginning to orient along the principal directions of the style subspace, but have not yet resolved fine geometric or textural details.
