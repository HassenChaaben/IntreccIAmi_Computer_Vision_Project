# SDXL LoRA Fine-Tuning Analysis (Epoch 1)
**Task Checkpoint:** Epoch 1 (7,080 training instances processed)  
**Architecture:** SDXL UNet LoRA (Rank 32, $\alpha = 32$)  
**Data Configuration:** 177 images, `dataset_repeat = 20`

---

## 1. Structural Consolidation
At Epoch 1, the model stabilizes the primary geometric layout, indicating that the low-rank updates are aligning with the coordinate grids of the training set.
- **Orthogonal Alignment:** Horizontal and vertical strand alignment on basic Intreccio sederhana patterns (Prompts 2 and 10) is significantly straighter, showing that the network is resolving spatial frequency grids.
- **Boundary Delineation:** The leather borders are decoupled from the weave area. The VAE decoder is beginning to resolve discrete stitching marks, though stitch intervals remain irregular.

---

## 2. Generalization Bottlenecks
- **Diagonal Perspective Distortion (Herringbone Weaves):** The model struggles with perspective grids on diagonal lines (spina weaves in Prompts 3 and 4), showing geometric drift near the margins. 
- **Attention Overlap:** Color bleeding at strand crossings is reduced but still present. The attention maps for color tokens are concentrating but still suffer from boundary leakage.

---

## 3. Training Dynamics Verdict
- **Status:** **Mid-Convergence.** The model shows stable macro-geometries but lacks micro-texture realism and sharp detail definition.
