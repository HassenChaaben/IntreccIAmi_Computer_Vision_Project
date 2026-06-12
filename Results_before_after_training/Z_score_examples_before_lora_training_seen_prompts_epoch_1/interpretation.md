# Z-Image LoRA Interpretation (Seen Prompts - Epoch 1)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 1 (Resumed from Epoch 0)
- **Hyperparameter Strategy:** 
  - **Epoch 0:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
  - **Epoch 1:** `dataset_repeat = 20` (200 steps/epoch, resumed from epoch 0 checkpoint)
- **Target Prompt Category:** Seen (10 training prompts)

---

## Qualitative Analysis (Before vs. After)

### 1. Style and Texture Alignment
- **Before LoRA:** The base model generated high-quality woven textures, but they were generic, often resembling stock photography of random straw or generic meshes without respecting the exact structure of Italian artisan weaves.
- **After LoRA (Epoch 1):** The model shows a clear, precise style transfer. The generated images perfectly align with the `intrecciami-style` prefix:
  - Clear **leather borders** with realistic stitching.
  - Symmetrical grid structures for simple weaves.
  - Convincing leather mignon and rattan material textures matching the prompts.

### 2. Pattern Compliance
- Specific complex patterns requested in the seen prompts (such as `Intreccio spina salto 2`, `Intreccio semplice con trama tripla`, and `Macramè` knot structures) are rendered with high structural fidelity.
- The double posts and wefts interlacing pattern in `spina salto 2` matches the training target closely, proving the LoRA successfully memorized and mapped these concepts to their descriptors.

---

## Training Success Assessment

**Did the model train well?**
- **Yes, exceptionally well.**
- By starting with `dataset_repeat = 50` in Epoch 0, the model quickly learned the visual aesthetics of the dataset (the leather borders, background contrast, and studio lighting). 
- Lowering the repeat to `20` in Epoch 1 stabilized the weights and prevented the model from collapsing into high-contrast or highly distorted patterns. The resulting textures are crisp, well-structured, and highly detailed, showing no signs of over-saturation.
