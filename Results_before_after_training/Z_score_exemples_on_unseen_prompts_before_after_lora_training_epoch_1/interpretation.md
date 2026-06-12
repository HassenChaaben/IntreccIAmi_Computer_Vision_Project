# Z-Image LoRA Interpretation (Unseen Prompts - Epoch 1)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 1 (Resumed from Epoch 0)
- **Hyperparameter Strategy:** 
  - **Epoch 0:** `dataset_repeat = 50`
  - **Epoch 1:** `dataset_repeat = 20`
- **Target Prompt Category:** Unseen (10 generalization test prompts)

---

## Qualitative Analysis (Before vs. After)

### 1. Style and Texture Alignment
- **Before LoRA:** Stock-like patterns lacking consistent branding, leather borders, or professional macro-lighting.
- **After LoRA (Epoch 1):** The style is fully integrated. The colors, leather borders, and textures blend seamlessly. Rattan and leather look natural and exhibit organic variations.

### 2. Pattern Generalization
- Compared to Epoch 0, the structural stiffness has decreased significantly. 
- The model generalizes much better to unseen prompt combinations:
  - For `Intreccio semplice` with dark blue wefts and rattan posts, the weaving is clean, uniform, and physically plausible.
  - For `Macramè / Vario` and `Rinfilo / Uncinetto`, the model successfully renders complex knots and crochet structures in leather without distorting.

---

## Training Success Assessment

**Did the model train well?**
- **Yes, the epoch-1 strategy was highly successful.**
- Lowering the repeat count from `50` (Epoch 0) to `20` (Epoch 1) allowed the model to generalize. It successfully retained the aesthetic identity (`intrecciami-style`) learned in Epoch 0 while softening the weights to prevent copying exact layouts.
- This results in a much more flexible and versatile model that can generate new and unique artisan woven designs that look realistic and manufacturable.
