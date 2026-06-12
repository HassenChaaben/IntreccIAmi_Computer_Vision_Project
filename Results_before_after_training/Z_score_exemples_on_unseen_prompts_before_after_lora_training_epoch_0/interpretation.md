# Z-Image LoRA Interpretation (Unseen Prompts - Epoch 0)

## Hyperparameters & Context
- **Base Model:** Z-Image (DiT)
- **Epoch:** 0 (Fresh training)
- **Hyperparameter Strategy:** `dataset_repeat = 50` (500 steps/epoch over 10 training images)
- **Target Prompt Category:** Unseen (10 generalization test prompts)

---

## Qualitative Analysis (Before vs. After)

### 1. Style and Texture Alignment
- **Before LoRA:** Generates stock-like weave images without cohesive styling. The trigger word `intrecciami-style` has no effect.
- **After LoRA (Epoch 0):** The style transfer is highly pronounced. The model successfully generates the distinct leather borders, macro photography depth of field, and the dark blue/white color palettes requested in the unseen prompts. 

### 2. Pattern Generalization
- The model shows an early ability to combine new materials with patterns (e.g. natural rattan posts with dark blue leather wefts).
- However, because Epoch 0 had a very high repeat count (`dataset_repeat = 50`), some unseen patterns exhibit structural stiffness. The model occasionally copies the exact spatial layouts of the seen training images rather than dynamically adapting to the new prompts.

---

## Training Success Assessment

**Did the model train well?**
- **Yes, but with signs of early overfitting.**
- The model successfully learned the aesthetic core (borders, lighting, texture feel) due to the high repeat count (50) in a single epoch.
- The high repetition accelerated learning but restricted generalization capability, making this checkpoint a strong baseline for style transfer but less flexible for diverse unseen prompts. Epoch 1's reduced repeat configuration is expected to resolve this stiffness.
