# Z-Image LoRA Results & Training Analysis

This document provides a comprehensive visual and technical analysis of all generated results for the **Z-Image (DiT)** model. It covers the full training curriculum — Epoch 0 (`dataset_repeat = 50`) and Epochs 1–4 (`dataset_repeat = 20`) — with quantitative metrics (MAE, PSNR, Structural Correlation, Color Histogram Similarity) and per-prompt image comparisons.

> All result folders live under: `Results_before_after_training/Z_score/`

---

## 1. Hyperparameters & Setup

| Parameter | Value |
| :--- | :--- |
| **Base Model** | Z-Image (DiT architecture) |
| **LoRA Rank / Alpha** | 32 / 32 |
| **Training Dataset** | 10 images (seen prompts only) |
| **Epoch 0 strategy** | `dataset_repeat = 50` → 500 steps (force style acquisition) |
| **Epochs 1–4 strategy** | `dataset_repeat = 20` → 200 steps/epoch (regularization) |
| **Total Steps (Epoch 4)** | 500 + (200 × 4) = **1,300 steps** |
| **Validation Set** | 10 seen + 50 unseen prompts |

> [!IMPORTANT]
> Z-Image uses only **10 training images** (vs. 177 for FLUX/SDXL). This makes it far more susceptible to overfitting at high `dataset_repeat`, which is the core design challenge of this training curriculum.

---

## 2. Folder Structure

| Folder | Contents | Epoch | Critique |
| :--- | :--- | :---: | :--- |
| [Z_score_examples_before_lora_training_seen_prompts/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts) | 10 seen baseline images | — | — |
| [Z_score_examples_before_lora_training_unseen_prompts/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts) | 50 unseen baseline images | — | — |
| [Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0) | 50 unseen after LoRA (sub-folder `after_unseen/`) | 0 | [interpretation.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/interpretation.md) |
| [Z_score_examples_after_lora_training_seen_prompts_epoch_1/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1) | 10 seen after LoRA | 1 | [interpretation.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/interpretation.md) |
| [Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1) | 10 seen + 10 unseen after LoRA | 1 | [interpretation.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/interpretation.md) |
| [Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4) | 10 seen + 10 unseen after LoRA | 4 | [interpretation.md](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/interpretation.md) |

---

## 3. Quantitative Metrics Summary

### Seen Prompts — Epoch 1 (10 training images)

> Avg MAE: **63.34** | Avg PSNR: **10.53 dB** | Avg Structural Corr: **0.3574** | Avg Color Hist: **0.4832**

| # | MAE | PSNR (dB) | Struct. Corr | Color Hist | Interpretation |
| :---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 80.83 | 8.42 | 0.43 | 0.44 | Large style shift; baseline layout fully replaced by leather border. |
| 2 | 50.50 | 11.78 | 0.34 | 0.43 | Moderate shift; glossy green rattan successfully injected. |
| 3 | 86.42 | 7.06 | **0.10** | 0.63 | Macramé collapse — threads merge into flat blurry shapes. Low corr = complete layout change. |
| 4 | 42.25 | 13.65 | **0.60** | 0.65 | Best seen result — simple orthogonal grid retained and polished. |
| 5 | 78.91 | 8.10 | 0.11 | 0.34 | Checkerboard partially reconstructed; color bleeding at crossings. |
| 6 | 54.18 | 11.01 | **0.05** | 0.37 | Lowest corr — baseline structure fully discarded. Border injection succeeded. |
| 7 | 44.82 | 13.47 | **0.72** | 0.44 | High corr — simple juniper bark grid preserved and micro-textured. |
| 8 | 46.75 | 13.12 | **0.70** | 0.51 | High corr — bamboo herringbone stable and correctly rendered. |
| 9 | 94.32 | 7.08 | 0.10 | 0.44 | Triple-weft collapse — blue and natural strands merge. |
| 10 | 54.41 | 11.64 | 0.42 | 0.57 | Moderate reconstruction; round filament vs. piattina not resolved. |

### Unseen Prompts — Epoch 0 (first 10 of 50 prompts)

> Avg MAE: **53.51** | Avg PSNR: **11.76 dB** | Avg Structural Corr: **0.4730** | Avg Color Hist: **0.5756**

| # | MAE | PSNR (dB) | Struct. Corr | Color Hist | Interpretation |
| :---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 93.67 | 6.80 | 0.15 | 0.47 | Style fully overrode baseline; wavy, overfitted rattan. |
| 2 | 48.02 | 12.70 | 0.35 | 0.26 | White gloss grid partially formed; low color match. |
| 3 | 103.41 | **6.61** | 0.18 | 0.23 | Worst result — complete geometric breakdown; disorganized noise. |
| 4 | 37.28 | 13.90 | 0.42 | **0.70** | Best result — black fettuccia grid stable; good color match. |
| 5 | 57.16 | 11.35 | 0.49 | 0.40 | Brown/white contrast present; color bleeding moderate. |
| 6 | 82.53 | 7.73 | 0.11 | 0.37 | Fuchsia color injected; structure collapsed. |
| 7 | **28.65** | **16.96** | **0.73** | **0.82** | Style rejection — LoRA barely modified the baseline; highest corr. |
| 8 | 38.69 | 14.60 | 0.32 | 0.60 | Gray strobel grid partially formed; muted color. |
| 9 | 43.88 | 12.69 | **0.71** | 0.60 | Checkerboard aligned; gloss not resolved. |
| 10 | 94.96 | 6.87 | 0.27 | 0.37 | High distortion — geometric breakdown for orange/rush bark combo. |

---

## 4. Baseline — Before LoRA Training

### 4.1 Seen Prompts Baseline (10 prompts)

| # | Baseline Image | Caption |
| :---: | :--- | :--- |
| **1** | [before_seen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_1.png) | *Intreccio spina salto 2 — natural rattan, Verde green dye, Grezzo finish, beige leather border with stitches.* |
| **2** | [before_seen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_2.png) | *Intreccio semplice — natural split rattan posts & wefts, 3 mm, deca green, glossy verniciatura finish.* |
| **3** | [before_seen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_3.png) | *Macramé Vario — square knots (Nodo quadro), black leather mignon strands, 9 mm.* |
| **4** | [before_seen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_4.png) | *Intreccio semplice & trama doppia — Corteccia di Giunco posts (2.5 mm), orange Pelle Mignon wefts, Grezzo finish.* |
| **5** | [before_seen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_5.png) | *Intreccio semplice — white & light brown leather (4 mm), checkerboard, posts spaced 9 mm.* |
| **6** | [before_seen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_6.png) | *Intreccio semplice — white leather mignon (4 mm), posts spaced 9 mm.* |
| **7** | [before_seen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_7.png) | *Intreccio semplice — juniper bark posts (6 mm), light yellow wefts (9 mm), Grezzo finish.* |
| **8** | [before_seen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_8.png) | *Intreccio spina salto 2 — natural bamboo strips (3 mm), Grezzo finish.* |
| **9** | [before_seen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_9.png) | *Intreccio semplice con trama tripla — dark blue & natural round filaments (2 mm), posts 17 mm apart.* |
| **10** | [before_seen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_10.png) | *Intreccio semplice — round rattan filaments (2 mm) posts, flat piattina wefts (4 mm).* |

### 4.2 Unseen Prompts Baseline (50 prompts — Groups A–E)

#### Group A — Intreccio Semplice (1–10)

| # | Baseline | Caption |
| :---: | :--- | :--- |
| **1** | [before_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_1.png) | *Intreccio semplice — natural rattan posts, dark blue leather wefts, Grezzo finish.* |
| **2** | [before_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_2.png) | *Intreccio semplice — white leather mignon strands, verniciatura gloss finish.* |
| **3** | [before_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_3.png) | *Intreccio semplice — red split rattan posts, natural rush bark wefts, Grezzo finish.* |
| **4** | [before_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_4.png) | *Intreccio semplice — thick black leather fettuccia, verniciatura opaca finish.* |
| **5** | [before_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_5.png) | *Intreccio semplice — light brown leather mignon posts, white wefts, Grezzo finish.* |
| **6** | [before_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_6.png) | *Intreccio semplice — fuchsia synthetic plastic posts, white leather wefts.* |
| **7** | [before_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_7.png) | *Intreccio semplice — natural midollino round filaments, marrone chiaro dye, Grezzo finish.* |
| **8** | [before_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_8.png) | *Intreccio semplice — gray leather strobel strands.* |
| **9** | [before_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_9.png) | *Intreccio semplice — yellow rattan & black leather wefts, checkerboard, verniciatura gloss.* |
| **10** | [before_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_10.png) | *Intreccio semplice — natural rush bark posts, orange leather wefts, Grezzo finish.* |

#### Group B — Intreccio Spina & Spina Salto 2 (11–20)

| # | Baseline | Caption |
| :---: | :--- | :--- |
| **11** | [before_unseen_11.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_11.png) | *Intreccio spina salto 2 — natural rattan, deep black dye, Grezzo finish.* |
| **12** | [before_unseen_12.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_12.png) | *Intreccio spina — natural split rattan posts, red leather wefts.* |
| **13** | [before_unseen_13.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_13.png) | *Intreccio spina salto 2 — double posts of blue leather, natural wefts, Grezzo finish.* |
| **14** | [before_unseen_14.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_14.png) | *Intreccio spina — white leather mignon, natural rush bark, diagonal herringbone.* |
| **15** | [before_unseen_15.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_15.png) | *Intreccio spina salto 2 — green midollino round filaments, dark brown wefts.* |
| **16** | [before_unseen_16.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_16.png) | *Intreccio spina — natural bamboo bark, gloss verniciatura finish.* |
| **17** | [before_unseen_17.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_17.png) | *Intreccio spina salto 2 — gray synthetic plastic, black leather.* |
| **18** | [before_unseen_18.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_18.png) | *Intreccio spina — natural rattan posts, red split rattan wefts, Grezzo finish.* |
| **19** | [before_unseen_19.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_19.png) | *Intreccio spina salto 2 — white leather fettuccia.* |
| **20** | [before_unseen_20.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_20.png) | *Intreccio spina — orange leather, natural midollino strands, Grezzo finish.* |

#### Group C — Pattern Scacco & Jacquard (21–30)

| # | Baseline | Caption |
| :---: | :--- | :--- |
| **21** | [before_unseen_21.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_21.png) | *Pattern scacco — white & dark blue leather strands, tight grid.* |
| **22** | [before_unseen_22.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_22.png) | *Pattern scacco — natural rattan & black leather, Grezzo finish.* |
| **23** | [before_unseen_23.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_23.png) | *Jacquard — fuchsia & white leather strands, repeating geometric grid.* |
| **24** | [before_unseen_24.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_24.png) | *Pattern scacco — natural rush bark & yellow leather.* |
| **25** | [before_unseen_25.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_25.png) | *Jacquard — red split rattan & natural midollino, verniciatura gloss.* |
| **26** | [before_unseen_26.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_26.png) | *Pattern scacco — gray leather mignon & white leather fettuccia.* |
| **27** | [before_unseen_27.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_27.png) | *Jacquard — green midollino round filaments, natural rattan wefts, Grezzo finish.* |
| **28** | [before_unseen_28.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_28.png) | *Pattern scacco — black & white leather strobel strands.* |
| **29** | [before_unseen_29.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_29.png) | *Jacquard — alternating orange leather & natural bamboo bark bands.* |
| **30** | [before_unseen_30.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_30.png) | *Pattern scacco — blue synthetic plastic & white leather.* |

#### Group D — Macramè & Vario (31–40)

| # | Baseline | Caption |
| :---: | :--- | :--- |
| **31** | [before_unseen_31.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_31.png) | *Macramè Vario — thick black leather strands, complex knot pattern.* |
| **32** | [before_unseen_32.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_32.png) | *Macramè — square knots in white leather mignon, Grezzo finish.* |
| **33** | [before_unseen_33.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_33.png) | *Vario — natural rush bark cords & brown leather.* |
| **34** | [before_unseen_34.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_34.png) | *Macramè — vertical knots in red & black leather fettuccia.* |
| **35** | [before_unseen_35.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_35.png) | *Vario — natural rattan fibers & blue leather mignon.* |
| **36** | [before_unseen_36.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_36.png) | *Macramè — green leather strobel & natural midollino.* |
| **37** | [before_unseen_37.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_37.png) | *Vario — yellow leather & white leather fettuccia, square knots.* |
| **38** | [before_unseen_38.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_38.png) | *Macramè — natural bamboo bark strands, Vario pattern, Grezzo finish.* |
| **39** | [before_unseen_39.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_39.png) | *Vario — orange leather & gray synthetic plastic.* |
| **40** | [before_unseen_40.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_40.png) | *Macramè — white leather mignon & natural split rattan.* |

#### Group E — Rinfilo & Uncinetto (41–50)

| # | Baseline | Caption |
| :---: | :--- | :--- |
| **41** | [before_unseen_41.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_41.png) | *Rinfilo — Paglia di Vienna base, beige leather fettuccia & mignon, Grezzo finish.* |
| **42** | [before_unseen_42.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_42.png) | *Uncinetto — thick black leather strands, repeating crochet grid.* |
| **43** | [before_unseen_43.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_43.png) | *Rinfilo — white leather mignon woven through Paglia di Vienna.* |
| **44** | [before_unseen_44.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_44.png) | *Uncinetto — natural rattan fibers, tight crochet pattern, Grezzo finish.* |
| **45** | [before_unseen_45.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_45.png) | *Rinfilo — red & black leather fettuccia on Paglia di Vienna.* |
| **46** | [before_unseen_46.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_46.png) | *Uncinetto — green midollino round filaments, verniciatura gloss finish.* |
| **47** | [before_unseen_47.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_47.png) | *Rinfilo — yellow leather mignon & natural rush bark on Paglia di Vienna.* |
| **48** | [before_unseen_48.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_48.png) | *Uncinetto — blue leather strobel crochet texture.* |
| **49** | [before_unseen_49.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_49.png) | *Rinfilo — orange leather fettuccia woven through Paglia di Vienna, Grezzo finish.* |
| **50** | [before_unseen_50.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_50.png) | *Uncinetto — white leather mignon crochet design.* |

---

## 5. Epoch 0 — Unseen Prompts After LoRA (500 steps, dataset_repeat=50)

> **Status:** Aggressive style injection. High repeat count forces leather border and studio lighting onto all outputs, but causes geometric distortions and complete structural breakdowns on complex patterns (Images 3, 10). Model either over-applies style (overfitting/distortion) or ignores it entirely (Image 7: style rejection). Verdict: **Not yet achieved.**

> [!WARNING]
> The epoch 0 `after_unseen` images are stored in the sub-folder `after_unseen/` inside `Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/`.

| # | Before LoRA | After LoRA (Epoch 0) | Caption | Analysis |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [before_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_1.png) | [after_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_1.png) | *Intreccio semplice — natural rattan posts, dark blue leather wefts, Grezzo finish.* | Style injected; wavy rattan grid; overfitted at Epoch 0. |
| **2** | [before_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_2.png) | [after_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_2.png) | *Intreccio semplice — white leather mignon, verniciatura gloss finish.* | White grid partially injected; low color match (0.26). |
| **3** | [before_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_3.png) | [after_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_3.png) | *Intreccio semplice — red split rattan posts, natural rush bark wefts, Grezzo finish.* | **Worst (MAE 103.41, PSNR 6.61 dB)** — complete geometric breakdown; disorganized noise output. |
| **4** | [before_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_4.png) | [after_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_4.png) | *Intreccio semplice — thick black leather fettuccia, verniciatura opaca finish.* | **Best (MAE 37.28, Color 0.70)** — simple black grid stable; border injection successful. |
| **5** | [before_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_5.png) | [after_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_5.png) | *Intreccio semplice — light brown leather mignon, white wefts, Grezzo finish.* | Color bleeding present; cross-attention leakage into brown posts. |
| **6** | [before_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_6.png) | [after_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_6.png) | *Intreccio semplice — fuchsia synthetic plastic posts, white leather wefts.* | Fuchsia color injected; structure collapsed (Struct. Corr 0.11). |
| **7** | [before_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_7.png) | [after_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_7.png) | *Intreccio semplice — natural midollino, marrone chiaro dye, Grezzo finish.* | **Style rejection** (PSNR 16.96, Corr 0.73) — LoRA barely modified baseline; faint color overlay only. |
| **8** | [before_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_8.png) | [after_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_8.png) | *Intreccio semplice — gray leather strobel strands.* | Gray grid partially formed; muted rendering (Color 0.60). |
| **9** | [before_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_9.png) | [after_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_9.png) | *Intreccio semplice — yellow rattan & black leather, checkerboard, verniciatura gloss.* | Checkerboard stable (Corr 0.71); gloss not resolved. |
| **10** | [before_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_10.png) | [after_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_0/after_unseen/after_unseen_10.png) | *Intreccio semplice — natural rush bark posts, orange leather wefts, Grezzo finish.* | **High distortion (MAE 94.96)** — geometric breakdown for orange/rush bark combo. |

---

## 6. Epoch 1 — Seen Prompts After LoRA (dataset_repeat=20)

> **Status:** Regularization phase. Reduction from repeat=50 to repeat=20 smooths out structural distortions. Simple orthogonal grids (Prompts 4, 7, 8) achieve high structural correlation. Complex topologies (Macramè #3, trama tripla #9) still collapse. Verdict: **Yes, for simple structures. No, for complex ones.**
> [interpretation.md →](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/interpretation.md)

| # | Before LoRA | After LoRA (Epoch 1) | Caption | MAE / PSNR / Corr | Analysis |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | [before_seen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_1.png) | [after_seen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_1.png) | *Intreccio spina salto 2 — Verde green rattan, beige border w/ stitches.* | 80.83 / 8.42 dB / 0.43 | Green rattan injected; beige border visible; diagonal still imperfect. |
| **2** | [before_seen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_2.png) | [after_seen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_2.png) | *Intreccio semplice — deca green, glossy verniciatura.* | 50.50 / 11.78 dB / 0.34 | Green glossy grid partially rendered; layout mostly replaced. |
| **3** | [before_seen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_3.png) | [after_seen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_3.png) | *Macramé Vario — black leather mignon, Nodo quadro, 9 mm.* | 86.42 / 7.06 dB / **0.10** | **Collapse** — threads merge into flat, blurry mass; no knot topology. |
| **4** | [before_seen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_4.png) | [after_seen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_4.png) | *Intreccio semplice & trama doppia — Corteccia di Giunco, orange wefts, Grezzo.* | 42.25 / 13.65 dB / **0.60** | **Best seen** — orthogonal grid preserved; orange/natural sharp contrast. |
| **5** | [before_seen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_5.png) | [after_seen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_5.png) | *Intreccio semplice — white & light brown leather, checkerboard.* | 78.91 / 8.10 dB / 0.11 | Color bleeding brown→white; checkerboard partially formed. |
| **6** | [before_seen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_6.png) | [after_seen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_6.png) | *Intreccio semplice — white leather mignon (4 mm), posts 9 mm.* | 54.18 / 11.01 dB / **0.05** | **Lowest corr** — baseline layout completely discarded; border injected. |
| **7** | [before_seen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_7.png) | [after_seen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_7.png) | *Intreccio semplice — juniper bark (6 mm), light yellow wefts (9 mm), Grezzo.* | 44.82 / 13.47 dB / **0.72** | **Best corr** — simple grid preserved; bark/yellow micro-texture injected cleanly. |
| **8** | [before_seen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_8.png) | [after_seen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_8.png) | *Intreccio spina salto 2 — natural bamboo strips (3 mm), Grezzo.* | 46.75 / 13.12 dB / **0.70** | High corr — bamboo herringbone correctly rendered and polished. |
| **9** | [before_seen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_9.png) | [after_seen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_9.png) | *Intreccio semplice con trama tripla — dark blue & natural filaments (2 mm).* | 94.32 / 7.08 dB / 0.10 | **Triple-weft collapse** — blue and natural strands fully merged. |
| **10** | [before_seen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_10.png) | [after_seen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_seen_prompts_epoch_1/after_seen_10.png) | *Intreccio semplice — round rattan filaments (2 mm), flat piattina wefts (4 mm).* | 54.41 / 11.64 dB / 0.42 | Moderate reconstruction; round/flat strand distinction not yet resolved. |

---

## 7. Epoch 1 — Unseen Prompts After LoRA (10 shown + full 50 evaluated)

> **Status:** Partial success. Macro-aesthetic cues (borders, lighting, material colors) generalized. Structural vulnerability on spina diagonals (spatial instability), Jacquard (wavy tiles), Macramè (knot collapse), and Rinfilo/Uncinetto (layer blending). Verdict: **Partially achieved.**
> [interpretation.md →](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/interpretation.md)

| # | Before LoRA | After LoRA (Epoch 1) | Caption | Analysis |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [before_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_1.png) | [after_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_1.png) | *Intreccio semplice — natural rattan posts, dark blue leather wefts, Grezzo finish.* | Excellent basic grid; flat rectangular profile reproduced; border stitched. |
| **2** | [before_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_2.png) | [after_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_2.png) | *Intreccio semplice — white leather mignon, verniciatura gloss.* | Gloss sheen visible; stitching consistent; color bleeding reduced. |
| **3** | [before_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_3.png) | [after_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_3.png) | *Intreccio semplice — red split rattan, natural rush bark, Grezzo.* | Red/natural color correct; perspective distortion at margins (fettuccia warping). |
| **4** | [before_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_4.png) | [after_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_4.png) | *Intreccio semplice — thick black leather fettuccia, verniciatura opaca.* | Black matte grid aligned; margin line convergence slight. |
| **5** | [before_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_5.png) | [after_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_5.png) | *Intreccio semplice — light brown mignon, white wefts, Grezzo.* | Color bleeding still present at crossings; cross-attention leakage (white→brown). |
| **6** | [before_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_6.png) | [after_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_6.png) | *Intreccio semplice — fuchsia synthetic plastic, white leather.* | Fuchsia color correct; diagonal spina at Epoch 1 — wavy tile boundaries. |
| **7** | [before_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_7.png) | [after_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_7.png) | *Intreccio semplice — natural midollino, marrone chiaro, Grezzo.* | Warm tone reproduced; diagonal grid transitions from 90° to herringbone correctly. |
| **8** | [before_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_8.png) | [after_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_8.png) | *Intreccio semplice — gray leather strobel.* | Gray grid stable; strobel micro-texture emerging; no border distortion. |
| **9** | [before_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_9.png) | [after_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_9.png) | *Intreccio semplice — yellow rattan & black leather, checkerboard, verniciatura gloss.* | Checkerboard delineated; Paglia di Vienna base visible; some layer blending. |
| **10** | [before_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_10.png) | [after_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_exemples_on_unseen_prompts_before_after_lora_training_epoch_1/after_unseen_10.png) | *Intreccio semplice — natural rush bark, orange leather, Grezzo.* | Orange/natural correct; crochet topology collapses to generic weave grid. |

---

## 8. Epoch 4 — Seen & Unseen After LoRA (Final Checkpoint, 1,300 steps)

> **Status:** Mature style representation. Orthogonal grids are perfectly straight; strand widths uniform. Stitched borders are beautifully aligned with realistic leather embossment. Generalization to unseen prompts (blue, red, black leather) successful. Macramè/Uncinetto knot topologies remain physically simplified. Verdict: **Yes, with high performance.**
> [interpretation.md →](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/interpretation.md)

### 8.1 Seen Prompts — Epoch 4

| # | Before LoRA | After LoRA (Epoch 4) | Caption | Analysis |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [before_seen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_1.png) | [after_seen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_1.png) | *Intreccio spina salto 2 — Verde green rattan, beige border w/ stitches, Grezzo finish.* | **Exceptional.** Verde saturation correct; stitched border perfectly straight; organic rattan grain. |
| **2** | [before_seen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_2.png) | [after_seen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_2.png) | *Intreccio semplice — deca green, glossy verniciatura.* | **Excellent.** Grid perfectly orthogonal; specular highlights crisp; no waviness. |
| **3** | [before_seen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_3.png) | [after_seen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_3.png) | *Macramé Vario — black leather mignon, Nodo quadro, 9 mm.* | Knot structures crisper than Epoch 1; residual merging at overlapping edges. Physical knot topology still simplified. |
| **4** | [before_seen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_4.png) | [after_seen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_4.png) | *Intreccio semplice & trama doppia — Corteccia di Giunco, orange wefts, Grezzo.* | **Excellent.** Orange/natural contrast clean; minimal cross-material bleeding. |
| **5** | [before_seen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_5.png) | [after_seen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_5.png) | *Intreccio semplice — white & light brown leather, checkerboard.* | **Mature.** Color bleeding resolved; checkerboard boundaries sharp with leather pore detail. |
| **6** | [before_seen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_6.png) | [after_seen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_6.png) | *Intreccio semplice — white leather mignon (4 mm), posts 9 mm.* | **Premium.** White leather pores, embossed grain, and stitched border all resolved. |
| **7** | [before_seen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_7.png) | [after_seen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_7.png) | *Intreccio semplice — juniper bark (6 mm), light yellow wefts (9 mm), Grezzo.* | **High realism.** Bark fiber grain and yellow weft fully decoupled; non-artificial organic texture. |
| **8** | [before_seen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_8.png) | [after_seen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_8.png) | *Intreccio spina salto 2 — natural bamboo (3 mm), Grezzo.* | **Strong.** Bamboo herringbone perfectly aligned across full frame; no margin drift. |
| **9** | [before_seen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_9.png) | [after_seen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_9.png) | *Intreccio semplice con trama tripla — dark blue & natural round filaments.* | Trama tripla pattern crisper than Epoch 1; minor strand merging remains. |
| **10** | [before_seen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_seen_prompts/before_seen_10.png) | [after_seen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_seen_10.png) | *Intreccio semplice — round rattan (2 mm), flat piattina (4 mm).* | **Uniform.** Round vs. flat strand thickness resolved; border depth shading visible. |

### 8.2 Unseen Prompts — Epoch 4

| # | Before LoRA | After LoRA (Epoch 4) | Caption | Analysis |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [before_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_1.png) | [after_unseen_1.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_1.png) | *Intreccio semplice — natural rattan posts, dark blue leather wefts, Grezzo finish.* | **Excellent generalization.** Blue leather on rattan — distinct material textures, flat profile, clean border. |
| **2** | [before_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_2.png) | [after_unseen_2.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_2.png) | *Intreccio semplice — white leather mignon, verniciatura gloss.* | Convincing specular highlights; leather pore micro-texture; strong generalization. |
| **3** | [before_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_3.png) | [after_unseen_3.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_3.png) | *Intreccio spina salto 2 — black rattan, diagonal herringbone, Grezzo finish.* | **Matte finish excellent.** Diagonal lines stable; minor strand warp at margins. |
| **4** | [before_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_4.png) | [after_unseen_4.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_4.png) | *Intreccio spina — red leather wefts on split rattan.* | Red/rattan color correct; spatial instability — diagonal strands warp near margins. |
| **5** | [before_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_5.png) | [after_unseen_5.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_5.png) | *Pattern scacco — white/blue leather checkerboard.* | Alternating colors well-delineated; minor tile corner rounding at intersections. |
| **6** | [before_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_6.png) | [after_unseen_6.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_6.png) | *Pattern scacco — rattan/black leather checkerboard.* | Checkerboard visually striking; small structural overlaps at corners. |
| **7** | [before_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_7.png) | [after_unseen_7.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_7.png) | *Macramè Vario — thick black leather strands.* | Strong 3D drop shadows; realistic leather depth; knot topology simplified to bands. |
| **8** | [before_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_8.png) | [after_unseen_8.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_8.png) | *Macramè — white leather mignon.* | White leather mignon with deep drop shadows; knot loops simplified. |
| **9** | [before_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_9.png) | [after_unseen_9.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_9.png) | *Rinfilo on Paglia di Vienna.* | Vienna straw base recognizable (pre-trained texture preserved); leather mignon woven through straw partially visible. |
| **10** | [before_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_before_lora_training_unseen_prompts/before_unseen_10.png) | [after_unseen_10.png](file:///c:/Users/user/Downloads/Dataset_preparation/mouhaymin/Results_before_after_training/Z_score/Z_score_examples_after_lora_training_unseen_and__seen_prompts_epoch_4/after_unseen_10.png) | *Uncinetto — thick black leather crochet grid.* | Black leather grid visible; crochet loop geometry collapses to generic weave; resolution bottleneck in decoder. |

---

## 9. Training Curriculum Progression Summary

| Dimension | Epoch 0 (repeat=50) | Epoch 1 (repeat=20) | Epoch 4 (repeat=20) |
| :--- | :---: | :---: | :---: |
| **Style Injection (Border)** | ✅ Forced | ✅ Stabilized | ✅ Consistent |
| **Grid Orthogonality** | ❌ Wavy/distorted | ⚠️ Improved | ✅ Perfect |
| **Color Bleeding** | ❌ Severe | ⚠️ Reduced | ✅ Resolved |
| **Micro-Texture Realism** | ❌ Absent | ⚠️ Emerging | ✅ Crisp |
| **Diagonal Spina Stability** | ❌ Collapsed | ⚠️ Partial | ⚠️ Slight margin warp |
| **Knot Topology (Macramè)** | ❌ Flat mass | ❌ Flat mass | ⚠️ 3D bands (simplified) |
| **Unseen Generalization** | ❌ Overfitting/rejection | ⚠️ Partial | ✅ High |
| **Paglia di Vienna** | ⚠️ Merged | ⚠️ Partially seen | ✅ Recognizable |

**Final Verdict:** ✅ **Goal Achieved at Epoch 4** for style, material, and simple geometry. ⚠️ **Partially achieved** for complex knot/crochet topologies — a known limitation of 2D diffusion models without geometric constraints.

**Recommendations for future runs:**
1. Expand training set from 10 → 50+ images to avoid overfitting/style-rejection conflict.
2. Add ControlNet edge-detection loss to enforce grid alignment on diagonal patterns.
3. For Macramè/Uncinetto: incorporate 3D-rendered reference images in the dataset.
