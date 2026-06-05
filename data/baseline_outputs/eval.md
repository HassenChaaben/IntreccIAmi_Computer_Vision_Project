# Z-Image Baseline Evaluation — Task 1 (IMG_6865)

## Caption Used

> **intrecciami-style:** A meticulously crafted woven square featuring an intricate Intreccio spina salto 2 pattern. The weave showcases a tight, symmetrical structure with double posts and wefts interlacing in a precise over-under sequence, creating a visually appealing grid-like texture. The natural rattan strands, flattened and evenly spaced, are uniformly dyed in a vibrant Verde (green) hue, highlighting the raw, unvarnished (Grezzo) finish. Encased in a light beige leather border with clean, stitched edges, the piece exemplifies artisanal craftsmanship. The weave's density and uniformity suggest a high level of skill and attention to detail, making it a striking example of traditional weaving techniques. Close-up studio photograph, premium texture, high resolution, macro photography.

---

## Token Length Analysis

| Metric | Value | Z-Image Target Range (Tokens) | Status |
|--------|-------|------------------------------|--------|
| **Exact Token Count (Qwen2 Tokenizer / Z-Image)** | 161 tokens | ~80–130 tokens | ⚠️ Slightly over (by 31 tokens) |
| **Exact Token Count (GPT2/CLIP Tokenizer)** | 167 tokens | ~80–130 tokens | ⚠️ Slightly over (by 37 tokens) |
| **Word Count** | 110 words | ~60–100 words | ⚠️ Slightly over (by 10 words) |
| **Character Count** | 793 characters | — | — |
| **Trigger Word** | `intrecciami-style` present | Required | ✅ |
| **Suffix Tags** | `Close-up studio photograph, premium texture, high resolution, macro photography` | Required | ✅ |

> [!NOTE]
> The caption contains exactly **161 tokens** (Qwen2 tokenizer) and **167 tokens** (GPT2/CLIP tokenizer). The original target range of 60–100 words roughly translates to **80–130 tokens**. Although the generated caption is slightly longer than this target, Z-Image's text encoder (Qwen2-based, with a max context limit of 512 tokens) is fully compatible and handles this length without truncation or loss of information.

---

## Caption ↔ Generated Image Compatibility

### Visual Match Assessment

| Caption Element | Described | Present in Image | Match |
|----------------|-----------|-----------------|-------|
| Woven square shape | ✅ | ✅ Square format with border | ✅ |
| Intreccio spina salto 2 pattern | ✅ | ✅ Diagonal herringbone-like weave visible | ✅ |
| Double posts and wefts | ✅ | ✅ Multiple strands per crossing clearly visible | ✅ |
| Over-under sequence | ✅ | ✅ Clear interlacing pattern | ✅ |
| Grid-like texture | ✅ | ✅ Regular repeating grid structure | ✅ |
| Natural rattan strands | ✅ | ✅ Organic natural fiber appearance | ✅ |
| Flattened and evenly spaced | ✅ | ✅ Flat strips uniformly arranged | ✅ |
| Verde (green) color | ✅ | ✅ Vibrant green throughout | ✅ |
| Grezzo (unvarnished) finish | ✅ | ✅ Matte, raw natural look (no gloss) | ✅ |
| Light beige leather border | ✅ | ✅ Tan/beige leather frame with stitching | ✅ |
| Stitched edges | ✅ | ✅ Visible stitching on leather border | ✅ |
| Close-up studio photograph | ✅ | ✅ Macro-level detail, studio lighting | ✅ |

### Compatibility Score: **11/11 elements matched = 100%**

---

## Quality Assessment

### 🟢 Strengths
- **Exceptional caption-image fidelity** — the Z-Image base model clearly understood ALL described elements
- **Material accuracy** — rattan fiber texture is convincingly rendered with natural strand variations
- **Color accuracy** — vibrant green matches "Verde" precisely with natural tonal variation
- **Structural accuracy** — the interlacing over-under pattern with double strands is physically correct
- **Composition** — leather border, stitching, and square format all match
- **Photography style** — lighting and macro detail match the "studio photograph" suffix

### 🟡 Observations
- The token length (161 Qwen2 tokens) exceeds the preferred ~80-130 token target range by about 23%, but this did not cause any generation errors or visual truncation.
- This is a **pre-LoRA baseline** — the result is already very strong, which suggests Z-Image's base model has strong textile/weave comprehension.

### 🔴 Implications for LoRA Training
- Since the baseline is already quite good at generating generic woven textures, the LoRA fine-tuning should focus on capturing the **specific identity** of your artisan's patterns (exact strand widths, precise spina salto crossing angles, authentic material imperfections).
- The comparison between this baseline and post-LoRA outputs will demonstrate style transfer, not capability transfer.

---

## Summary

| Metric | Result |
|--------|--------|
| **Caption Quality** | ✅ Excellent — all metadata faithfully described, no hallucinations |
| **Token Length** | ⚠️ 161 tokens (target: ~80–130) — slightly verbose but fully compatible |
| **Image-Caption Match** | ✅ 100% (11/11 elements) |
| **Baseline Usefulness** | ✅ Strong baseline for pre/post LoRA comparison |

