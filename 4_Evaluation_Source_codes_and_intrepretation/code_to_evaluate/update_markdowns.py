import re
import csv
import numpy as np
from pathlib import Path

root_dir = Path(r"c:\Users\user\Downloads\Final_computer_vision\Captioning")
results_dir = root_dir / "all_adopted_Results" / "Org_baseline_LoraGeneratedResults"
eval_dir = root_dir / "4_Evaluation_Source_codes_and_intrepretation"

models = {
    "flux": {
        "quant": results_dir / "flux" / "FLUX_quantitative_scoring_report.csv",
        "qual": results_dir / "flux" / "FLUX_qualitative_scoring_report.csv"
    },
    "SDXL": {
        "quant": results_dir / "SDXL" / "SDXL_quantitative_scoring_report.csv",
        "qual": results_dir / "SDXL" / "SDXL_qualitative_scoring_report.csv"
    },
    "Z_score": {
        "quant": results_dir / "Z_score" / "quantitative_scoring_report.csv",
        "qual": results_dir / "Z_score" / "qualitative_scoring_report.csv"
    }
}

model_stats = {}

for model_name, paths in models.items():
    # Load quantitative stats
    clips = []
    lpips = []
    iqas = []
    with open(paths["quant"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(float(row["clip_score"]))
            lpips.append(float(row["lpips_score"]))
            iqas.append(float(row["clip_iqa"]))

    # Load qualitative stats
    adherence = []
    identity = []
    manufacturability = []
    visual_quality = []
    originality = []
    mean_score = []
    with open(paths["qual"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adherence.append(float(row["prompt_adherence"]))
            identity.append(float(row["intreccio_identity"]))
            manufacturability.append(float(row["manufacturability"]))
            visual_quality.append(float(row["visual_quality"]))
            originality.append(float(row["controlled_originality"]))
            mean_score.append(float(row["mean_score"]))

    model_stats[model_name] = {
        "clip": (np.mean(clips), np.std(clips)),
        "lpips": (np.mean(lpips), np.std(lpips)),
        "iqa": (np.mean(iqas), np.std(iqas)),
        "adherence": (np.mean(adherence), np.std(adherence)),
        "identity": (np.mean(identity), np.std(identity)),
        "manufacturability": (np.mean(manufacturability), np.std(manufacturability)),
        "visual_quality": (np.mean(visual_quality), np.std(visual_quality)),
        "originality": (np.mean(originality), np.std(originality)),
        "mean_score": (np.mean(mean_score), np.std(mean_score))
    }

# Compute the LPIPS winner
lpips_winner = "FLUX"
if model_stats["Z_score"]["lpips"][0] < model_stats["flux"]["lpips"][0] and model_stats["Z_score"]["lpips"][0] < model_stats["SDXL"]["lpips"][0]:
    lpips_winner = "Z-Image"
elif model_stats["SDXL"]["lpips"][0] < model_stats["flux"]["lpips"][0] and model_stats["SDXL"]["lpips"][0] < model_stats["Z_score"]["lpips"][0]:
    lpips_winner = "SDXL"

# Construct the unified table for phase6.md
unified_table = f"""| Metric / Aspect | FLUX (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **CLIPScore** (Text Alignment) | {model_stats['flux']['clip'][0]:.4f} | {model_stats['SDXL']['clip'][0]:.4f} | **{model_stats['Z_score']['clip'][0]:.4f}** | **Z-Image** (Slight margin) |
| **CLIP-IQA** (Aesthetic Quality) | {model_stats['flux']['iqa'][0]:.4f} | **{model_stats['SDXL']['iqa'][0]:.4f}** | {model_stats['Z_score']['iqa'][0]:.4f} | **SDXL** |
| **LPIPS** (Style Distance to Real) | {model_stats['flux']['lpips'][0]:.4f} | {model_stats['SDXL']['lpips'][0]:.4f} | **{model_stats['Z_score']['lpips'][0]:.4f}** | **{lpips_winner}** |
| :--- | :---: | :---: | :---: | :---: |
| **Prompt Adherence** | **{model_stats['flux']['adherence'][0]:.2f}** | {model_stats['SDXL']['adherence'][0]:.2f} | {model_stats['Z_score']['adherence'][0]:.2f} | **FLUX** |
| **Intreccio Identity** | **{model_stats['flux']['identity'][0]:.2f}** | {model_stats['SDXL']['identity'][0]:.2f} | {model_stats['Z_score']['identity'][0]:.2f} | **FLUX** |
| **Manufacturability** | **{model_stats['flux']['manufacturability'][0]:.2f}** | {model_stats['SDXL']['manufacturability'][0]:.2f} | {model_stats['Z_score']['manufacturability'][0]:.2f} | **FLUX** |
| **Visual Quality** | **{model_stats['flux']['visual_quality'][0]:.2f}** | {model_stats['SDXL']['visual_quality'][0]:.2f} | {model_stats['Z_score']['visual_quality'][0]:.2f} | **FLUX** |
| **Controlled Originality** | **{model_stats['flux']['originality'][0]:.2f}** | {model_stats['SDXL']['originality'][0]:.2f} | {model_stats['Z_score']['originality'][0]:.2f} | **FLUX** |
| **OVERALL MEAN SCORE** | **{model_stats['flux']['mean_score'][0]:.2f}** | {model_stats['SDXL']['mean_score'][0]:.2f} | {model_stats['Z_score']['mean_score'][0]:.2f} | **FLUX** |"""

# Construct the comparison summary table for comparision.md (excluding the winner column and formatting as mean +- std)
comp_table_quant = f"""| Model | CLIPScore (Text Alignment) | CLIP-IQA (Aesthetic Quality) | LPIPS (Style Distance to Real) |
| :--- | :---: | :---: | :---: |
| **FLUX (DiT)** | {model_stats['flux']['clip'][0]:.4f} ± {model_stats['flux']['clip'][1]:.4f} | {model_stats['flux']['iqa'][0]:.4f} ± {model_stats['flux']['iqa'][1]:.4f} | {model_stats['flux']['lpips'][0]:.4f} ± {model_stats['flux']['lpips'][1]:.4f} |
| **SDXL (Latent Diffusion)** | {model_stats['SDXL']['clip'][0]:.4f} ± {model_stats['SDXL']['clip'][1]:.4f} | {model_stats['SDXL']['iqa'][0]:.4f} ± {model_stats['SDXL']['iqa'][1]:.4f} | {model_stats['SDXL']['lpips'][0]:.4f} ± {model_stats['SDXL']['lpips'][1]:.4f} |
| **Z-Image (DiT)** | {model_stats['Z_score']['clip'][0]:.4f} ± {model_stats['Z_score']['clip'][1]:.4f} | {model_stats['Z_score']['iqa'][0]:.4f} ± {model_stats['Z_score']['iqa'][1]:.4f} | {model_stats['Z_score']['lpips'][0]:.4f} ± {model_stats['Z_score']['lpips'][1]:.4f} |"""

comp_table_qual = f"""| Evaluation Criteria | FLUX (DiT) | SDXL (Latent Diffusion) | Z-Image (DiT) |
| :--- | :---: | :---: | :---: |
| **Prompt Adherence** | {model_stats['flux']['adherence'][0]:.2f} ± {model_stats['flux']['adherence'][1]:.2f} | {model_stats['SDXL']['adherence'][0]:.2f} ± {model_stats['SDXL']['adherence'][1]:.2f} | {model_stats['Z_score']['adherence'][0]:.2f} ± {model_stats['Z_score']['adherence'][1]:.2f} |
| **Intreccio Identity** | {model_stats['flux']['identity'][0]:.2f} ± {model_stats['flux']['identity'][1]:.2f} | {model_stats['SDXL']['identity'][0]:.2f} ± {model_stats['SDXL']['identity'][1]:.2f} | {model_stats['Z_score']['identity'][0]:.2f} ± {model_stats['Z_score']['identity'][1]:.2f} |
| **Manufacturability** | {model_stats['flux']['manufacturability'][0]:.2f} ± {model_stats['flux']['manufacturability'][1]:.2f} | {model_stats['SDXL']['manufacturability'][0]:.2f} ± {model_stats['SDXL']['manufacturability'][1]:.2f} | {model_stats['Z_score']['manufacturability'][0]:.2f} ± {model_stats['Z_score']['manufacturability'][1]:.2f} |
| **Visual Quality** | {model_stats['flux']['visual_quality'][0]:.2f} ± {model_stats['flux']['visual_quality'][1]:.2f} | {model_stats['SDXL']['visual_quality'][0]:.2f} ± {model_stats['SDXL']['visual_quality'][1]:.2f} | {model_stats['Z_score']['visual_quality'][0]:.2f} ± {model_stats['Z_score']['visual_quality'][1]:.2f} |
| **Controlled Originality** | {model_stats['flux']['originality'][0]:.2f} ± {model_stats['flux']['originality'][1]:.2f} | {model_stats['SDXL']['originality'][0]:.2f} ± {model_stats['SDXL']['originality'][1]:.2f} | {model_stats['Z_score']['originality'][0]:.2f} ± {model_stats['Z_score']['originality'][1]:.2f} |
| **OVERALL MEAN SCORE** | **{model_stats['flux']['mean_score'][0]:.2f} ± {model_stats['flux']['mean_score'][1]:.2f}** | **{model_stats['SDXL']['mean_score'][0]:.2f} ± {model_stats['SDXL']['mean_score'][1]:.2f}** | **{model_stats['Z_score']['mean_score'][0]:.2f} ± {model_stats['Z_score']['mean_score'][1]:.2f}** |"""

# ----------------- Update comparision.md -----------------
comp_path = eval_dir / "comparision.md"
with open(comp_path, "r", encoding="utf-8") as f:
    comp_content = f.read()

# Replace quantitative table
comp_content = re.sub(
    r"## 1\. Combined Quantitative Metrics Comparison\s*.*?(\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n)",
    f"## 1. Combined Quantitative Metrics Comparison\n\n{comp_table_quant}\n",
    comp_content,
    flags=re.DOTALL
)

# Replace qualitative table
comp_content = re.sub(
    r"## 2\. Combined Qualitative MLLM-as-a-Judge Ratings Comparison\s*.*?(\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n)",
    f"## 2. Combined Qualitative MLLM-as-a-Judge Ratings Comparison (Average / 5.0)\n\n{comp_table_qual}\n",
    comp_content,
    flags=re.DOTALL
)

with open(comp_path, "w", encoding="utf-8") as f:
    comp_content = f.write(comp_content)
print(f"Updated {comp_path}")

# ----------------- Update phase6.md -----------------
phase6_path = eval_dir / "phase6.md"
with open(phase6_path, "r", encoding="utf-8") as f:
    phase6_content = f.read()

# Replace unified table
phase6_content = re.sub(
    r"### Quantitative Metrics Comparison Summary\s*.*?(\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n\s*\|.*?\n)",
    f"### Quantitative Metrics Comparison Summary\n\n{unified_table}\n",
    phase6_content,
    flags=re.DOTALL
)

# Replace other hardcoded values in text
phase6_content = phase6_content.replace("0.3122", f"{model_stats['Z_score']['clip'][0]:.4f}")
phase6_content = phase6_content.replace("0.3106", f"{model_stats['flux']['clip'][0]:.4f}")
phase6_content = phase6_content.replace("0.3102", f"{model_stats['SDXL']['clip'][0]:.4f}")

phase6_content = phase6_content.replace("0.6722", f"{model_stats['SDXL']['iqa'][0]:.4f}")
phase6_content = phase6_content.replace("0.6444", f"{model_stats['flux']['iqa'][0]:.4f}")
phase6_content = phase6_content.replace("0.4850", f"{model_stats['Z_score']['iqa'][0]:.4f}")

phase6_content = re.sub(
    r"rates \*\*FLUX\*\* the highest at \*\*4\.70/5\.0\*\*",
    f"rates **FLUX** the highest at **{model_stats['flux']['adherence'][0]:.2f}/5.0**",
    phase6_content
)
phase6_content = re.sub(
    r"highest VLM Intreccio Identity score \(\*\*4\.50/5\.0\*\*\)",
    f"highest VLM Intreccio Identity score (**{model_stats['flux']['identity'][0]:.2f}/5.0**)",
    phase6_content
)
phase6_content = re.sub(
    r"leads with the lowest LPIPS distance \(\*\*0\.0018\*\*\)",
    f"leads with the highest VLM Intreccio Identity score (**{model_stats['flux']['identity'][0]:.2f}/5.0**), although Z-Image achieves the lowest quantitative LPIPS distance (**{model_stats['Z_score']['lpips'][0]:.4f}**)",
    phase6_content
)
phase6_content = re.sub(
    r"Controlled Originality rating of \*\*4\.57/5\.0\*\*",
    f"Controlled Originality rating of **{model_stats['flux']['originality'][0]:.2f}/5.0**",
    phase6_content
)
phase6_content = re.sub(
    r"whereas \*\*SDXL\*\* \(3\.99\) and \*\*Z-Image\*\* \(4\.37\) suffer",
    f"whereas **SDXL** ({model_stats['SDXL']['originality'][0]:.2f}) and **Z-Image** ({model_stats['Z_score']['originality'][0]:.2f}) suffer",
    phase6_content
)

# Let's also verify and replace any other occurrences of FLUX being rated highest in the summary
phase6_content = re.sub(
    r"rates \*\*FLUX\*\* the highest at \*\*4\.23/5\.0\*\*",
    f"rates **FLUX** the highest at **{model_stats['flux']['adherence'][0]:.2f}/5.0**",
    phase6_content
)

with open(phase6_path, "w", encoding="utf-8") as f:
    f.write(phase6_content)
print(f"Updated {phase6_path}")
print("[SUCCESS] update_markdowns.py executed successfully.")
