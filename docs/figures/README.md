# TIME-LLM Framework Figures (Publication-Ready)

This folder contains **five** complete, journal-style figures for the TIME-LLM event-driven reprogramming framework with label-free auxiliary deduction. All figures use a consistent color scheme and typography suitable for top-tier venues (e.g., NeurIPS, ICML, Nature Machine Intelligence).

---

## Figure Index

| File | Description | Suggested use |
|------|-------------|---------------|
| **fig0_minimal_main.svg** | Single-flow overview: TS+Prompts → Backbone → Joint Inference. | **Main paper**, one-column or small figure. |
| **fig1_framework_overview.svg** | Full pipeline: Input → Backbone → Joint Inference (Num + Aux + pseudo-labels) → Output. | Main paper or supplementary; clear data flow. |
| **fig2_training_vs_inference.svg** | Two-panel: Training (with pseudo-labels, losses) vs Inference (deterministic decoding, dispatch log). | Emphasize label-free training and inference behavior. |
| **fig3_joint_inference_detail.svg** | Zoom on Joint Inference: latent split to Num head (L_ZI-MSE) and Aux head (L_Aux), pseudo-label definition. | Supplementary; technical detail. |
| **fig4_full_pipeline_dispatch_log.svg** | End-to-end flow + three output types (numerical, auxiliary, dispatch log) and example log text. | Main or supplementary; ties pipeline to user-facing output. |

---

## Design Conventions

- **Input / data**: Blue tones (`#5a9fb8`, `#3d7a94`).
- **Backbone (frozen LLM)**: Purple tones (`#6b5b95`, `#4a3d6b`).
- **Joint Inference (Ours)**: Green tones (`#2d6a4f`, `#1b4332`).
- **Output / interpretation**: Amber/gold (`#b8860b`, `#8b6914`).
- **Pseudo-labels / training-only**: Neutral gray.
- **Font**: Georgia / Times New Roman (serif) for a classic journal look.
- **Arrows**: Dark gray (`#2c3e50`); dashed for “supervise” or optional paths.

---

## Usage

- **Insert in LaTeX**: `\includegraphics[width=\linewidth]{figures/fig1_framework_overview.pdf}`  
  (Export SVG to PDF for vector quality.)
- **Insert in Word / PPT**: Import the SVG directly, or export to PNG at 300 dpi for print.
- **Export to PDF**: Use Inkscape, Illustrator, or browser print (Save as PDF) from the SVG.
- **Caption suggestions**:  
  - Fig. 1: “Overview of TIME-LLM: time series and domain prompts are encoded by a frozen LLM; joint inference produces numerical forecasts and auxiliary dispatch deduction with data-derived pseudo-labels.”  
  - Fig. 2: “Training phase uses pseudo-labels from the future window; inference phase produces a deterministic dispatch log without future data.”

---

## File List

```
figures/
├── README.md
├── fig0_minimal_main.svg
├── fig1_framework_overview.svg
├── fig2_training_vs_inference.svg
├── fig3_joint_inference_detail.svg
└── fig4_full_pipeline_dispatch_log.svg
```

All diagrams are in English for international submission; captions and body text can be translated to Chinese for domestic venues while keeping the same figures.
