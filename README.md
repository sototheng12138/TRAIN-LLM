<div align="center">

# TRAIN-LLM:ecoupled Time-series Reprogramming with Auxiliary Inference Network for zero-inflated railway dispatching
**Companion repository for the associated manuscript (Elsevier CAS single-column style).**

</div>

---

## Highlights

- **Time-LLM** training, evaluation, and visualization scripts for multichannel daily iron-freight series.  
- **Data split and metrics** aligned with the paper: **Train:Val:Test = 7:1:2**, `StandardScaler` fit on training data only; standardized and inverse-transformed MAE/RMSE.  
- **Baselines:** DLinear, Autoformer, iTransformer, XGBoost, ARIMA, LSTM, Prophet (fair comparison under the same split).  
---

## Abstract

This repository contains the **experimental code** used for multivariate long-horizon forecasting with **Time-LLM** (default **seq_len=96, pred_len=48**) on the same dataset and split as the paper, plus deterministic and deep-learning baselines. Evaluation emits metrics in **standardized space** and after **inverse transform** to original units, and can export **2×2 four-channel prediction curves** (English labels, consistent styling with baselines).  

**Distributed package:** `checkpoints/` from full training are **not** included (large size). Training commands and eval scripts are provided so that reviewers and third parties can reproduce results after training locally or obtaining weights from the authors.

---

## Keywords

Time series forecasting · Large language models · Time-LLM · Railway freight · Multivariate · Reproducibility

---

## 1. Introduction

Time-LLM (ICLR 2024) reprograms time series for forecasting inside a frozen LLM backbone. This project extends the data pipeline, losses, and evaluation for **four commodity channels** of railway freight and reports comparisons against multiple baselines. Methodological claims and numeric results in the **peer-reviewed paper** are authoritative; this README documents **code layout, experimental settings, and artifact paths** so they can be matched to the manuscript.

**Citation — Time-LLM:**

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**Citation — associated railway-freight work (add when available):**

```bibtex
% @article{...,
%   title   = {...},
%   journal = {...},
%   year    = {...},
% }
```

---

## 2. Materials and methods

| Item | Setting |
|------|---------|
| Data file | `dataset/2023_2025_Iron_data.csv` |
| Variables | Four channels: 铁矿石, 铁矿砂, 铁矿粉, 铁精矿粉 (`features=M`) |
| Split | Train : Val : Test = **7 : 1 : 2** (temporal, no shuffle) |
| Default horizon | seq_len=**96**, label_len=**48**, pred_len=**48** |
| Metrics | Standardized MAE/RMSE (÷ training-set std); original-scale MAE/RMSE after inverse transform |
| Full protocol | `docs/Baseline_Experimental_Setup.md` |

---

## 3. Installation

```bash
conda create -n time-llm python=3.11 -y
conda activate time-llm
cd <clone-directory>
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

- Core dependencies: `requirements.txt`. Scripts that call **`accelerate launch`** require a working **Accelerate** install (included transitively with the pinned stack).  
- **LLM weights** are not shipped. Configure Hugging Face access for the backbone used in `run_main.py` (Time-LLM default; e.g. LLaMA-class models). Weights populate the local cache on first run where permitted.  
- **Prompt files** are loaded from `dataset/prompt_bank/<model_comment>.txt` via `utils/tools.py`; the repository includes the prompt files used in the reported experiments.

### README scope vs. full reproduction

This file is an **overview**. Executable steps are in **`docs/REPRODUCTION.md`**. Additional prerequisites:

| Requirement | Detail |
|-------------|--------|
| Missing `checkpoints/` | Run training scripts or obtain a weight archive from the authors and extract under `./checkpoints/`; otherwise `run_eval` / `Eval_*.sh` cannot load models. |
| Stage-2 linear training | `Iron_stage2_linear.sh` assumes a completed Stage-1 run from `Iron_stage2_linear.sh` (see comments in that script). |
| Incomplete `run_eval` lines | Placeholder `...` in examples is not runnable; copy the full argument list from the matching `scripts/Eval_*.sh` and append `--save_pred_true` if full-test arrays are needed. |
| Plot script `--tag` | Must match eval `--output_tag` (e.g. `Eval_Iron_stage2_linear.sh` uses `iron` → `pred_iron.npy` → `--tag iron`). If eval omits `output_tag`, use `--tag ""`. |
| Chronos | Extra dependencies beyond `requirements.txt`; see `scripts/Eval_Iron_chronos_zero_shot.py`. |

---

## 4. Reproducing results

Follow **`docs/REPRODUCTION.md`** step by step. Summary index:

| Step | Command / script |
|------|------------------|
| Train (stage-2 linear example) | `bash scripts/Iron_stage1_linear.sh` then `bash scripts/Iron_stage2_linear.sh` |
| Evaluate main model | `bash scripts/Eval_Iron_stage2_linear.sh` (default CPU; GPU flags in script comments) |
| Full-test paper_v4 figures | Add `--save_pred_true` to the same `run_eval.py` invocation as the chosen `Eval_*.sh`, then `scripts/plot_full_testset_from_npy.py` with matching `--tag` |
| Classical baselines | `python run_baseline_xgb_arima.py` (needs `requirements-optional.txt`) |
| Time-LLM (MSE) / DLinear / Autoformer / iTransformer | `bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh all` |
| Chronos zero-shot | `docs/REPRODUCTION.md`, section 6 |

---

## 5. Repository layout

```
<root>/
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── run_main.py
├── run_eval.py
├── run_baseline_xgb_arima.py
├── models/
├── layers/
├── data_provider/
├── data_provider_pretrain/
├── dataset/
│   └── README_DATA.md
├── scripts/
├── docs/
│   ├── REPRODUCTION.md
│   ├── Baseline_Experimental_Setup.md
│   └── PAPER_SYNC_CHECKLIST.md
├── figures/
├── iTransformer/
└── chronos-forecasting/
```

**Checkpoints:** The public tree omits `checkpoints/` (often hundreds of GB). After training or restoring weights, paths must match `run_eval` resolution: `checkpoints/<setting>-<model_comment>/`.

---

## 6. Manuscript figures ↔ artifacts

| Manuscript element (example) | Typical artifact |
|------------------------------|------------------|
| Main model, full-test 2×2 curves | `<ckpt>/plots_full_testset/pred_true_full_TEST_step1_*_paper_v4.pdf` |
| XGBoost / ARIMA / Prophet | `checkpoints/baselines/*_2023_2025_Iron_data_96_48/pred_true_full_TEST_step1_*_paper_v4.pdf` |
| Chronos zero-shot | `checkpoints/pred_true_full_*_chronos.pdf` (see `docs/REPRODUCTION.md`) |
| Baseline table | `baseline_eval_result.txt` (or path set in `run_baseline_xgb_arima.py`) |
| Ablations / other stages | Matching `scripts/Eval_Iron_*.sh` outputs (`eval_result*.txt`) |

---

## 7. Data availability

See **`dataset/README_DATA.md`**. Public release requires **clear rights** to redistribute the CSV; alternatives are a **synthetic surrogate** plus preprocessing code and a formal **data availability** statement in the paper.

---

## 8. License

- Upstream **Time-LLM**: **`LICENSE`** and **`LEGAL.md`**.  
- Derivative work in this repository should be described in the repository license or an `AUTHORS.md` / notice file as appropriate.

---

## 9. Contact

Maintainer / corresponding author: *(to be filled in for public release).*

---

*Section order follows common Elsevier **CAS single-column** article elements (Highlights, graphical abstract, abstract, keywords, body sections).*
