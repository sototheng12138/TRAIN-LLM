# Step-by-step reproduction (Iron 96→48, 7:1:2 split)

All commands assume repository root and Python env with `requirements.txt` installed.

## 0. Environment

```bash
conda create -n time-llm python=3.11 -y
conda activate time-llm
pip install -r requirements.txt
# Baselines (XGBoost / ARIMA / Prophet / LSTM):
pip install -r requirements-optional.txt
```

**LLM backbone:** Install Hugging Face access to the backbone used in `run_main.py` (default LLaMA family per Time-LLM). Weights are downloaded at first run if `HF_HOME` / cache is configured; this repo does not ship model weights.

## 1. Data

Default Iron CSV is already under `dataset/2023_2025_Iron_data.csv`.  
Schema: first column date, then four channels (铁矿石、铁矿砂、铁矿粉、铁精矿粉).  
See `dataset/README_DATA.md` when substituting another CSV (preserve column names expected by loaders).

## 2. Train main Time-LLM (example: stage-2 linear head)

**Important:** `scripts/Iron_stage2_linear.sh` **requires** a finished Stage-1 checkpoint from `scripts/Iron_stage1_linear.sh` (see comments at top of `Iron_stage2_linear.sh`). Order:

```bash
mkdir -p checkpoints
bash scripts/Iron_stage1_linear.sh    # first
bash scripts/Iron_stage2_linear.sh    # then (uses 2 GPUs in script; edit CUDA_VISIBLE_DEVICES if needed)
```

Other entry points: `bash scripts/Iron.sh` (single-stage), or browse `scripts/Iron*.sh`.

Training writes under `checkpoints/long_term_forecast_*`.

## 3. Evaluate Time-LLM

```bash
bash scripts/Eval_Iron_stage2_linear.sh   # or another Eval_*.sh matching the trained model_comment
# or generic:
bash scripts/Eval_Iron.sh
```

To save full-test predictions and paper_v4 figures, **append `--save_pred_true`** to the same arguments as your `Eval_*.sh`, then plot with the **same `--output_tag` as in that script** (suffix of `pred_<tag>.npy`).

Example for `scripts/Eval_Iron_stage2_linear.sh` (it uses `--output_tag iron`):

```bash
# Copy all flags from Eval_Iron_stage2_linear.sh and add:
python run_eval.py ... --save_pred_true   # + same flags as the .sh file

CKPT=checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_stage2_linear
python scripts/plot_full_testset_from_npy.py --ckpt_dir "$CKPT" --tag iron --style paper_v4
```

If your eval uses no `output_tag`, files are `pred.npy` / `true.npy` → use `--tag ""` (empty string) in the plot script.

## 4. Baselines (fair comparison, same split)

```bash
python run_baseline_xgb_arima.py
# outputs baseline_eval_result.txt and checkpoints/baselines/*/plots_full_testset/
```

## 5. DLinear / Autoformer / basic Time-LLM / iTransformer + full-test plots

```bash
# CPU (safe default)
bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh all

# GPU (pick a free card)
CUDA_VISIBLE_DEVICES=1 EVAL_DEVICE=cuda bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh all
```

## 6. Chronos zero-shot

```bash
python scripts/Eval_Iron_chronos_zero_shot.py --save_pred_true --out_dir checkpoints
python scripts/plot_full_testset_from_npy.py --ckpt_dir checkpoints --tag chronos2_small_unclipped_orig --style paper_v4
```

## 7. Experimental protocol (paper text)

See `docs/Baseline_Experimental_Setup.md` for train/val/test ratio, metrics (standardized vs original scale), and per-model settings.
