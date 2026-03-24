#!/bin/bash
# 对「最原始 Time-LLM（仅 MSE）」「DLinear」「Autoformer」「iTransformer」跑 eval 并生成：
#   1) 首窗四通道图（pred_true_4channels）
#   2) 全测试集 paper_v4 风格图（与 XGBoost/ARIMA/Prophet 主图格式一致）
#
# 用法（单个跑可避免 OOM）:
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh timellm     # 只跑 Time-LLM (MSE)
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh dlinear     # 只跑 DLinear
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh autoformer  # 只跑 Autoformer
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh itransformer # 只跑 iTransformer
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh             # 跑全部
#   bash scripts/Eval_TimeLLM_DLinear_Autoformer_plot.sh timellm --plot-only  # 仅重绘图（需已有 pred.npy）
#
# 默认 --device cpu 防 OOM；用 GPU 可加: export EVAL_DEVICE=cuda

set -e
cd "$(dirname "$0")/.."
PLOT_ONLY=false
RUN_WHICH=""
for a in "$@"; do
  [ "$a" = "--plot-only" ] && PLOT_ONLY=true
  [ "$a" = "timellm" ] || [ "$a" = "1" ] && RUN_WHICH="timellm"
  [ "$a" = "dlinear" ] || [ "$a" = "2" ] && RUN_WHICH="dlinear"
  [ "$a" = "autoformer" ] || [ "$a" = "3" ] && RUN_WHICH="autoformer"
  [ "$a" = "itransformer" ] || [ "$a" = "4" ] && RUN_WHICH="itransformer"
  [ "$a" = "all" ] && RUN_WHICH="all"
done
[ -z "$RUN_WHICH" ] && RUN_WHICH="all"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"

# 公共参数（与 Iron 数据集、seq=96 pred=48 一致）
COMMON=(
  --task_name long_term_forecast
  --model_id Iron_96_48
  --root_path ./dataset/
  --data_path 2023_2025_Iron_data.csv
  --data custom
  --features M
  --seq_len 96
  --label_len 48
  --pred_len 48
  --enc_in 4
  --dec_in 4
  --c_out 4
  --checkpoints ./checkpoints/
  --des 'Iron_Ore_Transport_Exp'
  --itr 1
)

# 1) Time-LLM 最原始版（仅 MSE）：不启用辅助头、不零阈值
time_llm_ckpt="checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron"
if [ "$RUN_WHICH" = "timellm" ] || [ "$RUN_WHICH" = "all" ]; then
  if [ "$PLOT_ONLY" = false ]; then
    echo "========== Time-LLM (MSE only, no aux / no zero_threshold) =========="
    python run_eval.py "${COMMON[@]}" \
      --device "$EVAL_DEVICE" \
      --model TimeLLM \
      --model_comment iron \
      --d_model 32 \
      --d_ff 128 \
      --llm_layers 32
    echo "  -> $time_llm_ckpt"
  fi
  [ -f "$time_llm_ckpt/pred_true_4channels_data.npz" ] && python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "$time_llm_ckpt" || echo "  跳过 Time-LLM 出图（无 npz，请先不传 --plot-only 跑一遍 eval）"
  echo ""
fi

# 2) DLinear（d_model=16, d_ff=32 与训练时一致）
dlinear_ckpt="checkpoints/long_term_forecast_Iron_96_48_DLinear_custom_ftM_sl96_ll48_pl48_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_dlinear"
if [ "$RUN_WHICH" = "dlinear" ] || [ "$RUN_WHICH" = "all" ]; then
  if [ "$PLOT_ONLY" = false ]; then
    echo "========== DLinear =========="
    python run_eval.py "${COMMON[@]}" \
      --device "$EVAL_DEVICE" \
      --model DLinear \
      --model_comment iron_dlinear \
      --d_model 16 \
      --d_ff 32
    echo "  -> $dlinear_ckpt"
  fi
  [ -f "$dlinear_ckpt/pred_true_4channels_data.npz" ] && python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "$dlinear_ckpt" || echo "  跳过 DLinear 出图（无 npz，请先不传 --plot-only 跑一遍 eval）"
  echo ""
fi

# 3) Autoformer（d_model=32, d_ff=128；训练时用了 --multivariate，eval 必须一致）
autoformer_ckpt="checkpoints/long_term_forecast_Iron_96_48_Autoformer_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_autoformer"
if [ "$RUN_WHICH" = "autoformer" ] || [ "$RUN_WHICH" = "all" ]; then
  if [ "$PLOT_ONLY" = false ]; then
    echo "========== Autoformer =========="
    python run_eval.py "${COMMON[@]}" \
      --device "$EVAL_DEVICE" \
      --model Autoformer \
      --model_comment iron_autoformer \
      --d_model 32 \
      --d_ff 128 \
      --moving_avg 25 \
      --freq d \
      --multivariate \
      --save_pred_true
    echo "  -> $autoformer_ckpt"
  fi
  [ -f "$autoformer_ckpt/pred_true_4channels_data.npz" ] && python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "$autoformer_ckpt" || echo "  跳过 Autoformer 首窗图（无 npz）"
  if [ -f "$autoformer_ckpt/pred.npy" ]; then
    python scripts/plot_full_testset_from_npy.py --ckpt_dir "$autoformer_ckpt" --tag "" --style paper_v4
    [ -f "$autoformer_ckpt/plots_full_testset/pred_true_full_TEST_step1_paper_v4.pdf" ] && cp "$autoformer_ckpt/plots_full_testset/pred_true_full_TEST_step1_paper_v4.pdf" "$autoformer_ckpt/pred_true_full_TEST_step1_Autoformer_paper_v4.pdf"
  else
    echo "  跳过 Autoformer 全测试集图（无 pred.npy）"
  fi
  echo ""
fi

# 4) iTransformer（d_model=64, d_ff=256, e_layers=3 对应 iron_itransformer_big checkpoint）
itransformer_ckpt="checkpoints/long_term_forecast_Iron_96_48_iTransformer_custom_ftM_sl96_ll48_pl48_dm64_nh8_el3_dl1_df256_fc3_ebtimeF_Iron_Ore_Transport_Exp_0-iron_itransformer_big"
if [ "$RUN_WHICH" = "itransformer" ] || [ "$RUN_WHICH" = "all" ]; then
  if [ "$PLOT_ONLY" = false ]; then
    echo "========== iTransformer =========="
    python run_eval.py "${COMMON[@]}" \
      --device "$EVAL_DEVICE" \
      --model iTransformer \
      --model_comment iron_itransformer_big \
      --d_model 64 \
      --d_ff 256 \
      --e_layers 3 \
      --save_pred_true
    echo "  -> $itransformer_ckpt"
  fi
  [ -f "$itransformer_ckpt/pred_true_4channels_data.npz" ] && python scripts/plot_pred_true_4channels_from_data.py --ckpt_dir "$itransformer_ckpt" || echo "  跳过 iTransformer 首窗图（无 npz）"
  if [ -f "$itransformer_ckpt/pred.npy" ]; then
    python scripts/plot_full_testset_from_npy.py --ckpt_dir "$itransformer_ckpt" --tag "" --style paper_v4
    [ -f "$itransformer_ckpt/plots_full_testset/pred_true_full_TEST_step1_paper_v4.pdf" ] && cp "$itransformer_ckpt/plots_full_testset/pred_true_full_TEST_step1_paper_v4.pdf" "$itransformer_ckpt/pred_true_full_TEST_step1_iTransformer_paper_v4.pdf"
  else
    echo "  跳过 iTransformer 全测试集图（无 pred.npy）"
  fi
  echo ""
fi

echo "Done. 首窗图见 pred_true_4channels.png；全测试集 paper_v4 图见 plots_full_testset/ 及 *_paper_v4.pdf"
