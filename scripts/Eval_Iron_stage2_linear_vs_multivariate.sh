#!/bin/bash
# 对比主实验（iron_stage2_linear）与 multivariate 模式：只跑 multivariate 评估，主实验沿用已有 eval_result.txt。
# 跑完后在 checkpoints 下生成对比报告 compare_linear_vs_multivariate.txt
# 默认用 CPU；有空闲 GPU 可去掉 --device cpu 并设 export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=2

set -e
CHECKPOINTS_DIR="./checkpoints"
COMPARE_FILE="$CHECKPOINTS_DIR/compare_linear_vs_multivariate.txt"
SETTING_PREFIX="long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Iron_Ore_Transport_Exp_0"

echo "========== 评估 multivariate 模式（主实验沿用已有结果）=========="
python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_stage2_multivariate_linear \
  --llm_layers 32 \
  --itr 1 \
  --multivariate \
  --channel_mixing \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1

echo ""
echo "========== 生成对比报告 =========="
LINEAR_DIR="$CHECKPOINTS_DIR/${SETTING_PREFIX}-iron_stage2_linear"
MULTI_DIR="$CHECKPOINTS_DIR/${SETTING_PREFIX}-iron_stage2_multivariate_linear"
{
  echo "=============================================="
  echo "  Linear vs Multivariate 评估对比"
  echo "  生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================="
  echo ""
  echo "---------- 主实验 (iron_stage2_linear) ----------"
  echo "Checkpoint: $LINEAR_DIR"
  echo ""
  if [ -f "$LINEAR_DIR/eval_result.txt" ]; then
    cat "$LINEAR_DIR/eval_result.txt"
  else
    echo "(eval_result.txt 不存在，主实验请先跑 Eval_Iron_stage2_linear.sh)"
  fi
  echo ""
  echo "---------- Multivariate (iron_stage2_multivariate_linear) ----------"
  echo "Checkpoint: $MULTI_DIR"
  echo ""
  if [ -f "$MULTI_DIR/eval_result.txt" ]; then
    cat "$MULTI_DIR/eval_result.txt"
  else
    echo "(eval_result.txt 不存在，请先完成上方 multivariate 评估)"
  fi
  echo ""
  echo "=============================================="
} > "$COMPARE_FILE"

echo "对比报告已保存: $COMPARE_FILE"
echo "主实验结果: $LINEAR_DIR/eval_result.txt"
echo "Multivariate 结果: $MULTI_DIR/eval_result.txt"
