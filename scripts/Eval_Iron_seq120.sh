#!/bin/bash
# 评估 Iron_seq120.sh（seq_len=120, pred_len=48），参数需与训练一致

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_120_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 120 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_seq120 \
  --llm_layers 32 \
  --itr 1
