#!/bin/bash
# 评估 Iron_pl36.sh（seq_len=96, pred_len=36），参数需与训练一致

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_36 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 36 \
  --pred_len 36 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_pl36 \
  --llm_layers 32 \
  --itr 1
