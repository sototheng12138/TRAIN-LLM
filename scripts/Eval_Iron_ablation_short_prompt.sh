#!/bin/bash
# 消融第一组：简短 Prompt 模型评估（参数需与 Iron_ablation_short_prompt.sh 一致）

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
  --model_comment iron_short \
  --prompt_type short \
  --prompt_domain 1 \
  --llm_layers 32 \
  --itr 1
