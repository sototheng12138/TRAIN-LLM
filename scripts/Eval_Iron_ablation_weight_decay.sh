#!/bin/bash
# 评估：开启 weight_decay 的消融（与 Iron_ablation_weight_decay.sh 对应）；有空闲 GPU 可去掉 --device cpu
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
  --model_comment iron \
  --llm_layers 32 \
  --itr 1 \
  --weight_decay 1e-5
