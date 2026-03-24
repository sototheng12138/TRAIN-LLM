#!/bin/bash
# 用训练好的 Iron_48 checkpoint 在测试集上计算 MAE 和 RMSE（参数需与 Iron_48.sh 一致）
# 默认用 CPU 避免 GPU 被占导致 OOM；有空闲 GPU 可去掉 --device cpu 并设 export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=2

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_48_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_48 \
  --llm_layers 32 \
  --itr 1
