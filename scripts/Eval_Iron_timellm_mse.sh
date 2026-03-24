#!/bin/bash
# Time-LLM baseline 评估：仅 MSE，不启用辅助头语义/门控，不做零阈值截断。
# 对 iron_timellm_mse checkpoint 在测试集上计算 MAE/RMSE，并输出四通道预测图到 checkpoint 目录。

# export CUDA_VISIBLE_DEVICES=2

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
  --model_comment iron_timellm_mse \
  --llm_layers 32 \
  --itr 1
