#!/bin/bash
# 用训练好的 DLinear Iron checkpoint 在测试集上计算 MAE/RMSE，与 Time-LLM 对比
# 参数与 Train_Iron_DLinear.sh 一致（含 d_model 等以匹配 checkpoint 路径）

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path 2023_2025_Iron_data.csv \
  --model_id Iron_96_48 \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 16 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 3 \
  --moving_avg 25 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_dlinear \
  --itr 1
