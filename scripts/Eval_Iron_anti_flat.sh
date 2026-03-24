#!/bin/bash
# 用训练好的 Iron_anti_flat checkpoint 在测试集上计算 MAE/RMSE，并输出四通道预测图到 checkpoint 目录
# 与 Eval_Iron.sh 一致，仅改 model_comment=iron_anti_flat 并加 --regression_head_mlp 以匹配训练时结构
# 默认用 CPU 避免 GPU 被占；有空闲 GPU 可去掉 --device cpu 并设 export CUDA_VISIBLE_DEVICES=2
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
  --model_comment iron_anti_flat \
  --llm_layers 32 \
  --itr 1 \
  --regression_head_mlp \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
