#!/bin/bash
# 评估 M5 十通道模型：在 m5_10ch 上训练得到的 checkpoint，输出 10 组泛化指标（各指标 MAE/RMSE）。

# export CUDA_VISIBLE_DEVICES=2

python run_eval.py \
  --device cpu \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path m5_10ch.csv \
  --model_id Iron_96_48 \
  --model TimeLLM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model 32 \
  --d_ff 128 \
  --factor 3 \
  --des 'Iron_Ore_Transport_Exp' \
  --model_comment iron_stage2_linear_m5_10ch \
  --llm_layers 32 \
  --itr 1 \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1
