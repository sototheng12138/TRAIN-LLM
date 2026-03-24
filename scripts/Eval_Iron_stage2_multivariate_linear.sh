#!/bin/bash
# 两阶段多元版 Stage 2（线性回归头）评估：加载 iron_stage2_multivariate_linear，需与训练一致加 --multivariate --channel_mixing
# 默认用 CPU；有空闲 GPU 可去掉 --device cpu
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
  --model_comment iron_stage2_multivariate_linear \
  --llm_layers 32 \
  --itr 1 \
  --multivariate \
  --channel_mixing \
  --output_aux_semantic \
  --aux_confidence_threshold 0.5 \
  --zero_threshold -1

